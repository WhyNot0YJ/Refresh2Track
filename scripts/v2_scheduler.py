"""V2 per-query MLP and query-set Transformer, with actual-decoded ST latents."""
import torch
from torch import nn

from main_message import FEATURES


class Scheduler(nn.Module):
    def __init__(self, variant='transformer'):
        super().__init__()
        if variant not in ('mlp', 'transformer'):
            raise ValueError('Unknown V2 scheduler variant: ' + variant)
        self.variant = variant
        self.latent_projection = nn.Linear(768, 64)
        self.input_projection = nn.Sequential(nn.Linear(212, 64), nn.LayerNorm(64), nn.ReLU())
        self.budget_embedding = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64))
        if variant == 'mlp':
            # V1 MLP pools across queries; the specified V2 control is per query.
            self.relation = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        else:
            layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.0)
            self.relation = nn.TransformerEncoder(layer, num_layers=2)
        self.output_head = nn.Linear(64, 1)
        if sum(parameter.numel() for parameter in self.parameters()) >= 500000:
            raise ValueError('V2 scheduler exceeds 0.5M parameter ceiling')

    def forward(self, tokens, budget_scalar, valid_mask=None):
        if tokens.ndim != 2 or tokens.shape[1] != 2324:
            raise ValueError('Expected N x 2324 temporal query features')
        valid = torch.ones(len(tokens), dtype=torch.bool, device=tokens.device) if valid_mask is None else valid_mask
        if valid.shape != (len(tokens),) or valid.dtype != torch.bool:
            raise ValueError('Invalid valid-query mask')
        observed = tokens[valid]
        latent = [self.latent_projection(observed[:, offset:offset + 768]) for offset in (0, 768, 1536)]
        query = self.input_projection(torch.cat(latent + [observed[:, 2304:]], dim=-1))
        scalar = torch.as_tensor(budget_scalar, device=tokens.device, dtype=tokens.dtype).reshape(1, 1)
        budget = self.budget_embedding(scalar)
        if self.variant == 'mlp':
            query = self.relation(torch.cat([query, budget.expand_as(query)], dim=-1))
        else:
            query = self.relation(torch.cat([query, budget], dim=0).unsqueeze(1))[:-1, 0]
        logits = self.output_head(query).squeeze(-1)
        output = tokens.new_full((len(tokens),), float('-inf'))
        output[valid] = logits
        return output


def straight_through_latents(decoded, current, last, probability, force, valid_mask=None):
    """Hard forward is real decoded D; gradient sees detached C minus old L."""
    valid = torch.ones_like(force) if valid_mask is None else valid_mask
    soft = torch.where(force, torch.ones_like(probability), probability)
    soft = torch.where(valid, soft, torch.zeros_like(soft))
    result = {name: value.detach() for name, value in decoded.items()}
    for name in FEATURES:
        surrogate = soft[:, None] * current[name].detach() + (1 - soft[:, None]) * last[name].detach()
        result[name] = decoded[name].detach() + (surrogate - surrogate.detach())
        if not torch.equal(result[name].detach(), decoded[name].detach()):
            raise RuntimeError('V2 ST forward differs from the actual receiver values')
    return result


def rate_loss(probability, force, mandatory, per_update, budget, reference, valid_mask=None):
    """One-sided soft cost penalty; hard packet bytes must be logged separately."""
    if per_update != 780 or reference <= 0:
        raise ValueError('V2 rate loss requires the inherited 780B bundle and positive train reference')
    valid = torch.ones_like(force) if valid_mask is None else valid_mask
    forced = valid & force
    optional = valid & ~force
    unavoidable = float(mandatory) + per_update * int(forced.sum())
    soft_bytes = probability.new_tensor(unavoidable) + per_update * probability[optional].sum()
    feasible = max(float(budget), unavoidable)
    return (torch.relu(soft_bytes - feasible) / reference).square(), soft_bytes, feasible


def self_check():
    from v2_message import FIELDS, TemporalCache, ReceiverCache, int8_roundtrip, pack, select_updates
    torch.manual_seed(23)
    tokens = torch.randn(5, 2324)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    counts = {}
    for variant in ('mlp', 'transformer'):
        model = Scheduler(variant).eval()
        counts[variant] = sum(parameter.numel() for parameter in model.parameters())
        assert torch.allclose(model(tokens[permutation], .5), model(tokens, .5)[permutation], atol=1e-6, rtol=1e-5)
        valid = torch.tensor([True, False, True, False, True])
        masked = model(tokens, .5, valid)
        assert torch.allclose(masked[valid], model(tokens[valid], .5), atol=1e-6, rtol=1e-5)
        assert torch.isneginf(masked[~valid]).all()
        assert model(torch.empty(0, 2324), .5).shape == (0,)
        if variant == 'mlp':
            assert torch.allclose(model(tokens[:1], .5), model(tokens, .5)[:1], atol=1e-6, rtol=1e-5)

    raw = {key: torch.randn(3, width) for key, width in zip(FIELDS, (256, 256, 256, 3, 10))}
    raw.update(obj_idxes=torch.tensor([8, 19, 5]), scores=torch.tensor([.8, .7, .6]))
    mirror, receiver = TemporalCache(), ReceiverCache()
    meta = dict(scene='st_contract', session=0, frame_idx=0, token='0', timestamp_us=0.)
    temporal = mirror.prepare(int8_roundtrip(raw), meta)
    decoded, _ = receiver.unpack(pack(raw, meta, temporal['force']), meta)
    mirror.commit(raw, temporal['force'], meta, decoded)
    current_raw = dict(raw)
    for key in FEATURES:
        current_raw[key] = raw[key] + 1
    meta.update(frame_idx=1, token='1', timestamp_us=100000.)
    current = int8_roundtrip(current_raw)
    temporal = mirror.prepare(current, meta)
    logits = torch.tensor([-2., 0., 2.], requires_grad=True)
    probability = logits.sigmoid()
    flags = select_updates(probability, temporal['force'])
    decoded, _ = receiver.unpack(pack(current_raw, meta, flags), meta)
    # Deliberately grad-enabled observations prove the ST helper detaches C/L/D.
    observed = {key: value.detach().requires_grad_() for key, value in current.items() if key in FEATURES}
    previous = {key: value.detach().requires_grad_() for key, value in temporal['last'].items()}
    student = straight_through_latents(decoded, observed, previous, probability, temporal['force'])
    assert all(torch.equal(student[key].detach(), decoded[key]) for key in FEATURES)
    assert all(torch.equal(student[key], decoded[key]) for key in FIELDS[3:])
    sum(student[key].sum() for key in FEATURES).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert all(value.grad is None for value in list(observed.values()) + list(previous.values()))
    mirror.commit(current_raw, flags, meta, decoded)
    p = torch.tensor([.2, .3, .4], requires_grad=True)
    force = torch.tensor([True, False, False])
    loss, soft_bytes, feasible = rate_loss(p, force, 1000, 780, 12198, 12840.189523169913)
    assert loss.item() == 0 and feasible == 12198
    loss, soft_bytes, feasible = rate_loss(p, force, 1000, 780, 1000, 12840.189523169913)
    assert feasible == 1780 and loss.item() > 0
    loss.backward()
    assert p.grad[0] == 0 and (p.grad[1:] > 0).all()
    all_forced = torch.ones(3, dtype=torch.bool)
    loss, _, feasible = rate_loss(p, all_forced, 1000, 780, 1, 12840.189523169913)
    assert loss.item() == 0 and feasible == 3340
    print('V2 constructed scheduler/ST/rate checks PASS; parameter counts=' + str(counts))
    return {'status': 'PASS', 'scope': 'constructed CPU checks; native task gradient is checked separately',
            'parameter_counts': counts, 'permutation_equivariance_both_models': True,
            'mlp_per_query_without_pooling': True, 'padding_excluded': True,
            'st_equals_actual_decoded_all_latents': True, 'st_observations_detached': True,
            'one_sided_rate_loss': True}


if __name__ == '__main__':
    self_check()
