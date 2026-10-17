"""V4 per-query MLP and direct optional latent-byte regularization."""
import argparse
import inspect
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn

from main_message import FEATURES
from v2_scheduler import straight_through_latents
import runtime


class Scheduler(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_projection = nn.Linear(768, 64)
        self.input_projection = nn.Sequential(nn.Linear(212, 64), nn.LayerNorm(64), nn.ReLU())
        self.relation = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.output_head = nn.Linear(64, 1)
        if sum(parameter.numel() for parameter in self.parameters()) != runtime.PARAMETER_COUNT:
            raise ValueError('V4 MLP differs from the prescribed minimal architecture')

    def forward(self, tokens, valid_mask=None):
        if tokens.ndim != 2 or tokens.shape[1] != 2324:
            raise ValueError('Expected N x 2324 temporal query features')
        valid = torch.ones(len(tokens), dtype=torch.bool, device=tokens.device) if valid_mask is None else valid_mask
        if valid.shape != (len(tokens),) or valid.dtype != torch.bool:
            raise ValueError('Invalid valid-query mask')
        observed = tokens[valid]
        latent = [self.latent_projection(observed[:, offset:offset + 768]) for offset in (0, 768, 1536)]
        query = self.input_projection(torch.cat(latent + [observed[:, 2304:]], dim=-1))
        logits = self.output_head(self.relation(query)).squeeze(-1)
        output = tokens.new_full((len(tokens),), float('-inf'))
        output[valid] = logits
        return output


def communication_loss(probability, force, reference, valid_mask=None):
    """Optional bytes per real frame, normalized by a fixed train-only constant."""
    if probability.ndim != 1 or probability.shape != force.shape or force.dtype != torch.bool:
        raise ValueError('Invalid probability or no-cache mask shape')
    if not math.isfinite(float(reference)) or reference <= 0:
        raise ValueError('Communication reference must be positive and finite')
    if not torch.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('UPDATE probabilities must lie in [0, 1]')
    valid = torch.ones_like(force) if valid_mask is None else valid_mask
    if valid.shape != force.shape or valid.dtype != torch.bool:
        raise ValueError('Invalid valid-query mask')
    estimated_optional_bytes = 780 * probability[valid & ~force].sum()
    return estimated_optional_bytes / reference, estimated_optional_bytes


def self_check(root):
    from v2_message import (FIELDS, ReceiverCache, TemporalCache, int8_roundtrip,
        message_sizes, pack, select_updates, self_check as inherited_checks)
    from v2_scheduler import rate_loss
    torch.manual_seed(23)
    model = Scheduler().eval()
    assert list(inspect.signature(model.forward).parameters) == ['tokens', 'valid_mask']
    assert not any('budget' in name for name, _ in model.named_parameters())
    tokens = torch.randn(5, 2324)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    assert torch.allclose(model(tokens[permutation]), model(tokens)[permutation], atol=1e-6, rtol=1e-5)
    assert torch.allclose(model(tokens[:1]), model(tokens)[:1], atol=1e-6, rtol=1e-5)
    valid = torch.tensor([True, False, True, False, True])
    masked = model(tokens, valid)
    assert torch.allclose(masked[valid], model(tokens[valid]), atol=1e-6, rtol=1e-5)
    assert torch.isneginf(masked[~valid]).all()
    assert model(torch.empty(0, 2324)).shape == (0,)
    mlp_comm, _ = communication_loss(model(tokens).sigmoid(), torch.tensor([True, False, False, False, False]), runtime.REFERENCE)
    mlp_gradients = torch.autograd.grad(mlp_comm, tuple(model.parameters()), allow_unused=True)
    assert all(torch.isfinite(value).all() for value in mlp_gradients if value is not None)
    mlp_gradient_norm = sum(float(value.square().sum()) for value in mlp_gradients if value is not None) ** .5
    assert mlp_gradient_norm > 0
    inherited = inherited_checks()

    raw = {name: torch.randn(3, width) for name, width in zip(FIELDS, (256, 256, 256, 3, 10))}
    raw.update(obj_idxes=torch.tensor([8, 19, 5]), scores=torch.tensor([.8, .7, .6]))
    sender, receiver = TemporalCache(), ReceiverCache()
    meta = dict(scene='v4_contract', session=0, frame_idx=0, token='0', timestamp_us=0.)
    cold = sender.prepare(int8_roundtrip(raw), meta)
    forced_flags = select_updates(torch.zeros(3), cold['force'])
    assert forced_flags.all()
    decoded, _ = receiver.unpack(pack(raw, meta, forced_flags), meta)
    sender.commit(raw, forced_flags, meta, decoded)
    meta.update(frame_idx=1, token='1', timestamp_us=100000.)
    changed = dict(raw)
    for name in FEATURES:
        changed[name] = raw[name] + 1
    current = int8_roundtrip(changed)
    warm = sender.prepare(current, meta)
    probability = torch.tensor([.49, .50, .51], requires_grad=True)
    flags = select_updates(probability.detach(), warm['force'])
    assert flags.tolist() == [False, True, True]
    mandatory, cost, _ = message_sizes(changed, meta)
    assert cost == 780
    flags_plus = flags.clone()
    flags_plus[0] = True
    assert len(pack(changed, meta, flags_plus)) - len(pack(changed, meta, flags)) == 780
    decoded, _ = receiver.unpack(pack(changed, meta, flags), meta)
    student = straight_through_latents(decoded, current, warm['last'], probability, warm['force'])
    assert all(torch.equal(student[name].detach(), decoded[name]) for name in FIELDS)
    assert all(torch.equal(decoded[name][~flags], warm['last'][name][~flags]) for name in FEATURES)
    latent_gradient = torch.autograd.grad(sum(student[name].sum() for name in FEATURES), probability)[0]
    assert torch.isfinite(latent_gradient).all() and latent_gradient.abs().sum() > 0

    p = torch.tensor([.2, .3, .4], dtype=torch.float64, requires_grad=True)
    force = torch.tensor([True, False, False])
    comm, soft = communication_loss(p, force, runtime.REFERENCE)
    gradient = torch.autograd.grad(comm, p, retain_graph=True)[0]
    assert math.isclose(float(soft), 546., abs_tol=1e-10)
    assert gradient[0] == 0 and torch.allclose(gradient[1:], torch.full((2,), 780 / runtime.REFERENCE, dtype=p.dtype))
    old, old_soft, feasible = rate_loss(p, force, 1000, 780, 1000, runtime.REFERENCE)
    old_gradient = torch.autograd.grad(old, p)[0]
    repeated = torch.stack([communication_loss(p, force, runtime.REFERENCE)[0] for _ in range(4)]).mean()
    repeated_gradient = torch.autograd.grad(repeated, p)[0]
    assert torch.allclose(repeated_gradient, gradient, atol=1e-15, rtol=1e-12)
    padded, padded_soft = communication_loss(p, force, runtime.REFERENCE, torch.tensor([True, True, False]))
    assert math.isclose(float(padded_soft), 234., abs_tol=1e-10)
    forced_loss, forced_soft = communication_loss(p, torch.ones_like(force), runtime.REFERENCE)
    assert forced_loss.item() == 0 and forced_soft.item() == 0
    assert torch.equal(torch.autograd.grad(forced_loss, p)[0], torch.zeros_like(p))
    empty, empty_soft = communication_loss(torch.empty(0), torch.empty(0, dtype=torch.bool), runtime.REFERENCE)
    assert empty.item() == 0 and empty_soft.item() == 0
    singleton = torch.stack([communication_loss(p, force, runtime.REFERENCE)[0]]).mean()
    assert torch.allclose(singleton, comm)
    uneven = torch.stack([comm, communication_loss(p[:1], torch.zeros(1, dtype=torch.bool), runtime.REFERENCE)[0]]).mean()
    assert math.isclose(float(uneven), (546 + 156) / (2 * runtime.REFERENCE), rel_tol=1e-12)
    empty_real_frame = torch.stack([comm, empty.to(comm)]).mean()
    assert torch.allclose(empty_real_frame, comm / 2)
    evidence = runtime.reference_evidence(root)
    settings = runtime.load_settings(root)
    comparison = {'mandatory': 1000, 'force': [True, False, False], 'probability': [.2, .3, .4],
        'v3_budget': 1000, 'v3_feasible': feasible, 'v3_soft_total_bytes': float(old_soft),
        'v3_rate_loss': float(old), 'v3_probability_gradient': old_gradient.tolist(),
        'v4_optional_soft_bytes': float(soft), 'v4_comm_loss': float(comm),
        'v4_probability_gradient': gradient.tolist(), 'v4_lambda_weighted_gradient': (settings['lambda_comm'] * gradient).tolist(),
        'repeated_four_frame_mean_gradient': repeated_gradient.tolist(),
        'reduction': 'mean of real per-frame probability sums; repeated identical frames preserve loss and shared-parameter gradient'}
    report = {'status': 'PASS', 'model': runtime.MODEL,
        'scope': 'constructed CPU protocol/loss checks; real native task gradient is required separately',
        'budget_input_removed': True, 'forward_arguments': list(inspect.signature(model.forward).parameters),
        'state_dimension': 2324, 'scheduler_parameters': sum(value.numel() for value in model.parameters()),
        'per_query_without_pooling': True, 'permutation_equivariance': True,
        'padding_excluded': True, 'empty_query_frame_safe': True,
        'hard_threshold_boundary': {'probability': [.49, .50, .51], 'flags': flags.tolist()},
        'missing_cache_forces_update': True, 'actual_packet_increment_bytes': 780,
        'latent_payload_increment_bytes': 780, 'real_decoded_hard_ST_forward': True,
        'latent_surrogate_gradient_finite_nonzero': True, 'forced_communication_gradient_zero': True,
        'communication_MLP_gradient_norm': mlp_gradient_norm,
        'real_frame_mean_gradient_scaling': True, 'uneven_query_counts_not_divided': True,
        'empty_real_frames_count_in_mean': True, 'lambda_comm': settings['lambda_comm'],
        'reference_evidence': evidence, 'objective_comparison': comparison,
        'inherited_protocol_checks': inherited, 'source_hashes': runtime.source_identity(root)}
    runtime.write(runtime.experiment(root) / 'audit/cpu_preflight.json', report)
    print('V4 CPU preflight PASS; 2324 state, 71361 parameters, no budget input, direct optional bytes', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--self-check', action='store_true', required=True)
    args = parser.parse_args()
    self_check(args.root.resolve())
