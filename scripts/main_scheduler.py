"""Budget-conditioned query set scheduler and matched MLP control."""
import torch
from torch import nn


class Scheduler(nn.Module):
    def __init__(self, variant='aptu'):
        super().__init__()
        if variant not in ('aptu', 'mlp', 'no_assoc'):
            raise ValueError('Unknown scheduler variant: ' + variant)
        self.variant = variant
        self.latent_projection = nn.Linear(768, 64)
        self.input_projection = nn.Sequential(nn.Linear(212, 64), nn.LayerNorm(64), nn.ReLU())
        self.budget_embedding = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64))
        if variant == 'mlp':
            # Same inputs and global set context; parameter count matches the attention control.
            self.relation = nn.Sequential(nn.Linear(256, 160), nn.ReLU(), nn.Linear(160, 160),
                                          nn.ReLU(), nn.Linear(160, 64))
        else:
            layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.0)
            self.relation = nn.TransformerEncoder(layer, num_layers=2)
        self.output_head = nn.Linear(64, 1)
        if sum(p.numel() for p in self.parameters()) >= 500000:
            raise ValueError('Scheduler exceeds the requested 0.5M parameter ceiling')

    def forward(self, tokens, budget_scalar):
        if tokens.ndim != 2 or tokens.shape[1] != 2324:
            raise ValueError('Expected N x 2324 temporal query features')
        latent = [self.latent_projection(tokens[:, offset:offset + 768]) for offset in (0, 768, 1536)]
        query = self.input_projection(torch.cat(latent + [tokens[:, 2304:]], dim=-1))
        budget = self.budget_embedding(tokens.new_tensor(float(budget_scalar)).reshape(1, 1))
        if self.variant == 'mlp':
            if len(query):
                mean, maximum = query.mean(0, keepdim=True), query.max(0, keepdim=True)[0]
                query = self.relation(torch.cat([query, mean.expand_as(query), maximum.expand_as(query),
                                                 budget.expand_as(query)], dim=-1))
        else:
            query = self.relation(torch.cat([query, budget], dim=0).unsqueeze(1))[:-1, 0]
        return self.output_head(query).squeeze(-1)


def straight_through_latents(decoded, current, last, probability, force):
    """The forward tensor equals the actual decoded packet; backward uses soft gates."""
    from main_message import FEATURES
    continuous = torch.where(force, torch.ones_like(probability), probability)
    result = dict(decoded)
    for name in FEATURES:
        soft = continuous[:, None] * current[name] + (1 - continuous[:, None]) * last[name]
        result[name] = decoded[name] + (soft - soft.detach())
        if not torch.equal(result[name].detach(), decoded[name]):
            raise RuntimeError('Straight-through forward differs from serialized receiver values')
    return result
