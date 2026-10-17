"""Only scheduler inputs differ; packet, caches, threshold and STE are shared."""
from pathlib import Path
import torch
import runtime

canonical = runtime.load_module('refresh2track_canonical_scheduler', Path(__file__).resolve().parents[1] / 'v4_budget_free/scheduler.py')


def mask_inputs(tokens, mode):
    if tokens.ndim != 2 or tokens.shape[1] != 2324 or mode not in runtime.MODES:
        raise ValueError('Invalid input contract or ablation mode')
    masked = tokens.clone()
    if mode == 'no_explicit_delta':
        masked[:, 1536:2304] = 0
    elif mode == 'no_receiver_state':
        masked[:, 768:2305] = 0
    return masked


class Scheduler(canonical.Scheduler):
    def __init__(self, mode=None):
        super().__init__()
        self.mode = mode or runtime.MODEL
        if self.mode not in runtime.MODES:
            raise ValueError('Unknown ablation mode')

    def forward(self, tokens, valid_mask=None):
        return super().forward(mask_inputs(tokens, self.mode), valid_mask)


def self_check(root):
    from v2_message import self_check as cache_checks
    from v2_scheduler import straight_through_latents
    from main_message import FEATURES, FIELDS
    rows = []
    initial = None
    for mode in runtime.MODES:
        tokens = torch.randn(5, 2324, dtype=torch.float32, requires_grad=True)
        masked = mask_inputs(tokens, mode)
        columns = torch.ones(2324, dtype=torch.bool)
        if mode == 'no_explicit_delta':
            columns[1536:2304] = False
        elif mode == 'no_receiver_state':
            columns[768:2305] = False
        assert masked.shape == tokens.shape and masked.dtype == tokens.dtype
        assert torch.equal(masked[:, columns], tokens[:, columns])
        assert not masked[:, ~columns].count_nonzero()
        torch.manual_seed(0)
        model = Scheduler(mode)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        if initial is None: initial = state
        assert all(torch.equal(value, initial[name]) for name, value in state.items())
        assert sum(p.numel() for p in model.parameters()) == 71361
        assert torch.equal(model(tokens), canonical.Scheduler.forward(model, masked))
        model(tokens).sum().backward()
        assert torch.isfinite(tokens.grad).all() and not tokens.grad[:, ~columns].count_nonzero()
        current = {name: torch.randn(5, width) for name, width in zip(FIELDS, (256, 256, 256, 3, 10))}
        last = {name: torch.randn_like(current[name]) for name in FEATURES}
        decoded = dict(current, **last)
        probability = model(tokens.detach()).sigmoid()
        force = torch.tensor([True, False, False, False, False])
        for name in FEATURES:
            decoded[name] = decoded[name].clone()
            decoded[name][force] = current[name][force]
        student = straight_through_latents(decoded, current, last, probability, force)
        assert all(torch.equal(student[name].detach(), decoded[name]) for name in FIELDS)
        task = student['query_feats'].square().mean()
        comm, _ = canonical.communication_loss(probability, force, runtime.REFERENCE)
        weight = 0.0 if mode == 'no_communication_loss' else runtime.read(runtime.reference(root) / 'common.json')['lambda_comm']
        total = task + weight * comm
        task_grad = torch.autograd.grad(task, tuple(model.parameters()), retain_graph=True, allow_unused=True)
        total_grad = torch.autograd.grad(total, tuple(model.parameters()), retain_graph=True, allow_unused=True)
        assert any(g is not None and bool(g.abs().sum()) for g in task_grad)
        if weight == 0:
            assert total.item() == task.item()
            assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(task_grad, total_grad))
        rows.append({'mode': mode, 'status': 'PASS', 'effective_dimensions': int(columns.sum()),
                     'lambda_comm': weight, 'fresh_identical_initialization': True,
                     'task_gradient_through_real_hard_STE': True})
    report = {'status': 'PASS', 'input_modes': rows, 'cache_serializer_checks': cache_checks(),
              'scope': 'constructed CPU functional checks; not dataset metrics'}
    runtime.write(Path(root) / 'reports/ablations/input_mask_tests.json', report)
    return report
