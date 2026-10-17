"""Instance-local GBA residual and branch gate over the unchanged native reasoner."""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
from types import MethodType

import torch
from torch import nn


GBA_MODULES = ('get_node_veh', 'get_node_inf', 'get_edge', 'ffn')
GBA_TENSORS = ('wq', 'wk', 'we')
FUSION5 = ('fuse_feats', 'fuse_motion', 'fuse_embed', 'cross_domain_query', 'cross_domain_motion')
GBA = GBA_MODULES + GBA_TENSORS
NEW = ('gba_state_adapter', 'fusion_branch_gate')
EXPERIMENTS = ('A_GBA', 'B_GATE', 'C_GBA_GATE')


def enabled(experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError('Unknown structure experiment: ' + str(experiment))
    return experiment != 'B_GATE', experiment != 'A_GBA'


def allowed(name, experiment='C_GBA_GATE'):
    gba, gate = enabled(experiment)
    modules = (() if not gba else GBA_MODULES + ('gba_state_adapter',)) + (
        () if not gate else FUSION5 + ('fusion_branch_gate',))
    return (gba and name in tuple('STReasoner.' + key for key in GBA_TENSORS)) or any(
        name.startswith('STReasoner.' + key + '.') for key in modules)


def new_parameter(name):
    return name.startswith(('STReasoner.gba_state_adapter.', 'STReasoner.fusion_branch_gate.'))


def add_structures(model, normalization, experiment='C_GBA_GATE'):
    gba, gate = enabled(experiment)
    reasoner = model.STReasoner
    if hasattr(reasoner, '_structure_experiment'):
        raise ValueError('Structures are already registered on this model')
    scales = {}
    for key, width, unchanged in (('pair_scale', 12, (8, 9)), ('gate_scale', 7, (6,))):
        scale = torch.as_tensor(normalization[key], dtype=torch.float32).clone()
        if scale.shape != (width,) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError('Invalid train-only normalization: ' + key)
        if not torch.equal(scale[list(unchanged)], torch.ones(len(unchanged))):
            raise ValueError('Sin/cos and confidence normalization must remain one')
        scales[key] = scale
    target = next(reasoner.parameters())
    for use, name, width, output in ((gba, 'gba_state_adapter', 12, 1),
                                     (gate, 'fusion_branch_gate', 7, 3)):
        if use:
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(0)
                module = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, output))
                nn.init.zeros_(module[-1].weight)
                nn.init.zeros_(module[-1].bias)
            setattr(reasoner, name, module.to(device=target.device, dtype=target.dtype))
    for name, scale in scales.items():
        reasoner.register_buffer('_structure_' + name, scale.to(device=target.device, dtype=target.dtype))
    reasoner._structure_experiment = experiment
    return {'experiment': experiment, 'normalization': {key: value.tolist() for key, value in scales.items()},
            'added_parameters': sum(parameter.numel() for name, parameter in model.named_parameters()
                                    if new_parameter(name))}


def geometry_features(reasoner, inf, veh, receiver_state, decode_boxes):
    state = torch.as_tensor(receiver_state, dtype=veh.cache_query_feats.dtype,
                            device=veh.cache_query_feats.device)
    if state.shape != (len(inf), 4) or state.requires_grad or not torch.isfinite(state).all():
        raise ValueError('Receiver state must be detached CAA-filtered Ni x 4 physical quantities')
    inf_boxes = decode_boxes(inf.cache_bboxes, reasoner.pc_range)
    veh_boxes = decode_boxes(veh.cache_bboxes, reasoner.pc_range)
    if inf_boxes.shape[-1] != 9 or veh_boxes.shape[-1] != 9:
        raise ValueError('Physical yaw and velocity require native nine-dimensional boxes')
    inf_position = reasoner._loc_denorm(inf.cache_ref_pts, reasoner.pc_range)
    veh_position = reasoner._loc_denorm(veh.cache_ref_pts, reasoner.pc_range)
    position = veh_position[:, None, :] - inf_position[None, :, :]
    velocity = veh_boxes[:, None, 7:9] - inf_boxes[None, :, 7:9]
    yaw = veh_boxes[:, None, 6] - inf_boxes[None, :, 6]
    history = state[None, :, :].expand(len(veh), -1, -1)
    pair = torch.cat((history, position, position.norm(dim=-1, keepdim=True),
                      yaw.sin().unsqueeze(-1), yaw.cos().unsqueeze(-1), velocity), dim=-1)
    gate = torch.cat((history, position.norm(dim=-1, keepdim=True),
                      velocity.norm(dim=-1, keepdim=True),
                      veh.cache_scores[:, None, None].expand(-1, len(inf), -1)), dim=-1)
    if not torch.isfinite(pair).all() or not torch.isfinite(gate).all():
        raise FloatingPointError('Nonfinite receiver-observable structure features')
    return pair, gate


def matching_instrument(source):
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('Expected one native _learn_matching method')
    function = tree.body[0]
    if function.name != '_learn_matching' or function.decorator_list:
        raise ValueError('Native matching method layout changed')
    expected = ast.dump(ast.parse('attn = self.ffn(attn).squeeze(-1)').body[0], include_attributes=False)
    rewritten, count = [], 0
    for statement in function.body:
        rewritten.append(statement)
        if ast.dump(statement, include_attributes=False) == expected:
            callback = ast.parse('attn = self._structure_adjust_logits(attn, inf, veh)').body[0]
            rewritten.append(ast.copy_location(callback, statement))
            count += 1
    if count != 1:
        raise ValueError('Expected exactly one native pre-sigmoid logits insertion')
    function.body = rewritten
    return ast.fix_missing_locations(tree)


def install(model, context, decode_boxes=None):
    reasoner = model.STReasoner
    gba, gate = enabled(reasoner._structure_experiment)
    if hasattr(reasoner, '_structure_adjust_logits'):
        raise ValueError('Structure hooks are already installed')
    original_match, original_fusion = reasoner._learn_matching, reasoner._query_fusion
    unwrapped = inspect.unwrap(original_match.__func__)
    source = inspect.getsource(unwrapped)
    tree = matching_instrument(source)
    namespace = dict(unwrapped.__globals__)
    exec(compile(tree, '<structure-native-matching>', 'exec'), namespace)
    replacement = namespace['_learn_matching']
    decoder = decode_boxes or unwrapped.__globals__['denormalize_bbox']

    def adjust_logits(self, logits, inf, veh):
        lane = context.get('matching_cursor', context.get('active_sample', 0))
        if not isinstance(lane, int) or not 0 <= lane < len(context['samples']):
            raise ValueError('Structure matching lane is missing or out of range')
        context['active_sample'] = lane
        sample = context['samples'][lane]
        pair, unused_gate = geometry_features(self, inf, veh, sample['receiver_state'], decoder)
        if pair.shape[:2] != logits.shape:
            raise ValueError('Pair-state vehicle mask or CAA infrastructure axis differs from native logits')
        if context.get('capture_evidence', True):
            sample['raw_pair_features'] = pair.detach()
        sample['pair_shape'] = list(pair.shape[:2])
        if gba:
            logits = logits + self.gba_state_adapter(pair / self._structure_pair_scale).squeeze(-1)
        return logits

    def query_fusion(self, inf, veh, inf_idx, veh_idx):
        lane = context['active_sample']
        sample = context['samples'][lane]
        sample['matched_pairs'] = len(veh_idx)
        if not gate:
            return original_fusion(inf, veh, inf_idx, veh_idx)
        matched_veh, matched_inf = veh[veh_idx], inf[inf_idx]
        state = torch.as_tensor(sample['receiver_state'], device=veh.cache_query_feats.device,
                                dtype=veh.cache_query_feats.dtype)[inf_idx]
        pair, gate_grid = geometry_features(self, matched_inf, matched_veh, state, decoder)
        matched = torch.arange(len(veh_idx), device=gate_grid.device)
        features = gate_grid[matched, matched]
        if context.get('capture_evidence', True):
            sample['raw_gate_features'] = features.detach()
        gains = 2 * self.fusion_branch_gate(features / self._structure_gate_scale).sigmoid()
        if context.get('capture_evidence', True):
            sample['gate_values'] = gains.detach()
        names = ('cache_query_feats', 'cache_motion_feats', 'cache_query_embeds')
        originals = [getattr(matched_veh, name).clone() for name in names]
        fused = original_fusion(inf, veh, inf_idx, veh_idx)
        for branch, name in enumerate(names):
            output = getattr(fused, name)
            setattr(fused, name, output + (gains[:, branch:branch + 1] - 1) * (output - originals[branch]))
        return fused

    reasoner._structure_adjust_logits = MethodType(adjust_logits, reasoner)
    reasoner._learn_matching = MethodType(replacement, reasoner)
    reasoner._query_fusion = MethodType(query_fusion, reasoner)

    def remove():
        reasoner._learn_matching = original_match
        reasoner._query_fusion = original_fusion
        del reasoner._structure_adjust_logits

    return remove, {'native_matching_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                    'instrumented_ast_sha256': hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
                    'logits_insertions': 1, 'native_source_files_modified': False,
                    'native_association_input_detached': False,
                    'receiver_state_columns': ['log1p_age_frames', 'anchor_position_delta_m',
                                               'anchor_abs_wrapped_yaw_delta_rad', 'anchor_velocity_delta_mps']}


def configure_trainables(model, experiment='C_GBA_GATE'):
    enabled(experiment)
    if model.STReasoner._structure_experiment != experiment:
        raise ValueError('Registered structures differ from the trainable experiment')
    model.eval().requires_grad_(False)
    model.training = True
    model.STReasoner.training = True
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(allowed(name, experiment))
    for name, module in model.named_modules():
        if allowed(name + '.weight', experiment):
            module.train()


def named_trainables(model, scheduler, experiment='C_GBA_GATE'):
    return [('scheduler.' + name, parameter) for name, parameter in scheduler.named_parameters()] + [
        (name, parameter) for name, parameter in model.named_parameters() if allowed(name, experiment)]


def make_optimizer(model, scheduler, settings, experiment='C_GBA_GATE'):
    native = [parameter for name, parameter in model.named_parameters()
              if allowed(name, experiment) and not new_parameter(name)]
    additions = [parameter for name, parameter in model.named_parameters()
                 if allowed(name, experiment) and new_parameter(name)]
    peak = settings['optimizer']['peak_lr']
    native_peak = settings.get('native_lr', settings.get('fusion_lr', 1e-5))
    return torch.optim.AdamW([
        {'params': list(scheduler.parameters()), 'lr': peak, 'initial_lr': peak, 'name': 'scheduler'},
        {'params': additions, 'lr': peak, 'initial_lr': peak, 'name': 'new_structure'},
        {'params': native, 'lr': native_peak, 'initial_lr': native_peak, 'name': 'original_modules'}],
        weight_decay=settings['optimizer']['weight_decay'])


def parameter_proof(model, scheduler, optimizer, experiment='C_GBA_GATE'):
    named = named_trainables(model, scheduler, experiment)
    actual = [parameter for group in optimizer.param_groups for parameter in group['params']]
    if len(actual) != len(named) or {id(parameter) for parameter in actual} != {
            id(parameter) for name, parameter in named}:
        raise ValueError('Structure optimizer parameter allowlist differs')
    if any(parameter.requires_grad != allowed(name, experiment) for name, parameter in model.named_parameters()):
        raise ValueError('Native requires_grad differs from the exact structure allowlist')
    if not all(parameter.requires_grad for parameter in scheduler.parameters()):
        raise ValueError('Scheduler contains frozen parameters')
    if any(module.training for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)):
        raise ValueError('Frozen native BatchNorm is in training mode')
    counts = {'scheduler': sum(parameter.numel() for parameter in scheduler.parameters()),
              'original_gba': sum(parameter.numel() for name, parameter in model.named_parameters()
                                  if allowed(name, experiment) and name.split('.')[1] in GBA_MODULES + GBA_TENSORS),
              'original_fusion5': sum(parameter.numel() for name, parameter in model.named_parameters()
                                      if allowed(name, experiment) and name.split('.')[1] in FUSION5),
              'new_structure': sum(parameter.numel() for name, parameter in model.named_parameters()
                                   if new_parameter(name))}
    if counts['scheduler'] != 71361:
        raise ValueError('Native V4 Scheduler parameter count changed')
    return {'experiment': experiment, 'module_counts': counts, 'total_trainable_parameters': sum(counts.values()),
            'optimizer_parameter_names': [name for name, parameter in named],
            'optimizer_group_names': [group['name'] for group in optimizer.param_groups],
            'batchnorm_statistics_frozen': True, 'reasoner_training': model.STReasoner.training}


def state_hash(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        value.update(name.encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(str(tensor.dtype).encode())
        value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def native_state(model, experiment='C_GBA_GATE'):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if allowed(name, experiment)}


def load_native_state(model, state, experiment='C_GBA_GATE'):
    expected = native_state(model, experiment)
    if set(state) != set(expected) or any(state[name].shape != expected[name].shape for name in expected):
        raise ValueError('Checkpoint native structure parameter names or shapes differ')
    model.load_state_dict(state, strict=False)
    if state_hash(native_state(model, experiment)) != state_hash(state):
        raise ValueError('Actual native structure weights differ after checkpoint load')


def frozen_hash(model, experiment='C_GBA_GATE'):
    return state_hash({name: value for name, value in model.state_dict().items() if not allowed(name, experiment)})


def frozen_snapshot(model, experiment='C_GBA_GATE'):
    parameters = tuple((name, parameter) for name, parameter in model.named_parameters()
                       if not allowed(name, experiment))
    buffers = tuple(model.named_buffers())
    return {'versions': {name: parameter._version for name, parameter in parameters},
            'parameters': parameters,
            'buffers': {name: value.detach().cpu().clone() for name, value in buffers},
            'device_buffers': tuple((name, value, value.detach().clone()) for name, value in buffers),
            'sha256': frozen_hash(model, experiment)}


def assert_frozen(model, snapshot, full=False, experiment='C_GBA_GATE'):
    for name, parameter in snapshot['parameters']:
        if parameter.requires_grad or parameter.grad is not None or parameter._version != snapshot['versions'][name]:
            raise ValueError('Frozen native parameter changed or received gradients: ' + name)
    comparisons = []
    for name, value, expected in snapshot['device_buffers']:
        if value.shape != expected.shape:
            raise ValueError('Frozen native buffer changed: ' + name)
        comparisons.append(torch.eq(value.detach(), expected).all())
    unchanged = torch.stack(comparisons).cpu().tolist() if comparisons else []
    for (name, value, expected), passed in zip(snapshot['device_buffers'], unchanged):
        if not passed:
            raise ValueError('Frozen native buffer changed: ' + name)
    if full and frozen_hash(model, experiment) != snapshot['sha256']:
        raise ValueError('Frozen native weights differ by SHA256')


def gradient_statistics(groups):
    groups = [(name, [value for value in gradients if value is not None]) for name, gradients in groups]
    gradients = [value for name, values in groups for value in values]
    finite = [torch.isfinite(value).all() for value in gradients]
    squares = [value.detach().square().sum() for value in gradients]
    scalars = torch.stack(finite + squares).detach().cpu().tolist() if gradients else []
    if not all(scalars[:len(gradients)]):
        raise FloatingPointError('Nonfinite structure optimizer gradient')
    result, offset = {}, len(gradients)
    for name, values in groups:
        result[name] = sum(scalars[offset:offset + len(values)]) ** .5
        offset += len(values)
    return result


def gradient_norm(parameters):
    return gradient_statistics([('norm', [parameter.grad for parameter in parameters])])['norm']


def gradient_norms(model, scheduler, experiment='C_GBA_GATE'):
    result = {'scheduler': gradient_norm(scheduler.parameters())}
    for name in GBA_MODULES + GBA_TENSORS + FUSION5 + ('gba_state_adapter', 'fusion_branch_gate'):
        value = getattr(model.STReasoner, name, None)
        if value is not None and allowed('STReasoner.' + name + ('' if name in GBA_TENSORS else '.weight'), experiment):
            result[name] = gradient_norm([value] if isinstance(value, nn.Parameter) else value.parameters())
    return result


def observation_self_check():
    import struct

    groups = [('first', [torch.tensor([1., -2., 3.]), None,
                          torch.arange(13, dtype=torch.float32) / 7]),
              ('empty', []), ('zero', [torch.zeros(4), torch.empty(0)]),
              ('mixed', [torch.tensor([1e-20, -1e10]), torch.tensor([.25], dtype=torch.float64)])]
    expected = {}
    for name, values in groups:
        values = [value for value in values if value is not None]
        if any(not torch.isfinite(value).all() for value in values):
            raise FloatingPointError('Nonfinite legacy diagnostic gradient')
        expected[name] = sum(float(value.detach().square().sum()) for value in values) ** .5
    actual = gradient_statistics(groups)
    assert all(struct.pack('!d', actual[name]) == struct.pack('!d', expected[name]) for name in expected)
    try:
        gradient_statistics([('nonfinite', [torch.tensor([float('inf')])])])
    except FloatingPointError:
        pass
    else:
        raise AssertionError('Nonfinite gradient was not rejected')
    model = nn.Module()
    model.register_parameter('frozen_parameter', nn.Parameter(torch.tensor([1., 2.]), requires_grad=False))
    model.register_buffer('frozen_buffer', torch.tensor([1., 2., 3.]))
    snapshot = frozen_snapshot(model)
    assert_frozen(model, snapshot, full=True)
    model.frozen_buffer.add_(1)
    try:
        assert_frozen(model, snapshot)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed buffer was not rejected')
    model.frozen_buffer.sub_(1)
    assert_frozen(model, snapshot, full=True)
    model.frozen_parameter.grad = torch.ones_like(model.frozen_parameter)
    try:
        assert_frozen(model, snapshot)
    except ValueError:
        pass
    else:
        raise AssertionError('Frozen parameter gradient was not rejected')
    return {'status': 'PASS', 'gradient_norm_python_float_bytes_exact': True,
            'empty_and_mixed_dtype_groups': True, 'nonfinite_gradient_rejected': True,
            'buffer_mutation_rejected': True, 'frozen_gradient_rejected': True}


def self_check():
    observation_proof = observation_self_check()
    class Instances:
        def __init__(self, **fields):
            self.__dict__.update(fields)

        def __len__(self):
            return len(self.cache_query_feats)

        def __getitem__(self, indices):
            return Instances(**{name: value[indices] for name, value in self.__dict__.items()})

    class Reasoner(nn.Module):
        def __init__(self):
            super().__init__()
            self.pc_range = (-10, -10, -2, 10, 10, 2)
            for name in GBA_MODULES[:2] + FUSION5:
                setattr(self, name, nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256)))
            self.get_edge = nn.Sequential(nn.Linear(3, 256), nn.ReLU(), nn.Linear(256, 256))
            self.ffn = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 1))
            for name in GBA_TENSORS:
                setattr(self, name, nn.Parameter(torch.randn(256, 256) * .01))

        def _loc_denorm(self, points, bounds):
            lower = points.new_tensor(bounds[:3])
            return points * (points.new_tensor(bounds[3:]) - lower) + lower

        def _learn_matching(self, inf, veh, inf_pts, veh_pts, mask):
            if self.training:
                veh = veh[mask]
            query = self.get_node_veh(torch.cat((veh.cache_query_feats, veh.cache_motion_feats), dim=-1)) @ self.wq
            key = self.get_node_inf(torch.cat((inf.cache_query_feats, inf.cache_motion_feats), dim=-1)) @ self.wk
            edge = self.get_edge(veh.cache_ref_pts[:, None] - inf.cache_ref_pts) @ self.we
            attn = (query @ key.T).unsqueeze(-1) + edge
            attn = self.ffn(attn).squeeze(-1)
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), attn

        def _query_fusion(self, inf, veh, inf_idx, veh_idx):
            result = veh[veh_idx]
            other = inf[inf_idx]
            for name, module in zip(('cache_query_feats', 'cache_motion_feats', 'cache_query_embeds'),
                                    (self.fuse_feats, self.fuse_motion, self.fuse_embed)):
                setattr(result, name, module(torch.cat((getattr(result, name), getattr(other, name)), dim=-1)))
            result.cache_ref_pts = (result.cache_ref_pts + other.cache_ref_pts) / 2
            return result

    util_path = Path(__file__).resolve().parents[2] / 'code/CoopTrack/projects/mmdet3d_plugin/core/bbox/util.py'
    util = {}
    exec(compile(util_path.read_text(encoding='utf-8'), str(util_path), 'exec'), util)
    torch.manual_seed(0)
    model = nn.Module()
    model.STReasoner = Reasoner()
    model.frozen = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
    scheduler = nn.Module()
    scheduler.values = nn.Parameter(torch.ones(71361))

    def instances(count):
        boxes = torch.zeros(count, 10)
        boxes[:, 7] = 1
        return Instances(cache_query_feats=torch.randn(count, 256) + scheduler.values[0],
                         cache_motion_feats=torch.randn(count, 256), cache_query_embeds=torch.randn(count, 256),
                         cache_ref_pts=torch.rand(count, 3), cache_scores=torch.ones(count), cache_bboxes=boxes)

    inf, veh = instances(2), instances(3)
    mask = torch.tensor([True, False, True])
    indices = torch.tensor([0, 1])
    reasoner = model.STReasoner
    baseline_logits = reasoner._learn_matching(inf, veh, None, None, mask)[2].detach()
    baseline_fusion = reasoner._query_fusion(inf, veh, indices, indices)
    initial_rng = torch.get_rng_state().clone()
    add_structures(model, {'pair_scale': [1.] * 12, 'gate_scale': [1.] * 7})
    assert torch.equal(initial_rng, torch.get_rng_state())
    configure_trainables(model)
    context = {'samples': [{'receiver_state': torch.zeros(2, 4)}], 'matching_cursor': 0}
    remove, proof = install(model, context, util['denormalize_bbox'])
    actual_logits = reasoner._learn_matching(inf, veh, None, None, mask)[2]
    assert torch.equal(actual_logits.detach(), baseline_logits)
    context['matching_cursor'] += 1
    actual_fusion = reasoner._query_fusion(inf, veh, indices, indices)
    for name in ('cache_query_feats', 'cache_motion_feats', 'cache_query_embeds', 'cache_ref_pts'):
        assert torch.equal(getattr(actual_fusion, name).detach(), getattr(baseline_fusion, name).detach())
    assert context['samples'][0]['raw_pair_features'].shape == (2, 2, 12)
    assert context['samples'][0]['raw_gate_features'].shape == (2, 7)
    settings = {'optimizer': {'peak_lr': 1e-4, 'weight_decay': 1e-4}, 'native_lr': 1e-5}
    optimizer = make_optimizer(model, scheduler, settings)
    parameters = parameter_proof(model, scheduler, optimizer)
    assert parameters['total_trainable_parameters'] == 1781990
    snapshot = frozen_snapshot(model)
    before = native_state(model)
    loss = actual_logits.square().mean() + actual_fusion.cache_query_feats.square().mean()
    loss = loss + actual_fusion.cache_motion_feats.square().mean() + actual_fusion.cache_query_embeds.square().mean()
    loss = loss + reasoner.cross_domain_query(torch.cat((actual_fusion.cache_query_feats,
                                                        actual_fusion.cache_query_embeds), dim=-1)).square().mean()
    loss = loss + reasoner.cross_domain_motion(torch.cat((actual_fusion.cache_motion_feats,
                                                         actual_fusion.cache_query_embeds), dim=-1)).square().mean()
    loss.backward()
    norms = gradient_norms(model, scheduler)
    assert all(value > 0 for value in norms.values())
    optimizer.step()
    assert_frozen(model, snapshot, full=True)
    assert state_hash(native_state(model)) != state_hash(before)
    load_native_state(model, before)
    context['matching_cursor'] = 0
    context['samples'][0]['receiver_state'] = torch.empty(0, 4)
    empty = inf[torch.empty(0, dtype=torch.long)]
    assert reasoner._learn_matching(empty, veh, None, None, mask)[2].shape == (2, 0)
    assert len(reasoner._query_fusion(empty, veh, [], [])) == 0
    remove()
    return {'status': 'PASS', 'scope': 'constructed CPU structures; no dataset training',
            'step0_logits_and_fusion_exact': True, 'masked_vehicle_axis': True,
            'lane_cursor_not_modified': True, 'empty_pairs': True, 'frozen_sha256_unchanged': True,
            'all_allowed_module_gradients_nonzero': True, 'parameter_proof': parameters, 'hook_proof': proof,
            'observation_helper_proof': observation_proof}


if __name__ == '__main__':
    import json
    print(json.dumps(self_check(), sort_keys=True, allow_nan=False))
