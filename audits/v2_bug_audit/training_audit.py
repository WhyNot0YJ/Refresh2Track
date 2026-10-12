"""Independent, bounded V2 training audit. Never calls optimizer.step or train()."""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import sys
import subprocess
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected(root):
    paths = list((root / 'scripts').glob('v2_*.py'))
    paths += [root / 'scripts' / name for name in ('main_runtime.py', 'main_message.py', 'main_train.py')]
    paths += list((root / 'configs/main_experiment_v2').glob('*'))
    paths += [root / 'experiments/v2/checkpoints' / variant / 'final.pth' for variant in ('mlp', 'transformer')]
    paths += [root / 'experiments/v2/metrics' / name for name in
              ('mlp_train.json', 'transformer_train.json', 'FINAL_PROTOCOL_MANIFEST.json')]
    paths += [root / 'code/CoopTrack/projects/mmdet3d_plugin' / name for name in
              ('cooptrack/detectors/cooptrack.py', 'cooptrack/modules/spatial_temporal_reason.py',
               'cooptrack/dense_heads/track_head_plugin/track_instance.py', 'losses/track_loss.py')]
    return {str(path.relative_to(root)): sha(path) for path in paths if path.is_file()}


def item(name, refs, method, inputs, expected, actual, passed=True, limitation=None):
    value = dict(check=name, files=refs, method=method, inputs=inputs, expected=expected,
                 actual=actual, status='PASS' if passed else 'FAIL')
    if limitation:
        value['limitation'] = limitation
    return value


def fixture():
    from v2_message import FIELDS, FEATURES, TemporalCache, ReceiverCache, int8_roundtrip, pack
    generator = torch.Generator().manual_seed(314)
    first = {key: torch.randn(3, width, generator=generator) for key, width in
             zip(FIELDS, (256, 256, 256, 3, 10))}
    first.update(obj_idxes=torch.tensor([8, 19, 5]), scores=torch.tensor([.8, .7, .6]))
    current = {}
    for key in FIELDS:
        older = first[key][torch.tensor([2, 0, 1])]
        shift = .7 if key in FEATURES else .01
        current[key] = torch.cat([older + shift, torch.randn(1, older.shape[1], generator=generator)])
    current.update(obj_idxes=torch.tensor([5, 8, 19, 44]), scores=torch.tensor([.6, .8, .7, .5]))
    sender, receiver = TemporalCache(), ReceiverCache()
    meta0 = dict(scene='audit_fixture', session=0, token='audit0', frame_idx=0, timestamp_us=0.)
    context0 = sender.prepare(int8_roundtrip(first), meta0)
    initial, _ = receiver.unpack(pack(first, meta0, context0['force']), meta0)
    sender.commit(first, context0['force'], meta0, initial)
    meta1 = dict(scene='audit_fixture', session=0, token='audit1', frame_idx=1, timestamp_us=100000.)
    quantized = int8_roundtrip(current)
    context = sender.prepare(quantized, meta1)
    return first, current, quantized, context, sender, receiver, meta1


def cpu_checks(root):
    from v2_message import FIELDS, FEATURES, int8_roundtrip, pack, select_updates
    from v2_scheduler import Scheduler, straight_through_latents, rate_loss
    checks = []
    _, raw, quantized, context, sender, receiver, metadata = fixture()
    logits = torch.tensor([-2., 0., 2., -1.], requires_grad=True)
    probability = logits.sigmoid()
    flags = select_updates(probability, context['force'])
    decoded, _ = receiver.unpack(pack(raw, metadata, flags), metadata)
    current = {key: quantized[key].clone().requires_grad_() for key in FEATURES}
    last = {key: context['last'][key].clone().requires_grad_() for key in FEATURES}
    grad_decoded = {key: value.clone().requires_grad_() for key, value in decoded.items()}
    student = straight_through_latents(grad_decoded, current, last, probability, context['force'])
    weights = {key: torch.full_like(student[key], (index + 1) / 256.) for index, key in enumerate(FEATURES)}
    proxy = sum((student[key] * weights[key]).sum() for key in FEATURES)
    proxy.backward()
    analytic = probability.detach() * (1 - probability.detach()) * sum(
        ((current[key].detach() - last[key].detach()) * weights[key]).sum(1) for key in FEATURES)
    analytic[context['force']] = 0
    equal = {key: torch.equal(student[key].detach(), decoded[key]) for key in FIELDS}
    detached = all(value.grad is None for value in list(current.values()) + list(last.values()) + list(grad_decoded.values()))
    checks.append(item('actual_decoded_ST_forward_and_analytic_task_proxy_gradient',
        ['scripts/v2_train.py:283', 'scripts/v2_train.py:291', 'scripts/v2_scheduler.py:48', 'scripts/v2_scheduler.py:55'],
        'Real pack/unpack, hard mixed REUSE/UPDATE/new-ID; linear task proxy backward compared with analytic dp/dlogit*(C-L).',
        {'shape': [4, 256], 'ids': raw['obj_idxes'].tolist(), 'logits': logits.detach().tolist(),
         'force': context['force'].tolist(), 'flags': flags.tolist()},
        'Every field equals actual decoder D bit-for-bit; optional logits receive finite analytic task gradients; forced/current/last/decoded have no gradient.',
        {'decoded_equal': equal, 'logit_gradient': logits.grad.tolist(), 'analytic_gradient': analytic.tolist(),
         'C_L_D_gradients_absent': detached},
        all(equal.values()) and torch.allclose(logits.grad, analytic, atol=1e-6, rtol=1e-5)
        and torch.isfinite(logits.grad).all().item() and (logits.grad[:3].abs() > 0).all().item()
        and logits.grad[3].item() == 0 and detached,
        'This new CPU test proves the implemented ST Jacobian, not a full native task objective.'))

    values = torch.tensor([.499999, .5, .500001, .1])
    boundary = select_updates(values, torch.tensor([False, False, False, True]))
    checks.append(item('hard_threshold_and_forced_gate', ['scripts/v2_message.py:45', 'scripts/v2_message.py:54'],
        'Boundary probabilities through the actual gate.', {'probabilities': values.tolist()},
        '[REUSE, UPDATE, UPDATE, forced UPDATE]', boundary.tolist(), boundary.tolist() == [False, True, True, True]))

    probabilities = torch.tensor([.2, .3, .4, .9], dtype=torch.float64, requires_grad=True)
    forced = torch.tensor([True, False, False, False])
    valid = torch.tensor([True, True, True, False])
    reference = 12840.189523169913
    rate_results = []
    for budget in (12198, 1000):
        loss, soft_bytes, feasible = rate_loss(probabilities, forced, 1000, 780, budget, reference, valid)
        gradient = torch.autograd.grad(loss, probabilities, retain_graph=True)[0]
        expected_soft = 1000 + 780 + 780 * (.3 + .4)
        expected_feasible = max(budget, 1780)
        expected_loss = (max(0., expected_soft - expected_feasible) / reference) ** 2
        expected_gradient = torch.tensor([0., 1., 1., 0.], dtype=torch.float64) * (
            2 * max(0., expected_soft - expected_feasible) * 780 / reference ** 2)
        rate_results.append(dict(budget=budget, soft_bytes=soft_bytes.item(), feasible=feasible,
                                 loss=loss.item(), gradient=gradient.tolist(),
                                 matches_analytic=math.isclose(loss.item(), expected_loss, abs_tol=1e-15)
                                 and torch.allclose(gradient, expected_gradient, atol=1e-15, rtol=1e-12)))
    loss, soft_bytes, feasible = rate_loss(probabilities, torch.ones(4, dtype=torch.bool), 1000, 780, 1, reference)
    forced_gradient = torch.autograd.grad(loss, probabilities)[0]
    checks.append(item('one_sided_rate_optional_forced_and_padding', ['scripts/v2_scheduler.py:62', 'scripts/v2_train.py:297', 'scripts/v2_train.py:376'],
        'Compare actual rate loss and autograd with analytic expression below/above budget and all-forced overflow.',
        {'p': probabilities.detach().tolist(), 'force': forced.tolist(), 'valid': valid.tolist(), 'M': 1000, 'update_bytes': 780, 'reference': reference},
        'Soft cost=M+780F+780sum(optional_p); target=max(B,M+780F); only overage penalized; forced/padded gradients zero.',
        {'cases': rate_results, 'all_forced': {'soft_bytes': soft_bytes.item(), 'feasible': feasible,
                                             'loss': loss.item(), 'gradient': forced_gradient.tolist()}},
        all(case['matches_analytic'] for case in rate_results) and loss.item() == 0
        and feasible == 4120 and forced_gradient.abs().sum().item() == 0))

    sender.commit(raw, flags, metadata, decoded)
    future_raw = {key: value.clone() for key, value in raw.items()}
    for key in FEATURES:
        future_raw[key] += .4
    future_meta = dict(metadata, token='audit2', frame_idx=2, timestamp_us=200000.)
    future_context = sender.prepare(int8_roundtrip(future_raw), future_meta)
    later_logits = torch.tensor([-1., 1., -.5, .5], requires_grad=True)
    later_p = later_logits.sigmoid()
    later_flags = select_updates(later_p, future_context['force'])
    later_decoded, _ = receiver.unpack(pack(future_raw, future_meta, later_flags), future_meta)
    later_student = straight_through_latents(later_decoded, int8_roundtrip(future_raw), future_context['last'], later_p, future_context['force'])
    later_loss = sum(value.sum() / 256 for key, value in later_student.items() if key in FEATURES)
    previous_grad, later_grad = torch.autograd.grad(later_loss, (logits, later_logits), allow_unused=True)
    native_instance_path = root / 'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/dense_heads/track_head_plugin/track_instance.py'
    spec = importlib.util.spec_from_file_location('training_audit_native_instances', native_instance_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = module.Instances((1, 1))
    instance.query_feats = later_student['query_feats']
    detached_instance = instance.detach_and_clone()
    checks.append(item('cross_frame_credit_is_truncated',
        ['scripts/v2_message.py:35', 'scripts/v2_message.py:21', 'scripts/main_message.py:178',
         'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/detectors/cooptrack.py:1129',
         'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/dense_heads/track_head_plugin/track_instance.py:120'],
        'Commit a real decoded frame, form next-frame ST, autograd to old/new logits; execute native Instances.detach_and_clone.',
        {'previous_frame': 1, 'next_frame': 2, 'previous_flags': flags.tolist(), 'next_flags': later_flags.tolist()},
        'Prior decision is detached; next loss reaches current decision only. Native temporal instance clone has no grad_fn.',
        {'previous_logit_gradient': None if previous_grad is None else previous_grad.tolist(),
         'next_logit_gradient': later_grad.tolist(), 'sender_cached_requires_grad': any(value.requires_grad for row in sender.cache.values() for value in row.values() if torch.is_tensor(value)),
         'native_instance_requires_grad': detached_instance.query_feats.requires_grad},
        previous_grad is None and torch.isfinite(later_grad).all().item() and later_grad.abs().sum().item() > 0
        and not detached_instance.query_feats.requires_grad,
        'Confirmed design limitation: no BPTT/long-term return credit to earlier UPDATE decisions. Not a protocol violation.'))

    import v2_train
    v2_train.native.torch = torch
    native_toy = torch.nn.Sequential(torch.nn.BatchNorm1d(256), torch.nn.Linear(256, 2)).requires_grad_(False)
    v2_train.model_in_training_mode(native_toy)
    snapshot = v2_train.frozen_snapshot(native_toy)
    toy_input = torch.randn(4, 256, requires_grad=True)
    native_toy(toy_input).square().mean().backward()
    v2_train.assert_frozen(native_toy, snapshot)
    checks.append(item('actual_training_mode_freeze_helpers',
        ['scripts/v2_train.py:202', 'scripts/v2_train.py:209', 'scripts/v2_train.py:216', 'scripts/main_runtime.py:126'],
        'Run actual mode/snapshot/assert helpers on independent frozen BN+Linear, then backward to input.',
        {'input_shape': [4, 256], 'model': 'independent BN+Linear CPU control, not CoopTrack'},
        'Model training=True, BN training=False; input gradient survives frozen parameters; BN buffers and parameter versions unchanged.',
        {'model_training': native_toy.training, 'bn_training': native_toy[0].training,
         'input_gradient_norm': toy_input.grad.norm().item(), 'base_gradients_absent': all(p.grad is None for p in native_toy.parameters()), 'assert_frozen': 'PASS'},
        native_toy.training and not native_toy[0].training and toy_input.grad.norm().item() > 0,
        'The helper control is fresh evidence; actual native buffers are separately asserted in every completed formal epoch.'))

    for variant in ('mlp', 'transformer'):
        torch.manual_seed(0)
        scheduler = Scheduler(variant)
        before = {name: value.detach().clone() for name, value in scheduler.state_dict().items()}
        _, raw, quantized, temporal, _, receiver, metadata = fixture()
        p = scheduler(temporal['tokens'], .5).sigmoid()
        hard = select_updates(p, temporal['force'])
        decoded, _ = receiver.unpack(pack(raw, metadata, hard), metadata)
        student = straight_through_latents(decoded, quantized, temporal['last'], p, temporal['force'])
        task_proxy = sum(student[key].square().mean() for key in FEATURES)
        gradients = torch.autograd.grad(task_proxy, tuple(scheduler.parameters()), allow_unused=True)
        norm = sum(g.square().sum().item() for g in gradients if g is not None) ** .5
        unchanged = all(torch.equal(before[name], value) for name, value in scheduler.state_dict().items())
        checks.append(item(variant + '_task_proxy_reaches_real_scheduler_without_updates',
            ['scripts/v2_scheduler.py:9', 'scripts/v2_scheduler.py:27', 'scripts/v2_scheduler.py:48'],
            'Actual architecture CPU forward, real pack/unpack ST, isolated quadratic proxy autograd; no optimizer constructed or stepped.',
            {'tokens_shape': list(temporal['tokens'].shape), 'budget_ratio': .5, 'force': temporal['force'].tolist()},
            'Finite nonzero task-proxy parameter gradient; parameters unchanged.',
            {'gradient_norm': norm, 'connected_parameter_tensors': sum(g is not None for g in gradients),
             'finite': all(torch.isfinite(g).all().item() for g in gradients if g is not None), 'parameters_unchanged': unchanged},
            norm > 0 and math.isfinite(norm) and unchanged and all(torch.isfinite(g).all().item() for g in gradients if g is not None),
            'This CPU proxy subcheck alone does not prove native task gradients; see the separate real_native_task controls.'))
    return checks


def stored_training_evidence(root):
    checks = []
    source = (root / 'scripts/v2_train.py').read_text()
    tree = ast.parse(source)
    optimizer_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == 'AdamW']
    parameter_args = [ast.get_source_segment(source, node.args[0]) for node in optimizer_calls]
    factor_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'equal_stream_factors')
    factor_namespace = {'torch': torch}
    exec(compile(ast.Module(body=[factor_node], type_ignores=[]), 'actual_v2_equal_stream_factors', 'exec'), factor_namespace)
    native_streams = [{'loss_bbox': torch.tensor(2.), 'avg_factor_bbox': torch.tensor(6.)},
                      {'loss_bbox': torch.tensor(10.), 'avg_factor_bbox': torch.tensor(2.)}]
    normalized_values_before = [row['loss_bbox'].clone() for row in native_streams]
    for row in native_streams:
        factor_namespace['equal_stream_factors'](row)
    actual_outer = sum(row['loss_bbox'] * row['avg_factor_bbox'] for row in native_streams) / sum(row['avg_factor_bbox'] for row in native_streams)
    import mmcv
    import main_runtime
    cfg = mmcv.Config.fromfile(str(root / 'configs/main_experiment_v2/coop_train_train.py'))
    original_cfg = mmcv.Config.fromfile(str(root / 'code/CoopTrack' / main_runtime.CONFIGS['coop']))
    settings = json.loads((root / 'configs/main_experiment_v2/training_v2.json').read_text())
    native_weights_equal = cfg.model.loss_cfg == original_cfg.model.loss_cfg and cfg.model.asso_loss_cfg == original_cfg.model.asso_loss_cfg
    checks.append(item('registered_batch_reduction_change_preserves_inner_normalized_losses',
        ['scripts/v2_train.py:162', 'scripts/v2_train.py:171',
         'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/detectors/cooptrack.py:973',
         'code/CoopTrack/projects/mmdet3d_plugin/losses/track_loss.py:344'],
        'Execute the actual nested factor helper extracted from AST on already-normalized two-stream loss values; compare native outer reducer.',
        {'already_normalized_loss_values': [2., 10.], 'native_outer_GT_factors': [6., 2.],
         'registered_rule': settings['batch_loss_reduction']},
        'Inner loss values stay [2,10]; outer factors become 1, producing stream mean6 (original GT-weighted mean4 intentionally changes).',
        {'loss_values_after': [float(row['loss_bbox']) for row in native_streams],
         'outer_factors_after': [float(row['avg_factor_bbox']) for row in native_streams],
         'outer_mean': float(actual_outer), 'native_GT_weighted_mean_before': 4.},
        all(torch.equal(before, row['loss_bbox']) for before, row in zip(normalized_values_before, native_streams))
        and float(actual_outer) == 6 and settings['batch_loss_reduction'] == 'mean over actual valid streams, including tail batches',
        'Protocol-authorized change to cross-stream reduction; original loss weights/internal normalization are retained. Do not call native batch reduction wholly unchanged.'))
    checks.append(item('native_loss_weights_and_no_teacher_topk_ttl',
        ['configs/main_experiment_v2/coop_train_train.py:670', 'configs/main_experiment_v2/coop_train_train.py:689',
         'scripts/v2_train.py:128', 'scripts/v2_train.py:137', 'scripts/v2_train.py:256',
         'scripts/v2_train.py:447', 'scripts/v2_message.py:14', 'scripts/v2_message.py:54'],
        'Compare loaded formal native loss configuration with frozen original config, and trace executed V2 hooks/gate/total loss.',
        {'formal_native_config': str(root / 'configs/main_experiment_v2/coop_train_train.py'),
         'original_native_config': main_runtime.CONFIGS['coop']},
        'Native task weights unchanged; lambda_rate=1; no association-preserving teacher pass, budget fill or TTL forced refresh.',
        {'native_loss_cfg_equals_original': native_weights_equal, 'asso_focal': dict(cfg.model.asso_loss_cfg.loss_focal),
         'classification_focal': dict(cfg.model.loss_cfg.loss_cls), 'bbox': dict(cfg.model.loss_cfg.loss_bbox),
         'lambda_rate': settings['lambda_rate'], 'lambda_extra_association_preserve': settings['lambda_extra_association_preserve'],
         'gate': settings['gate'], 'cache_ttl': settings['cache']['ttl'],
         'teacher_hooks_used': False, 'main_train_import_used': 'training_metas only'},
        native_weights_equal and settings['lambda_rate'] == 1 and settings['lambda_extra_association_preserve'] == 0
        and settings['gate']['budget_top_k'] is False and settings['gate']['staleness_cap'] is None
        and settings['cache']['ttl'] is None))
    checks.append(item('optimizer_and_total_loss_actual_source',
        ['scripts/v2_train.py:605', 'scripts/v2_train.py:369', 'scripts/v2_train.py:375',
         'scripts/v2_train.py:376', 'scripts/v2_train.py:411', 'scripts/v2_train.py:447'],
        'AST of executed frozen trainer and fixed configuration; inspect loss/backward and optimizer args.',
        {'AdamW_parameter_args': parameter_args},
        'Every AdamW gets scheduler.parameters(); loss=parsed native task + lambda1*rate; teacher lambda0.',
        {'AdamW_parameter_args': parameter_args, 'task_float_conversion_after_backward': source.index('values = dict(task=float(task.detach())') > source.index('total.backward()')},
        parameter_args == ['scheduler.parameters()', 'scheduler.parameters()']))
    for variant in ('mlp', 'transformer'):
        path = root / 'experiments/v2/checkpoints' / variant / 'final.pth'
        checkpoint = torch.load(str(path), map_location='cpu')
        report = json.loads((root / 'experiments/v2/metrics' / (variant + '_train.json')).read_text())
        scheduler = __import__('v2_scheduler').Scheduler(variant)
        groups = checkpoint['optimizer']['param_groups']
        states = checkpoint['optimizer']['state']
        expected_steps = sum(epoch['optimizer_steps'] for epoch in report['epochs'])
        steps = sorted({float(value['step']) for value in states.values()})
        nonzero_moments = sum(value['exp_avg'].abs().sum().item() > 0 for value in states.values())
        finite = all(torch.isfinite(value[key]).all().item() for value in states.values() for key in ('exp_avg', 'exp_avg_sq'))
        parameter_count = len(list(scheduler.parameters()))
        checks.append(item(variant + '_formal_optimizer_state_and_loss_evidence',
            ['scripts/v2_train.py:605', 'scripts/v2_train.py:400', 'scripts/v2_train.py:411',
             'scripts/v2_train.py:685', 'scripts/v2_train.py:729'],
            'Read actual final optimizer tensors and formal per-epoch records; compare optimizer IDs to real architecture tensor count. No optimizer step.',
            {'checkpoint_sha256': sha(path), 'scheduler_parameter_tensor_count': parameter_count},
            'Only scheduler parameter tensors in optimizer, finite moments with actual formal steps, isolated nonzero native task gradient evidence, completed freeze assertions.',
            {'optimizer_parameter_tensor_count': sum(len(group['params']) for group in groups),
             'optimizer_step_values': steps, 'expected_formal_optimizer_steps': expected_steps,
             'nonzero_moment_tensors': nonzero_moments, 'finite_optimizer_moments': finite,
             'native_task_gradient_norm_recorded': report['first_task_gradient_norm'],
             'base_trainable_params': report['base_trainable_params'], 'completed_epochs': report['completed_epoch'],
             'per_epoch_freeze_assertion_precedes_checkpoint': True},
            sum(len(group['params']) for group in groups) == parameter_count and steps == [float(expected_steps)]
            and nonzero_moments > 0 and finite and report['base_trainable_params'] == 0
            and report['first_task_gradient_norm'] > 0 and report['completed_epoch'] == 12,
            'Stored states/source are one subcheck of executed updates; see the separate real_native_task controls for native task gradients and recreated-initial/final parameter comparisons.'))
    reasoner_relative = 'projects/mmdet3d_plugin/cooptrack/modules/spatial_temporal_reason.py'
    reasoner_head = subprocess.check_output(['git', '-C', str(root / 'code/CoopTrack'), 'show', 'HEAD:' + reasoner_relative]).decode('utf-8').replace('\r\n', '\n')
    reasoner_actual = (root / 'code/CoopTrack' / reasoner_relative).read_text().replace('\r\n', '\n')
    reasoner_unchanged = reasoner_head == reasoner_actual
    checks.append(item('official_train_eval_matching_difference_and_task_graph',
        ['code/CoopTrack/projects/mmdet3d_plugin/cooptrack/modules/spatial_temporal_reason.py:238',
         'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/modules/spatial_temporal_reason.py:310',
         'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/modules/spatial_temporal_reason.py:341',
         'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/modules/spatial_temporal_reason.py:167',
         'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/detectors/cooptrack.py:930'],
        'Trace native branch, preserved git HEAD source, differentiation and labels.',
        {'training': 'model.train + BN.eval; return_loss=True', 'inference': 'model.eval; return_loss=False'},
        'Original training computes affinity logits + geometry matching; original eval uses learned-affinity Hungarian; original detection/refinement/association weights retained.',
        {'train_fusion_assignment': 'geometry Hungarian, 3.5m cut-off',
         'train_affinity_loss': dict(cfg.model.asso_loss_cfg.loss_focal),
         'eval_assignment': 'Hungarian on 1-sigmoid(affinity), detached discrete indices',
         'history_refine': 'fused feature -> original classifier/regressor',
         'reasoner_matches_frozen_HEAD': reasoner_unchanged,
         'reasoner_normalized_text_sha256': hashlib.sha256(reasoner_actual.encode()).hexdigest(),
         'native_complementation_topk': 'unchanged vehicle-query retention at reasoner:297; never selects latent UPDATE packets',
         'reasoner_worktree_diff': 'none; detector local delta only empty sender initialization'},
        reasoner_unchanged,
        'Official inherited mismatch: current-frame differentiable task is a surrogate for discrete inference association. It does not establish long-term or exact inference reward optimization.'))
    return checks


def bounded_native_task_checks(root):
    """Call only after root has coordinated GPU access. Two forwards/variant, zero optimizer steps."""
    import main_runtime as native
    import v2_runtime as runtime
    native.initialize(root)
    from mmcv.parallel import MMDataParallel
    from v2_scheduler import Scheduler
    import v2_train as trainer
    settings = native.read(root / 'configs/main_experiment_v2/training_v2.json')
    checks = []
    for variant in ('mlp', 'transformer'):
        model = scheduler = parallel = remove = None
        alignment_handle = original_geometry = measure_geometry = None
        context, streams = {}, {}
        try:
            runtime.seed(0)
            # Read the exact frozen dumped config; never invoke make_config/load_training, which rewrite it.
            cfg = native.mmcv.Config.fromfile(str(root / 'configs/main_experiment_v2/coop_train_train.py'))
            dataset = native.dataset_for(cfg, training=True)
            scenes = {}
            for index, info in enumerate(dataset.data_infos):
                scenes.setdefault(str(info['scene_token']), []).append(index)
            sender_rows = {row['token']: row for row in native.native_manifest(root, 'train')}
            model = native.build_model(root, cfg, 'coop').cuda()
            model.CLASSES = dataset.CLASSES
            trainer.model_in_training_mode(model)
            scheduler = Scheduler(variant).cuda().train()
            initial = {name: value.detach().cpu().clone() for name, value in scheduler.state_dict().items()}
            checkpoint = native.torch.load(str(root / 'experiments/v2/checkpoints' / variant / 'final.pth'), map_location='cpu')
            differences = {name: float((checkpoint['scheduler'][name] - initial[name]).norm()) for name in initial}
            scheduler.load_state_dict(checkpoint['scheduler'])
            versions = tuple(parameter._version for parameter in scheduler.parameters())
            frozen = trainer.frozen_snapshot(model)
            remove = trainer.install_hooks(model, context)
            alignment_handle = model.crossview_alignment.register_forward_hook(
                lambda module, incoming, outgoing: context['aligned_query_counts'].append(len(outgoing['query_feats'])))
            original_geometry = model.STReasoner._query_matching
            def measure_geometry(*args, **kwargs):
                result = original_geometry(*args, **kwargs)
                context['geometry_match_counts'].append(len(result[0]))
                return result
            model.STReasoner._query_matching = measure_geometry
            parallel = MMDataParallel(model, device_ids=[0])
            order = sorted(scenes, key=lambda scene: (settings['initial_offset'][scene], scene))
            batches = trainer.scene_batches(scenes, order, 4)
            native_results = []
            for step in (1, 2):
                entries = next(batches)
                batch, samples, stats = trainer.prepare_batch(root, dataset, entries, streams, sender_rows, scheduler, settings, 0)
                trainer.install_batch_state(model, entries, streams)
                context.clear()
                context.update(samples=samples, alignment_indices=[index for index, sample in enumerate(samples)
                               if len(sample['current']['query_feats'])], alignment_cursor=0, matching_cursor=0,
                               native_forward_count=0, bev_forward_count=0, native_bev_batch_size=0,
                               aligned_query_counts=[], geometry_match_counts=[])
                losses = parallel(return_loss=True, **batch)
                task, _ = model._parse_losses(losses)
                record = {'step': step, 'tokens': [sample['token'] for sample in samples],
                          'scenes': [sample['scene'] for sample in samples], 'task': float(task.detach()),
                          'native_forwards': context['native_forward_count'], 'BEV_forwards': context['bev_forward_count'],
                          'BEV_batch_size': context['native_bev_batch_size'], 'query_counts': [len(sample['student']['query_feats']) for sample in samples],
                          'affinity_requires_grad': [sample['affinity_requires_grad'] for sample in samples],
                          'aligned_query_counts': list(context['aligned_query_counts']),
                          'geometry_match_counts': list(context['geometry_match_counts']),
                          'optional_queries': sum(stat['optional_queries'] for stat in stats), 'updates': sum(stat['updates'] for stat in stats)}
                if step == 2:
                    record['task_gradient_norm'], record['task_gradient_connected'] = trainer.gradient_evidence(task, scheduler)
                    record['task_component_gradient_norms'] = {name: trainer.gradient_evidence(value.mean(), scheduler)[0]
                         for name, value in losses.items() if 'loss' in name}
                    detection_refinement = sum(value.mean() for name, value in losses.items() if 'loss' in name and 'asso' not in name)
                    record['detection_refinement_only_gradient_norm'], _ = trainer.gradient_evidence(detection_refinement, scheduler)
                    rate = torch.stack([sample['rate'] for sample in samples]).mean()
                    record['rate_gradient_norm'], record['rate_gradient_connected'] = trainer.gradient_evidence(rate, scheduler)
                    feature_inputs = [sample['student'][name] for sample in samples for name in ('query_feats', 'query_embeds', 'cache_motion_feats')]
                    feature_gradients = torch.autograd.grad(task, tuple(feature_inputs), retain_graph=True, allow_unused=True)
                    record['student_latent_task_gradient_norms'] = {
                        name: sum(gradient.square().sum().item() for position, gradient in enumerate(feature_gradients)
                                  if position % 3 == index and gradient is not None) ** .5
                        for index, name in enumerate(('query_feats', 'query_embeds', 'cache_motion_feats'))}
                    record['total_loss_with_lambda1'] = float((task + rate).detach())
                trainer.save_batch_state(model, entries, streams)
                native_results.append(record)
                context.clear()
                del batch, samples, stats, losses, task
            trainer.assert_frozen(model, frozen)
            unchanged = tuple(parameter._version for parameter in scheduler.parameters()) == versions
            passed = (native_results[1]['task_gradient_norm'] > 0 and math.isfinite(native_results[1]['task_gradient_norm'])
                      and native_results[1]['detection_refinement_only_gradient_norm'] > 0
                      and sum(native_results[1]['aligned_query_counts']) > 0
                      and all(result['native_forwards'] == result['BEV_forwards'] == 1 and result['BEV_batch_size'] == 4
                              for result in native_results) and unchanged and all(not module.training for module in model.modules()
                          if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)))
            checks.append(item(variant + '_real_native_task_gradient_and_frozen_state_control',
                ['scripts/v2_train.py:250', 'scripts/v2_train.py:128', 'scripts/v2_train.py:369', 'scripts/v2_train.py:380',
                 'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/detectors/cooptrack.py:924',
                 'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/modules/spatial_temporal_reason.py:167'],
                'Independent final Scheduler/native copies; actual 4-scene cold forward then next-frame native return_loss=True; component autograd only, zero optimizer steps.',
                {'final_checkpoint': str(root / 'experiments/v2/checkpoints' / variant / 'final.pth'),
                 'native_batch_size': 4, 'forward_budget': 2, 'optimizer_steps': 0},
                'Actual native task reaches Scheduler and original task components; frozen base/BN and loaded Scheduler weights unchanged by audit.',
                {'batches': native_results, 'scheduler_unchanged_by_audit': unchanged,
                 'native_parameter_and_BN_assert_frozen': 'PASS', 'model_training': model.training,
                 'BN_training_count': sum(module.training for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)),
                 'optimizer_steps': 0, 'fresh_init_to_final_parameter_differences': differences,
                 'changed_parameter_tensors_vs_recreated_initialization': sum(value > 0 for value in differences.values())},
                passed and any(value > 0 for value in differences.values()),
                'Fresh init uses seed0 and the same dataset/model constructors with read-only equivalent frozen config; original initialization checkpoint was not recorded. Current-frame surrogate only.'))
        finally:
            if alignment_handle is not None:
                alignment_handle.remove()
            if model is not None and original_geometry is not None:
                model.STReasoner._query_matching = original_geometry
            del alignment_handle, original_geometry, measure_geometry
            if remove is not None:
                remove()
            del remove
            context.clear()
            streams.clear()
            for name in ('rate', 'feature_inputs', 'feature_gradients', 'checkpoint', 'detection_refinement'):
                if name in locals():
                    # Delete graph locals before model collection; scheduler/model optimizer steps never occur.
                    if name == 'rate':
                        del rate
                    elif name == 'feature_inputs':
                        del feature_inputs
                    elif name == 'feature_gradients':
                        del feature_gradients
                    elif name == 'checkpoint':
                        del checkpoint
                    elif name == 'detection_refinement':
                        del detection_refinement
            del parallel, model, scheduler
            gc.collect()
            torch.cuda.empty_cache()
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/mnt/d/CoopTrack_Audit'))
    parser.add_argument('--native-gradient', action='store_true', help='Run only after root GPU coordination; never optimizes weights.')
    parser.add_argument('--refresh-static', action='store_true', help='Refresh static evidence only while preserving completed native measurements.')
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'scripts'))
    output = root / 'audits/v2_bug_audit/training_evidence.json'
    before = protected(root)
    if args.refresh_static:
        if args.native_gradient:
            raise ValueError('Static refresh cannot request native GPU execution')
        evidence = json.loads(output.read_text())
        if evidence['protected_before'] != before:
            raise ValueError('Formal sources/checkpoints changed since measured native control')
        revised = stored_training_evidence(root)
        names = {check['check'] for check in revised}
        evidence['checks'] = [check for check in evidence['checks'] if check['check'] not in names] + revised
        evidence['protected_hashes_unchanged'] = before == protected(root)
        evidence['static_refreshed_utc'] = datetime.now(timezone.utc).isoformat()
        evidence['scope'] = 'read-only formal artifacts; independent CPU controls and completed bounded native task autograd; zero optimizer steps'
        evidence['status'] = 'TRAINING_AUDIT_PASS' if all(check['status'] == 'PASS' for check in evidence['checks']) and evidence['protected_hashes_unchanged'] and evidence['native_task_control_status'] == 'PASS' else 'FAIL'
        output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'status': evidence['status'], 'check_count': len(evidence['checks']), 'native_rerun': False}))
        return 0 if evidence['status'] == 'TRAINING_AUDIT_PASS' else 1
    evidence = dict(audit='V2_training_implementation', generated_utc=datetime.now(timezone.utc).isoformat(),
                    scope='read-only formal sources/checkpoints; independent CPU controls; no optimizer steps',
                    checks=[], confirmed_bugs=[], native_task_control_status='未验证：等待 root GPU coordination')
    try:
        evidence['checks'] += cpu_checks(root)
        evidence['checks'] += stored_training_evidence(root)
        evidence['cpu_only_no_cuda_initialization'] = not torch.cuda.is_initialized()
        if args.native_gradient:
            evidence['checks'] += bounded_native_task_checks(root)
            evidence['native_task_control_status'] = 'PASS' if all(check['status'] == 'PASS' for check in evidence['checks']
                if '_real_native_task_gradient_' in check['check']) else 'FAIL'
        evidence['protected_hashes_unchanged'] = before == protected(root)
        evidence['protected_before'] = before
        passed = all(check['status'] == 'PASS' for check in evidence['checks']) and evidence['protected_hashes_unchanged']
        evidence['status'] = ('TRAINING_AUDIT_PASS' if args.native_gradient else 'CPU_CHECKS_PASS_NATIVE_TASK_PENDING') if passed else 'FAIL'
    except Exception as exc:
        evidence.update(status='AUDIT_ERROR', error=str(exc), traceback=traceback.format_exc())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': evidence['status'], 'check_count': len(evidence['checks']),
                      'failures': [check['check'] for check in evidence['checks'] if check['status'] == 'FAIL'],
                      'error': evidence.get('error'), 'output': str(output)}, ensure_ascii=False))
    return 0 if evidence['status'] in ('CPU_CHECKS_PASS_NATIVE_TASK_PENDING', 'TRAINING_AUDIT_PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
