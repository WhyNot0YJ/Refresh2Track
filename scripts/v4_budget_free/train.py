"""Train one complete budget-free V4 epoch; validation and selection run in the orchestrator."""
import argparse
import csv
import gc
import json
import math
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import main_runtime as native
import v2_runtime as inherited
import v2_train as training
from v2_message import (FIELDS, FEATURES, ReceiverCache, TemporalCache, int8_roundtrip,
                        message_sizes, pack, select_updates)
from v2_scheduler import straight_through_latents
import native_hooks
import runtime
from scheduler import Scheduler, communication_loss

MODELS = runtime.MODELS
DIAGNOSTIC_STEPS = (2, 10)
CORRECTNESS_MAX_CANDIDATE_BATCHES = 7


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def load_training(root):
    cfg = runtime.make_config(root, 'coop', 'train', training=True, batch_size=4)
    dataset = native.dataset_for(cfg, training=True)
    scenes = {}
    for index, info in enumerate(dataset.data_infos):
        scenes.setdefault(str(info['scene_token']), []).append(index)
    official = set(runtime.read(Path(root) / 'code/CoopTrack/data/split_datas/cooperative-split-data-spd.json')['batch_split']['train'])
    if len(dataset) != 7362 or len(scenes) != 46 or set(scenes) != official:
        raise ValueError('V4 requires all 46 official train scenes / 7362 samples')
    for scene, indices in scenes.items():
        stamps = [dataset.data_infos[index]['timestamp'] for index in indices]
        if stamps != sorted(stamps):
            raise ValueError('Train scene is not chronological: ' + scene)
    sender_rows = {row['token']: row for row in native.native_manifest(root, 'train')}
    if any(str(info['token_inf']) not in sender_rows for info in dataset.data_infos):
        raise ValueError('Official train cache mapping incomplete')
    return cfg, dataset, scenes, sender_rows


def prepare_batch(root, dataset, entries, streams, sender_rows, scheduler, settings, epoch, model_name):
    from mmcv.parallel import collate
    from main_train import training_metas
    torch = native.torch
    prepared, expected, samples, records = [], [], [], []
    for entry in entries:
        scene, index = entry['scene'], entry['index']
        if entry['first']:
            if scene in streams:
                raise ValueError('New scene entered an already live lane')
            streams[scene] = {'sender': TemporalCache(), 'receiver': ReceiverCache(),
                              'native': training.native_slot(),
                              'rng': native_hooks.scene_rng(settings['training_seed'], scene, epoch)}
        stream = streams[scene]
        info = dataset.data_infos[index]
        with native_hooks.stream_rng(stream):
            data = dataset.prepare_train_data(index)
        if data is None:
            raise ValueError('Native pipeline rejected official training sample: ' + str(info['token']))
        metas = training_metas(data['img_metas'])
        if len(metas) != 1 or str(metas[0]['sample_idx']) != str(info['token']) or str(metas[0]['sample_idx_inf']) != str(info['token_inf']):
            raise ValueError('Native pipeline substituted/reordered an official frame')
        metadata = sender_rows[str(info['token_inf'])]
        if str(metadata['scene']) != scene:
            raise ValueError('Sender/vehicle scene mismatch')
        raw = native.fields(Path(root) / 'query_cache/train' / (metadata['token'] + '.pkl'))
        raw = {name: raw[name] for name in FIELDS + ('obj_idxes', 'scores')}
        expected.append({name: raw[name] for name in FIELDS})
        current = int8_roundtrip(raw)
        temporal = stream['sender'].prepare(current, metadata)
        probability = scheduler(temporal['tokens'].cuda()).sigmoid()
        if not torch.isfinite(probability).all():
            raise FloatingPointError('Nonfinite Scheduler UPDATE probability')
        flags = select_updates(probability.detach().cpu(), temporal['force'])
        mandatory, cost, _ = message_sizes(raw, metadata)
        packet = pack(raw, metadata, flags)
        if cost != 780 or len(packet) != mandatory + cost * int(flags.sum()):
            raise ValueError('Actual packet differs from mandatory + 780 * UPDATE')
        decoded, _ = stream['receiver'].unpack(packet, metadata)
        for name in FEATURES:
            if not torch.equal(decoded[name][flags], current[name][flags]) or not torch.equal(decoded[name][~flags], temporal['last'][name][~flags]):
                raise ValueError('Actual UPDATE/REUSE does not use current INT8/received cache')
        current_gpu = {name: value.cuda() for name, value in current.items() if torch.is_tensor(value)}
        last_gpu = {name: value.cuda() for name, value in temporal['last'].items()}
        decoded_gpu = {name: value.cuda() for name, value in decoded.items()}
        forced = temporal['force'].cuda()
        student = straight_through_latents(decoded_gpu, current_gpu, last_gpu, probability, forced)
        if any(not torch.equal(student[name].detach(), decoded_gpu[name]) for name in FIELDS):
            raise ValueError('Hard STE forward differs from actual decoded packet')
        comm, estimated = communication_loss(probability, forced, settings['full_int8_mean_bytes'])
        changed = bool((~forced).any()) and any(bool((current_gpu[name][~forced] != last_gpu[name][~forced]).any()) for name in FEATURES)
        samples.append({'scene': scene, 'token': str(info['token']),
                        'current': current_gpu, 'student': student, 'comm': comm,
                        'optional': bool((~forced).any()), 'optional_changed': changed,
                        'received_queries': 0, 'affinity_requires_grad': False})
        optional = ~temporal['force']
        optional_p = probability.detach().cpu()[optional]
        actual_optional = 780 * int((flags & optional).sum())
        records.append({'scene': scene, 'token': str(info['token']),
                        'sender_token': metadata['token'], 'session': metadata['session'],
                        'frame_idx': metadata['frame_idx'], 'timestamp_us': metadata['timestamp_us'],
                        'queries': len(flags), 'updates': int(flags.sum()),
                        'forced_init': int(temporal['force'].sum()),
                        'optional_queries': int(optional.sum()),
                        'optional_updates': int((flags & optional).sum()),
                        'actual_bytes': len(packet),
                        'geometry_bytes': 52 * len(flags), 'latent_bytes': 780 * int(flags.sum()),
                        'feature_payload_bytes': 52 * len(flags) + 780 * int(flags.sum()),
                        'forced_init_bytes': 780 * int(temporal['force'].sum()),
                        'estimated_optional_bytes': float(estimated.detach()),
                        'actual_optional_bytes': actual_optional,
                        'soft_hard_gap_bytes': float(estimated.detach()) - actual_optional,
                        'optional_probabilities': optional_p.tolist(),
                        'near_threshold_count': int(((optional_p >= .45) & (optional_p <= .55)).sum()),
                        'comm_loss': float(comm.detach()),
                        'weighted_comm_loss': settings['lambda_comm'] * float(comm.detach()),
                        'cache_ages': temporal['elapsed'][optional].tolist()})
        stream['sender'].commit(raw, flags, metadata, decoded)
        prepared.append(data)
    batch = collate(prepared, samples_per_gpu=len(prepared))
    inherited.assert_loaded_queries(batch, expected)
    return batch, samples, records


def parameter_proof(model, scheduler, optimizer):
    torch = native.torch
    scheduler_named = list(scheduler.named_parameters())
    expected = {id(parameter) for _, parameter in scheduler_named}
    actual = [parameter for group in optimizer.param_groups for parameter in group['params']]
    if len(actual) != len(expected) or {id(parameter) for parameter in actual} != expected:
        raise ValueError('Optimizer contains parameters outside Scheduler')
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('Native CoopTrack parameter is trainable')
    if any(module.training for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)):
        raise ValueError('Native BatchNorm is in training mode')
    value = {'total_parameters': sum(p.numel() for p in model.parameters()) + sum(p.numel() for _, p in scheduler_named),
             'trainable_scheduler_parameters': sum(p.numel() for _, p in scheduler_named),
             'frozen_cooptrack_parameters': sum(p.numel() for p in model.parameters()),
             'optimizer_parameter_names': [name for name, _ in scheduler_named],
             'native_trainable_parameters': 0, 'batchnorm_statistics_frozen': True}
    if value['trainable_scheduler_parameters'] != 71361:
        raise ValueError('V4 MLP architecture/parameter count changed')
    emit(dict(phase='parameter_proof', **value))
    return value


def run_batch(root, model_name, epoch, step, entries, dataset, streams, sender_rows,
              model, scheduler, parallel, optimizer, settings, context, evidence, optimize=True, diagnose=False):
    torch = native.torch
    optimizer.zero_grad(set_to_none=True)
    batch, samples, records = prepare_batch(root, dataset, entries, streams, sender_rows,
                                           scheduler, settings, epoch, model_name)
    training.install_batch_state(model, entries, streams)
    context.clear()
    context.update(samples=samples, streams=streams,
                   alignment_indices=[index for index, item in enumerate(samples) if len(item['current']['query_feats'])],
                   alignment_cursor=0, matching_cursor=0, native_forward_count=0,
                   bev_forward_count=0, native_bev_batch_size=0)
    losses = parallel(return_loss=True, **batch)
    task, _ = model._parse_losses(losses)
    if context['native_forward_count'] != 1 or context['bev_forward_count'] != 1:
        raise ValueError('V4 training requires one genuinely batched native forward')
    if context['alignment_cursor'] != len(context['alignment_indices']) or context['matching_cursor'] != len(samples):
        raise ValueError('Native hooks failed to visit each real lane exactly once')
    if any(not item.get('native_loss_captured') for item in samples):
        raise ValueError('Exact native per-frame task loss was not captured')
    mean_task = sum(item['task_loss'] for item in samples) / len(samples)
    if not math.isclose(float(task.detach()), mean_task, rel_tol=3e-5, abs_tol=1e-5):
        raise ValueError('Captured per-frame loss does not reproduce original native task reducer')
    comm = torch.stack([item['comm'] for item in samples]).mean()
    total = task + settings['lambda_comm'] * comm
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite V4 training loss')
    diagnostic = step in DIAGNOSTIC_STEPS or diagnose
    verify_task = optimize and not evidence.get('task_gradient_verified') and any(item['optional_changed'] for item in samples)
    task_norm = comm_norm = None
    if diagnostic or verify_task:
        task_norm, _ = training.gradient_evidence(task, scheduler)
        comm_norm, _ = training.gradient_evidence(comm, scheduler)
    if verify_task and task_norm > 0:
        if not any(item['affinity_requires_grad'] for item in samples if item['optional_changed']):
            raise ValueError('Native task gradient excludes student association affinity')
        evidence.update(task_gradient_verified=True, first_task_gradient_norm=task_norm,
                        first_comm_gradient_norm=comm_norm,
                        task_gradient_tokens=[item['token'] for item in samples])
    updated, gradient_norm = False, 0.0
    if optimize and any(item['optional'] for item in samples) and total.requires_grad:
        total.backward()
        gradients = [p.grad for p in scheduler.parameters() if p.grad is not None]
        if any(not torch.isfinite(value).all() for value in gradients):
            raise FloatingPointError('Absent or nonfinite scheduler gradients')
        if gradients:
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(scheduler.parameters(), settings['optimizer']['grad_clip_norm']))
            if not math.isfinite(gradient_norm):
                raise FloatingPointError('Nonfinite scheduler gradient norm')
            optimizer.step()
            updated = True
        elif not diagnose:
            raise FloatingPointError('Absent scheduler gradients in a formal optional-query batch')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Frozen native model received parameter gradients')
    training.save_batch_state(model, entries, streams)
    for record, sample in zip(records, samples):
        record.update(task_loss=sample['task_loss'],
                      total_loss=sample['task_loss'] + settings['lambda_comm'] * record['comm_loss'])
    value = {'task_loss': float(task.detach()), 'comm_loss': float(comm.detach()),
             'weighted_comm_loss': settings['lambda_comm'] * float(comm.detach()),
             'total_loss': float(total.detach()), 'optimizer_step': updated,
             'scheduler_gradient_norm': gradient_norm, 'task_gradient_norm': task_norm,
             'comm_gradient_norm': comm_norm, 'diagnostic': diagnostic,
             'native_batch_size': context['native_bev_batch_size'], 'records': records}
    context.clear()
    del batch, samples, losses, task, comm, total
    return value


def statistics(records):
    np = native.np
    count = len(records)
    if not count:
        raise ValueError('Statistics require real frames')
    queries = sum(row['queries'] for row in records)
    optional = sum(row['optional_queries'] for row in records)
    ages = [age for row in records for age in row['cache_ages']]
    probabilities = [p for row in records for p in row['optional_probabilities']]
    age_values = {'mean': float(np.mean(ages)), 'median': float(np.median(ages)),
                  'p95': float(np.percentile(ages, 95)), 'max': float(np.max(ages))} if ages else dict.fromkeys(('mean', 'median', 'p95', 'max'))
    mean = lambda name: sum(row[name] for row in records) / count
    value = {'samples': count, 'scenes': len({row['scene'] for row in records}),
             **{name: mean(name) for name in ('task_loss', 'comm_loss', 'weighted_comm_loss', 'total_loss')},
             'mean_actual_bytes_per_frame': mean('actual_bytes'),
             'p95_actual_bytes_per_frame': float(np.percentile([row['actual_bytes'] for row in records], 95)),
             'feature_payload_mean_B_per_frame': mean('feature_payload_bytes'),
             'feature_payload_BPS': 10 * mean('feature_payload_bytes'),
             'feature_payload_latent_BPS': 10 * mean('latent_bytes'),
             'geometry_BPS': 10 * mean('geometry_bytes'),
             'serialized_message_mean_B_per_frame': mean('actual_bytes'),
             'serialized_message_P95_B_per_frame': float(np.percentile([row['actual_bytes'] for row in records], 95)),
             'serialized_message_BPS': 10 * mean('actual_bytes'),
             'estimated_optional_mean_B_per_frame': mean('estimated_optional_bytes'),
             'actual_optional_mean_B_per_frame': mean('actual_optional_bytes'),
             'soft_hard_gap_mean_B_per_frame': mean('soft_hard_gap_bytes'),
             'soft_hard_gap_absolute_mean_B_per_frame': sum(abs(row['soft_hard_gap_bytes']) for row in records) / count,
             'queries': queries, 'optional_queries': optional,
             'forced_init': sum(row['forced_init'] for row in records),
             'optional_updates': sum(row['optional_updates'] for row in records),
             'optional_update_ratio': sum(row['optional_updates'] for row in records) / optional if optional else None,
             'forced_init_ratio': sum(row['forced_init'] for row in records) / queries if queries else None,
             'total_update_ratio': sum(row['updates'] for row in records) / queries if queries else None,
             'near_threshold_fraction': sum(row['near_threshold_count'] for row in records) / optional if optional else None,
             'optional_probability_mean': float(np.mean(probabilities)) if probabilities else None,
             'optional_probability_percentiles': [float(v) for v in np.percentile(probabilities, [0, 5, 25, 50, 75, 95, 100])] if probabilities else None,
             'optional_probability_histogram': np.histogram(probabilities, bins=[0, .1, .25, .45, .55, .75, .9, 1])[0].tolist(),
             'cache_age_scope': 'optional queries before decision; initialization excluded',
             'BPS_scope': 'nominal 10Hz model-field payload; excludes protocol/network overhead',
             **{'cache_age_' + name: number for name, number in age_values.items()}}
    value.update(sample_count=count, scene_count=value['scenes'], mean_bytes=value['mean_actual_bytes_per_frame'], p95_bytes=value['p95_actual_bytes_per_frame'])
    return value


def save_history(path, history):
    runtime.write(path.with_suffix('.json'), history)
    columns = list(dict.fromkeys(key for row in history for key in row))
    temporary = path.with_suffix('.csv.writing')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in history)
    temporary.replace(path.with_suffix('.csv'))


def save_checkpoint(path, scheduler, optimizer, epoch, identity, history, selection):
    value = {'scheduler': scheduler.state_dict(), 'optimizer': optimizer.state_dict(),
             'epoch': epoch, 'variant': 'mlp', 'phase': 'train', 'identity': identity,
             'rng': inherited.rng_state(), 'lr_progress': float(epoch),
             'training_result': {'status': 'EPOCH_COMPLETE', 'completed_epoch': epoch, 'epochs': history},
             'selection': selection}
    temporary = path.with_suffix('.pth.writing')
    native.torch.save(value, str(temporary))
    temporary.replace(path)
    return value


def same_state(left, right):
    """Compare optimizer tensors after device-aware load_state_dict."""
    if native.torch.is_tensor(left):
        return native.torch.is_tensor(right) and native.torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(same_state(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, (list, tuple)) and len(left) == len(right) and all(same_state(a, b) for a, b in zip(left, right))
    return left == right


def train_epoch(root, model_name, epoch):
    root = Path(root).resolve()
    runtime.initialize(root)
    torch = native.torch
    from mmcv.parallel import MMDataParallel
    settings = runtime.load_settings(root, model_name)
    if model_name not in MODELS or not 1 <= epoch <= settings['max_epochs']:
        raise ValueError('Unknown model or epoch outside the V4 protocol')
    identity = runtime.identity(root, model_name)
    directory = runtime.experiment(root, model_name)
    metrics = directory / 'metrics'
    metrics.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = directory / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = checkpoint_dir / 'latest.pth'
    previous = None
    if latest.is_file() or (checkpoint_dir / 'last.pth').is_file():
        previous = torch.load(str(latest if latest.is_file() else checkpoint_dir / 'last.pth'), map_location='cpu')
        if previous['identity'] != identity or previous['epoch'] != epoch - 1:
            raise ValueError('V4 epoch must follow the exact compatible completed checkpoint')
    elif epoch != 1:
        raise ValueError('Previous complete epoch checkpoint is required')
    preflight = runtime.read(directory / 'audit/preflight.json')
    if preflight.get('status') != 'PASS' or preflight.get('identity') != identity:
        raise ValueError('Matching complete V4 preflight is required before formal training')
    proof_path = metrics / 'correctness.json'
    if not proof_path.is_file() or runtime.read(proof_path).get('status') != 'PASS' or runtime.read(proof_path).get('identity') != identity:
        raise ValueError('Matching V4 correctness proof is required before formal training')
    inherited.seed(settings['training_seed'])
    cfg, dataset, scenes, sender_rows = load_training(root)
    model = native.build_model(root, cfg, 'coop').cuda()
    model.CLASSES = dataset.CLASSES
    training.model_in_training_mode(model)
    scheduler = Scheduler().cuda().train()
    optimizer = torch.optim.AdamW(scheduler.parameters(), lr=settings['optimizer']['peak_lr'],
                                  weight_decay=settings['optimizer']['weight_decay'])
    history, selection = [], {}
    if previous is not None:
        scheduler.load_state_dict(previous['scheduler'], strict=True)
        optimizer.load_state_dict(previous['optimizer'])
        inherited.restore_rng(previous['rng'])
        history = previous['training_result']['epochs']
        selection = previous.get('selection', {})
        if len(history) != epoch - 1 or previous['lr_progress'] != float(epoch - 1):
            raise ValueError('Checkpoint epoch/history/LR progress disagree')
    proof = parameter_proof(model, scheduler, optimizer)
    frozen = training.frozen_snapshot(model)
    context, streams, evidence = {}, {}, {'task_gradient_verified': False}
    remove_original = training.install_hooks(model, context)
    remove_v3, instrumentation = native_hooks.install(model, context)
    parallel = MMDataParallel(model, device_ids=[0])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    records, gradient_norms, task_norms, comm_norms = [], [], [], []
    counters, processed, diagnostic_rows = Counter(), [], []
    try:
        model.reset_memory()
        # A separate sampler RNG makes order identical across models and resumes.
        order = sorted(scenes)
        random.Random(settings['training_seed'] + epoch - 1).shuffle(order)
        batches = list(training.scene_batches(scenes, order, 4))
        for step, entries in enumerate(batches, 1):
            progress = (epoch - 1) + (step - 1) / max(1, len(batches) - 1)
            lr = training.learning_rate(progress, settings)
            for group in optimizer.param_groups:
                group['lr'] = lr
            value = run_batch(root, model_name, epoch, step, entries, dataset, streams, sender_rows,
                              model, scheduler, parallel, optimizer, settings, context, evidence)
            with (metrics / ('train_epoch%03d_batches.jsonl' % epoch)).open('a', encoding='utf-8') as batch_log:
                batch_log.write(json.dumps({key: item for key, item in value.items() if key != 'records'}, allow_nan=False) + '\n')
            records.extend(value['records'])
            processed.extend(entry['index'] for entry in entries)
            counters['optimizer_steps'] += int(value['optimizer_step'])
            counters['zero_gradient_batches'] += int(value['scheduler_gradient_norm'] == 0)
            counters['gradient_not_computed_batches'] += int(not value['optimizer_step'])
            counters['batches'] += 1
            gradient_norms.append(value['scheduler_gradient_norm'])
            if value['diagnostic']:
                task_norms.append(value['task_gradient_norm'])
                comm_norms.append(value['comm_gradient_norm'])
                diagnostic_rows.append({name: value[name] for name in ('task_gradient_norm', 'comm_gradient_norm', 'scheduler_gradient_norm')})
                diagnostic_rows[-1]['step'] = step
            if step == 1 or step % 25 == 0 or step == len(batches):
                emit({'phase': 'train', 'model': model_name, 'epoch': epoch, 'step': step,
                      'steps': len(batches), 'samples': len(processed), 'total_samples': 7362,
                      'task_loss': value['task_loss'], 'comm_loss': value['comm_loss'],
                      'total_loss': value['total_loss'], 'lr': lr})
        if len(processed) != 7362 or len(set(processed)) != 7362 or streams:
            raise ValueError('Full official train samples were not processed exactly once')
        if not evidence['task_gradient_verified']:
            raise ValueError('Native Scheduler task gradient was not proven nonzero')
        training.assert_frozen(model, frozen)
        row = dict(statistics(records), epoch=epoch, optimizer_steps=counters['optimizer_steps'],
                   batches=counters['batches'], learning_rate=lr, lr_progress=float(epoch),
                   scheduler_gradient_norm=sum(gradient_norms) / len(gradient_norms),
                   task_gradient_norm=sum(task_norms) / len(task_norms) if task_norms else None,
                   comm_gradient_norm=sum(comm_norms) / len(comm_norms) if comm_norms else None,
                   zero_gradient_batch_count=counters['zero_gradient_batches'],
                   gradient_not_computed_batch_count=counters['gradient_not_computed_batches'],
                   non_finite_batch_count=0, task_gradient_verified=evidence['task_gradient_verified'],
                   first_task_gradient_norm=evidence.get('first_task_gradient_norm'),
                   first_comm_gradient_norm=evidence.get('first_comm_gradient_norm'),
                   gradient_diagnostic_steps=list(DIAGNOSTIC_STEPS), gradient_diagnostics=diagnostic_rows,
                   seconds=time.perf_counter() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   parameter_proof=proof, native_instrumentation=instrumentation,
                   experiment_id=runtime.MODEL)
        runtime.write(directory / 'communication' / ('train_epoch%03d.json' % epoch), {'summary': row, 'frames': records})
        history.append(row)
        save_checkpoint(latest, scheduler, optimizer, epoch, identity, history, selection)
        save_history(metrics / 'train_history', history)
        runtime.write(metrics / 'training_state.json', {'status': 'EPOCH_COMPLETE', 'model': model_name,
                      'completed_epoch': epoch, 'checkpoint': str(latest),
                      'checkpoint_sha256': runtime.digest(latest), 'identity': identity})
        emit({'phase': 'train_epoch_complete', 'model': model_name, 'epoch': epoch,
              'mean_bytes': row['mean_actual_bytes_per_frame'], 'seconds': row['seconds']})
        return row
    except Exception as exc:
        nonfinite = isinstance(exc, FloatingPointError) or any(word in str(exc).lower() for word in ('nonfinite', 'non-finite', 'nan/inf'))
        runtime.write(metrics / 'training_state.json', {'status': 'FAILED', 'model': model_name,
                      'epoch': epoch, 'completed_samples': len(processed),
                      'non_finite_batch_count': int(nonfinite),
                      'error': str(exc), 'traceback': traceback.format_exc(), 'identity': identity})
        raise
    finally:
        remove_v3()
        remove_original()
        context.clear()
        streams.clear()
        del parallel, model, scheduler, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def correctness_check(root, model_name):
    root = Path(root).resolve()
    runtime.initialize(root)
    torch = native.torch
    from mmcv.parallel import MMDataParallel
    settings = runtime.load_settings(root, model_name)
    identity = runtime.identity(root, model_name)
    metrics = runtime.experiment(root) / 'metrics'
    metrics.mkdir(parents=True, exist_ok=True)
    path = metrics / 'correctness.json'
    if path.exists():
        previous = runtime.read(path)
        if previous.get('status') == 'PASS' and previous.get('identity') == identity:
            return previous
        raise ValueError('Preserve existing failed or incompatible correctness evidence')
    inherited.seed(settings['training_seed'])
    cfg, dataset, scenes, sender_rows = load_training(root)
    if cfg.model.spatial_temporal_reason.future_reasoning:
        raise ValueError('Future reasoning must be disabled')
    model = native.build_model(root, cfg, 'coop').cuda()
    model.CLASSES = dataset.CLASSES
    training.model_in_training_mode(model)
    scheduler = Scheduler().cuda().train()
    optimizer = torch.optim.AdamW(scheduler.parameters(), lr=settings['optimizer']['peak_lr'], weight_decay=settings['optimizer']['weight_decay'])
    parameter_details = parameter_proof(model, scheduler, optimizer)
    frozen = training.frozen_snapshot(model)
    context, streams, evidence = {}, {}, {}
    remove_original = training.install_hooks(model, context)
    remove_native, instrumentation = native_hooks.install(model, context)
    parallel = MMDataParallel(model, device_ids=[0])
    report = {'status': 'RUNNING', 'model': model_name, 'identity': identity,
              'weights_discarded': True, 'official_val_backpropagation': False,
              'parameter_proof': parameter_details, 'native_instrumentation': instrumentation,
              'correctness_rule': 'first four sorted train scenes; cold frame then at most seven chronological frames per lane; stop at first nonzero finite task AND comm gradient; discard all check weights',
              'checked_batches': [], 'max_frames_per_lane': 8}
    try:
        model.reset_memory()
        batches = iter(training.scene_batches(scenes, sorted(scenes), 4))
        checked = None
        for step in range(1, 9):
            entries = next(batches)
            value = run_batch(root, model_name, 1, step, entries, dataset, streams, sender_rows, model, scheduler, parallel, optimizer, settings, context, evidence, optimize=step > 1, diagnose=True)
            row = {key: item for key, item in value.items() if key != 'records'}
            row.update(step=step, sample_count=len(entries), frames=value['records'])
            report['checked_batches'].append(row)
            emit(dict(phase='correctness_batch', **{key: item for key, item in row.items() if key != 'frames'}))
            training.assert_frozen(model, frozen)
            if step == 1 and (value['optimizer_step'] or any(frame['optional_queries'] for frame in value['records'])):
                raise ValueError('Cold initialization must have no optional decision or optimizer update')
            if step > 1 and value['task_gradient_norm'] > 0 and value['comm_gradient_norm'] > 0 and value['scheduler_gradient_norm'] > 0 and evidence.get('task_gradient_verified'):
                checked = value
                break
        if checked is None:
            raise ValueError('No finite nonzero task AND communication gradients in the predetermined 2-8 frame snippet')
        for field in ('sender', 'receiver', 'rng'):
            if len({id(stream[field]) for stream in streams.values()}) != len(streams):
                raise ValueError('Scene lane state isolation failed: ' + field)
        for scene, stream in streams.items():
            if stream['receiver'].identity != stream['sender'].identity or stream['sender'].identity[0] != scene or stream['native']['scene_token'] != scene:
                raise ValueError('Native/sender/receiver scene identities leaked')
        temporary = metrics / 'correctness_resume_roundtrip.pth'
        if temporary.exists():
            raise ValueError('Unexpected previous correctness checkpoint')
        save_checkpoint(temporary, scheduler, optimizer, 1, identity, [{'epoch': 1}], {'best_epoch': None})
        loaded = torch.load(str(temporary), map_location='cpu')
        clone = Scheduler().cuda()
        clone.load_state_dict(loaded['scheduler'], strict=True)
        clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=.01)
        clone_optimizer.load_state_dict(loaded['optimizer'])
        if not same_state(scheduler.state_dict(), clone.state_dict()) or not same_state(optimizer.state_dict(), clone_optimizer.state_dict()) or loaded['identity'] != identity or loaded['lr_progress'] != 1.0:
            raise ValueError('Scheduler/optimizer/identity/LR resume roundtrip failed')
        inherited.restore_rng(loaded['rng'])
        first = (random.random(), float(native.np.random.rand()), float(torch.rand(1)), float(torch.rand(1, device='cuda')))
        inherited.restore_rng(loaded['rng'])
        second = (random.random(), float(native.np.random.rand()), float(torch.rand(1)), float(torch.rand(1, device='cuda')))
        if first != second:
            raise ValueError('Python/NumPy/CPU/CUDA RNG resume roundtrip failed')
        temporary.unlink()
        del clone, clone_optimizer
        report.update(status='PASS', native_task_gradient_nonzero=True, communication_gradient_nonzero=True,
                      task_gradient_norm=checked['task_gradient_norm'], comm_gradient_norm=checked['comm_gradient_norm'],
                      scheduler_gradient_norm=checked['scheduler_gradient_norm'],
                      native_batch_size=4, lane_cache_rng_isolation=True,
                      real_packet_INT8_UPDATE_REUSE=True, hard_STE_equals_actual_decoded=True,
                      native_parameters_and_BN_frozen=True, future_reasoning_disabled=True,
                      resume_rng_optimizer_lr_identity_roundtrip=True, **evidence)
        runtime.write(path, report)
        emit(dict(phase='correctness', status='PASS', task_gradient_norm=report['task_gradient_norm'], comm_gradient_norm=report['comm_gradient_norm']))
        return report
    except Exception as exc:
        report.update(status='FAILED', error=str(exc), traceback=traceback.format_exc())
        runtime.write(path, report)
        raise
    finally:
        remove_native()
        remove_original()
        context.clear()
        streams.clear()
        del parallel, model, scheduler, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def self_check():
    checks = native_hooks.self_check()
    scenes = {'a': [0, 1, 2], 'b': [3], 'c': [4, 5], 'd': [6], 'e': [7, 8]}
    batches = list(training.scene_batches(scenes, sorted(scenes), 4))
    assert sorted(entry['index'] for batch in batches for entry in batch) == list(range(9))
    assert all(len({entry['scene'] for entry in batch}) == len(batch) for batch in batches)
    assert all([entry['index'] for batch in batches for entry in batch if entry['scene'] == scene] == indices for scene, indices in scenes.items())
    settings = {'optimizer': {'peak_lr': 1e-4}, 'lr_schedule': {'warmup_epochs': 1,
                'warmup_start_lr': 1e-5, 'min_lr': 1e-6, 'cosine_total_horizon_epochs': 36}}
    assert training.learning_rate(0, settings) == 1e-5
    assert abs(training.learning_rate(1, settings) - 1e-4) < 1e-12
    assert abs(training.learning_rate(36, settings) - 1e-6) < 1e-12
    assert sum(p.numel() for p in Scheduler().parameters()) == 71361
    emit(dict(checks, scene_batch_temporal_isolation=True, horizon36_LR_endpoints=True,
              reused_MLP_parameter_count=71361))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE.parents[1])
    parser.add_argument('--model', choices=MODELS, default=runtime.MODEL)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--epoch', type=int)
    group.add_argument('--correctness-check', action='store_true')
    group.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.correctness_check:
        correctness_check(args.root, args.model)
    else:
        train_epoch(args.root, args.model, args.epoch)


if __name__ == '__main__':
    main()
