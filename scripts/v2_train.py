"""Train each V2 gate for 12 fixed epochs on all 46 official train scenes."""
import argparse
import csv
import gc
import json
import math
import random
import shutil
import time
import traceback
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import main_runtime as native
import v2_runtime as runtime


STATE_FIELDS = ('scene_token', 'prev_timestamp', 'prev_bev', 'track_instances',
                'l2g_r_mat', 'l2g_t', 'prev_pos', 'prev_angle')
STALENESS_BUCKETS = ('0-1', '2-3', '4-7', '8-15', '16-31', '32+',
                    'negative_elapsed', 'new_initialization')


def staleness_counts(elapsed, forced, probabilities, actions):
    """Small CPU counters; initialization never enters optional gate statistics."""
    counts = {name: Counter() for name in STALENESS_BUCKETS}
    for age, initialize, probability, update in zip(elapsed, forced, probabilities, actions):
        if initialize:
            name = 'new_initialization'
        elif age < 0:
            name = 'negative_elapsed'
        else:
            name = next((label for bound, label in zip((2, 4, 8, 16, 32), STALENESS_BUCKETS)
                         if age < bound), '32+')
        counts[name]['count'] += 1
        counts[name]['update_count'] += int(update)
        if not initialize:
            counts[name]['optional_count'] += 1
            counts[name]['probability_sum'] += probability
    return counts


def staleness_summary(counts):
    return {name: {'count': values['count'], 'update_count': values['update_count'],
                   'mean_probability': values['probability_sum'] / values['optional_count']
                   if values['optional_count'] else None,
                   'optional_update_ratio': values['update_count'] / values['optional_count']
                   if values['optional_count'] else None}
            for name, values in counts.items()}


def scene_batches(scenes, scene_order, batch_size):
    """Keep each scene in one lane, replacing a completed scene with the next."""
    queue = deque(scene_order)
    lanes = [None] * batch_size
    while queue or any(lanes):
        batch = []
        for slot in range(batch_size):
            if lanes[slot] is None and queue:
                lanes[slot] = [queue.popleft(), 0]
            if lanes[slot] is None:
                continue
            scene, position = lanes[slot]
            indices = scenes[scene]
            batch.append({'scene': scene, 'index': indices[position], 'slot': slot,
                          'first': position == 0, 'last': position == len(indices) - 1})
            lanes[slot][1] += 1
            if batch[-1]['last']:
                lanes[slot] = None
        if batch:
            yield batch


def learning_rate(progress, settings):
    peak = settings['optimizer']['peak_lr']
    schedule = settings['lr_schedule']
    warmup = schedule['warmup_epochs']
    if progress <= warmup:
        return schedule['warmup_start_lr'] + (peak - schedule['warmup_start_lr']) * progress / warmup
    horizon = schedule['cosine_total_horizon_epochs']
    return schedule['min_lr'] + 0.5 * (peak - schedule['min_lr']) * (
        1 + math.cos(math.pi * (progress - warmup) / (horizon - warmup)))


def emit(root, value):
    line = json.dumps(value, ensure_ascii=False, allow_nan=False)
    print(line, flush=True)
    path = Path(root) / 'logs/v2/train.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')


def resource_row(root, value):
    path = Path(root) / 'logs/v2/resource.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size
    names = ('timestamp_utc', 'phase', 'variant', 'epoch', 'step', 'samples',
             'native_batch_size', 'effective_optimizer_batch_size', 'data_ms',
             'forward_ms', 'backward_ms', 'optimizer_ms', 'elapsed_ms',
             'samples_per_second', 'allocated_bytes', 'reserved_bytes',
             'peak_allocated_bytes', 'peak_reserved_bytes', 'host_rss_bytes',
             'host_mem_available_bytes', 'host_commit_limit_bytes', 'host_committed_bytes')
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow(dict(value, timestamp_utc=datetime.now(timezone.utc).isoformat()))


def host_memory():
    values = {}
    try:
        meminfo = {line.split(':')[0]: int(line.split()[1]) * 1024
                   for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
        values.update(host_mem_available_bytes=meminfo.get('MemAvailable'),
                      host_commit_limit_bytes=meminfo.get('CommitLimit'),
                      host_committed_bytes=meminfo.get('Committed_AS'))
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                values['host_rss_bytes'] = int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return values


def install_hooks(model, context):
    """V2-only instance hooks: no teacher pass, per-scene student and task mean."""
    torch = native.torch
    from v2_message import FIELDS
    originals = {name: getattr(model, name) for name in
                 ('forward_track_stream_train', 'get_bevs', 'loss_single_batch')}
    original_match = model.STReasoner._learn_matching
    original_memory = model.criterion.loss_mem_bank

    def alignment_input(module, incoming):
        loaded, transform = incoming
        index = context['alignment_indices'][context['alignment_cursor']]
        context['alignment_cursor'] += 1
        sample = context['samples'][index]
        for name in ('ref_pts', 'pred_boxes'):
            if not torch.equal(loaded[name], sample['current'][name]):
                raise ValueError('Native geometry differs from the current sender at batch sample %d' % index)
        sample['received_queries'] = len(loaded['query_feats'])
        return ({name: sample['student'][name] for name in FIELDS}, transform)

    def learn_matching(*args, **kwargs):
        result = original_match(*args, **kwargs)
        index = context['matching_cursor']
        context['matching_cursor'] += 1
        context['samples'][index]['affinity_requires_grad'] = result[2].requires_grad
        return result

    def get_bevs(img, *args, **kwargs):
        context['bev_forward_count'] += 1
        context['native_bev_batch_size'] = int(img.shape[0])
        if context['native_bev_batch_size'] != len(context['samples']):
            raise ValueError('High-cost BEV forward did not receive the full active batch')
        return originals['get_bevs'](img, *args, **kwargs)

    def equal_stream_factors(losses):
        # Original loss values remain normalized inside their native loss functions.
        # The outer batch reducer now averages streams rather than weighting GT counts.
        for name in list(losses):
            if 'avg_factor' in name:
                value = losses[name]
                losses[name] = torch.ones_like(value) if torch.is_tensor(value) else 1.0
        return losses

    def loss_single_batch(*args, **kwargs):
        result = originals['loss_single_batch'](*args, **kwargs)
        equal_stream_factors(model.criterion.losses_dict)
        return result

    def loss_mem_bank(*args, **kwargs):
        return equal_stream_factors(original_memory(*args, **kwargs))

    def raw_stream(*args, **kwargs):
        context['native_forward_count'] += 1
        losses = originals['forward_track_stream_train'](*args, **kwargs)
        if any(not torch.isfinite(value).all() for value in losses.values()):
            raise ValueError('Native task loss contains NaN/Inf before outer nan_to_num')
        return losses

    handle = model.crossview_alignment.register_forward_pre_hook(alignment_input)
    model.STReasoner._learn_matching = learn_matching
    model.criterion.loss_mem_bank = loss_mem_bank
    model.get_bevs = get_bevs
    model.loss_single_batch = loss_single_batch
    model.forward_track_stream_train = raw_stream

    def remove():
        handle.remove()
        for name, original in originals.items():
            setattr(model, name, original)
        model.STReasoner._learn_matching = original_match
        model.criterion.loss_mem_bank = original_memory
    return remove


def model_in_training_mode(model):
    model.train()
    for module in model.modules():
        if isinstance(module, native.torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def frozen_snapshot(model):
    torch = native.torch
    return {'versions': tuple(parameter._version for parameter in model.parameters()),
            'bn': {name: value.detach().cpu().clone() for name, value in model.named_buffers()
                   if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}}


def assert_frozen(model, snapshot):
    if tuple(parameter._version for parameter in model.parameters()) != snapshot['versions']:
        raise ValueError('Frozen native model parameter was updated')
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Frozen native parameter received gradients or was unfrozen')
    for name, value in model.named_buffers():
        if name in snapshot['bn'] and not native.torch.equal(value.detach().cpu(), snapshot['bn'][name]):
            raise ValueError('Frozen native BN persistent statistics changed: ' + name)


def native_slot():
    return {name: (0 if name in ('prev_pos', 'prev_angle') else None) for name in STATE_FIELDS}


def load_training(root, batch_size):
    cfg = runtime.make_config(root, 'coop', 'train', training=True, batch_size=batch_size)
    dataset = native.dataset_for(cfg, training=True)
    manifest = native.read(Path(root) / 'configs/main_experiment_v2/train_manifest.json')
    scenes = {}
    for index, info in enumerate(dataset.data_infos):
        scenes.setdefault(str(info['scene_token']), []).append(index)
    official = set(native.read(Path(root) / 'code/CoopTrack/data/split_datas/cooperative-split-data-spd.json')['batch_split']['train'])
    if len(dataset) != 7362 or len(scenes) != 46 or set(scenes) != official or set(scenes) != set(manifest['all_train_scenes']):
        raise ValueError('V2 training does not contain all official cooperative train samples/scenes')
    for scene, indices in scenes.items():
        timestamps = [dataset.data_infos[index]['timestamp'] for index in indices]
        if timestamps != sorted(timestamps):
            raise ValueError('Native train scene is not in chronological order: ' + scene)
    sender_rows = {row['token']: row for row in native.native_manifest(root, 'train')}
    if any(str(info['token_inf']) not in sender_rows for info in dataset.data_infos):
        raise ValueError('Native train sender mapping is incomplete')
    return cfg, dataset, scenes, sender_rows, manifest


def prepare_batch(root, dataset, entries, streams, sender_rows, scheduler, settings, epoch_index):
    torch = native.torch
    from mmcv.parallel import collate
    from main_train import training_metas
    from v2_message import (FIELDS, FEATURES, TemporalCache, ReceiverCache,
                            int8_roundtrip, message_sizes, pack, select_updates)
    from v2_scheduler import straight_through_latents, rate_loss
    prepared, expected, sample_context, stats = [], [], [], []
    for entry in entries:
        index, scene = entry['index'], entry['scene']
        if entry['first']:
            if scene in streams:
                raise ValueError('New scene entered a live lane twice')
            streams[scene] = {'sender': TemporalCache(), 'receiver': ReceiverCache(), 'native': native_slot()}
        stream = streams[scene]
        info = dataset.data_infos[index]
        data = dataset.prepare_train_data(index)
        if data is None:
            raise ValueError('Native pipeline rejected an official training frame: ' + str(info['token']))
        metas = training_metas(data['img_metas'])
        if len(metas) != 1 or str(metas[0]['sample_idx']) != str(info['token']) or str(metas[0]['sample_idx_inf']) != str(info['token_inf']):
            raise ValueError('Native pipeline substituted or reordered an official training frame')
        metadata = sender_rows[str(info['token_inf'])]
        if str(metadata['scene']) != scene:
            raise ValueError('Sender and vehicle scene identities differ')
        raw = native.fields(Path(root) / 'query_cache/train' / (metadata['token'] + '.pkl'))
        raw = {name: raw[name] for name in FIELDS + ('obj_idxes', 'scores')}
        expected.append({name: raw[name] for name in FIELDS})
        current = int8_roundtrip(raw)
        temporal = stream['sender'].prepare(current, metadata)
        budget_index = (settings['initial_offset'][scene] + epoch_index) % 4
        budget = settings['budgets_bytes'][budget_index]
        probability = scheduler(temporal['tokens'].cuda(), budget / settings['full_int8_mean_bytes']).sigmoid()
        hard = select_updates(probability.detach().cpu(), temporal['force'])
        mandatory, per_update, full = message_sizes(raw, metadata)
        payload = pack(raw, metadata, hard)
        decoded, _ = stream['receiver'].unpack(payload, metadata)
        for name in FEATURES:
            if not torch.equal(decoded[name][hard], current[name][hard]) or not torch.equal(decoded[name][~hard], temporal['last'][name][~hard]):
                raise ValueError('Decoded UPDATE/REUSE differs from current INT8 or last-received')
        current_gpu = {name: value.cuda() for name, value in current.items() if torch.is_tensor(value)}
        last_gpu = {name: value.cuda() for name, value in temporal['last'].items()}
        decoded_gpu = {name: value.cuda() for name, value in decoded.items()}
        force_gpu = temporal['force'].cuda()
        student = straight_through_latents(decoded_gpu, current_gpu, last_gpu, probability, force_gpu)
        if any(not torch.equal(student[name].detach(), decoded_gpu[name]) for name in FIELDS):
            raise ValueError('V2 straight-through hard forward differs from the actual decoded packet')
        rate, soft_bytes, feasible = rate_loss(probability, force_gpu, mandatory, per_update,
                                               budget, settings['full_int8_mean_bytes'])
        # The surrogate reads the old mirror before the UPDATE commit.
        stream['sender'].commit(raw, hard, metadata, decoded)
        changed = bool((~force_gpu).any()) and any(
            bool((current_gpu[name][~force_gpu] != last_gpu[name][~force_gpu]).any()) for name in FEATURES)
        sample_context.append({'current': current_gpu, 'student': student, 'rate': rate,
                               'optional_changed': changed, 'optional': bool((~force_gpu).any()),
                               'received_queries': 0, 'affinity_requires_grad': False,
                               'token': str(info['token']), 'scene': scene})
        probability_cpu = probability.detach().cpu()
        optional_p = probability_cpu[~temporal['force']]
        histogram = torch.histc(optional_p, bins=10, min=0, max=1).to(torch.int64).tolist() if len(optional_p) else [0] * 10
        stats.append({'queries': len(hard), 'updates': int(hard.sum()), 'init': int(temporal['force'].sum()),
                      'optional_queries': int((~temporal['force']).sum()),
                      'optional_updates': int((hard & ~temporal['force']).sum()),
                      'bytes': len(payload), 'soft_bytes': float(soft_bytes.detach()),
                      'overflow': max(0, len(payload) - budget),
                      'init_overflow': max(0, mandatory + per_update * int(temporal['force'].sum()) - budget),
                      'budget_name': 'B' + str(budget_index + 1), 'histogram': histogram,
                      'staleness': staleness_counts(temporal['elapsed'].tolist(), temporal['force'].tolist(),
                                                    probability_cpu.tolist(), hard.tolist())})
        prepared.append(data)
    batch = collate(prepared, samples_per_gpu=len(prepared))
    runtime.assert_loaded_queries(batch, expected)
    return batch, sample_context, stats


def install_batch_state(model, entries, streams):
    model.batch_size = len(entries)
    model.train_prev_infos = {name: [streams[entry['scene']]['native'][name] for entry in entries]
                              for name in STATE_FIELDS}


def save_batch_state(model, entries, streams):
    for index, entry in enumerate(entries):
        scene = entry['scene']
        if model.train_prev_infos['scene_token'][index] != scene:
            raise ValueError('Native temporal state leaked between scene lanes')
        streams[scene]['native'] = {name: model.train_prev_infos[name][index] for name in STATE_FIELDS}
        if entry['last']:
            del streams[scene]


def gradient_evidence(loss, scheduler):
    torch = native.torch
    if not loss.requires_grad:
        return 0.0, False
    gradients = torch.autograd.grad(loss, tuple(scheduler.parameters()), retain_graph=True, allow_unused=True)
    if any(not torch.isfinite(value).all() for value in gradients if value is not None):
        raise ValueError('Nonfinite standalone native/rate gradient')
    norm = sum(float(value.detach().square().sum()) for value in gradients if value is not None) ** 0.5
    if not math.isfinite(norm):
        raise ValueError('Nonfinite standalone native/rate gradient norm')
    return norm, any(value is not None for value in gradients)


def run_batch(root, variant, phase, epoch, step, entries, dataset, streams, sender_rows,
              model, scheduler, parallel, optimizer, settings, context, evidence, optimize=True):
    torch = native.torch
    torch.cuda.synchronize()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    batch, samples, stats = prepare_batch(root, dataset, entries, streams, sender_rows, scheduler, settings, epoch - 1)
    install_batch_state(model, entries, streams)
    context.clear()
    context.update(samples=samples, alignment_indices=[index for index, item in enumerate(samples)
                                                     if len(item['current']['query_feats'])],
                   alignment_cursor=0, matching_cursor=0, native_forward_count=0, bev_forward_count=0,
                   native_bev_batch_size=0)
    torch.cuda.synchronize()
    data_done = time.perf_counter()
    losses = parallel(return_loss=True, **batch)
    task, logs = model._parse_losses(losses)
    if context['native_forward_count'] != 1 or context['bev_forward_count'] != 1:
        raise ValueError('V2 batch was implemented as multiple native model forwards')
    if context['alignment_cursor'] != len(context['alignment_indices']) or context['matching_cursor'] != len(samples):
        raise ValueError('Native batch hook did not visit each scene exactly once')
    rate = torch.stack([item['rate'] for item in samples]).mean()
    total = task + settings['lambda_rate'] * rate
    if not torch.isfinite(total):
        raise ValueError('Nonfinite V2 training loss')
    if optimize and not evidence.get('task_gradient_verified') and any(item['optional_changed'] for item in samples):
        norm, _ = gradient_evidence(task, scheduler)
        evidence['task_gradient_checked_batches'] = evidence.get('task_gradient_checked_batches', 0) + 1
        if norm > 0:
            if not any(item['affinity_requires_grad'] for item in samples if item['optional_changed']):
                raise ValueError('Native task graph did not include student association affinity')
            evidence.update(task_gradient_verified=True, first_task_gradient_norm=norm,
                            task_gradient_tokens=[item['token'] for item in samples],
                            task_gradient_native_batch_size=len(samples))
            emit(root, {'phase': 'task_gradient_verified', 'variant': variant, 'norm': norm,
                        'tokens': evidence['task_gradient_tokens'], 'native_batch_size': len(samples)})
    if optimize and not evidence.get('rate_gradient_nonzero_seen') and rate.requires_grad:
        norm, connected = gradient_evidence(rate, scheduler)
        if connected:
            evidence.update(rate_gradient_verified=True, first_rate_gradient_norm=norm,
                            rate_gradient_nonzero_seen=norm > 0,
                            rate_loss_active_at_check=float(rate.detach()) > 0)
    torch.cuda.synchronize()
    forward_done = time.perf_counter()
    gradient_norm, updated = None, False
    if optimize and any(item['optional'] for item in samples) and total.requires_grad:
        total.backward()
        gradients = [parameter.grad for parameter in scheduler.parameters() if parameter.grad is not None]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise ValueError('Absent or nonfinite V2 scheduler gradients')
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(scheduler.parameters(), settings['optimizer']['grad_clip_norm']))
        if not math.isfinite(gradient_norm):
            raise ValueError('Nonfinite clipped scheduler gradient norm')
        updated = True
    torch.cuda.synchronize()
    backward_done = time.perf_counter()
    if updated:
        optimizer.step()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Frozen native model received parameter gradients')
    save_batch_state(model, entries, streams)
    torch.cuda.synchronize()
    finished = time.perf_counter()
    measure = dict(phase=phase, variant=variant, epoch=epoch, step=step, samples=len(entries),
                   native_batch_size=context['native_bev_batch_size'],
                   effective_optimizer_batch_size=len(entries) if updated else 0,
                   data_ms=(data_done - started) * 1000, forward_ms=(forward_done - data_done) * 1000,
                   backward_ms=(backward_done - forward_done) * 1000,
                   optimizer_ms=(finished - backward_done) * 1000,
                   elapsed_ms=(finished - started) * 1000,
                   samples_per_second=len(entries) / (finished - started),
                   allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(), **host_memory())
    resource_row(root, measure)
    values = dict(task=float(task.detach()), rate=float(rate.detach()), total=float(total.detach()),
                  components={name: float(value) for name, value in logs.items() if name != 'loss'},
                  updated=updated, gradient_norm=gradient_norm, measure=measure, samples=stats)
    context.clear()
    del batch, samples, losses, task, rate, total
    return values


def validate_settings(settings):
    if settings['max_epochs'] != 12 or settings.get('epochs', 12) != 12:
        raise ValueError('Formal V2 requires exactly 12 training epochs')
    if settings['lr_schedule']['cosine_total_horizon_epochs'] != 12:
        raise ValueError('Formal V2 cosine horizon must be 12 epochs')
    for name in ('requested_native_batch_size', 'actual_native_batch_size', 'effective_optimizer_batch_size'):
        if settings.get(name, 4) != 4:
            raise ValueError('Formal V2 requires native batch 4 without fallback: ' + name)
    if settings['budgets_bytes'] != [3474, 6382, 9290, 12198]:
        raise ValueError('Formal V2 budgets differ from the registered protocol')
    if settings['lambda_rate'] != 1.0 or settings.get('lambda_extra_association_preserve', 0.0) != 0.0:
        raise ValueError('Formal V2 loss weights differ from the registered protocol')


def correctness_check(root, variant, resume=True):
    """One cold forward, then one real optimizer step; discard these check weights."""
    root = Path(root).resolve()
    if variant not in ('mlp', 'transformer'):
        raise ValueError('Unknown formal V2 architecture')
    runtime.initialize(root)
    torch = native.torch
    from mmcv.parallel import MMDataParallel
    from v2_scheduler import Scheduler
    settings = native.read(root / 'configs/main_experiment_v2/training_v2.json')
    validate_settings(settings)
    identity = runtime.training_identity(root, variant, 'train', 4)
    path = root / 'experiments/v2/metrics' / ('correctness_' + variant + '.json')
    if resume and path.exists():
        previous = native.read(path)
        if previous.get('status') != 'PASS' or previous.get('identity') != identity:
            raise ValueError('Existing correctness proof is incomplete or belongs to another protocol')
        return previous
    runtime.seed(settings['training_seed'])
    cfg, dataset, scenes, sender_rows, manifest = load_training(root, 4)
    model = scheduler = optimizer = parallel = remove = None
    context, streams, evidence = {}, {}, {}
    report = {'status': 'RUNNING', 'variant': variant, 'identity': identity,
              'phase': 'correctness_check', 'native_batch_size': 4,
              'source': 'complete official train pipeline; one cold forward and one optimizer batch',
              'weights_discarded': True, 'optimizer_steps': 0}
    try:
        model = native.build_model(root, cfg, 'coop').cuda()
        model.CLASSES = dataset.CLASSES
        model_in_training_mode(model)
        scheduler = Scheduler(variant).cuda().train()
        optimizer = torch.optim.AdamW(scheduler.parameters(), lr=settings['optimizer']['peak_lr'],
                                      weight_decay=settings['optimizer']['weight_decay'])
        frozen = frozen_snapshot(model)
        initial_versions = tuple(parameter._version for parameter in scheduler.parameters())
        remove = install_hooks(model, context)
        parallel = MMDataParallel(model, device_ids=[0])
        torch.cuda.reset_peak_memory_stats()
        # Use registered B1 scenes for the check; scene budgets remain unchanged.
        order = sorted(scenes, key=lambda scene: (settings['initial_offset'][scene], scene))
        batches = iter(scene_batches(scenes, order, 4))
        warm_entries = next(batches)
        warm = run_batch(root, variant, 'correctness_warm', 1, 1, warm_entries, dataset,
                         streams, sender_rows, model, scheduler, parallel, optimizer,
                         settings, context, evidence, optimize=False)
        if warm['updated'] or tuple(parameter._version for parameter in scheduler.parameters()) != initial_versions:
            raise ValueError('Cold correctness forward unexpectedly optimized the scheduler')
        entries = next(batches)
        if len(entries) != 4:
            raise ValueError('Correctness optimizer batch must contain four real scene streams')
        checked = run_batch(root, variant, 'correctness_check', 1, 2, entries, dataset,
                            streams, sender_rows, model, scheduler, parallel, optimizer,
                            settings, context, evidence)
        norm = checked['gradient_norm']
        if not checked['updated'] or norm is None or not math.isfinite(norm) or norm <= 0:
            raise RuntimeError('CORRECTNESS_GRADIENT_BLOCKED: one real optimizer batch has no finite nonzero scheduler gradient')
        if not evidence.get('task_gradient_verified'):
            raise RuntimeError('TASK_GRADIENT_BLOCKED: one real optimizer batch did not prove a finite nonzero native task gradient')
        if tuple(parameter._version for parameter in scheduler.parameters()) == initial_versions:
            raise ValueError('Correctness optimizer step did not update scheduler parameters')
        assert_frozen(model, frozen)
        report.update(status='PASS', optimizer_steps=1, warm_forward_batches=1,
                      effective_optimizer_batch_size=4, gradient_norm=norm,
                      task_loss=checked['task'], rate_loss=checked['rate'], total_loss=checked['total'],
                      scene_ids=[entry['scene'] for entry in entries],
                      sample_tokens=[str(dataset.data_infos[entry['index']]['token']) for entry in entries],
                      one_native_forward_per_batch=True, one_batched_bev_forward_per_batch=True,
                      scene_state_isolated=True, native_parameters_and_bn_statistics_frozen=True,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), **evidence)
    except Exception as exc:
        report.update(status='V2_FORMAL_EXPERIMENT_BLOCKED', exception_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc(), **evidence)
    finally:
        if remove is not None:
            remove()
        del remove
        context.clear()
        streams.clear()
        del parallel, model, scheduler, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    emit(root, report)
    native.write(path, report)
    if report['status'] != 'PASS':
        raise RuntimeError(report['error'])
    return report


def save_checkpoint(path, scheduler, optimizer, epoch, variant, identity, result, progress):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {'scheduler': scheduler.state_dict(), 'optimizer': optimizer.state_dict(),
                  'epoch': epoch, 'variant': variant, 'phase': 'train', 'identity': identity,
                  'training_result': result, 'lr_progress': progress,
                  'lr_scheduler': {'type': 'normalized_epoch_formula', 'horizon': 12},
                  'rng': runtime.rng_state()}
    temp = path.with_suffix(path.suffix + '.writing')
    native.torch.save(checkpoint, str(temp))
    temp.replace(path)


def copy_checkpoint(source, destination):
    destination = Path(destination)
    temp = destination.with_suffix(destination.suffix + '.writing')
    shutil.copyfile(source, temp)
    temp.replace(destination)


def train(root, variant, resume=True):
    """Fresh scheduler initialization; fixed epoch 12 is the sole formal endpoint."""
    root = Path(root).resolve()
    if variant not in ('mlp', 'transformer'):
        raise ValueError('Formal V2 expects mlp or transformer')
    runtime.initialize(root)
    torch = native.torch
    from mmcv.parallel import MMDataParallel
    from v2_scheduler import Scheduler
    settings = native.read(root / 'configs/main_experiment_v2/training_v2.json')
    validate_settings(settings)
    identity = runtime.training_identity(root, variant, 'train', 4)
    directory = root / 'experiments/v2/checkpoints' / variant
    latest, final = directory / 'latest.pth', directory / 'final.pth'
    report_path = root / 'experiments/v2/metrics' / (variant + '_train.json')
    checkpoint = None
    if final.exists() or latest.exists():
        if not resume:
            raise ValueError('Formal V2 checkpoint already exists; resume is required')
        checkpoint = torch.load(str(final if final.exists() else latest), map_location='cpu')
        if checkpoint['identity'] != identity or checkpoint.get('phase') != 'train':
            raise ValueError('Formal V2 resume checkpoint belongs to another source/configuration/protocol')
        if not 0 <= checkpoint['epoch'] <= 12:
            raise ValueError('Formal V2 resume epoch lies outside the fixed 12 epochs')
        completed = checkpoint['training_result']
        if checkpoint['epoch'] == 12 and completed['status'] == 'PASS':
            if not final.exists():
                copy_checkpoint(latest, final)
            completed.update(final_checkpoint=str(final), final_sha256=runtime.digest(final))
            native.write(report_path, completed)
            return completed
    correctness_check(root, variant, resume=True)
    # The correctness optimizer batch is discarded. Reconstruct the formal model
    # and gate from the same initialization seed before any formal epoch begins.
    runtime.seed(settings['training_seed'])
    cfg, dataset, scenes, sender_rows, manifest = load_training(root, 4)
    if len(dataset) != 7362 or len(scenes) != 46:
        raise ValueError('Formal V2 requires all 7362 frames from all 46 train scenes')
    if set(settings['initial_offset']) != set(scenes):
        raise ValueError('Formal budget offset map does not cover exactly the official train scenes')
    directory.mkdir(parents=True, exist_ok=True)
    model = native.build_model(root, cfg, 'coop').cuda()
    model.CLASSES = dataset.CLASSES
    model_in_training_mode(model)
    scheduler = Scheduler(variant).cuda().train()
    optimizer = torch.optim.AdamW(scheduler.parameters(), lr=settings['optimizer']['peak_lr'],
                                  weight_decay=settings['optimizer']['weight_decay'])
    result = {'status': 'RUNNING', 'variant': variant, 'phase': 'train', 'identity': identity,
              'epochs': [], 'train_samples': 7362, 'train_scenes': 46,
              'native_batch_size': 4, 'effective_optimizer_batch_size': 4,
              'trainable_params': sum(parameter.numel() for parameter in scheduler.parameters()),
              'base_trainable_params': 0, 'precision': 'FP32', 'max_epochs': 12,
              'cosine_total_horizon_epochs': 12, 'training_seed': settings['training_seed'],
              'task_gradient_verified': False, 'rate_gradient_verified': False,
              'peak_vram_bytes': 0, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'checkpoint_policy': 'fixed_epoch_12; latest used only for complete-epoch crash resume',
              'correctness_proof': str(root / 'experiments/v2/metrics' / ('correctness_' + variant + '.json')),
              'scene_budget_history': {scene: [] for scene in sorted(scenes)}}
    start_epoch = 0
    if checkpoint is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        runtime.restore_rng(checkpoint['rng'])
        start_epoch = checkpoint['epoch']
        result = checkpoint['training_result']
    frozen = frozen_snapshot(model)
    context, streams = {}, {}
    remove = install_hooks(model, context)
    parallel = MMDataParallel(model, device_ids=[0])
    torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(start_epoch + 1, 13):
            model.batch_size = 4
            model.reset_memory()
            scene_order = sorted(scenes)
            random.shuffle(scene_order)
            batches = list(scene_batches(scenes, scene_order, 4))
            streams, totals, budgets = {}, Counter(), Counter()
            components, histogram, gradient_norms = Counter(), [0] * 10, []
            native_batches, effective_batches, gradient_batches = Counter(), Counter(), Counter()
            task_verified_before_epoch = result.get('task_gradient_verified', False)
            task_checks_before_epoch = result.get('task_gradient_checked_batches', 0)
            staleness = {name: Counter() for name in STALENESS_BUCKETS}
            epoch_started = time.perf_counter()
            processed = []
            for step, entries in enumerate(batches, 1):
                progress = (epoch - 1) + (step - 1) / len(batches)
                lr = learning_rate(progress, settings)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                values = run_batch(root, variant, 'train', epoch, step, entries, dataset,
                                   streams, sender_rows, model, scheduler, parallel,
                                   optimizer, settings, context, result)
                count = len(entries)
                processed.extend(entry['index'] for entry in entries)
                totals['samples'] += count
                totals['optimizer_steps'] += int(values['updated'])
                totals['skipped_steps'] += int(not values['updated'])
                native_batches[values['measure']['native_batch_size']] += 1
                effective_batches[values['measure']['effective_optimizer_batch_size']] += 1
                for name in ('task', 'rate', 'total'):
                    totals[name] += values[name] * count
                for name, value in values['components'].items():
                    components[name] += value * count
                if values['gradient_norm'] is not None:
                    gradient_norms.append(values['gradient_norm'])
                    gradient_batches['nonzero' if values['gradient_norm'] > 0 else 'zero'] += 1
                else:
                    gradient_batches['not_computed'] += 1
                for sample in values['samples']:
                    for name in ('queries', 'updates', 'init', 'optional_queries', 'optional_updates',
                                 'bytes', 'soft_bytes', 'overflow', 'init_overflow'):
                        totals[name] += sample[name]
                    budgets[sample['budget_name']] += 1
                    histogram = [left + right for left, right in zip(histogram, sample['histogram'])]
                    for name, by_bucket in sample['staleness'].items():
                        staleness[name].update(by_bucket)
                result['peak_vram_bytes'] = max(result['peak_vram_bytes'], values['measure']['peak_allocated_bytes'])
                if step == 1 or step % 10 == 0 or step == len(batches):
                    emit(root, {'phase': 'train', 'variant': variant, 'epoch': epoch, 'step': step,
                                'steps': len(batches), 'frames': totals['samples'], 'count': 7362,
                                'native_batch_size': count, 'task': values['task'], 'rate': values['rate'],
                                'total': values['total'], 'lr': lr, 'peak_vram_bytes': result['peak_vram_bytes']})
            if len(processed) != 7362 or len(set(processed)) != 7362 or streams:
                raise ValueError('Formal scene sampler did not process all 7362 samples exactly once')
            assert_frozen(model, frozen)
            if not result['task_gradient_verified']:
                raise RuntimeError('TASK_GRADIENT_BLOCKED: formal training did not establish a finite nonzero native task gradient')
            for scene in sorted(scenes):
                result['scene_budget_history'][scene].append(
                    'B' + str((settings['initial_offset'][scene] + epoch - 1) % 4 + 1))
            elapsed = time.perf_counter() - epoch_started
            record = {'epoch': epoch, 'frames': 7362, 'scenes': 46, 'steps': len(batches),
                      'optimizer_steps': totals['optimizer_steps'], 'skipped_steps': totals['skipped_steps'],
                      'means': {name: totals[name] / 7362 for name in ('task', 'rate', 'total', 'bytes', 'soft_bytes')},
                      'task_components': {name: value / 7362 for name, value in components.items()},
                      'lr': lr, 'lr_progress': progress, 'native_batch_size': 4,
                      'actual_native_batch_counts': dict(native_batches),
                      'actual_effective_optimizer_batch_counts': dict(effective_batches),
                      'scheduler_gradient_status': 'FINITE_NONZERO_OBSERVED' if gradient_batches['nonzero']
                      else 'FINITE_ZERO_ONLY' if gradient_batches['zero'] else 'NO_OPTIMIZER_GRADIENT_COMPUTED',
                      'nonzero_gradient_batch_count': gradient_batches['nonzero'],
                      'zero_gradient_batch_count': gradient_batches['zero'],
                      'gradient_not_computed_batch_count': gradient_batches['not_computed'],
                      'task_gradient_verified': result['task_gradient_verified'],
                      'native_task_gradient_checked_batches': result.get('task_gradient_checked_batches', 0) - task_checks_before_epoch,
                      'native_task_gradient_nonzero_verified_this_epoch':
                      not task_verified_before_epoch and result['task_gradient_verified'],
                      'query_count': totals['queries'], 'update_count': totals['updates'],
                      'forced_init_count': totals['init'],
                      'query_weighted_update_ratio': totals['updates'] / totals['queries'] if totals['queries'] else None,
                      'optional_update_ratio': totals['optional_updates'] / totals['optional_queries'] if totals['optional_queries'] else None,
                      'optional_probability_histogram': histogram, 'budget_coverage': dict(budgets),
                      'staleness_before_decision_buckets': staleness_summary(staleness),
                      'mean_gradient_norm': sum(gradient_norms) / len(gradient_norms) if gradient_norms else None,
                      'elapsed_seconds': elapsed, 'samples_per_second': 7362 / elapsed,
                      'peak_vram_bytes': result['peak_vram_bytes'],
                      'hard_minus_soft_bytes': (totals['bytes'] - totals['soft_bytes']) / 7362,
                      'overflow_bytes': totals['overflow'],
                      'unavoidable_init_overflow_bytes': totals['init_overflow']}
            result['epochs'].append(record)
            result['completed_epoch'] = epoch
            result['updated_utc'] = datetime.now(timezone.utc).isoformat()
            if epoch == 12:
                for scene, history in result['scene_budget_history'].items():
                    if Counter(history) != Counter({'B1': 3, 'B2': 3, 'B3': 3, 'B4': 3}):
                        raise ValueError('Formal 12-epoch scene budget rotation is not balanced: ' + scene)
                result.update(status='PASS', stop_reason='fixed_12_epochs_complete',
                              final_checkpoint=str(final))
            save_checkpoint(latest, scheduler, optimizer, epoch, variant, identity, result, progress)
            if epoch == 12:
                copy_checkpoint(latest, final)
                result['final_sha256'] = runtime.digest(final)
            native.write(report_path, result)
    except Exception as exc:
        result.update(status='V2_FORMAL_EXPERIMENT_BLOCKED', error=str(exc),
                      traceback=traceback.format_exc(), updated_utc=datetime.now(timezone.utc).isoformat())
        native.write(report_path, result)
        emit(root, {'phase': 'train', 'variant': variant, 'status': result['status'], 'error': str(exc)})
        raise
    finally:
        remove()
        del remove
        context.clear()
        streams.clear()
        del parallel, model, scheduler, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return result


def self_check():
    scenes = {'a': [0, 1, 2], 'b': [3], 'c': [4, 5], 'd': [6], 'e': [7, 8]}
    batches = list(scene_batches(scenes, list(scenes), 4))
    flattened = [entry['index'] for batch in batches for entry in batch]
    assert sorted(flattened) == list(range(9)) and len(set(flattened)) == 9
    assert all(len({entry['scene'] for entry in batch}) == len(batch) for batch in batches)
    assert all([entry['index'] for batch in batches for entry in batch if entry['scene'] == scene] == indices
               for scene, indices in scenes.items())
    assert len(batches[-1]) < 4
    settings = {'optimizer': {'peak_lr': 1e-4}, 'lr_schedule': {
        'warmup_epochs': 1, 'warmup_start_lr': 1e-5, 'min_lr': 1e-6,
        'cosine_total_horizon_epochs': 12}}
    assert abs(learning_rate(0, settings) - 1e-5) < 1e-12
    assert abs(learning_rate(1, settings) - 1e-4) < 1e-12
    assert abs(learning_rate(12, settings) - 1e-6) < 1e-12
    assert learning_rate(6, settings) > learning_rate(11, settings)
    for offset in range(4):
        history = [(offset + epoch) % 4 for epoch in range(12)]
        assert Counter(history) == Counter({0: 3, 1: 3, 2: 3, 3: 3})
    ages = [0, -1, 0, 1, 2, 4, 8, 16, 32, 100]
    counts = staleness_counts(ages, [True] + [False] * 9, [.9] + [.25] * 9,
                              [True, False, False, True, False, False, False, False, True, False])
    summary = staleness_summary(counts)
    assert summary['new_initialization']['count'] == 1
    assert summary['new_initialization']['mean_probability'] is None
    assert summary['negative_elapsed']['count'] == 1
    assert summary['0-1']['count'] == 2 and summary['0-1']['optional_update_ratio'] == .5
    assert summary['32+']['count'] == 2 and summary['32+']['optional_update_ratio'] == .5
    assert summary['32+']['mean_probability'] == .25
    assert sum(item['count'] for item in summary.values()) == len(ages)
    print('Formal V2 scene batching / 12-epoch LR / 3x budget rotation / staleness buckets PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--variant', choices=('mlp', 'transformer'), default='mlp')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--correctness-check', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.correctness_check:
        correctness_check(args.root, args.variant, args.resume)
    else:
        train(args.root, args.variant, args.resume)
