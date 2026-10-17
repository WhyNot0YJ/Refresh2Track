"""Real-data calibration, equivalence, gradient gates and isolated structure training."""
import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import runtime
import modules
import native_targets
from receiver import ReceiverCache


def emit(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def canonical(root):
    sys.modules['runtime'] = runtime
    scheduler = runtime.load_module('structure_scheduler', Path(root) / 'scripts/v4_budget_free/scheduler.py')
    sys.modules['scheduler'] = scheduler
    base = runtime.load_module('structure_canonical_training', Path(root) / 'scripts/v4_budget_free/train.py')
    base.ReceiverCache = ReceiverCache
    return base


def normalization(root):
    path = runtime.experiment(root) / 'normalization.json'
    return runtime.read(path) if path.exists() else {'pair_scale': [1.] * 12, 'gate_scale': [1.] * 7}


def build(root, base, cfg, dataset, settings, old_scheduler=False):
    torch = runtime.native.torch
    runtime.inherited.seed(0)
    model = runtime.native.build_model(root, cfg, 'coop').cuda()
    model.CLASSES = dataset.CLASSES
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(0)
        scheduler = base.Scheduler().cuda().train()
    if old_scheduler:
        state = torch.load(str(Path(root) / 'experiments/v4_budget_free/checkpoints/best.pth'), map_location='cpu')
        scheduler.load_state_dict(state['scheduler'], strict=True)
    modules.add_structures(model, normalization(root), experiment=runtime.MODEL)
    model.cuda()
    modules.configure_trainables(model, experiment=runtime.MODEL)
    native_targets.install(model.criterion)
    optimizer = modules.make_optimizer(model, scheduler, settings, experiment=runtime.MODEL)
    return model, scheduler, optimizer


def attach(base, model, context, structured=True):
    from mmcv.parallel import MMDataParallel
    removers = []
    if structured:
        removal, proof = modules.install(model, context)
        removers.append(removal)
    removers.append(base.training.install_hooks(model, context))
    removal, instrumentation = base.native_hooks.install(model, context)
    removers.append(removal)
    original_pts = model.crossview_alignment.transform_pts
    original_labels = model.STReasoner._gen_asso_label

    def transform_pts(points, transform):
        output, keep = original_pts(points, transform)
        lane = context['alignment_indices'][context['alignment_cursor'] - 1]
        sample = context['samples'][lane]
        receiver = context['streams'][sample['scene']]['receiver']
        state = receiver.receiver_state.to(device=points.device, dtype=points.dtype)
        if len(state) != len(points):
            raise ValueError('Receiver sidecar and CAA input rows differ')
        sample['receiver_state'] = state[keep]
        if context.get('capture_evidence', True):
            sample['caa_keep_indices'] = keep.nonzero().flatten().detach().cpu().tolist()
        return output, keep

    def association_labels(*args, **kwargs):
        labels = original_labels(*args, **kwargs)
        lane = context['matching_cursor']
        sample = context['samples'][lane]
        sample['association_positive_pairs'] = int((labels > 0).sum())
        sample['association_negative_pairs'] = int((labels == 0).sum())
        return labels

    model.crossview_alignment.transform_pts = transform_pts
    model.STReasoner._gen_asso_label = association_labels

    def remove():
        model.crossview_alignment.transform_pts = original_pts
        model.STReasoner._gen_asso_label = original_labels
        for callback in reversed(removers):
            callback()
    return MMDataParallel(model, device_ids=[0]), remove, instrumentation


def gradient(loss, parameters):
    torch = runtime.native.torch
    parameters = list(parameters)
    if not loss.requires_grad:
        return 0.
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return modules.gradient_statistics([('objective', values)])['objective']


def run_batch(root, base, epoch, step, entries, dataset, streams, sender_rows,
              model, scheduler, parallel, optimizer, settings, context, optimize=False, diagnose=False,
              capture_evidence=True):
    torch = runtime.native.torch
    optimizer.zero_grad(set_to_none=True)
    packet_hashes = []
    original_pack = base.pack

    def record_pack(*args, **kwargs):
        packet = original_pack(*args, **kwargs)
        packet_hashes.append(hashlib.sha256(packet).hexdigest())
        return packet

    base.pack = record_pack
    try:
        batch, samples, records = base.prepare_batch(root, dataset, entries, streams, sender_rows,
                                                   scheduler, settings, epoch, runtime.MODEL)
    finally:
        base.pack = original_pack
    for record, packet_hash in zip(records, packet_hashes):
        record['packet_sha256'] = packet_hash
    for sample in samples:
        sample['receiver_state'] = torch.empty(0, 4, device='cuda')
    base.training.install_batch_state(model, entries, streams)
    context.clear()
    context.update(samples=samples, streams=streams,
                   alignment_indices=[index for index, sample in enumerate(samples) if len(sample['current']['query_feats'])],
                   alignment_cursor=0, matching_cursor=0, native_forward_count=0,
                   bev_forward_count=0, native_bev_batch_size=0, capture_evidence=capture_evidence)
    losses = parallel(return_loss=True, **batch)
    task, logs = model._parse_losses(losses)
    comm = torch.stack([sample['comm'] for sample in samples]).mean()
    total = task + settings['lambda_comm'] * comm
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite structure training objective')
    if context['native_forward_count'] != 1 or context['bev_forward_count'] != 1:
        raise ValueError('Expected one genuinely batched native forward')
    if context['matching_cursor'] != len(samples) or context['alignment_cursor'] != len(context['alignment_indices']):
        raise ValueError('Native lane coverage differs')
    captured = sum(sample['task_loss'] for sample in samples) / len(samples)
    if not math.isclose(float(task.detach()), captured, rel_tol=3e-5, abs_tol=1e-5):
        raise ValueError('Original per-frame loss reducer changed')
    diagnostics = {}
    if diagnose:
        asso = sum(value.mean() for name, value in losses.items() if 'asso_loss' in name)
        tracking = task - asso
        diagnostics['scheduler_task'] = gradient(task, scheduler.parameters())
        diagnostics['scheduler_comm'] = gradient(comm, scheduler.parameters())
        for name in modules.GBA + modules.FUSION5 + modules.NEW:
            if not modules.allowed('STReasoner.' + name, runtime.MODEL) and not modules.allowed('STReasoner.' + name + '.weight', runtime.MODEL):
                continue
            target = getattr(model.STReasoner, name)
            parameters = target.parameters() if hasattr(target, 'parameters') else [target]
            diagnostics[name] = gradient(asso if name in modules.GBA or name == 'gba_state_adapter' else tracking, parameters)
    norms = {}
    if optimize:
        total.backward()
        norms = modules.gradient_statistics([
            (group['name'], [parameter.grad for parameter in group['params']]) for group in optimizer.param_groups])
        clipped = torch.nn.utils.clip_grad_norm_([parameter for _, parameter in modules.named_trainables(model, scheduler, experiment=runtime.MODEL)], 5.)
        if not math.isfinite(float(clipped)):
            raise FloatingPointError('Nonfinite clipped gradient norm')
        optimizer.step()
    snapshots = []
    if capture_evidence:
        for lane in range(len(entries)):
            tracks = model.train_prev_infos['track_instances'][lane]
            snapshots.append({name: value.detach().cpu().clone() for name, value in tracks.get_fields().items()
                              if torch.is_tensor(value)})
    base.training.save_batch_state(model, entries, streams)
    for record, sample in zip(records, samples):
        record.update(task_loss=sample['task_loss'], total_loss=sample['task_loss'] + .05 * record['comm_loss'])
    features = {name: [sample[name].detach().cpu() for sample in samples if name in sample]
                for name in ('raw_pair_features', 'raw_gate_features')} if capture_evidence else {}
    value = {'step': step, 'task_loss': float(task.detach()), 'comm_loss': float(comm.detach()),
             'total_loss': float(total.detach()), 'loss_components': {key: float(item) for key, item in logs.items()},
             'gradient_norms': norms, 'objective_gradients': diagnostics, 'records': records,
             'positive_pairs': sum(sample.get('association_positive_pairs', 0) for sample in samples),
             'negative_pairs': sum(sample.get('association_negative_pairs', 0) for sample in samples),
             'snapshots': snapshots, 'features': features}
    context.clear()
    return value


def entries_for(base, scenes, epoch):
    order = sorted(scenes)
    random.Random(epoch - 1).shuffle(order)
    return list(base.training.scene_batches(scenes, order, 4))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row if not isinstance(row[key], (list, dict))))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def preflight(root):
    runtime.initialize(root)
    source_hashes = runtime.source_identity(root)
    base = canonical(root)
    torch = runtime.native.torch
    settings = runtime.load_settings(root)
    directory = runtime.experiment(root)
    cfg, dataset, scenes, sender_rows = base.load_training(root)
    batches = entries_for(base, scenes, 1)
    results = {}
    for structured in (False, True):
        model, scheduler, optimizer = build(root, base, cfg, dataset, settings, old_scheduler=True)
        context, streams = {}, {}
        parallel, remove, _ = attach(base, model, context, structured)
        runtime.inherited.seed(0)
        model.reset_memory()
        rows = []
        try:
            for step, entries in enumerate(batches[:4], 1):
                with torch.no_grad():
                    value = run_batch(root, base, 1, step, entries, dataset, streams, sender_rows,
                                      model, scheduler, parallel, optimizer, settings, context)
                rows.append(value)
                emit(phase='step0_equivalence', structured=structured, step=step)
            results[structured] = rows
        finally:
            remove()
            del parallel, model, scheduler, optimizer
            streams.clear()
            gc.collect()
            torch.cuda.empty_cache()
    for reference, changed in zip(results[False], results[True]):
        for before, after in zip(reference['snapshots'], changed['snapshots']):
            if set(before) != set(after):
                raise ValueError('Step0 tracking field set differs')
            for name in before:
                if before[name].dtype.is_floating_point:
                    same = torch.allclose(before[name], after[name], atol=2e-5, rtol=2e-5)
                else:
                    same = torch.equal(before[name], after[name])
                if not same:
                    raise ValueError('Step0 tracking tensor differs: ' + name)
        for before, after in zip(reference['records'], changed['records']):
            for key in ('queries', 'updates', 'optional_updates', 'actual_bytes', 'feature_payload_bytes', 'packet_sha256'):
                if before[key] != after[key]:
                    raise ValueError('Step0 communication differs: ' + key)
    runtime.write(directory / 'preflight/step0_training_equivalence.json',
                  {'status': 'PASS', 'real_batches': 4, 'atol': 2e-5, 'rtol': 2e-5,
                   'scheduler_checkpoint': 'old V4 best; equivalence only; never formal training'})
    del results
    if runtime.MODEL == 'C_GBA_GATE':
        model, scheduler, optimizer = build(root, base, cfg, dataset, settings)
        context, streams = {}, {}
        parallel, remove, _ = attach(base, model, context)
        totals = {'raw_pair_features': torch.zeros(12, dtype=torch.float64),
                  'raw_gate_features': torch.zeros(7, dtype=torch.float64)}
        counts = dict.fromkeys(totals, 0)
        try:
            model.reset_memory()
            for step, entries in enumerate(batches[:24], 1):
                with torch.no_grad():
                    value = run_batch(root, base, 1, step, entries, dataset, streams, sender_rows,
                                      model, scheduler, parallel, optimizer, settings, context)
                for name, tensors in value['features'].items():
                    for tensor in tensors:
                        width = totals[name].numel()
                        tensor = tensor.reshape(-1, width).double()
                        totals[name] += tensor.square().sum(0)
                        counts[name] += len(tensor)
                emit(phase='train_only_normalization', step=step, steps=24)
        finally:
            remove()
            del parallel, model, scheduler, optimizer
            streams.clear()
            gc.collect()
            torch.cuda.empty_cache()
        if not all(counts.values()):
            raise ValueError('Train-only normalization has no real pair/gate observations')
        scales = {key: (totals[name] / counts[name]).sqrt().clamp_min(1.).tolist()
                  for key, name in (('pair_scale', 'raw_pair_features'), ('gate_scale', 'raw_gate_features'))}
        scales['pair_scale'][8:10] = [1., 1.]
        scales['gate_scale'][:4] = scales['pair_scale'][:4]
        scales['gate_scale'][4] = scales['pair_scale'][7]
        scales['gate_scale'][6] = 1.
        scales.update(status='PASS', split='train only; first 24 seed0 batches; no optimization', counts=counts,
                      token_order=[dataset.data_infos[entry['index']]['token'] for batch in batches[:24] for entry in batch])
        path = directory / 'normalization.json'
        if path.exists() and runtime.read(path) != scales:
            raise ValueError('Existing normalization constants differ')
        runtime.write(path, scales)
    else:
        shared = Path(root) / 'configs/structure_v4_joint/normalization.json'
        path = directory / 'normalization.json'
        scales = runtime.read(shared)
        tokens = [dataset.data_infos[entry['index']]['token'] for batch in batches[:24] for entry in batch]
        if scales.get('status') != 'PASS' or scales.get('token_order') != tokens:
            raise ValueError('Remote requires the identical passed local train-only calibration')
        if runtime.digest(path) != runtime.digest(shared):
            raise ValueError('Remote normalization bytes differ from the shared local constants')
        runtime.write(directory / 'preflight/shared_normalization.json',
                      {'status': 'PASS', 'sha256': runtime.digest(shared), 'source': str(shared),
                       'scope': 'reuse fixed local train-only constants; actual seed0 token order verified'})
    model, scheduler, optimizer = build(root, base, cfg, dataset, settings)
    initial_hash = modules.state_hash(scheduler.state_dict())
    proof = modules.parameter_proof(model, scheduler, optimizer, experiment=runtime.MODEL)
    frozen = modules.frozen_snapshot(model, experiment=runtime.MODEL)
    before = modules.native_state(model, experiment=runtime.MODEL)
    context, streams = {}, {}
    parallel, remove, _ = attach(base, model, context)
    diagnostics, positive, negative = {}, 0, 0
    rows = []
    try:
        model.reset_memory()
        for step, entries in enumerate(batches[:64], 1):
            progress = (step - 1) / max(1, len(batches) - 1)
            learning_rate = base.training.learning_rate(progress, settings)
            for group in optimizer.param_groups:
                group['lr'] = learning_rate * group['initial_lr'] / settings['optimizer']['peak_lr']
            value = run_batch(root, base, 1, step, entries, dataset, streams, sender_rows,
                              model, scheduler, parallel, optimizer, settings, context,
                              optimize=True, diagnose=True)
            positive += value['positive_pairs']
            negative += value['negative_pairs']
            for name, norm in value['objective_gradients'].items():
                diagnostics[name] = max(diagnostics.get(name, 0.), norm)
            modules.assert_frozen(model, frozen, experiment=runtime.MODEL)
            rows.append({key: value[key] for key in ('step', 'task_loss', 'comm_loss', 'total_loss', 'objective_gradients')})
            emit(phase='gradient_preflight', step=step, positive_pairs=positive,
                 objective_gradients=diagnostics)
            if step >= 3 and positive and negative and diagnostics and all(norm > 0 for norm in diagnostics.values()):
                break
        if not positive or not negative or not diagnostics or not all(norm > 0 for norm in diagnostics.values()):
            raise ValueError('Real-data task/communication/association/fusion gradients failed')
        after = modules.native_state(model, experiment=runtime.MODEL)
        changed = {name: float((after[name] - tensor).abs().max()) for name, tensor in before.items()}
        for name in modules.GBA + modules.FUSION5 + modules.NEW:
            if not modules.allowed('STReasoner.' + name, runtime.MODEL) and not modules.allowed('STReasoner.' + name + '.weight', runtime.MODEL):
                continue
            if not any(delta > 0 for key, delta in changed.items() if key == 'STReasoner.' + name or key.startswith('STReasoner.' + name + '.')):
                raise ValueError('Allowed module did not update: ' + name)
        modules.assert_frozen(model, frozen, full=True, experiment=runtime.MODEL)
        runtime.write(directory / 'preflight/grad_and_weight_delta.json',
                      dict(status='PASS', objective_gradients=diagnostics, max_weight_deltas=changed,
                           positive_pairs=positive, negative_pairs=negative, batches=rows,
                           frozen_weights_and_BN='PASS', scheduler_initial_sha256=initial_hash,
                           native_association_loss='unchanged native FocalLoss weight10 gamma1 alpha-1; no extra loss'))
        runtime.write(directory / 'preflight/parameter_allowlist.json', proof)
        write_csv(directory / 'preflight/freeze_trainable_params.csv',
                  [{'name': name, 'parameters': parameter.numel(), 'requires_grad': parameter.requires_grad}
                   for name, parameter in list(model.named_parameters()) + [('scheduler.' + name, parameter) for name, parameter in scheduler.named_parameters()]])
    finally:
        remove()
        del parallel, model, scheduler, optimizer
        streams.clear()
        gc.collect()
        torch.cuda.empty_cache()
    subprocess.run([sys.executable, '-B', str(HERE / 'step0.py'), '--root', str(root)], check=True)
    if source_hashes != runtime.source_identity(root):
        raise ValueError('Experiment source changed during preflight; rerun the final stable implementation')
    runtime.freeze_manifest(root)
    runtime.write(directory / 'preflight/preflight.json',
                  {'status': 'PASS', 'identity': runtime.identity(root),
                   'scheduler_initial_sha256': initial_hash, 'real_gradient_preflight': True,
                   'real_matching_inference_equivalence': True,
                   'no_receiver_leakage': True, 'frozen_weights_and_BN': True})
    emit(phase='preflight_complete', status='PASS')


def train_epoch(root, epoch):
    runtime.initialize(root)
    base = canonical(root)
    torch = runtime.native.torch
    directory = runtime.experiment(root)
    settings, identity = runtime.load_settings(root), runtime.identity(root)
    if runtime.read(directory / 'preflight/preflight.json').get('identity') != identity:
        raise ValueError('Matching passed preflight is required')
    cfg, dataset, scenes, sender_rows = base.load_training(root)
    model, scheduler, optimizer = build(root, base, cfg, dataset, settings)
    latest = directory / 'checkpoints/latest.pth'
    history, selection = [], {}
    if latest.exists():
        state = torch.load(str(latest), map_location='cpu')
        if state['identity'] != identity or state['epoch'] != epoch - 1:
            raise ValueError('Resume requires the exact preceding C checkpoint')
        scheduler.load_state_dict(state['scheduler'], strict=True)
        modules.load_native_state(model, state['native_trainable'], experiment=runtime.MODEL)
        optimizer.load_state_dict(state['optimizer'])
        runtime.inherited.restore_rng(state['rng'])
        history, selection = state['training_result']['epochs'], state.get('selection', {})
    elif epoch != 1:
        raise ValueError('Previous checkpoint is missing')
    elif modules.state_hash(scheduler.state_dict()) != runtime.read(directory / 'preflight/preflight.json')['scheduler_initial_sha256']:
        raise ValueError('Formal Scheduler did not start from the verified fresh seed0 state')
    frozen = modules.frozen_snapshot(model, experiment=runtime.MODEL)
    context, streams = {}, {}
    parallel, remove, instrumentation = attach(base, model, context)
    batches = entries_for(base, scenes, epoch)
    records, processed, rows = [], [], []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        model.reset_memory()
        for step, entries in enumerate(batches, 1):
            progress = epoch - 1 + (step - 1) / max(1, len(batches) - 1)
            learning_rate = base.training.learning_rate(progress, settings)
            for group in optimizer.param_groups:
                group['lr'] = learning_rate * group['initial_lr'] / settings['optimizer']['peak_lr']
            value = run_batch(root, base, epoch, step, entries, dataset, streams, sender_rows,
                              model, scheduler, parallel, optimizer, settings, context,
                              optimize=True, diagnose=step in (2, 10), capture_evidence=False)
            modules.assert_frozen(model, frozen, experiment=runtime.MODEL)
            processed.extend(entry['index'] for entry in entries)
            records.extend(value['records'])
            row = {key: item for key, item in value.items() if key not in ('snapshots', 'features', 'records')}
            rows.append(row)
            log = directory / 'logs/train.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(epoch=epoch, **row), allow_nan=False) + '\n')
            if step == 1 or step % 25 == 0:
                emit(phase='training', epoch=epoch, step=step, steps=len(batches), samples=len(processed),
                     task_loss=value['task_loss'], comm_loss=value['comm_loss'], lr=learning_rate)
        if sorted(processed) != list(range(7362)) or streams:
            raise ValueError('Training epoch did not visit every official frame exactly once')
        modules.assert_frozen(model, frozen, full=True, experiment=runtime.MODEL)
        summary = dict(base.statistics(records), epoch=epoch, optimizer_steps=len(batches),
                       seconds=time.perf_counter() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated())
        history.append(summary)
        checkpoint = {'epoch': epoch, 'identity': identity, 'scheduler': scheduler.state_dict(),
                      'native_trainable': modules.native_state(model, experiment=runtime.MODEL), 'optimizer': optimizer.state_dict(),
                      'rng': runtime.inherited.rng_state(), 'lr_progress': float(epoch),
                      'training_result': {'epochs': history}, 'selection': selection,
                      'normalization': normalization(root), 'train_cache_state': {},
                      'cache_resume_boundary': 'all scene lanes drained at completed epoch; next epoch cold resets'}
        latest.parent.mkdir(parents=True, exist_ok=True)
        temporary = latest.with_suffix('.pth.writing')
        torch.save(checkpoint, str(temporary))
        temporary.replace(latest)
        write_csv(directory / 'communication' / ('train_epoch%03d.csv' % epoch), records)
        runtime.write(directory / 'metrics/train_history.json', history)
        return summary
    finally:
        remove()
        del parallel, model, scheduler, optimizer
        streams.clear()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--epoch', type=int)
    args = parser.parse_args()
    if args.preflight:
        preflight(args.root)
    elif args.epoch is not None and 1 <= args.epoch <= 36:
        train_epoch(args.root, args.epoch)
    else:
        parser.error('Select --preflight or --epoch 1..36')
