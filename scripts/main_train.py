"""Train only APTU parameters on every official cooperative train frame."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import traceback
import time

import main_runtime as native


def training_metas(value):
    value = native.unwrap(value)
    if isinstance(value, dict):
        if 'sample_idx' in value:
            return [value]
        return [meta for item in value.values() for meta in training_metas(item)]
    if isinstance(value, (list, tuple)):
        return [meta for item in value for meta in training_metas(item)]
    return []


def identity(root):
    return {name: hashlib.sha256((root / 'configs/main_experiment' / name).read_bytes()).hexdigest()
            for name in ('training.json', 'budgets.json')}


def append_resource(root, variant, epoch, frame, elapsed_ms):
    torch = native.torch
    path = root / 'logs/main/resource.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        if not exists:
            writer.writerow(['timestamp_utc', 'phase', 'variant', 'epoch', 'frame', 'elapsed_ms',
                             'allocated_bytes', 'reserved_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes'])
        writer.writerow([datetime.now(timezone.utc).isoformat(), 'train', variant, epoch, frame, elapsed_ms,
                         torch.cuda.memory_allocated(), torch.cuda.memory_reserved(),
                         torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()])


def install_native_hooks(model, context, association_weight):
    """Teacher association reuses the student's exact current vehicle instance state."""
    torch = native.torch
    from main_message import FIELDS
    original_match = model.STReasoner._learn_matching
    original_stream = model.forward_track_stream_train

    def alignment_input(module, incoming):
        loaded, transform = incoming
        for name in ('ref_pts', 'pred_boxes'):
            if not torch.equal(loaded[name], context['current'][name]):
                raise ValueError('Native loaded geometry differs from the current sender frame')
        if association_weight:
            with torch.no_grad():
                teacher_fields = {name: context['current'][name].detach().clone() for name in FIELDS}
                teacher_fields = module.forward(teacher_fields, transform)
                context['teacher_inf'] = model._init_inf_tracks(teacher_fields)
        return ({name: context['student'][name] for name in FIELDS}, transform)

    def receiver_output(module, incoming, outgoing):
        context['received_queries'] = len(outgoing['query_feats'])

    def learn_matching(inf, veh, inf_points, veh_points, mask):
        result = original_match(inf, veh, inf_points, veh_points, mask)
        student = result[2]
        context['student_affinity_requires_grad'] = student.requires_grad
        if association_weight and student.numel():
            # Only the three fields read by native matching need copying.
            with torch.no_grad():
                vehicle_copy = type(veh)(veh.image_size)
                for name in ('cache_query_feats', 'cache_motion_feats', 'cache_ref_pts'):
                    vehicle_copy.set(name, veh.get(name).detach().clone())
                teacher = original_match(context['teacher_inf'], vehicle_copy,
                                         inf_points.detach(), veh_points.detach(), mask)[2]
            if student.shape != teacher.shape:
                raise ValueError('Teacher/student association query sets differ')
            if not torch.isfinite(teacher).all() or not torch.isfinite(student).all():
                raise ValueError('Nonfinite native affinity logits')
            context['association_loss'] = torch.nn.functional.smooth_l1_loss(student, teacher)
        return result

    def raw_stream(*args, **kwargs):
        losses = original_stream(*args, **kwargs)
        if any(not torch.isfinite(value).all() for value in losses.values()):
            raise ValueError('Native task loss contains NaN/Inf before outer nan_to_num')
        return losses

    handles = [model.crossview_alignment.register_forward_pre_hook(alignment_input),
               model.crossview_alignment.register_forward_hook(receiver_output)]
    model.STReasoner._learn_matching = learn_matching
    model.forward_track_stream_train = raw_stream
    return handles


def save_checkpoint(path, scheduler, optimizer, epoch, variant, fingerprints, result):
    torch = native.torch
    checkpoint = {'scheduler': scheduler.state_dict(), 'optimizer': optimizer.state_dict(),
                  'lr_scheduler': None, 'epoch': epoch, 'variant': variant,
                  'trainable_params': sum(p.numel() for p in scheduler.parameters()),
                  'identity': fingerprints, 'training_result': result,
                  'rng': {'python': random.getstate(), 'numpy': native.np.random.get_state(),
                          'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}}
    temp = path.with_suffix('.writing')
    torch.save(checkpoint, str(temp))
    temp.replace(path)


def train(root, variant, epochs, resume):
    root = root.resolve()
    native.initialize(root)
    torch = native.torch
    from mmcv.parallel import MMDataParallel
    from torch.utils.data import Subset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from main_message import (FIELDS, FEATURES, TemporalCache, ReceiverCache, int8_roundtrip,
                              message_sizes, select_updates, pack)
    from main_scheduler import Scheduler, straight_through_latents
    settings = native.read(root / 'configs/main_experiment/training.json')
    budget_config = native.read(root / 'configs/main_experiment/budgets.json')
    budgets = budget_config['budgets_bytes']
    baseline_bytes = float(budget_config['full_int8_mean_bytes'])
    if len(budgets) != 4 or any(budget <= 0 for budget in budgets) or baseline_bytes <= 0:
        raise ValueError('Four positive full-train locked budgets are required')
    if not 1 <= epochs <= settings['epochs']:
        raise ValueError('Requested epoch count exceeds the locked training configuration')
    cache_manifest = native.read(root / 'query_cache/train/manifest.json')
    if cache_manifest.get('status') != 'FULL_TRAIN_CACHE_PASS' or cache_manifest.get('frames') != 7445:
        raise ValueError('All 7445 native infrastructure train caches must complete before training')
    random.seed(settings['seed'])
    native.np.random.seed(settings['seed'])
    torch.manual_seed(settings['seed'])
    torch.cuda.manual_seed_all(settings['seed'])
    cfg = native.make_config(root, 'coop', 'train', training=True)
    cfg.model.batch_size = 1
    cfg.data.train.samples_per_gpu = 1
    cfg.data.train.filter_empty_gt = False
    cfg.dump(str(root / 'configs/main_experiment/coop_train_train.py'))
    dataset = native.dataset_for(cfg, training=True)
    scenes = {}
    for index, info in enumerate(dataset.data_infos):
        scenes.setdefault(str(info['scene_token']), []).append(index)
    official_scenes = set(native.read(root / 'code/CoopTrack/data/split_datas/cooperative-split-data-spd.json')['batch_split']['train'])
    if len(dataset) != 7362 or len(scenes) != 46 or set(scenes) != official_scenes:
        raise ValueError('Formal training requires all 7362 samples in all 46 official train scenes')
    sender_rows = {row['token']: row for row in native.native_manifest(root, 'train')}
    if any(str(info['token_inf']) not in sender_rows for info in dataset.data_infos):
        raise ValueError('Missing full-train infrastructure mapping')
    model = native.build_model(root, cfg, 'coop').cuda()
    model.CLASSES = dataset.CLASSES
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    scheduler = Scheduler(variant).cuda().train()
    trainable_params = sum(p.numel() for p in scheduler.parameters())
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('Original CoopTrack parameters must be frozen')
    optimizer = torch.optim.AdamW(scheduler.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    directory = root / 'experiments/main/checkpoints' / variant
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / 'latest.pth'
    report = root / 'experiments/main/metrics' / (variant + '_train.json')
    fingerprints = identity(root)
    start_epoch = 0
    result = {'status': 'RUNNING', 'variant': variant, 'epochs': [], 'train_samples': 7362,
              'train_scenes': 46, 'trainable_params': trainable_params, 'base_trainable_params': 0,
              'peak_vram_bytes': 0, 'precision': 'FP32', 'identity': fingerprints,
              'checkpoint_selection': 'last_stable_epoch', 'val_selection': False,
              'task_gradient_verified': False, 'started_utc': datetime.now(timezone.utc).isoformat()}
    if latest.exists():
        if not resume:
            raise ValueError('A checkpoint exists; resume is required to preserve the formal experiment')
        checkpoint = torch.load(str(latest), map_location='cpu')
        if checkpoint['identity'] != fingerprints or checkpoint['variant'] != variant:
            raise ValueError('Resume checkpoint differs from locked experiment identity')
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        result = checkpoint['training_result']
        random.setstate(checkpoint['rng']['python'])
        native.np.random.set_state(checkpoint['rng']['numpy'])
        torch.set_rng_state(checkpoint['rng']['torch'])
        torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
    if start_epoch >= epochs:
        if result['status'] != 'PASS':
            raise ValueError('Completed checkpoint lacks a stable training result')
        native.write(report, result)
        return result
    association_weight = 0.0 if variant == 'no_assoc' else settings['lambda_assoc']
    context = {}
    handles = install_native_hooks(model, context, association_weight)
    parallel = MMDataParallel(model, device_ids=[0])
    torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(start_epoch + 1, epochs + 1):
            model.reset_memory()
            sender, receiver = TemporalCache(), ReceiverCache()
            scene_order = sorted(scenes)
            random.shuffle(scene_order)
            order = [index for scene in scene_order for index in scenes[scene]]
            loader = build_dataloader(Subset(dataset, order), samples_per_gpu=1, workers_per_gpu=0,
                                      dist=False, shuffle=False, nonshuffler_sampler=cfg.data.nonshuffler_sampler)
            totals = {name: 0.0 for name in ('task', 'association', 'rate', 'total', 'bytes')}
            updates, forced_over_budget = 0, 0
            epoch_started = time.perf_counter()
            for position, data in enumerate(loader):
                info = dataset.data_infos[order[position]]
                metadata = training_metas(data['img_metas'])
                if len(metadata) != 1 or str(metadata[0]['sample_idx']) != str(info['token']) or str(metadata[0]['sample_idx_inf']) != str(info['token_inf']):
                    raise ValueError('Training loader substituted or reordered an official frame')
                meta = sender_rows[str(info['token_inf'])]
                raw = native.fields(root / 'query_cache/train' / (meta['token'] + '.pkl'))
                native.assert_loaded_queries(data, {name: raw[name] for name in FIELDS})
                raw = {name: raw[name] for name in FIELDS + ('obj_idxes', 'scores')}
                current = int8_roundtrip(raw)
                temporal = sender.prepare(current, meta)
                budget = int(random.choice(budgets))
                torch.cuda.synchronize()
                started = time.perf_counter()
                tokens = temporal['tokens'].cuda()
                logits = scheduler(tokens, budget / baseline_bytes)
                uniform = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
                probability = torch.sigmoid((logits + torch.log(uniform) - torch.log1p(-uniform)) / settings['gate_temperature'])
                mandatory, per_update, full_bytes = message_sizes(raw, meta)
                flags = select_updates(probability.detach().cpu(), temporal['force'], mandatory, per_update, budget)
                payload = pack(raw, meta, flags)
                decoded, header = receiver.unpack(payload, meta)
                for name in FEATURES:
                    if not torch.equal(decoded[name][flags], current[name][flags]) or not torch.equal(decoded[name][~flags], temporal['last'][name][~flags]):
                        raise ValueError('Decoded UPDATE/REUSE differs from the full INT8 teacher or sender mirror')
                sender.commit(raw, flags, meta)
                if len(payload) > budget:
                    forced_over_budget += 1
                current_gpu = {name: value.cuda() for name, value in current.items() if torch.is_tensor(value)}
                last_gpu = {name: value.cuda() for name, value in temporal['last'].items() if torch.is_tensor(value)}
                decoded_gpu = {name: value.cuda() for name, value in decoded.items()}
                force_gpu = temporal['force'].cuda()
                student = straight_through_latents(decoded_gpu, current_gpu, last_gpu, probability, force_gpu)
                context.clear()
                context.update(current=current_gpu, student=student, association_loss=logits.sum() * 0,
                               received_queries=0, teacher_inf=None, student_affinity_requires_grad=False)
                optimizer.zero_grad(set_to_none=True)
                losses = parallel(return_loss=True, **data)
                task, log_vars = model._parse_losses(losses)
                expected = mandatory + (torch.where(force_gpu, torch.ones_like(probability), probability) *
                                        torch.as_tensor(per_update, device=logits.device, dtype=logits.dtype)).sum()
                feasible_budget = max(min(budget, full_bytes), mandatory + float((temporal['force'].to(torch.float64) *
                                     torch.as_tensor(per_update, dtype=torch.float64)).sum()))
                rate = ((expected - feasible_budget) / baseline_bytes).square()
                association = context['association_loss']
                total = task + association_weight * association + settings['lambda_rate'] * rate
                if not torch.isfinite(total):
                    raise ValueError('Nonfinite formal training loss')
                optional = bool((~force_gpu).any())
                if not result['task_gradient_verified'] and optional and context['received_queries'] and task.requires_grad:
                    gradients = torch.autograd.grad(task, tuple(scheduler.parameters()), retain_graph=True, allow_unused=True)
                    norm = sum(float(gradient.detach().square().sum()) for gradient in gradients if gradient is not None) ** 0.5
                    if norm > 0:
                        if not context['student_affinity_requires_grad']:
                            raise ValueError('Task graph does not include native student affinity')
                        result['task_gradient_verified'] = True
                        result['first_task_gradient_norm'] = norm
                        result['first_task_gradient_token'] = str(info['token'])
                        print(json.dumps({'phase': 'task_gradient_verified', 'variant': variant,
                                          'token': str(info['token']), 'gradient_norm': norm}), flush=True)
                # All-forced or empty sets have no optional scheduling action to optimize.
                if optional and total.requires_grad:
                    total.backward()
                    gradients = [parameter.grad for parameter in scheduler.parameters() if parameter.grad is not None]
                    if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                        raise ValueError('Absent or nonfinite scheduler gradients')
                    torch.nn.utils.clip_grad_norm_(scheduler.parameters(), settings['gradient_clip'])
                    optimizer.step()
                    updates += 1
                if any(parameter.grad is not None for parameter in model.parameters()):
                    raise ValueError('Frozen official model received parameter gradients')
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000
                for name, value in (('task', task), ('association', association), ('rate', rate), ('total', total)):
                    totals[name] += float(value.detach())
                totals['bytes'] += len(payload)
                result['peak_vram_bytes'] = max(result['peak_vram_bytes'], torch.cuda.max_memory_allocated())
                append_resource(root, variant, epoch, position + 1, elapsed_ms)
                if position % 10 == 0 or position == len(order) - 1:
                    print(json.dumps({'phase': 'train', 'variant': variant, 'epoch': epoch, 'frame': position + 1,
                                      'count': len(order), 'token': str(info['token']), 'loss': float(total.detach()),
                                      'task': float(task.detach()), 'association': float(association.detach()),
                                      'rate': float(rate.detach()), 'budget': budget, 'payload_bytes': len(payload),
                                      'updates': updates, 'peak_vram_bytes': result['peak_vram_bytes']}), flush=True)
                # Release both task and teacher graph references after the current frame's backward.
                context.clear()
                del losses, task, association, rate, total, student, logits, probability
            if not updates or not result['task_gradient_verified']:
                raise ValueError('A complete epoch did not establish task-loss gradients to the scheduler')
            result['epochs'].append({'epoch': epoch, 'frames': len(order), 'scenes': len(scenes),
                                     'optimizer_updates': updates, 'forced_over_budget_frames': forced_over_budget,
                                     'means': {name: value / len(order) for name, value in totals.items()},
                                     'elapsed_seconds': time.perf_counter() - epoch_started})
            result.update(status='PASS' if epoch == epochs else 'RUNNING', last_stable_epoch=epoch,
                          checkpoint=str(latest), completed_utc=datetime.now(timezone.utc).isoformat())
            save_checkpoint(directory / ('epoch_%02d.pth' % epoch), scheduler, optimizer, epoch, variant, fingerprints, result)
            save_checkpoint(latest, scheduler, optimizer, epoch, variant, fingerprints, result)
            native.write(report, result)
            print(json.dumps({'epoch_complete': epoch, 'variant': variant, 'frames': 7362, 'status': result['status']}), flush=True)
        return result
    except Exception:
        result.update(status='FAILED', exact_blocker=traceback.format_exc(),
                      peak_vram_bytes=max(result['peak_vram_bytes'], torch.cuda.max_memory_allocated()),
                      minimal_next_action='Resume the same full-data experiment from the last stable epoch; for CUDA OOM use the requested 32GB GPU.')
        native.write(report, result)
        raise
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--variant', choices=('aptu', 'mlp', 'no_assoc'), default='aptu')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--resume', action='store_true', default=True)
    args = parser.parse_args()
    result = train(args.root, args.variant, args.epochs, args.resume)
    print(json.dumps({'training': result['status'], 'variant': args.variant, 'epochs': result['last_stable_epoch'],
                      'trainable_params': result['trainable_params'], 'peak_vram_bytes': result['peak_vram_bytes']}), flush=True)


if __name__ == '__main__':
    main()
