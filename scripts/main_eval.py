"""Formal full-train cache/statistics and full-val temporal communication runs."""
import argparse
import base64
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import random
import time
import traceback
import main_runtime as runtime
from main_message import (FIELDS, FEATURES, ReceiverCache, TemporalCache, baseline_int8_pack,
                          int8_roundtrip, message_sizes, pack, select_updates, validate)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save_rng():
    torch, np = runtime.torch, runtime.np
    numpy_state = np.random.get_state()
    encode = lambda state: base64.b64encode(state.cpu().numpy().tobytes()).decode('ascii')
    return {'python': random.getstate(), 'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            'torch': encode(torch.get_rng_state()), 'cuda': [encode(state) for state in torch.cuda.get_rng_state_all()]}


def restore_rng(saved):
    torch, np = runtime.torch, runtime.np
    python_state = saved['python']
    random.setstate((python_state[0], tuple(python_state[1]), python_state[2]))
    numpy_state = saved['numpy']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    decode = lambda state: torch.from_numpy(np.frombuffer(base64.b64decode(state), dtype=np.uint8).copy())
    torch.set_rng_state(decode(saved['torch']))
    torch.cuda.set_rng_state_all([decode(state) for state in saved['cuda']])


def preflight(root):
    """Only imports, commit, formal metadata/cache reads and CPU checkpoint load."""
    torch = runtime.torch
    official = runtime.read(root / 'code/CoopTrack/data/split_datas/cooperative-split-data-spd.json')['batch_split']
    expected = {'veh': {'train': 7110, 'val': 3316}, 'inf': {'train': 7445, 'val': 3316},
                'coop': {'train': 7362, 'val': 3316}}
    counts = {}
    for role, split_counts in expected.items():
        counts[role] = {}
        for split, count in split_counts.items():
            infos = runtime.load_infos(root, role, split)
            scenes = {str(row['scene_token']) for row in infos}
            if len(infos) != count or scenes != set(official[split]):
                raise ValueError('Formal inherited infos identity differs: ' + role + '/' + split)
            counts[role][split] = {'frames': len(infos), 'scenes': len(scenes)}
    val_rows = runtime.native_manifest(root, 'val')
    expected_files = {row['token'] + '.pkl' for row in val_rows}
    actual_files = {path.name for path in (root / 'query_cache/val').glob('*.pkl')}
    if actual_files != expected_files:
        raise ValueError('Protected full-val cache paths do not match official infrastructure tokens')
    first = root / 'query_cache/val' / (val_rows[0]['token'] + '.pkl')
    cache_count = validate(runtime.fields(first))
    checkpoints = {}
    for role in ('inf', 'coop'):
        path = root / 'checkpoints' / runtime.CHECKPOINTS[role][0]
        checkpoint = torch.load(str(path), map_location='cpu')
        if not isinstance(checkpoint, dict):
            raise ValueError('Official checkpoint must be a dictionary: ' + role)
        state = checkpoint.get('state_dict', checkpoint)
        if not isinstance(state, dict) or not state or not any(torch.is_tensor(value) for value in state.values()):
            raise ValueError('Official checkpoint has no tensor state: ' + role)
        if any(value.device.type != 'cpu' for value in state.values() if torch.is_tensor(value)):
            raise ValueError('Minimal checkpoint check must remain CPU only')
        checkpoints[role] = {'file': str(path), 'loaded': True, 'state_keys': len(state),
                             'inherited_md5': runtime.CHECKPOINTS[role][1], 'md5_recomputed': False}
        del state, checkpoint
    result = {'status': 'PREFLIGHT_PASS', 'commit': runtime.COMMIT, 'infos': counts,
              'val_cache_files': len(actual_files), 'read_cache_file': str(first),
              'read_cache_queries': cache_count, 'checkpoints': checkpoints,
              'model_forward_executed': False, 'GPU_model_loaded': False, 'hash_audit_executed': False}
    runtime.write(root / 'experiments/main/metrics/preflight.json', result)
    print(runtime.json.dumps(result, ensure_ascii=False), flush=True)
    return result


def cache_train(root, resume=False):
    """Replay all native infrastructure frames; resume only completed whole scenes."""
    torch, mmcv = runtime.torch, runtime.mmcv
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from torch.utils.data import Subset
    from mmcv.parallel import MMDataParallel
    rows = runtime.native_manifest(root, 'train')
    cache_root = root / 'query_cache/train'
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / 'manifest.json'
    identity = {'commit': runtime.COMMIT, 'checkpoint': runtime.CHECKPOINTS['inf'][0],
                'checkpoint_md5': runtime.CHECKPOINTS['inf'][1],
                'infos_sha256': sha(root / 'data_infos/V2X-Seq-SPD-CoopTrack-full/infrastructure-side/spd_infos_temporal_train.pkl'),
                'split_sha256': sha(root / 'code/CoopTrack/data/split_datas/cooperative-split-data-spd.json'),
                'native_tokens_sha256': hashlib.sha256(str([row['token'] for row in rows]).encode()).hexdigest(),
                'precision': 'FP32', 'seed': 0,
                'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32}
    completed, audited, resume_rng = [], [], None
    if resume and manifest_path.is_file():
        previous = runtime.read(manifest_path)
        if previous.get('identity') != identity:
            raise ValueError('Full train cache resume identity differs')
        completed = previous['completed_scenes']
        audited = previous['rows']
        resume_rng = previous.get('rng_state')
        if any(not (cache_root / (row['token'] + '.pkl')).is_file() for row in audited):
            raise ValueError('A completed scene cache is missing; do not silently resume')
        if previous.get('status') == 'FULL_TRAIN_CACHE_PASS':
            runtime.write(root / 'experiments/main/metrics/train_cache.json', previous)
            print('FULL_TRAIN_CACHE_PASS frames=7445 scenes=46 reused=True', flush=True)
            return previous
        if completed and resume_rng is None:
            raise ValueError('Completed scene cache lacks native RNG state for exact scene-boundary resume')
    cfg = runtime.make_config(root, 'inf', 'train')
    dataset = runtime.dataset_for(cfg)
    if [str(info['token']) for info in dataset.data_infos] != [row['token'] for row in rows]:
        raise ValueError('Full sender dataset differs from native manifest order')
    indices = [index for index, row in enumerate(rows) if row['scene'] not in completed]
    model = runtime.build_model(root, cfg, 'inf').cuda()
    model.CLASSES = dataset.CLASSES
    parallel = MMDataParallel(model, device_ids=[0])
    loader = build_dataloader(Subset(dataset, indices), samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False, nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    report = {'status': 'RUNNING', 'identity': identity, 'rows': audited,
              'completed_scenes': completed, 'frames': len(audited), 'scenes': len(completed),
              'native_reset': 'scene change; dt>0.5s; reversed timestamp', 'started_utc': now()}
    if resume_rng is not None:
        report['rng_state'] = resume_rng
    scene_rows = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        iterator = iter(loader)
        if resume_rng is not None:
            restore_rng(resume_rng)
        for position, data in enumerate(iterator):
            index = indices[position]
            metadata = rows[index]
            metas = runtime.collect_metas(data['img_metas'])
            if len(metas) != 1 or str(metas[0]['sample_idx']) != metadata['token']:
                raise ValueError('Full train sender sample token mapping differs')
            torch.cuda.synchronize()
            frame_started = time.perf_counter()
            with torch.no_grad():
                prediction = parallel(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - frame_started) * 1000
            if len(prediction) != 1 or str(prediction[0]['token']) != metadata['token']:
                raise ValueError('Full train sender prediction token differs')
            path = cache_root / (metadata['token'] + '.pkl')
            current = runtime.fields(path)
            count = validate(current)
            scene_rows.append(dict(metadata, query_count=count, file=metadata['token'] + '.pkl', inference_ms=inference_ms))
            last_scene_frame = position + 1 == len(indices) or rows[indices[position + 1]]['scene'] != metadata['scene']
            if last_scene_frame:
                audited.extend(scene_rows)
                completed.append(metadata['scene'])
                scene_rows = []
                report.update(frames=len(audited), scenes=len(completed),
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(), rng_state=save_rng())
                runtime.write(manifest_path, report)
            if position % 25 == 0 or last_scene_frame:
                print('FULL_TRAIN_CACHE frame=' + str(position + 1 + (len(rows) - len(indices))) + '/7445 scene=' + metadata['scene'], flush=True)
        cooperative = runtime.load_infos(root, 'coop', 'train')
        tokens = {row['token'] for row in audited}
        if len(audited) != 7445 or len(completed) != 46 or len(cooperative) != 7362 or any(str(info['token_inf']) not in tokens for info in cooperative):
            raise ValueError('Full 46-scene train cache coverage differs from the official split')
        if len({str(info['token_inf']) for info in cooperative}) != 7362:
            raise ValueError('Repeated infrastructure references require one-packet-per-sender-frame routing')
        report.update(status='FULL_TRAIN_CACHE_PASS', cooperative_samples=len(cooperative),
                      mean_infrastructure_ms=float(runtime.np.mean([row['inference_ms'] for row in audited])),
                      p95_infrastructure_ms=float(runtime.np.percentile([row['inference_ms'] for row in audited], 95)),
                      runtime_seconds=time.perf_counter() - started, completed_utc=now())
        runtime.write(manifest_path, report)
        runtime.write(root / 'experiments/main/metrics/train_cache.json', report)
        print('FULL_TRAIN_CACHE_PASS frames=7445 cooperative_samples=7362 scenes=46', flush=True)
        return report
    except Exception:
        report.update(status='FAILED', traceback=traceback.format_exc(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        runtime.write(manifest_path, report)
        raise


def distribution(values):
    np = runtime.np
    return {'count': len(values), 'mean': float(np.mean(values)), 'p50': float(np.percentile(values, 50)),
            'p95': float(np.percentile(values, 95)), 'min': float(np.min(values)), 'max': float(np.max(values))}


def collect_train_stats(root):
    """Lock four budgets and delta thresholds from complete official train only."""
    torch, np = runtime.torch, runtime.np
    cache_manifest = runtime.read(root / 'query_cache/train/manifest.json')
    if cache_manifest['status'] != 'FULL_TRAIN_CACHE_PASS' or cache_manifest['frames'] != 7445 or cache_manifest['scenes'] != 46:
        raise ValueError('A complete native full train cache is required before locking budgets')
    target = root / 'configs/main_experiment/budgets.json'
    if target.is_file():
        locked = runtime.read(target)
        if locked.get('source_cache_manifest_sha256') != sha(root / 'query_cache/train/manifest.json'):
            raise ValueError('Locked budgets refer to a different full train cache')
        print('FULL_TRAIN_BUDGETS_LOCKED reused=True budgets=' + str(locked['budgets_bytes']), flush=True)
        return locked
    baseline, mandatory, temporal_full, query_counts, delta = [], [], [], [], []
    mirror = TemporalCache()
    for position, metadata in enumerate(runtime.native_manifest(root, 'train')):
        current = runtime.fields(root / 'query_cache/train' / (metadata['token'] + '.pkl'))
        query_counts.append(validate(current))
        reference_frame = {key: metadata[key] for key in ('token', 'scene', 'timestamp_us')}
        baseline.append(len(baseline_int8_pack(current, reference_frame)))
        fixed, _, full = message_sizes(current, metadata)
        mandatory.append(fixed)
        temporal_full.append(full)
        rounded = int8_roundtrip(current)
        context = mirror.prepare(rounded, metadata)
        matched = ~context['force']
        if bool(matched.any()):
            old = torch.cat([context['last'][key] for key in FEATURES], dim=-1)
            new = torch.cat([rounded[key] for key in FEATURES], dim=-1)
            change = (new - old).norm(dim=-1) / old.norm(dim=-1).clamp_min(1e-8)
            delta.extend(change[matched].tolist())
        mirror.commit(current, torch.ones(len(rounded['obj_idxes']), dtype=torch.bool), metadata)
        if position % 500 == 0:
            print('FULL_TRAIN_STATS frame=' + str(position + 1) + '/7445', flush=True)
    if not delta:
        raise ValueError('Full train has no valid same-predicted-ID latent deltas')
    baseline_stats = distribution(baseline)
    mandatory_stats = distribution(mandatory)
    low = max(mandatory_stats['p95'], mandatory_stats['mean'] + 0.15 * (baseline_stats['mean'] - mandatory_stats['mean']))
    high = max(low + 3, baseline_stats['mean'] * 0.95)
    budgets = [int(round(value)) for value in np.linspace(low, high, 4)]
    quantiles = [90, 75, 50, 25]
    thresholds = [float(np.percentile(delta, percentile)) for percentile in quantiles]
    locked = {'status': 'LOCKED_FROM_FULL_TRAIN', 'budgets_bytes': budgets,
              'full_int8_mean_bytes': baseline_stats['mean'], 'full_int8': baseline_stats,
              'mandatory_state': mandatory_stats, 'temporal_full_int8': distribution(temporal_full),
              'query_count': distribution(query_counts), 'frames': 7445, 'cooperative_samples': 7362, 'scenes': 46,
              'delta_thresholds': thresholds, 'delta_percentiles': quantiles,
              'delta_definition': 'L2 of concatenated three current INT8-dequantized latents minus last_received, divided by last_received L2; train full INT8 updates each native sender frame',
              'delta_distribution': distribution(delta), 'max_staleness_frames': 4,
              'source_cache_manifest_sha256': sha(root / 'query_cache/train/manifest.json'),
              'created_utc': now(), 'val_used': False,
              'budget_rule': 'B1=max(P95 mandatory,mean mandatory+15% latent gap); B4=95% mean inherited full INT8; linear intermediate points',
              'rate_scope': 'len(serialized_message); native mandatory predicted IDs/flags/geometry/header plus all per-query scales',
              'baseline_scope': 'exact inherited CTKMSG01 FEAT_INT8 serializer; temporal protocol additionally transmits predicted IDs and UPDATE flags',
              'overflow_rule': 'new track and age>=4 force UPDATE; actual serialized overflow is reported'}
    runtime.write(target, locked)
    runtime.write(root / 'experiments/main/metrics/train_communication.json', locked)
    print('FULL_TRAIN_BUDGETS_LOCKED budgets=' + str(budgets) + ' delta=' + str(thresholds), flush=True)
    return locked


def car_only_evaluation(dataset, predictions, output):
    """Reuse the paper-aligned native class/config semantics; score only car."""
    from nuscenes.eval.common.config import config_factory
    from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval import NuScenesEval_custom, TrackingEval_custom
    tracking_file, _ = dataset.format_results(predictions, jsonfile_prefix=str(output))
    detection_file, _ = dataset.format_results_det(predictions, jsonfile_prefix=str(output))
    det_config = copy.copy(dataset.eval_detection_configs)
    det_config.class_names = ['car']
    det_config.class_range = dataset.class_range
    track_config = config_factory('tracking_nips_2019')
    track_config.class_names = track_config.tracking_names = ['car']
    track_config.class_range = dataset.class_range
    splits = dataset.create_splits_spd()
    runtime.write(output / 'evaluation_config.json', {'det_class_names': ['car'], 'tracking_names': ['car'],
                  'dataset_classes_unchanged': list(dataset.CLASSES), 'class_range_unchanged': dict(dataset.class_range),
                  'official_split': dataset.split_datas_file, 'samples': 3316, 'scenes': 21})
    detection = NuScenesEval_custom(dataset.nusc, config=det_config, result_path=detection_file,
                                    eval_set='val', output_dir=str(output / 'det'), verbose=True,
                                    overlap_test=dataset.overlap_test, data_infos=dataset.data_infos,
                                    splits=splits, category_to_type_name=dataset.category_to_detection_name)
    detection.main(plot_examples=0, render_curves=False)
    del detection
    tracking = TrackingEval_custom(config=track_config, result_path=tracking_file, eval_set='val',
                                  output_dir=str(output / 'track'), verbose=True, nusc_version=dataset.version,
                                  nusc_dataroot=dataset.data_root, splits=splits,
                                  category_to_type_name=dataset.category_to_tracking_name)
    tracking.main(render_curves=False)
    det = runtime.read(output / 'det/metrics_summary.json')
    track = runtime.read(output / 'track/metrics_summary.json')
    return {'mAP': det['mean_ap'], 'AMOTA': track['amota'], 'AMOTP': track['amotp'], 'IDS': track['ids'], 'MOTA': track['mota']}


def evaluate(root, method, budget_name=None, checkpoint=None, resume=False):
    torch, np, mmcv = runtime.torch, runtime.np, runtime.mmcv
    from mmdet.datasets.builder import PIPELINES
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from mmcv.parallel import MMDataParallel
    learned = method in ('aptu', 'mlp', 'no_assoc')
    if method not in ('periodic2', 'periodic4') and budget_name is None:
        raise ValueError('Learned and Delta cases require one of the four train-locked budgets')
    budget_cfg = runtime.read(root / 'configs/main_experiment/budgets.json')
    budget_index = int(budget_name[1:]) - 1 if budget_name else None
    budget = budget_cfg['budgets_bytes'][budget_index] if budget_index is not None else None
    label = method + ('_' + budget_name if budget_name else '')
    metrics_path = root / 'experiments/main/metrics' / (label + '.json')
    output = root / 'experiments/main/predictions' / label
    output.mkdir(parents=True, exist_ok=True)
    communication_path = root / 'experiments/main/communication' / (label + '.json')
    if learned:
        checkpoint = Path(checkpoint or root / 'experiments/main/checkpoints' / method / 'latest.pth')
    identity = {'commit': runtime.COMMIT, 'method': method, 'budget': budget_name,
                'budgets_sha256': sha(root / 'configs/main_experiment/budgets.json'),
                'scheduler_checkpoint_sha256': sha(checkpoint) if learned else None,
                'cooperative_checkpoint_md5': runtime.CHECKPOINTS['coop'][1], 'seed': 0,
                'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32}
    if resume and metrics_path.is_file():
        old = runtime.read(metrics_path)
        if (old.get('status') == 'PASS' and old.get('identity') == identity
                and Path(old['predictions_file']).is_file() and Path(old['communication_file']).is_file()
                and sha(old['predictions_file']) == old['predictions_sha256']):
            print('FULL_VAL_PASS reused=True method=' + label, flush=True)
            return old
    scheduler = None
    if learned:
        from main_scheduler import Scheduler
        state = torch.load(str(checkpoint), map_location='cpu')
        if state['variant'] != method:
            raise ValueError('Scheduler checkpoint variant differs from requested method')
        if state.get('identity', {}).get('budgets.json') != identity['budgets_sha256']:
            raise ValueError('Scheduler checkpoint differs from the train-locked budgets')
        # Loading the CPU scheduler must not advance native track-selection RNG.
        with torch.random.fork_rng(devices=[]):
            scheduler = Scheduler(method).float().eval()
            scheduler.load_state_dict(state['scheduler'], strict=True)
    manifest = {row['token']: row for row in runtime.native_manifest(root, 'val')}
    mirror, receiver = TemporalCache(), ReceiverCache()
    context = {'last': None}

    @PIPELINES.register_module(force=True)
    class MainTemporalReceiver:
        def __init__(self, keys, load_file_root):
            if tuple(keys) != FIELDS:
                raise ValueError('Native receiver field contract differs')
            self.cache_root = Path(load_file_root)

        def __call__(self, results):
            token = str(results['sample_idx_inf'])
            metadata = manifest[token]
            if str(results['scene_token']) != metadata['scene']:
                raise ValueError('Vehicle and infrastructure scene identities differ')
            current = runtime.fields(self.cache_root / (token + '.pkl'))
            rounded = int8_roundtrip(current)
            temporal = mirror.prepare(rounded, metadata)
            mandatory, per_update, full = message_sizes(current, metadata)
            start = time.perf_counter()
            if method.startswith('periodic'):
                period = 2 if method == 'periodic2' else 4
                flags = temporal['force'] | (temporal['elapsed'] >= period)
            elif method == 'delta':
                old = torch.cat([temporal['last'][key] for key in FEATURES], dim=-1)
                new = torch.cat([rounded[key] for key in FEATURES], dim=-1)
                delta = (new - old).norm(dim=-1) / old.norm(dim=-1).clamp_min(1e-8)
                flags = temporal['force'] | (delta >= budget_cfg['delta_thresholds'][budget_index])
            else:
                with torch.no_grad():
                    logits = scheduler(temporal['tokens'], budget / budget_cfg['full_int8_mean_bytes'])
                flags = select_updates(logits, temporal['force'], mandatory, per_update, budget)
            scheduler_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            packet = pack(current, metadata, flags)
            pack_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            decoded, header = receiver.unpack(packet, metadata)
            unpack_ms = (time.perf_counter() - start) * 1000
            mirror.commit(current, flags, metadata)
            for position, track_id in enumerate(current['obj_idxes'].tolist()):
                if any(not torch.equal(decoded[key][position], mirror.cache[track_id][key]) for key in FEATURES):
                    raise ValueError('Sender mirror differs from actual received latent')
            context['last'] = {'token_inf': token, 'bytes': len(packet), 'mandatory_bytes': mandatory,
                               'full_temporal_bytes': full, 'updates': header['updates'], 'queries': header['queries'],
                               'forced_updates': int(temporal['force'].sum()), 'scheduler_ms': scheduler_ms,
                               'pack_ms': pack_ms, 'unpack_ms': unpack_ms,
                               'budget_overflow_bytes': max(0, len(packet) - budget) if learned else None, 'decoded': decoded}
            results.update(decoded)
            return results

    cfg = runtime.make_config(root, 'coop', 'val')

    def replace(pipeline):
        count = 0
        for transform in pipeline:
            if transform['type'] == 'LoadInfInformation':
                transform['type'] = 'MainTemporalReceiver'
                count += 1
            if 'transforms' in transform:
                count += replace(transform['transforms'])
        return count

    if replace(cfg.data.test.pipeline) != 1:
        raise ValueError('Exactly one native receiver loader must be replaced')
    dataset = runtime.dataset_for(cfg)
    if len(dataset) != 3316 or len({info['scene_token'] for info in dataset.data_infos}) != 21:
        raise ValueError('Formal validation requires all official 3316 samples and 21 scenes')
    for info in dataset.data_infos:
        metadata = manifest[str(info['token_inf'])]
        if str(info['scene_token']) != metadata['scene'] or float(info['timestamp_inf']) != metadata['timestamp_us']:
            raise ValueError('Official cooperative-infrastructure pairing differs')
    model = runtime.build_model(root, cfg, 'coop').cuda()
    model.CLASSES = dataset.CLASSES
    parallel = MMDataParallel(model, device_ids=[0])
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False, nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    report = {'status': 'RUNNING', 'identity': identity, 'method': method, 'budget': budget_name,
              'budget_bytes': budget, 'samples': 3316, 'scenes': 21, 'started_utc': now(),
              'precision': 'FP32', 'category': 'car-only',
              'timing_scope': 'scheduler CPU forward and hard selection, or periodic/delta decision; CPU actual-message pack/unpack; synchronized vehicle cooperative forward; pipeline additionally includes images/cache/INT8 preprocessing/temporal features/message sizing/mirror checks but excludes sender image inference and evaluation',
              'sender_backbone_cached': True, 'scheduler_device': 'CPU',
              'delta_threshold': budget_cfg['delta_thresholds'][budget_index] if method == 'delta' else None}
    outputs, frame_records = [], []
    iterator = iter(loader)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        for position, info in enumerate(dataset.data_infos):
            cycle = time.perf_counter()
            data = next(iterator)
            record = context['last']
            if record['token_inf'] != str(info['token_inf']):
                raise ValueError('Actual full-val loader pairing differs')
            runtime.assert_loaded_queries(data, record['decoded'])
            torch.cuda.synchronize()
            inference_start = time.perf_counter()
            with torch.no_grad():
                prediction = parallel(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            cooperative_ms = (time.perf_counter() - inference_start) * 1000
            if len(prediction) != 1 or str(prediction[0]['token']) != str(info['token']):
                raise ValueError('Native full-val prediction token differs')
            outputs.extend(prediction)
            row = {key: value for key, value in record.items() if key != 'decoded'}
            row.update(token=str(info['token']), scene=str(info['scene_token']), cooperative_ms=cooperative_ms,
                       pipeline_ms=(time.perf_counter() - cycle) * 1000)
            frame_records.append(row)
            if position % 25 == 0 or position == 3315:
                print('FULL_VAL method=' + label + ' frame=' + str(position + 1) + '/3316 bytes=' + str(row['bytes']), flush=True)
        predictions_file = output / 'predictions.pkl'
        mmcv.dump(outputs, str(predictions_file))
        peak = torch.cuda.max_memory_allocated()
        metrics = car_only_evaluation(dataset, outputs, output)
        total_queries = sum(row['queries'] for row in frame_records)
        mean_bytes = float(np.mean([row['bytes'] for row in frame_records]))
        communication = {'mean_bytes_per_frame': mean_bytes,
                         'p95_bytes_per_frame': float(np.percentile([row['bytes'] for row in frame_records], 95)),
                         'bytes_per_second': mean_bytes * 10,
                         'update_ratio': sum(row['updates'] for row in frame_records) / max(total_queries, 1),
                         'forced_update_ratio': sum(row['forced_updates'] for row in frame_records) / max(total_queries, 1),
                         'budget_overflow_frames': sum(bool(row['budget_overflow_bytes']) for row in frame_records) if learned else None,
                         'maximum_budget_overflow_bytes': max(row['budget_overflow_bytes'] for row in frame_records) if learned else None}
        latency = {key: float(np.mean([row[key] for row in frame_records]))
                   for key in ('scheduler_ms', 'pack_ms', 'unpack_ms', 'cooperative_ms', 'pipeline_ms')}
        runtime.write(communication_path, {'method': method, 'budget': budget_name,
                      'summary': communication, 'latency': latency, 'frames': frame_records})
        report.update(status='PASS', metrics=metrics, communication=communication, latency=latency,
                      peak_allocated_bytes=peak, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      predictions_file=str(predictions_file), predictions_sha256=sha(predictions_file),
                      communication_file=str(communication_path), runtime_seconds=time.perf_counter() - started,
                      completed_utc=now())
        runtime.write(metrics_path, report)
        print('FULL_VAL_PASS method=' + label + ' samples=3316 scenes=21 metrics=' + str(metrics), flush=True)
        return report
    except Exception:
        report.update(status='FAILED', completed_frames=len(outputs), traceback=traceback.format_exc(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        runtime.write(metrics_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', '--audit-root', type=Path, default=Path('/mnt/d/CoopTrack_Audit'))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preflight', action='store_true')
    action.add_argument('--generate-train-cache', action='store_true')
    action.add_argument('--collect-train-stats', action='store_true')
    action.add_argument('--method', choices=('periodic2', 'periodic4', 'delta', 'aptu', 'mlp', 'no_assoc'))
    parser.add_argument('--budget', choices=('B1', 'B2', 'B3', 'B4'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    runtime.initialize(root)
    if args.preflight:
        preflight(root)
    elif args.generate_train_cache:
        cache_train(root, args.resume)
    elif args.collect_train_stats:
        collect_train_stats(root)
    else:
        evaluate(root, args.method, args.budget, args.checkpoint, args.resume)


if __name__ == '__main__':
    main()
