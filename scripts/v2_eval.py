"""V2 full-scene evaluation with actual ordered INT8 packets and native metrics."""
import copy
import csv
import gc
import json
import gzip
import hashlib
from pathlib import Path
import shutil
import time
import traceback

import main_runtime as runtime
from main_eval import now, sha
from main_message import FIELDS, FEATURES, ReceiverCache, int8_roundtrip, message_sizes, pack


def source_identity(root, learned=False):
    names = ['main_runtime.py', 'main_message.py', 'v2_runtime.py', 'v2_message.py', 'v2_eval.py']
    if learned:
        names.append('v2_scheduler.py')
    return {name: sha(root / 'scripts' / name) for name in names}


def split_infos(root, split):
    if split != 'val':
        raise ValueError('Formal V2 evaluation accepts only official val')
    infos = runtime.load_infos(root, 'coop', 'val')
    official = runtime.read(root / 'code/CoopTrack/data/split_datas/cooperative-split-data-spd.json')['batch_split']['val']
    if len(infos) != 3316 or {str(row['scene_token']) for row in infos} != set(official):
        raise ValueError('Official val must contain exactly 21 scenes / 3316 samples')
    return infos


def car_evaluation(dataset, predictions, output, scenes):
    """Run the inherited car-only evaluator on the complete official val split."""
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
    splits['val'] = list(scenes)
    runtime.write(output / 'evaluation_config.json', {
        'det_class_names': ['car'], 'tracking_names': ['car'],
        'class_range': dict(dataset.class_range), 'samples': len(dataset), 'scenes': list(scenes),
        'GT_filter': 'complete official val scene names'})
    detection = NuScenesEval_custom(dataset.nusc, config=det_config, result_path=detection_file,
        eval_set='val', output_dir=str(output / 'det'), verbose=True, overlap_test=dataset.overlap_test,
        data_infos=dataset.data_infos, splits=splits, category_to_type_name=dataset.category_to_detection_name)
    detection.main(plot_examples=0, render_curves=False)
    del detection
    tracking = TrackingEval_custom(config=track_config, result_path=tracking_file, eval_set='val',
        output_dir=str(output / 'track'), verbose=True, nusc_version=dataset.version,
        nusc_dataroot=dataset.data_root, splits=splits, category_to_type_name=dataset.category_to_tracking_name)
    tracking.main(render_curves=False)
    det = runtime.read(output / 'det/metrics_summary.json')
    track = runtime.read(output / 'track/metrics_summary.json')
    return {'mAP': det['mean_ap'], 'AMOTA': track['amota'], 'AMOTP': track['amotp'],
            'MOTA': track['mota'], 'IDS': track['ids']}


def stats(values):
    np = runtime.np
    return {'mean': float(np.mean(values)), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'max': float(np.max(values))} if values else {
            'mean': None, 'median': None, 'p95': None, 'max': None}


def archive_predictions(output):
    """Keep complete V2 evaluator inputs losslessly compressed on the limited D: disk."""
    for name in ('predictions.pkl', 'results_nusc.json', 'results_nusc_det.json'):
        path = output / name
        if not path.is_file():
            continue
        target = path.with_suffix(path.suffix + '.gz')
        temp = target.with_suffix(target.suffix + '.writing')
        with path.open('rb') as source, gzip.open(str(temp), 'wb', compresslevel=1) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        temp.replace(target)
        path.unlink()


def evaluate(root, method, split='val', budget_name=None, checkpoint=None, resume=True, seed=0):
    from mmcv.parallel import MMDataParallel
    from mmdet.datasets.builder import PIPELINES
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from v2_message import TemporalCache, periodic_updates, select_updates as learned_updates
    from v2_runtime import make_config, seed as configure_seed, verify_final_protocol
    torch, np = runtime.torch, runtime.np
    root = Path(root)
    protocol = verify_final_protocol(root)
    if split != 'val' or seed != 0:
        raise ValueError('Formal V2 is official val with primary seed 0 only')
    learned = method in ('mlp', 'transformer')
    if not learned and method not in protocol['periodic_points']:
        raise ValueError('Only the eight pre-registered new periodic points may run; P1/P2/P4 are reused')
    period = int(method[1:]) if method.startswith('P') else None
    if not learned and period is None and method != 'INIT_ONLY':
        raise ValueError('Unknown V2 method: ' + method)
    budget_cfg = runtime.read(root / 'configs/main_experiment/budgets.json')
    if budget_name is not None and budget_name not in ('B1', 'B2', 'B3', 'B4'):
        raise ValueError('Unknown formal budget')
    budget = budget_cfg['budgets_bytes'][int(budget_name[1:]) - 1] if budget_name else None
    if learned and (checkpoint is None or budget is None):
        raise ValueError('Learned eval requires a frozen checkpoint and budget')
    label = method + ('_' + budget_name if budget_name else '') + '_seed' + str(seed)
    if learned and (Path(checkpoint).resolve() != (root / protocol['final_checkpoints'][method]['path']).resolve() or
                    sha(checkpoint) != protocol['final_checkpoints'][method]['sha256']):
        raise ValueError('Only the manifest-locked epoch-12 final checkpoint may be evaluated')
    run_dir = root / 'experiments/v2/predictions/official_val' / label
    metric_path = root / 'experiments/v2/metrics/official_val' / label / 'result.json'
    comm_path = root / 'experiments/v2/communication/official_val' / (label + '.json')
    infos = split_infos(root, split)
    scenes = sorted({str(info['scene_token']) for info in infos})
    split_identity = hashlib.sha256(str([str(x['token']) for x in infos]).encode()).hexdigest()
    identity = {'backbone_commit': runtime.COMMIT, 'sources': source_identity(root, learned),
        'method': method, 'split': split, 'split_identity': split_identity, 'budget_name': budget_name,
        'budget': budget, 'seed': seed, 'checkpoint_sha256': sha(checkpoint) if learned else None,
        'budgets_sha256': sha(root / 'configs/main_experiment/budgets.json'),
        'rate_normalizer': budget_cfg['full_int8_mean_bytes'],
        'cooperative_checkpoint_md5': runtime.CHECKPOINTS['coop'][1],
        'serializer': 'APTU0001', 'no_TTL': True, 'forced': 'no-cache only', 'threshold': .5,
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32}
    identity['protocol_manifest_sha256'] = sha(root / 'experiments/v2/metrics/FINAL_PROTOCOL_MANIFEST.json')
    if resume and metric_path.is_file():
        old = runtime.read(metric_path)
        if old.get('status') != 'PASS' or old.get('identity') != identity:
            raise ValueError('Existing V2 run is incomplete or incompatible: ' + str(metric_path))
        if not comm_path.is_file() or not Path(old['predictions_file']).is_file():
            raise ValueError('Completed V2 run is missing its real predictions/communication')
        print('V2_EVAL_REUSED ' + label + ' split=' + split, flush=True)
        return old
    run_dir.mkdir(parents=True, exist_ok=True)
    scheduler = None
    if learned:
        from v2_scheduler import Scheduler
        state = torch.load(str(checkpoint), map_location='cpu')
        if state['variant'] != method or state['epoch'] != 12 or state.get('phase') != 'train':
            raise ValueError('Checkpoint must be the formal architecture at epoch 12')
        scheduler = Scheduler(method).float().eval()
        scheduler.load_state_dict(state['scheduler'], strict=True)
    source_split = 'val'
    manifest = {row['token']: row for row in runtime.native_manifest(root, source_split)}
    mirror, receiver = TemporalCache(), ReceiverCache()
    context, staleness, intervals = {}, [], []
    probability_histogram = np.zeros(10, dtype=np.int64)
    bucket_ranges = (('0-1', 0, 2), ('2-3', 2, 4), ('4-7', 4, 8),
                     ('8-15', 8, 16), ('16-31', 16, 32), ('32+', 32, float('inf')))
    buckets = {name: {'count': 0, 'update_count': 0, 'probability_sum': 0.0}
               for name, _, _ in bucket_ranges}
    previous_identity = None
    resets = {'scene_reset_count': 0, 'native_session_reset_count': 0}

    @PIPELINES.register_module(force=True)
    class V2TemporalReceiver:
        def __init__(self, keys, load_file_root):
            if tuple(keys) != FIELDS:
                raise ValueError('Native receiver field contract differs')
            self.cache_root = Path(load_file_root)

        def __call__(self, results):
            nonlocal previous_identity
            token = str(results['sample_idx_inf'])
            metadata = manifest[token]
            if str(results['scene_token']) != metadata['scene']:
                raise ValueError('Cooperative-infrastructure scene mismatch')
            current = runtime.fields(self.cache_root / (token + '.pkl'))
            rounded = int8_roundtrip(current)
            temporal = mirror.prepare(rounded, metadata)
            mandatory, cost, full = message_sizes(current, metadata)
            started = time.perf_counter()
            if learned:
                with torch.no_grad():
                    probability = scheduler(temporal['tokens'], budget / budget_cfg['full_int8_mean_bytes']).sigmoid()
                flags = learned_updates(probability, temporal['force'])
                soft = mandatory + cost * float(torch.where(temporal['force'], torch.ones_like(probability), probability).sum())
            else:
                flags = periodic_updates(temporal, period, init_only=(method == 'INIT_ONLY'))
                probability, soft = None, None
            scheduler_ms = (time.perf_counter() - started) * 1000
            forced = temporal['force']
            elapsed_optional = temporal['elapsed'][~forced]
            flags_optional = flags[~forced]
            staleness.extend(elapsed_optional.tolist())
            if learned:
                probability_histogram[:] += np.histogram(probability[~forced].cpu().numpy(), bins=np.linspace(0, 1, 11))[0]
            for name, lower, upper in bucket_ranges:
                mask = (elapsed_optional >= lower) & (elapsed_optional < upper)
                item = buckets[name]
                item['count'] += int(mask.sum())
                item['update_count'] += int(flags_optional[mask].sum())
                if learned:
                    item['probability_sum'] += float(probability[~forced][mask].sum())
            ids = current['obj_idxes'].tolist()
            for index, track_id in enumerate(ids):
                if bool(flags[index]) and not bool(forced[index]):
                    intervals.append((metadata['timestamp_us'] - mirror.cache[track_id]['timestamp_us']) / 1e6)
            started = time.perf_counter()
            packet = pack(current, metadata, flags)
            decoded, header = receiver.unpack(packet, metadata)
            if len(packet) != mandatory + cost * int(flags.sum()):
                raise ValueError('Actual packet size is not M + 780*U')
            mirror.commit(current, flags, metadata, decoded)
            if any(not torch.equal(decoded[key][index], mirror.cache[track_id][key]) for index, track_id in enumerate(ids) for key in FEATURES):
                raise ValueError('Sender mirror differs from decoded receiver')
            pack_unpack_ms = (time.perf_counter() - started) * 1000
            identity_now = (metadata['scene'], metadata['session'])
            if identity_now != previous_identity:
                if previous_identity is None or identity_now[0] != previous_identity[0]:
                    resets['scene_reset_count'] += 1
                else:
                    resets['native_session_reset_count'] += 1
                previous_identity = identity_now
            context['last'] = dict(metadata, bytes=len(packet), mandatory_bytes=mandatory,
                full_temporal_bytes=full, queries=header['queries'], updates=header['updates'],
                new_init_count=int(forced.sum()), refresh_count=int((flags & ~forced).sum()),
                optional_queries=int((~forced).sum()), S_soft=soft,
                soft_hard_gap=len(packet) - soft if soft is not None else None,
                overflow_bytes=max(0, len(packet) - budget) if budget is not None else None,
                unavoidable_init_overflow=max(0, mandatory + cost * int(forced.sum()) - budget) if budget is not None else None,
                scheduler_ms=scheduler_ms, pack_unpack_ms=pack_unpack_ms, decoded=decoded)
            results.update(decoded)
            return results

    cfg = make_config(root, 'coop', source_split, training=False)

    def replace(pipeline):
        count = 0
        for transform in pipeline:
            if transform['type'] == 'LoadInfInformation':
                transform['type'] = 'V2TemporalReceiver'
                count += 1
            if 'transforms' in transform:
                count += replace(transform['transforms'])
        return count

    if replace(cfg.data.test.pipeline) != 1:
        raise ValueError('Expected exactly one native receiver transform')
    dataset = runtime.dataset_for(cfg)
    expected_tokens = {str(info['token']) for info in infos}
    dataset.data_infos = [info for info in dataset.data_infos if str(info['token']) in expected_tokens]
    if len(dataset) != len(infos) or {str(info['token']) for info in dataset.data_infos} != expected_tokens:
        raise ValueError('V2 full split sample coverage mismatch')
    model = runtime.build_model(root, cfg, 'coop').cuda()
    model.CLASSES = dataset.CLASSES
    parallel = MMDataParallel(model, device_ids=[0])
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False,
        shuffle=False, nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    report = {'status': 'RUNNING', 'identity': identity, 'method': method, 'split': split,
        'sample_count': len(infos), 'scene_count': len(scenes), 'seed': seed, 'started_utc': now(),
        'timing_scope': 'cached infrastructure queries, receiver preprocessing and synchronized vehicle forward; excludes sender image inference'}
    outputs, records = [], []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        # Construction/loading and loader creation happen before resetting inference RNG.
        iterator = iter(loader)
        configure_seed(seed)
        for position, info in enumerate(dataset.data_infos):
            cycle = time.perf_counter()
            data = next(iterator)
            received = context['last']
            if received['token'] != str(info['token_inf']):
                raise ValueError('Data loader substituted/reordered sender sample')
            runtime.assert_loaded_queries(data, received['decoded'])
            torch.cuda.synchronize()
            inference_start = time.perf_counter()
            with torch.no_grad():
                predictions = parallel(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - inference_start) * 1000
            if len(predictions) != 1 or str(predictions[0]['token']) != str(info['token']):
                raise ValueError('Native prediction token mismatch')
            outputs.extend(predictions)
            row = {key: value for key, value in received.items() if key != 'decoded'}
            row.update(cooperative_token=str(info['token']), cooperative_ms=inference_ms,
                pipeline_ms=(time.perf_counter() - cycle) * 1000)
            records.append(row)
            if position % 25 == 0 or position + 1 == len(infos):
                print('V2_EVAL split=%s method=%s frame=%d/%d bytes=%d' % (split, label, position + 1, len(infos), row['bytes']), flush=True)
        temp = run_dir / 'predictions.writing.pkl'
        runtime.mmcv.dump(outputs, str(temp))
        temp.replace(run_dir / 'predictions.pkl')
        metrics = car_evaluation(dataset, outputs, run_dir, scenes)
        archive_predictions(run_dir)
        query_count = sum(row['queries'] for row in records)
        update_count = sum(row['updates'] for row in records)
        optional_count = sum(row['optional_queries'] for row in records)
        mean_bytes = float(np.mean([row['bytes'] for row in records]))
        communication = {'mean_bytes_per_frame': mean_bytes,
            'p95_bytes_per_frame': float(np.percentile([row['bytes'] for row in records], 95)),
            'bytes_per_second_at_10Hz': mean_bytes * 10, 'query_count': query_count, 'update_count': update_count,
            'query_weighted_update_ratio': update_count / query_count if query_count else None,
            'hard_UPDATE_ratio': update_count / query_count if query_count else None,
            'new_init_count': sum(row['new_init_count'] for row in records),
            'forced_init_ratio': sum(row['new_init_count'] for row in records) / query_count if query_count else None,
            'optional_query_count': optional_count,
            'periodic_refresh_count': sum(row['refresh_count'] for row in records) if not learned else None,
            'optional_learned_update_count': sum(row['refresh_count'] for row in records) if learned else None,
            'optional_update_ratio': sum(row['refresh_count'] for row in records) / optional_count if optional_count else None,
            'mean_mandatory_bytes': float(np.mean([row['mandatory_bytes'] for row in records])),
            'mean_latent_update_bytes': update_count * 780 / len(infos),
            'staleness': stats(staleness), 'inter_update_seconds': stats(intervals), **resets}
        communication['staleness_scope'] = 'optional queries before decision; no-cache initialization excluded'
        communication['staleness_before_decision_buckets'] = {
            name: {'count': value['count'], 'update_count': value['update_count'],
                   'optional_update_ratio': value['update_count'] / value['count'] if value['count'] else None,
                   'mean_probability': value['probability_sum'] / value['count'] if learned and value['count'] else None}
            for name, value in buckets.items()}
        if learned:
            communication['optional_probability_histogram'] = {'edges': np.linspace(0, 1, 11).tolist(),
                'counts': probability_histogram.tolist(), 'scope': 'optional queries; forced initialization excluded'}
            if int(probability_histogram.sum()) != optional_count:
                raise ValueError('Optional probability histogram coverage mismatch')
            communication.update(mean_S_soft=float(np.mean([row['S_soft'] for row in records])),
                mean_S_hard=mean_bytes, mean_soft_hard_gap=float(np.mean([row['soft_hard_gap'] for row in records])),
                overflow_frame_count=sum(row['overflow_bytes'] > 0 for row in records),
                max_frame_overflow=max(row['overflow_bytes'] for row in records),
                mean_unavoidable_init_overflow=float(np.mean([row['unavoidable_init_overflow'] for row in records])))
        runtime.write(comm_path, {'identity': identity, 'summary': communication, 'frames': records})
        metrics['mean_bytes'] = mean_bytes
        report.update(status='PASS', metrics=metrics, communication=communication, mean_bytes=mean_bytes, period_frames=period,
            nominal_period_seconds=period / 10 if period is not None else None,
            runtime_seconds=time.perf_counter() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), predictions_file=str(run_dir / 'predictions.pkl.gz'),
            communication_file=str(comm_path), completed_utc=now())
        runtime.write(metric_path, report)
        print('V2_EVAL_PASS %s split=%s metrics=%s' % (label, split, metrics), flush=True)
        return report
    except Exception:
        report.update(status='FAILED', completed_frames=len(outputs), traceback=traceback.format_exc())
        runtime.write(metric_path.with_name('failure.json'), report)
        raise
    finally:
        context.clear()
        del parallel, model, dataset, loader, outputs
        gc.collect()
        torch.cuda.empty_cache()


def write_table(path, results):
    rows = []
    for result in results:
        row = {key: result.get(key) for key in ('method', 'period_frames', 'nominal_period_seconds', 'sample_count', 'scene_count', 'status', 'seed', 'mean_bytes')}
        row['budget_name'] = result.get('identity', {}).get('budget_name')
        row['reused'] = result.get('reused', False)
        row.update(result['metrics'])
        row.update({key: value for key, value in result.get('communication', {}).items() if not isinstance(value, dict)})
        for name in ('staleness', 'inter_update_seconds'):
            row.update({name + '_' + k: v for k, v in result.get('communication', {}).get(name, {}).items()})
        for name in ('optional_probability_histogram', 'staleness_before_decision_buckets'):
            row[name] = json.dumps(result.get('communication', {}).get(name), sort_keys=True)
        rows.append(row)
    if rows:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.writing')
        with temp.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)

