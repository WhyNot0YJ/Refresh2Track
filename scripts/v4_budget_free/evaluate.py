"""Isolated complete official-val V4/P16 inference and payload accounting."""
import argparse
import copy
import gc
import gzip
import hashlib
import inspect
import json
import os
import pickle
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import runtime
from runtime import native, read, write, digest, experiment
from scheduler import Scheduler
from v2_message import (FIELDS, FEATURES, TemporalCache, ReceiverCache,
                        int8_roundtrip, message_sizes, pack, select_updates, periodic_updates)

SEEDS = (0, 2025, 2026)
RNG_RULE = 'after scheduler/model construction, checkpoint load and iter(loader); before first next(iterator); no scene reseeding'


def utc():
    return datetime.now(timezone.utc).isoformat()


def statistics(values):
    np = native.np
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(np.max(values))) if values else dict.fromkeys(('mean', 'median', 'p95', 'max'))


def rng_digest():
    torch = native.torch
    return {'python': hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest(),
            'numpy': hashlib.sha256(pickle.dumps(native.np.random.get_state(), protocol=4)).hexdigest(),
            'torch_cpu': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            'torch_cuda': [hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
                           for value in torch.cuda.get_rng_state_all()]}


def environment():
    torch = native.torch
    return {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0), 'precision': 'FP32', 'batch': 1, 'workers': 0,
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'threads': torch.get_num_threads(), 'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED')}


def communication_summary(records):
    count = len(records)
    if not count:
        raise ValueError('No actual frames for communication statistics')
    totals = {key: sum(row[key] for row in records)
              for key in ('queries', 'updates', 'forced', 'optional_queries', 'optional_updates')}
    message = statistics([row['bytes'] for row in records])
    payload = statistics([row['feature_payload_bytes'] for row in records])
    ages = statistics([age for row in records for age in row['cache_ages']])
    estimated = sum(row['estimated_optional_bytes'] for row in records)
    actual = sum(row['actual_optional_bytes'] for row in records)
    probability_count = sum(row['probability_count'] for row in records)
    near = sum(row['probability_near_threshold_count'] for row in records)
    return dict(**totals, mean_bytes=message['mean'], p95_bytes=message['p95'],
        bytes_per_second=message['mean'] * 10,
        serialized_message_mean_B_per_frame=message['mean'],
        serialized_message_P95_B_per_frame=message['p95'],
        serialized_message_BPS=message['mean'] * 10, serialized_BPS=message['mean'] * 10,
        feature_payload_mean_B_per_frame=payload['mean'], feature_payload_P95_B_per_frame=payload['p95'],
        feature_payload_BPS=payload['mean'] * 10,
        feature_payload_latent_BPS=780 * totals['updates'] / count * 10,
        geometry_BPS=52 * totals['queries'] / count * 10,
        optional_update_ratio=totals['optional_updates'] / totals['optional_queries'] if totals['optional_queries'] else None,
        forced_init_ratio=totals['forced'] / totals['queries'] if totals['queries'] else None,
        total_update_ratio=totals['updates'] / totals['queries'] if totals['queries'] else None,
        estimated_optional_bytes=estimated, actual_optional_bytes=actual,
        estimated_optional_mean_B_per_frame=estimated / count,
        actual_optional_mean_B_per_frame=actual / count,
        soft_hard_optional_gap_bytes=estimated - actual,
        soft_hard_optional_gap_mean_B_per_frame=(estimated - actual) / count,
        soft_hard_optional_gap_absolute_mean_B_per_frame=sum(abs(row['estimated_optional_bytes'] - row['actual_optional_bytes']) for row in records) / count,
        probability_near_threshold_fraction=near / probability_count if probability_count else None,
        probability_histogram=[sum(row['probability_histogram'][i] for row in records) for i in range(10)],
        probability_histogram_bins=[i / 10 for i in range(11)],
        probability_count=probability_count, probability_scope='optional queries only',
        cache_age_mean=ages['mean'], cache_age_p95=ages['p95'], cache_age_max=ages['max'],
        cache_age_scope='optional queries before decision, infrastructure frames',
        BPS_scope='nominal 10Hz mean bytes/frame times 10; not measured network throughput')


def bind_and_evaluate(dataset, outputs, directory, scenes, used_checkpoint, selection):
    """Inherited native evaluators with complete input hashes saved before evaluation."""
    from nuscenes.eval.common.config import config_factory
    from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval import NuScenesEval_custom, TrackingEval_custom
    raw = directory / 'predictions.pkl'
    native.mmcv.dump(outputs, str(raw))
    tracking_file, _ = dataset.format_results(outputs, jsonfile_prefix=str(directory))
    detection_file, _ = dataset.format_results_det(outputs, jsonfile_prefix=str(directory))
    inputs = {'checkpoint': {'path': str(used_checkpoint), 'sha256': digest(used_checkpoint)},
              'predictions': {'path': str(raw), 'sha256': digest(raw)},
              'tracking': {'path': str(tracking_file), 'sha256': digest(tracking_file)},
              'detection': {'path': str(detection_file), 'sha256': digest(detection_file)}}
    write(directory / 'evaluator_inputs.json', inputs)
    write(directory / 'prediction_sha256_before_evaluation.json',
          dict(inputs['predictions'], bytes=raw.stat().st_size, recorded_utc=utc()))
    det_config = copy.copy(dataset.eval_detection_configs)
    det_config.class_names, det_config.class_range = ['car'], dataset.class_range
    track_config = config_factory('tracking_nips_2019')
    track_config.class_names = track_config.tracking_names = ['car']
    track_config.class_range = dataset.class_range
    splits = dataset.create_splits_spd()
    splits['val'] = list(scenes)
    write(directory / 'evaluation_config.json', {'samples': len(dataset), 'scenes': scenes,
          'classes': ['car'], 'frequency_hz': 10, 'checkpoint_selection': selection})
    detection = NuScenesEval_custom(dataset.nusc, config=det_config, result_path=detection_file,
        eval_set='val', output_dir=str(directory / 'det'), verbose=True, overlap_test=dataset.overlap_test,
        data_infos=dataset.data_infos, splits=splits, category_to_type_name=dataset.category_to_detection_name)
    detection.main(plot_examples=0, render_curves=False)
    del detection
    tracking = TrackingEval_custom(config=track_config, result_path=tracking_file, eval_set='val',
        output_dir=str(directory / 'track'), verbose=True, nusc_version=dataset.version,
        nusc_dataroot=dataset.data_root, splits=splits, category_to_type_name=dataset.category_to_tracking_name)
    tracking.main(render_curves=False)
    det, track = read(directory / 'det/metrics_summary.json'), read(directory / 'track/metrics_summary.json')
    metrics = {'mAP': det['mean_ap'], 'AMOTA': track['amota'], 'AMOTP': track['amotp'],
        'MOTA': track['mota'], 'IDS': track['ids'], 'FP': track['fp'], 'FN': track['fn'], 'recall': track['recall']}
    if not all(native.np.isfinite(value) for value in metrics.values()):
        raise ValueError('Non-finite official evaluator metrics')
    for name in ('predictions', 'tracking', 'detection'):
        path = Path(inputs[name]['path'])
        if digest(path) != inputs[name]['sha256']:
            raise ValueError('Evaluator changed bound input: ' + name)
        target = path.with_suffix(path.suffix + '.gz')
        temporary = target.with_suffix(target.suffix + '.writing')
        with path.open('rb') as source, gzip.open(str(temporary), 'wb', compresslevel=1) as destination:
            shutil.copyfileobj(source, destination, 1024 * 1024)
        check = hashlib.sha256()
        with gzip.open(str(temporary), 'rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                check.update(block)
        if check.hexdigest() != inputs[name]['sha256']:
            raise ValueError('Lossless input archive verification failed')
        temporary.replace(target)
        inputs[name].update(archive=str(target), archive_sha256=digest(target))
    write(directory / 'evaluator_inputs.json', inputs)
    return metrics, inputs


def evaluate(root, method, checkpoint, epoch=None, inference_seed=0, selection=False):
    root, checkpoint = Path(root).resolve(), Path(checkpoint).resolve()
    if method not in ('V4', 'P16') or inference_seed not in SEEDS:
        raise ValueError('Only registered V4/P16 inference seeds 0/2025/2026 may run')
    if selection and (method != 'V4' or inference_seed != 0 or epoch is None):
        raise ValueError('Selection requires V4 epoch and seed 0')
    if checkpoint.parent != experiment(root) / 'checkpoints':
        raise ValueError('Pair anchor must be an isolated V4 checkpoint')
    runtime.initialize(root)
    from mmcv.parallel import MMDataParallel
    from mmdet.datasets.builder import PIPELINES
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from v2_runtime import seed
    torch, np = native.torch, native.np
    protocol = runtime.identity(root)
    state = torch.load(str(checkpoint), map_location='cpu')
    if state.get('identity') != protocol or (epoch is not None and state.get('epoch') != epoch):
        raise ValueError('V4 pair anchor source/protocol/epoch differs')
    epoch = state['epoch']
    used_checkpoint = checkpoint if method == 'V4' else root / 'checkpoints/cooptrack_r50_coop.pth'
    directory = experiment(root) / 'validation'
    directory = directory / ('epoch%03d' % epoch) if selection else directory / 'final' / (method + '_seed' + str(inference_seed))
    identity = {'protocol': protocol, 'method': method, 'epoch': epoch,
                'inference_seed': inference_seed, 'selection': selection,
                'checkpoint_sha256': digest(checkpoint),
                'used_checkpoint_sha256': digest(used_checkpoint)}
    path = directory / 'result.json'
    if path.exists():
        result = read(path)
        if result.get('status') != 'PASS' or result['identity'] != identity:
            raise ValueError('Existing validation is incompatible')
        if result.get('sample_count') != 3316 or result.get('scene_count') != 21:
            raise ValueError('Existing validation coverage incomplete')
        for item in result['evaluator_inputs'].values():
            if digest(item['path']) != item['sha256'] or ('archive' in item and digest(item['archive']) != item['archive_sha256']):
                raise ValueError('Validation input identity changed')
        return result
    directory.mkdir(parents=True, exist_ok=True)
    infos = native.load_infos(root, 'coop', 'val')
    scenes = sorted({str(info['scene_token']) for info in infos})
    official = read(root / 'code/CoopTrack/data/split_datas/cooperative-split-data-spd.json')['batch_split']['val']
    if len(infos) != 3316 or scenes != sorted(official):
        raise ValueError('Validation requires complete official val 3316/21')
    gate = Scheduler().float().eval()
    gate.load_state_dict(state['scheduler'], strict=True)
    del state
    manifest = {row['token']: row for row in native.native_manifest(root, 'val')}
    mirror, receiver = TemporalCache(), ReceiverCache()
    context = {}

    @PIPELINES.register_module(force=True)
    class V4TemporalReceiver:
        def __init__(self, keys, load_file_root):
            if tuple(keys) != FIELDS:
                raise ValueError('Native receiver field contract differs')
            self.cache_root = Path(load_file_root)

        def __call__(self, results):
            token = str(results['sample_idx_inf'])
            meta = manifest[token]
            if str(results['scene_token']) != meta['scene']:
                raise ValueError('Sender/receiver scene mismatch')
            raw = native.fields(self.cache_root / (token + '.pkl'))
            current = int8_roundtrip(raw)
            temporal = mirror.prepare(current, meta)
            mandatory, cost, _ = message_sizes(raw, meta)
            if method == 'V4':
                with torch.no_grad():
                    probability = gate(temporal['tokens']).sigmoid()
                hard = select_updates(probability, temporal['force'])
            else:
                hard = periodic_updates(temporal, 16)
            packet = pack(raw, meta, hard)
            decoded, _ = receiver.unpack(packet, meta)
            if cost != 780 or len(packet) != mandatory + cost * int(hard.sum()):
                raise ValueError('Packet byte accounting differs from real serializer')
            for name in FEATURES:
                if not torch.equal(decoded[name][hard], current[name][hard]) or not torch.equal(decoded[name][~hard], temporal['last'][name][~hard]):
                    raise ValueError('Real UPDATE/REUSE receiver values differ')
            optional = ~temporal['force']
            probabilities = probability[optional] if method == 'V4' else hard[optional].float()
            ages = temporal['elapsed'][optional].tolist()
            mirror.commit(raw, hard, meta, decoded)
            optional_updates = int((hard & optional).sum())
            context['last'] = dict(meta, bytes=len(packet), mandatory_bytes=mandatory,
                queries=len(hard), updates=int(hard.sum()), forced=int(temporal['force'].sum()),
                optional_queries=int(optional.sum()), optional_updates=optional_updates, cache_ages=ages,
                feature_payload_bytes=52 * len(hard) + 780 * int(hard.sum()),
                feature_payload_latent_bytes=780 * int(hard.sum()), geometry_bytes=52 * len(hard),
                estimated_optional_bytes=780 * float(probabilities.sum()),
                actual_optional_bytes=780 * optional_updates,
                soft_hard_optional_gap_bytes=780 * float(probabilities.sum()) - 780 * optional_updates,
                probability_count=len(probabilities) if method == 'V4' else 0,
                probability_near_threshold_count=int(((probabilities >= .45) & (probabilities <= .55)).sum()) if method == 'V4' else 0,
                probability_histogram=np.histogram(probabilities.cpu().numpy(), bins=np.linspace(0, 1, 11))[0].tolist() if method == 'V4' else [0] * 10,
                decision_probabilities=probabilities.tolist() if method == 'V4' else None,
                decoded=decoded)
            results.update(decoded)
            return results

    cfg = runtime.make_config(root, 'coop', 'val')

    def replace(pipeline):
        count = 0
        for transform in pipeline:
            if transform['type'] == 'LoadInfInformation':
                transform['type'] = 'V4TemporalReceiver'
                count += 1
            if 'transforms' in transform:
                count += replace(transform['transforms'])
        return count

    if replace(cfg.data.test.pipeline) != 1:
        raise ValueError('Exactly one receiver transform required')
    dataset = native.dataset_for(cfg)
    if len(dataset) != 3316 or [str(row['token']) for row in dataset.data_infos] != [str(row['token']) for row in infos]:
        raise ValueError('Val dataset ordering/coverage changed')
    model_native = native.build_model(root, cfg, 'coop').cuda().eval()
    model_native.CLASSES = dataset.CLASSES
    parallel = MMDataParallel(model_native, device_ids=[0])
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False,
        shuffle=False, nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    cache_hashes = {token: digest(root / 'query_cache/val' / (token + '.pkl')) for token in sorted(manifest)}
    pair_identity = {'protocol': protocol,
        'seed': inference_seed, 'native_checkpoint_sha256': digest(root / 'checkpoints/cooptrack_r50_coop.pth'),
        'config_sha256': hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
        'cache_sha256': hashlib.sha256(json.dumps(cache_hashes, sort_keys=True).encode()).hexdigest(),
        'cache_file_count': len(cache_hashes),
        'val_order_sha256': hashlib.sha256(json.dumps([(str(row['token']), str(row['token_inf']), str(row['scene_token'])) for row in infos]).encode()).hexdigest(),
        'native_source_sha256': digest(inspect.getfile(type(model_native))),
        'dataset_source_sha256': digest(inspect.getfile(type(dataset))),
        'fields': list(FIELDS), 'filter_stage': 'all cached sender rows before vehicle spatial filtering',
        'serializer': 'APTU0001', 'payload': '52*N+780*U', 'frequency_hz': 10,
        'rng_reset': RNG_RULE, 'environment': environment()}
    outputs, records = [], []
    started = time.perf_counter()
    try:
        iterator = iter(loader)
        seed(inference_seed)
        pair_identity['rng_after_reset'] = rng_digest()
        write(directory / 'run_manifest.json', {'status': 'RUNNING', 'identity': identity,
            'pairing_identity': pair_identity, 'started_utc': utc(), 'command': list(sys.argv)})
        with torch.no_grad():
            for position, info in enumerate(dataset.data_infos):
                data = next(iterator)
                received = context['last']
                if received['token'] != str(info['token_inf']):
                    raise ValueError('Loader reordered an official frame')
                native.assert_loaded_queries(data, received['decoded'])
                predictions = parallel(return_loss=False, rescale=True, **data)
                if len(predictions) != 1 or str(predictions[0]['token']) != str(info['token']):
                    raise ValueError('Prediction token mismatch')
                outputs.extend(predictions)
                records.append(dict({key: value for key, value in received.items() if key != 'decoded'}, cooperative_token=str(info['token'])))
                if position % 100 == 0 or position == 3315:
                    print('V4_VAL method=%s epoch=%s seed=%s frame=%d/3316' % (method, epoch, inference_seed, position + 1), flush=True)
        if len(outputs) != 3316:
            raise ValueError('Incomplete validation cannot select checkpoints')
        after_inference = rng_digest()
        metrics, inputs = bind_and_evaluate(dataset, outputs, directory, scenes, used_checkpoint, selection)
        communication = communication_summary(records)
        communication['optional_estimate_type'] = 'learned optional probability sum' if method == 'V4' else 'deterministic Periodic hard flags; no learned soft estimate'
        write(directory / 'communication.json', {'identity': identity, 'summary': communication, 'frames': records})
        result = dict(status='PASS', identity=identity, pairing_identity=pair_identity, pairing=pair_identity, method=method,
            model=runtime.MODEL, epoch=epoch, seed=inference_seed, sample_count=3316, scene_count=21,
            metrics=metrics, communication=communication, **metrics, **communication,
            checkpoint_sha256=identity['checkpoint_sha256'], used_checkpoint_sha256=identity['used_checkpoint_sha256'],
            used_checkpoint=str(used_checkpoint), checkpoint_role='V4 scheduler' if method == 'V4' else 'native CoopTrack; V4 checkpoint is pairing anchor only',
            prediction_sha256=inputs['predictions']['sha256'], evaluator_inputs=inputs,
            result_file=str(path), seconds=time.perf_counter() - started, completed_utc=utc())
        write(path, result)
        write(directory / 'run_manifest.json', {'status': 'PASS', 'identity': identity,
            'pairing_identity': pair_identity, 'completed_utc': result['completed_utc'],
            'rng_after_inference': after_inference, 'evaluator_inputs': inputs,
            'metrics_sha256': {kind: digest(directory / kind / 'metrics_summary.json') for kind in ('det', 'track')},
            'command': list(sys.argv)})
        return result
    except Exception:
        write(directory / 'failure.json', {'status': 'FAIL', 'completed_frames': len(outputs), 'traceback': traceback.format_exc()})
        raise
    finally:
        context.clear()
        del parallel, model_native, loader, dataset, outputs, gate
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE.parents[1])
    parser.add_argument('--method', choices=('V4', 'P16'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=0)
    parser.add_argument('--selection', action='store_true')
    args = parser.parse_args()
    evaluate(args.root, args.method, args.checkpoint, args.epoch, args.seed, args.selection)
