"""Independent frozen-model geometry/latent refresh ablation; never writes formal V2."""
import argparse
import copy
import gc
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import traceback
import zlib

BASE = 'experiments/v2/ablations/refresh_2x2'
MAGIC = b'CTKR2X01'
PREFIX = struct.Struct('<8sII')
FIELDS = ('query_feats', 'query_embeds', 'cache_motion_feats', 'ref_pts', 'pred_boxes')
FEATURES = FIELDS[:3]
WIDTHS = (256, 256, 256, 3, 10)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pack(current, metadata, geometry_flags, latent_flags):
    import numpy as np
    import torch
    from main_message import quantized, validate
    n = validate(current)
    flags = [torch.as_tensor(x, dtype=torch.bool).cpu() for x in (geometry_flags, latent_flags)]
    if any(x.shape != (n,) for x in flags):
        raise ValueError('Independent flags must match query count')
    header = {'version': 1, 'frame': metadata, 'queries': n, 'byte_order': 'little',
              'identity': 'predicted_track_id_int64', 'flags': ['geometry_u8', 'latent_u8'],
              'fields': list(FIELDS), 'widths': list(WIDTHS),
              'geometry_encoding': 'fp32_updated_rows',
              'latent_encoding': 'int8_per_query_symmetric_fp32_scale_updated_rows'}
    encoded = json.dumps(header, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    body = bytearray(current['obj_idxes'].cpu().numpy().astype('<i8').tobytes())
    for flag in flags:
        body.extend(flag.numpy().astype('u1').tobytes())
    for name in FIELDS[3:]:
        body.extend(current[name][flags[0]].cpu().numpy().astype('<f4').tobytes())
    for name in FEATURES:
        values, scales = quantized(current[name][flags[1]])
        body.extend(scales.tobytes())
        body.extend(values.tobytes())
    content = encoded + bytes(body)
    packet = PREFIX.pack(MAGIC, len(encoded), zlib.crc32(content)) + content
    accounting = {'control_bytes': PREFIX.size + len(encoded) + 10 * n,
                  'geometry_bytes': 52 * int(flags[0].sum()),
                  'latent_bytes': 780 * int(flags[1].sum())}
    if len(packet) != sum(accounting.values()):
        raise ValueError('Packet component accounting failed')
    return packet, accounting


class Receiver:
    """Separate received geometry and latent caches, isolated by scene/session/ID."""
    def __init__(self, pc_range):
        self.pc_range = pc_range
        self.geometry, self.latent, self.previous = {}, {}, {}

    def flags(self, ids, metadata, method):
        import torch
        namespace = (metadata['scene'], metadata['session'])
        force = torch.tensor([namespace + (int(x),) not in self.geometry or
                              namespace + (int(x),) not in self.latent for x in ids], dtype=torch.bool)
        return force | (method in ('B', 'D')), force | (method in ('C', 'D')), force

    def unpack(self, packet, expected):
        import numpy as np
        import torch
        if not PREFIX.size <= len(packet) <= 64 * 1024 * 1024:
            raise ValueError('Invalid packet size')
        magic, length, crc = PREFIX.unpack_from(packet)
        content = packet[PREFIX.size:]
        if magic != MAGIC or not 0 < length <= min(len(content), 1048576) or zlib.crc32(content) != crc:
            raise ValueError('Invalid packet prefix/CRC')
        header = json.loads(content[:length])
        metadata = header['frame']
        if metadata != expected or not {'scene', 'session', 'frame_idx', 'timestamp_us', 'token'} <= set(metadata):
            raise ValueError('Invalid frame identity')
        rotation_check = np.asarray(metadata.get('lidar2global_rotation'), dtype=np.float64)
        translation_check = np.asarray(metadata.get('lidar2global_translation'), dtype=np.float64)
        if (rotation_check.shape != (3, 3) or translation_check.shape != (3,) or
            not np.isfinite(rotation_check).all() or not np.isfinite(translation_check).all() or
            not np.allclose(rotation_check @ rotation_check.T, np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(rotation_check), 1, atol=1e-6) or
            not isinstance(metadata['session'], int) or not isinstance(metadata['frame_idx'], int) or
            not np.isfinite(metadata['timestamp_us'])):
            raise ValueError('Invalid transmitted timing/calibration metadata')
        n = header['queries']
        if (not isinstance(n, int) or not 0 <= n <= 16384 or header['version'] != 1 or
            header['byte_order'] != 'little' or header['fields'] != list(FIELDS) or header['widths'] != list(WIDTHS) or
            header['flags'] != ['geometry_u8', 'latent_u8'] or header['identity'] != 'predicted_track_id_int64' or
            header['geometry_encoding'] != 'fp32_updated_rows' or
            header['latent_encoding'] != 'int8_per_query_symmetric_fp32_scale_updated_rows'):
            raise ValueError('Unsupported packet schema')
        namespace = (metadata['scene'], metadata['session'])
        previous = self.previous.get(namespace)
        if previous is not None and (metadata['frame_idx'] <= previous['frame_idx'] or
                                     metadata['timestamp_us'] <= previous['timestamp_us']):
            raise ValueError('Unordered packet within native session')
        body, cursor = content[length:], 0

        def take(dtype, shape):
            nonlocal cursor
            count = int(np.prod(shape))
            size = count * np.dtype(dtype).itemsize
            if cursor + size > len(body):
                raise ValueError('Truncated packet')
            result = np.frombuffer(body, dtype=dtype, count=count, offset=cursor).copy().reshape(shape)
            cursor += size
            return result

        ids = take('<i8', (n,))
        g, l = take('u1', (n,)), take('u1', (n,))
        if (ids < 0).any() or len(set(ids.tolist())) != n or (g > 1).any() or (l > 1).any():
            raise ValueError('Invalid identities/flags')
        geometry_rows, latent_rows = np.flatnonzero(g), np.flatnonzero(l)
        geometry = {name: torch.from_numpy(take('<f4', (len(geometry_rows), width)))
                    for name, width in zip(FIELDS[3:], WIDTHS[3:])}
        latent = {}
        for name in FEATURES:
            scale = take('<f4', (len(latent_rows), 1))
            values = take('i1', (len(latent_rows), 256))
            if not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError('Invalid quantization scales')
            latent[name] = torch.from_numpy(values.astype(np.float32) * scale)
        if cursor != len(body) or any(not torch.isfinite(x).all() for x in [*geometry.values(), *latent.values()]):
            raise ValueError('Invalid tensor body')
        # Validate before committing so malformed packets cannot poison receiver state.
        keys = [namespace + (int(track_id),) for track_id in ids]
        for row, key in enumerate(keys):
            if key not in self.geometry or key not in self.latent:
                if not (g[row] and l[row]):
                    raise ValueError('New identity requires both geometry and latent initialization')
        geometry_cache, latent_cache = self.geometry.copy(), self.latent.copy()
        for pos, row in enumerate(geometry_rows):
            geometry_cache[keys[row]] = {name: value[pos].clone() for name, value in geometry.items()}
            geometry_cache[keys[row]].update(timestamp_us=metadata['timestamp_us'], frame_idx=metadata['frame_idx'])
            geometry_cache[keys[row]].update(rotation=metadata['lidar2global_rotation'], translation=metadata['lidar2global_translation'])
        for pos, row in enumerate(latent_rows):
            latent_cache[keys[row]] = {name: value[pos].clone() for name, value in latent.items()}
            latent_cache[keys[row]].update(timestamp_us=metadata['timestamp_us'], frame_idx=metadata['frame_idx'])
        output = {name: [] for name in FIELDS}
        geometry_age, latent_age, elapsed = [], [], []
        span = torch.tensor(self.pc_range[3:]) - torch.tensor(self.pc_range[:3])
        for key in keys:
            geo, lat = geometry_cache[key], latent_cache[key]
            dt = (metadata['timestamp_us'] - geo['timestamp_us']) / 1e6
            if dt < 0:
                raise ValueError('Negative prediction interval')
            box, ref = geo['pred_boxes'].clone(), geo['ref_pts'].clone()
            displacement = box[8:10] * dt
            box[:2] += displacement
            ref[:2] += displacement / span[:2]
            if dt:
                rotation = torch.tensor(geo['rotation'], dtype=torch.float64)
                current_rotation = torch.tensor(metadata['lidar2global_rotation'], dtype=torch.float64)
                relative = rotation @ torch.linalg.inv(current_rotation)
                shift = (torch.tensor(geo['translation'], dtype=torch.float64) -
                         torch.tensor(metadata['lidar2global_translation'], dtype=torch.float64)) @ torch.linalg.inv(current_rotation)
                center = box[[0, 1, 4]].double() @ relative + shift
                reference = (ref.double() * span.double() + torch.tensor(self.pc_range[:3], dtype=torch.float64)) @ relative + shift
                box[[0, 1, 4]] = center.float()
                ref = ((reference - torch.tensor(self.pc_range[:3], dtype=torch.float64)) / span.double()).float()
                velocity = torch.cat([box[8:10].double(), torch.zeros(1, dtype=torch.float64)]) @ relative
                yaw_delta = torch.atan2(relative[1, 0], relative[0, 0])
                sine, cosine = box[6].double(), box[7].double()
                box[6:8] = torch.stack([sine * yaw_delta.cos() + cosine * yaw_delta.sin(),
                                       cosine * yaw_delta.cos() - sine * yaw_delta.sin()]).float()
                box[8:10] = velocity[:2].float()
            output['pred_boxes'].append(box)
            output['ref_pts'].append(ref)
            for name in FEATURES:
                output[name].append(lat[name].clone())
            geometry_age.append(metadata['frame_idx'] - geo['frame_idx'])
            latent_age.append(metadata['frame_idx'] - lat['frame_idx'])
            elapsed.append(dt)
        output = {name: torch.stack(rows) if n else torch.empty((0, width), dtype=torch.float32)
                  for (name, rows), width in zip(output.items(), WIDTHS)}
        if any(not torch.isfinite(x).all() for x in output.values()):
            raise ValueError('Nonfinite receiver reconstruction')
        centers = output['ref_pts'] * span + torch.tensor(self.pc_range[:3])
        if not torch.allclose(centers, output['pred_boxes'][:, [0, 1, 4]], atol=3e-5, rtol=1e-6):
            raise ValueError('Receiver reference points and predicted box centers disagree')
        self.geometry, self.latent = geometry_cache, latent_cache
        self.previous[namespace] = metadata.copy()
        header.update(geometry_updates=int(g.sum()), latent_updates=int(l.sum()),
                      geometry_age=geometry_age, latent_age=latent_age, geometry_elapsed_seconds=elapsed)
        return output, header


def config(root):
    import main_runtime as runtime
    cfg = runtime.mmcv.Config.fromfile(str(root / 'configs/main_experiment_v2/coop_val.py'))
    if cfg.get('fp16') or cfg.model.get('fp16'):
        raise ValueError('Frozen config must use FP32')
    return cfg


def frame_manifest(root):
    import main_runtime as runtime
    import numpy as np
    from pyquaternion import Quaternion
    rows = {x['token']: x for x in runtime.native_manifest(root, 'val')}
    for info in runtime.load_infos(root, 'inf', 'val'):
        r_lidar = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        r_ego = Quaternion(info['ego2global_rotation']).rotation_matrix
        rotation = r_lidar.T @ r_ego.T
        translation = np.asarray(info['lidar2ego_translation']) @ r_ego.T + np.asarray(info['ego2global_translation'])
        rows[str(info['token'])].update(lidar2global_rotation=rotation.tolist(), lidar2global_translation=translation.tolist())
    return rows


def verify_frozen(root):
    import main_runtime as runtime
    manifest = runtime.read(root / 'experiments/v2/metrics/FINAL_PROTOCOL_MANIFEST.json')
    for name, value in manifest['source_hashes'].items():
        if sha(root / 'scripts' / name) != value:
            raise ValueError('Frozen source changed: ' + name)
    for name, item in manifest['final_checkpoints'].items():
        if sha(root / item['path']) != item['sha256']:
            raise ValueError('Frozen checkpoint changed: ' + name)
    for key, path in [('training_config_sha256', 'configs/main_experiment_v2/training_v2.json'),
                      ('train_manifest_sha256', 'configs/main_experiment_v2/train_manifest.json'),
                      ('native_runtime_sha256', 'scripts/main_runtime.py'),
                      ('native_message_sha256', 'scripts/main_message.py')]:
        if sha(root / path) != manifest[key]:
            raise ValueError('Frozen dependency changed: ' + path)
    return manifest


def stats(values):
    import numpy as np
    return {name: float(fn(values)) for name, fn in
            [('mean', np.mean), ('p95', lambda x: np.percentile(x, 95)), ('max', np.max)]} if values else {
                'mean': None, 'p95': None, 'max': None}


def diagnostic(current, decoded, header, ids):
    import torch
    import numpy as np
    # Offline only: these references never enter receiver reconstruction or model input.
    box, reference = decoded['pred_boxes'], current['pred_boxes']
    delta = box[:, [0, 1, 4]] - reference[:, [0, 1, 4]]
    yaw = torch.atan2(box[:, 6], box[:, 7]) - torch.atan2(reference[:, 6], reference[:, 7])
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw)).abs()
    speed_delta = torch.linalg.vector_norm(box[:, 8:10] - reference[:, 8:10], dim=1)
    return [{'id': int(track_id), 'geometry_age_frames': header['geometry_age'][row],
             'latent_age_frames': header['latent_age'][row], 'elapsed_seconds': header['geometry_elapsed_seconds'][row],
             'position_error_m': float(torch.linalg.vector_norm(delta[row])), 'yaw_error_rad': float(yaw[row]),
             'velocity_error_mps': float(speed_delta[row]), 'received_speed_mps': float(torch.linalg.vector_norm(box[row, 8:10]))}
            for row, track_id in enumerate(ids)]


def summarize(records):
    import numpy as np
    queries = [q for row in records for q in row['query_diagnostics']]
    n = sum(row['queries'] for row in records)
    summary = {'mean_bytes_per_frame': float(np.mean([x['bytes'] for x in records])),
               'p95_bytes_per_frame': float(np.percentile([x['bytes'] for x in records], 95)),
               'query_count': n, 'forced_init_count': sum(x['forced_init_count'] for x in records)}
    summary['bytes_per_second_at_10Hz'] = summary['mean_bytes_per_frame'] * 10
    for kind in ('control', 'geometry', 'latent'):
        summary['mean_' + kind + '_bytes'] = float(np.mean([x[kind + '_bytes'] for x in records]))
    for kind in ('geometry', 'latent'):
        summary[kind + '_update_ratio'] = sum(x[kind + '_updates'] for x in records) / n
    for field in ('geometry_age_frames', 'latent_age_frames', 'elapsed_seconds', 'position_error_m',
                  'yaw_error_rad', 'velocity_error_mps', 'received_speed_mps'):
        summary[field] = stats([x[field] for x in queries])
    summary['age_buckets'] = {}
    for name, low, high in [('0-1', 0, 2), ('2-3', 2, 4), ('4-7', 4, 8), ('8-15', 8, 16),
                            ('16-31', 16, 32), ('32+', 32, float('inf'))]:
        selected = [x for x in queries if low <= x['geometry_age_frames'] < high]
        summary['age_buckets'][name] = {'count': len(selected),
            **{field: stats([x[field] for x in selected]) for field in ('position_error_m', 'yaw_error_rad', 'velocity_error_mps')}}
    return summary


def replay_B(root):
    import main_runtime as runtime
    from main_message import ReceiverCache, pack as old_pack
    from v2_eval import split_infos
    import torch
    receiver, old_receiver = Receiver([0, -51.2, -5, 102.4, 51.2, 3]), ReceiverCache()
    manifest = frame_manifest(root)
    records = []
    for info in split_infos(root, 'val'):
        meta = manifest[str(info['token_inf'])]
        current = runtime.fields(root / 'query_cache/val' / (meta['token'] + '.pkl'))
        g, l, force = receiver.flags(current['obj_idxes'], meta, 'B')
        packet, accounting = pack(current, meta, g, l)
        decoded, header = receiver.unpack(packet, meta)
        legacy_meta = {key: value for key, value in meta.items() if not key.startswith('lidar2global_')}
        old, _ = old_receiver.unpack(old_pack(current, legacy_meta, l), legacy_meta)
        if any(not torch.equal(old[key], decoded[key]) for key in FIELDS):
            raise ValueError('B is not tensor-equivalent to frozen INIT_ONLY')
        records.append(dict(meta, cooperative_token=str(info['token']), bytes=len(packet), **accounting,
            queries=len(g), geometry_updates=int(g.sum()), latent_updates=int(l.sum()), forced_init_count=int(force.sum()),
            query_diagnostics=diagnostic(current, decoded, header, current['obj_idxes'])))
    original = runtime.read(root / 'experiments/v2/metrics/official_val/INIT_ONLY_seed0/result.json')
    original_comm = runtime.read(root / 'experiments/v2/communication/official_val/INIT_ONLY_seed0.json')
    if original['status'] != 'PASS' or len(original_comm['frames']) != len(records):
        raise ValueError('Frozen INIT_ONLY evidence incomplete')
    for old, new in zip(original_comm['frames'], records):
        if any(old[key] != new[key] for key in ('token', 'scene', 'session', 'cooperative_token')) or old['updates'] != new['latent_updates']:
            raise ValueError('B replay identities or flags differ from original official-val')
    summary = summarize(records)
    result = {'status': 'PASS', 'method': 'B', 'reused': True, 'sample_count': 3316, 'scene_count': 21,
        'metrics': {key: original['metrics'][key] for key in ('mAP', 'AMOTA', 'AMOTP', 'MOTA', 'IDS')},
        'communication': summary, 'equivalence': 'all five receiver tensors bitwise equal on all 3316 official-val frames',
        'original_result': 'experiments/v2/metrics/official_val/INIT_ONLY_seed0/result.json',
        'original_result_sha256': sha(root / 'experiments/v2/metrics/official_val/INIT_ONLY_seed0/result.json'),
        'predictions_file': original['predictions_file'], 'prediction_inference_repeated': False}
    runtime.write(root / BASE / 'communication/B.json', {'summary': summary, 'frames': records})
    runtime.write(root / BASE / 'metrics/B/result.json', result)
    print('REFRESH_B_REUSE_PASS frames=3316', flush=True)
    return result


def evaluate(root, method, correctness=False):
    import main_runtime as runtime
    from v2_runtime import seed
    from v2_eval import split_infos, car_evaluation, archive_predictions
    from mmcv.parallel import MMDataParallel
    from mmdet.datasets.builder import PIPELINES
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    torch = runtime.torch
    cfg = config(root)
    manifest = frame_manifest(root)
    receiver = Receiver([0, -51.2, -5, 102.4, 51.2, 3])
    context = {}

    @PIPELINES.register_module(force=True)
    class RefreshReceiver:
        def __init__(self, keys, load_file_root):
            if tuple(keys) != FIELDS:
                raise ValueError('Native receiver contract changed')
            self.cache_root = Path(load_file_root)

        def __call__(self, results):
            meta = manifest[str(results['sample_idx_inf'])]
            if str(results['scene_token']) != meta['scene']:
                raise ValueError('Scene identity mismatch')
            current = runtime.fields(self.cache_root / (meta['token'] + '.pkl'))
            g, l, force = receiver.flags(current['obj_idxes'], meta, method)
            clean_packet, accounting = pack(current, meta, g, l)
            actual_packet = clean_packet
            poison_checked = False
            if correctness and int((~g).sum() + (~l).sum()) > 0:
                altered = {k: v.clone() for k, v in current.items() if torch.is_tensor(v)}
                for name in FIELDS[3:]:
                    altered[name][~g] += 1234.5
                for name in FEATURES:
                    altered[name][~l] -= 4321.0
                actual_packet, other_accounting = pack(altered, meta, g, l)
                if actual_packet != clean_packet or other_accounting != accounting:
                    raise ValueError('Unsent geometry/latent leaked into serialized packet')
                poison_checked = True
            decoded, header = receiver.unpack(actual_packet, meta)
            context['last'] = dict(meta, bytes=len(actual_packet), **accounting, queries=len(g),
                geometry_updates=int(g.sum()), latent_updates=int(l.sum()), forced_init_count=int(force.sum()),
                query_diagnostics=diagnostic(current, decoded, header, current['obj_idxes']),
                decoded=decoded, poison_checked=poison_checked)
            results.update(decoded)
            return results

    def replace(pipeline):
        count = 0
        for transform in pipeline:
            if transform['type'] == 'LoadInfInformation':
                transform['type'] = 'RefreshReceiver'
                count += 1
            if 'transforms' in transform:
                count += replace(transform['transforms'])
        return count

    if replace(cfg.data.test.pipeline) != 1:
        raise ValueError('Expected exactly one native receiver transform')
    dataset = runtime.dataset_for(cfg)
    infos = split_infos(root, 'val')
    if correctness:
        first = str(infos[0]['scene_token'])
        second = next(str(x['scene_token']) for x in infos if str(x['scene_token']) != first)
        selected = [x for x in infos if str(x['scene_token']) == first][:40] + [x for x in infos if str(x['scene_token']) == second][:3]
        reset_index = next(i for i in range(1, len(infos)) if
            manifest[str(infos[i]['token_inf'])]['scene'] == manifest[str(infos[i - 1]['token_inf'])]['scene'] and
            manifest[str(infos[i]['token_inf'])]['session'] != manifest[str(infos[i - 1]['token_inf'])]['session'])
        selected += infos[reset_index - 3:reset_index + 3]
        wanted_correctness = {str(x['token']) for x in selected}
        infos = [x for x in infos if str(x['token']) in wanted_correctness]
    wanted = {str(x['token']) for x in infos}
    dataset.data_infos = [x for x in dataset.data_infos if str(x['token']) in wanted]
    if len(dataset) != len(infos):
        raise ValueError('Dataset coverage mismatch')
    model = runtime.build_model(root, cfg, 'coop').cuda()
    model.CLASSES = dataset.CLASSES
    fusion_checks, aggregation_checks, expected_aggregation = [], [], []

    def check_fusion(module, args):
        actual = args[0]
        expected = context['last']['decoded']
        for key in FIELDS:
            if not torch.equal(actual[key].detach().cpu(), expected[key]):
                raise ValueError('Unsent information bypasses receiver at real fusion input: ' + key)
        fusion_checks.append(context['last']['token'])

    hook = model.crossview_alignment.register_forward_pre_hook(check_fusion)

    def aligned_output(module, args, output):
        context['aligned'] = {key: value.detach().cpu().clone() for key, value in output.items()}
        if len(output['ref_pts']) or model.STReasoner.learn_match:
            expected_aggregation.append(context['last']['token'])

    aligned_hook = model.crossview_alignment.register_forward_hook(aligned_output)
    native_aggregation = model.STReasoner.aggregation

    def checked_aggregation(vehicle, infrastructure, sample_idx):
        for source, target in [('query_feats', 'cache_query_feats'), ('query_embeds', 'cache_query_embeds'),
                               ('cache_motion_feats', 'cache_motion_feats'), ('ref_pts', 'cache_ref_pts'),
                               ('pred_boxes', 'cache_bboxes')]:
            if not torch.equal(getattr(infrastructure, target).detach().cpu(), context['aligned'][source]):
                raise ValueError('Actual aggregation input differs from received aligned field: ' + source)
        aggregation_checks.append(context['last']['token'])
        return native_aggregation(vehicle, infrastructure, sample_idx)

    model.STReasoner.aggregation = checked_aggregation
    parallel = MMDataParallel(model, device_ids=[0])
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False,
        shuffle=False, nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    outputs, records = [], []
    out = root / BASE / 'predictions' / method
    result_path = root / BASE / 'metrics' / method / 'result.json'
    started = time.perf_counter()
    try:
        iterator = iter(loader)
        seed(0)
        for position, info in enumerate(dataset.data_infos):
            data = next(iterator)
            received = context['last']
            if received['token'] != str(info['token_inf']):
                raise ValueError('Native sender token mismatch')
            runtime.assert_loaded_queries(data, received['decoded'])
            with torch.no_grad():
                predictions = parallel(return_loss=False, rescale=True, **data)
            if len(predictions) != 1 or str(predictions[0]['token']) != str(info['token']):
                raise ValueError('Prediction token mismatch')
            def finite_output(value):
                if hasattr(value, 'tensor'):
                    value = value.tensor
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError('Nonfinite model output')
                if isinstance(value, dict):
                    for child in value.values():
                        finite_output(child)
                elif isinstance(value, (list, tuple)):
                    for child in value:
                        finite_output(child)
            finite_output(predictions[0])
            outputs.extend(predictions)
            records.append({**{k: v for k, v in received.items() if k != 'decoded'}, 'cooperative_token': str(info['token'])})
            if position % 50 == 0 or position + 1 == len(infos):
                print('REFRESH_EVAL method=%s correctness=%s frame=%d/%d bytes=%d' %
                      (method, correctness, position + 1, len(infos), received['bytes']), flush=True)
        if len(fusion_checks) != sum(x['queries'] > 0 for x in records):
            raise ValueError('Fusion hook did not inspect all nonempty receiver inputs')
        if aggregation_checks != expected_aggregation:
            raise ValueError('Aggregation inspection coverage differs from actual aligned inputs')
        summary = summarize(records)
        result = {'status': 'PASS', 'method': method, 'reused': False, 'sample_count': len(infos),
            'scene_count': len({str(x['scene_token']) for x in infos}), 'seed': 0,
            'communication': summary, 'fusion_input_checks': len(fusion_checks),
            'aggregation_input_checks': len(aggregation_checks),
            'poison_checks': sum(x['poison_checked'] for x in records), 'runtime_seconds': time.perf_counter() - started}
        if correctness:
            longest, lengths, previous_keys = 0, {}, set()
            for row in records:
                keys = {(row['scene'], row['session'], q['id']) for q in row['query_diagnostics']}
                lengths = {key: lengths.get(key, 0) + 1 if key in previous_keys else 1 for key in keys}
                longest = max(longest, max(lengths.values(), default=0))
                previous_keys = keys
            identities = [(row['scene'], row['session']) for row in records]
            session_resets = sum(a[0] == b[0] and a[1] != b[1] for a, b in zip(identities, identities[1:]))
            if longest <= 32 or session_resets < 1 or not any(x['forced_init_count'] > 0 for x in records):
                raise ValueError('Real correctness sample lacks required long-lived ID/init/session-reset coverage')
            result.update(longest_consecutive_real_id_frames=longest, real_session_resets=session_resets)
            result['frames'] = records
        else:
            out.mkdir(parents=True, exist_ok=True)
            runtime.mmcv.dump(outputs, str(out / 'predictions.pkl'))
            result['metrics'] = car_evaluation(dataset, outputs, out, sorted({str(x['scene_token']) for x in infos}))
            archive_predictions(out)
            result['predictions_file'] = str(out / 'predictions.pkl.gz')
            runtime.write(root / BASE / 'communication' / (method + '.json'), {'summary': summary, 'frames': records})
            runtime.write(result_path, result)
        return result
    except Exception:
        runtime.write(root / BASE / 'metrics' / (method + '_failure.json'),
            {'status': 'FAILED', 'correctness': correctness, 'completed_frames': len(outputs), 'traceback': traceback.format_exc()})
        raise
    finally:
        hook.remove()
        aligned_hook.remove()
        model.STReasoner.aggregation = native_aggregation
        context.clear()
        del model, parallel, dataset, loader, outputs
        gc.collect()
        torch.cuda.empty_cache()


def self_check():
    import torch
    from main_message import int8_roundtrip
    current = {name: torch.arange(3 * width, dtype=torch.float32).reshape(3, width) / 100
               for name, width in zip(FIELDS, WIDTHS)}
    current.update(obj_idxes=torch.tensor([1, 7, 11]), scores=torch.tensor([.8, .7, .6]))
    current['pred_boxes'][:, 8:10] = torch.tensor([2., -3.])
    current['ref_pts'] = (current['pred_boxes'][:, [0, 1, 4]] - torch.tensor([0., -51.2, -5.])) / torch.tensor([102.4, 102.4, 8.])
    metadata = {'scene': 'test', 'session': 0, 'frame_idx': 0, 'token': 'a', 'timestamp_us': 0.,
                'lidar2global_rotation': torch.eye(3).tolist(), 'lidar2global_translation': [0., 0., 0.]}
    receiver = Receiver([0, -51.2, -5, 102.4, 51.2, 3])
    flags = torch.ones(3, dtype=torch.bool)
    packet, parts = pack(current, metadata, flags, flags)
    decoded, _ = receiver.unpack(packet, metadata)
    assert len(packet) == sum(parts.values()) and parts['latent_bytes'] == 2340 and parts['geometry_bytes'] == 156
    assert all(torch.equal(decoded[k], int8_roundtrip(current)[k]) for k in FEATURES)
    metadata.update(frame_idx=40, token='b', timestamp_us=4123456.)
    g, l = torch.tensor([False, True, False]), torch.tensor([False, False, True])
    changed = {key: value.clone() for key, value in current.items()}
    changed['pred_boxes'][1, :2] += 10
    changed['ref_pts'][1, :2] += 10 / 102.4
    for name in FEATURES:
        changed[name] += 100
    packet, parts = pack(changed, metadata, g, l)
    decoded, header = receiver.unpack(packet, metadata)
    assert torch.allclose(decoded['pred_boxes'][0, :2], current['pred_boxes'][0, :2] + torch.tensor([2., -3.]) * 4.123456)
    assert header['geometry_age'] == [40, 0, 40] and header['latent_age'] == [40, 40, 0]
    assert parts['geometry_bytes'] == 52 and parts['latent_bytes'] == 780
    assert torch.equal(decoded['pred_boxes'][1], changed['pred_boxes'][1])
    assert torch.equal(decoded['query_feats'][0], int8_roundtrip(current)['query_feats'][0])
    assert torch.equal(decoded['query_feats'][2], int8_roundtrip(changed)['query_feats'][2])
    metadata.update(scene='parallel', frame_idx=0, timestamp_us=0., token='c')
    assert receiver.flags(current['obj_idxes'], metadata, 'A')[2].all()
    receiver.unpack(pack(current, metadata, flags, flags)[0], metadata)
    metadata.update(scene='test', session=1, token='d')
    try:
        receiver.unpack(pack(current, metadata, ~flags, ~flags)[0], metadata)
    except ValueError:
        pass
    else:
        raise AssertionError('Session reset reused stale cache')
    # Known 90-degree coordinate change and translation checks native row/yaw convention.
    pose_receiver = Receiver([0, -51.2, -5, 102.4, 51.2, 3])
    pose_meta = dict(metadata, scene='pose', session=0, frame_idx=0, token='p0', timestamp_us=0.)
    pose_receiver.unpack(pack(current, pose_meta, flags, flags)[0], pose_meta)
    pose_meta.update(frame_idx=1, token='p1', timestamp_us=1e6,
                     lidar2global_rotation=[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
                     lidar2global_translation=[10., 0., 0.])
    output, _ = pose_receiver.unpack(pack(current, pose_meta, ~flags, ~flags)[0], pose_meta)
    assert torch.allclose(output['pred_boxes'][:, 0], -current['pred_boxes'][:, 1] + 3.)
    assert torch.allclose(output['pred_boxes'][:, 1], current['pred_boxes'][:, 0] - 8.)
    assert torch.allclose(output['pred_boxes'][:, 8:10], torch.tensor([[3., 2.]]).expand(3, 2))
    assert torch.allclose(output['pred_boxes'][:, 6], -current['pred_boxes'][:, 7])
    assert torch.allclose(output['pred_boxes'][:, 7], current['pred_boxes'][:, 6])
    print('REFRESH_WIRE_SELF_CHECK_PASS', flush=True)


def inside(args):
    import main_runtime as runtime
    runtime.initialize(args.root)
    verify_frozen(args.root)
    identity = {
        'configuration_sha256': sha(args.root / 'configs/main_experiment_v2/refresh_ablation.json'),
        'source_sha256': sha(Path(__file__)), 'formal_manifest_sha256': sha(args.root / 'experiments/v2/metrics/FINAL_PROTOCOL_MANIFEST.json'),
        'cooperative_checkpoint_sha256': sha(args.root / 'checkpoints/cooptrack_r50_coop.pth'),
        'config': runtime.read(args.root / 'configs/main_experiment_v2/refresh_ablation.json')}
    manifest_path = args.root / BASE / 'metrics/ablation_manifest.json'
    if args.phase == 'eval' and (not manifest_path.is_file() or runtime.read(manifest_path) != identity):
        raise ValueError('Ablation source/config changed since correctness; rerun correctness before evaluation')
    if args.phase in ('correctness', 'all'):
        if any((args.root / BASE / 'metrics' / method / 'result.json').is_file() for method in 'ACD'):
            if runtime.read(manifest_path) != identity:
                raise ValueError('Ablation source/config changed after official-val results')
        runtime.write(manifest_path, identity)
    if args.phase in ('correctness', 'all'):
        self_check()
        checks = {method: evaluate(args.root, method, True) for method in 'ABCD'}
        runtime.write(args.root / BASE / 'metrics/correctness.json', {'status': 'PASS', 'methods': checks})
    if args.phase in ('eval', 'all'):
        checks = runtime.read(args.root / BASE / 'metrics/correctness.json')
        if checks['status'] != 'PASS':
            raise ValueError('Correctness must pass before official-val evaluation')
        for method in args.methods:
            path = args.root / BASE / 'metrics' / method / 'result.json'
            if path.is_file():
                if runtime.read(path)['status'] != 'PASS':
                    raise ValueError('Existing result incomplete')
                print('REFRESH_RESULT_REUSED ' + method, flush=True)
                continue
            replay_B(args.root) if method == 'B' else evaluate(args.root, method)
    verify_frozen(args.root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--phase', choices=('correctness', 'eval', 'all'), default='all')
    parser.add_argument('--methods', default='BACD')
    parser.add_argument('--inside', action='store_true')
    args = parser.parse_args()
    if not args.methods or any(x not in 'ABCD' for x in args.methods):
        parser.error('Methods must be a sequence of A/B/C/D')
    if args.inside:
        inside(args)
        return 0
    from run_main import IMAGE, CONTAINER, LINUX_ROOT
    info = json.loads(subprocess.check_output(['docker', 'inspect', CONTAINER], text=True))[0]
    if info['Image'] != IMAGE or not info['State']['Running']:
        raise RuntimeError('Inherited running container required')
    log = args.root / BASE / 'metrics/run.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    command = ['docker', 'exec', '-e', 'PYTHONHASHSEED=0', '-e', 'PYTHONDONTWRITEBYTECODE=1', CONTAINER,
               'python', LINUX_ROOT + '/scripts/refresh_ablation.py', '--inside', '--root', LINUX_ROOT,
               '--phase', args.phase, '--methods', args.methods]
    with log.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'command': command}) + '\n')
        stream.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end='', flush=True)
        return process.wait()


if __name__ == '__main__':
    sys.exit(main())
