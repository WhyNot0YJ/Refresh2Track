"""Independent CPU/read-only checks of the completed formal V2 evaluation chain.

Only this audit's evidence JSON is written. No runtime.initialize, model forward,
evaluator execution, config mutation, or formal artifact modification is used.
"""
import argparse
import ast
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import struct
import sys
import zlib

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True

import numpy as np
import torch
import pyquaternion
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox

torch.set_num_threads(2)


class BoxStub:
    """The three read-only properties used by the frozen native JSON formatter.

    Identical to the installed native LiDARInstance3DBoxes/base_box3d properties;
    avoids importing native model registries or any CUDA extension.
    """
    def __len__(self):
        return len(self.tensor)

    @property
    def gravity_center(self):
        value = self.tensor[:, :3].clone()
        value[:, 2] += self.tensor[:, 5] * .5
        return value

    @property
    def dims(self):
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        return self.tensor[:, 6]


class InstanceStub:
    def get_fields(self):
        return self._fields


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name.endswith('Instance3DBoxes'):
            return BoxStub
        if name == 'Instances':
            return InstanceStub
        return super().find_class(module, name)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def infos(root, side, split):
    path = root / 'data_infos/V2X-Seq-SPD-CoopTrack-full' / side / ('spd_infos_temporal_' + split + '.pkl')
    with path.open('rb') as stream:
        return sorted(pickle.load(stream)['infos'], key=lambda row: row['timestamp'])


def decode_json(path):
    with gzip.open(path, 'rb') as stream:
        payload = stream.read()
    return json.loads(payload), hashlib.sha256(payload).hexdigest(), len(payload)


def path_from_run(root, value):
    prefix = '/mnt/d/CoopTrack_Audit/'
    assert value.startswith(prefix), value
    return root / value[len(prefix):]


def native_coordinate_helpers(root):
    path = root / 'code/CoopTrack/projects/mmdet3d_plugin/datasets/data_utils/data_utils.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = ('output_to_nusc_box', 'output_to_nusc_box_det', 'lidar_nusc_box_to_global')
    namespace = dict(np=np, Quaternion=Quaternion, pyquaternion=pyquaternion, NuScenesBox=NuScenesBox)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def coordinate_check(predictions, info_rows, formatted, helper, detection):
    # Fixed spacing plus first nonempty sample: does not select frames by success.
    indices = set(range(0, len(predictions), 257)) | {len(predictions) - 1}
    key = 'boxes_3d_det' if detection else 'boxes_3d'
    indices.add(next(index for index, row in enumerate(predictions) if len(row[key])))
    count, max_error = 0, 0.
    converter = helper['output_to_nusc_box_det' if detection else 'output_to_nusc_box']
    for index in sorted(indices):
        boxes = converter(predictions[index])
        boxes, _ = helper['lidar_nusc_box_to_global'](
            info_rows[index], boxes, ['car', 'bicycle', 'pedestrian'],
            {'car': 50, 'bicycle': 40, 'pedestrian': 40})
        expected = formatted[str(info_rows[index]['token'])]
        assert len(boxes) == len(expected), (index, len(boxes), len(expected))
        for box, row in zip(boxes, expected):
            for name, value in [('translation', box.center), ('size', box.wlh),
                                ('rotation', box.orientation.elements), ('velocity', box.velocity[:2])]:
                error = float(np.max(np.abs(np.asarray(value) - np.asarray(row[name]))))
                max_error = max(max_error, error)
                assert error <= 1e-10, (index, name, error)
            score_name = 'detection_score' if detection else 'tracking_score'
            class_name = 'detection_name' if detection else 'tracking_name'
            assert float(box.score) == row[score_name]
            assert ['car', 'bicycle', 'pedestrian'][box.label] == row[class_name]
            if not detection:
                assert str(int(box.token)) == str(row['tracking_id'])
            count += 1
    return {'status': 'PASS', 'frame_indices': sorted(indices), 'boxes_checked': count,
            'max_absolute_error': max_error, 'scope': 'archived predictions -> actual native helper -> archived evaluator JSON; CPU only'}


def main(root):
    root = Path(root).resolve()
    evidence = {'scope': 'evaluation/data separation/fairness/artifact chain', 'GPU_used': False,
                'full_val_rerun': False, 'formal_files_modified': False, 'checks': {}, 'runs': {},
                'test_execution_status': 'RUNNING'}
    target = root / 'audits/v2_bug_audit/evaluation_evidence.json'
    checks = evidence['checks']
    manifest_path = root / 'experiments/v2/metrics/FINAL_PROTOCOL_MANIFEST.json'
    manifest = read(manifest_path)
    manifest_sha = sha(manifest_path)
    protected = [manifest_path, root / 'configs/main_experiment_v2/training_v2.json',
                 root / 'configs/main_experiment_v2/train_manifest.json']
    protected += list((root / 'scripts').glob('v2_*.py'))
    protected += [root / 'scripts/main_runtime.py', root / 'scripts/main_message.py']
    protected += list((root / 'experiments/v2/metrics/official_val').glob('*/result.json'))
    protected += [root / item['path'] for item in manifest['final_checkpoints'].values()]
    before = {str(path.relative_to(root)): sha(path) for path in protected}
    official = read(root / 'code/CoopTrack/data/split_datas/cooperative-split-data-spd.json')['batch_split']
    train = infos(root, 'cooperative', 'train')
    val = infos(root, 'cooperative', 'val')
    infra = infos(root, 'infrastructure-side', 'val')
    train_tokens = {str(row['token']) for row in train}
    tokens = [str(row['token']) for row in val]
    assert len(train) == len(train_tokens) == 7362
    assert len(val) == len(set(tokens)) == len(infra) == 3316
    assert not train_tokens & set(tokens)
    assert set(official['train']).isdisjoint(official['val'])
    assert {str(row['scene_token']) for row in train} == set(official['train'])
    assert {str(row['scene_token']) for row in val} == set(official['val'])
    assert len(official['train']) == 46 and len(official['val']) == 21
    train_manifest = read(root / 'configs/main_experiment_v2/train_manifest.json')
    assert train_manifest['train_tokens'] == [str(row['token']) for row in train]
    checks['train_val_separation'] = {'status': 'PASS', 'train': [46, 7362], 'val': [21, 3316],
        'scene_overlap': [], 'cooperative_token_overlap': [], 'train_manifest_tokens_exact': True}
    del train
    native_rows, previous, session, frame = {}, None, -1, -1
    for row in infra:
        timestamp, scene = float(row['timestamp']), str(row['scene_token'])
        reset = previous is None or scene != previous['scene'] or timestamp < previous['timestamp_us'] or timestamp - previous['timestamp_us'] > 500000
        if reset:
            session, frame = session + 1, 0
        else:
            frame += 1
        current = dict(token=str(row['token']), scene=scene, session=session, frame_idx=frame,
                       native_frame_idx=int(row['frame_idx']), timestamp_us=timestamp)
        assert current['token'] not in native_rows
        native_rows[current['token']] = current
        previous = current
    del infra
    delta, calibration_errors = [], []
    for row in val:
        metadata = native_rows[str(row['token_inf'])]
        assert metadata['scene'] == str(row['scene_token'])
        assert metadata['timestamp_us'] == float(row['timestamp_inf'])
        delta.append(float(row['timestamp_inf']) - float(row['timestamp']))
        # Current cooperative record's calibrated transform; inverse is the native alignment direction.
        transform = np.eye(4)
        transform[:3, :3] = row['VehLidar2InfLidar_rotation']
        transform[:3, 3] = row['VehLidar2InfLidar_translation']
        assert np.isfinite(transform).all()
        calibration_errors.append(float(np.max(np.abs(np.linalg.inv(transform) @ transform - np.eye(4)))))
    checks['cooperative_infrastructure_mapping'] = {'status': 'PASS', 'samples_checked': len(val),
        'timestamp_delta_inf_minus_vehicle_us': {'min': min(delta), 'max': max(delta),
        'positive_count': sum(value > 0 for value in delta), 'zero_count': sum(value == 0 for value in delta)},
        'calibration_inverse_max_error': max(calibration_errors),
        'scope': 'official paired current rows, all frame/scene/timestamp; rigid transform finite/invertible; output global coordinates checked below'}
    helper = native_coordinate_helpers(root)
    logs = (root / 'logs/v2/eval.log').read_text(encoding='utf-8-sig')
    frame_fingerprints, cfg_hashes = {}, set()
    labels = ['P6_seed0', 'P8_seed0', 'P10_seed0', 'P12_seed0', 'P16_seed0', 'P24_seed0',
              'P32_seed0', 'INIT_ONLY_seed0'] + [f'{variant}_B{budget}_seed0' for variant in ('mlp', 'transformer') for budget in range(1, 5)]
    for label in labels:
        result_path = root / 'experiments/v2/metrics/official_val' / label / 'result.json'
        result = read(result_path)
        print('AUDIT_RUN', label, flush=True)
        assert result['status'] == 'PASS' and result['sample_count'] == 3316 and result['scene_count'] == 21
        identity = result['identity']
        assert identity['method'] == result['method'] and identity['split'] == 'val' and identity['seed'] == 0
        assert identity['protocol_manifest_sha256'] == manifest_sha
        assert identity['split_identity'] == hashlib.sha256(str(tokens).encode()).hexdigest()
        for name, digest in identity['sources'].items():
            assert sha(root / 'scripts' / name) == digest
        assert 'V2_EVAL_PASS ' + label + ' split=val' in logs
        assert 'V2_EVAL_REUSED ' + label not in logs
        assert manifest['frozen_utc'] < result['started_utc'] < result['completed_utc']
        if result['method'] in ('mlp', 'transformer'):
            assert identity['checkpoint_sha256'] == manifest['final_checkpoints'][result['method']]['sha256']
            assert identity['budget_name'] in label and identity['budget'] == manifest['budgets'][int(identity['budget_name'][1:]) - 1]
        else:
            assert identity['checkpoint_sha256'] is None and identity['budget'] is None
        comm_path = path_from_run(root, result['communication_file'])
        assert comm_path == root / 'experiments/v2/communication/official_val' / (label + '.json')
        communication = read(comm_path)
        assert communication['identity'] == identity
        rows = communication['frames']
        assert [row['cooperative_token'] for row in rows] == tokens
        for row, info in zip(rows, val):
            expected = native_rows[str(info['token_inf'])]
            assert all(row[key] == value for key, value in expected.items())
        output = root / 'experiments/v2/predictions/official_val' / label
        prediction_path = path_from_run(root, result['predictions_file'])
        assert prediction_path == output / 'predictions.pkl.gz'
        with gzip.open(prediction_path, 'rb') as stream:
            predictions = CPUUnpickler(stream).load()
            assert stream.read(1) == b''
        assert len(predictions) == 3316 and [str(row['token']) for row in predictions] == tokens
        for prediction in predictions:
            for key in ('boxes_3d', 'boxes_3d_det'):
                assert prediction[key].tensor.device.type == 'cpu' and prediction[key].tensor.shape[1] == 9
                assert torch.isfinite(prediction[key].tensor).all()
        run = {'status': 'PASS', 'result_path': str(result_path.relative_to(root)),
               'checkpoint_sha256': identity['checkpoint_sha256'], 'prediction_path': str(prediction_path.relative_to(root)),
               'predictions_compressed_sha256': sha(prediction_path), 'mapping_frames_checked': len(rows),
               'raw_prediction_shape': '3316 dictionaries; LiDAR boxes [N,9], scores/labels/track IDs [N], CPU tensors',
               'archived_json': {}}
        for detection, filename in [(False, 'results_nusc.json.gz'), (True, 'results_nusc_det.json.gz')]:
            path = output / filename
            formatted, payload_sha, payload_bytes = decode_json(path)
            assert set(formatted['results']) == set(tokens)
            assert all(box['sample_token'] == token for token, boxes in formatted['results'].items() for box in boxes)
            run['archived_json'][filename] = {'compressed_sha256': sha(path), 'uncompressed_sha256': payload_sha,
                'uncompressed_bytes': payload_bytes, 'samples': len(formatted['results']),
                'coordinate_check': coordinate_check(predictions, val, formatted['results'], helper, detection)}
            if not detection:
                frame_fingerprints[label] = [canonical(formatted['results'][token]) for token in tokens]
            del formatted
        evaluation_cfg = read(output / 'evaluation_config.json')
        assert evaluation_cfg['det_class_names'] == evaluation_cfg['tracking_names'] == ['car']
        assert evaluation_cfg['scenes'] == sorted(official['val']) and evaluation_cfg['samples'] == 3316
        det, track = read(output / 'det/metrics_summary.json'), read(output / 'track/metrics_summary.json')
        assert set(det['label_aps']) == set(track['label_metrics']['amota']) == {'car'}
        for metric, value in [('mAP', det['mean_ap']), ('AMOTA', track['amota']), ('AMOTP', track['amotp']), ('MOTA', track['mota']), ('IDS', track['ids'])]:
            assert result['metrics'][metric] == value
        cfg_hashes.add(canonical({'det': det['cfg'], 'track': track['cfg'], 'evaluation': evaluation_cfg}))
        run['five_metrics_exact'] = True
        run['evaluator_configuration'] = {'det': det['cfg'], 'track': track['cfg']}
        evidence['runs'][label] = run
        target.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        del predictions, rows, communication
        gc.collect()
    assert len(cfg_hashes) == 1
    differences = {}
    for index, label in enumerate(labels):
        for other in labels[:index]:
            unequal = sum(a != b for a, b in zip(frame_fingerprints[label], frame_fingerprints[other]))
            assert unequal > 0, ('identical full tracking content', label, other)
            differences[label + '__' + other] = unequal
    checks['distinct_prediction_content'] = {'status': 'PASS', 'pair_count': len(differences),
        'tracking_differing_frames_by_pair': differences, 'min_different_frames': min(differences.values()),
        'scope': 'canonical tracking box dictionaries including geometry/scores/IDs; gzip headers excluded'}
    checks['evaluator_settings_and_five_metrics'] = {'status': 'PASS', 'runs': 16,
        'all_configuration_hashes_equal': True, 'configuration_sha256': list(cfg_hashes)[0],
        'metrics': ['mAP', 'AMOTA', 'AMOTP', 'MOTA', 'IDS'], 'scope': 'native car-only summaries, 21 official scenes, full3316'}
    checks['original_prediction_hash_provenance'] = {'status': 'UNVERIFIED',
        'reason': 'Formal result.json and FINAL_PROTOCOL_MANIFEST record paths/source/checkpoint identities but do not record original prediction content SHA. Current audit computes file/content SHA; historical immutability cannot be established cryptographically.'}
    reused = read(root / 'configs/main_experiment_v2/reused_val.json')
    historical = {}
    for label in ('P1', 'P2', 'P4'):
        inherited = reused['methods'][label]
        metric = read(path_from_run(root, inherited['metric_path']))
        prediction = path_from_run(root, inherited['prediction_path'])
        assert prediction.is_file()
        if label == 'P1':
            assert inherited['communication']['mean_bytes_per_frame'] == 12161.645355850422
            raw = json.loads(prediction.read_bytes())
            assert set(raw['results']) == set(tokens)
            digest = sha(prediction)
            recorded = metric['source_hashes_before']['tracking_json']['sha256']
            assert digest == recorded == metric['source_hashes_after']['tracking_json']
            del raw
        else:
            digest = sha(prediction)
            assert digest == metric['predictions_sha256']
            with prediction.open('rb') as stream:
                saved = CPUUnpickler(stream).load()
            assert [str(row['token']) for row in saved] == tokens
            del saved
        historical[label] = {'status': 'PASS_REFERENCED_EXISTING_RESULT', 'prediction_sha256': digest,
            'recorded_sha_matches': True, 'rerun': False, 'limitation': inherited['limitation'],
            'serializer_difference': inherited.get('serializer_difference'), 'historical_mean_bytes': inherited['communication']['mean_bytes_per_frame']}
    checks['historical_references'] = historical
    checks['GT_future_input_static_chain'] = {'status': 'PASS_STATIC_ONLY',
        'current_GT_present_in_pipeline': True, 'future_annotations_enabled': False, 'forecasting_annotations_enabled': False,
        'evidence': ['configs/main_experiment_v2/coop_val.py:355 current annotations, with_future_anns=False',
            'datasets/pipelines/loading.py:220-226 with_forecasting defaults False',
            'cooptrack/detectors/cooptrack.py:272 forward_test current input; :1136 inference consumes only 5 communication kwargs; :1250 simple_test_track',
            'cooptrack/modules/motion_extractor.py:36-64 only predicted cache boxes/features',
            'cooptrack/modules/spatial_temporal_reason.py:222-250 predicted association and feature fusion'],
        'scope': 'Static reachable inference branch verified; this audit does not execute native model forward or GT-intervention black-box test.'}
    after = {str(path.relative_to(root)): sha(path) for path in protected}
    assert before == after
    checks['formal_files_unchanged'] = {'status': 'PASS', 'files': len(before), 'before_after_sha256': before}
    evidence['FINAL_PROTOCOL_MANIFEST_sha256'] = manifest_sha
    evidence['evaluation_conclusion'] = 'No concrete evaluation/data mapping bug reproduced; original prediction hash provenance remains unverified; no GPU/model forward performed.'
    evidence['test_execution_status'] = 'PASS_COMPLETED_CURRENT_ARTIFACT_CHECKS'
    evidence['overall_required_audit_status'] = 'AUDIT_INCOMPLETE_PROVENANCE_GAPS'
    evidence['audit_source_sha256'] = sha(Path(__file__))
    target = root / 'audits/v2_bug_audit/evaluation_evidence.json'
    target.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('EVALUATION_AUDIT_DONE', target, flush=True)


def supplement(root):
    """Check external prepared metadata and inherited evaluator configurations."""
    root = Path(root).resolve()
    target = root / 'audits/v2_bug_audit/evaluation_evidence.json'
    evidence = read(target)
    search_roots = ['logs/v2', 'reports/v2', 'experiments/v2/metrics',
                    'configs/main_experiment_v2', 'experiments/v2/predictions']
    pattern = re.compile(r'prediction\w*[_ ](?:sha256|hash)|(?:sha256|hash)[_ ]prediction|(?:prediction.{0,80}sha256|sha256.{0,80}prediction)', re.I)
    searched, matches = [], []
    excluded_current_audit_reports = ['reports/v2/V2_BUG_AUDIT.md']
    for directory in search_roots:
        for path in (root / directory).rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.log', '.md', '.csv'):
                if str(path.relative_to(root)).replace('\\', '/') in excluded_current_audit_reports:
                    continue
                searched.append(str(path.relative_to(root)))
                text = path.read_text(encoding='utf-8-sig')
                if pattern.search(text):
                    matches.append(str(path.relative_to(root)))
    evidence['checks']['original_prediction_hash_search'] = {'status': 'NO_ORIGINAL_PREDICTION_HASH_FOUND' if not matches else 'MATCH_REQUIRES_REVIEW',
        'search_roots': search_roots, 'text_files_searched': len(searched), 'matching_files': matches,
        'excluded_current_audit_reports': excluded_current_audit_reports,
        'scope': 'all existing logs/reports/final_summary/run_state/raw result and uncompressed per-run evaluator metadata; current audit hashes excluded'}
    manifest = read(root / 'experiments/v2/metrics/FINAL_PROTOCOL_MANIFEST.json')
    source_matches = {}
    for name, digest in manifest['source_hashes'].items():
        current = sha(root / 'scripts' / name)
        assert current == digest
        source_matches[name] = current
    assert sha(root / 'configs/main_experiment_v2/training_v2.json') == manifest['training_config_sha256']
    assert sha(root / 'configs/main_experiment_v2/train_manifest.json') == manifest['train_manifest_sha256']
    assert sha(root / 'scripts/main_runtime.py') == manifest['native_runtime_sha256']
    assert sha(root / 'scripts/main_message.py') == manifest['native_message_sha256']
    for checkpoint in manifest['final_checkpoints'].values():
        assert sha(root / checkpoint['path']) == checkpoint['sha256']
    digest = hashlib.md5()
    with (root / 'checkpoints/cooptrack_r50_coop.pth').open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == '12d4658b70f21931f3b4a793cd1bf033'
    evidence['checks']['frozen_sources_config_checkpoints'] = {'status': 'PASS', 'eight_V2_source_sha256': source_matches,
        'two_current_final_checkpoint_sha_match': True, 'native_coop_R50_checkpoint_actual_MD5': digest.hexdigest(),
        'scope': 'current bytes equal original final manifest for all recorded identity fields'}
    native_files = ['cooptrack/detectors/cooptrack.py', 'cooptrack/modules/pf_utils.py',
                    'cooptrack/modules/latent_transformation.py', 'datasets/spd_dataset.py',
                    'datasets/data_utils/data_utils.py', 'datasets/eval_utils/nuscenes_eval.py']
    native_root = root / 'code/CoopTrack/projects/mmdet3d_plugin'
    evidence['checks']['native_working_tree_historical_provenance'] = {'status': 'UNVERIFIED_HISTORICAL_HASH',
        'current_native_file_sha256': {name: sha(native_root / name) for name in native_files},
        'known_inherited_patches': ['cooptrack.py:904-908 training empty sender _init_inf_tracks is outside nonempty condition',
                                  'pf_utils.py:37 inverse computed on CPU and transferred back to the original device'],
        'reason': 'FINAL_PROTOCOL_MANIFEST locks git HEAD and native main_runtime/main_message, but not native model/evaluator working-tree file SHA; preexisting native patches differ from HEAD. No formal-before native file hash is available.'}
    val = infos(root, 'cooperative', 'val')
    prepared = root / 'datasets/prepared/V2X-Seq-SPD-CoopTrack-full_v1'
    vehicle = {row['frame_id']: row for row in read(prepared / 'vehicle-side/data_info.json')}
    infra = {row['frame_id']: row for row in read(prepared / 'infrastructure-side/data_info.json')}
    pairs = {row['vehicle_frame']: row for row in read(prepared / 'cooperative/data_info.json')}
    samples = {row['token']: row for row in read(prepared / 'cooperative/v1.0-trainval/sample.json')}
    scenes = {row['token']: row['name'] for row in read(prepared / 'cooperative/v1.0-trainval/scene.json')}
    image_pc_delta, rotation_errors, translation_errors = [], [], []
    for row in val:
        token, inf_token = str(row['token']), str(row['token_inf'])
        veh, inf, pair, sample = vehicle[token], infra[inf_token], pairs[token], samples[token]
        assert pair['infrastructure_frame'] == inf_token
        assert pair['vehicle_sequence'] == pair['infrastructure_sequence'] == str(row['scene_token'])
        assert veh['sequence_id'] == inf['sequence_id'] == scenes[sample['scene_token']] == str(row['scene_token'])
        assert float(veh['pointcloud_timestamp']) == float(row['timestamp'])
        assert float(inf['pointcloud_timestamp']) == float(row['timestamp_inf'])
        assert sample['timestamp'] == float(veh['image_timestamp'])
        assert row['cams']['VEHICLE_CAM_FRONT']['data_path'] == veh['image_path']
        assert row['system_error_offset'] == pair['system_error_offset']
        image_pc_delta.append(sample['timestamp'] - float(row['timestamp']))
        calib = read(prepared / 'infrastructure-side' / inf['calib_virtuallidar_to_world_path'])
        vr = Quaternion(row['lidar2ego_rotation']).rotation_matrix
        vg = Quaternion(row['ego2global_rotation']).rotation_matrix
        vt, vgt = np.asarray(row['lidar2ego_translation']), np.asarray(row['ego2global_translation'])
        ir, it = np.asarray(calib['rotation']), np.asarray(calib['translation']).reshape(3)
        offset = np.array([pair['system_error_offset']['delta_x'], pair['system_error_offset']['delta_y'], 0.])
        expected_r = (vr.T @ vg.T @ np.linalg.inv(ir).T).T
        expected_t = (-offset @ vr.T @ vg.T + vt @ vg.T + vgt - it) @ np.linalg.inv(ir).T
        rotation_errors.append(float(np.max(np.abs(expected_r - row['VehLidar2InfLidar_rotation']))))
        translation_errors.append(float(np.max(np.abs(expected_t - row['VehLidar2InfLidar_translation']))))
    assert max(rotation_errors) <= 1e-10 and max(translation_errors) <= 1e-9
    evidence['checks']['external_prepared_metadata_chain'] = {'status': 'PASS', 'samples_checked': len(val),
        'pairs_source': 'datasets/prepared/V2X-Seq-SPD-CoopTrack-full_v1/cooperative/data_info.json',
        'native_inference_time': 'vehicle/infrastructure pointcloud_timestamp for each official current paired frame',
        'native_evaluator_sample_time': 'same vehicle frame image_timestamp, inherited prepared NuScenes table',
        'camera_path_same_vehicle_frame': True, 'GT_sample_scene_and_token_match': True,
        'image_minus_pointcloud_us': {'min': min(image_pc_delta), 'max': max(image_pc_delta)},
        'calibration_rotation_max_error': max(rotation_errors), 'calibration_translation_max_error': max(translation_errors),
        'calibration_source': 'all3316 official infrastructure calib/virtuallidar_to_world/<token>.json plus official pair system_error_offset',
        'calibration_equations': 'spd_to_uniad.py:264-270 vehicle lidar -> global -> infrastructure lidar; native latent_transformation.py:142 uses inverse transpose convention',
        'cadence_scope': '10Hz is nominal; actual timestamp deltas and inherited image/pointcloud time bases are retained, not rewritten'}
    current = root / 'experiments/v2/predictions/official_val/P6_seed0'
    historical_cfg = {}
    for name, output in [('P1', root / 'artifacts/core/r50_alignment/communication_car_only/FEAT_INT8_seed0_val'),
                         ('P2', root / 'experiments/main/predictions/periodic2'),
                         ('P4', root / 'experiments/main/predictions/periodic4')]:
        hashes = {}
        for kind in ('det', 'track'):
            source = read(output / kind / 'metrics_summary.json')['cfg']
            reference = read(current / kind / 'metrics_summary.json')['cfg']
            assert source == reference
            hashes[kind] = canonical(source)
        historical_cfg[name] = hashes
        reused = read(root / 'configs/main_experiment_v2/reused_val.json')['methods'][name]
        reference_metric = read(path_from_run(root, reused['metric_path']))
        track_metric = read(output / 'track/metrics_summary.json')
        for metric_name in ('mAP', 'AMOTA', 'AMOTP', 'MOTA', 'IDS'):
            value = reference_metric['metrics'].get(metric_name)
            if value is None:
                value = track_metric['mota']
            assert reused['metrics'][metric_name] == value
    evidence['checks']['historical_evaluator_configuration_exact'] = {'status': 'PASS',
        'same_det_and_track_cfg_all_P1_P2_P4_and_16_new': True, 'configuration_sha256': historical_cfg,
        'fairness_limit': 'CTKMSG01 vs APTU0001 serializer and historical RNG placement differ; historical comparisons are observational references, not controlled causal refresh-policy ablations; no historical byte rebill'}
    assert not torch.cuda.is_initialized()
    evidence['checks']['no_CUDA_initialization'] = {'status': 'PASS', 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
                                                 'torch_cuda_initialized': False}
    evidence['checks']['GT_future_input_static_chain']['evidence'][0] = 'configs/main_experiment_v2/coop_val.py:334-370 current annotations; with_future_anns=False'
    evidence['evaluation_conclusion'] = 'No concrete evaluation/data mapping bug reproduced. Current recorded identities and actual artifacts pass; original prediction-content SHA and historical native working-tree SHA remain unverified. No GPU/model forward/evaluator executed by this audit.'
    evidence['audit_source_sha256'] = sha(Path(__file__))
    target.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('EVALUATION_SUPPLEMENT_DONE', target, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--supplement', action='store_true')
    args = parser.parse_args()
    if args.supplement:
        supplement(args.root)
    else:
        main(args.root)
        supplement(args.root)
