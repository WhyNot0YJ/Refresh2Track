"""Frozen native CoopTrack runtime for the formal full-data experiment."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

COMMIT = '29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399'
CONFIGS = {role: 'projects/configs_spd_' + role + '/cooptrack/tiny_track_r50_stream_bs8_48epoch_3cls.py'
           for role in ('inf', 'coop', 'veh')}
SIDES = {'inf': 'infrastructure-side', 'coop': 'cooperative', 'veh': 'vehicle-side'}
CHECKPOINTS = {'inf': ('cooptrack_r50_inf.pth', '0657439da18e951724f6b4e5a623ea23'),
               'coop': ('cooptrack_r50_coop.pth', '12d4658b70f21931f3b4a793cd1bf033'),
               'veh': ('cooptrack_r50_veh.pth', '34d9f0553e4eb421bdb4bb3b7dae41aa')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def initialize(root):
    if not Path('/.dockerenv').is_file():
        raise RuntimeError('Use the inherited CoopTrack Docker image; no host framework installation.')
    repo = Path(root) / 'code/CoopTrack'
    actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise ValueError('Frozen CoopTrack commit differs: ' + actual)
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    global torch, np, mmcv, DataContainer
    import torch
    import numpy as np
    import mmcv
    from mmcv.parallel import DataContainer
    importlib.import_module('projects.mmdet3d_plugin')
    import cv2
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    from mmdet.apis import set_random_seed
    set_random_seed(0, deterministic=False)
    torch.backends.cudnn.benchmark = False
    return repo


def patch_pipeline(pipeline, data_root, query_root):
    for transform in pipeline:
        if transform.get('type') == 'LoadMultiViewImageFromFilesInCeph':
            transform['img_root'] = str(data_root)
        elif transform.get('type') == 'LoadInfInformation':
            transform['load_file_root'] = str(query_root)
        if 'transforms' in transform:
            patch_pipeline(transform['transforms'], data_root, query_root)


def make_config(root, role, split='val', training=False):
    root = Path(root)
    repo = root / 'code/CoopTrack'
    cfg = mmcv.Config.fromfile(str(repo / CONFIGS[role]))
    if cfg.get('fp16') is not None:
        raise ValueError('Formal runtime requires the inherited FP32 configuration')
    prepared = Path('/data/spd_prepared')
    if not prepared.is_dir():
        prepared = root / 'datasets/prepared/V2X-Seq-SPD-CoopTrack-full_v1'
    data_root = prepared / SIDES[role]
    info_root = root / 'data_infos/V2X-Seq-SPD-CoopTrack-full' / SIDES[role]
    query_root = root / 'query_cache' / split
    split_file = repo / 'data/split_datas/cooperative-split-data-spd.json'
    cfg.data_root, cfg.info_root = str(data_root), str(info_root)
    cfg.ann_file_train = str(info_root / 'spd_infos_temporal_train.pkl')
    cfg.ann_file_val = str(info_root / 'spd_infos_temporal_val.pkl')
    cfg.ann_file_test = cfg.ann_file_train if split == 'train' else cfg.ann_file_val
    cfg.split_datas_file = str(split_file)
    for name in ('train_pipeline', 'test_pipeline', 'eval_pipeline'):
        if name in cfg:
            patch_pipeline(cfg[name], data_root, query_root)
    for name in ('train', 'val', 'test'):
        ds = cfg.data[name]
        ds.data_root = str(data_root)
        ds.ann_file = cfg.ann_file_train if name == 'train' else cfg.ann_file_test
        ds.split_datas_file = str(split_file)
        patch_pipeline(ds.pipeline, data_root, query_root)
    cfg.data.test.test_mode = True
    cfg.data.test.samples_per_gpu = 1
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0
    cfg.model.pretrained = None
    cfg.load_from = cfg.resume_from = None
    if role == 'inf':
        cfg.save_track_query = cfg.model.save_track_query = True
        cfg.save_track_query_file_root = cfg.model.save_track_query_file_root = str(query_root)
    elif role == 'coop':
        cfg.is_cooperation = cfg.model.is_cooperation = True
        cfg.read_track_query_file_root = cfg.model.read_track_query_file_root = str(query_root)
    if cfg.get('evaluation', {}).get('pipeline'):
        patch_pipeline(cfg.evaluation.pipeline, data_root, query_root)
    target = root / 'configs/main_experiment' / (role + '_' + split + ('_train' if training else '') + '.py')
    target.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(target))
    return cfg


def build_model(root, cfg, role):
    from mmdet3d.models import build_model as native_build
    from mmcv.runner import load_checkpoint
    model = native_build(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = Path(root) / 'checkpoints' / CHECKPOINTS[role][0]
    loaded = load_checkpoint(model, str(checkpoint), map_location='cpu', strict=False)
    own = model.state_dict()
    state = {(key[7:] if key.startswith('module.') else key): value
             for key, value in loaded.get('state_dict', loaded).items()}
    missing = set(own) - set(state)
    unexpected = set(state) - set(own)
    bad_missing = [key for key in missing if not key.endswith('.num_batches_tracked')]
    if bad_missing or unexpected or any(own[key].shape != state[key].shape for key in set(own) & set(state)):
        raise ValueError('Official checkpoint/model key or shape mismatch')
    model.float().eval().requires_grad_(False)
    if any(getattr(module, 'fp16_enabled', False) for module in model.modules()):
        raise ValueError('Unexpected FP16 module')
    return model


def dataset_for(cfg, training=False):
    from mmdet3d.datasets import build_dataset
    options = copy.deepcopy(cfg.data.train if training else cfg.data.test)
    options.pop('samples_per_gpu', None)
    return build_dataset(options)


def fields(path):
    value = mmcv.load(str(path))
    result = dict(value.get_fields() if hasattr(value, 'get_fields') else value)
    ids = result.get('obj_idxes')
    # Native export Instances.to(torch.zeros(1)) casts predicted integer IDs to FP32.
    # Recover identities in memory; protected native cache files stay unchanged.
    if ids is not None and ids.dtype != torch.int64:
        if not ids.is_floating_point() or not torch.isfinite(ids).all() or not torch.equal(ids, ids.round()) or (ids.abs() > 2 ** 24).any():
            raise ValueError('Native exported predicted IDs cannot be recovered exactly')
        result['obj_idxes'] = ids.to(torch.int64)
    return result


def load_infos(root, role, split):
    path = Path(root) / 'data_infos/V2X-Seq-SPD-CoopTrack-full' / SIDES[role] / ('spd_infos_temporal_' + split + '.pkl')
    return sorted(mmcv.load(str(path))['infos'], key=lambda row: row['timestamp'])


def native_manifest(root, split):
    """Preserve native sender resets even where cooperative pairing skips frames."""
    previous, session, frame_idx = None, -1, -1
    rows = []
    for info in load_infos(root, 'inf', split):
        timestamp = float(info['timestamp'])
        scene = str(info['scene_token'])
        reset = previous is None or scene != previous['scene'] or timestamp < previous['timestamp_us'] or timestamp - previous['timestamp_us'] > 500000
        if reset:
            session += 1
            frame_idx = 0
        else:
            frame_idx += 1
        row = {'token': str(info['token']), 'scene': scene, 'session': session,
               'frame_idx': frame_idx, 'native_frame_idx': int(info['frame_idx']), 'timestamp_us': timestamp}
        rows.append(row)
        previous = row
    expected = 7445 if split == 'train' else 3316
    scenes = 46 if split == 'train' else 21
    if len(rows) != expected or len({row['scene'] for row in rows}) != scenes:
        raise ValueError('Official full infrastructure split count mismatch')
    official = read(Path(root) / 'code/CoopTrack/data/split_datas/cooperative-split-data-spd.json')['batch_split'][split]
    if {row['scene'] for row in rows} != set(official):
        raise ValueError('Official full infrastructure scene identity mismatch')
    seen = set()
    previous_scene = None
    for row in rows:
        if row['scene'] != previous_scene:
            if row['scene'] in seen:
                raise ValueError('Native timestamp-sorted scenes interleave; explicit per-scene replay is required')
            seen.add(row['scene'])
            previous_scene = row['scene']
    return rows


def unwrap(value):
    if isinstance(value, DataContainer):
        return unwrap(value.data)
    if isinstance(value, (list, tuple)):
        return [unwrap(item) for item in value]
    if isinstance(value, dict):
        return {key: unwrap(item) for key, item in value.items()}
    return value


def collect_metas(value):
    value = unwrap(value)
    if isinstance(value, dict):
        return [value] if 'sample_idx' in value else []
    if isinstance(value, list):
        return [meta for item in value for meta in collect_metas(item)]
    return []


def assert_loaded_queries(data, expected):
    for key, value in expected.items():
        loaded = unwrap(data[key])
        while isinstance(loaded, list):
            if len(loaded) != 1:
                raise ValueError('Expected batch of one in formal evaluation')
            loaded = loaded[0]
        if loaded.ndim == value.ndim + 1:
            loaded = loaded[0]
        if not torch.equal(loaded.cpu(), value.cpu()):
            raise ValueError('Serialized receiver fields were not used by model: ' + key)
