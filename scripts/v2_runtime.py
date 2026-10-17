"""V2 configuration adapter; inherited inputs remain read-only."""
import hashlib
import json
import random
from pathlib import Path

import main_runtime as native


def initialize(root):
    return native.initialize(root)


def make_config(root, role, split='val', training=False, batch_size=1):
    root = Path(root)
    if split not in ('train', 'val'):
        raise ValueError('Formal V2 accepts only the complete official train or val dataset')
    if training and (split != 'train' or batch_size != 4):
        raise ValueError('Formal V2 training requires all official train samples and native batch 4')
    repo = root / 'code/CoopTrack'
    cfg = native.mmcv.Config.fromfile(str(repo / native.CONFIGS[role]))
    if cfg.get('fp16') is not None:
        raise ValueError('V2 requires the inherited FP32 configuration')
    prepared = Path('/data/spd_prepared')
    if not prepared.is_dir():
        prepared = root / 'datasets/prepared/V2X-Seq-SPD-CoopTrack-full_v1'
    data_root = prepared / native.SIDES[role]
    info_root = root / 'data_infos/V2X-Seq-SPD-CoopTrack-full' / native.SIDES[role]
    cache_split = split
    query_root = root / 'query_cache' / cache_split
    cfg.data_root, cfg.info_root = str(data_root), str(info_root)
    cfg.ann_file_train = str(info_root / 'spd_infos_temporal_train.pkl')
    cfg.ann_file_val = str(info_root / 'spd_infos_temporal_val.pkl')
    cfg.ann_file_test = cfg.ann_file_train if cache_split == 'train' else cfg.ann_file_val
    cfg.split_datas_file = str(repo / 'data/split_datas/cooperative-split-data-spd.json')
    for name in ('train_pipeline', 'test_pipeline', 'eval_pipeline'):
        if name in cfg:
            native.patch_pipeline(cfg[name], data_root, query_root)
    for name in ('train', 'val', 'test'):
        ds = cfg.data[name]
        ds.data_root = str(data_root)
        ds.ann_file = cfg.ann_file_train if name == 'train' else cfg.ann_file_test
        ds.split_datas_file = cfg.split_datas_file
        native.patch_pipeline(ds.pipeline, data_root, query_root)
    cfg.data.test.test_mode = True
    cfg.data.test.samples_per_gpu = 1
    cfg.data.samples_per_gpu = batch_size if training else 1
    cfg.data.workers_per_gpu = 0
    cfg.model.pretrained = None
    cfg.load_from = cfg.resume_from = None
    if training:
        cfg.model.batch_size = batch_size
        cfg.data.train.samples_per_gpu = batch_size
        cfg.data.train.filter_empty_gt = False
    if role == 'inf':
        cfg.save_track_query = cfg.model.save_track_query = True
        cfg.save_track_query_file_root = cfg.model.save_track_query_file_root = str(query_root)
    elif role == 'coop':
        cfg.is_cooperation = cfg.model.is_cooperation = True
        cfg.read_track_query_file_root = cfg.model.read_track_query_file_root = str(query_root)
    if cfg.get('evaluation', {}).get('pipeline'):
        native.patch_pipeline(cfg.evaluation.pipeline, data_root, query_root)
    target = root / 'configs/main_experiment_v2' / (role + '_' + split + ('_train' if training else '') + '.py')
    target.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(target))
    return cfg


def seed(seed_value):
    torch = native.torch
    random.seed(seed_value)
    native.np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    # Match the inherited numerical setting; do not promise deterministic CUDA kernels.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def rng_state():
    torch = native.torch
    return {'python': random.getstate(), 'numpy': native.np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    torch = native.torch
    random.setstate(state['python'])
    native.np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(root):
    root = Path(root)
    files = {path.name: digest(path) for path in sorted((root / 'scripts').glob('v2_*.py'))}
    return {'files': files, 'sha256': hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()}


def training_identity(root, variant, phase='train', batch_size=4):
    root = Path(root)
    if variant not in ('mlp', 'transformer') or phase != 'train' or batch_size != 4:
        raise ValueError('Formal V2 training identity requires MLP/Transformer, train phase and native batch 4')
    return {'backbone_commit': native.COMMIT, 'variant': variant, 'phase': phase,
            'protocol_version': 'V2-FORMAL-20261007-FULL46-12E',
            'native_batch_size': batch_size, 'v2_source': source_identity(root),
            'training_config_sha256': digest(root / 'configs/main_experiment_v2/training_v2.json'),
            'train_manifest_sha256': digest(root / 'configs/main_experiment_v2/train_manifest.json'),
            'cooperative_checkpoint': native.CHECKPOINTS['coop'][0]}


def verify_final_protocol(root):
    """Reject official-val entry until both fixed-epoch models and sources are frozen."""
    root = Path(root)
    path = root / 'experiments/v2/metrics/FINAL_PROTOCOL_MANIFEST.json'
    if not path.is_file():
        raise ValueError('Official val requires FINAL_PROTOCOL_MANIFEST.json after both formal trainings')
    manifest = native.read(path)
    expected = {'protocol': 'full46_fixed12_official_val', 'epochs': 12, 'batch': 4,
                'threshold': 0.5, 'seed': 0, 'budgets': [3474, 6382, 9290, 12198],
                'periodic_points': ['P6', 'P8', 'P10', 'P12', 'P16', 'P24', 'P32', 'INIT_ONLY']}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError('Frozen formal protocol differs: ' + key)
    if manifest.get('source_hashes') != source_identity(root)['files']:
        raise ValueError('V2 source changed after final official-val protocol freeze')
    settings = native.read(root / 'configs/main_experiment_v2/training_v2.json')
    if (settings.get('train_scenes') != 46 or settings.get('train_samples') != 7362 or
            settings.get('epochs') != 12 or settings.get('max_epochs') != 12 or
            settings.get('actual_native_batch_size') != 4 or settings.get('effective_optimizer_batch_size') != 4 or
            settings.get('training_seed') != 0 or settings.get('primary_inference_seed') != 0 or
            settings.get('budgets_bytes') != expected['budgets'] or settings.get('gate', {}).get('threshold') != 0.5 or
            settings.get('lr_schedule', {}).get('cosine_total_horizon_epochs') != 12):
        raise ValueError('Formal training configuration differs from full46/fixed12/batch4/seed0 protocol')
    if ('training_config_sha256' in manifest and
            manifest['training_config_sha256'] != digest(root / 'configs/main_experiment_v2/training_v2.json')):
        raise ValueError('Training configuration changed after final protocol freeze')
    checkpoints = manifest.get('final_checkpoints', {})
    if set(checkpoints) != {'mlp', 'transformer'}:
        raise ValueError('Both frozen formal final checkpoints are required')
    for variant in ('mlp', 'transformer'):
        relative = 'experiments/v2/checkpoints/' + variant + '/final.pth'
        checkpoint = checkpoints[variant]
        if (checkpoint.get('path') != relative or not (root / relative).is_file() or
                checkpoint.get('sha256') != digest(root / relative)):
            raise ValueError('Frozen formal final checkpoint differs: ' + variant)
        result = native.read(root / 'experiments/v2/metrics' / (variant + '_train.json'))
        if (result.get('status') != 'PASS' or result.get('completed_epoch') != 12 or
                result.get('train_scenes') != 46 or result.get('train_samples') != 7362):
            raise ValueError('Complete full46 fixed12 formal training evidence missing: ' + variant)
    return manifest


def assert_loaded_queries(data, expected_batch):
    """Check each collated sample without assuming batch size one."""
    for name in expected_batch[0]:
        values = native.unwrap(data[name])
        # mmcv.collate groups GPU samples; queue_length=1 supplies the final nesting.
        if len(values) != 1 or len(values[0]) != len(expected_batch):
            raise ValueError('V2 collated sender batch shape differs: ' + name)
        for index, expected in enumerate(expected_batch):
            sample = values[0][index]
            if not isinstance(sample, list) or len(sample) != 1:
                raise ValueError('V2 requires one temporal frame per sample')
            if not native.torch.equal(sample[0].cpu(), expected[name].cpu()):
                raise ValueError('Loaded sender fields differ at batch sample %d: %s' % (index, name))


def state_copy(model):
    """Preserve references to detached native stream history."""
    return {'batch_size': model.batch_size, 'test_flag': model.test_flag,
            'train_prev_infos': {name: list(values) for name, values in model.train_prev_infos.items()},
            'training': model.training}


def restore_state(model, state):
    model.batch_size = state['batch_size']
    model.train_prev_infos = state['train_prev_infos']
    model.test_flag = state['test_flag']
    model.train(state['training'])
    for module in model.modules():
        if isinstance(module, native.torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
