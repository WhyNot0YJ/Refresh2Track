"""Isolated V4 paths and immutable shared-asset protocol bindings."""
import ast
from datetime import datetime, timezone
import hashlib
import inspect
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import textwrap

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main_runtime as native
import v2_runtime as inherited

MODEL = 'v4_budget_free_mlp_lambda005_seed0'
MODELS = (MODEL,)
REFERENCE = 12840.189523169913
PARAMETER_COUNT = 71361
ARCHIVE = 'archives/v3_fixed_budget_20261012'
read, write = native.read, native.write


def retirement(root):
    """Explicit maintenance migration, preserving the frozen computation identity."""
    path = experiment(root) / 'audit/v3_retirement_transition.json'
    if not path.is_file():
        return None
    record = read(path)
    if (record.get('status') != 'V3_RETIRED_DOCUMENTS_ONLY'
            or digest(experiment(root) / 'manifest.json') != record['v4_manifest_sha256']):
        raise ValueError('V3 retirement does not bind the unchanged V4 manifest')
    merge_path = experiment(root) / 'audit/git_merge_resume.json'
    if merge_path.is_file():
        merge = read(merge_path)
        source = record['approved_sources']['scripts/v4_budget_free/runtime.py']
        if (merge.get('status') != 'VERIFIED_COMPUTATION_PRESERVING_MERGE'
                or merge['retirement_sha256'] != digest(path)
                or merge['v4_manifest_sha256'] != record['v4_manifest_sha256']
                or merge['previous_runtime_sha256'] != source['current_sha256']):
            raise ValueError('Git merge resume does not bind the prior maintenance identity')
        source['current_sha256'] = merge['runtime_sha256']
    for name, row in record['approved_sources'].items():
        if digest(_relative(root, row['current_path'])) != row['current_sha256']:
            raise ValueError('V4 maintenance source changed: ' + name)
    return record


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def experiment(root, model=None):
    if model not in (None, MODEL):
        raise ValueError('Only the preregistered V4 model is authorized')
    return Path(root) / 'experiments/v4_budget_free'


def load_settings(root, model=None):
    experiment(root, model)
    settings = read(Path(root) / 'configs/v4_budget_free/common.json')
    expected = {'model': MODEL, 'protocol': 'V4_BUDGET_FREE_FULL46_LAMBDA005_SEED0',
        'max_epochs': 36, 'training_seed': 0, 'inference_seed': 0,
        'paired_inference_seeds': [0, 2025, 2026], 'precision': 'FP32', 'batch_size': 4,
        'train_samples': 7362, 'train_scenes': 46, 'val_samples': 3316, 'val_scenes': 21,
        'full_int8_mean_bytes': REFERENCE, 'lambda_comm': 0.05, 'threshold': 0.5,
        'scheduler_state_dimension': 2324, 'scheduler_parameters': PARAMETER_COUNT,
        'latent_update_bytes': 780, 'geometry_bytes_per_query': 52, 'frequency_hz': 10,
        'validation_interval': 3, 'patience': 4, 'minimum_stop_epoch': 12,
        'diagnostic_steps': [2, 10],
        'optimizer': {'type': 'AdamW', 'peak_lr': 1e-4, 'weight_decay': 1e-4, 'grad_clip_norm': 5.0},
        'lr_schedule': {'warmup_epochs': 1, 'warmup_start_lr': 1e-5, 'min_lr': 1e-6,
                        'cosine_total_horizon_epochs': 36}}
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError('V4 settings differ from the single preregistered protocol')
    if any('budget' in key.lower() or key == 'lambda_rate' for key in settings):
        raise ValueError('V4 configuration contains a budget condition or old penalty')
    return settings


def source_identity(root):
    root = Path(root)
    migration = retirement(root)
    if migration:
        frozen = read(experiment(root) / 'manifest.json')['source_hashes']
        approved = migration['approved_sources']
        for name, expected in frozen.items():
            row = approved.get(name)
            if row:
                if row['original_sha256'] != expected:
                    raise ValueError('Maintenance source lacks its original frozen binding: ' + name)
            elif digest(_relative(root, name)) != expected:
                raise ValueError('Frozen V4 computation changed: ' + name)
        actual = {p.relative_to(root).as_posix() for p in (root / 'scripts/v4_budget_free').glob('*.py')}
        expected = {approved.get(name, {}).get('current_path', name) for name in frozen
                    if name.startswith(('scripts/v4_budget_free/', 'scripts/v3_budget/'))}
        if actual != expected:
            raise ValueError('V4 maintenance source coverage changed')
        return frozen
    paths = list((root / 'scripts/v4_budget_free').glob('*.py'))
    paths += list((root / 'configs/v4_budget_free').glob('*.json'))
    paths += [root / 'scripts' / name for name in ('main_runtime.py', 'main_message.py',
        'main_train.py', 'main_eval.py', 'v2_runtime.py', 'v2_train.py',
        'v2_message.py', 'v2_scheduler.py', 'run_main.py', 'v3_budget/native_hooks.py')]
    paths += list((root / 'code/CoopTrack/projects/mmdet3d_plugin').rglob('*.py'))
    return {path.relative_to(root).as_posix(): digest(path) for path in sorted(set(paths))}


def _relative(root, name):
    parts = PurePosixPath(name)
    if parts.is_absolute() or '..' in parts.parts:
        raise ValueError('Archive manifest contains an unsafe relative path')
    target = (Path(root) / Path(*parts.parts)).resolve()
    if Path(root).resolve() not in target.parents:
        raise ValueError('Archive manifest path escapes its root')
    return target


def verify_protection(root):
    """Rehash every archived copy and protected original; never write old paths."""
    root = Path(root).resolve()
    migration = retirement(root)
    if migration:
        source_identity(root)
        mismatches = []
        for name, row in migration['protected_shared_files'].items():
            path = _relative(root, name)
            if not path.is_file() or path.stat().st_size != row['size'] or digest(path) != row['sha256']:
                mismatches.append({'path': name, 'reason': 'protected shared asset changed or missing'})
        return {'status': 'FAIL' if mismatches else 'PASS',
                'archive_status': migration['status'], 'archive_copies_checked': 0,
                'originals_checked': len(migration['protected_shared_files']),
                'scope': 'remaining shared assets; V3 copies explicitly deleted by user request',
                'migration_path': 'experiments/v4_budget_free/audit/v3_retirement_transition.json',
                'mismatches': mismatches, 'checked_utc': datetime.now(timezone.utc).isoformat()}
    archive = root / ARCHIVE
    manifest = read(archive / 'manifest.json')
    baseline = read(archive / 'protection_before.json')
    mismatches = []
    if manifest.get('status') != 'V3_ARCHIVE_VERIFIED':
        mismatches.append({'path': ARCHIVE, 'reason': 'archive gate is not verified'})
    if len(baseline['files']) != manifest['protected_file_count'] or len(manifest['files']) != manifest['file_count']:
        mismatches.append({'path': ARCHIVE, 'reason': 'archive/protection file coverage differs'})
    for relative, row in baseline['files'].items():
        path = _relative(root, relative)
        if not path.is_file():
            mismatches.append({'path': relative, 'reason': 'protected original missing'})
        elif path.stat().st_size != row['size'] or digest(path) != row['sha256']:
            mismatches.append({'path': relative, 'reason': 'protected original size/SHA256 changed'})
    for row in manifest['files']:
        relative = row['archive_relative_path']
        path = _relative(archive, relative)
        if not path.is_file():
            mismatches.append({'path': ARCHIVE + '/' + relative, 'reason': 'archive copy missing'})
        elif path.stat().st_size != row['size_bytes'] or digest(path) != row['archive_sha256_after']:
            mismatches.append({'path': ARCHIVE + '/' + relative, 'reason': 'archive copy size/SHA256 changed'})
        if row['source_sha256_before'] != row['source_sha256_after'] or row['source_sha256_before'] != row['archive_sha256_after']:
            mismatches.append({'path': row['source_relative_path'], 'reason': 'archive provenance hashes disagree'})
        if baseline['hashes'].get(row['source_relative_path']) != row['source_sha256_before']:
            mismatches.append({'path': row['source_relative_path'], 'reason': 'archive/protection source binding differs'})
    return {'status': 'FAIL' if mismatches else 'PASS', 'archive_status': manifest.get('status'),
        'originals_checked': len(baseline['files']), 'archive_copies_checked': len(manifest['files']),
        'scope': 'full protected original files and all archived copies; dataset/cache directories locked by existing manifests',
        'archive_manifest_sha256': digest(archive / 'manifest.json'),
        'protection_baseline_sha256': digest(archive / 'protection_before.json'),
        'mismatches': mismatches, 'checked_utc': datetime.now(timezone.utc).isoformat()}


def reference_evidence(root):
    root = Path(root)
    path = root / 'configs/main_experiment/budgets.json'
    values = read(path)
    cache = root / 'query_cache/train/manifest.json'
    if (values.get('status') != 'LOCKED_FROM_FULL_TRAIN' or values.get('val_used') is not False
            or values['full_int8_mean_bytes'] != REFERENCE or values['frames'] != 7445
            or values['cooperative_samples'] != 7362 or values['scenes'] != 46
            or values['source_cache_manifest_sha256'] != digest(cache)):
        raise ValueError('Inherited R lacks a matching train-only cache reference')
    return {'value': REFERENCE, 'units': 'B/frame', 'normalization_only': True,
        'source': path.relative_to(root).as_posix(), 'source_sha256': digest(path),
        'cache_manifest_sha256': digest(cache), 'infrastructure_frames': 7445,
        'cooperative_samples': 7362, 'train_scenes': 46, 'val_used': False,
        'definition': 'train-only CTKMSG01 FEAT_INT8 full serialized mean, including its existing overhead'}


def freeze_manifest(root):
    """Freeze only after all V4 files exist and before any formal GPU training."""
    root = Path(root).resolve()
    settings = load_settings(root)
    if retirement(root):
        identity(root)
        protection = verify_protection(root)
        if protection['status'] != 'PASS':
            raise ValueError('Shared asset protection failed after V3 retirement')
        write(experiment(root) / 'audit/protection_before_v4.json', protection)
        return read(experiment(root) / 'manifest.json')
    if any(not (root / 'scripts/v4_budget_free' / name).is_file() for name in
           ('runtime.py', 'scheduler.py', 'train.py', 'evaluate.py', 'run.py', 'report.py')):
        raise ValueError('All V4 entrypoints must exist before freezing the experiment')
    protection = verify_protection(root)
    if protection['status'] != 'PASS':
        raise ValueError('V3 preservation verification failed: ' + str(protection['mismatches']))
    archive = root / ARCHIVE
    archived = read(archive / 'manifest.json')
    shared = read(archive / archived['shared_asset_reference_manifest'])
    official_path = root / 'code/CoopTrack/data/split_datas/cooperative-split-data-spd.json'
    official = read(official_path)['batch_split']
    if len(official['train']) != 46 or len(official['val']) != 21:
        raise ValueError('Official split scene coverage differs')
    baseline = read(archive / 'protection_before.json')
    manifest = {'protocol': settings['protocol'], 'model': MODEL, 'settings': settings,
        'source_hashes': source_identity(root), 'reference_evidence': reference_evidence(root),
        'train_scenes': sorted(official['train']), 'val_scenes': sorted(official['val']),
        'train_samples': 7362, 'val_samples': 3316, 'training_seed': 0,
        'inference_seed': 0, 'paired_inference_seeds': [0, 2025, 2026],
        'lambda_comm': 0.05, 'scheduler_parameters': PARAMETER_COUNT, 'state_dimension': 2324,
        'objective': settings['objective'], 'selection': 'strict raw official-val AMOTA; ties do not improve',
        'resume_boundary': 'complete epoch', 'official_val_use': settings['official_val_use'],
        'archive': {'path': ARCHIVE, 'status': archived['status'],
            'manifest_sha256': protection['archive_manifest_sha256'],
            'protection_baseline_sha256': protection['protection_baseline_sha256'],
            'shared_assets_sha256': digest(archive / archived['shared_asset_reference_manifest'])},
        'shared_assets': shared,
        'asset_locks': {name: value for name, value in baseline['hashes'].items()
            if name.startswith(('checkpoints/', 'data_infos/', 'query_cache/'))
                or name == official_path.relative_to(root).as_posix()
                or name == 'artifacts/core/val_cache_manifest.json'},
        'rng': {'per_scene': 'persistent independent preprocessing and native lane RNG',
            'batch': 'separate RNG for genuinely batched operations',
            'inference_reset': 'after construction/loading/iter(loader), before first next',
            'PYTHONHASHSEED': '0', 'cudnn_deterministic': False, 'cudnn_benchmark': False}}
    path = experiment(root) / 'manifest.json'
    if path.exists():
        if read(path) != manifest:
            raise ValueError('Frozen V4 source/protocol changed; refusing incompatible resume')
    else:
        write(path, manifest)
        for name in manifest['source_hashes']:
            if name.startswith(('scripts/v4_budget_free/', 'configs/v4_budget_free/')):
                destination = experiment(root) / 'audit/source_snapshot' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, destination)
    write(experiment(root) / 'audit/protection_before_v4.json', protection)
    return manifest


def identity(root, model=None):
    experiment(root, model)
    manifest = read(experiment(root) / 'manifest.json')
    if manifest['source_hashes'] != source_identity(root) or manifest['settings'] != load_settings(root):
        raise ValueError('V4 source/configuration changed after freeze')
    if not retirement(root) and manifest['archive']['manifest_sha256'] != digest(Path(root) / ARCHIVE / 'manifest.json'):
        raise ValueError('V3 archive manifest changed after V4 freeze')
    return {'protocol': manifest['protocol'], 'model': MODEL,
        'manifest_sha256': digest(experiment(root) / 'manifest.json'), 'settings': manifest['settings']}


def make_config(root, role, split='val', training=False, batch_size=1):
    tree = ast.parse(textwrap.dedent(inspect.getsource(inherited.make_config)))
    replacements = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 'configs/main_experiment_v2':
            node.value = 'configs/v4_budget_free/generated'
            replacements += 1
    if replacements != 1:
        raise ValueError('Inherited config adapter shape changed')
    namespace = dict(inherited.make_config.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<v4-config-adapter>', 'exec'), namespace)
    return namespace['make_config'](root, role, split, training, batch_size)


def initialize(root):
    root = Path(root).resolve()
    actual = subprocess.check_output(['git', '-C', str(root / 'code/CoopTrack'),
                                      'rev-parse', 'HEAD'], universal_newlines=True).strip()
    if actual == native.COMMIT:
        return native.initialize(root)
    path = experiment(root) / 'audit/git_merge_resume.json'
    if not path.is_file():
        raise ValueError('Frozen CoopTrack commit differs: ' + actual)
    merge = read(path)
    identity(root)  # All frozen computation and approved maintenance hashes still match.
    if actual != merge['accepted_commit'] or native.COMMIT != merge['original_commit']:
        raise ValueError('CoopTrack HEAD differs from the verified computation-preserving merge')
    original = native.COMMIT
    try:
        # The shared runtime file stays frozen; this V4 process accepts only the audited HEAD.
        native.COMMIT = actual
        return native.initialize(root)
    finally:
        native.COMMIT = original
