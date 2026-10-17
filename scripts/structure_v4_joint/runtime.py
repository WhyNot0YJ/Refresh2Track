"""Isolated source, configuration and identity for the local joint structure run."""
import ast
import copy
import hashlib
import importlib
import importlib.util
import inspect
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import textwrap

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
import main_runtime as native
import v2_runtime as inherited

MODEL = os.environ.get('REFRESH2TRACK_STRUCTURE_MODEL', 'C_GBA_GATE')
MODELS = ('A_GBA', 'B_GATE', 'C_GBA_GATE')
if MODEL not in MODELS:
    raise ValueError('Unknown preregistered structure experiment: ' + MODEL)
COMMIT = '212a08ee6d61382bec5ad32931d834e95c26306e'
PARAMETER_COUNT = 71361
REFERENCE = 12840.189523169913
read, write = native.read, native.write


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load_module(name, path):
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != Path(path).resolve():
            raise ValueError('Module name is already bound to another source: ' + name)
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = list(sys.path)
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.path[:] = previous
    return module


def experiment(root, model=None):
    if model not in (None, MODEL):
        raise ValueError('Requested experiment differs from the process experiment: ' + MODEL)
    return Path(root) / 'experiments/structure_v4_joint' / MODEL


def bind(root, model=None):
    experiment(root, model)
    sys.modules['runtime'] = sys.modules[__name__]
    sys.modules['scheduler'] = load_module(
        'structure_scheduler', Path(root) / 'scripts/v4_budget_free/scheduler.py')


def precheck(root):
    path = Path(root) / 'experiments/structure_v4_joint/precheck.json'
    record = read(path)
    if record.get('status') != 'PASS' or record.get('git_head') != COMMIT:
        raise ValueError('A passed local precheck with the audited CoopTrack HEAD is required')
    return record


def git(root, *arguments):
    return subprocess.check_output(
        ['git', '-c', 'safe.directory=*', '-C', str(Path(root) / 'code/CoopTrack')] + list(arguments))


def source_snapshot(root):
    root = Path(root).resolve()
    audit = precheck(root)
    if git(root, 'rev-parse', 'HEAD').decode().strip() != COMMIT:
        raise ValueError('Current CoopTrack HEAD differs from the audited local source')
    original = root / 'code/CoopTrack'
    snapshot = root / 'experiments/structure_v4_joint/source/CoopTrack'
    files = {}
    for name in git(root, 'ls-files', '-z').decode('utf-8').split('\0'):
        if not name:
            continue
        parts = PurePosixPath(name)
        if parts.is_absolute() or '..' in parts.parts:
            raise ValueError('Unsafe tracked source path: ' + name)
        source = original.joinpath(*parts.parts)
        if source.is_symlink() or not source.is_file():
            continue
        target = snapshot.joinpath(*parts.parts)
        expected = digest(source)
        if target.exists():
            if digest(target) != expected:
                raise ValueError('Independent CoopTrack source differs: ' + name)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(target))
        files[name] = expected
    for name, expected in audit['source_hashes'].items():
        relative = PurePosixPath(name.replace('\\', '/')).relative_to('code/CoopTrack').as_posix()
        if files.get(relative) != expected:
            raise ValueError('CoopTrack source differs from the completed precheck: ' + name)
    if not files or (snapshot / '.git').exists():
        raise ValueError('The independent source must contain tracked files and no Git directory')
    patch = git(root, 'diff', '--binary', 'HEAD', '--')
    record = {'git_head': COMMIT, 'git_diff_sha256': hashlib.sha256(patch).hexdigest(),
              'tracked_regular_files': files, 'precheck_sha256': digest(
                  root / 'experiments/structure_v4_joint/precheck.json')}
    destination = experiment(root) / 'provenance/source_snapshot.json'
    if destination.exists() and read(destination) != record:
        raise ValueError('Independent source provenance changed')
    if not destination.exists():
        write(destination, record)
        (destination.parent / 'git_diff.patch').write_bytes(patch)
    return snapshot, record


def initialize(root):
    root = Path(root).resolve()
    snapshot, _ = source_snapshot(root)
    sys.path.insert(0, str(snapshot))
    importlib.import_module('projects.mmdet3d_plugin')
    original = native.COMMIT
    try:
        native.COMMIT = COMMIT
        native.initialize(root)
    finally:
        native.COMMIT = original
    sys.path.insert(0, str(snapshot))
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if name.startswith('projects.') and filename and snapshot not in Path(filename).resolve().parents:
            raise ValueError('CoopTrack import escaped the independent source: ' + name)
    os.chdir(str(snapshot))
    bind(root)
    return snapshot


def load_settings(root, model=None):
    experiment(root, model)
    settings = copy.deepcopy(read(Path(root) / 'configs/v4_budget_free/common.json'))
    settings.update(model=MODEL, protocol='REFRESH2TRACK_V4_STRUCTURE_JOINT_SEED0',
                    training_seed=0, inference_seed=0, paired_inference_seeds=[0],
                    max_epochs=36, batch_size=4, precision='FP32', original_peak_lr=1e-5,
                    native_lr=1e-5, fusion_lr=1e-5,
                    enable_gba_state_residual=True, enable_fusion_state_gate=True,
                    trainable_modules=['Scheduler', 'STReasoner.get_node_veh', 'STReasoner.get_node_inf',
                        'STReasoner.get_edge', 'STReasoner.wq', 'STReasoner.wk', 'STReasoner.we',
                        'STReasoner.ffn', 'STReasoner.gba_state_adapter', 'STReasoner.fuse_feats',
                        'STReasoner.fuse_motion', 'STReasoner.fuse_embed',
                        'STReasoner.cross_domain_query', 'STReasoner.cross_domain_motion',
                        'STReasoner.fusion_branch_gate'],
                    initialization='original CoopTrack base plus a fresh seed0 Scheduler; no V4 Scheduler inheritance',
                    objective='unchanged native tracking and association losses + 0.05 * mean_real_frames(780 * sum(p_optional) / R)',
                    checkpoint_rule='complete epoch latest; strict raw official-val AMOTA improvement for best; retain earlier ties')
    if MODEL != 'C_GBA_GATE':
        import modules
        gba, gate = modules.enabled(MODEL)
        settings.update(enable_gba_state_residual=gba, enable_fusion_state_gate=gate)
        settings['trainable_modules'] = [name for name in settings['trainable_modules']
                                        if name == 'Scheduler' or modules.allowed(name, MODEL)
                                        or modules.allowed(name + '.weight', MODEL)]
    settings['optimizer'] = {'type': 'AdamW', 'peak_lr': 1e-4, 'original_peak_lr': 1e-5,
                             'weight_decay': 1e-4, 'grad_clip_norm': 5.}
    settings['lr_schedule'] = {'warmup_epochs': 1, 'warmup_start_lr': 1e-5,
                               'min_lr': 1e-6, 'cosine_total_horizon_epochs': 36,
                               'original_warmup_start_lr': 1e-6, 'original_min_lr': 1e-7}
    expected = {'train_samples': 7362, 'train_scenes': 46, 'val_samples': 3316, 'val_scenes': 21,
                'lambda_comm': .05, 'threshold': .5, 'scheduler_parameters': PARAMETER_COUNT,
                'scheduler_state_dimension': 2324, 'full_int8_mean_bytes': REFERENCE,
                'validation_interval': 3, 'patience': 4, 'minimum_stop_epoch': 12,
                'latent_update_bytes': 780, 'geometry_bytes_per_query': 52, 'frequency_hz': 10}
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError('Inherited communication/data protocol differs from the preregistered run')
    return settings


def config_tree():
    tree = ast.parse(textwrap.dedent(inspect.getsource(inherited.make_config)))
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 'configs/main_experiment_v2':
            node.value = 'experiments/structure_v4_joint/' + MODEL + '/configs/generated'
            count += 1
    if count != 1:
        raise ValueError('Inherited config output adapter changed')
    return ast.fix_missing_locations(tree)


def make_config(root, role, split='val', training=False, batch_size=1):
    namespace = dict(inherited.make_config.__globals__)
    exec(compile(config_tree(), '<structure-joint-config>', 'exec'), namespace)
    return namespace['make_config'](root, role, split, training, batch_size)


def source_identity(root):
    root = Path(root).resolve()
    paths = list(HERE.glob('*.py'))
    paths += [root / 'scripts' / name for name in (
        'main_runtime.py', 'main_message.py', 'main_train.py', 'main_eval.py',
        'v2_runtime.py', 'v2_train.py', 'v2_message.py', 'v2_scheduler.py', 'v4_budget_free/native_hooks.py',
        'v4_budget_free/train.py', 'v4_budget_free/evaluate.py', 'v4_budget_free/scheduler.py')]
    paths += [root / 'configs/v4_budget_free/common.json']
    return {path.relative_to(root).as_posix(): digest(path) for path in sorted(set(paths))}


def manifest_value(root):
    root = Path(root).resolve()
    _, source = source_snapshot(root)
    normalization = read(experiment(root) / 'normalization.json')
    if normalization.get('status') != 'PASS':
        raise ValueError('Train-only normalization must pass before freezing the formal identity')
    for key, width in (('pair_scale', 12), ('gate_scale', 7)):
        values = normalization[key]
        if len(values) != width or any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError('Invalid frozen normalization: ' + key)
    audit = precheck(root)
    return {'model': MODEL, 'settings': load_settings(root), 'source_hashes': source_identity(root),
            'source_snapshot': source, 'normalization': normalization,
            'normalization_sha256': digest(experiment(root) / 'normalization.json'),
            'checkpoints': {name: audit['checkpoints'][name] for name in ('coop', 'inf')},
            'data_splits': audit['splits'], 'train_val_scene_disjoint': audit['train_val_scene_disjoint']}


def freeze_manifest(root):
    value = manifest_value(root)
    directory = experiment(root)
    path = directory / 'manifest.json'
    if path.exists() and read(path) != value:
        raise ValueError('Frozen structure source/protocol changed; refusing incompatible continuation')
    if not path.exists():
        write(directory / 'configs/resolved_config.json', value['settings'])
        write(directory / 'provenance/model_manifest.json', value)
        write(path, value)
    return value


def identity(root, model=None):
    directory = experiment(root, model)
    manifest = read(directory / 'manifest.json')
    if manifest != manifest_value(root):
        raise ValueError('Structure computation changed after formal identity freeze')
    return {'protocol': manifest['settings']['protocol'], 'model': MODEL,
            'manifest_sha256': digest(directory / 'manifest.json'), 'settings': manifest['settings']}


def self_check(root):
    settings = load_settings(root)
    compile(config_tree(), '<structure-config-self-check>', 'exec')
    assert experiment(root).name == MODEL
    assert settings['optimizer']['peak_lr'] / settings['original_peak_lr'] == 10
    assert settings['paired_inference_seeds'] == [0]
    assert settings['batch_size'] == 4 and settings['precision'] == 'FP32'
    assert settings['max_epochs'] == 36 and settings['scheduler_parameters'] == 71361
    assert all(path.is_file() for path in (HERE / 'runtime.py', HERE / 'evaluate.py'))
    return {'status': 'PASS', 'scope': 'CPU static configuration; no training or checkpoint loading'}


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE.parents[1])
    parser.add_argument('--self-check', action='store_true', required=True)
    args = parser.parse_args()
    print(json.dumps(self_check(args.root.resolve()), sort_keys=True))
