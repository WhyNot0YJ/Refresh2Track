"""Isolated ablation identities over the verified, unchanged V4 execution code."""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import textwrap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import main_runtime as native
import v2_runtime as inherited

MODES = ('no_receiver_state', 'no_explicit_delta', 'no_communication_loss')
MODELS = MODES
MODEL = None
PARAMETER_COUNT = 71361
REFERENCE = 12840.189523169913
read, write = native.read, native.write


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference(root):
    return Path(root) / 'experiments/v4_ablations/reference_v4_full'


def bind(root, mode):
    global MODEL
    if mode not in MODES:
        raise ValueError('Unknown ablation')
    MODEL = mode
    sys.modules['runtime'] = sys.modules[__name__]
    verify_reference(root)


def verify_reference(root):
    root = Path(root)
    directory = reference(root)
    info = read(directory / 'v4_full_reference.json')
    for relative, expected in info['canonical_source_hashes'].items():
        if digest(root / relative) != expected:
            raise ValueError('Frozen V4 execution source differs: ' + relative)
    for name, expected in info['reference_files'].items():
        if digest(directory / name) != expected:
            raise ValueError('Frozen V4 reference differs: ' + name)
    for name, expected in info['source_commits'].items():
        actual = subprocess.check_output(['git', '-C', str(root / 'code' / name), 'rev-parse', 'HEAD'], universal_newlines=True).strip()
        if actual != expected:
            raise ValueError('Frozen source commit differs: ' + name)
    return info


def experiment(root, model=None):
    if MODEL not in MODES or model not in (None, MODEL):
        raise ValueError('Ablation is not bound to this process')
    return Path(root) / 'experiments/v4_ablations' / MODEL


def expected_settings(root):
    settings = copy.deepcopy(read(reference(root) / 'common.json'))
    settings.update(model=MODEL, protocol='REFRESH2TRACK_ABLATION_' + MODEL.upper(), input_mode=MODEL,
                    paired_inference_seeds=[0],
                    effective_input_dimensions={'no_explicit_delta': 1556, 'no_receiver_state': 787, 'no_communication_loss': 2324}[MODEL])
    if MODEL == 'no_communication_loss':
        settings['lambda_comm'] = 0.0
    return settings


def freeze_manifest(root):
    directory = experiment(root)
    settings = expected_settings(root)
    sources = {path.name: digest(path) for path in sorted(HERE.glob('*.py'))}
    info = verify_reference(root)
    manifest = {'model': MODEL, 'settings': settings, 'ablation_sources': sources,
                'canonical_source_hashes': info['canonical_source_hashes'],
                'reference_checkpoint_sha256': info['checkpoint_sha256'],
                'reference_manifest_sha256': digest(reference(root) / 'manifest.json'),
                'initialization': 'fresh V4 MLP; same seed and construction order; never load reference scheduler'}
    for path, value in ((directory / 'config.json', settings), (directory / 'manifest.json', manifest)):
        if path.exists() and read(path) != value:
            raise ValueError('Existing ablation identity differs: ' + str(path))
        if not path.exists():
            write(path, value)
    return manifest


def load_settings(root, model=None):
    directory = experiment(root, model)
    settings = read(directory / 'config.json')
    if settings != expected_settings(root):
        raise ValueError('Frozen ablation settings changed')
    return settings


def identity(root, model=None):
    directory = experiment(root, model)
    manifest = read(directory / 'manifest.json')
    if manifest['settings'] != load_settings(root) or manifest['ablation_sources'] != {p.name: digest(p) for p in sorted(HERE.glob('*.py'))}:
        raise ValueError('Ablation computation changed after freeze')
    return {'protocol': manifest['settings']['protocol'], 'model': MODEL,
            'manifest_sha256': digest(directory / 'manifest.json'), 'settings': manifest['settings']}


def initialize(root):
    info = verify_reference(root)
    original = native.COMMIT
    try:
        native.COMMIT = info['source_commits']['CoopTrack']
        repo = native.initialize(root)
    finally:
        native.COMMIT = original
    baseline = read(reference(root) / 'final_validation.json')
    row = next(row for row in baseline['rows'] if row['method'] == 'V4' and row['seed'] == 0)
    environment = row['pairing_identity']['environment']
    native.torch.backends.cuda.matmul.allow_tf32 = environment['tf32_matmul']
    native.torch.backends.cudnn.allow_tf32 = environment['tf32_cudnn']
    return repo


def make_config(root, role, split='val', training=False, batch_size=1):
    tree = ast.parse(textwrap.dedent(inspect.getsource(inherited.make_config)))
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 'configs/main_experiment_v2':
            node.value = 'configs/refresh2track_ablations/' + MODEL + '/generated'
            changed += 1
    if changed != 1:
        raise ValueError('Canonical V4 configuration adapter differs')
    namespace = dict(inherited.make_config.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<ablation-output-config>', 'exec'), namespace)
    return namespace['make_config'](root, role, split, training, batch_size)
