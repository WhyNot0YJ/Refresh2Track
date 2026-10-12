"""Execute the full-46, fixed-12 V2 protocol in the inherited Docker container."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback


PERIODIC_POINTS = ('P6', 'P8', 'P10', 'P12', 'P16', 'P24', 'P32', 'INIT_ONLY')
BUDGETS = [3474, 6382, 9290, 12198]
PROTOCOL = 'full46_fixed12_official_val'


def freeze_training_source(root):
    import main_runtime as runtime
    from v2_runtime import digest, source_identity
    path = root / 'experiments/v2/metrics/source_freeze.json'
    identity = {'protocol': PROTOCOL, 'source': source_identity(root),
        'training_config_sha256': digest(root / 'configs/main_experiment_v2/training_v2.json'),
        'train_manifest_sha256': digest(root / 'configs/main_experiment_v2/train_manifest.json'),
        'native_runtime_sha256': digest(root / 'scripts/main_runtime.py'),
        'native_message_sha256': digest(root / 'scripts/main_message.py')}
    if path.is_file():
        previous = runtime.read(path)
        if previous['identity'] != identity:
            raise ValueError('Formal training source/configuration changed after freeze')
        return previous
    result = {'status': 'LOCKED', 'identity': identity,
        'frozen_utc': datetime.now(timezone.utc).isoformat()}
    runtime.write(path, result)
    return result


def freeze_protocol(root):
    import main_runtime as runtime
    from v2_runtime import digest, source_identity, verify_final_protocol
    path = root / 'experiments/v2/metrics/FINAL_PROTOCOL_MANIFEST.json'
    if path.is_file():
        return verify_final_protocol(root)
    finals = {}
    for variant in ('mlp', 'transformer'):
        checkpoint = root / 'experiments/v2/checkpoints' / variant / 'final.pth'
        state = runtime.torch.load(str(checkpoint), map_location='cpu')
        training = runtime.read(root / 'experiments/v2/metrics' / (variant + '_train.json'))
        if state.get('variant') != variant or state.get('epoch') != 12 or state.get('phase') != 'train':
            raise ValueError('Final manifest requires both formal epoch-12 checkpoints')
        if training.get('status') != 'PASS':
            raise ValueError('Final manifest requires both complete formal training records')
        finals[variant] = {'path': str(checkpoint.relative_to(root)), 'sha256': digest(checkpoint)}
    settings = runtime.read(root / 'configs/main_experiment_v2/training_v2.json')
    manifest = {'protocol': PROTOCOL, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'backbone_commit': runtime.COMMIT, 'source_hashes': source_identity(root)['files'],
        'final_checkpoints': finals, 'epochs': 12, 'batch': 4, 'threshold': .5,
        'budgets': BUDGETS, 'seed': 0, 'train_scenes': 46, 'train_samples': 7362,
        'internal_dev': None, 'val_scenes': 21, 'val_samples': 3316,
        'periodic_points': list(PERIODIC_POINTS), 'reused_periodic_points': ['P1', 'P2', 'P4'],
        'optimizer': {'name': 'AdamW', 'peak_lr': 1e-4, 'weight_decay': 1e-4, 'grad_clip': 5.0},
        'lr_schedule': {'horizon_epochs': 12, 'warmup_epochs': 1,
            'warmup_start_lr': 1e-5, 'peak_lr': 1e-4, 'after_warmup': 'cosine', 'min_lr': 1e-6},
        'precision': 'FP32', 'serializer_version': 'APTU0001 INT8; M + 780 * updates',
        'evaluator_version': 'inherited native NuScenesEval_custom / TrackingEval_custom; car-only',
        'training_config_sha256': digest(root / 'configs/main_experiment_v2/training_v2.json'),
        'train_manifest_sha256': digest(root / 'configs/main_experiment_v2/train_manifest.json'),
        'native_runtime_sha256': digest(root / 'scripts/main_runtime.py'),
        'native_message_sha256': digest(root / 'scripts/main_message.py'),
        'training_settings': settings,
        'rng': {'PYTHONHASHSEED': '0', 'python': 0, 'numpy': 0, 'torch_cpu': 0, 'torch_cuda': 0,
            'cudnn_benchmark': runtime.torch.backends.cudnn.benchmark,
            'cudnn_deterministic': runtime.torch.backends.cudnn.deterministic,
            'tf32_matmul': runtime.torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': runtime.torch.backends.cudnn.allow_tf32}}
    runtime.write(path, manifest)
    return verify_final_protocol(root)


def inherited_periodic(root):
    import main_runtime as runtime
    inherited = runtime.read(root / 'configs/main_experiment_v2/reused_val.json')
    rows = []
    for method in ('P1', 'P2', 'P4'):
        row = copy.deepcopy(inherited['methods'][method])
        row.update(method=method, status='PASS', split='val', reused=True,
            sample_count=3316, scene_count=21, seed=None,
            period_frames=int(method[1:]), nominal_period_seconds=int(method[1:]) / 10)
        row.setdefault('communication', {})
        row['mean_bytes'] = row['communication'].get('mean_bytes_per_frame',
            row.get('mean_bytes', row['metrics'].get('mean_bytes')))
        if row['mean_bytes'] is None:
            raise ValueError('Inherited Periodic reference lacks historical actual bytes')
        row['communication'].setdefault('mean_bytes_per_frame', row['mean_bytes'])
        rows.append(row)
    return rows


def cleanup_completed(root):
    """Remove only redundant resume checkpoints and incomplete temporary files."""
    import main_runtime as runtime
    from v2_runtime import verify_final_protocol
    verify_final_protocol(root)
    base = (root / 'experiments/v2').resolve()
    candidates = []
    for variant in ('mlp', 'transformer'):
        directory = base / 'checkpoints' / variant
        candidates.extend(directory.glob('epoch*.pth'))
        if (directory / 'latest.pth').is_file():
            candidates.append(directory / 'latest.pth')
    for pattern in ('*.tmp', '*.partial', '*.writing', '.eval_hook'):
        candidates.extend(path for path in base.rglob(pattern) if path.is_file())
    for failure in (base / 'metrics/official_val').glob('*/failure.json'):
        success = failure.with_name('result.json')
        if success.is_file() and runtime.read(success).get('status') == 'PASS':
            candidates.append(failure)
    for path in sorted(set(candidates)):
        if base not in path.resolve().parents or path.is_symlink():
            raise ValueError('Unsafe final cleanup path: ' + str(path))
    if candidates:
        print('FINAL_DELETE_PLAN ' + json.dumps([str(path) for path in sorted(set(candidates))]), flush=True)
    for path in set(candidates):
        path.unlink()
    return {'deleted': [str(path) for path in sorted(set(candidates))]}


def inside(args):
    import main_runtime as runtime
    runtime.initialize(args.root)
    root = args.root
    if not (root / 'reports/v2/CLEANUP_RESET.md').is_file():
        raise RuntimeError('Formal V2 requires the completed reset preservation report')
    if (root / 'configs/main_experiment_v2/dev_split.json').exists():
        raise RuntimeError('Abandoned internal split must be removed before formal V2')
    path = root / 'experiments/v2/metrics/run_state.json'
    state = runtime.read(path) if path.is_file() else {'completed': [], 'protocol': PROTOCOL}
    if state.get('protocol') != PROTOCOL:
        raise ValueError('Existing run identity belongs to a different protocol')
    if state['completed'] and not args.resume and args.phase != 'report':
        raise ValueError('Existing formal run found; use --resume')
    state.pop('error', None)
    state.pop('traceback', None)

    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        runtime.write(path, state)

    def phase(name, action, log='train'):
        print('V2_LOG_PHASE=' + log, flush=True)
        state.update(status='RUNNING', phase=name)
        save()
        print('V2_FORMAL_PHASE ' + name, flush=True)
        result = action()
        if name not in state['completed']:
            state['completed'].append(name)
        save()
        return result

    try:
        if args.phase in ('prepare', 'all'):
            from v2_prepare import prepare
            phase('prepare', lambda: prepare(root))
            from v2_message import self_check as message_check
            from v2_scheduler import self_check as scheduler_check

            def constructed_checks():
                checks = {'status': 'PASS', 'message': message_check(), 'scheduler': scheduler_check(),
                    'scope': 'constructed wire/cache/gate/ST/rate checks; real native gradient checked separately'}
                runtime.write(root / 'experiments/v2/metrics/implementation_checks.json', checks)
                return checks
            phase('implementation_checks', constructed_checks)
        if args.phase in ('correctness', 'train', 'all'):
            from v2_train import correctness_check, train
            phase('freeze-training-source', lambda: freeze_training_source(root))
            for variant in ('mlp', 'transformer'):
                phase('correctness-' + variant, lambda variant=variant: correctness_check(root, variant, resume=args.resume))
            if args.phase in ('train', 'all'):
                for variant in ('mlp', 'transformer'):
                    phase('train-' + variant, lambda variant=variant: train(root, variant, resume=args.resume))
        if args.phase in ('eval', 'all'):
            from v2_eval import evaluate, write_table
            phase('freeze-final-protocol', lambda: freeze_protocol(root))
            periodic = inherited_periodic(root)
            for method in PERIODIC_POINTS:
                periodic.append(phase('official-val-' + method,
                    lambda method=method: evaluate(root, method, resume=args.resume), log='eval'))
                write_table(root / 'experiments/v2/metrics/periodic_official_val.csv', periodic)
            learned = []
            for variant in ('mlp', 'transformer'):
                checkpoint = root / 'experiments/v2/checkpoints' / variant / 'final.pth'
                for index in range(1, 5):
                    name = 'B' + str(index)
                    learned.append(phase('official-val-' + variant + '-' + name,
                        lambda variant=variant, name=name, checkpoint=checkpoint:
                        evaluate(root, variant, budget_name=name, checkpoint=checkpoint, resume=args.resume), log='eval'))
                    write_table(root / 'experiments/v2/metrics/v2_official_val.csv', learned)
        if args.phase in ('report', 'all'):
            from v2_report import report
            summary = phase('report', lambda: report(root), log='eval')
            if summary.get('status') != 'V2_FORMAL_EXPERIMENT_COMPLETE':
                raise RuntimeError('Formal report is incomplete: ' + str(summary.get('missing', [])))
            phase('cleanup-final', lambda: cleanup_completed(root), log='eval')
            required = ('reports/v2/V2_FINAL.md', 'experiments/v2/metrics/periodic_official_val.csv',
                'experiments/v2/metrics/v2_official_val.csv', 'experiments/v2/metrics/mlp_train.json',
                'experiments/v2/metrics/transformer_train.json', 'logs/v2/train.log', 'logs/v2/eval.log')
            if any(not (root / name).is_file() for name in required):
                raise RuntimeError('Required formal output files are missing')
            state.update(status='V2_FORMAL_EXPERIMENT_COMPLETE', phase='complete')
        else:
            state['status'] = 'PHASE_COMPLETE'
        save()
        print(json.dumps({'status': state['status'], 'phase': state['phase']}), flush=True)
        return 0
    except Exception as exc:
        state.update(status='V2_FORMAL_EXPERIMENT_BLOCKED', error=str(exc), traceback=traceback.format_exc())
        save()
        print(state['traceback'], flush=True)
        print(json.dumps({'status': state['status'], 'phase': state.get('phase'), 'error': str(exc)}), flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--phase', choices=('prepare', 'correctness', 'train', 'eval', 'report', 'all'), default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inside:
        return inside(args)
    from run_main import IMAGE, CONTAINER, LINUX_ROOT
    info = json.loads(subprocess.check_output(['docker', 'inspect', CONTAINER], text=True))[0]
    if info['Image'] != IMAGE or not info['State']['Running']:
        raise RuntimeError('The inherited running CoopTrack container is required')
    log_dir = args.root / 'logs/v2'
    log_dir.mkdir(parents=True, exist_ok=True)
    command = ['docker', 'exec', '-e', 'PYTHONHASHSEED=0', '-e', 'PYTHONDONTWRITEBYTECODE=1',
        CONTAINER, 'python', LINUX_ROOT + '/scripts/v2_run.py', '--inside', '--root', LINUX_ROOT,
        '--phase', args.phase] + (['--resume'] if args.resume else [])
    logs = {name: (log_dir / (name + '.log')).open('a', encoding='utf-8') for name in ('train', 'eval')}
    current = 'eval' if args.phase in ('eval', 'report') else 'train'
    try:
        logs[current].write('\n' + json.dumps({'actual_command': command}) + '\n')
        logs[current].flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            if line.startswith('V2_LOG_PHASE='):
                current = line.strip().split('=', 1)[1]
            logs[current].write(line)
            logs[current].flush()
            print(line, end='', flush=True)
        return process.wait()
    finally:
        for log in logs.values():
            log.close()


if __name__ == '__main__':
    sys.exit(main())
