"""Gate all three ablations, then run serially or in independent branch processes."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
import runtime
import report as reporting

_LOCK_DESCRIPTORS = ()
_EXECUTION_MODE = 'serial_B_A_C'


def utc():
    return datetime.now(timezone.utc).isoformat()


def require_storage(root):
    free = shutil.disk_usage(root).free
    if free < 2 * 1024 ** 3:
        raise RuntimeError('Less than 2 GiB free for checkpoints and full evaluator outputs; release storage before continuing')


def log_event(root, **values):
    values.update(utc=utc(), model=runtime.MODEL, coordinator_pid=os.getpid(), execution_mode=_EXECUTION_MODE)
    print(json.dumps(values, allow_nan=False), flush=True)
    path = runtime.experiment(root) / 'run_state.json'
    old = runtime.read(path) if path.exists() else {}
    runtime.write(path, dict(old, **values))
    reporting.report(root)


def worker(root, script, options=(), phase='train'):
    # Canonical coordinator options carry --model; the isolated worker uses --mode.
    options = list(options)
    if '--model' in options:
        index = options.index('--model')
        if index + 1 >= len(options) or options[index + 1] != runtime.MODEL:
            raise ValueError('Canonical worker mode differs from the bound ablation')
        del options[index:index + 2]
    if script == 'train.py':
        action = 'correctness' if '--correctness-check' in options else 'train'
        if '--correctness-check' in options:
            options.remove('--correctness-check')
    elif script == 'evaluate.py':
        action = 'evaluate'
    else:
        action = script
    if action in ('train', 'evaluate'):
        require_storage(root)
    command = [sys.executable, '-B', str(HERE / 'worker.py'), '--root', str(root),
               '--mode', runtime.MODEL, '--action', action] + options
    path = Path(root) / 'logs/refresh2track_ablations' / runtime.MODEL / (phase + '.log')
    path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')
    with path.open('a', encoding='utf-8', buffering=1) as output:
        output.write(json.dumps({'command': command, 'utc': utc()}) + '\n')
        # Workers inherit the branch flock descriptors. If the parent crashes,
        # running GPU work still owns those locks until it exits.
        subprocess.run(command, check=True, env=environment, stdout=output,
                       stderr=subprocess.STDOUT, pass_fds=_LOCK_DESCRIPTORS)


def canonical(root, mode):
    runtime.bind(root, mode)
    module = runtime.load_module('refresh2track_canonical_coordinator', HERE.parent / 'v4_budget_free/run.py')
    module.runtime, module.MODEL = runtime, mode
    module.worker, module.log_event = worker, log_event
    return module


def epoch_diagnostic(row, settings):
    for name in ('task_loss', 'comm_loss', 'total_loss', 'scheduler_gradient_norm', 'optional_update_ratio'):
        value = row.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return 'Missing or nonfinite epoch ' + name
    if row.get('sample_count', row.get('samples')) != 7362 or row.get('scene_count', row.get('scenes')) != 46:
        return 'Epoch lacks all 7362 official samples / 46 scenes'
    if row.get('non_finite_batch_count', 0) or row['scheduler_gradient_norm'] <= 0 or not row.get('task_gradient_verified'):
        return 'Nonfinite or vanished native task/Scheduler gradient'
    for name in ('first_task_gradient_norm', 'first_comm_gradient_norm'):
        value = row.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            return 'Missing finite nonzero native gradient proof: ' + name
    if not 0 <= row['optional_update_ratio'] <= 1:
        return 'Actual optional UPDATE ratio is outside [0, 1]'
    if not math.isclose(row['total_loss'], row['task_loss'] + settings['lambda_comm'] * row['comm_loss'], rel_tol=1e-5, abs_tol=1e-6):
        return 'Total loss differs from the frozen communication coefficient'
    # Extreme UPDATE/REUSE is a possible ablation result, not a stop condition.
    return None


def preflight(root):
    import audit
    runtime.bind(root, runtime.MODES[0])
    log_event(root, status='RUNNING', phase='read_only_audit')
    with reporting.report_lock(root):
        evidence = audit.audit(root)
    if evidence.get('status') != 'PASS':
        raise ValueError('V4/data/config audit failed')
    with reporting.report_lock(root):
        worker(root, 'cpu', phase='cpu')
    masks = runtime.read(Path(root) / 'reports/ablations/input_mask_tests.json')
    rows = masks.get('input_modes', [])
    cache = masks.get('cache_serializer_checks', {})
    if (masks.get('status') != 'PASS' or {row['mode'] for row in rows} != set(runtime.MODES) or
            any(row.get('status') != 'PASS' for row in rows) or cache.get('status') != 'PASS'):
        raise ValueError('All three input masks and cache/serializer CPU checks must pass')
    for check in ('session_scene_reset', 'id_permutation_cache_alignment', 'serializer_actual_780_bytes_per_update',
                  'empty_and_padded_queries', 'received_vs_observed_history'):
        if cache.get(check) is not True:
            raise ValueError('CPU cache correctness not proven: ' + check)
    for mode in runtime.MODES:
        runtime.bind(root, mode)
        runtime.freeze_manifest(root)
    # Correctness workers discard all their weights. Every branch is proven
    # before the first branch is allowed to process a formal training epoch.
    proofs = {}
    for mode in runtime.MODES:
        runtime.bind(root, mode)
        identity = runtime.identity(root)
        path = runtime.experiment(root) / 'audit/preflight.json'
        if path.exists():
            old = runtime.read(path)
            if old.get('status') != 'PASS' or old.get('identity') != identity:
                raise ValueError('Existing preflight differs or failed: ' + mode)
        log_event(root, status='RUNNING', phase='native_gpu_correctness')
        worker(root, 'train.py', ['--model', mode, '--correctness-check'], 'correctness')
        proof_path = runtime.experiment(root) / 'metrics/correctness.json'
        proof = runtime.read(proof_path)
        if proof.get('status') != 'PASS' or proof.get('identity') != identity or proof.get('native_batch_size') != 4:
            raise ValueError('Matching batch-four native correctness proof required: ' + mode)
        checks = ('native_task_gradient_nonzero', 'communication_gradient_nonzero',
                  'native_parameters_and_BN_frozen', 'real_packet_INT8_UPDATE_REUSE',
                  'hard_STE_equals_actual_decoded', 'lane_cache_rng_isolation',
                  'future_reasoning_disabled', 'resume_rng_optimizer_lr_identity_roundtrip', 'weights_discarded')
        if any(proof.get(name) is not True for name in checks):
            raise ValueError('Native correctness check failed: ' + mode)
        for name in ('task_gradient_norm', 'comm_gradient_norm', 'scheduler_gradient_norm'):
            value = proof.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Nonzero finite task/communication/STE gradient required: ' + mode)
        mode_mask = next(row for row in rows if row['mode'] == mode)
        if mode_mask.get('lambda_comm') != runtime.load_settings(root)['lambda_comm']:
            raise ValueError('CPU loss coefficient differs from the frozen settings')
        proofs[mode] = {'status': 'PASS', 'identity': identity, 'cpu_proof': str(Path(root) / 'reports/ablations/input_mask_tests.json'),
                        'native_proof': str(proof_path), 'reference_audit': str(Path(root) / 'reports/ablations/v4_preflight.json'),
                        'assertions': dict.fromkeys(checks, True), 'input_mask': mode_mask,
                        'training_started': (runtime.experiment(root) / 'checkpoints/latest.pth').exists(),
                        'utc': utc(), 'all_branches_preflight_before_formal_training': True}
    for mode, proof in proofs.items():
        runtime.bind(root, mode)
        path = runtime.experiment(root) / 'audit/preflight.json'
        if not path.exists():
            runtime.write(path, proof)
        log_event(root, status='PREFLIGHT_PASS', phase='preflight_complete')
    with reporting.report_lock(root):
        runtime.write(Path(root) / 'reports/ablations/all_preflight.json', {
            'status': 'PASS', 'utc': utc(), 'branches': proofs, 'cpu_mask_tests_sha256':
            runtime.digest(Path(root) / 'reports/ablations/input_mask_tests.json')})


def require_all_preflight(root, selected_mode):
    """Independent branch startup consumes prior gates without rewriting them."""
    path = Path(root) / 'reports/ablations/all_preflight.json'
    if not path.is_file():
        raise ValueError('Run --phase preflight for all three modes before launching independent branches')
    all_proofs = runtime.read(path)
    if (all_proofs.get('status') != 'PASS' or set(all_proofs.get('branches', {})) != set(runtime.MODES) or
            runtime.digest(Path(root) / 'reports/ablations/input_mask_tests.json') != all_proofs.get('cpu_mask_tests_sha256')):
        raise ValueError('Matching real all-three CPU/native preflight evidence required')
    for mode in runtime.MODES:
        runtime.bind(root, mode)
        identity = runtime.identity(root)
        proof = runtime.read(runtime.experiment(root) / 'audit/preflight.json')
        native = runtime.read(runtime.experiment(root) / 'metrics/correctness.json')
        recorded = all_proofs['branches'][mode]
        if (proof.get('status') != 'PASS' or proof.get('identity') != identity or
                recorded.get('status') != 'PASS' or recorded.get('identity') != identity or
                native.get('status') != 'PASS' or native.get('identity') != identity or
                native.get('native_batch_size') != 4 or not proof.get('all_branches_preflight_before_formal_training') or
                not recorded.get('all_branches_preflight_before_formal_training') or
                not proof.get('assertions') or recorded.get('assertions') != proof['assertions'] or
                not all(value is True for value in proof['assertions'].values()) or
                not all(native.get(name) is True for name in proof['assertions'])):
            raise ValueError('All-three preflight identity or native correctness differs: ' + mode)
        for name in ('task_gradient_norm', 'comm_gradient_norm', 'scheduler_gradient_norm'):
            value = native.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Prepared native gradients are invalid: ' + mode)
    runtime.bind(root, selected_mode)


def chosen_modes(mode=None):
    if mode is not None and mode not in runtime.MODES:
        raise ValueError('Unknown ablation mode')
    return (mode,) if mode is not None else tuple(runtime.MODES)


def lock_paths(root, mode=None):
    return tuple(Path(root) / 'experiments/v4_ablations' / branch / 'coordinator.lock'
                 for branch in chosen_modes(mode))


def acquire_locks(root, mode=None):
    """Serial runs acquire every branch lock; independent runs acquire one."""
    import fcntl
    descriptors = []
    try:
        for path in lock_paths(root, mode):
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
            descriptors.append(descriptor)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('An ablation coordinator or its GPU worker already holds ' + str(path))
            os.ftruncate(descriptor, 0)
            os.write(descriptor, json.dumps({'pid': os.getpid(), 'utc': utc(), 'root': str(root), 'mode': mode}).encode())
        return tuple(descriptors)
    except Exception:
        for descriptor in descriptors:
            os.close(descriptor)
        raise


def train_branch(root, mode, resume):
    runner = canonical(root, mode)
    directory = runtime.experiment(root)
    settings = runtime.load_settings(root)
    checkpoints = directory / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest = checkpoints / 'latest.pth'
    selection_path = directory / 'metrics/best_selection.json'
    state = runner.checkpoint_load(root, latest) if latest.exists() else None
    if state and not resume:
        raise ValueError('Existing completed ablation epoch requires --resume')
    if not state and (selection_path.exists() or (checkpoints / 'best.pth').exists()):
        raise ValueError('Selection exists without its completed training state')
    completed = state['epoch'] if state else 0
    selection = runtime.read(selection_path) if selection_path.exists() else {}
    if state:
        runner.write_history(directory / 'metrics/train_history.json', state['training_result']['epochs'])
        for row in state['training_result']['epochs']:
            reason = epoch_diagnostic(row, settings)
            if reason:
                raise ValueError(reason)
    # A finalization checkpoint marks training as PASS. Do not continue its
    # training after a crash between training completion and final evaluation.
    finalized = bool(state and state['training_result'].get('status') == 'PASS')
    if not finalized and completed and completed % settings['validation_interval'] == 0:
        selection = runner.validate_epoch(root, completed)
    stopped = completed >= settings['minimum_stop_epoch'] and selection.get('patience', 0) >= settings['patience']
    if not finalized:
        for epoch in range(completed + 1, settings['max_epochs'] + 1):
            if stopped:
                break
            log_event(root, status='RUNNING', phase='training', epoch=epoch)
            worker(root, 'train.py', ['--model', mode, '--epoch', str(epoch)])
            state = runner.checkpoint_load(root, latest, epoch)
            completed = epoch
            row = state['training_result']['epochs'][-1]
            reason = epoch_diagnostic(row, settings)
            runtime.write(directory / 'audit' / ('epoch%03d_diagnostic.json' % epoch), {
                'status': 'FAIL' if reason else 'PASS', 'reason': reason, 'epoch': epoch,
                'optional_update_ratio': row.get('optional_update_ratio'),
                'extreme_optional_update_ratio': row.get('optional_update_ratio', .5) <= .001 or row.get('optional_update_ratio', .5) >= .999,
                'extreme_ratio_action': 'record only; no tuning or performance-based stopping',
                'checkpoint_sha256': runtime.digest(latest)})
            if reason:
                raise ValueError(reason)
            log_event(root, status='RUNNING', phase='epoch_complete', completed_epoch=epoch)
            if epoch % settings['validation_interval'] == 0:
                selection = runner.validate_epoch(root, epoch)
                log_event(root, status='RUNNING', phase='selection_complete', completed_epoch=epoch,
                          best_epoch=selection['best_epoch'], patience=selection['patience'])
                stopped = epoch >= settings['minimum_stop_epoch'] and selection.get('patience', 0) >= settings['patience']
        if selection.get('best_epoch') is None:
            raise ValueError('Cannot finish without a selected complete official validation')
        selection.update(status='PASS', stop_epoch=completed,
                         stop_reason='early_stopping' if stopped else 'max_epochs_36')
        state = runner.checkpoint_load(root, latest, completed)
        state['selection'] = selection
        state['training_result'].update(status='PASS', stop_epoch=completed, stop_reason=selection['stop_reason'])
        runner.checkpoint_save(latest, state)
        runner.checkpoint_save(checkpoints / 'last.pth', state)
        runtime.write(selection_path, selection)
        runtime.write(directory / 'metrics/training_state.json', state['training_result'])
    else:
        # Finalization is idempotent if a crash occurred before its ledgers were written.
        selection = state['selection']
        runtime.write(selection_path, selection)
        runtime.write(directory / 'metrics/training_state.json', state['training_result'])
        runner.checkpoint_save(checkpoints / 'last.pth', state)
    best = checkpoints / 'best.pth'
    chosen = runner.checkpoint_load(root, best, selection['best_epoch'])
    if runtime.digest(best) != selection['checkpoint_sha256'] or runner.scheduler_hash(chosen) != selection['scheduler_state_sha256']:
        raise ValueError('Selected best checkpoint changed')
    frozen = {'status': 'FROZEN', 'identity': runtime.identity(root), 'best_epoch': selection['best_epoch'],
              'best_score': selection['best_score'], 'checkpoint': str(best), 'checkpoint_sha256': runtime.digest(best),
              'scheduler_state_sha256': runner.scheduler_hash(chosen), 'stop_epoch': completed,
              'selection_rule': 'full official val raw AMOTA, strict improvement; first tie retained', 'inference_seed': 0}
    path = directory / 'metrics/frozen_selection.json'
    if path.exists() and runtime.read(path) != frozen:
        raise ValueError('Frozen final checkpoint selection differs')
    if not path.exists():
        runtime.write(path, frozen)
    log_event(root, status='RUNNING', phase='final_validation', epoch=selection['best_epoch'], seed=0)
    worker(root, 'evaluate.py', ['--method', 'V4', '--checkpoint', str(best),
                                '--epoch', str(selection['best_epoch']), '--seed', '0'], 'eval')
    result = runner.require_result(root, directory / 'validation/final/V4_seed0/result.json', best, 'V4', selection['best_epoch'], 0)
    if result.get('identity', {}).get('protocol') != frozen['identity'] or runtime.digest(best) != frozen['checkpoint_sha256']:
        raise ValueError('Final evaluation identity or frozen checkpoint differs')
    runtime.verify_reference(root)
    runtime.write(directory / 'metrics/final_validation.json', {'status': 'PASS', 'identity': frozen['identity'], 'rows': [result],
        'training_seed': 0, 'inference_seeds': [0], 'checkpoint_sha256': runtime.digest(best),
        'official_val_use': 'checkpoint selection and reporting; not independent test generalization'})
    log_event(root, status='COMPLETE', phase='complete', best_epoch=selection['best_epoch'],
              checkpoint=str(best), completed_epoch=completed)


def execute(args):
    global _LOCK_DESCRIPTORS, _EXECUTION_MODE
    root = args.root.resolve()
    if args.phase == 'report':
        print(json.dumps(reporting.report(root)), flush=True)
        return
    if sys.platform != 'linux':
        raise RuntimeError('Execute in the restored Linux/CUDA environment on the cloud server')
    _EXECUTION_MODE = 'independent_branch_processes' if args.mode else 'serial_B_A_C'
    try:
        _LOCK_DESCRIPTORS = acquire_locks(root, args.mode)
        modes = chosen_modes(args.mode)
        if args.mode:
            # Bind early so failures belong to this process's own branch only.
            runtime.bind(root, args.mode)
        if not args.resume and any((root / 'experiments/v4_ablations' / mode / 'checkpoints/latest.pth').exists() for mode in modes):
            raise ValueError('Existing completed ablation epochs require --resume')
        require_storage(root)
        if args.mode:
            require_all_preflight(root, args.mode)
            log_event(root, status='PREFLIGHT_PASS', phase='prepared_preflight_verified')
        else:
            preflight(root)
        if args.phase == 'preflight':
            return
        for mode in modes:
            train_branch(root, mode, args.resume)
        reporting.report(root)
    except Exception as exc:
        if not _LOCK_DESCRIPTORS:
            # A rejected duplicate must not overwrite the active run's state.
            print(str(exc), file=sys.stderr, flush=True)
            raise
        failure = {'status': 'FAILED', 'phase': 'stopped', 'error': str(exc), 'traceback': traceback.format_exc(),
                   'utc': utc(), 'automatic_retry': False, 'mode': args.mode or runtime.MODEL,
                   'coordinator_pid': os.getpid(), 'execution_mode': _EXECUTION_MODE}
        if args.mode:
            # The cross-mode evidence verifier may have been bound to a sibling
            # when it failed. Never write into that sibling's active state.
            runtime.MODEL = args.mode
        if runtime.MODEL in runtime.MODES:
            runtime.write(runtime.experiment(root) / 'audit/coordinator_failure.json', failure)
            runtime.write(runtime.experiment(root) / 'run_state.json', failure)
        try:
            reporting.report(root)
        except Exception:
            pass
        print(json.dumps(failure), file=sys.stderr, flush=True)
        raise
    finally:
        # Do not explicitly LOCK_UN while an orphan worker may still hold its
        # inherited descriptor; closing this process's descriptor is sufficient.
        for descriptor in _LOCK_DESCRIPTORS:
            os.close(descriptor)
        _LOCK_DESCRIPTORS = ()


def self_check():
    assert chosen_modes() == tuple(runtime.MODES)
    assert all(chosen_modes(mode) == (mode,) for mode in runtime.MODES)
    branch_locks = lock_paths(Path('/fixture'))
    assert len(branch_locks) == 3 and len(set(branch_locks)) == 3
    assert all(lock_paths(Path('/fixture'), mode) == (branch_locks[index],)
               for index, mode in enumerate(runtime.MODES))
    assert parser().parse_args(['--mode', runtime.MODES[1], '--resume']).mode == runtime.MODES[1]
    selection_module = runtime.load_module('refresh2track_selection_selfcheck', HERE.parent / 'v4_budget_free/run.py')
    selection, improved = selection_module.selection_update({}, .7, 3)
    assert improved
    for epoch in (6, 9, 12, 15):
        selection, improved = selection_module.selection_update(selection, .7, epoch)
        assert not improved and selection['best_epoch'] == 3
    assert selection['patience'] == 4
    row = {'task_loss': 1., 'comm_loss': .2, 'total_loss': 1.01, 'scheduler_gradient_norm': .02,
           'optional_update_ratio': 0., 'samples': 7362, 'scenes': 46,
           'first_task_gradient_norm': .01, 'first_comm_gradient_norm': .02, 'task_gradient_verified': True}
    assert epoch_diagnostic(row, {'lambda_comm': .05}) is None
    assert epoch_diagnostic(dict(row, optional_update_ratio=1.), {'lambda_comm': .05}) is None
    assert epoch_diagnostic(dict(row, total_loss=1.), {'lambda_comm': 0.}) is None
    assert epoch_diagnostic(dict(row, total_loss=float('nan')), {'lambda_comm': .05})
    assert epoch_diagnostic(dict(row, samples=1), {'lambda_comm': .05})
    print('Refresh2Track mode/branch-lock selection, strict selection/coverage/finite gradients/extreme-ratio recording self-check PASS')


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--root', type=Path, default=HERE.parents[1])
    command.add_argument('--mode', choices=runtime.MODES, help='run this branch independently using already prepared all-three gates')
    command.add_argument('--phase', choices=('all', 'preflight', 'report'), default='all')
    command.add_argument('--resume', action='store_true')
    command.add_argument('--launch', action='store_true')
    command.add_argument('--self-check', action='store_true')
    return command


def main():
    command_parser = parser()
    args = command_parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.mode and args.phase == 'preflight':
        command_parser.error('--phase preflight prepares all three branches together; omit --mode')
    if args.launch:
        directory = args.root / 'logs/refresh2track_ablations'
        if args.mode:
            directory /= args.mode
        path = directory / 'coordinator.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-B', str(HERE / 'run.py'), '--root', str(args.root), '--phase', args.phase]
        if args.mode:
            command += ['--mode', args.mode]
        if args.resume:
            command.append('--resume')
        with path.open('a', encoding='utf-8', buffering=1) as output:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=dict(os.environ, PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1'))
        print(json.dumps({'status': 'LAUNCHED', 'mode': args.mode, 'pid': child.pid, 'log': str(path)}), flush=True)
        return
    execute(args)


if __name__ == '__main__':
    main()
