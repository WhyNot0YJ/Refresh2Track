"""Run the single preregistered V4 model and six paired inference evaluations."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone

MODEL = 'v4_budget_free_mlp_lambda005_seed0'
SEEDS = (0, 2025, 2026)
runtime = None


def log_event(root, **values):
    values['utc'] = datetime.now(timezone.utc).isoformat()
    values['model'] = MODEL
    print(json.dumps(values, allow_nan=False), flush=True)
    path = runtime.experiment(root) / 'run_state.json'
    previous = runtime.read(path) if path.exists() else {}
    runtime.write(path, dict(previous, **values))


def worker(root, script, options=(), phase='train'):
    path = Path(root) / 'logs/v4_budget_free' / (phase + '.log')
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-B', str(Path(root) / 'scripts/v4_budget_free' / script), '--root', str(root)] + list(options)
    environment = dict(os.environ, PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')
    with path.open('a', encoding='utf-8', buffering=1) as output:
        output.write(json.dumps({'command': command, 'utc': datetime.now(timezone.utc).isoformat()}) + '\n')
        subprocess.run(command, check=True, env=environment, stdout=output, stderr=subprocess.STDOUT)


def write_history(path, rows):
    runtime.write(path, rows)
    if rows:
        keys = list(dict.fromkeys(key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))))
        temporary = path.with_suffix('.csv.writing')
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path.with_suffix('.csv'))


def selection_update(previous, score, epoch):
    if not math.isfinite(score):
        raise ValueError('Selection requires finite complete raw AMOTA')
    improved = previous.get('best_score') is None or score > previous['best_score']
    current = dict(previous, patience=0 if improved else previous.get('patience', 0) + 1,
                   last_validation_epoch=epoch, last_score=score)
    if improved:
        current.update(best_score=score, best_epoch=epoch)
    return current, improved


def checkpoint_load(root, path, epoch=None):
    import torch
    state = torch.load(str(path), map_location='cpu')
    if state.get('identity') != runtime.identity(root, MODEL):
        raise ValueError('Checkpoint source/config/seed/weight identity differs: ' + str(path))
    if epoch is not None and state.get('epoch') != epoch:
        raise ValueError('Checkpoint does not contain the required completed epoch')
    required = ('scheduler', 'optimizer', 'rng', 'lr_progress', 'training_result')
    if any(key not in state for key in required) or state['lr_progress'] != float(state['epoch']):
        raise ValueError('Checkpoint lacks completed optimizer/LR/RNG state')
    epochs = state['training_result'].get('epochs', [])
    if len(epochs) != state['epoch'] or [row.get('epoch') for row in epochs] != list(range(1, state['epoch'] + 1)):
        raise ValueError('Checkpoint epoch and complete history disagree')
    return state


def checkpoint_save(path, state):
    import torch
    temporary = path.with_suffix('.pth.writing')
    torch.save(state, str(temporary))
    temporary.replace(path)


def scheduler_hash(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state['scheduler'].items()):
        value.update(name.encode())
        value.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def require_result(root, path, checkpoint, method, epoch, seed):
    row = runtime.read(path)
    expected = {'status': 'PASS', 'sample_count': 3316, 'scene_count': 21,
                'checkpoint_sha256': runtime.digest(checkpoint), 'method': method, 'epoch': epoch, 'seed': seed}
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError('Incomplete or mismatched evaluation: ' + str(path))
    for name in ('AMOTA', 'mAP', 'feature_payload_BPS', 'serialized_BPS'):
        if not isinstance(row.get(name), (float, int)) or not math.isfinite(row[name]):
            raise ValueError('Missing finite evaluation metric ' + name)
    return row


def validate_epoch(root, epoch):
    directory = runtime.experiment(root)
    metrics = directory / 'metrics'
    history_path = metrics / 'val_history.json'
    histories = runtime.read(history_path) if history_path.exists() else []
    selection_path = metrics / 'best_selection.json'
    previous = runtime.read(selection_path) if selection_path.exists() else {}
    latest = directory / 'checkpoints/latest.pth'
    state = checkpoint_load(root, latest, epoch)
    completed = [row for row in histories if row['epoch'] == epoch]
    if completed:
        if len(completed) != 1 or previous.get('last_validation_epoch') != epoch:
            raise ValueError('Incomplete validation selection transaction')
        if state.get('selection') != previous:
            state['selection'] = previous
            checkpoint_save(latest, state)
        candidate = latest.with_name('candidate.pth')
        if candidate.exists():
            if runtime.digest(candidate) != completed[0]['checkpoint_sha256']:
                raise ValueError('Unexpected interrupted validation candidate')
            candidate.unlink()
        return previous
    candidate = latest.with_name('candidate.pth')
    if not candidate.exists():
        shutil.copyfile(latest, candidate)
    candidate_state = checkpoint_load(root, candidate, epoch)
    log_event(root, status='RUNNING', phase='selection_validation', epoch=epoch, seed=0)
    worker(root, 'evaluate.py', ['--method', 'V4', '--checkpoint', str(candidate), '--epoch', str(epoch), '--seed', '0', '--selection'], 'eval')
    row = require_result(root, directory / 'validation' / ('epoch%03d' % epoch) / 'result.json', candidate, 'V4', epoch, 0)
    replay = {}
    for historic in sorted(histories, key=lambda item: item['epoch']):
        replay, _ = selection_update(replay, historic['AMOTA'], historic['epoch'])
    current, improved = selection_update(dict(previous, **replay), row['AMOTA'], epoch)
    current['status'] = 'RUNNING'
    if improved:
        # Commit current selection into best while preserving scheduler/optimizer/RNG.
        # The ledger binds both evaluation candidate and resulting checkpoint hashes.
        best = latest.with_name('best.pth')
        selected = dict(candidate_state)
        selected['selection'] = {key: current[key] for key in ('best_score', 'best_epoch', 'patience', 'last_validation_epoch', 'last_score', 'status')}
        checkpoint_save(best, selected)
        if scheduler_hash(selected) != scheduler_hash(candidate_state):
            raise ValueError('Selection metadata changed scheduler weights')
        current.update(best_checkpoint=str(best), checkpoint_sha256=runtime.digest(best),
                       validation_checkpoint_sha256=row['checkpoint_sha256'],
                       scheduler_state_sha256=scheduler_hash(selected), best_results=row,
                       selection_metadata_commit_only=True)
    runtime.write(selection_path, current)
    write_history(history_path, histories + [row])
    state['selection'] = current
    checkpoint_save(latest, state)
    if runtime.digest(candidate) != row['checkpoint_sha256']:
        raise ValueError('Validation candidate changed during selection commit')
    candidate.unlink()
    return current


def epoch_diagnostic(row):
    for name in ('task_loss', 'comm_loss', 'total_loss', 'scheduler_gradient_norm'):
        if not isinstance(row.get(name), (float, int)) or not math.isfinite(row[name]):
            return 'Missing or nonfinite epoch ' + name
    task_norm = row.get('first_task_gradient_norm', row.get('task_gradient_norm'))
    comm_norm = row.get('first_comm_gradient_norm', row.get('comm_gradient_norm'))
    if not all(isinstance(value, (float, int)) and math.isfinite(value) and value > 0 for value in (task_norm, comm_norm)):
        return 'Missing finite nonzero native task/communication gradient proof'
    if (row.get('non_finite_batch_count', 0) or row['scheduler_gradient_norm'] <= 0 or not row.get('task_gradient_verified')):
        return 'Nonfinite or vanished native task/Scheduler gradient'
    ratio = row.get('optional_update_ratio')
    if not isinstance(ratio, (float, int)) or not math.isfinite(ratio):
        return 'Missing finite actual optional UPDATE ratio'
    if ratio <= 0.001 or ratio >= 0.999:
        return 'Degenerate actual optional UPDATE ratio: ' + str(ratio)
    return None


def preflight(root):
    directory = runtime.experiment(root)
    identity = runtime.identity(root, MODEL)
    path = directory / 'audit/preflight.json'
    if path.exists():
        old = runtime.read(path)
        if old.get('status') == 'PASS' and old.get('identity') == identity:
            print('V4_PREFLIGHT_PASS', flush=True)
            return
        raise ValueError('Existing preflight differs or failed; preserve evidence and stop')
    log_event(root, status='RUNNING', phase='preflight')
    worker(root, 'scheduler.py', ['--self-check'], 'cpu')
    cpu = runtime.read(directory / 'audit/cpu_preflight.json')
    if cpu.get('status') != 'PASS' or cpu.get('model') != MODEL:
        raise ValueError('CPU scheduler/protocol preflight failed')
    proof_path = directory / 'metrics/correctness.json'
    if not proof_path.exists():
        worker(root, 'train.py', ['--model', MODEL, '--correctness-check'])
    native = runtime.read(proof_path)
    if native.get('status') != 'PASS' or native.get('identity') != identity:
        raise ValueError('Matching native short-rollout gradient proof required')
    for check in ('native_task_gradient_nonzero', 'communication_gradient_nonzero',
                  'native_parameters_and_BN_frozen', 'real_packet_INT8_UPDATE_REUSE',
                  'hard_STE_equals_actual_decoded', 'lane_cache_rng_isolation'):
        if native.get(check) is not True:
            raise ValueError('Native preflight did not establish ' + check)
    protection = runtime.verify_protection(root)
    if protection.get('status') != 'PASS':
        raise ValueError('V2/V3 preservation failed')
    cpu_path = str(directory / 'audit/cpu_preflight.json')
    native_path = str(proof_path)
    inherited = cpu.get('inherited_protocol_checks', {})
    boundary = cpu.get('hard_threshold_boundary', {})
    assertions = {
        'V3_read_only_protection': protection['status'] == 'PASS',
        'budget_input_removed_2324_state': cpu.get('budget_input_removed') is True and cpu.get('state_dimension') == 2324
            and cpu.get('forward_arguments') == ['tokens', 'valid_mask'],
        'hard_forward_threshold_and_receiver_reuse': boundary.get('probability') == [0.49, 0.5, 0.51]
            and boundary.get('flags') == [False, True, True] and cpu.get('real_decoded_hard_ST_forward') is True
            and native.get('hard_STE_equals_actual_decoded') is True,
        'communication_increment_exactly_780B': cpu.get('actual_packet_increment_bytes') == 780
            and cpu.get('latent_payload_increment_bytes') == 780,
        'forced_initialization_reset_ID_alignment_empty_queries': cpu.get('missing_cache_forces_update') is True
            and all(inherited.get(key) is True for key in ('session_scene_reset', 'id_permutation_cache_alignment', 'empty_and_padded_queries')),
        'real_native_task_and_communication_gradients': native.get('native_task_gradient_nonzero') is True
            and native.get('communication_gradient_nonzero') is True
            and all(math.isfinite(native.get(key, float('nan'))) and native.get(key, 0) > 0 for key in ('task_gradient_norm', 'comm_gradient_norm')),
        'frozen_native_model_and_no_future_information': native.get('native_parameters_and_BN_frozen') is True
            and native.get('future_reasoning_disabled') is True,
        'reference_results_and_protocol_unchanged': protection['status'] == 'PASS' and inherited.get('status') == 'PASS',
    }
    if not all(assertions.values()):
        runtime.write(directory / 'audit/preflight_gate_failure.json', {'status': 'FAIL', 'assertions': assertions})
        raise ValueError('A required V4 preflight gate failed: ' + str([key for key, passed in assertions.items() if not passed]))
    commands = [[sys.executable, '-B', str(Path(root) / 'scripts/v4_budget_free/scheduler.py'), '--root', str(root), '--self-check'],
                [sys.executable, '-B', str(Path(root) / 'scripts/v4_budget_free/train.py'), '--root', str(root), '--model', MODEL, '--correctness-check']]
    evidence = [cpu_path, native_path, str(directory / 'audit/protection_before_v4.json'),
                str(Path(root) / 'logs/v4_budget_free/cpu.log'), str(Path(root) / 'logs/v4_budget_free/train.log')]
    runtime.write(path, {'status': 'PASS', 'identity': identity, 'cpu_proof': str(directory / 'audit/cpu_preflight.json'),
                        'native_proof': str(proof_path), 'protection': protection,
                        'commands': commands, 'python_api_calls': ['runtime.verify_protection(root)'],
                        'assertions': assertions, 'evidence': evidence,
                        'gates': [{'id': index, 'assertion': key, 'passed': passed, 'evidence': evidence}
                                  for index, (key, passed) in enumerate(assertions.items(), 1)],
                        'reference_gate_scope': 'Protected FULL/Periodic results, source/config/weight hashes and inherited CPU protocol checks; full paired P16 inference runs follow training.',
                        'utc': datetime.now(timezone.utc).isoformat(), 'training_started': False})
    print('V4_PREFLIGHT_PASS', flush=True)
    log_event(root, status='PREFLIGHT_PASS', phase='preflight_complete')


def no_v3_writers():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ')
            stat = (proc / 'stat').read_text()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if stat[stat.rfind(')') + 2:].split()[0] != 'Z' and any(name in command for name in
                (b'scripts/v3_budget/run.py', b'scripts/v3_budget/train.py', b'scripts/v3_budget/evaluate.py')):
            found.append({'pid': int(proc.name), 'command': command.decode(errors='replace')})
    if found:
        raise RuntimeError('WAIT_FOR_V3_PROCESS_CONFIRMATION: ' + json.dumps(found))


def acquire_lock(path, resume):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        previous = runtime.read(path)
        command = Path('/proc') / str(previous['pid']) / 'cmdline'
        if command.exists() and b'v4_budget_free/run.py' in command.read_bytes():
            raise RuntimeError('A V4 coordinator is already running')
        if not resume:
            raise RuntimeError('Stale V4 lock; explicit --resume is required')
        path.unlink()
        descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump({'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat()}, stream)


def inside(args):
    root = args.root.resolve()
    if not Path('/.dockerenv').is_file():
        raise RuntimeError('V4 GPU work requires the pinned inherited Docker container')
    if args.phase == 'report':
        worker(root, 'report.py', [], 'report')
        return
    no_v3_writers()
    directory = runtime.experiment(root)
    lock = directory / 'run.lock'
    acquire_lock(lock, args.resume)
    preflight_passed = False
    try:
        log_event(root, status='RUNNING', phase='provenance_freeze')
        runtime.freeze_manifest(root)
        settings = runtime.load_settings(root, MODEL)
        checkpoints = directory / 'checkpoints'
        latest = checkpoints / 'latest.pth'
        if latest.exists() and not args.resume:
            raise ValueError('Existing completed V4 epoch requires --resume; weights cannot be reused as from-scratch')
        preflight(root)
        preflight_passed = True
        worker(root, 'report.py', [], 'report')
        if args.phase == 'preflight':
            return
        state = checkpoint_load(root, latest) if latest.exists() else None
        completed = state['epoch'] if state else 0
        selection_path = directory / 'metrics/best_selection.json'
        selection = runtime.read(selection_path) if selection_path.exists() else {}
        if state:
            write_history(directory / 'metrics/train_history.json', state['training_result']['epochs'])
            reason = epoch_diagnostic(state['training_result']['epochs'][-1])
            if reason:
                raise RuntimeError(reason)
        if completed and completed % settings['validation_interval'] == 0:
            selection = validate_epoch(root, completed)
        stopped = completed >= settings['minimum_stop_epoch'] and selection.get('patience', 0) >= settings['patience']
        for epoch in range(completed + 1, settings['max_epochs'] + 1):
            if stopped:
                break
            no_v3_writers()
            log_event(root, status='RUNNING', phase='training', epoch=epoch)
            if epoch == 1:
                worker(root, 'report.py', [], 'report')
            worker(root, 'train.py', ['--model', MODEL, '--epoch', str(epoch)])
            state = checkpoint_load(root, latest, epoch)
            completed = epoch
            reason = epoch_diagnostic(state['training_result']['epochs'][-1])
            if reason:
                runtime.write(directory / 'audit/training_stop_diagnostic.json', {'status': 'BLOCKED', 'epoch': epoch,
                    'reason': reason, 'epoch_metrics': state['training_result']['epochs'][-1],
                    'checkpoint_sha256': runtime.digest(latest), 'automatic_retry': False})
                raise RuntimeError(reason)
            worker(root, 'report.py', [], 'report')
            if epoch % settings['validation_interval'] == 0:
                selection = validate_epoch(root, epoch)
                worker(root, 'report.py', [], 'report')
                stopped = epoch >= settings['minimum_stop_epoch'] and selection.get('patience', 0) >= settings['patience']
        if selection.get('best_epoch') is None:
            raise ValueError('Cannot finish training without a selected complete validation')
        selection.update(status='PASS', stop_epoch=completed, stop_reason='early_stopping' if stopped else 'max_epochs_36')
        state = checkpoint_load(root, latest, completed)
        state['selection'] = selection
        state['training_result'].update(status='PASS', stop_epoch=completed, stop_reason=selection['stop_reason'])
        checkpoint_save(latest, state)
        checkpoint_save(checkpoints / 'last.pth', state)
        runtime.write(selection_path, selection)
        runtime.write(directory / 'metrics/training_state.json', state['training_result'])
        best = checkpoints / 'best.pth'
        chosen = checkpoint_load(root, best, selection['best_epoch'])
        if runtime.digest(best) != selection['checkpoint_sha256'] or scheduler_hash(chosen) != selection['scheduler_state_sha256']:
            raise ValueError('Selected best checkpoint changed')
        final_rows = []
        for seed in SEEDS:
            for method in ('V4', 'P16'):
                no_v3_writers()
                log_event(root, status='RUNNING', phase='final_validation', method=method, epoch=selection['best_epoch'], seed=seed)
                worker(root, 'evaluate.py', ['--method', method, '--checkpoint', str(best),
                    '--epoch', str(selection['best_epoch']), '--seed', str(seed)], 'eval')
                final_rows.append(require_result(root, directory / 'validation/final' / ('%s_seed%d' % (method, seed)) / 'result.json',
                    best, method, selection['best_epoch'], seed))
        for seed in SEEDS:
            pair = [row for row in final_rows if row['seed'] == seed]
            if len(pair) != 2 or not pair[0].get('pairing') or pair[0]['pairing'] != pair[1].get('pairing'):
                raise ValueError('V4/P16 protocol/cache/RNG pairing differs for seed ' + str(seed))
        protection = runtime.verify_protection(root)
        if protection.get('status') != 'PASS' or len(final_rows) != 6:
            raise ValueError('Final six-run validation or V2/V3 preservation failed')
        runtime.write(directory / 'audit/protection_final.json', protection)
        runtime.write(directory / 'metrics/final_validation.json', {'status': 'PASS', 'rows': final_rows,
            'training_seed': 0, 'inference_seeds': list(SEEDS), 'checkpoint_sha256': runtime.digest(best),
            'official_val_use': 'checkpoint selection and reporting; not independent test generalization'})
        log_event(root, status='COMPLETED', phase='complete', best_epoch=selection['best_epoch'], checkpoint=str(best), completed_epoch=completed)
        worker(root, 'report.py', [], 'report')
    except Exception as exc:
        if not preflight_passed:
            print('V4_PREFLIGHT_FAIL', flush=True)
            runtime.write(directory / 'audit/preflight_failure.json', {'status': 'FAIL', 'error': str(exc),
                'utc': datetime.now(timezone.utc).isoformat(), 'automatic_retry': False,
                'commands': [[sys.executable, '-B', str(root / 'scripts/v4_budget_free/scheduler.py'), '--root', str(root), '--self-check'],
                             [sys.executable, '-B', str(root / 'scripts/v4_budget_free/train.py'), '--root', str(root), '--model', MODEL, '--correctness-check']],
                'evidence': [str(directory / 'audit/cpu_preflight.json'), str(directory / 'metrics/correctness.json'),
                             str(directory / 'audit/preflight_gate_failure.json'), str(root / 'logs/v4_budget_free/cpu.log'),
                             str(root / 'logs/v4_budget_free/train.log')]})
        log_event(root, status='BLOCKED', phase='stopped', error=str(exc), traceback=traceback.format_exc(), automatic_retry=False)
        try:
            worker(root, 'report.py', [], 'report')
        except Exception:
            pass
        raise
    finally:
        if lock.exists() and runtime.read(lock).get('pid') == os.getpid():
            lock.unlink()


def self_check():
    state, improved = selection_update({}, 0.7, 3)
    assert improved and state['best_epoch'] == 3
    for epoch in (6, 9, 12, 15):
        state, improved = selection_update(state, 0.7, epoch)
        assert not improved and state['best_epoch'] == 3
    assert state['patience'] == 4
    state, improved = selection_update(state, 0.700000000000001, 18)
    assert improved and state['patience'] == 0
    row = dict(task_loss=1., comm_loss=0.1, total_loss=1.1, task_gradient_norm=0.01,
               comm_gradient_norm=0.02, scheduler_gradient_norm=0.02, optional_update_ratio=0.1,
               task_gradient_verified=True)
    assert epoch_diagnostic(row) is None
    assert epoch_diagnostic(dict(row, optional_update_ratio=0.001))
    assert epoch_diagnostic(dict(row, optional_update_ratio=0.999))
    assert epoch_diagnostic(dict(row, task_gradient_norm=0.))
    assert epoch_diagnostic(dict(row, total_loss=float('nan')))
    assert epoch_diagnostic(dict(row, task_gradient_norm=0., comm_gradient_norm=0.,
                                first_task_gradient_norm=0.01, first_comm_gradient_norm=0.02)) is None
    print('V4 coordinator strict AMOTA/ties/patience/degeneracy/gradient self-check PASS')


def main():
    global runtime
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--phase', choices=('all', 'preflight', 'report'), default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.inside:
        import runtime as runtime_module
        runtime = runtime_module
        if args.launch:
            log = args.root / 'logs/v4_budget_free/coordinator.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a', encoding='utf-8', buffering=1) as stream:
                sys.stdout, sys.stderr = stream, stream
                inside(args)
        else:
            inside(args)
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_main import CONTAINER, IMAGE, LINUX_ROOT
    info = json.loads(subprocess.check_output(['docker', 'inspect', CONTAINER], universal_newlines=True, timeout=20))[0]
    if info['Image'] != IMAGE or not info['State']['Running']:
        raise ValueError('The pinned running CoopTrack container is required')
    command = ['docker', 'exec'] + (['-d'] if args.launch else [])
    command += ['-e', 'PYTHONHASHSEED=0', '-e', 'PYTHONDONTWRITEBYTECODE=1', CONTAINER,
        'python', '-B', LINUX_ROOT + '/scripts/v4_budget_free/run.py', '--root', LINUX_ROOT, '--inside', '--phase', args.phase]
    command += (['--resume'] if args.resume else []) + (['--launch'] if args.launch else [])
    subprocess.run(command, check=True)
    if args.launch:
        print('V4 coordinator launched; progress: experiments/v4_budget_free/run_state.json', flush=True)


if __name__ == '__main__':
    main()
