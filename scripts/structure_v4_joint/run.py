"""Run the single local C experiment, with mandatory gates and full validation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import runtime
LOCK_DESCRIPTOR = None


def state(root, **values):
    directory = runtime.experiment(root)
    path = directory / 'run_state.json'
    previous = runtime.read(path) if path.exists() else {}
    if values.get('status') in ('RUNNING', 'PREFLIGHT_PASS', 'COMPLETED'):
        previous.pop('error', None)
        previous.pop('traceback', None)
    values.update(utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), experiment=runtime.MODEL)
    values = dict(previous, **values)
    runtime.write(path, values)
    report = '# ' + runtime.MODEL + ' execution\n\n' + json.dumps(values, ensure_ascii=False, indent=2) + '\n'
    report += '\nDeployment: A_GBA/B_GATE on remote; C_GBA_GATE on local. See each process state.\n'
    report += '\nTraining is FP32, batch 4, seed 0, with a fresh Scheduler and original CoopTrack base.\n'
    (directory / 'reports').mkdir(parents=True, exist_ok=True)
    (directory / 'reports/FINAL.md').write_text(report, encoding='utf-8')
    comparison(root, values)
    print(json.dumps(values, allow_nan=False), flush=True)


def comparison(root, values):
    import csv
    shared = Path(root) / 'experiments/structure_v4_joint'
    rows = [{'method': 'V4 Full historical', 'status': 'HISTORICAL', 'best_epoch': 12,
             'AMOTA': .375331, 'mAP': .365131, 'IDS': 249,
             'feature_payload_BPS': 16501.53, 'serialized_BPS': 27258.58}]
    for method in ('A_GBA', 'B_GATE', 'C_GBA_GATE'):
        row = dict.fromkeys(rows[0], 'NA')
        path = shared / method / 'run_state.json'
        other = runtime.read(path) if path.exists() else {}
        row.update(method=method, status=values['status'] if method == runtime.MODEL else other.get('status', 'PENDING_OTHER_HOST'))
        if method == runtime.MODEL:
            row.update(best_epoch=values.get('best_epoch', 'NA'))
            best = runtime.experiment(root) / 'metrics/best_full_val.json'
            if best.exists():
                metrics = runtime.read(best)
                row.update({key: metrics.get(key, 'NA') for key in ('AMOTA', 'mAP', 'IDS', 'feature_payload_BPS', 'serialized_BPS')})
        rows.append(row)
    with (shared / 'COMPARE_THREE.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    header = list(rows[0])
    table = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |']
    table.extend('| ' + ' | '.join(str(row[key]) for key in header) + ' |' for row in rows)
    (shared / 'COMPARE_THREE.md').write_text('# Three structure experiments\n\n' + '\n'.join(table) +
        '\n\nOnly A/B/C are registered new training runs. Check each host process state for live status.\n'
        '\nComparison includes both new structures and original module fine-tuning.\n', encoding='utf-8')


def worker(root, script, arguments):
    directory = runtime.experiment(root)
    log = directory / 'logs' / ('preflight.log' if '--preflight' in arguments else script.replace('.py', '.log'))
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-B', str(HERE / script), '--root', str(root)] + arguments
    with log.open('a', encoding='utf-8', buffering=1) as stream:
        stream.write(json.dumps({'command': command, 'utc': datetime.now(timezone.utc).isoformat()}) + '\n')
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                       pass_fds=(LOCK_DESCRIPTOR,),
                       env=dict(os.environ, PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1'))


def execute(args):
    global LOCK_DESCRIPTOR
    import fcntl
    root = args.root.resolve()
    directory = runtime.experiment(root)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'coordinator.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        LOCK_DESCRIPTOR = lock.fileno()
        try:
            proof = directory / 'preflight/preflight.json'
            if not proof.exists():
                state(root, status='RUNNING', phase='preflight')
                worker(root, 'train.py', ['--preflight'])
            runtime.initialize(root)
            if runtime.read(proof).get('status') != 'PASS' or runtime.read(proof).get('identity') != runtime.identity(root):
                raise ValueError('Preflight is missing or incompatible')
            if args.phase == 'preflight':
                state(root, status='PREFLIGHT_PASS', phase='ready_for_training')
                return
            import torch
            latest = directory / 'checkpoints/latest.pth'
            completed, selection, history = 0, {}, []
            if latest.exists():
                if not args.resume:
                    raise ValueError('Use --resume for the existing C checkpoint')
                checkpoint = torch.load(str(latest), map_location='cpu')
                if checkpoint['identity'] != runtime.identity(root):
                    raise ValueError('Resume checkpoint computation identity changed')
                completed, selection = checkpoint['epoch'], checkpoint.get('selection', {})
            val_path = directory / 'metrics/epoch_validation.json'
            if val_path.exists():
                history = runtime.read(val_path)
            epochs = ([completed] if completed and completed % 3 == 0 and selection.get('last_validation_epoch') != completed else [])
            epochs += list(range(completed + 1, 37))
            if completed >= 12 and selection.get('patience', 0) >= 4:
                epochs = []
            for epoch in epochs:
                if epoch > completed:
                    state(root, status='RUNNING', phase='training', epoch=epoch, completed_epoch=completed)
                    worker(root, 'train.py', ['--epoch', str(epoch)])
                    completed = epoch
                    state(root, status='RUNNING', phase='epoch_complete', epoch=epoch, completed_epoch=completed)
                if epoch % 3:
                    continue
                candidate = latest.with_name('candidate.pth')
                shutil.copyfile(latest, candidate)
                state(root, status='RUNNING', phase='full_validation', epoch=epoch)
                worker(root, 'evaluate.py', ['--checkpoint', str(candidate), '--epoch', str(epoch), '--selection'])
                result = runtime.read(directory / 'validation' / ('epoch%03d' % epoch) / 'result.json')
                if result.get('sample_count') != 3316 or result.get('scene_count') != 21 or result.get('status') != 'PASS':
                    raise ValueError('Selection requires complete official full validation')
                improved = selection.get('best_score') is None or result['AMOTA'] > selection['best_score']
                selection.update(last_validation_epoch=epoch, patience=0 if improved else selection.get('patience', 0) + 1)
                if improved:
                    selection.update(best_score=result['AMOTA'], best_epoch=epoch)
                    shutil.copyfile(candidate, latest.with_name('best.pth'))
                    runtime.write(directory / 'metrics/best_full_val.json', result)
                history = [row for row in history if row['epoch'] != epoch] + [result]
                runtime.write(val_path, history)
                from train import write_csv
                write_csv(directory / 'metrics/epoch_validation.csv', history)
                checkpoint = torch.load(str(latest), map_location='cpu')
                checkpoint['selection'] = selection
                temporary = latest.with_suffix('.pth.writing')
                torch.save(checkpoint, str(temporary))
                temporary.replace(latest)
                state(root, status='RUNNING', phase='validation_complete', epoch=epoch,
                      best_epoch=selection['best_epoch'], AMOTA=result['AMOTA'])
                if epoch >= 12 and selection['patience'] >= 4:
                    break
            best = latest.with_name('best.pth')
            chosen = torch.load(str(best), map_location='cpu')
            state(root, status='RUNNING', phase='best_full_reevaluation', best_epoch=chosen['epoch'])
            worker(root, 'evaluate.py', ['--checkpoint', str(best), '--epoch', str(chosen['epoch'])])
            result = runtime.read(directory / 'validation/final/V4_seed0/result.json')
            runtime.write(directory / 'metrics/best_full_val.json', result)
            state(root, status='COMPLETED', phase='complete', completed_epoch=completed,
                  best_epoch=chosen['epoch'], AMOTA=result['AMOTA'], mAP=result['mAP'],
                  feature_payload_BPS=result['feature_payload_BPS'], serialized_BPS=result['serialized_BPS'])
        except Exception as error:
            state(root, status='PREFLIGHT_FAIL' if not (directory / 'preflight/preflight.json').exists() else 'FAILED',
                  phase='stopped', error=str(error), traceback=traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('all', 'preflight'), default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        log = runtime.experiment(args.root) / 'logs/coordinator.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-B', str(HERE / 'run.py'), '--root', str(args.root), '--phase', args.phase]
        if args.resume:
            command.append('--resume')
        with log.open('a', encoding='utf-8') as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       env=dict(os.environ, PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1'))
        print(json.dumps({'status': 'LAUNCHED', 'pid': process.pid, 'log': str(log)}), flush=True)
        return
    execute(args)


if __name__ == '__main__':
    main()
