"""Reload only the V4 coordinator after its untouched GPU worker commits an epoch."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'experiments/v4_budget_free/audit/v3_retirement_handoff.json'


def process(pid):
    try:
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'state': fields[0], 'start_time': fields[19],
                'command': (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def write(row):
    pending = AUDIT.with_suffix('.tmp')
    pending.write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
    pending.replace(AUDIT)


def main():
    lock = json.loads((ROOT / 'experiments/v4_budget_free/run.lock').read_text())
    coordinator = process(lock['pid'])
    assert coordinator and coordinator['state'] in ('T', 't')
    assert coordinator['command'].startswith('python -B ' + str(ROOT / 'scripts/v4_budget_free/run.py') + ' ')
    prefix = str(ROOT / 'scripts/v4_budget_free/train.py')
    workers = [row for p in Path('/proc').iterdir() if p.name.isdigit()
               for row in [process(int(p.name))] if row and prefix in row['command'] and row['state'] != 'Z']
    assert len(workers) == 1, workers
    worker = workers[0]
    epoch = int(worker['command'].split('--epoch ')[1].split()[0])
    row = {'status': 'WAITING_FOR_COMPLETE_EPOCH', 'handoff_pid': os.getpid(),
           'coordinator': coordinator, 'worker': worker, 'epoch': epoch,
           'gpu_worker_interrupted': False}
    write(row)
    try:
        while True:
            current = process(worker['pid'])
            if not current or current['start_time'] != worker['start_time'] or current['state'] == 'Z':
                break
            time.sleep(5)
        state = json.loads((ROOT / 'experiments/v4_budget_free/metrics/training_state.json').read_text())
        assert state['status'] == 'EPOCH_COMPLETE' and state['completed_epoch'] == epoch, state
        checkpoint = ROOT / 'experiments/v4_budget_free/checkpoints/latest.pth'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == state['checkpoint_sha256']
        current = process(coordinator['pid'])
        if current and current['start_time'] == coordinator['start_time']:
            assert current['command'] == coordinator['command'] and current['state'] in ('T', 't')
            os.kill(coordinator['pid'], signal.SIGKILL)
        command = [sys.executable, '-B', str(ROOT / 'scripts/v4_budget_free/run.py'),
                   '--root', str(ROOT), '--inside', '--phase', 'all', '--resume', '--launch']
        new = subprocess.Popen(command, start_new_session=True)
        row.update(status='COORDINATOR_RESTARTED', new_coordinator_pid=new.pid,
                   completed_epoch=epoch, checkpoint_sha256=state['checkpoint_sha256'])
        write(row)
    except Exception as exc:
        row.update(status='BLOCKED', error=str(exc))
        write(row)
        raise


if __name__ == '__main__':
    main()
