"""Run the two preregistered remote experiments sequentially on one GPU."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def execute(root):
    directory = root / 'experiments/structure_v4_joint'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'remote_queue.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def save(**values):
            values.update(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat())
            temporary = directory / 'remote_queue.json.writing'
            temporary.write_text(json.dumps(values, indent=2), encoding='utf-8')
            temporary.replace(directory / 'remote_queue.json')
            print(json.dumps(values), flush=True)

        try:
            for phase in ('preflight', 'all'):
                for model in ('A_GBA', 'B_GATE'):
                    required = (16 if model == 'A_GBA' else 8) * 1024 ** 3
                    if phase == 'all' and shutil.disk_usage(root).free < required:
                        save(status='BLOCKED_STORAGE', phase=phase, experiment=model,
                             required_free_bytes=required, available_free_bytes=shutil.disk_usage(root).free)
                        return
                    save(status='RUNNING', phase=phase, experiment=model,
                         policy='one GPU, sequential; both preflights must pass before formal A then B')
                    environment = dict(os.environ, REFRESH2TRACK_STRUCTURE_MODEL=model,
                                       PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1',
                                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                    subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('run.py')),
                                    '--root', str(root), '--phase', phase], check=True, env=environment,
                                   pass_fds=(lock.fileno(),))
            save(status='COMPLETED', phase='complete')
        except Exception as error:
            save(status='FAILED', phase=phase, experiment=model, error=str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        log = args.root / 'experiments/structure_v4_joint/remote_queue.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8') as stream:
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                        '--root', str(args.root.resolve())], stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'status': 'LAUNCHED', 'pid': process.pid, 'log': str(log)}))
    else:
        execute(args.root.resolve())
