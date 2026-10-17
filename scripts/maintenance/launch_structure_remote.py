"""Run unchanged A/B training concurrently, optionally adopting an active A."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def storage_action(free_bytes, paused):
    if not paused and free_bytes < 2 * 1024 ** 3:
        return 'pause'
    if paused and free_bytes >= 3 * 1024 ** 3:
        return 'resume'
    return 'wait'


def process_token(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None


def validate_process(root, pid, script, model=None):
    directory = Path('/proc') / str(pid)
    command = directory.joinpath('cmdline').read_bytes().decode().split('\0')
    environment = directory.joinpath('environ').read_bytes().split(b'\0')
    if str(script) not in command or '--root' not in command or command[command.index('--root') + 1] != str(root):
        raise ValueError('Process does not belong to this launcher/root: ' + str(pid))
    if model and ('REFRESH2TRACK_STRUCTURE_MODEL=' + model).encode() not in environment:
        raise ValueError('Process experiment differs: ' + str(pid))
    token = process_token(pid)
    if token is None or os.getpgid(pid) != pid:
        raise ValueError('Process must be alive in its own process group: ' + str(pid))
    return token


def execute(root, adopt_a_pid=None, replace_queue_pid=None):
    import fcntl
    directory = root / 'experiments/structure_v4_joint'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'remote_parallel.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def save(**values):
            values.update(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(),
                          mode='PARALLEL', experiment='A_GBA+B_GATE', phase='all',
                          storage_policy='user approved: no 16/8 GiB gate; pause below 2 GiB, resume at 3 GiB',
                          available_free_bytes=shutil.disk_usage(root).free)
            temporary = directory / 'remote_queue.json.writing'
            temporary.write_text(json.dumps(values, indent=2), encoding='utf-8')
            temporary.replace(directory / 'remote_queue.json')
            print(json.dumps(values), flush=True)

        children, tokens, owned, paused = {}, {}, {}, set()
        try:
            for model in ('A_GBA', 'B_GATE'):
                proof = directory / model / 'preflight/preflight.json'
                if not proof.exists() or json.loads(proof.read_text()).get('status') != 'PASS':
                    raise ValueError('Both independent preflights must already pass: ' + model)
                if not (model == 'A_GBA' and adopt_a_pid) and (directory / model / 'checkpoints/latest.pth').exists():
                    raise ValueError('Use explicit checkpoint resume instead of a fresh launch: ' + model)
            if adopt_a_pid:
                tokens['A_GBA'] = validate_process(root, adopt_a_pid,
                    root / 'scripts/structure_v4_joint/run.py', 'A_GBA')
                children['A_GBA'] = adopt_a_pid
            if replace_queue_pid:
                if not adopt_a_pid or replace_queue_pid == adopt_a_pid:
                    raise ValueError('Replacing a queue requires its independently running A child')
                previous = json.loads((directory / 'remote_queue.json').read_text())
                if previous.get('pid') != replace_queue_pid or previous.get('child_pid') != adopt_a_pid:
                    raise ValueError('Previous queue/child ownership differs')
                previous_token = validate_process(root, replace_queue_pid, Path(__file__).resolve())
                (directory / 'sequential_queue_handoff.json').write_text(json.dumps(previous, indent=2))
                os.kill(replace_queue_pid, signal.SIGTERM)
                deadline = time.monotonic() + 10
                while process_token(replace_queue_pid) == previous_token:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Previous queue did not stop; A was not signalled')
                    time.sleep(.1)
            for model in ('A_GBA', 'B_GATE'):
                if model in children:
                    continue
                if shutil.disk_usage(root).free < 2 * 1024 ** 3:
                    raise RuntimeError('Less than 2 GiB free before parallel launch')
                environment = dict(os.environ, REFRESH2TRACK_STRUCTURE_MODEL=model,
                                   PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1',
                                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                command = [sys.executable, '-B', str(root / 'scripts/structure_v4_joint/run.py'),
                           '--root', str(root), '--phase', 'all']
                process = subprocess.Popen(command, env=environment, start_new_session=True,
                                           pass_fds=(lock.fileno(),))
                owned[model], children[model] = process, process.pid
                tokens[model] = process_token(process.pid)
                if tokens[model] is None:
                    raise RuntimeError('Child exited during startup: ' + model)
            while True:
                states, active = {}, []
                for model, pid in children.items():
                    if model in owned:
                        owned[model].poll()
                    path = directory / model / 'run_state.json'
                    states[model] = json.loads(path.read_text()) if path.exists() else {}
                    if process_token(pid) == tokens[model]:
                        active.append(model)
                    elif states[model].get('status') != 'COMPLETED':
                        states[model].update(status='FAILED', error='Coordinator exited before completion')
                if not active:
                    save(status='COMPLETED' if all(value.get('status') == 'COMPLETED' for value in states.values()) else 'FAILED',
                         children=children, states=states)
                    break
                for model in active:
                    action = storage_action(shutil.disk_usage(root).free, model in paused)
                    if action in ('pause', 'resume'):
                        pid = children[model]
                        if process_token(pid) != tokens[model] or os.getpgid(pid) != pid:
                            continue
                        os.killpg(pid, signal.SIGSTOP if action == 'pause' else signal.SIGCONT)
                        if action == 'pause':
                            paused.add(model)
                        else:
                            paused.discard(model)
                save(status='PAUSED_LOW_STORAGE' if paused else 'RUNNING', children=children,
                     states=states, active=active, paused=sorted(paused))
                time.sleep(5)
        except Exception as error:
            save(status='SUPERVISOR_FAILED', children=children, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--adopt-a-pid', type=int)
    parser.add_argument('--replace-queue-pid', type=int)
    args = parser.parse_args()
    if args.self_check:
        for free_gib, paused, expected in ((1, False, 'pause'), (2, False, 'wait'),
                                           (2, True, 'wait'), (3, True, 'resume'),
                                           (13, False, 'wait')):
            assert storage_action(free_gib * 1024 ** 3, paused) == expected
        print('PASS: storage policy boundaries; no training code or settings modified')
        return
    if args.root is None:
        parser.error('--root is required for launching')
    root = args.root.resolve()
    if args.launch:
        log = root / 'experiments/structure_v4_joint/remote_parallel.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8') as stream:
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--root', str(root)]
            if args.adopt_a_pid:
                command += ['--adopt-a-pid', str(args.adopt_a_pid)]
            if args.replace_queue_pid:
                command += ['--replace-queue-pid', str(args.replace_queue_pid)]
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'status': 'LAUNCHED', 'pid': process.pid, 'log': str(log)}))
    else:
        execute(root, args.adopt_a_pid, args.replace_queue_pid)


if __name__ == '__main__':
    main()
