"""Run the complete official experiment in the inherited Docker image."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

IMAGE = 'sha256:22197870c646dcd6221210f814fe3494218e3e3fb22c6030faa47afc6dc649f8'
CONTAINER = 'cooptrack-main'
LINUX_ROOT = '/mnt/d/CoopTrack_Audit'


def container(root):
    exists = subprocess.run(['docker', 'inspect', CONTAINER], capture_output=True, text=True)
    if exists.returncode == 0:
        info = json.loads(exists.stdout)[0]
        if info['Image'] != IMAGE:
            raise RuntimeError('Existing main container does not use the inherited image')
        if not info['State']['Running']:
            subprocess.run(['docker', 'start', CONTAINER], check=True, stdout=subprocess.DEVNULL)
        return
    command = ['docker', 'run', '-d', '--name', CONTAINER, '--gpus', 'all',
               '--shm-size=2g', '--workdir', LINUX_ROOT + '/code/CoopTrack',
               '--mount', 'type=bind,source=%s,target=%s' % (root, LINUX_ROOT)]
    for relative in ('datasets', 'checkpoints', 'data', 'data_infos', 'query_cache/val', 'artifacts/core'):
        command += ['--mount', 'type=bind,source=%s,target=%s,readonly' %
                    (root / relative, LINUX_ROOT + '/' + relative)]
    command += ['--mount', 'type=bind,source=%s,target=/data/spd_prepared,readonly' %
                (root / 'datasets/prepared/V2X-Seq-SPD-CoopTrack-full_v1')]
    command += ['--mount', 'type=bind,source=%s,target=/data/spd_raw,readonly' %
                (root / 'datasets/V2X-Seq-SPD-raw')]
    for variable in ('PYTHONUNBUFFERED=1', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=2',
                     'OPENBLAS_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'PYTHONHASHSEED=0'):
        command += ['--env', variable]
    command += [IMAGE, 'sleep', 'infinity']
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cleanup-script', type=Path, help='One-time reviewed PowerShell cleanup, after existing IDS jobs finish')
    args = parser.parse_args()
    root = args.root.resolve()
    state_path = root / 'experiments/main/metrics/run_state.json'
    state_path.parent.mkdir(parents=True, exist_ok=True)
    (root / 'logs/main').mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'completed': []}
    if state['completed'] and not args.resume:
        raise RuntimeError('Existing main run found; use --resume')

    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = state_path.with_suffix('.writing')
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(state_path)

    def run(name, script, options, log):
        state.update(status='RUNNING', phase=name)
        state.pop('error', None)
        save()
        (root / 'reports/MAIN_EXPERIMENT_FINAL.md').write_text(
            '# 主实验执行状态\n\n**RUNNING**\n\nPhase: `' + name +
            '`\n\n执行状态见 [run_state.json](../experiments/main/metrics/run_state.json)。'
            '缓存、训练和完整验证完成后生成最终结果；当前没有新增正式指标。\n', encoding='utf-8')
        command = ['docker', 'exec', CONTAINER, 'python', LINUX_ROOT + '/scripts/' + script,
                   '--root', LINUX_ROOT] + options
        print(json.dumps({'phase': name, 'status': 'RUNNING'}), flush=True)
        with (root / 'logs/main' / log).open('a', encoding='utf-8') as stream:
            stream.write('\n' + json.dumps({'phase': name, 'command': command}) + '\n')
            stream.flush()
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            code = process.wait()
        if code:
            raise RuntimeError('%s exited %d; see logs/main/%s' % (name, code, log))
        if name not in state['completed']:
            state['completed'].append(name)
        save()
        print(json.dumps({'phase': name, 'status': 'PASS'}), flush=True)

    try:
        if (root / 'stage2').exists():
            if args.cleanup_script is None or not args.cleanup_script.is_file():
                raise RuntimeError('One-time cleanup is not complete; pass the reviewed --cleanup-script')
            state.update(status='RUNNING', phase='waiting_for_existing_ids')
            save()
            while True:
                command = "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object {$_.CommandLine -match 'scripts[/\\\\]ids_diagnosis[/\\\\]run_diagnosis\\.py'} | Select-Object -ExpandProperty ProcessId"
                launcher = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                                          check=True, capture_output=True, text=True).stdout.strip()
                active = bool(launcher)
                for name in ('cooptrack-ids-gpu', 'cooptrack-ids-cpu'):
                    info = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
                    if info.returncode or not json.loads(info.stdout)[0]['State']['Running']:
                        continue
                    process = subprocess.run(['docker', 'top', name, '-eo', 'pid,args'],
                                             check=True, capture_output=True, text=True)
                    active = active or any(script in process.stdout
                                           for script in ('inference_probe.py', 'evaluate_runs.py'))
                if not active:
                    break
                time.sleep(30)
            state.update(phase='cleanup')
            save()
            cleanup = subprocess.run(['powershell.exe', '-NoProfile', '-File', str(args.cleanup_script.resolve())],
                                     capture_output=True, text=True)
            if cleanup.returncode:
                raise RuntimeError('Cleanup failed: ' + cleanup.stderr.strip())
            state['cleanup'] = 'PASS'
            save()
        for relative in ('data_infos/V2X-Seq-SPD-CoopTrack-full', 'query_cache/val', 'artifacts/core'):
            if not (root / relative).is_dir():
                raise RuntimeError('Canonical experiment input is missing: ' + relative)
        container(root)
        run('preflight', 'main_eval.py', ['--preflight'], 'train.log')
        run('full_train_cache', 'main_eval.py', ['--generate-train-cache', '--resume'], 'train.log')
        run('lock_train_budgets', 'main_eval.py', ['--collect-train-stats'], 'train.log')
        for variant in ('aptu', 'mlp', 'no_assoc'):
            run('train_' + variant, 'main_train.py',
                ['--variant', variant, '--epochs', '12', '--resume'], 'train.log')
        for method in ('periodic2', 'periodic4'):
            run('eval_' + method, 'main_eval.py', ['--method', method, '--resume'], 'eval.log')
        for method in ('delta', 'mlp', 'no_assoc', 'aptu'):
            for budget in ('B1', 'B2', 'B3', 'B4'):
                run('eval_' + method + '_' + budget, 'main_eval.py',
                    ['--method', method, '--budget', budget, '--resume'], 'eval.log')
        run('final_report', 'main_report.py', [], 'eval.log')
        state.update(status='MAIN_EXPERIMENT_COMPLETE', phase='complete')
        save()
    except Exception as exc:
        state.update(status='MAIN_EXPERIMENT_BLOCKED', error=str(exc))
        save()
        from main_report import write_blocked
        write_blocked(root, state)
        print(json.dumps(state), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
