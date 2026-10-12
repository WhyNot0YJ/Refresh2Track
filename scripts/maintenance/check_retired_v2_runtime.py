"""Check protected imports and constructed contracts without loading H code or data."""
import importlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys


class RejectH(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + '.')
               for prefix in ('budget_free', 'scripts.budget_free')):
            raise ImportError('Retired H runtime import forbidden: ' + fullname)


def main():
    if not sys.dont_write_bytecode:
        raise RuntimeError('Run this check with python -B')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'scripts'))
    sys.meta_path.insert(0, RejectH())
    modules = ('run_main', 'main_runtime', 'main_message', 'main_scheduler', 'main_eval',
               'main_train', 'main_report', 'v2_message', 'v2_scheduler', 'v2_runtime',
               'v2_prepare', 'v2_train', 'v2_eval', 'v2_report', 'v2_run', 'refresh_ablation',
               'refresh_ablation_report', 'refresh_ablation_protection')
    for name in modules:
        importlib.import_module(name)
    checks = ('main_message', 'v2_message', 'v2_scheduler')
    for name in checks:
        importlib.import_module(name).self_check()
    for script in ('run_main.py', 'v2_run.py', 'refresh_ablation.py'):
        subprocess.run([sys.executable, '-B', str(root / 'scripts' / script), '--help'],
                       check=True, capture_output=True, text=True, timeout=10)
    assert not any(name == 'budget_free' or name.startswith('budget_free.')
                   or name == 'scripts.budget_free' or name.startswith('scripts.budget_free.')
                   for name in sys.modules)
    print(json.dumps(dict(status='PASS', imported_modules=modules, self_checks=checks,
                          retired_import_guard='PASS', cli_help='PASS', bytecode_disabled=True,
                          native_initialize_called=False, training_started=False,
                          official_val_started=False)))


if __name__ == '__main__':
    main()
