"""Verify merge-compatible resume and rejection of later source/HEAD drift."""
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts/v4_budget_free'))
import runtime
import run

record = runtime.read(runtime.experiment(ROOT) / 'audit/git_merge_resume.json')
manifest = runtime.experiment(ROOT) / 'manifest.json'
latest = runtime.experiment(ROOT) / 'checkpoints/latest.pth'
best = latest.with_name('best.pth')
before = {p: runtime.digest(p) for p in (manifest, latest, best)}
original_commit = runtime.native.COMMIT
runtime.initialize(ROOT)
assert runtime.native.COMMIT == original_commit == record['original_commit']
assert runtime.identity(ROOT)['manifest_sha256'] == record['v4_manifest_sha256']
run.runtime = runtime
state = run.checkpoint_load(ROOT, latest, 17)
assert state['training_result']['completed_epoch'] == 17 and state['lr_progress'] == 17.0
assert len(state['optimizer']['state']) == 12 and set(state['rng']) == {'python', 'numpy', 'torch', 'cuda'}
assert run.checkpoint_load(ROOT, best, 12)['epoch'] == 12

real_digest = runtime.digest
changed = ROOT / 'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/detectors/cooptrack.py'
with patch.object(runtime, 'digest', side_effect=lambda p: 'changed' if Path(p) == changed else real_digest(p)):
    try:
        runtime.initialize(ROOT)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed frozen native computation was accepted')
with patch.object(runtime.subprocess, 'check_output', return_value='0' * 40):
    try:
        runtime.initialize(ROOT)
    except ValueError:
        pass
    else:
        raise AssertionError('An unapproved Git HEAD was accepted')
with patch.object(runtime.native, 'initialize', side_effect=RuntimeError('constructed init failure')):
    try:
        runtime.initialize(ROOT)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Initialization failure did not propagate')
assert runtime.native.COMMIT == original_commit
assert before == {p: runtime.digest(p) for p in before}
print('PASS: exact merged HEAD initializes; epoch17 optimizer/LR/RNG/identity retained; source/HEAD drift rejected; shared commit and checkpoint bytes preserved')
