"""Verify that V3 cleanup preserves V4 computation and rejects source drift."""
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts/v4_budget_free'))
import runtime
import native_hooks

record = runtime.retirement(ROOT)
assert record and record['computation_change'] is False
manifest = runtime.read(runtime.experiment(ROOT) / 'manifest.json')
assert runtime.source_identity(ROOT) == manifest['source_hashes']
assert runtime.identity(ROOT)['manifest_sha256'] == record['v4_manifest_sha256']
assert native_hooks.self_check()['status'] == 'PASS'
assert runtime.verify_protection(ROOT)['status'] == 'PASS'
assert runtime.freeze_manifest(ROOT) == manifest
assert all(not (ROOT / row['path']).exists() for row in record['deletion_scopes'])
real_digest = runtime.digest
changed = ROOT / 'scripts/v4_budget_free/scheduler.py'
with patch.object(runtime, 'digest', side_effect=lambda p: 'changed' if Path(p) == changed else real_digest(p)):
    try:
        runtime.identity(ROOT)
    except ValueError:
        pass
    else:
        raise AssertionError('Training computation drift was accepted')
with patch.object(runtime, 'digest', side_effect=lambda p: 'changed' if Path(p) == ROOT / 'scripts/v4_budget_free/train.py' else real_digest(p)):
    try:
        runtime.identity(ROOT)
    except ValueError:
        pass
    else:
        raise AssertionError('Maintenance source drift was accepted')
print('PASS: unchanged V4 identity, helper arithmetic, shared assets, resume freeze; source drift rejected')
