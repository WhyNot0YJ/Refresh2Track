import importlib.util
import json
from pathlib import Path
import torch
import traceback

audit = Path('/mnt/d/CoopTrack_Audit')
def module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value
before = module(audit / 'docker/probes/pf_utils_before_40xx.py', 'pf_before')
after = module(audit / 'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/modules/pf_utils.py', 'pf_after')
xyz = torch.tensor([[1., 2., 3.], [-2., 4., 1.]])
r1 = torch.eye(3)
r2 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
t1 = torch.tensor([1., 2., 3.])
t2 = torch.tensor([0.5, 0.25, -0.5])
inputs = [xyz, r1, t1, r2, t2]
report = {}
try:
    before.xyz_ego_transformation(*[item.cuda() for item in inputs], None, False, False)
    torch.cuda.synchronize()
    report['before'] = 'PASS'
except Exception:
    report['before'] = 'CUSOLVER_FAIL'
    report['before_error'] = traceback.format_exc()
expected = before.xyz_ego_transformation(*inputs, None, False, False)
actual = after.xyz_ego_transformation(*[item.cuda() for item in inputs], None, False, False)
torch.cuda.synchronize()
report['after'] = 'PASS' if torch.allclose(expected, actual.cpu(), atol=1e-6, rtol=1e-6) else 'FAIL'
report['max_abs_error'] = float((expected - actual.cpu()).abs().max())
report['device'] = str(actual.device)
report['dtype'] = str(actual.dtype)
(audit / 'artifacts/core/docker_provenance/40xx_inverse_verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert report['after'] == 'PASS'
