"""Actual import and CUDA extension gate, not inference or training."""
import importlib
import json
import platform
from pathlib import Path
import sys
import traceback

audit = Path('/mnt/d/CoopTrack_Audit')
sys.path.insert(0, str(audit / 'code/CoopTrack'))
report = {'python': platform.python_version(), 'imports': {}, 'cuda_checks': {},
          'execution_context': 'docker' if Path('/.dockerenv').exists() else 'wsl',
          'hostname': platform.node()}
try:
    for name in ['torch', 'mmcv', 'mmdet', 'mmseg', 'mmdet3d']:
        package = importlib.import_module(name)
        report['imports'][name] = {'version': package.__version__, 'path': package.__file__}
    import torch
    report['torch_cuda'] = torch.version.cuda
    report['cuda_available'] = torch.cuda.is_available()
    assert report['cuda_available']
    report['gpu'] = torch.cuda.get_device_name(0)
    report['capability'] = torch.cuda.get_device_capability(0)
    report['arch_list'] = torch.cuda.get_arch_list()
    matrix = torch.eye(3, device='cuda')
    report['cuda_checks']['basic_tensor'] = bool(torch.isfinite(matrix @ matrix).all())
    torch.cuda.synchronize()
    # Real operator used by BEVFormer spatial/temporal deformable attention.
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
    value = torch.rand(1, 4, 1, 32, device='cuda')
    spatial_shapes = torch.tensor([[2, 2]], dtype=torch.long, device='cuda')
    level_start_index = torch.tensor([0], dtype=torch.long, device='cuda')
    sampling_locations = torch.rand(1, 2, 1, 1, 2, 2, device='cuda')
    attention_weights = torch.full((1, 2, 1, 1, 2), 0.5, device='cuda')
    out = MultiScaleDeformableAttnFunction.apply(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights, 64)
    torch.cuda.synchronize()
    assert out.shape == (1, 2, 32) and torch.isfinite(out).all()
    report['cuda_checks']['ms_deform_attn'] = {'shape': list(out.shape), 'finite': True}
    from mmdet3d.ops.iou3d import boxes_iou_bev
    boxes = torch.tensor([[0., 0., 2., 2., 0.]], device='cuda')
    iou = boxes_iou_bev(boxes, boxes)
    torch.cuda.synchronize()
    assert torch.allclose(iou, torch.ones_like(iou), atol=1e-5)
    report['cuda_checks']['mmdet3d_iou3d'] = iou.cpu().tolist()
    # Probe documented 40xx error independently; model is patched only if needed.
    try:
        inverse = torch.linalg.inv(matrix)
        torch.cuda.synchronize()
        report['cuda_checks']['cusolver_inverse'] = {'status': 'PASS', 'finite': bool(torch.isfinite(inverse).all())}
    except Exception:
        report['cuda_checks']['cusolver_inverse'] = {'status': 'FAIL', 'error': traceback.format_exc()}
    import projects.mmdet3d_plugin
    report['plugin_import'] = 'PASS'
    report['status'] = 'ENV_IMPORT_CUSTOM_OP_PASS'
except Exception:
    report['status'] = 'ENV_SETUP_FAIL'
    report['error'] = traceback.format_exc()
finally:
    (audit / 'artifacts/core/docker_provenance/environment_gate.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
if report['status'] == 'ENV_SETUP_FAIL':
    raise SystemExit(1)
