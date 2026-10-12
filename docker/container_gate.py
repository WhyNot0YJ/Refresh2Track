"""Run existing smoke probes in Docker while preserving earlier host evidence."""
import json
from pathlib import Path
import platform
import traceback


AUDIT = Path('/mnt/d/CoopTrack_Audit')
PREFIX = '/home/yujie/miniconda/envs/cooptrack-smoke'
report = {'status': 'DOCKER_ENV_GATE_FAIL', 'container_hostname': platform.node(),
          'libc': platform.libc_ver(), 'prefix': PREFIX}


def run_probe(filename, previous_output, container_output):
    path = AUDIT / 'docker/probes' / filename
    source = path.read_text()
    assert source.count(previous_output) == 1, filename
    # Redirect only the evidence destination; use the existing GPU checks unchanged.
    source = source.replace(previous_output, container_output)
    exec(compile(source, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path)})


try:
    assert Path('/.dockerenv').exists(), 'Run this probe inside Docker.'
    prefix_mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        point = line.split()[4]
        if point != '/' and (point == PREFIX or point.startswith(PREFIX + '/')
                             or PREFIX.startswith(point + '/')):
            prefix_mounts.append(point)
    report['prefix_mountpoints'] = prefix_mounts
    report['host_prefix_bound'] = bool(prefix_mounts)
    assert not prefix_mounts, 'Container must own its prefix, without a host prefix bind.'
    run_probe('environment_gate.py', 'artifacts/core/docker_provenance/environment_gate.json', 'artifacts/core/docker_provenance/docker_environment_gate.json')
    run_probe('verify_40xx_inverse.py', 'artifacts/core/docker_provenance/40xx_inverse_verification.json',
              'artifacts/core/docker_provenance/docker_40xx_inverse_verification.json')
    environment = json.loads((AUDIT / 'artifacts/core/docker_provenance/docker_environment_gate.json').read_text())
    inverse = json.loads((AUDIT / 'artifacts/core/docker_provenance/docker_40xx_inverse_verification.json').read_text())
    assert environment['status'] == 'ENV_IMPORT_CUSTOM_OP_PASS'
    assert inverse['after'] == 'PASS'
    import torch
    import cv2
    import open3d
    report['runtime_imports'] = {'torch': torch.__version__, 'cv2': cv2.__version__,
                                 'open3d': open3d.__version__}
    report['gpu'] = environment['gpu']
    report['custom_ops'] = environment['cuda_checks']
    report['patched_inverse'] = inverse['after']
    report['original_example_bound'] = Path('/mnt/c/Users/yujie/Desktop/CoopTrack/V2X-Seq-SPD-Example').is_dir()
    report['status'] = 'DOCKER_ENV_GATE_PASS'
except BaseException:
    report['error'] = traceback.format_exc()
finally:
    (AUDIT / 'artifacts/core/docker_provenance/docker_migration_gate.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
if report['status'] != 'DOCKER_ENV_GATE_PASS':
    raise SystemExit(1)
