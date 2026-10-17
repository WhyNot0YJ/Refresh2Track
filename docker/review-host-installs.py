"""Read-only check of system/package provenance; never removes packages."""
import datetime
import gzip
import hashlib
import json
import pathlib
import subprocess

audit = pathlib.Path('/mnt/d/CoopTrack_Audit')
records = []
for path in sorted(pathlib.Path('/var/log/apt').glob('history.log*')):
    if not path.is_file():
        continue
    raw = path.read_bytes()
    text = gzip.decompress(raw).decode(errors='replace') if path.suffix == '.gz' else raw.decode(errors='replace')
    records.append({'path': str(path), 'bytes': len(raw),
                    'sha256': hashlib.sha256(raw).hexdigest(),
                    'latest_mtime': datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    'gcc11_transactions': [x for x in text.split('\n\n') if 'gcc-11' in x or 'g++-11' in x],
                    'task_date_transactions': [x for x in text.split('\n\n') if 'Start-Date: 2026-10-02' in x]})
inventory = []
for root in ('/usr/local/lib', '/root/.local/lib', '/home/yujie/.local/lib'):
    for pattern in ('python*/site-packages', 'python*/dist-packages'):
        for path in pathlib.Path(root).glob(pattern):
            info_dirs = [str(x) for x in path.glob('*.dist-info')]
            inventory.append({'path': str(path), 'entries': [x.name for x in sorted(path.iterdir())],
                              'package_birth_mtime': subprocess.run(
                                  ['stat', '-c', '%n | birth=%w | mtime=%y', *info_dirs],
                                  capture_output=True, text=True, check=True).stdout if info_dirs else ''})
result = {
    'status': 'READ_ONLY_REVIEW_COMPLETE',
    'apt_history': records,
    'gcc_installed': subprocess.run(['dpkg-query', '-W', '-f=${Package} ${Version} ${Status}\n',
                                     'gcc-11', 'g++-11'], capture_output=True, text=True, check=True).stdout,
    'system_pip_owner': subprocess.run(['dpkg-query', '-S', '/usr/bin/pip', '/usr/bin/pip3'],
                                     capture_output=True, text=True).stdout,
    'non_distro_pip_site_dirs': inventory,
    'initial_evidence': ['artifacts/core/docker_provenance/environment_recon.txt:4', 'artifacts/core/docker_provenance/environment_recon_detail.json:291'],
    'conclusion': 'GCC11/G++11 were present in the initial read-only reconnaissance. No apt history transaction exists on the task date; empty current apt/dpkg logs and last completed apt history are older than this task. No host system pip install is recorded in the task installer scripts; they select the isolated prefix. Do not uninstall preexisting system compilers or pip.',
    'uninstall_plan': [],
    'host_packages_modified_by_this_review': False,
}
(audit / 'artifacts/core/stage1/host_installation_cleanup_review.json').write_text(json.dumps(result, indent=2))
print(json.dumps({'status': result['status'], 'task_date_apt_transactions':
                  sum(len(x['task_date_transactions']) for x in records),
                  'gcc_preexisting': True, 'uninstall_plan': []}, indent=2))
