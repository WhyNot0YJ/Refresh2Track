"""Capture exact published artifacts from retained install reports, without installing."""
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path('/mnt/d/CoopTrack_Audit')
OUT = ROOT / 'docker/locks'


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


installed = json.loads((OUT / 'pip-installed-host-original.json').read_text())
wanted = {canonical(x['name']): x['version'] for x in installed}
wanted.pop('pip')  # Included in the explicit Conda lock.
wanted.pop('mmdet3d')  # Built from its official, locked Git commit.
wanted['ninja'] = '1.11.1.3'  # Corrected only inside the migration image.
artifacts = {}
for path in sorted((ROOT / 'artifacts/core/docker_provenance').glob('*install_report.json')):
    for item in json.loads(path.read_text()).get('install', []):
        name = canonical(item['metadata']['name'])
        if wanted.get(name) != item['metadata']['version']:
            continue
        info = item['download_info']
        sha = info.get('archive_info', {}).get('hashes', {}).get('sha256')
        if sha:
            artifacts[name] = {'name': name, 'version': wanted[name],
                               'url': info['url'], 'sha256': sha,
                               'evidence': str(path.relative_to(ROOT))}

for line in (OUT / 'pip-rebuild-requirements.txt').read_text().splitlines():
    if ' @ file://' not in line:
        continue
    name, url = line.split(' @ ', 1)
    name = canonical(name)
    base, fragment = url.split('#sha256=', 1)
    artifacts[name] = {'name': name, 'version': wanted[name], 'url': base,
                       'sha256': fragment, 'evidence': 'retained verified wheel'}

# These tools were installed before pip report capture. Fetch only official PyPI JSON.
for name in ('setuptools', 'wheel'):
    url = 'https://pypi.org/pypi/{}/{}/json'.format(name, wanted[name])
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                metadata = json.load(response)
            break
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2)
    (OUT / ('pypi-' + name + '-metadata.json')).write_text(json.dumps(metadata, indent=2))
    matches = [x for x in metadata['urls'] if x['filename'].endswith('-py3-none-any.whl')]
    assert len(matches) == 1, (name, matches)
    item = matches[0]
    artifacts[name] = {'name': name, 'version': wanted[name], 'url': item['url'],
                       'sha256': item['digests']['sha256'], 'evidence': url}

assert set(artifacts) == set(wanted), sorted(set(wanted) - set(artifacts))
allowed = {'files.pythonhosted.org', 'download.pytorch.org', 'download.openmmlab.com'}
for item in artifacts.values():
    url = urllib.parse.urlsplit(item['url'])
    assert url.scheme == 'file' or (url.scheme == 'https' and url.hostname in allowed), item
    assert re.fullmatch('[0-9a-f]{64}', item['sha256']), item


def requirement(item):
    return '{} @ {} --hash=sha256:{}'.format(item['name'], item['url'], item['sha256'])


(OUT / 'pip-artifacts-linux-64.lock').write_text(
    '# Exact cp38/linux artifacts captured from the successful official install.\n'
    + '\n'.join(requirement(artifacts[x]) for x in sorted(artifacts)) + '\n')
(OUT / 'pip-bootstrap.lock').write_text(
    '\n'.join(requirement(artifacts[x]) for x in ('setuptools', 'wheel', 'numpy')) + '\n')
(OUT / 'pip-artifact-provenance.json').write_text(json.dumps(artifacts, indent=2))
print(json.dumps({'artifacts': len(artifacts), 'missing': [],
                  'downloaded_package_bodies': 0, 'output': str(OUT)}, indent=2))
