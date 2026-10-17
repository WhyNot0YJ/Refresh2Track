"""Small, dependency-free checks used by the Docker-only rebuild."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import subprocess
import tarfile


def digest(path, algorithm):
    value = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(data)
    return value.hexdigest()


def headers(root, prefix):
    resources = json.loads((root / 'locks/cuda-header-archives.json').read_text())
    extracted = []
    for item in resources:
        path = root / 'cuda_packages' / item['filename']
        assert path.stat().st_size == item['size'], path
        assert digest(path, 'md5') == item['md5'], path
        # The original NVIDIA metadata publishes MD5 only. This SHA was computed
        # locally after that official MD5 was verified; it is a retention lock.
        assert digest(path, 'sha256') == item['actual_sha256'], path
        allowed = {x['name'] for x in item['header_members']}
        with tarfile.open(str(path), 'r|bz2') as archive:
            for member in archive:
                if member.name not in allowed:
                    continue
                relative = pathlib.PurePosixPath(member.name)
                assert member.isfile() and relative.parts[0] == 'include'
                assert '..' not in relative.parts and not relative.is_absolute()
                target = prefix.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                assert source is not None
                target.write_bytes(source.read())
                extracted.append(member.name)
    assert len(extracted) == 14, extracted
    for name in ('cublas_v2.h', 'cusparse.h', 'cusolverDn.h'):
        assert (prefix / 'include' / name).is_file(), name
    return {'headers': extracted, 'libraries_extracted': False}


def sources(root):
    locks = json.loads((root / 'locks/official-source-lock.json').read_text())
    repos = {
        'CoopTrack': ('https://github.com/zhongjiaru/CoopTrack.git', '/opt/cooptrack/CoopTrack'),
        'mmdetection3d': ('https://github.com/open-mmlab/mmdetection3d.git', '/opt/cooptrack/mmdetection3d'),
    }
    for name, (url, target) in repos.items():
        commit = locks[name]['commit']
        assert re.fullmatch('[0-9a-f]{40}', commit), commit
        subprocess.run(['git', 'init', target], check=True)
        subprocess.run(['git', '-C', target, 'remote', 'add', 'origin', url], check=True)
        subprocess.run(['git', '-C', target, 'fetch', '--depth=1', 'origin', commit], check=True)
        subprocess.run(['git', '-C', target, 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        actual = subprocess.check_output(['git', '-C', target, 'rev-parse', 'HEAD'], text=True).strip()
        assert actual == commit, (name, actual)
        patch = root / 'locks' / (name + '.patch')
        assert digest(patch, 'sha256') == locks[name]['diff_sha256'], patch
        if patch.stat().st_size:
            subprocess.run(['git', '-C', target, 'apply', '--check', str(patch)], check=True)
            subprocess.run(['git', '-C', target, 'apply', str(patch)], check=True)
    return locks


def versions(root):
    count = 0
    for line in (root / 'locks/pip-exact-versions.txt').read_text().splitlines():
        name, expected = line.split('==', 1)
        actual = importlib.metadata.version(name)
        assert actual == expected, (name, expected, actual)
        count += 1
    assert importlib.metadata.version('pip') == '24.3.1'
    assert importlib.metadata.version('mmdet3d') == '0.17.1'
    import torch, mmcv, mmdet, mmseg, mmdet3d, cv2, open3d
    assert torch.__version__ == '1.9.1+cu111'
    assert mmcv.__version__ == '1.4.0'
    assert mmdet.__version__ == '2.14.0'
    assert mmseg.__version__ == '0.14.1'
    assert mmdet3d.__version__ == '0.17.1'
    return {'exact_pip_packages_checked': count, 'imports': 'PASS',
            'GPU_check': 'Run container_gate.py with --gpus all after build.'}


parser = argparse.ArgumentParser()
parser.add_argument('operation', choices=('headers', 'sources', 'versions'))
parser.add_argument('--root', type=pathlib.Path, default=pathlib.Path('/opt/rebuild'))
parser.add_argument('--prefix', type=pathlib.Path,
                    default=pathlib.Path('/home/yujie/miniconda/envs/cooptrack-smoke'))
args = parser.parse_args()
result = headers(args.root, args.prefix) if args.operation == 'headers' else (
    sources(args.root) if args.operation == 'sources' else versions(args.root))
print(json.dumps(result, indent=2))
