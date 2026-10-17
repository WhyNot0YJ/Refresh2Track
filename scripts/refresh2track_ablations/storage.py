"""Retain SHA-verified lossless evaluator archives without duplicate raw files."""
import gzip
import hashlib
from pathlib import Path
import shutil
import tempfile


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def archive_digest(path):
    value = hashlib.sha256()
    with gzip.open(str(path), 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def artifacts(result, directory):
    for name in ('predictions', 'tracking', 'detection'):
        item = result['evaluator_inputs'][name]
        path, archive = Path(item['path']), Path(item['archive'])
        path.resolve().relative_to(Path(directory).resolve())
        archive.resolve().relative_to(Path(directory).resolve())
        if digest(archive) != item['archive_sha256']:
            raise ValueError('Evaluator archive changed: ' + name)
        yield path, archive, item['sha256']


def restore(result, directory):
    for path, archive, expected in artifacts(result, directory):
        if not path.exists():
            temporary = path.with_name(path.name + '.restore-writing')
            with gzip.open(str(archive), 'rb') as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
            if digest(temporary) != expected:
                raise ValueError('Lossless evaluator restore failed')
            temporary.replace(path)
        elif digest(path) != expected:
            raise ValueError('Evaluator raw input changed')


def retain_archives(result, directory):
    freed = 0
    for path, archive, expected in artifacts(result, directory):
        if archive_digest(archive) != expected:
            raise ValueError('Lossless evaluator archive differs from raw input')
        if path.exists():
            if digest(path) != expected:
                raise ValueError('Evaluator input changed before duplicate removal')
            freed += path.stat().st_size
            path.unlink()
    return freed


def self_check():
    with tempfile.TemporaryDirectory(prefix='refresh2track-storage-') as temporary:
        directory = Path(temporary)
        inputs = {}
        for name in ('predictions', 'tracking', 'detection'):
            path = directory / name
            path.write_bytes((name.encode() + b'\0') * 101)
            archive = path.with_suffix('.gz')
            with gzip.open(str(archive), 'wb') as stream:
                stream.write(path.read_bytes())
            inputs[name] = {'path': str(path), 'archive': str(archive), 'sha256': digest(path), 'archive_sha256': digest(archive)}
        result = {'evaluator_inputs': inputs}
        assert retain_archives(result, directory) > 0
        assert all(not Path(item['path']).exists() for item in inputs.values())
        restore(result, directory)
        assert all(digest(item['path']) == item['sha256'] for item in inputs.values())
    return {'status': 'PASS', 'lossless_archive_restore_sha256_roundtrip': True}
