"""Recorded preflight for the exact user-authorized cleanup on 2026-10-13."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / 'audits/cleanup_review_20261012'
OUT = ROOT / 'audits/cleanup_execution_20261013'
OUT.mkdir(parents=True, exist_ok=True)
WHITELIST_SHA = '00e159dae228882bbaa7d042e54b3cd62831defe0ca85859345a99ec36895c55'


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def safe(relative):
    parts = Path(relative)
    assert not parts.is_absolute() and '..' not in parts.parts and ':' not in relative
    path = ROOT / parts
    assert path.is_relative_to(ROOT)
    for parent in path.parents:
        if parent == ROOT:
            break
        assert not (parent.lstat().st_file_attributes & 1024), 'Linked ancestor: ' + str(parent)
    return path


whitelist = REVIEW / 'safe_candidates.csv'
assert sha(whitelist) == WHITELIST_SHA
rows = list(csv.DictReader(whitelist.open(encoding='utf-8-sig')))
assert len(rows) == len({row['path'] for row in rows}) == 186
assert sum(int(row['bytes']) for row in rows) == 100830055
sys.path[:0] = [str(ROOT / 'scripts/v4_budget_free'), str(ROOT / 'scripts')]
import runtime
protection = runtime.verify_protection(ROOT)
assert protection['status'] == 'PASS', protection
locked = runtime.retirement(ROOT)['protected_shared_files']
before_identity = runtime.identity(ROOT)
files = []
for row in rows:
    assert row['decision'] == '可清理' and not row['protection'] and row['path'] not in locked
    path = safe(row['path'])
    stat = path.lstat()
    assert stat.st_size == int(row['bytes']) and stat.st_mtime_ns == int(row['mtime_ns']), row['path']
    linked = bool(stat.st_file_attributes & 1024)
    assert linked == (row['kind'] == 'link'), row['path']
    files.append(dict(row, absolute=str(path), sha256=None if linked else sha(path)))
directories = list(csv.DictReader((REVIEW / 'directories.csv').open(encoding='utf-8-sig')))
safe_directory_names = {row['path'] for row in directories if row['decision'] == '可清理'
                        and int(row['files']) == 0 and '.git' not in Path(row['path']).parts
                        and not row['path'].startswith('datasets/')}
empty, excluded = [], []
for row in directories:
    if row['decision'] != '可清理' or int(row['files']) != 0:
        continue
    name = row['path']
    if '.git' in Path(name).parts or name.startswith('datasets/'):
        row.update(decision='保留', reason='Git 内部结构或正式数据布局目录，不能仅因为空删除。')
        excluded.append(name)
        continue
    path = safe(name)
    assert path.is_dir() and not path.lstat().st_file_attributes & 1024
    for child in path.iterdir():
        assert child.is_dir() and not child.lstat().st_file_attributes & 1024
        assert child.relative_to(ROOT).as_posix() in safe_directory_names, 'New directory content: ' + str(child)
    row['reason'] = '目录树没有文件；执行时只在每个目录实际为空后移除，并复核没有 writer。'
    empty.append({'path': name, 'absolute': str(path)})
with (REVIEW / 'directories.csv').open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(directories[0]))
    writer.writeheader()
    writer.writerows(directories)
summary_path = REVIEW / 'summary.json'
summary = json.loads(summary_path.read_text(encoding='utf-8'))
summary['directory_summary'] = [dict(row, **{key: int(row[key]) for key in
    ('files', 'bytes', 'safe_files', 'conditional_files', 'keep_files')}) for row in directories]
summary['directory_correction_20261013'] = {'retained_structural_empty_dirs': excluded}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

scopes = ['data/V2X-Seq-SPD-Example', 'data_infos/V2X-Seq-SPD-Example',
          'code/CoopTrack/output', 'code/CoopTrack/datasets']
for name in scopes:
    path = safe(name)
    # os.walk does not follow the WSL reparse files on the Windows host.
    import os
    for base, names, _ in os.walk(path, followlinks=False):
        for candidate in [Path(base), *(Path(base) / item for item in names)]:
            relative = candidate.relative_to(ROOT).as_posix()
            if relative not in {row['path'] for row in empty}:
                empty.append({'path': relative, 'absolute': str(candidate), 'after_whitelist_deletion': True})
plan = {'date_asia_shanghai': '2026-10-13', 'prepared_local': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'whitelist_sha256': WHITELIST_SHA, 'authorization': 'User explicitly requested cleaning confirmed safe items.',
        'files': files, 'directories': sorted(empty, key=lambda row: len(row['absolute']), reverse=True),
        'excluded_empty_dirs': excluded, 'before_v4_identity': before_identity,
        'protected_verification_before': protection, 'expected_file_count': 186,
        'expected_logical_bytes': 100830055}
(OUT / 'verified_plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({'status': 'VERIFIED', 'files': len(files), 'directories': len(empty),
                  'retained_structural_dirs': len(excluded), 'bytes': plan['expected_logical_bytes']}, ensure_ascii=False))
