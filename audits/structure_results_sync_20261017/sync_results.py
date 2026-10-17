"""Copy completed remote A/B result records, never checkpoint contents."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
SHARED = ROOT / 'experiments/structure_v4_joint'
MODELS = ('A_GBA', 'B_GATE')
METRICS = ('AMOTA', 'mAP', 'IDS', 'feature_payload_BPS', 'serialized_BPS')
REMOTE = r'''
import hashlib, io, json, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path

root = Path('/root/autodl-tmp/Refresh2Track/experiments/structure_v4_joint')
models = ('A_GBA', 'B_GATE')
excluded_names = {
    'predictions.pkl', 'predictions.pkl.gz', 'results_nusc.json',
    'results_nusc.json.gz', 'results_nusc_det.json', 'results_nusc_det.json.gz',
    'communication.json', 'structure_frames.json'
}
selected, excluded = [], []
for model in models:
    state = json.loads((root / model / 'run_state.json').read_text())
    if state['status'] != 'COMPLETED':
        raise RuntimeError('Remote experiment is not completed: ' + model)
    for path in sorted((root / model).rglob('*')):
        if path.is_symlink():
            raise RuntimeError('Unexpected symlink: ' + str(path))
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if ('checkpoints' in relative.parts or path.suffix in ('.pth', '.pt', '.ckpt')
                or path.name.endswith(('.lock', '.writing'))
                or ('validation' in relative.parts and path.name in excluded_names)):
            excluded.append({'path': relative.as_posix(), 'bytes': path.stat().st_size})
        else:
            selected.append(path)
for name in ('remote_queue.json', 'remote_parallel.log', 'sequential_queue_handoff.json'):
    path = root / name
    if path.is_file():
        selected.append(path)
entries = []
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for path in selected:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        relative = path.relative_to(root).as_posix()
        entries.append({'path': relative, 'bytes': path.stat().st_size,
                        'sha256': digest.hexdigest()})
        archive.add(str(path), arcname=relative, recursive=False)
    manifest = {'remote_root': str(root), 'captured_utc': datetime.now(timezone.utc).isoformat(),
                'checkpoint_contents_transferred': False, 'remote_files_modified': False,
                'files': entries, 'excluded': excluded}
    payload = json.dumps(manifest, indent=2).encode()
    member = tarfile.TarInfo('_sync_manifest.json')
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))
'''


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def check_results(stage):
    validated = 0
    for model in MODELS:
        directory = stage / model
        best = read(directory / 'metrics/best_full_val.json')
        final = read(directory / 'validation/final/V4_seed0/result.json')
        assert best['epoch'] == final['epoch']
        assert all(best[key] == final[key] for key in METRICS)
        for path in (directory / 'validation').rglob('result.json'):
            result = read(path)
            assert result['status'] == 'PASS' and result['sample_count'] == 3316
            assert result['scene_count'] == 21
            proof = read(path.with_name('run_manifest.json'))
            assert proof['structure_result_sha256'] == digest(path)
            for kind, expected in proof['metrics_sha256'].items():
                assert expected == digest(path.parent / kind / 'metrics_summary.json')
            validated += 1
    return validated


def summarize(manifest):
    captured = datetime.now(timezone.utc).isoformat()
    models = {}
    rows = [dict(method='V4 Full historical', status='HISTORICAL', best_epoch=12,
                 AMOTA=.375331, mAP=.365131, IDS=249,
                 feature_payload_BPS=16501.53, serialized_BPS=27258.58)]
    for model in (*MODELS, 'C_GBA_GATE'):
        directory = SHARED / model
        state = read(directory / 'run_state.json')
        best = read(directory / 'metrics/best_full_val.json')
        history = read(directory / 'metrics/epoch_validation.json')
        models[model] = {'state': state, 'best': best, 'latest_validation': history[-1]}
        rows.append(dict(method=model, status=state['status'], best_epoch=best['epoch'],
                         **{key: best[key] for key in METRICS}))
    fields = list(rows[0])
    with (SHARED / 'COMPARE_THREE.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    local_time = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    table = ['| 实验 | 状态 | 最佳轮次 | AMOTA | mAP | IDS | Feature B/s | Serialized B/s |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        table.append('| {method} | {status} | {best_epoch} | {AMOTA:.8f} | {mAP:.8f} | '
                     '{IDS:.0f} | {feature_payload_BPS:.2f} | {serialized_BPS:.2f} |'.format(**row))
    report = ['# 三组结构实验结果同步记录', '', '本地快照时间：' + local_time, '', *table, '',
              'A/B 为远程已完成实验，最终最佳模型重评与已保存 best 指标一致；C 仍在本地训练，表中为快照时最佳结果。',
              'V4 Full historical 保留原有历史参照记录，不是本次新实验或新同步评估。',
              '所有表中结果均为完整官方验证集（3316 帧、21 场景）；不是测试集成绩。', '',
              '## 最新一次验证', '']
    for model, values in models.items():
        latest = values['latest_validation']
        report.append('- %s：epoch %d，AMOTA %.8f，mAP %.8f，IDS %d。' %
                      (model, latest['epoch'], latest['AMOTA'], latest['mAP'], latest['IDS']))
    report += ['', '## 同步范围', '',
               '- 已同步 A/B 的指标、逐轮训练历史、评估明细、报告、完整日志、配置及来源记录。',
               '- checkpoint 全部保留在远程，未下载任何权重；未删除或修改远程文件。',
               '- 大体积原始预测、逐帧 communication.json / structure_frames.json 保留远程；其 SHA256 引用仍保留在评估 manifest。',
               '- 每个下载文件已核对远程 SHA256；各次验证结果及检测/跟踪 metrics_summary 已核对 manifest。',
               '- 本文为固定快照；COMPARE_THREE 文件可能被后续 C 协调器重新生成。', '']
    text = '\n'.join(report)
    (SHARED / 'RESULTS_SYNC.md').write_text(text, encoding='utf-8')
    (SHARED / 'COMPARE_THREE.md').write_text(text, encoding='utf-8')
    write(SHARED / 'remote_status.json', {
        'remote_root': '/root/autodl-tmp/Refresh2Track', 'snapshot_utc': captured,
        'queue': read(SHARED / 'remote_queue.json'),
        'experiments': {model: models[model] for model in MODELS},
        'checkpoint_contents_transferred': False})
    manifest.update(status='PASS', verified_utc=captured, local_results_snapshot=models)
    write(AUDIT / 'sync_manifest.json', manifest)


def main():
    command = ['ssh', '-o', 'ClearAllForwardings=yes', '-o', 'BatchMode=yes',
               '-o', 'ConnectTimeout=12', '-o', 'ServerAliveInterval=15',
               '-o', 'ServerAliveCountMax=3', 'connect.nmb2.seetacloud.com', 'python3 -']
    with tempfile.TemporaryDirectory(prefix='staging_', dir=AUDIT) as temporary:
        stage = Path(temporary).resolve()
        assert stage.is_relative_to(AUDIT.resolve())
        archive_path = stage / 'records.tar.gz'
        print('Downloading result records; checkpoints excluded.', flush=True)
        with archive_path.open('wb') as output:
            result = subprocess.run(command, input=REMOTE.encode('utf-8'), stdout=output,
                                    stderr=subprocess.PIPE, timeout=300)
        if result.returncode:
            raise RuntimeError(result.stderr.decode('utf-8', errors='replace')[-4000:])
        with tarfile.open(archive_path, 'r:gz') as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                if (not member.isfile() or relative.is_absolute() or '..' in relative.parts
                        or '\\' in member.name or ':' in member.name
                        or 'checkpoints' in relative.parts
                        or relative.suffix in ('.pth', '.pt', '.ckpt')):
                    raise ValueError('Unexpected archive path: ' + member.name)
                target = stage.joinpath(*relative.parts).resolve()
                if not target.is_relative_to(stage):
                    raise ValueError('Archive path escapes staging directory')
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
        manifest = read(stage / '_sync_manifest.json')
        for entry in manifest['files']:
            relative = PurePosixPath(entry['path'])
            assert (relative.parts[0] in MODELS or entry['path'] in
                    ('remote_queue.json', 'remote_parallel.log', 'sequential_queue_handoff.json'))
            source = stage.joinpath(*relative.parts)
            assert source.stat().st_size == entry['bytes'] and digest(source) == entry['sha256']
        manifest['validated_full_validation_results'] = check_results(stage)
        copied = 0
        for entry in manifest['files']:
            relative = PurePosixPath(entry['path'])
            source, target = stage.joinpath(*relative.parts), SHARED.joinpath(*relative.parts).resolve()
            assert target.is_relative_to(SHARED.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or digest(target) != entry['sha256']:
                source.replace(target)
                copied += 1
            assert digest(target) == entry['sha256']
        manifest['copied_or_updated_files'] = copied
        summarize(manifest)
        print(json.dumps({'status': 'PASS', 'files': len(manifest['files']),
                          'bytes': sum(entry['bytes'] for entry in manifest['files']),
                          'validated_full_validation_results': manifest['validated_full_validation_results'],
                          'checkpoint_contents_transferred': False,
                          'report': str(SHARED / 'RESULTS_SYNC.md')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
