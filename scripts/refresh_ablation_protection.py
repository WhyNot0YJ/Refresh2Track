"""Read-only protection and directory audit for the refresh 2x2 experiment.

Only this experiment's two metrics JSON files are written. Large data/cache
trees are inventoried by file metadata, without copying or hashing payloads.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import stat as stat_mode
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
EXPERIMENT = 'experiments/v2/ablations/refresh_2x2'
ROOTS = (
    'scripts', 'configs', 'reports', 'logs', 'experiments/main', 'experiments/v2',
    'checkpoints', 'artifacts/core', 'datasets', 'data', 'data_infos',
    'query_cache', 'code/CoopTrack', 'code/mmdetection3d', 'docker', 'patches',
)
PURPOSES = {
    'scripts': 'Existing experiment entrypoints and shared runtime/protocol code',
    'configs': 'Frozen experiment settings, manifests and locked budgets',
    'reports': 'Existing V1/V2 formal reports and provenance',
    'logs': 'Existing training, evaluation and resource evidence',
    'experiments/main': 'V1 formal checkpoints, metrics, predictions and communication',
    'experiments/v2': 'Frozen V2 formal outputs; new refresh_2x2 subtree excluded',
    'checkpoints': 'Original official ResNet50 weights',
    'artifacts/core': 'Inherited data/model identities and stage evidence',
    'datasets': 'Prepared and original dataset volumes',
    'data': 'Existing example data and environment packages',
    'data_infos': 'Official prepared train/validation infos',
    'query_cache': 'Existing train/validation roadside predictions',
    'code/CoopTrack': 'Official source, inherited compatibility changes and Git history',
    'code/mmdetection3d': 'Existing native dependencies and Git history',
    'docker': 'Frozen runtime environment provenance',
    'patches': 'Existing compatibility patches',
}
NEW_FILES = {
    'configs/main_experiment_v2/refresh_ablation.json',
    'reports/v2/REFRESH_2X2_ABLATION_FINAL.md',
}
TEXT_SUFFIXES = {'.py', '.ps1', '.sh', '.md', '.txt', '.json', '.jsonl', '.csv',
                 '.yml', '.yaml', '.toml', '.ini', '.cfg', '.cpp', '.cu', '.h', '.c'}
NATIVE_REPOSITORIES = ('code/CoopTrack', 'code/mmdetection3d')
DRIFT_EVIDENCE = EXPERIMENT + '/metrics/git_metadata_drift_verification.json'
# The fixed supplemental capture is retained; later autofetch is never a new baseline.
DRIFT_EVIDENCE_SHA256 = 'c290588403f6fc921e6968faf36463ed7d3ed140bb9f7fa449b2267847625fe4'
CLEANUP_EVIDENCE = 'audits/project_structure_cleanup/cleanup_evidence.json'
CLEANUP_INVENTORY = 'audits/project_structure_cleanup/inventory_after.csv.gz'
CLEANUP_APPLY = Path('C:/Users/yujie/AppData/Local/Temp/cooptrack_structure_cleanup/apply_current.json')
CLEANUP_HASHES = {
    CLEANUP_EVIDENCE: 'd4bd8a1f1c5b2d563a32d82ad8630b39306615af1562cf02df07e873c286652b',
    CLEANUP_INVENTORY: '00d68d4912fb266af358e1399526daa11d8fe97734c1d41c9d87ca840a98ad4e',
    str(CLEANUP_APPLY): '1028cce1ab048ff30917cb872eea4b68dfd0eebcaaa1db28bd8d092333e74d40',
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def excluded(relative):
    if relative == EXPERIMENT or relative.startswith(EXPERIMENT + '/'):
        return True
    if relative in NEW_FILES:
        return True
    if relative.startswith('scripts/'):
        name = Path(relative).name
        return name.startswith('refresh_ablation') or name.startswith('v2_refresh_ablation')
    return False


def should_hash(relative, size):
    parts = Path(relative).parts
    if '.git' in parts:
        return False
    name = Path(relative).name.lower()
    if parts[0] in {'datasets', 'data', 'query_cache'}:
        return 'manifest' in name or name in {'identity.json', 'metadata.json'}
    if 'predictions' in parts or name.startswith('results_nusc_') or name == 'bbox_results.pkl':
        return 'manifest' in name or name in {'result.json', 'identity.json'}
    return Path(relative).suffix.lower() in TEXT_SUFFIXES | {'.pth', '.pkl'} or size <= 1024 * 1024


def walk_files(root, relative):
    """Sorted traversal never follows symlink/junction directory targets."""
    start = root / relative
    if start.is_file():
        yield start
        return
    for current, dirs, files in os.walk(start, followlinks=False):
        current = Path(current)
        kept = []
        for name in sorted(dirs):
            path = current / name
            rel = path.relative_to(root).as_posix()
            if excluded(rel):
                continue
            directory_stat = path.lstat()
            if (getattr(directory_stat, 'st_file_attributes', 0) & 0x400 or
                    stat_mode.S_ISLNK(directory_stat.st_mode)):
                yield path
            else:
                kept.append(name)
        dirs[:] = kept
        for name in sorted(files):
            path = current / name
            if not excluded(path.relative_to(root).as_posix()):
                yield path


def git_identity(path):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    commands = {
        'head': ['rev-parse', 'HEAD'],
        'status': ['status', '--porcelain=v1', '--untracked-files=all'],
        'tracked_diff_sha256': ['diff', '--binary', 'HEAD', '--'],
    }
    result = {}
    for key, args in commands.items():
        process = subprocess.run(['git', '-C', str(path), *args], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 check=False)
        if process.returncode:
            result[key] = {'error': process.stderr.decode('utf-8', 'replace').strip()}
        elif key.endswith('_sha256'):
            result[key] = hashlib.sha256(process.stdout).hexdigest()
        else:
            result[key] = process.stdout.decode('utf-8', 'replace').strip()
    return result


def snapshot_tree(root, relative, metadata_overrides=None):
    """Use the original inventory encoding, including for disclosed reconstruction."""
    metadata_overrides = metadata_overrides or {}
    digest = hashlib.sha256()
    count, total = 0, 0
    suffixes = Counter()
    git_files, temporary, strong_files = [], [], {}
    for path in walk_files(root, relative):
        rel = path.relative_to(root).as_posix()
        if any(part in path.name.lower() for part in
               ('.tmp', '.partial', '.writing', '.eval_hook')):
            temporary.append(rel)
        stat = path.lstat()
        record = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if getattr(stat, 'st_file_attributes', 0) & 0x400:
            # WSL links have reparse tags but cannot be read/followed by Windows.
            record['reparse_tag'] = getattr(stat, 'st_reparse_tag', None)
        elif stat_mode.S_ISLNK(stat.st_mode):
            record['symlink_target'] = os.readlink(path)
        elif should_hash(rel, stat.st_size):
            record['sha256'] = sha256(path)
            strong_files[rel] = record
        record.update(metadata_overrides.get(rel, {}))
        digest.update(json.dumps([rel, record], sort_keys=True,
                                 separators=(',', ':')).encode('utf-8') + b'\n')
        count += 1
        total += record['size']
        suffixes[path.suffix.lower() or '<none>'] += 1
        if '.git' in path.parts:
            git_files.append([rel, record['size'], record['mtime_ns']])
    tree = {
        'exists': (root / relative).exists(), 'file_count': count,
        'total_file_bytes': total, 'metadata_and_selected_sha256': digest.hexdigest(),
        'suffix_counts': dict(sorted(suffixes.items())),
    }
    if git_files:
        tree['git_history_metadata'] = {
            'file_count': len(git_files), 'sha256': hashlib.sha256(
                json.dumps(git_files, separators=(',', ':')).encode()).hexdigest(),
        }
    return tree, strong_files, temporary


def snapshot(root):
    root = Path(root).resolve()
    trees, strong_files = {}, {}
    temporary = []
    for relative in ROOTS:
        tree, files, candidates = snapshot_tree(root, relative)
        trees[relative] = tree
        strong_files.update(files)
        temporary.extend(candidates)
    for path in sorted(root.glob('*')):
        if path.is_file():
            stat = path.stat()
            strong_files[path.name] = {'size': stat.st_size,
                'mtime_ns': stat.st_mtime_ns, 'sha256': sha256(path)}
    return {'schema': 1, 'root': str(root), 'created_utc': utc_now(),
            'method': 'All protected files: path/size/mtime_ns metadata digest; selected files additionally SHA256; Git objects metadata only; no data copy',
            'trees': trees, 'sha256_files': strong_files,
            'temporary_name_candidates': temporary,
            'native_git': {rel: git_identity(root / rel) for rel in NATIVE_REPOSITORIES},
            'excluded_new_artifacts': [EXPERIMENT, *sorted(NEW_FILES),
                'scripts/refresh_ablation*.py and bytecode',
                'scripts/v2_refresh_ablation.py and bytecode']}


def metrics_dir(root):
    root = Path(root).resolve()
    path = root / EXPERIMENT / 'metrics'
    # Resolve before writing: a new experiment symlink must never target protected output.
    path.resolve().relative_to(root / EXPERIMENT)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def structure_audit(before):
    hashes = before['sha256_files']
    duplicates = {}
    for path, record in hashes.items():
        if path.startswith(('scripts/', 'configs/')):
            duplicates.setdefault(record['sha256'], []).append(path)
    return {'schema': 1, 'status': 'INSPECTED_PRESERVED', 'created_utc': utc_now(),
        'directory_purposes': PURPOSES, 'protected_trees': before['trees'],
        'native_git_before': before['native_git'],
        'duplicate_script_or_config_contents': [paths for paths in duplicates.values() if len(paths) > 1],
        'temporary_name_candidates': before.get('temporary_name_candidates', []), 'files_cleaned': [],
        'cleanup_decision': 'Preserve existing artifacts; no obsolete unreferenced generated files were established. Keep this experiment in one isolated subtree.',
        'prediction_and_data_duplicate_scan': 'Payloads were not re-read or copied; existing data, cache and prediction paths were inventoried by metadata. Different names alone are not evidence of useless duplication.',
        'protected_artifacts': 'BASELINE_RECORDED',
        'protection_limits': 'Large data/cache/prediction payloads and Git objects use metadata, so no complete content hash claim is made for those trees.'}


def create_before(root):
    output = metrics_dir(root)
    path = output / 'protection_before.json'
    if path.exists():
        before = json.loads(path.read_text(encoding='utf-8'))
        if Path(before['root']).resolve() != Path(root).resolve():
            raise ValueError('Protection baseline belongs to a different project root')
        return before
    before = snapshot(root)
    write_json(path, before, exclusive=True)
    audit_path = output / 'structure_audit.json'
    if audit_path.exists():
        raise FileExistsError('Refusing to replace an existing structure audit before baseline creation')
    write_json(audit_path, structure_audit(before), exclusive=True)
    return before


def reconcile_git_metadata(root, before, raw_result):
    """Accept only the fixed, disclosed FETCH_HEAD bookkeeping evidence.

    This never changes a Git file, restores a timestamp, or replaces the baseline.
    Original aggregate Git metadata equality remains FAIL and is kept separately.
    """
    root = Path(root).resolve()
    result = {'status': 'FAIL', 'checked_utc': utc_now(),
              'evidence_path': DRIFT_EVIDENCE, 'verified_conditions': [],
              'accepted_FETCH_HEAD_mtime_changes': [], 'reconstructed_tree_checks': {}}

    def check(condition, reason):
        if not condition:
            raise ValueError(reason)
        result['verified_conditions'].append(reason)

    try:
        check(raw_result['status'] == 'FAIL', 'Raw metadata verdict is retained as FAIL')
        check(set(raw_result['changed_trees']) == set(NATIVE_REPOSITORIES),
              'Only the two explicitly recorded native repository trees differ')
        check(not raw_result['changed_or_missing_sha256_files'] and
              not raw_result['added_protected_sha256_files'] and
              not raw_result['native_git_changed'],
              'All original selected SHA256 files and native Git identities match baseline')
        check(raw_result['native_git_after'] == before['native_git'],
              'Raw current HEAD/status/tracked diff equal the original baseline')
        evidence_path = root / DRIFT_EVIDENCE
        evidence_stat = evidence_path.stat()
        evidence_payload = evidence_path.read_bytes()
        result['evidence_sha256'] = hashlib.sha256(evidence_payload).hexdigest()
        check(result['evidence_sha256'] == DRIFT_EVIDENCE_SHA256,
              'Supplemental evidence is the exact fixed 17:34 capture')
        evidence = json.loads(evidence_payload)
        check(evidence_path.stat().st_mtime_ns == evidence_stat.st_mtime_ns and
              evidence_path.stat().st_size == evidence_stat.st_size,
              'Supplemental evidence is stable during read')
        check(evidence['status'] == 'EVIDENCE_COMPLETE' and
              evidence['baseline_path'] == EXPERIMENT + '/metrics/protection_before.json' and
              evidence['baseline_sha256'] == sha256(root / evidence['baseline_path']) and
              evidence['baseline_created_utc'] == before['created_utc'],
              'Supplemental evidence references the unchanged original baseline SHA256/time')
        check(evidence['raw_protection_status'] == 'FAIL' and
              set(evidence['raw_changed_trees']) == set(NATIVE_REPOSITORIES) and
              not evidence['raw_strong_file_changes'] and
              not evidence['raw_added_strong_files'] and
              not evidence['raw_native_git_changed'],
              'Supplemental raw mismatch has the same narrow scope')
        check(set(evidence['repositories']) == set(NATIVE_REPOSITORIES),
              'Supplemental repository set is exactly the two named repositories')
        proof_limits = {}
        for relative in NATIVE_REPOSITORIES:
            recorded = evidence['repositories'][relative]
            baseline_tree = before['trees'][relative]
            current_tree, current_files, _ = snapshot_tree(root, relative)
            check(current_tree == raw_result['after_trees'][relative],
                  relative + ': current tree remains stable since the raw check')
            check(recorded['baseline_tree'] == baseline_tree and
                  all(current_tree[field] == baseline_tree[field] ==
                      recorded['raw_current_tree'][field]
                      for field in ('exists', 'file_count', 'total_file_bytes', 'suffix_counts')),
                  relative + ': original tree existence/count/bytes/suffix counts are preserved')
            check(current_files == {name: value for name, value in before['sha256_files'].items()
                                    if name.startswith(relative + '/')},
                  relative + ': fresh selected-file SHA256/size/mtime equality with baseline')
            locked_git = before['native_git'][relative]
            current_git = git_identity(root / relative)
            check(current_git == locked_git == recorded['baseline_native_git'] ==
                  recorded['current_native_git'] and recorded['native_git_equal_to_baseline'],
                  relative + ': fresh HEAD/status/tracked diff equal baseline and supplemental capture')
            fetch_path = relative + '/.git/FETCH_HEAD'
            check(recorded['all_reads_stable'] and
                  recorded['only_FETCH_HEAD_mtime_after_baseline'] and
                  recorded['files_with_mtime_after_baseline'] == [fetch_path] and
                  recorded['FETCH_HEAD_points_to_locked_HEAD'] and
                  recorded['FETCH_HEAD_commits'] == [locked_git['head']] and
                  recorded['history_fsck']['exit_code'] == 0,
                  relative + ': supplemental fsck/stability/only-FETCH_HEAD/locked-HEAD checks pass')
            expected = {item['path']: item for item in recorded['current_git_files']}
            current_paths = {path.relative_to(root).as_posix(): path
                             for path in walk_files(root, relative + '/.git')}
            check(set(expected) == set(current_paths) and
                  len(expected) == recorded['current_git_file_count'] and
                  sum(item['size'] for item in expected.values()) == recorded['current_git_total_bytes'],
                  relative + ': every current Git path is exactly represented by supplemental records')
            overrides = {}
            for name, path in current_paths.items():
                item = expected[name]
                stat_before = path.stat()
                content_sha256 = sha256(path)
                stat_after = path.stat()
                check(item['stable_during_read'] and
                      (stat_before.st_size, stat_before.st_mtime_ns) ==
                      (stat_after.st_size, stat_after.st_mtime_ns) and
                      stat_after.st_size == item['size'] and content_sha256 == item['sha256'],
                      name + ': unchanged supplemental content/size and stable current read')
                if name == fetch_path:
                    check(stat_after.st_mtime_ns >= item['mtime_ns'],
                          name + ': timestamp is unchanged or advanced, never moved backwards')
                    check(path.read_text(encoding='utf-8') == recorded['FETCH_HEAD_content'] and
                          [line.split('\t', 1)[0] for line in recorded['FETCH_HEAD_content'].splitlines()] ==
                          [locked_git['head']],
                          name + ': exact recorded fetch content still points to the locked HEAD')
                    overrides[name] = {'mtime_ns': item['mtime_ns']}
                    if stat_after.st_mtime_ns != item['mtime_ns']:
                        result['accepted_FETCH_HEAD_mtime_changes'].append({
                            'path': name, 'size': item['size'], 'current_sha256': content_sha256,
                            'supplemental_mtime_ns': item['mtime_ns'],
                            'current_mtime_ns': stat_after.st_mtime_ns})
                else:
                    check(stat_after.st_mtime_ns == item['mtime_ns'],
                          name + ': exact supplemental Git metadata is unchanged')
            reconstructed, _, _ = snapshot_tree(root, relative, overrides)
            check(reconstructed == recorded['raw_current_tree'],
                  relative + ': substituting only exact FETCH_HEAD recorded mtimes reconstructs the fixed supplemental tree')
            process = subprocess.run(['git', '-C', str(root / relative), 'fsck', '--full'],
                env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False)
            check(process.returncode == 0, relative + ': fresh full Git object/reference integrity check passes')
            result['reconstructed_tree_checks'][relative] = {
                'status': 'PASS', 'supplemental_tree_reconstructed': reconstructed,
                'git_file_content_checks': len(current_paths), 'selected_file_checks': len(current_files),
                'fresh_fsck': {'exit_code': process.returncode,
                    'stdout': process.stdout.decode('utf-8', 'replace'),
                    'stderr': process.stderr.decode('utf-8', 'replace')}}
            proof_limits[relative] = recorded['proof_limits']
        result.update(status='PASS',
            reason='All protected content checks pass; only the two named FETCH_HEAD bookkeeping files have recorded metadata drift. Further accepted changes are mtime-only with identical size/SHA256/locked HEAD. Background autofetch attribution is an inference.',
            proof_limits=proof_limits,
            original_git_equality_limit='The original baseline has aggregate Git metadata digests only. Complete original before/after Git byte or mtime equality is not established; raw metadata FAIL remains preserved.')
    except (OSError, KeyError, TypeError, ValueError) as error:
        result['reason'] = str(error)
    return result


def verify(root):
    output = metrics_dir(root)
    before = json.loads((output / 'protection_before.json').read_text(encoding='utf-8'))
    after = snapshot(root)
    changed_trees = [name for name, value in before['trees'].items()
                     if after['trees'].get(name) != value]
    changed_files = sorted(path for path in before['sha256_files']
        if before['sha256_files'][path] != after['sha256_files'].get(path))
    added_files = sorted(set(after['sha256_files']) - set(before['sha256_files']))
    git_changed = before['native_git'] != after['native_git']
    status = 'PASS' if not (changed_trees or changed_files or added_files or git_changed) else 'FAIL'
    result = {'status': status, 'checked_utc': utc_now(), 'changed_trees': changed_trees,
              'changed_or_missing_sha256_files': changed_files,
              'added_protected_sha256_files': added_files, 'native_git_changed': git_changed,
              'protected_tree_count': len(before['trees']),
              'protected_sha256_file_count': len(before['sha256_files']),
              'after_trees': after['trees'], 'native_git_after': after['native_git']}
    reconciliation = reconcile_git_metadata(root, before, result) if status == 'FAIL' else {
        'status': 'NOT_NEEDED', 'reason': 'Original strict metadata comparison passed'}
    effective_status = 'PASS' if status == 'PASS' or reconciliation['status'] == 'PASS' else 'FAIL'
    audit_path = output / 'structure_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    audit.update(protected_artifacts=effective_status, raw_metadata_status=status,
                 protection_after=result, git_metadata_reconciliation=reconciliation,
                 temporary_name_candidates=after['temporary_name_candidates'])
    write_json(audit_path, audit)
    return {**result, 'status': effective_status, 'raw_metadata_status': status,
            'git_metadata_reconciliation': reconciliation}


def record_required_assets(root):
    """Separate required immutable inputs from externally removed derivatives.

    This records evidence, never changes the raw full-inventory FAIL or silently
    exempts losses. It is specific to the externally documented cleanup event.
    """
    root = Path(root).resolve()
    output = metrics_dir(root)
    audit_path = output / 'structure_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    before = json.loads((output / 'protection_before.json').read_text(encoding='utf-8'))
    raw = audit['protection_after']
    required = {'status': 'FAIL', 'assessment_utc': utc_now(),
        'original_inventory_checked_utc': raw['checked_utc'],
        'scope': 'Original frozen source/config/report/checkpoint/formal-result/data/query-cache and retained canonical runtime assets; excludes only the exact externally documented generated cleanup targets',
        'checks': [], 'full_original_inventory_status': raw['status'],
        'proof_limits': 'Full original inventory equality is FAIL. Large dataset/cache/prediction payloads use the original metadata inventories, not new complete content hashes. Original complete Git byte/mtime equality was never recorded.'}
    external = {'assessment_utc': required['assessment_utc'], 'external_cleanup': {},
                'added_file_inventory': [], 'process_snapshot': {}}

    def check(condition, message):
        if not condition:
            raise ValueError(message)
        required['checks'].append(message)

    try:
        check(raw['status'] == 'FAIL', 'Original full-inventory FAIL is retained without exemptions')
        for name, expected in CLEANUP_HASHES.items():
            path = Path(name) if Path(name).is_absolute() else root / name
            check(sha256(path) == expected, 'Pinned external cleanup evidence SHA256: ' + name)
        plan = json.loads((root / CLEANUP_EVIDENCE).read_text(encoding='utf-8'))
        applied = json.loads(CLEANUP_APPLY.read_text(encoding='utf-8-sig'))
        files = plan['merged_files'] + plan['deleted_files']
        targets = {entry['path']: entry for entry in files}
        directories = {entry['path'] for entry in plan['removed_directories']}
        check(len(files) == len(targets) == len(applied['deleted_paths']) == 116 and
              set(applied['deleted_paths']) == set(targets) and
              len(directories) == len(applied['removed_directory_paths']) == 232 and
              set(applied['removed_directory_paths']) == directories,
              'Actual external Apply lists exactly the fixed 116 files and 232 empty directories')
        check(applied['mode'] == 'APPLY' and applied['preflight'] == applied['execution'] == 'PASS' and
              applied['final'] == 'COMPLETE' and not applied['errors'] and not applied['missing'] and
              applied['files_deleted'] == 116 and applied['directories_removed'] == 232 and
              applied['bytes_reclaimed'] == sum(entry['size'] for entry in files) == 2056555438 and
              applied['checkpoint_hashes'] == applied['protected_hashes'] == 'PASS',
              'External Apply succeeded with pinned counts/bytes and protected/checkpoint checks')
        check(all(not os.path.lexists(root / name) for name in targets) and
              all(not os.path.lexists(root / name) for name in directories),
              'All 116 exact cleanup files and 232 exact empty-directory targets are absent')
        target_rows = {}
        known_missing = []
        affected = set(raw['changed_trees'])
        with gzip.open(root / CLEANUP_INVENTORY, 'rt', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                name = row['path']
                if name in targets:
                    check(name not in target_rows, 'Unique reviewed inventory target: ' + name)
                    target_rows[name] = row
                if (row['kind'] == 'file' and not excluded(name) and
                    any(name.startswith(relative + '/') for relative in affected) and
                    not os.path.lexists(root / name)):
                    known_missing.append(name)
        check(set(target_rows) == set(targets) and set(known_missing) == set(targets),
              'Every known original file loss in affected trees is exactly on the documented whitelist')
        for name, entry in targets.items():
            row = target_rows[name]
            check(row['kind'] == 'file' and int(row['size']) == entry['size'],
                  'Reviewed original cleanup target type/size: ' + name)
            if name in before['sha256_files']:
                old = before['sha256_files'][name]
                check(old['size'] == int(row['size']) and old['mtime_ns'] == int(row['mtime_ns']),
                      'Cleanup inventory agrees with original baseline target metadata: ' + name)
        missing_selected = [name for name in raw['changed_or_missing_sha256_files']
                            if not os.path.lexists(root / name)]
        check(len(missing_selected) == 51 and set(missing_selected) ==
              set(targets).intersection(before['sha256_files']) and
              len(missing_selected) == len(raw['changed_or_missing_sha256_files']) and
              not raw['native_git_changed'],
              'Exactly 51 original selected paths are whitelisted ancillary losses; no surviving original selected content changes')
        required['missing_ancillary_selected_paths'] = sorted(missing_selected)
        required['original_selected_file_count'] = len(before['sha256_files'])
        required['unchanged_original_selected_file_count'] = len(before['sha256_files']) - len(missing_selected)
        required['surviving_original_selected_changes'] = []
        for name, original in before['sha256_files'].items():
            if name in targets:
                continue
            path = root / name
            stat = path.stat()
            digest = sha256(path)
            after_read = path.stat()
            check(stat.st_size == original['size'] and digest == original['sha256'] and
                  (stat.st_size, stat.st_mtime_ns) == (after_read.st_size, after_read.st_mtime_ns),
                  'Fresh surviving original selected content matches baseline: ' + name)
        required['original_selected_content_checked_utc'] = utc_now()
        expected_trees = ('experiments/main', 'experiments/v2', 'checkpoints', 'artifacts/core',
                          'datasets', 'data_infos', 'query_cache', 'docker', 'patches')
        check(all(before['trees'][name] == raw['after_trees'][name] for name in expected_trees),
              'All nine required formal-result/data/cache/checkpoint/environment tree inventories equal original baseline')
        required['unchanged_required_trees'] = list(expected_trees)
        groups = {}
        for name, count in (('data', 62), ('scripts', 7), ('code/mmdetection3d', 47)):
            members = [entry for entry in files if entry['path'].startswith(name + '/')]
            check(len(members) == count, 'Exact externally removed derivative count: ' + name)
            groups[name] = {'file_count': count, 'bytes': sum(entry['size'] for entry in members)}
        check(before['trees']['data']['file_count'] - raw['after_trees']['data']['file_count'] == 62 and
              before['trees']['data']['total_file_bytes'] - raw['after_trees']['data']['total_file_bytes'] == groups['data']['bytes'] and
              before['trees']['code/mmdetection3d']['file_count'] - raw['after_trees']['code/mmdetection3d']['file_count'] == 47,
              'Raw data count/bytes and native file-count losses equal the documented derivative groups')
        native_current, _, _ = snapshot_tree(root, 'code/mmdetection3d')
        check(before['trees']['code/mmdetection3d']['file_count'] - native_current['file_count'] == 47 and
              before['trees']['code/mmdetection3d']['total_file_bytes'] - native_current['total_file_bytes'] == groups['code/mmdetection3d']['bytes'],
              'Fresh native file-count/byte losses exactly equal the 47 documented derivatives')
        required['raw_native_aggregate_discrepancy'] = {
            'raw_total_file_bytes': raw['after_trees']['code/mmdetection3d']['total_file_bytes'],
            'current_total_file_bytes': native_current['total_file_bytes'],
            'difference_bytes': native_current['total_file_bytes'] - raw['after_trees']['code/mmdetection3d']['total_file_bytes'],
            'interpretation': 'Raw inventory FAIL is preserved. A transient FETCH_HEAD truncation during background fetch is an inference; only the current stable Git contents are verified below.'}
        retained = []
        runtime_merges = [entry for entry in plan['merged_files']
                          if entry['path'].startswith('code/mmdetection3d/build/')]
        check(len(runtime_merges) == 11, 'Exactly eleven reviewed duplicate extensions have retained canonical targets')
        for entry in runtime_merges:
            target = root / entry['target']
            check(entry['target'] not in targets and target.is_file() and
                  target.stat().st_size == entry['size'] and sha256(target) == entry['sha256'],
                  'Canonical runtime extension remains content-identical: ' + entry['target'])
            retained.append({'path': entry['target'], 'size': entry['size'], 'sha256': entry['sha256']})
        wheel = plan['environment_before']['wheel_fragments']['concat_proof']
        wheel_path = root / wheel['complete_wheel']
        check(wheel['status'] == 'PASS' and wheel['all_ranges_contiguous'] and
              wheel['all_lengths_match'] and wheel_path.stat().st_size == wheel['total_payload_bytes'] and
              sha256(wheel_path) == wheel['recorded_sha256'] == wheel['combined_sha256'],
              'Retained complete Torch wheel freshly matches the pinned complete/combined SHA256')
        required['retained_runtime_extensions'] = retained
        required['retained_complete_wheel'] = {'path': wheel['complete_wheel'],
            'size': wheel_path.stat().st_size, 'sha256': wheel['recorded_sha256']}
        check(sha256(root / DRIFT_EVIDENCE) == DRIFT_EVIDENCE_SHA256,
              'Fixed supplemental Git evidence remains unchanged')
        git_evidence = json.loads((root / DRIFT_EVIDENCE).read_text(encoding='utf-8'))
        check(git_evidence['baseline_sha256'] == sha256(output / 'protection_before.json'),
              'Original protection baseline SHA256 remains unchanged')
        git_checks = {}
        for relative in NATIVE_REPOSITORIES:
            recorded = git_evidence['repositories'][relative]
            check(git_identity(root / relative) == before['native_git'][relative],
                  relative + ': current HEAD/status/tracked diff match original baseline')
            expected = {item['path']: item for item in recorded['current_git_files']}
            current = {path.relative_to(root).as_posix(): path for path in walk_files(root, relative + '/.git')}
            check(set(current) == set(expected), relative + ': Git path inventory matches fixed supplemental capture')
            fetch_changes = []
            other_metadata_changes = []
            for name, path in current.items():
                item = expected[name]
                stat = path.stat()
                digest = sha256(path)
                after_read = path.stat()
                check(stat.st_size == item['size'] and digest == item['sha256'] and
                      (stat.st_size, stat.st_mtime_ns) == (after_read.st_size, after_read.st_mtime_ns),
                      name + ': content/size match supplemental Git evidence')
                if stat.st_mtime_ns != item['mtime_ns'] and not name.endswith('/FETCH_HEAD'):
                    other_metadata_changes.append({'path': name, 'recorded_mtime_ns': item['mtime_ns'],
                                                   'current_mtime_ns': stat.st_mtime_ns,
                                                   'same_size': True, 'same_sha256': True})
                if name.endswith('/FETCH_HEAD'):
                    check(path.read_text(encoding='utf-8').split('\t', 1)[0] == before['native_git'][relative]['head'],
                          name + ': fetch content points to the locked HEAD')
                    if stat.st_mtime_ns != item['mtime_ns']:
                        fetch_changes.append({'path': name, 'recorded_mtime_ns': item['mtime_ns'],
                                              'current_mtime_ns': stat.st_mtime_ns})
            fsck = subprocess.run(['git', '-C', str(root / relative), 'fsck', '--full'],
                env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            check(fsck.returncode == 0, relative + ': fresh Git history integrity check passes')
            git_checks[relative] = {'status': 'PASS', 'Git_content_records': len(current),
                                   'FETCH_HEAD_mtime_changes': fetch_changes,
                                   'other_metadata_changes': other_metadata_changes,
                                   'metadata_equality_status': 'FAIL' if fetch_changes or other_metadata_changes else 'PASS',
                                   'fsck_exit_code': fsck.returncode,
                                   'scope': 'Current Git contents/path sizes and original HEAD/status/diffs; timestamp equality is separately disclosed and remains failed'}
        required['native_source_and_git_checks'] = git_checks
        for name in raw['added_protected_sha256_files']:
            path = root / name
            stat = path.stat()
            digest = sha256(path)
            after = path.stat()
            external['added_file_inventory'].append({'path': name, 'size': stat.st_size,
                'mtime_ns': stat.st_mtime_ns, 'sha256': digest,
                'stable_during_read': (stat.st_size, stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns)})
        process = subprocess.run(['docker', 'exec', 'cooptrack-main', 'ps', '-eo', 'pid,args'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        external['process_snapshot'] = {'checked_utc': utc_now(), 'exit_code': process.returncode,
            'stdout': process.stdout.decode('utf-8', 'replace'), 'stderr': process.stderr.decode('utf-8', 'replace'),
            'interpretation': 'Read-only snapshot; unrelated work is not stopped. Native refresh run exit 0 was separately observed by the coordinating agent; no global absence-of-writers claim.'}
        external['external_cleanup'] = {'status': 'VERIFIED_EXTERNAL_CLEANUP',
            'evidence_path': CLEANUP_EVIDENCE, 'evidence_sha256': CLEANUP_HASHES[CLEANUP_EVIDENCE],
            'inventory_path': CLEANUP_INVENTORY, 'inventory_sha256': CLEANUP_HASHES[CLEANUP_INVENTORY],
            'apply_result_path': str(CLEANUP_APPLY), 'apply_result_sha256': CLEANUP_HASHES[str(CLEANUP_APPLY)],
            'completed_utc': applied['completed_at'], 'deleted_file_count': 116,
            'removed_empty_directory_count': 232, 'deleted_bytes': 2056555438,
            'exact_deleted_files': files, 'exact_removed_directories': sorted(directories),
            'removed_groups': groups, 'missing_selected_count': len(missing_selected),
            'original_loss_check': 'All 116 missing known files in affected trees exactly match the fixed cleanup whitelist; original selected losses are its exact 51-path intersection',
            'cleanup_done_by_refresh_task': False}
        required.update(status='PASS',
            reason='Required immutable source/config/report/checkpoint/formal-result/dataset/query-cache and canonical runtime assets are preserved in the checked scope. Original inventory equality remains FAIL due to independently documented derivative cleanup, new concurrent files, Git timestamp changes and the disclosed 103-byte raw native aggregate anomaly.')
    except (OSError, KeyError, TypeError, ValueError) as error:
        required['reason'] = str(error)
    audit.update(required_protected_artifacts=required, external_changes=external)
    write_json(audit_path, audit)
    return required


def self_check():
    assert excluded(EXPERIMENT + '/metrics/result.json')
    assert excluded('scripts/__pycache__/refresh_ablation_message.cpython-311.pyc')
    assert not excluded('scripts/v2_message.py')
    assert not excluded('experiments/v2/metrics/final_summary.json')
    assert should_hash('experiments/v2/checkpoints/mlp/final.pth', 20000000)
    assert should_hash('query_cache/val/manifest.json', 5000000)
    assert not should_hash('query_cache/val/000902.pkl', 1000000)
    assert not should_hash('code/CoopTrack/.git/objects/ab/1234', 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--phase', choices=('before', 'after', 'required-assets'), required=True)
    args = parser.parse_args()
    self_check()
    if args.phase == 'before':
        result = create_before(args.root)
    elif args.phase == 'after':
        result = verify(args.root)
    else:
        result = record_required_assets(args.root)
    print(json.dumps({'phase': args.phase, 'status': result.get('status', 'BASELINE_RECORDED'),
                      'sha256_file_count': result.get('protected_sha256_file_count',
                                                    len(result.get('sha256_files', {}))),
                      'metrics_dir': str(metrics_dir(args.root))}))
    if result.get('status') == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
