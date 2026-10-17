"""CPU-only benchmark comparison, exact scalar checks and conditional training estimates."""
import argparse
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import pickle
import random
import statistics

DURATIONS = {'seconds', 'elapsed_seconds', 'duration_seconds', 'elapsed_ms',
             'data_ms', 'forward_ms', 'backward_ms', 'optimizer_ms'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def flatten(value, prefix=''):
    if isinstance(value, dict):
        if not value:
            return {prefix: {}}
        return {name: item for key in sorted(value) if key not in DURATIONS
                for name, item in flatten(value[key], prefix + '.' + key).items()}
    if isinstance(value, list):
        result = {prefix + '.length': len(value)}
        for index, item in enumerate(value):
            result.update(flatten(item, prefix + '[%d]' % index))
        return result
    return {prefix: value}


def compare_logs(old_path, new_path, expected):
    rows = [[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            for path in (old_path, new_path)]
    coverage = all(len(items) == expected and [item['step'] for item in items] == list(range(1, expected + 1))
                   and all(item['epoch'] == 1 for item in items) for items in rows)
    old, new = map(flatten, rows)
    mismatches = [{'field': key, 'old': old.get(key, '<missing>'), 'new': new.get(key, '<missing>')}
                  for key in sorted(old.keys() | new.keys()) if key not in old or key not in new or old[key] != new[key]]
    return {'status': 'PASS' if coverage and not mismatches else 'FAIL',
            'old_rows': len(rows[0]), 'new_rows': len(rows[1]), 'expected_rows': expected,
            'step_coverage_equal': coverage, 'compared_scalar_fields': len(old.keys() & new.keys()),
            'mismatch_count': len(mismatches), 'first_mismatches': mismatches[:20],
            'ignored_duration_keys': sorted(DURATIONS), 'comparison': 'exact equality; no numerical tolerance',
            'log_sha256': {'old': digest(old_path), 'new': digest(new_path)}}


def schedule(root):
    info_path = root / 'data_infos/V2X-Seq-SPD-CoopTrack-full/cooperative/spd_infos_temporal_train.pkl'
    with info_path.open('rb') as stream:
        infos = sorted(pickle.load(stream)['infos'], key=lambda row: row['timestamp'])
    scenes = {}
    for index, info in enumerate(infos):
        scenes.setdefault(str(info['scene_token']), []).append(index)
    if len(infos) != 7362 or len(scenes) != 46:
        raise ValueError('Expected the complete official 7362-sample, 46-scene training set')
    source = root / 'scripts/v2_train.py'
    function = next(node for node in ast.parse(source.read_text(encoding='utf-8')).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'scene_batches')
    namespace = {'deque': deque}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual-scene-batches>', 'exec'), namespace)
    counts = []
    for epoch in range(1, 37):
        order = sorted(scenes)
        random.Random(epoch - 1).shuffle(order)
        batches = list(namespace['scene_batches'](scenes, order, 4))
        if sorted(entry['index'] for batch in batches for entry in batch) != list(range(7362)):
            raise ValueError('Scene sampler did not cover the actual train set exactly once')
        counts.append(len(batches))
    return {'epoch_batches': counts, 'total_batches': sum(counts),
            'train_info_sha256': digest(info_path), 'scene_sampler_sha256': digest(source),
            'rule': 'actual scene_batches; sorted scenes shuffled by Random(epoch-1); four active scene lanes'}


def compare(root, old_path, new_path):
    old, new = read(old_path), read(new_path)
    keys = ('settings', 'initialization', 'normalization_sha256', 'data_identity', 'environment',
            'sample_order_sha256', 'sample_indices', 'warmup_batches', 'measured_batches',
            'first_measured_step', 'last_measured_step', 'epoch_total_batches', 'measured_frames',
            'diagnostic_steps', 'timing_scope')
    equal = {key: key in old and key in new and old[key] == new[key] for key in keys}
    shared = old['source_hashes'].keys() & new['source_hashes'].keys()
    equal['shared_dependency_hashes'] = bool(shared) and all(old['source_hashes'][key] == new['source_hashes'][key] for key in shared)
    equal['benchmark_passed'] = old['status'] == new['status'] == 'PASS'
    equal['gpu_sampling'] = (old['gpu']['status'] == new['gpu']['status'] == 'PASS'
                             and old['gpu']['interval_seconds'] == new['gpu']['interval_seconds']
                             and old['gpu']['device_index'] == new['gpu']['device_index'])
    logs = compare_logs(old_path.with_suffix('.train.jsonl'), new_path.with_suffix('.train.jsonl'),
                        old['warmup_batches'] + old['measured_batches'])
    performance = {}
    for label, row in (('old', old), ('new', new)):
        performance[label] = {'label': row['label'], 'measured_window_seconds': row['seconds'],
                              'batch_per_second': row['measured_batches'] / row['seconds'],
                              'seconds_per_batch': row['seconds'] / row['measured_batches'],
                              'median_batch_seconds': row['batch_seconds']['median'],
                              'p95_batch_seconds': row['batch_seconds']['p95'],
                              'frames_per_second': row['frames_per_second'],
                              'gpu_mean_percent': row['gpu']['statistics']['gpu_utilization_percent']['mean'],
                              'gpu_p95_percent': row['gpu']['statistics']['gpu_utilization_percent']['p95']}
    ratio = performance['new']['batch_per_second'] / performance['old']['batch_per_second']
    performance.update(speed_multiplier=ratio, throughput_increase_percent=(ratio - 1) * 100,
                       time_per_batch_reduction_percent=(1 - 1 / ratio) * 100)
    actual = schedule(root)
    historical = []
    for path in sorted((root / 'experiments/v4_budget_free/validation').glob('epoch*/result.json')):
        row = read(path)
        if row.get('status') == 'PASS' and row.get('sample_count') == 3316:
            historical.append({'path': str(path), 'seconds': row['seconds'],
                               'gpu': row['pairing_identity']['environment']['gpu']})
    mean_validation = statistics.mean(row['seconds'] for row in historical) if historical else None
    estimates = {}
    for label in ('old', 'new'):
        seconds = actual['total_batches'] / performance[label]['batch_per_second']
        validation = 13 * mean_validation if mean_validation is not None else None
        estimates[label] = {'training_hours': seconds / 3600,
                            'twelve_selection_validations_plus_best_hours': validation / 3600 if validation else None,
                            'total_hours': (seconds + validation) / 3600 if validation else None}
    return {'status': 'PASS' if all(equal.values()) and logs['status'] == 'PASS' else 'FAIL',
            'inputs': {'old': {'path': str(old_path), 'sha256': digest(old_path)},
                       'new': {'path': str(new_path), 'sha256': digest(new_path)}},
            'identity_equal': equal, 'scalar_training_logs': logs, 'performance': performance,
            'performance_interpretation': 'One pair of short windows measures an observed difference; it does not establish a repeatable speed benefit. Reverse-order repeated measurements are needed to assess variability.',
            'gpu_scope': old['gpu']['scope'], 'schedule': actual,
            'conditional_36_epoch_projection': estimates,
            'projection_limits': '64 timed early-epoch batches are a short window, not a full-run guarantee. Assumes this rate persists; later scenes, variable active lanes, thermal/load changes and diagnostics can alter throughput. Excludes construction, checkpoint saving and interruptions. Early stopping can reduce total epochs; update ETA after complete epochs.',
            'historical_validation': {'mean_seconds': mean_validation, 'runs': historical,
                                      'scope': 'old local full3316 validation: inference, official evaluators and input archives; excludes model/dataset construction; new C may differ'}}


def markdown(result):
    performance = result['performance']
    lines = ['# Structure performance comparison', '', 'Exact identity and scalar checks: ' + result['status'], '',
             '| Metric | Old | New |', '| --- | ---: | ---: |']
    for key in ('measured_window_seconds', 'batch_per_second', 'seconds_per_batch', 'median_batch_seconds', 'p95_batch_seconds',
                'frames_per_second', 'gpu_mean_percent', 'gpu_p95_percent'):
        values = [performance[label][key] for label in ('old', 'new')]
        lines.append('| %s | %s | %s |' % (key, *('NA' if value is None else '%.6f' % value for value in values)))
    lines += ['', 'Throughput change: %+.2f%%; time per batch reduction: %.2f%%.' % (
        performance['throughput_increase_percent'], performance['time_per_batch_reduction_percent']),
        '', result['performance_interpretation'],
        '', 'Exact scalar log comparison: %d differences across %d fields and %d steps.' % (
            result['scalar_training_logs']['mismatch_count'], result['scalar_training_logs']['compared_scalar_fields'],
            result['scalar_training_logs']['expected_rows']), '', result['gpu_scope'],
        'GPU readings come from nvidia-smi; no Windows Task Manager 3D metric is used.', '',
        'Actual 36-epoch schedule: %d batches.' % result['schedule']['total_batches'], '',
        json.dumps(result['conditional_36_epoch_projection'], indent=2), '', result['projection_limits'], '']
    return '\n'.join(lines)


def self_check():
    assert flatten({'loss': 1., 'seconds': 2.}) == {'.loss': 1.}
    assert flatten([{'loss': 1.}]) != flatten([{'loss': 1.000000000000001}])
    assert flatten({'empty': []}) != flatten({})
    return {'status': 'PASS', 'scope': 'CPU exact scalar comparison and duration exclusion'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--old', type=Path)
    parser.add_argument('--new', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), sort_keys=True))
    else:
        if any(value is None for value in (args.old, args.new, args.output)) or args.output.suffix != '.json':
            parser.error('--old, --new and --output ending in .json are required')
        if args.output.resolve() in (args.old.resolve(), args.new.resolve()):
            parser.error('Output must be separate from benchmark evidence')
        result = compare(args.root.resolve(), args.old.resolve(), args.new.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        args.output.with_suffix('.md').write_text(markdown(result), encoding='utf-8')
        print(json.dumps({'status': result['status'], 'performance': result['performance'],
                          'scalar_logs': result['scalar_training_logs'], 'output': str(args.output)}, allow_nan=False))
        raise SystemExit(0 if result['status'] == 'PASS' else 1)
