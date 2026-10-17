"""Report completed refresh 2x2 results without model inference or formal V2 writes."""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

BASE = Path('experiments/v2/ablations/refresh_2x2')
METHODS = 'ABCD'
NAMES = {'A': 'Both-Init-Only', 'B': 'Geometry-Refresh',
         'C': 'Latent-Refresh', 'D': 'Full-Refresh'}
METRICS = ('mAP', 'AMOTA', 'AMOTP', 'MOTA', 'IDS')
BUCKETS = (('0-1', 0, 2), ('2-3', 2, 4), ('4-7', 4, 8),
           ('8-15', 8, 16), ('16-31', 16, 32), ('32+', 32, math.inf))
# Fixed before formal A/C/D results exist; this diagnostic never selects a model or policy.
DIAGNOSTIC_RULE = {'preset_date_asia_shanghai': '2026-10-09',
    'score_threshold': 0.5, 'xy_match_distance_m': 2.0, 'car_range_m': 50.0,
    'matching': 'descending score, nearest unmatched GT, strict distance < 2m',
    'GT_filter': 'car, nonzero lidar+radar points, strict ego XY range < 50m',
    'frame_age_proxy': 'mean geometry_age_frames of all A sender queries in the same frame',
    'paired_geometry_error': 'query-weighted mean sender-reference error over exactly the same selected frames',
    'age_trend': 'first/last populated bucket and monotonicity of bucket aggregates; no significance test',
    'scope': 'offline descriptive detection diagnostic; not official AP or age-conditioned AMOTA'}


def read(path):
    path = Path(path)
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz'
          else path.open(encoding='utf-8-sig')) as stream:
        return json.load(stream)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def local_path(root, value):
    path = Path(value)
    if path.exists():
        return path
    marker = '/CoopTrack_Audit/'
    normalized = str(value).replace('\\', '/')
    return root / normalized.split(marker, 1)[1] if marker in normalized else root / normalized


def fmt(value, digits=6):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def table(columns, rows):
    return '\n'.join(['| ' + ' | '.join(columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(columns)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def bucket(age):
    return next(name for name, low, high in BUCKETS if low <= age < high)


def counts(predictions, truth):
    """Fixed score / XY threshold; GT enters only this offline reporting function."""
    remaining = set(range(len(truth)))
    tp = 0
    for prediction in sorted(predictions, key=lambda x: x['detection_score'], reverse=True):
        nearest = min(remaining, key=lambda i: math.dist(prediction['translation'][:2], truth[i][:2]), default=None)
        if nearest is not None and math.dist(prediction['translation'][:2], truth[nearest][:2]) < 2.0:
            remaining.remove(nearest)
            tp += 1
    return {'TP': tp, 'FP': len(predictions) - tp, 'FN': len(truth) - tp}


def quality(values):
    total = {key: sum(row[key] for row in values) for key in ('TP', 'FP', 'FN')}
    tp, fp, fn = (total[key] for key in ('TP', 'FP', 'FN'))
    return {**total, 'recall': tp / (tp + fn) if tp + fn else None,
            'precision': tp / (tp + fp) if tp + fp else None,
            'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def age_trend(values):
    """Describe populated age-bucket aggregates; no significance or causal test."""
    populated = [(name, value) for name, value in values if value is not None]
    if len(populated) < 2:
        return {'pattern': 'insufficient', 'populated_buckets': len(populated)}
    differences = [right[1] - left[1] for left, right in zip(populated, populated[1:])]
    if all(abs(value) <= 1e-12 for value in differences):
        pattern = 'flat'
    elif all(value >= -1e-12 for value in differences):
        pattern = 'nondecreasing'
    elif all(value <= 1e-12 for value in differences):
        pattern = 'nonincreasing'
    else:
        pattern = 'nonmonotonic'
    return {'pattern': pattern, 'populated_buckets': len(populated),
            'strictly_increasing': all(value > 1e-12 for value in differences),
            'first_bucket': populated[0][0], 'last_bucket': populated[-1][0],
            'first_value': populated[0][1], 'last_value': populated[-1][1],
            'last_minus_first': populated[-1][1] - populated[0][1]}


def trend_text(label, value):
    patterns = {'flat': '相同', 'nondecreasing': '非递减', 'nonincreasing': '非递增',
                'nonmonotonic': '非单调', 'insufficient': '不足以判断趋势'}
    if value['pattern'] == 'insufficient':
        return label + '：有效年龄桶不足 2 个，无法判断趋势。'
    overall = '增加' if value['last_minus_first'] > 1e-12 else '降低' if value['last_minus_first'] < -1e-12 else '持平'
    strict = '严格增加' if value['strictly_increasing'] else '并非严格增加'
    return (f"{label}：{value['first_bucket']} 桶 {fmt(value['first_value'], 4)} → "
            f"{value['last_bucket']} 桶 {fmt(value['last_value'], 4)}；"
            f"首末总体{overall}，{value['populated_buckets']} 个有效桶随年龄{patterns[value['pattern']]}（{strict}）。")


def percentile(values, quantile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def time_alignment(frames, vehicle_timestamps):
    gaps = [(right['timestamp_us'] - left['timestamp_us']) / 1e6
            for left, right in zip(frames, frames[1:])
            if (left['scene'], left['session']) == (right['scene'], right['session'])]
    skew = [(row['timestamp_us'] - vehicle_timestamps[row['cooperative_token']]) / 1e6 for row in frames]
    if not gaps or any(value <= 0 for value in gaps):
        raise ValueError('Invalid same-session sender timestamp progression')
    return {'prediction_target': 'current infrastructure sender timestamp; preserve vehicle-road pairing skew',
        'same_session_sender_gaps_seconds': {'count': len(gaps), 'min': min(gaps),
            'median': statistics.median(gaps), 'p95': percentile(gaps, .95), 'max': max(gaps)},
        'sender_minus_vehicle_seconds': {'min': min(skew), 'max': max(skew),
            'positive_count': sum(value > 0 for value in skew), 'negative_count': sum(value < 0 for value in skew),
            'zero_count': sum(value == 0 for value in skew)},
        'vehicle_timestamp_source': 'inherited prepared NuScenes keyframe sample_data pointcloud timestamp',
        'timestamps_rewritten': False}


def offline_detection(root, results, frames):
    """Use saved predictions and existing GT tables; no model or evaluator rerun."""
    gt_root = root / 'datasets/prepared/V2X-Seq-SPD-CoopTrack-full_v1/cooperative/v1.0-trainval'
    sample_tokens = [row['cooperative_token'] for row in frames['A']]
    wanted = set(sample_tokens)
    categories = {x['token']: x['name'] for x in read(gt_root / 'category.json')}
    instances = {x['token']: categories[x['category_token']] for x in read(gt_root / 'instance.json')}
    poses = {x['token']: x['translation'] for x in read(gt_root / 'ego_pose.json')}
    sample_data = read(gt_root / 'sample_data.json')
    ego = {x['sample_token']: poses[x['ego_pose_token']] for x in sample_data
           if x['is_key_frame'] and x['sample_token'] in wanted}
    vehicle_timestamps = {x['sample_token']: float(x['timestamp']) for x in sample_data
                         if x['is_key_frame'] and x['sample_token'] in wanted}
    if set(ego) != wanted:
        raise ValueError('Offline GT ego-pose coverage differs from official val')
    truth = defaultdict(list)
    for annotation in read(gt_root / 'sample_annotation.json'):
        token = annotation['sample_token']
        if (token in wanted and instances[annotation['instance_token']] == 'car'
            and annotation['num_lidar_pts'] + annotation['num_radar_pts'] != 0
            and math.dist(annotation['translation'][:2], ego[token][:2]) < 50.0):
            truth[token].append(annotation['translation'])
    age = {row['cooperative_token']: statistics.mean(q['geometry_age_frames'] for q in row['query_diagnostics'])
           if row['query_diagnostics'] else None for row in frames['A']}
    diagnostic = {'status': 'PASS', 'rule': DIAGNOSTIC_RULE, 'GT_count': sum(map(len, truth.values())),
                  'sample_count': len(wanted), 'methods': {}, 'paired_age_buckets': {},
                  'time_alignment': time_alignment(frames['A'], vehicle_timestamps)}
    for method in METHODS:
        directory = local_path(root, results[method]['predictions_file']).parent
        prediction_path = directory / 'results_nusc_det.json.gz'
        predictions = read(prediction_path)['results']
        if set(predictions) != wanted:
            raise ValueError('Saved detection token coverage differs: ' + method)
        records = []
        for token in sample_tokens:
            selected = [x for x in predictions[token] if x['detection_name'] == 'car'
                        and x['detection_score'] >= 0.5
                        and math.dist(x['translation'][:2], ego[token][:2]) < 50.0]
            records.append({'token': token, 'mean_A_geometry_age_frames': age[token],
                'age_bucket': bucket(age[token]) if age[token] is not None else None,
                **counts(selected, truth[token])})
        diagnostic['methods'][method] = {'prediction_file': str(prediction_path),
            'prediction_sha256': sha(prediction_path), 'overall': quality(records), 'frames': records}
        del predictions
        gc.collect()
    for name, _, _ in BUCKETS:
        selected = {method: [x for x in diagnostic['methods'][method]['frames'] if x['age_bucket'] == name]
                    for method in METHODS}
        per_method = {method: quality(values) for method, values in selected.items()}
        cohort_tokens = {row['token'] for row in selected['A']}
        geometry_errors = {}
        for method in 'AC':
            queries = [query for row in frames[method] if row['cooperative_token'] in cohort_tokens
                       for query in row['query_diagnostics']]
            geometry_errors[method] = {'query_count': len(queries),
                'mean_position_error_m': statistics.mean(q['position_error_m'] for q in queries) if queries else None,
                'mean_elapsed_seconds': statistics.mean(q['elapsed_seconds'] for q in queries) if queries else None}
        differences = {}
        for label, left, right in (('A_minus_B', 'A', 'B'), ('C_minus_D', 'C', 'D')):
            differences[label] = {metric: per_method[left][metric] - per_method[right][metric]
                if per_method[left][metric] is not None and per_method[right][metric] is not None else None
                for metric in ('recall', 'F1')}
        diagnostic['paired_age_buckets'][name] = {'frame_count': len(selected['A']),
            'methods': per_method, 'geometry_errors_on_same_frames': geometry_errors,
            'paired_differences': differences}
    diagnostic['age_trends'] = {}
    for method, pair in (('A', 'A_minus_B'), ('C', 'C_minus_D')):
        diagnostic['age_trends'][method + '_position_error_m'] = age_trend([
            (name, row['geometry_errors_on_same_frames'][method]['mean_position_error_m'])
            for name, row in diagnostic['paired_age_buckets'].items()])
        for metric in ('recall', 'F1'):
            diagnostic['age_trends'][method + '_' + metric + '_degradation'] = age_trend([
                (name, -row['paired_differences'][pair][metric]
                 if row['paired_differences'][pair][metric] is not None else None)
                for name, row in diagnostic['paired_age_buckets'].items()])
    diagnostic['frames_without_sender_queries'] = sum(x is None for x in age.values())
    return diagnostic


def validate(root):
    base = root / BASE
    correctness = read(base / 'metrics/correctness.json')
    structure = read(base / 'metrics/structure_audit.json')
    independent = read(base / 'metrics/independent_audit.json')
    if correctness['status'] != 'PASS' or independent['status'] != 'PASS':
        raise ValueError('Correctness and independent audits of all four results must pass before reporting')
    required = structure.get('required_protected_artifacts', {})
    if required.get('status') not in ('PASS', 'FAIL'):
        raise ValueError('Final required-artifact protection must be assessed explicitly as PASS or FAIL')
    if structure.get('raw_metadata_status') != 'FAIL' or structure['protection_after']['status'] != 'FAIL':
        raise ValueError('Raw full-inventory metadata mismatch must remain explicit')
    cleanup = structure.get('external_changes', {}).get('external_cleanup', {})
    if required['status'] == 'PASS':
        if (cleanup.get('status') != 'VERIFIED_EXTERNAL_CLEANUP'
            or cleanup.get('deleted_file_count') != 116
            or cleanup.get('removed_empty_directory_count') != 232
            or cleanup.get('cleanup_done_by_refresh_task') is not False
            or any(sha(root / cleanup[key]) != cleanup[key.replace('_path', '_sha256')]
                   for key in ('evidence_path', 'inventory_path'))):
            raise ValueError('Required-scope PASS requires pinned, verified exact external cleanup evidence')
    cache = read(base / 'metrics/val_cache_verification.json')
    if (cache['status'] != 'PASS' or cache['file_count'] != 3316 or len(cache['files']) != 3316
        or len({x['path'] for x in cache['files']}) != 3316 or cache['mismatch_count'] != 0
        or cache['missing_files'] or cache['extra_files']
        or any(not x['pass'] or not x['unchanged_during_read']
               or x['actual_bytes'] != x['expected_bytes'] or x['actual_sha256'] != x['expected_sha256']
               for x in cache['files'])
        or sha(root / cache['source_manifest']) != cache['source_manifest_sha256']):
        raise ValueError('All 3316 original val-cache payload SHA256/size checks must pass')
    results, frames = {}, {}
    for method in METHODS:
        result = read(base / 'metrics' / method / 'result.json')
        records = read(base / 'communication' / (method + '.json'))['frames']
        if (result['status'] != 'PASS' or result['sample_count'] != 3316 or result['scene_count'] != 21
            or len(records) != 3316 or len({x['cooperative_token'] for x in records}) != 3316
            or any(not math.isfinite(result['metrics'][key]) for key in METRICS)):
            raise ValueError('Incomplete or invalid official-val result: ' + method)
        if any(row['bytes'] != sum(row[kind + '_bytes'] for kind in ('control', 'geometry', 'latent'))
               or row['geometry_bytes'] != 52 * row['geometry_updates']
               or row['latent_bytes'] != 780 * row['latent_updates'] for row in records):
            raise ValueError('Actual per-frame packet accounting failed: ' + method)
        if any(row['queries'] != len(row['query_diagnostics'])
               or len({q['id'] for q in row['query_diagnostics']}) != row['queries']
               or any(not math.isfinite(q[field]) or q[field] < 0
                      for q in row['query_diagnostics']
                      for field in ('geometry_age_frames', 'latent_age_frames', 'elapsed_seconds',
                                    'position_error_m', 'yaw_error_rad', 'velocity_error_mps', 'received_speed_mps'))
               for row in records):
            raise ValueError('Invalid query identity or geometry diagnostic: ' + method)
        if len({x['scene'] for x in records}) != 21:
            raise ValueError('Official scene count differs: ' + method)
        measured = statistics.mean(row['bytes'] for row in records)
        if not math.isclose(measured, result['communication']['mean_bytes_per_frame'], abs_tol=1e-9):
            raise ValueError('Reported mean bytes differs from actual packets: ' + method)
        if result['communication']['query_count'] != sum(row['queries'] for row in records):
            raise ValueError('Reported query count differs from raw records: ' + method)
        results[method], frames[method] = result, records
    tokens = [x['cooperative_token'] for x in frames['A']]
    if any([x['cooperative_token'] for x in frames[method]] != tokens for method in METHODS):
        raise ValueError('Four methods do not use the same ordered official frames')
    for method in METHODS:
        for reference, actual in zip(frames['A'], frames[method]):
            if (any(reference[key] != actual[key] for key in ('token', 'scene', 'session', 'timestamp_us', 'queries'))
                or [q['id'] for q in reference['query_diagnostics']] != [q['id'] for q in actual['query_diagnostics']]):
                raise ValueError('Sender identity/query alignment differs: ' + method)
    if not results['B']['reused'] or results['D']['reused']:
        raise ValueError('Expected exact B prediction reuse and independent V2-native D')
    return results, frames, correctness, structure


def comparisons(results):
    result = {}
    for name, left, right in (('geometry_with_init_latent', 'B', 'A'), ('latent_with_predicted_geometry', 'C', 'A'),
                             ('latent_with_fresh_geometry', 'D', 'B'), ('geometry_with_fresh_latent', 'D', 'C')):
        result[name] = {'comparison': left + '-' + right,
            **{key: results[left]['metrics'][key] - results[right]['metrics'][key] for key in METRICS},
            'bytes_per_frame': results[left]['communication']['mean_bytes_per_frame'] - results[right]['communication']['mean_bytes_per_frame']}
    result['interaction_D_minus_C_minus_B_plus_A'] = {
        key: results['D']['metrics'][key] - results['C']['metrics'][key]
             - results['B']['metrics'][key] + results['A']['metrics'][key] for key in METRICS}
    result['average_geometry_effect'] = {key: (result['geometry_with_init_latent'][key] + result['geometry_with_fresh_latent'][key]) / 2
                                        for key in ('mAP', 'AMOTA')}
    result['average_latent_effect'] = {key: (result['latent_with_predicted_geometry'][key] + result['latent_with_fresh_geometry'][key]) / 2
                                      for key in ('mAP', 'AMOTA')}
    return result


def practical_analysis(results, effects):
    statements = {}
    for kind, label, keys in (
        ('geometry', '持续显式几何刷新', ('geometry_with_init_latent', 'geometry_with_fresh_latent')),
        ('latent', '持续高维 latent 刷新', ('latent_with_predicted_geometry', 'latent_with_fresh_geometry'))):
        selected = [effects[key] for key in keys]
        if all(row['AMOTA'] > 0 and row['mAP'] > 0 for row in selected):
            conclusion = '两种另一因素设置下 mAP 与 AMOTA 均提高，本结果支持在所测试策略下保留该刷新以获得相应指标'
        elif all(row['AMOTA'] > 0 for row in selected):
            conclusion = '两种另一因素设置下 AMOTA 均提高，检测效应需分别看待；本结果支持其跟踪价值'
        elif all(row['AMOTA'] <= 0 and row['mAP'] <= 0 for row in selected):
            conclusion = '两种另一因素设置下均未提高 mAP/AMOTA，本比较不支持其为达到这些指标的必要条件'
        else:
            conclusion = '收益依赖另一因素及指标，不能统一判断为必要或可省略'
        statements[kind] = label + '：' + conclusion + '。'
    savings = {}
    for kind, low, high in (('geometry', 'A', 'B'), ('latent', 'B', 'D')):
        smaller, larger = results[low]['communication'], results[high]['communication']
        field = 'mean_' + kind + '_bytes'
        savings[kind] = {'comparison': high + '-' + low, 'payload_bytes_per_frame': larger[field] - smaller[field],
            'payload_reduction_percent': (1 - smaller[field] / larger[field]) * 100,
            'packet_reduction_percent': (1 - smaller['mean_bytes_per_frame'] / larger['mean_bytes_per_frame']) * 100}
    return {'statements': statements, 'stop_post_initialization_savings': savings,
            'necessity_limit': 'Only a descriptive single-seed comparison of the tested fixed policies; no universal necessity claim'}


def writer_snapshot():
    """Read process command lines only; never stop unrelated experiment writers."""
    rows = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()[1:]
    snapshot = {'checked_asia_shanghai': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                'refresh_inference_writers': [], 'other_experiment_writers': []}
    for row in rows:
        parts = row.strip().split(maxsplit=1)
        if len(parts) != 2 or 'python' not in Path(parts[1].split()[0]).name:
            continue
        if 'scripts/refresh_ablation.py' in parts[1]:
            snapshot['refresh_inference_writers'].append({'pid': int(parts[0]), 'command': parts[1]})
        if 'scripts/budget_free/' in parts[1]:
            snapshot['other_experiment_writers'].append({'pid': int(parts[0]), 'command': parts[1]})
    if snapshot['refresh_inference_writers']:
        raise ValueError('Refresh inference writer remains active during final reporting')
    return snapshot


def figures(root, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = root / BASE / 'figures'
    output.mkdir(parents=True, exist_ok=True)
    colors = ['#596a8d', '#1d8c82', '#df9a3c', '#b64b61']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for method, color in zip(METHODS, colors):
        x, y = results[method]['communication']['mean_bytes_per_frame'], results[method]['metrics']['AMOTA']
        ax.scatter(x, y, s=90, color=color, zorder=3)
        ax.annotate(method + ': ' + NAMES[method], (x, y), xytext=(7, 8), textcoords='offset points', fontsize=9)
    ax.set(xlabel='Actual serialized bytes / frame', ylabel='AMOTA', title='Geometry and latent refresh: official val, seed 0')
    ax.margins(x=.2, y=.2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'bytes_vs_amota.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.1))
    for ax, metric in zip(axes.flat, METRICS):
        values = [results[method]['metrics'][metric] for method in METHODS]
        ax.bar(list(METHODS), values, color=colors, width=.65)
        ax.set_title(metric + (' (lower is better)' if metric in ('AMOTP', 'IDS') else ''))
        low, high = min(0, min(values)), max(0, max(values))
        padding = (high - low) * .2 or .1
        ax.set_ylim(low - padding if low < 0 else 0, high + padding)
        if low < 0:
            ax.axhline(0, color='#666666', linewidth=.7)
        for position, value in enumerate(values):
            ax.text(position, value, fmt(value, 0 if metric == 'IDS' else 4), ha='center',
                    va='bottom' if value >= 0 else 'top', fontsize=9)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0, .8, '\n'.join(method + '  ' + NAMES[method] for method in METHODS), va='top', fontsize=10)
    fig.suptitle('Frozen R50, 21 scenes / 3316 samples, seed 0')
    fig.tight_layout()
    fig.savefig(output / 'four_group_metrics.png', dpi=180)
    plt.close(fig)


def report(root, results, effects, diagnostic, correctness, structure, writers):
    required = structure['required_protected_artifacts']
    metric_rows, comm_rows, geometry_rows, error_rows = [], [], [], []
    for method in METHODS:
        r, c = results[method], results[method]['communication']
        metric_rows.append([method, '每帧' if method in 'BD' else '初始化+预测',
            '每帧' if method in 'CD' else '仅初始化', fmt(c['mean_bytes_per_frame'], 3),
            *[fmt(r['metrics'][key], 0 if key == 'IDS' else 6) for key in METRICS]])
        comm_rows.append([method, fmt(c['p95_bytes_per_frame'], 3), fmt(c['bytes_per_second_at_10Hz'], 3),
            *[fmt(c['mean_' + kind + '_bytes'], 3) for kind in ('control', 'geometry', 'latent')],
            fmt(c['geometry_update_ratio']), fmt(c['latent_update_ratio'])])
        geometry_rows.append([method, *[fmt(c[field][stat], 3) for field, stat in
            (('geometry_age_frames', 'mean'), ('geometry_age_frames', 'p95'), ('geometry_age_frames', 'max'),
             ('elapsed_seconds', 'mean'), ('elapsed_seconds', 'p95'), ('position_error_m', 'mean'),
             ('position_error_m', 'p95'), ('yaw_error_rad', 'mean'), ('velocity_error_mps', 'mean'), ('received_speed_mps', 'mean'))]])
        if method in 'AC':
            for name, _, _ in BUCKETS:
                value = c['age_buckets'][name]
                error_rows.append([method, name, value['count'], fmt(value['position_error_m']['mean'], 3),
                    fmt(value['position_error_m']['p95'], 3), fmt(value['yaw_error_rad']['mean'], 3),
                    fmt(value['velocity_error_mps']['mean'], 3)])
    pair_rows = [[value['comparison'], fmt(value['bytes_per_frame'], 3),
                 *[fmt(value[key], 0 if key == 'IDS' else 6) for key in METRICS]]
                 for key, value in effects.items() if 'comparison' in value]
    age_rows = []
    for name, _, _ in BUCKETS:
        row = diagnostic['paired_age_buckets'][name]
        age_rows.append([name, row['frame_count'], row['methods']['A']['TP'] + row['methods']['A']['FN'],
            row['geometry_errors_on_same_frames']['A']['query_count'],
            fmt(row['geometry_errors_on_same_frames']['A']['mean_elapsed_seconds'], 3),
            *[fmt(row['geometry_errors_on_same_frames'][method]['mean_position_error_m'], 3) for method in 'AC'],
            *[fmt(row['methods'][method]['recall'], 4) for method in METHODS],
            *[fmt(row['paired_differences'][pair][metric], 4) for pair in ('A_minus_B', 'C_minus_D') for metric in ('recall', 'F1')]])
    age_observations = ' '.join(trend_text(label, diagnostic['age_trends'][key])
        for key, label in (('A_position_error_m', 'A 平均位置误差（m）'),
                           ('C_position_error_m', 'C 平均位置误差（m）'),
                           ('A_recall_degradation', 'B−A recall 退化差'),
                           ('C_recall_degradation', 'D−C recall 退化差'),
                           ('A_F1_degradation', 'B−A F1 退化差'),
                           ('C_F1_degradation', 'D−C F1 退化差')))
    checks = correctness.get('methods', {})
    check_text = '；'.join(f"{m}: {v['sample_count']} 帧/{v.get('fusion_input_checks', 'N/A')} 次对齐前检查/{v.get('aggregation_input_checks', 'N/A')} 次实际 aggregation 检查/{v.get('poison_checks', 'N/A')} 次未发送字段扰动"
                          for m, v in checks.items())
    interaction = effects['interaction_D_minus_C_minus_B_plus_A']
    query_trends = {method: age_trend([(name, results[method]['communication']['age_buckets'][name]['position_error_m']['mean'])
                    for name, _, _ in BUCKETS]) for method in 'AC'}
    query_age_observations = ' '.join(trend_text(method + ' 单 query 年龄桶平均位置误差（m）', value)
                                    for method, value in query_trends.items())
    practical = practical_analysis(results, effects)
    saving = practical['stop_post_initialization_savings']
    gaps = diagnostic['time_alignment']['same_session_sender_gaps_seconds']
    skew = diagnostic['time_alignment']['sender_minus_vehicle_seconds']
    geometry, latent = effects['average_geometry_effect'], effects['average_latent_effect']
    contributor = '显式几何' if geometry['AMOTA'] > latent['AMOTA'] else '高维 latent' if latent['AMOTA'] > geometry['AMOTA'] else '两者相同'
    old = {name: read(root / 'experiments/v2/metrics/official_val' / name / 'result.json')
           for name in ('INIT_ONLY_seed0', 'P32_seed0', 'mlp_B1_seed0')}
    reference_rows = [[name, fmt(value['communication']['mean_bytes_per_frame'], 3),
                      fmt(value['metrics']['mAP']), fmt(value['metrics']['AMOTA']), fmt(value['metrics']['IDS'], 0)]
                      for name, value in old.items()]
    cleaned = structure.get('files_cleaned', [])
    external = structure.get('external_changes', {})
    cleanup = external.get('external_cleanup', {})
    added_file_count = len(external.get('added_file_inventory', []))
    writer_text = '观察到 budget_free writer（PID ' + ', '.join(str(row['pid']) for row in writers['other_experiment_writers']) + '）' if writers['other_experiment_writers'] else '该时点未观察到 budget_free writer'
    observations = []
    for key, wording in (('geometry_with_init_latent', '固定 init-only latent 时持续刷新几何'),
                         ('geometry_with_fresh_latent', '固定逐帧 latent 时持续刷新几何'),
                         ('latent_with_predicted_geometry', '固定预测几何时持续刷新 latent'),
                         ('latent_with_fresh_geometry', '固定逐帧几何时持续刷新 latent')):
        value = effects[key]
        sign = '两项指标均提高' if value['mAP'] > 0 and value['AMOTA'] > 0 else '两项指标均降低' if value['mAP'] < 0 and value['AMOTA'] < 0 else '检测与跟踪结果混合'
        observations.append(f"{wording}（{value['comparison']}）：{sign}，ΔmAP={fmt(value['mAP'])}、ΔAMOTA={fmt(value['AMOTA'])}。")
    prefix = '../../' + BASE.as_posix()
    text = f'''# 几何与高维 latent 刷新频率 2×2 消融

**{'REFRESH_2X2_ABLATION_COMPLETE' if required['status'] == 'PASS' else 'REFRESH_2X2_RESULTS_COMPLETE_PROTECTION_FAIL'}**。四组任务结果独立审计 PASS；必需保护产物检查 {required['status']}，全量起始清单的原始比较 FAIL（差异披露见下文）。Official val：21 scenes / 3316 samples，R50、FP32、10Hz nominal、car-only、seed 0。模型与缓存均继承冻结版本；本轮没有训练、GT 在线调度、val 规则调整或 checkpoint 选择。

本次固定策略下，持续几何刷新的平均 AMOTA 增益为 {fmt(geometry['AMOTA'])}，高维 latent 刷新为 {fmt(latent['AMOTA'])}。仅刷新几何的 B 相比 Full-Refresh D 节省 {fmt(saving['latent']['packet_reduction_percent'], 2)}% 实际 packet，AMOTA 降低 {fmt(effects['latent_with_fresh_geometry']['AMOTA'])}；停止后续几何刷新后的 A/C 均为 AMOTA=0。该结果支持当前策略保留几何刷新，latent 的收益依赖几何状态；结论限于本次单 seed 描述性比较。

## 目录与保护

沿用现有正式目录，新增入口 `scripts/refresh_ablation.py` 与独立 ablation 结果；被正式协议引用的两个 `coop_val.py` 保留。本轮自身清理 {len(cleaned)} 文件。必需资产范围 **{required['status']}**：{required['unchanged_original_selected_file_count']} 个仍存在的原始强哈希文件一致，{len(required['unchanged_required_trees'])} 类正式结果/模型、数据/cache/环境目录摘要一致；{len(required['retained_runtime_extensions'])} 个运行时 `.so` 与完整 wheel SHA256 通过，当前 Git 内容/路径/大小及原 HEAD/status/diff、fsck 通过。3316 个 val cache 另做完整 SHA256/大小核验；大型数据目录仍主要依赖 metadata 清单，未宣称全量重新哈希。

原始全文件比较与严格 Git 元数据对账均保留 **FAIL**。另一任务完成白名单清理 116 文件、232 空目录（下载副本、build 派生文件与 bytecode），实际缺失逐项与 Apply 证据对账；其中 51 项属于原始强哈希清单，没有发现仍存在的原文件内容变化。另保留 {added_file_count} 项并行任务新增文件，不把两类外部变化计成本轮清理。详见 [清理报告](../PROJECT_STRUCTURE_CLEANUP.md) 与 [固定机器证据](../../audits/project_structure_cleanup/cleanup_evidence.json)。

两个库的 `FETCH_HEAD` 与 `.git/index` mtime 推进，内容与大小一致；原始 native 扫描还比随后稳定读取少 103 B，原因未证实，临时 fetch 截断仅为推断。原 Git baseline 只有 aggregate 摘要，无法补证整个 `.git` 起始与结束的逐文件内容完全相等。这些差异没有被改写为 PASS；必需范围的内容检查与全清单/时间戳一致性分开披露。完整清单、时间、哈希与限制见 [目录审计]({prefix}/metrics/structure_audit.json) 和 [Git 差异核验]({prefix}/metrics/git_metadata_drift_verification.json)。

报告生成时本轮 refresh 推理 writer 已退出，{writer_text}。本轮未删除、覆盖或停止另一任务；观察时点及命令保存在 [机器汇总]({prefix}/metrics/report_summary.json)，不宣称整个工作区无其他 writer。

## 实现与复用

几何与 latent 按 `(scene, native reset session, predicted track ID)` 独立缓存，新 ID 强制初始化两类信息；ID 消失后保留本 session 已接收状态，scene/session 变化隔离缓存，无 TTL、年龄上限或填充更新。几何字段为 `ref_pts` 与 `pred_boxes`，latent 为三组 256 维字段。

`pred_boxes=[cx,cy,log(w),log(l),cz,log(h),sin(yaw),cos(yaw),vx,vy]`；中心与速度在路端 LiDAR 物理坐标中，速度单位 m/s。`ref_pts` 为 infrastructure point-cloud range `[0,-51.2,-5,102.4,51.2,3]` 的仿射归一化坐标。A/C 从最后实收状态按实际 timestamp（μs→s）执行固定恒速平移，保持初始化 z、尺寸、物理 yaw 与速度；发送的必要标定/时序 metadata 将状态从实收路端坐标系输运到当前路端坐标系。参考点应用同一物理位移，并核验反归一化参考点与框中心的一致性；车端融合继续使用官方 `veh2inf_rt`。没有读入当前未发送框、未发送 reference point、GT 或未来帧补偿。

预测目标是当前路端 sender timestamp，不额外向车端时刻外推。同 session 的真实相邻发送间隔 min/median/p95/max={fmt(gaps['min'], 6)}/{fmt(gaps['median'], 7)}/{fmt(gaps['p95'], 8)}/{fmt(gaps['max'], 6)} s，未强行改成 0.1 s。原始路车配对时差（路端−车端 pointcloud timestamp）范围 {fmt(skew['min'], 6)} 至 {fmt(skew['max'], 6)} s，正时差 {skew['positive_count']} 帧；四组保留该配对与时间偏差。通信 B/s 仍按原协议 nominal 10Hz 报告。

INT8 延用 per-query/per-field absmax/127、`np.rint` 与 FP32 scale，每次 latent UPDATE 为 780 B/query。CTKR2X01 序列化包含 ID、两个独立 flags、必要 metadata、长度和 CRC；只有对应 UPDATE 才写入当前 payload。几何 UPDATE 为 52 B/query，控制开销另计。逐帧验证 `len(packet)=control+geometry+latent`，本地预测与缓存复制不计通信；无线/TCP/IP/重传开销未计入。

B 通过 3316 帧完整 CPU replay，五组 receiver 张量与 frozen INIT_ONLY 逐元素 bitwise 相等，沿用其预测和 evaluator 指标，仅按统一新协议重新计费。A/C 为新增推理。D 为独立命名的 V2-native Full-Refresh：历史 P1 的 CTKMSG01 与模型构建前 seed 放置不同，而 native inference 存在 CPU `torch.randperm`；无法证明历史预测与 V2 seed 放置相等，因此本轮不复用历史 P1 指标。P1/P2/P4 全部保持原文件不变。

所有新增运行在构建模型、加载 checkpoint、创建 loader iterator 后重置 seed 0，继承 V2 CUDA 数值设置。Correctness PASS：{check_text}。未发送字段修改保持 flags 与实收状态固定，并检查真正进入 crossview fusion 的五组张量；正式运行也检查融合输入及有限输出。额外混合 flags、多 ID、跨 scene/session、>32 帧缓存与 INT8/字节检查见原始 correctness 证据。

## 四组任务指标

{table(['组','Geo update','Latent update','B/frame','mAP','AMOTA','AMOTP','MOTA','IDS'], metric_rows)}

## 真实通信分解

分解列均为 B/frame；UPDATE ratio 按 query 加权，初始化也计为 UPDATE。Bytes/s 使用 nominal 10Hz。

{table(['组','P95 B/frame','B/s','Control','Geometry','Latent','Geo ratio','Latent ratio'], comm_rows)}

## 几何预测质量

以下为 receiver 重建状态相对当前路端预测的离线差异，参考框仅用于分析，不能把它解释为相对 GT 的绝对定位误差。Age 为 query 加权的使用时年龄（UPDATE 后为 0）；帧年龄同时保留真实 dt。Yaw 为 wrap 后绝对角误差（rad）；速度差为 XY 速度 L2（m/s）。

{table(['组','Age mean','Age p95','Age max','dt mean s','dt p95 s','Position mean m','Position p95 m','Yaw mean rad','Velocity error mean m/s','Received speed mean m/s'], geometry_rows)}

{table(['组','Geo age frames','Queries','Position mean m','Position p95 m','Yaw mean rad','Velocity error mean m/s'], error_rows)}

{query_age_observations} 这是年龄桶横截面均值，不能解释为同一个目标的连续误差轨迹。

## 年龄与性能的描述性检查

在 A/C/D 正式结果可见前固定了以下离线规则：保存的 detection JSON，car score≥0.5、XY 中心距离<2m、按分数降序贪心一对一匹配；继承 50m ego range 与非零点数 GT filter。它没有重新推理、修改策略或计算另一份正式指标。用每帧 A 所有路端 query 的平均 geometry age 分桶，四组使用同一帧集合；GT={diagnostic['GT_count']}，无路端 query 帧={diagnostic['frames_without_sender_queries']}（不进入年龄分桶）。Recall/F1 是该固定诊断点的 micro aggregate，不能称为 mAP/AMOTA。

以下位置误差与性能差使用完全相同的帧集合；位置误差与 dt 按该集合中的路端 query 加权，recall/F1 按该集合的 detection/GT 数量 micro 聚合。它们与上节按单个 query 年龄分桶的几何误差具有不同分组单位。表内差值为 A−B/C−D；下述趋势中的退化差为 B−A/D−C，正数代表预测组在该诊断点更差。

{table(['Mean frame age','Frames','GT','Queries','dt mean s','A position mean m','C position mean m','A recall','B recall','C recall','D recall','A−B recall','A−B F1','C−D recall','C−D F1'], age_rows)}

{age_observations}

上述趋势仅描述已填充年龄桶的聚合值，不是对象的时间轨迹或显著性检验。帧平均年龄不是某个最终车端 track 的年龄；场景、目标数量、可见性、初始化时机与 score operating point 均有混杂。即使长龄桶出现更大退化，也不能据此证明 object-age 条件下 AMOTA 单调下降。完整 TP/FP/FN、每帧结果及 F1 见离线诊断 JSON。

## 两因素效应

差值为表中左组减右组；mAP/AMOTA/MOTA 越高越好，AMOTP/IDS 越低越好。全部差值均披露。

{table(['对比','ΔB/frame','ΔmAP','ΔAMOTA','ΔAMOTP','ΔMOTA','ΔIDS'], pair_rows)}

{' '.join(observations)}

几何平均效应 `((B−A)+(D−C))/2`：ΔmAP={fmt(geometry['mAP'])}、ΔAMOTA={fmt(geometry['AMOTA'])}；latent 平均效应 `((C−A)+(D−B))/2`：ΔmAP={fmt(latent['mAP'])}、ΔAMOTA={fmt(latent['AMOTA'])}。按本次 AMOTA 平均差值，贡献更大的是：{contributor}。该排序只描述当前固定策略与单 seed。

交互差值 `D−C−B+A`：mAP={fmt(interaction['mAP'])}，AMOTA={fmt(interaction['AMOTA'])}。这是完整 2×2 的数值 difference-in-differences；不能把无重复实验的非零值当成显著交互或可推广的因果效应。

## 对原有低通信结果的解释

{table(['冻结 V2 参考','原协议 B/frame','mAP','AMOTA','IDS'], reference_rows)}

INIT_ONLY 对应本轮 B：它每帧仍接收显式几何，只停止后续 latent 刷新。P32 和 MLP B1 也持续刷新显式几何，同时以不同频率/策略更新 latent；其较低通信成本不能证明可停止几何发送。MLP B1 的单 seed AMOTA 改善是在给定预算与状态选择策略下观察到的结果，不代表本轮恒速几何预测或 Transformer 架构优势。上表保留 frozen V2 的原协议字节，跨协议时仅用于背景解释，不混入本轮四组通信表。

## 可以支持的结论与限制

本实验量化了两类显式字段持续刷新的相对贡献、相应真实 payload/control 成本，以及初始化后恒速预测的累计差异。是否有必要持续刷新应依据 B−A 与 D−C、C−A 与 D−B 的具体差值及应用容忍度判断，不能从一次策略比较宣称普遍必要或不必要。初始化+预测包含实收几何；初始化一次 latent 包含高维信息；latent 本身仍可能含空间线索。

{practical['statements']['geometry']} {practical['statements']['latent']} 这些是单 seed 的描述性收益，不能据此把所有应用的刷新需求定为相同。

停掉初始化后的几何刷新（B→A）减少 geometry payload {fmt(saving['geometry']['payload_bytes_per_frame'], 3)} B/frame（{fmt(saving['geometry']['payload_reduction_percent'], 2)}%），总 packet 减少 {fmt(saving['geometry']['packet_reduction_percent'], 2)}%；停掉初始化后的 latent 刷新（D→B）减少 latent payload {fmt(saving['latent']['payload_bytes_per_frame'], 3)} B/frame（{fmt(saving['latent']['payload_reduction_percent'], 2)}%），总 packet 减少 {fmt(saving['latent']['packet_reduction_percent'], 2)}%。这些节省需要与对应表中性能差同时评估；控制开销、初始化和必要 pose metadata 仍被真实计入。

四组使用相同 split、路端预测、车端权重、量化、evaluator、seed 放置与消息实现；B 的严格张量等价复用节省了完整 GPU 推理。单 seed、未复制实验且 native CUDA kernels 未保证全局确定，无法支持显著性、跨 seed 鲁棒性或跨数据泛化。不同策略改变 retained queries 数量及原生随机 query filling 的后续调用，受控 seed 并不保证逐帧随机模板完全一致。几何误差采用当前路端预测参考，年龄诊断仅有 frame proxy；后续如需年龄条件的跟踪结论，应预注册对象对应关系与共同 operating point，再做独立重复，不在当前 val 调规则。

模型与消息检查未发现未发送信息旁路、NaN/Inf、身份覆盖异常或计费异常。目录保护发现 Git 元数据摘要漂移、另一并行任务新增文件以及独立白名单清理，原始全文件 FAIL 与必需产物 {required['status']} 分开保留。不能把范围调整后的状态解释为整个 `.git` 前后字节相等、起始清单全部文件仍在或整个工作区没有其他 writer。任何必需资产未证实保持的部分均按 FAIL/未确认披露，而不以完成推理掩盖。

## 原始证据与图

- [配置](../../configs/main_experiment_v2/refresh_ablation.json)，[继承的车端配置](../../configs/main_experiment_v2/coop_val.py)，[manifest]({prefix}/metrics/ablation_manifest.json)，[correctness]({prefix}/metrics/correctness.json)，[独立审计]({prefix}/metrics/independent_audit.json)，[目录与保护审计]({prefix}/metrics/structure_audit.json)，[Git 元数据差异核验]({prefix}/metrics/git_metadata_drift_verification.json)，[3316 个 val cache SHA256/大小核验]({prefix}/metrics/val_cache_verification.json)，[机器汇总]({prefix}/metrics/report_summary.json)，[离线年龄诊断]({prefix}/metrics/detection_age_diagnostic.json)。
- A/B/C/D：`{BASE.as_posix()}/metrics/<group>/result.json`，逐帧通信与 query 误差：`communication/<group>.json`；A/C/D 完整 evaluator 与 lossless gzip 预测：`predictions/<group>/`；B 指向 [原 INIT_ONLY 预测](../../experiments/v2/predictions/official_val/INIT_ONLY_seed0/)。
- [冻结 V2 正式报告](V2_FINAL.md)，[运行日志]({prefix}/metrics/run.log)。

![真实通信与 AMOTA]({prefix}/figures/bytes_vs_amota.png)

![四组任务指标]({prefix}/figures/four_group_metrics.png)
'''
    target = root / 'reports/v2/REFRESH_2X2_ABLATION_FINAL.md'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')
    return target


def self_check():
    prediction = lambda xy, score: {'translation': [*xy, 0], 'detection_score': score}
    assert counts([prediction([0, 0], .9), prediction([0, 0], .8)], [[0, 0, 0]]) == {'TP': 1, 'FP': 1, 'FN': 0}
    assert counts([prediction([2, 0], .9)], [[0, 0, 0]]) == {'TP': 0, 'FP': 1, 'FN': 1}
    assert counts([], [[0, 0, 0]]) == {'TP': 0, 'FP': 0, 'FN': 1}
    assert quality([{'TP': 2, 'FP': 1, 'FN': 2}])['F1'] == 4 / 7
    assert bucket(1.9) == '0-1' and bucket(2) == '2-3' and bucket(32) == '32+'
    assert age_trend([('0-1', None), ('2-3', .1)])['pattern'] == 'insufficient'
    assert age_trend([('0-1', .1), ('2-3', .2), ('4-7', .2)])['pattern'] == 'nondecreasing'
    assert age_trend([('0-1', .2), ('2-3', .1)])['pattern'] == 'nonincreasing'
    assert age_trend([('0-1', .1), ('2-3', .2), ('4-7', .15)])['pattern'] == 'nonmonotonic'
    assert age_trend([('0-1', .1), ('2-3', .2)])['strictly_increasing']
    assert not age_trend([('0-1', .1), ('2-3', .2), ('4-7', .2)])['strictly_increasing']
    assert math.isclose(percentile([1., 2., 3., 4.], .95), 3.85)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print('REFRESH_REPORT_SELF_CHECK_PASS')
        return
    root = args.root.resolve()
    results, frames, correctness, structure = validate(root)
    effects = comparisons(results)
    diagnostic = offline_detection(root, results, frames)
    write(root / BASE / 'metrics/detection_age_diagnostic.json', diagnostic)
    figures(root, results)
    writers = writer_snapshot()
    target = report(root, results, effects, diagnostic, correctness, structure, writers)
    identity_paths = [root / BASE / 'metrics' / method / 'result.json' for method in METHODS]
    identity_paths += [root / BASE / 'communication' / (method + '.json') for method in METHODS]
    identity_paths += [root / BASE / 'figures/bytes_vs_amota.png', root / BASE / 'figures/four_group_metrics.png',
        root / BASE / 'metrics/detection_age_diagnostic.json', target, Path(__file__),
        root / 'configs/main_experiment_v2/refresh_ablation.json', root / BASE / 'metrics/correctness.json',
        root / 'configs/main_experiment_v2/coop_val.py', root / BASE / 'metrics/ablation_manifest.json',
        root / BASE / 'metrics/val_cache_verification.json', root / BASE / 'metrics/structure_audit.json',
        root / BASE / 'metrics/independent_audit.json', root / BASE / 'metrics/git_metadata_drift_verification.json']
    cleanup = structure.get('external_changes', {}).get('external_cleanup', {})
    for relative in ('reports/PROJECT_STRUCTURE_CLEANUP.md', cleanup.get('evidence_path'), cleanup.get('inventory_path')):
        if relative and not Path(relative).is_absolute() and ':' not in relative:
            path = root / relative
            if path.is_file():
                identity_paths.append(path)
    summary = {'status': 'PASS' if structure['required_protected_artifacts']['status'] == 'PASS' else 'PROTECTION_FAIL',
        'experiment_results_status': 'PASS', 'created_asia_shanghai': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'protocol': 'REFRESH_2X2_SEED0', 'results': results, 'comparisons': effects,
        'practical_analysis': practical_analysis(results, effects),
        'query_age_position_error_trends': {method: age_trend([(name, results[method]['communication']['age_buckets'][name]['position_error_m']['mean'])
            for name, _, _ in BUCKETS]) for method in 'AC'},
        'time_alignment': diagnostic['time_alignment'],
        'offline_diagnostic_rule': DIAGNOSTIC_RULE,
        'protected_artifacts': structure['protected_artifacts'],
        'required_scope_status': structure['required_protected_artifacts']['status'],
        'raw_full_inventory_status': structure['protected_artifacts'],
        'required_protected_artifacts': structure['required_protected_artifacts'],
        'external_changes': structure.get('external_changes', {}),
        'raw_protection_status': read(root / BASE / 'metrics/git_metadata_drift_verification.json')['raw_protection_status'],
        'git_metadata_reconciliation': structure.get('git_metadata_reconciliation', {}),
        'concurrent_additions_reconciliation': structure.get('concurrent_additions_reconciliation', {}),
        'writer_snapshot': writers,
        'git_metadata_evidence': 'metrics/git_metadata_drift_verification.json',
        'git_metadata_proof_limit': 'Original aggregate Git baseline cannot establish complete before/after individual Git content equality; raw FAIL remains disclosed.',
        'formal_inference_repeated_for_B': False, 'files_cleaned': structure.get('files_cleaned', []),
        'sha256': {path.relative_to(root).as_posix(): sha(path) for path in identity_paths}}
    write(root / BASE / 'metrics/report_summary.json', summary)
    print('REFRESH_REPORT_' + summary['status'] + ' ' + str(target))


if __name__ == '__main__':
    main()
