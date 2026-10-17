"""Report full-46-scene, fixed-12-epoch V2 results from real official-val evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import main_runtime as runtime

METRICS = ('mAP', 'AMOTA', 'AMOTP', 'MOTA', 'IDS')
BUDGETS = {'B1': 3474, 'B2': 6382, 'B3': 9290, 'B4': 12198}
PERIODIC = ('P6', 'P8', 'P10', 'P12', 'P16', 'P24', 'P32', 'INIT_ONLY')
BUCKETS = ('0-1', '2-3', '4-7', '8-15', '16-31', '32+')
PROTOCOL = 'full46_fixed12_official_val'


def read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else {}


def digest(path):
    path = Path(path)
    if not path.is_file():
        return None
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def local_path(root, value):
    if not value:
        return None
    normalized = str(value).replace('\\', '/')
    for prefix in ('/mnt/d/CoopTrack_Audit/', 'D:/CoopTrack_Audit/'):
        if normalized.lower().startswith(prefix.lower()):
            return root / normalized[len(prefix):]
    path = Path(value)
    return path if path.is_absolute() else root / path


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalized_buckets(values):
    """Merge optional ages zero and one by counts, excluding initialization."""
    groups = {'0-1': ('0-1', 'elapsed=0', '[1,2)'), '2-3': ('2-3', '[2,4)'),
              '4-7': ('4-7', '[4,8)'), '8-15': ('8-15', '[8,16)'),
              '16-31': ('16-31', '[16,32)'), '32+': ('32+', '[32,+inf)')}
    output = {}
    for label, aliases in groups.items():
        entries = [values[label]] if label in values else [values[key] for key in aliases if key in values]
        if not entries:
            continue
        count = sum(item.get('count', 0) for item in entries)
        updates = sum(item.get('update_count', 0) for item in entries)
        weighted = [item for item in entries if item.get('count', 0)]
        probability = (sum(item['mean_probability'] * item['count'] for item in weighted) / count
                       if count and all(number(item.get('mean_probability')) for item in weighted) else None)
        output[label] = {'count': count, 'update_count': updates, 'mean_probability': probability,
                         'optional_update_ratio': updates / count if count else None}
    return output


def run_row(result, path, frames):
    identity, comm = result.get('identity', {}), result['communication']
    query_count = sum(frame['queries'] for frame in frames)
    updates = sum(frame['updates'] for frame in frames)
    forced = sum(frame['new_init_count'] for frame in frames)
    optional = sum(frame['optional_queries'] for frame in frames)
    refresh = sum(frame['refresh_count'] for frame in frames)
    histogram = comm.get('optional_probability_histogram')
    histogram_counts = histogram.get('counts') if isinstance(histogram, dict) else histogram
    row = {'method': result['method'], 'budget_name': identity.get('budget_name'),
           'target_budget': identity.get('budget'), 'seed': result.get('seed', identity.get('seed')),
           'origin': 'new_formal_v2', 'status': 'PASS_COMPLETE_RUN', 'split': 'val',
           'sample_count': 3316, 'scene_count': 21, 'period_frames': result.get('period_frames'),
           'mean_bytes_per_frame': statistics.mean(frame['bytes'] for frame in frames),
           'p95_bytes_per_frame': comm.get('p95_bytes_per_frame'),
           'mean_mandatory_bytes': statistics.mean(frame['mandatory_bytes'] for frame in frames),
           'mean_latent_update_bytes': updates * 780 / len(frames),
           'query_count': query_count, 'update_count': updates, 'new_init_count': forced,
           'optional_query_count': optional, 'optional_refresh_count': refresh,
           'hard_UPDATE_ratio': updates / query_count if query_count else None,
           'forced_init_ratio': forced / query_count if query_count else None,
           'optional_learned_update_ratio': refresh / optional if optional and result['method'] in ('mlp', 'transformer') else None,
           'optional_periodic_update_ratio': refresh / optional if optional and result['method'] not in ('mlp', 'transformer') else None,
           'mean_init_bytes': forced * 780 / len(frames), 'mean_optional_refresh_bytes': refresh * 780 / len(frames),
           'optional_probability_histogram': histogram_counts,
           'optional_probability_histogram_metadata': histogram if isinstance(histogram, dict) else None,
           'staleness_before_decision_buckets': normalized_buckets(comm.get('staleness_before_decision_buckets', {})),
           'checkpoint_identity': identity.get('checkpoint_sha256'),
           'protocol_manifest_sha256': identity.get('protocol_manifest_sha256'),
           'source_identity': identity.get('sources', {}), 'result_path': str(path),
           'predictions_file': result.get('predictions_file'), 'communication_file': result.get('communication_file'),
           'runtime_seconds': result.get('runtime_seconds'), 'timing_scope': result.get('timing_scope')}
    row.update({name: result['metrics'][name] for name in METRICS})
    row['bytes_per_second_at_10Hz'] = row['mean_bytes_per_frame'] * 10
    row['communication_reduction_vs_historical_P1'] = 1 - row['mean_bytes_per_frame'] / 12161.645355850422
    for field in ('staleness', 'inter_update_seconds'):
        row.update({field + '_' + name: comm.get(field, {}).get(name) for name in ('mean', 'median', 'p95', 'max')})
    for key in ('mean_S_soft', 'mean_S_hard', 'mean_soft_hard_gap', 'overflow_frame_count',
                'max_frame_overflow', 'mean_unavoidable_init_overflow', 'scene_reset_count', 'native_session_reset_count'):
        row[key] = comm.get(key)
    row['mean_budget_constraint_pass'] = (row['mean_bytes_per_frame'] <= 1.05 * row['target_budget']
                                          if number(row['target_budget']) else None)
    return row


def packet_issue(result, evidence):
    frames = evidence.get('frames', [])
    keys = ('bytes', 'mandatory_bytes', 'queries', 'updates', 'new_init_count', 'optional_queries', 'refresh_count')
    measured = result.get('communication', {}).get('mean_bytes_per_frame')
    if evidence.get('identity') != result.get('identity'):
        return 'packet and result identities differ'
    if len(frames) != 3316 or not all(all(number(frame.get(key)) for key in keys) for frame in frames):
        return 'complete actual packet accounting missing'
    if not number(measured) or not math.isclose(statistics.mean(frame['bytes'] for frame in frames), measured, rel_tol=1e-10, abs_tol=1e-7):
        return 'reported mean bytes differs from actual packets'
    if not all(frame['bytes'] == frame['mandatory_bytes'] + 780 * frame['updates'] for frame in frames):
        return 'actual packets violate serializer M + 780*updates accounting'
    if not all(0 <= frame['new_init_count'] <= frame['updates'] <= frame['queries']
               and frame['optional_queries'] == frame['queries'] - frame['new_init_count']
               and frame['refresh_count'] == frame['updates'] - frame['new_init_count'] for frame in frames):
        return 'initialization and optional query/update counts are inconsistent'
    if len({frame.get('cooperative_token') for frame in frames}) != 3316:
        return 'distinct full official-val cooperative token coverage missing'
    return None


def load_runs(root, manifest):
    rows, excluded = [], []
    directory = root / 'experiments/v2/metrics/official_val'
    for path in sorted(directory.rglob('result.json')) if directory.is_dir() else []:
        result, reason = read(path), None
        method, identity = result.get('method'), result.get('identity', {})
        if result.get('status') != 'PASS':
            reason = 'status is not PASS'
        elif result.get('split') != 'val' or result.get('sample_count') != 3316 or result.get('scene_count') != 21:
            reason = 'full official-val 3316-sample / 21-scene coverage missing'
        elif not all(number(result.get('metrics', {}).get(name)) for name in METRICS):
            reason = 'native car-only task metrics missing'
        elif method not in PERIODIC + ('mlp', 'transformer') or result.get('seed') != 0:
            reason = 'not a preregistered seed-0 formal run'
        elif identity.get('protocol_manifest_sha256') != digest(root / 'experiments/v2/metrics/FINAL_PROTOCOL_MANIFEST.json'):
            reason = 'evaluation is not bound to the frozen formal manifest'
        if reason is None:
            try:
                started = datetime.fromisoformat(result['started_utc'].replace('Z', '+00:00'))
                frozen = datetime.fromisoformat(manifest['frozen_utc'].replace('Z', '+00:00'))
                if started < frozen:
                    reason = 'official-val evaluation started before formal protocol freeze'
            except (KeyError, TypeError, ValueError):
                reason = 'formal freeze / evaluation start timestamp missing or invalid'
        if reason is None and method in ('mlp', 'transformer'):
            entry = manifest.get('final_checkpoints', {}).get(method, {})
            if identity.get('checkpoint_sha256') != entry.get('sha256') or not entry.get('sha256'):
                reason = 'evaluation checkpoint differs from the frozen epoch-12 final checkpoint'
            elif identity.get('budget_name') not in BUDGETS or identity.get('budget') != BUDGETS.get(identity.get('budget_name')):
                reason = 'evaluation budget differs from the locked four budgets'
        if reason is None and method in PERIODIC:
            period = int(method[1:]) if method.startswith('P') else None
            if result.get('period_frames') != period:
                reason = 'Periodic interval differs from the preregistered method'
        predictions, communication = local_path(root, result.get('predictions_file')), local_path(root, result.get('communication_file'))
        if reason is None and (predictions is None or not predictions.is_file() or communication is None or not communication.is_file()):
            reason = 'actual prediction or communication evidence file missing'
        frames = []
        if reason is None:
            evidence = read(communication)
            reason = packet_issue(result, evidence)
            frames = evidence.get('frames', [])
        if reason:
            excluded.append({'path': str(path), 'reason': reason})
        else:
            rows.append(run_row(result, path, frames))
    groups = {}
    for row in rows:
        groups.setdefault((row['method'], row['budget_name'], row['seed']), []).append(row)
    rows = []
    for candidates in groups.values():
        if len(candidates) == 1:
            rows.extend(candidates)
        else:
            excluded.extend({'path': row['result_path'], 'reason': 'duplicate formal method/budget/seed result; no metric-based selection'} for row in candidates)
    return rows, excluded


def inherited_rows(root):
    reused, output = read(root / 'configs/main_experiment_v2/reused_val.json'), []
    for method in ('P1', 'P2', 'P4'):
        item = reused.get('methods', {}).get(method, {})
        if item.get('sample_count') != 3316 or item.get('scene_count') != 21 or not item.get('status', '').startswith('REUSED_'):
            continue
        proof_path = local_path(root, item.get('metric_path'))
        proof = read(proof_path) if proof_path else {}
        if proof.get('status') != 'PASS':
            continue
        track = read(proof_path.parent / 'track/metrics_summary.json')
        metrics = {key: item.get('metrics', {}).get(key, track.get(key.lower())) for key in METRICS}
        comm, measured = item.get('communication', {}), item.get('communication', {}).get('mean_bytes_per_frame')
        if not all(number(value) for value in metrics.values()) or not number(measured):
            continue
        if method == 'P1' and not math.isclose(measured, 12161.645355850422, rel_tol=0, abs_tol=1e-9):
            continue
        output.append({'method': method, 'budget_name': None, 'target_budget': None, 'seed': 0,
                       'origin': 'inherited_official_val_reference', 'status': 'REUSED', 'split': 'val',
                       'sample_count': 3316, 'scene_count': 21, 'period_frames': int(method[1:]),
                       'mean_bytes_per_frame': measured, 'p95_bytes_per_frame': comm.get('p95_bytes_per_frame'),
                       'bytes_per_second_at_10Hz': measured * 10, **metrics,
                       'hard_UPDATE_ratio': comm.get('update_ratio'), 'forced_init_ratio': None,
                       'staleness_before_decision_buckets': {}, 'optional_probability_histogram': None,
                       'serializer_scope': item.get('serializer_difference', 'historical APTU0001'),
                       'inference_seed_scope': item.get('limitation', 'historical RNG placement differs'),
                       'source_identity': item.get('inherited_identity', {}), 'result_path': str(proof_path),
                       'predictions_file': item.get('prediction_path'), 'new_model_forward_executed': False})
    return output


def find(rows, method, budget=None):
    return next((row for row in rows if row['method'] == method and row.get('budget_name') == budget and row['seed'] == 0), None)


def comparison(left, right):
    if left is None or right is None:
        return {'status': 'NOT_EVALUATED'}
    relative = abs(left['mean_bytes_per_frame'] - right['mean_bytes_per_frame']) / right['mean_bytes_per_frame']
    better = left['AMOTA'] >= right['AMOTA'] and left['mAP'] >= right['mAP'] and (left['AMOTA'] > right['AMOTA'] or left['mAP'] > right['mAP'])
    return {'status': 'MATCHED_RATE' if relative <= .05 else 'RATE_MISMATCH',
            'relative_actual_rate_difference': relative, 'rate_match': relative <= .05,
            'delta_mAP': left['mAP'] - right['mAP'], 'delta_AMOTA': left['AMOTA'] - right['AMOTA'],
            'numerical_both_metrics_nonworse_one_better': better,
            'single_seed_same_rate_advantage_observed': relative <= .05 and better,
            'historical_reference': right['origin'] == 'inherited_official_val_reference',
            'statistical_robustness_established': False}


def periodic_neighbors(adaptive, periodic):
    if not periodic:
        return {'nearest': None, 'lower_bracket': None, 'upper_bracket': None}
    ordered = sorted(periodic, key=lambda row: (row['mean_bytes_per_frame'], row['method']))
    value = adaptive['mean_bytes_per_frame']
    lower = [row for row in ordered if row['mean_bytes_per_frame'] < value]
    upper = [row for row in ordered if row['mean_bytes_per_frame'] > value]
    # Byte-only deterministic matching; task metrics never choose a reference.
    nearest = min(ordered, key=lambda row: (abs(row['mean_bytes_per_frame'] - value), row['mean_bytes_per_frame'], row['method']))
    return {'nearest': nearest, 'lower_bracket': lower[-1] if lower else None, 'upper_bracket': upper[0] if upper else None}


def training_issues(root, variant, result, manifest):
    problems, epochs = [], result.get('epochs', [])
    if result.get('status') != 'PASS' or result.get('phase') != 'train':
        problems.append(variant + ' full official-train training PASS missing')
    if result.get('train_samples') != 7362 or result.get('train_scenes') != 46:
        problems.append(variant + ' training is not full 46 scenes / 7362 samples')
    if [epoch.get('epoch') for epoch in epochs] != list(range(1, 13)) or any(epoch.get('frames') != 7362 or epoch.get('scenes') != 46 for epoch in epochs):
        problems.append(variant + ' exactly 12 complete full-train epochs missing')
    if result.get('completed_epoch') != 12 or result.get('cosine_total_horizon_epochs') != 12 or result.get('training_seed') != 0:
        problems.append(variant + ' fixed final epoch / 12-epoch LR horizon / seed-0 evidence differs')
    if any(not all(number(epoch.get('means', {}).get(key)) for key in ('task', 'rate', 'total', 'bytes'))
           or not number(epoch.get('samples_per_second')) or not number(epoch.get('peak_vram_bytes')) for epoch in epochs):
        problems.append(variant + ' full per-epoch finite loss / throughput / VRAM evidence missing')
    if result.get('native_batch_size') != 4 or result.get('effective_optimizer_batch_size') != 4:
        problems.append(variant + ' genuine multi-scene optimizer batch 4 evidence missing')
    if not result.get('task_gradient_verified') or result.get('base_trainable_params') != 0:
        problems.append(variant + ' native task gradient / frozen-backbone evidence missing')
    entry = manifest.get('final_checkpoints', {}).get(variant, {})
    path = local_path(root, entry.get('path')) or root / ('experiments/v2/checkpoints/' + variant + '/final.pth')
    if not entry.get('sha256') or digest(path) != entry.get('sha256'):
        problems.append(variant + ' frozen epoch-12 final checkpoint SHA256 missing or mismatched')
    return problems


def manifest_issues(root, manifest):
    problems = []
    for key, expected in (('protocol', PROTOCOL), ('epochs', 12), ('batch', 4), ('threshold', .5), ('seed', 0), ('budgets', list(BUDGETS.values()))):
        if manifest.get(key) != expected:
            problems.append('FINAL_PROTOCOL_MANIFEST ' + key + ' differs from fixed formal protocol')
    if set(manifest.get('periodic_points', [])) != set(PERIODIC):
        problems.append('FINAL_PROTOCOL_MANIFEST all eight preregistered Periodic points missing')
    if not manifest.get('frozen_utc') or not manifest.get('source_hashes'):
        problems.append('FINAL_PROTOCOL_MANIFEST freeze timestamp / source hashes missing')
    for key in ('optimizer', 'lr_schedule', 'serializer_version', 'evaluator_version'):
        if not manifest.get(key):
            problems.append('FINAL_PROTOCOL_MANIFEST ' + key + ' missing')
    if manifest.get('cooptrack_commit', manifest.get('backbone_commit')) != runtime.COMMIT:
        problems.append('FINAL_PROTOCOL_MANIFEST CoopTrack commit differs')
    for name, expected in manifest.get('source_hashes', {}).items():
        path = root / name if '/' in name else root / 'scripts' / name
        if digest(path) != expected:
            problems.append('frozen source changed: ' + name)
    return problems


def completion_status(missing):
    return 'INCOMPLETE' if missing else 'V2_FORMAL_EXPERIMENT_COMPLETE'


def plots(root, rows, training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = root / 'experiments/v2/figures'
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for metric in ('AMOTA', 'mAP'):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name in ('periodic', 'mlp', 'transformer'):
            points = sorted([row for row in rows if row['origin'] == 'new_formal_v2'
                             and (row['method'] in PERIODIC if name == 'periodic' else row['method'] == name)],
                            key=lambda row: row['mean_bytes_per_frame'])
            if points:
                ax.plot([row['mean_bytes_per_frame'] for row in points], [row[metric] for row in points], 'o-', label=name)
        inherited = [row for row in rows if row['origin'] == 'inherited_official_val_reference']
        if inherited:
            ax.scatter([row['mean_bytes_per_frame'] for row in inherited], [row[metric] for row in inherited],
                       marker='x', color='gray', label='Historical P1/P2/P4; RNG/serializer scope differs')
        for row in rows:
            ax.annotate(row['method'] + (' ' + row['budget_name'] if row.get('budget_name') else ''),
                        (row['mean_bytes_per_frame'], row[metric]), fontsize=7)
        ax.set(xlabel='Actual serialized bytes/frame', ylabel=metric, title='Official val: 21 scenes / 3316 samples, seed 0')
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = directory / ('official_val_bytes_vs_' + metric.lower() + '.png')
        fig.savefig(str(path), dpi=160)
        plt.close(fig)
        paths.append(str(path))
    for variant in ('mlp', 'transformer'):
        measured = [row for row in rows if row['method'] == variant]
        final = next((epoch for epoch in training[variant].get('epochs', []) if epoch.get('epoch') == 12), {})
        series = [(row['budget_name'], row.get('optional_probability_histogram'), row.get('staleness_before_decision_buckets', {})) for row in measured]
        series.append(('train epoch 12', final.get('optional_probability_histogram'), normalized_buckets(final.get('staleness_before_decision_buckets', {}))))
        for kind in ('histogram', 'staleness'):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            present = False
            for label, histogram, buckets in series:
                if kind == 'histogram' and isinstance(histogram, list) and len(histogram) == 10 and sum(histogram):
                    present = True
                    ax.plot([.05 + index / 10 for index in range(10)], [count / sum(histogram) for count in histogram], 'o-', label=label)
                elif kind == 'staleness':
                    values = [buckets.get(bucket, {}).get('mean_probability') for bucket in BUCKETS]
                    if any(number(value) for value in values):
                        present = True
                        ax.plot(range(6), [value if number(value) else math.nan for value in values], 'o-', label=label + ' mean p')
                        hard = [buckets.get(bucket, {}).get('optional_update_ratio') for bucket in BUCKETS]
                        ax.plot(range(6), [value if number(value) else math.nan for value in hard], 'x--', label=label + ' hard UPDATE')
            if present:
                ax.set(xlabel='p(update), ten fixed bins' if kind == 'histogram' else 'Staleness before decision (frames)',
                       ylabel='Optional query fraction' if kind == 'histogram' else 'Mean probability / UPDATE ratio',
                       title=variant + '; initialization excluded', ylim=(0, 1))
                if kind == 'staleness':
                    ax.set(xticks=range(6), xticklabels=BUCKETS)
                ax.legend(fontsize=7)
                fig.tight_layout()
                path = directory / ('formal_' + variant + '_' + kind + '.png')
                fig.savefig(str(path), dpi=160)
                paths.append(str(path))
            plt.close(fig)
    return paths


def display(value):
    return ('%.6f' % value) if number(value) else 'N/A' if value is None else str(value)


def report(root):
    root = Path(root)
    cleanup = root / 'reports/v2/CLEANUP_RESET.md'
    if not cleanup.is_file():
        raise RuntimeError('CLEANUP_DONE evidence missing; no formal reports may be generated yet')
    metrics_dir = root / 'experiments/v2/metrics'
    manifest = read(metrics_dir / 'FINAL_PROTOCOL_MANIFEST.json')
    state = read(metrics_dir / 'run_state.json')
    training = {variant: read(metrics_dir / (variant + '_train.json')) for variant in ('mlp', 'transformer')}
    fresh, excluded = load_runs(root, manifest)
    inherited = inherited_rows(root)
    rows = inherited + fresh
    periodic = [row for row in rows if row['method'] in PERIODIC + ('P1', 'P2', 'P4')]
    adaptive = [row for row in rows if row['method'] in ('mlp', 'transformer')]
    missing = manifest_issues(root, manifest)
    for variant in ('mlp', 'transformer'):
        missing += training_issues(root, variant, training[variant], manifest)
        for budget in BUDGETS:
            if find(rows, variant, budget) is None:
                missing.append('official val ' + variant + ' ' + budget + ' complete PASS missing')
    for method in ('P1', 'P2', 'P4') + PERIODIC:
        if find(rows, method) is None:
            missing.append('official val ' + method + ' complete/reused evidence missing')
    comparisons, comparison_rows = [], []
    for row in adaptive:
        neighbors = periodic_neighbors(row, periodic)
        comparisons.append({'adaptive': row, **neighbors,
                            'comparisons': {role: comparison(row, point) for role, point in neighbors.items()}})
        for role, point in neighbors.items():
            comparison_rows.append({'method': row['method'], 'budget_name': row['budget_name'], 'seed': 0,
                                    'adaptive_actual_bytes': row['mean_bytes_per_frame'], 'periodic_role': role,
                                    'periodic_method': point['method'] if point else None,
                                    'periodic_actual_bytes': point['mean_bytes_per_frame'] if point else None,
                                    **comparison(row, point)})
    relation = [{'budget': budget, 'transformer': find(rows, 'transformer', budget),
                 'mlp': find(rows, 'mlp', budget),
                 'comparison': comparison(find(rows, 'transformer', budget), find(rows, 'mlp', budget))} for budget in BUDGETS]
    failed = [row for row in adaptive if row['mean_budget_constraint_pass'] is False]
    diagnostic_missing = []
    for row in adaptive:
        for key in ('p95_bytes_per_frame', 'staleness_mean', 'staleness_median', 'staleness_p95', 'staleness_max', 'optional_probability_histogram'):
            if row.get(key) is None:
                diagnostic_missing.append(row['method'] + ' ' + row['budget_name'] + ': ' + key + ' NOT_RECORDED')
        absent = [bucket for bucket in BUCKETS if bucket not in row['staleness_before_decision_buckets']]
        if absent:
            diagnostic_missing.append(row['method'] + ' ' + row['budget_name'] + ': staleness buckets missing ' + ','.join(absent))
        histogram = row.get('optional_probability_histogram')
        if not isinstance(histogram, list) or len(histogram) != 10 or any(not number(value) or value < 0 for value in histogram) or sum(histogram) != row['optional_query_count']:
            diagnostic_missing.append(row['method'] + ' ' + row['budget_name'] + ': complete optional-query histogram count differs')
        if sum(item['count'] for item in row['staleness_before_decision_buckets'].values()) != row['optional_query_count']:
            diagnostic_missing.append(row['method'] + ' ' + row['budget_name'] + ': complete optional-query staleness bucket count differs')
    missing += diagnostic_missing
    status = completion_status(missing)
    constraint = 'COMMUNICATION_CONSTRAINT_FAIL' if failed else 'PASS' if len(adaptive) == 8 else 'INCOMPLETE'
    verdict = {'adaptive_refresh_advantage': 'LEARNED_SCHEDULER_ADVANTAGE_NOT_ESTABLISHED' if len(adaptive) == 8 and len(periodic) == 11 else 'NOT_EVALUATED',
               'transformer_over_mlp': 'TRANSFORMER_ADVANTAGE_NOT_ESTABLISHED' if len(adaptive) == 8 else 'NOT_EVALUATED',
               'communication_constraint': constraint,
               'communication_reduction': [{'method': row['method'], 'budget': row['budget_name'],
                                            'actual_bytes': row['mean_bytes_per_frame'],
                                            'fraction_vs_historical_P1': row['communication_reduction_vs_historical_P1']} for row in adaptive],
               'single_seed_numeric_trends': comparison_rows, 'statistical_robustness_established': False,
               'explanation': 'Seed 0 is the fixed primary result. Numeric differences do not establish stable replicated improvement.'}
    summary = {'status': status, 'missing': missing, 'generated_utc': datetime.now(timezone.utc).isoformat(), 'protocol': PROTOCOL,
               'formal_protocol': {'train_scenes': 46, 'train_samples': 7362, 'internal_dev': 'NONE',
                                   'epochs_mlp': 12, 'epochs_transformer': 12, 'batch': 4, 'threshold': .5,
                                   'primary_seed': 0, 'budgets': BUDGETS, 'checkpoint_rule': 'epoch 12 final, no validation selection'},
               'manifest': manifest, 'manifest_sha256': digest(metrics_dir / 'FINAL_PROTOCOL_MANIFEST.json'),
               'training': training, 'official_val_results': rows, 'nearest_bracket_comparisons': comparisons,
               'transformer_vs_mlp_same_budget': relation, 'verdict': verdict,
               'limitations': ['P1 is inherited FEAT_INT8, measured historical CTKMSG01 bytes; P2/P4 inherit historical RNG placement.',
                               'All nearest and lower/upper Periodic references are selected by actual bytes alone, including unfavorable task results.',
                               'Matched-rate flag uses a descriptive 5% actual-byte tolerance; nominal budgets never replace actual costs.',
                               'One training seed is available; no robustness, significance, or independent replicate claim.',
                               'AMOTP and IDS are lower-is-better; other task metrics are higher-is-better.',
                               'Bytes/s assumes 10Hz; cached replay runtime excludes online infrastructure-image inference.',
                               'Unrecorded historical communication/staleness diagnostics remain N/A.'],
               'run_state_status': state.get('status'), 'completed_stages': state.get('completed', []),
               'excluded_runs': excluded, 'exact_blocker': state.get('error'),
               'minimal_next_action': 'Resolve the recorded blocker and complete only unfinished phases under the fixed protocol.' if missing else None,
               'evidence': {'cleanup': str(cleanup), 'run_state': str(metrics_dir / 'run_state.json'),
                            'protocol_manifest': str(metrics_dir / 'FINAL_PROTOCOL_MANIFEST.json')}}
    figures = plots(root, rows, training) if rows else []
    summary['figures'] = figures
    runtime.write(metrics_dir / 'final_summary.json', summary)
    lines = ['# V2 formal experiment', '', '**' + status + '**', '',
             '46 official train scenes / 7362 samples; fixed 12 epochs per scheduler; epoch-12 final checkpoints. Official val: 21 scenes / 3316 samples.',
             'The fixed primary seed is 0. All costs are actual serialized bytes; no validation checkpoint selection or model adjustment is applied.', '',
             '| Method | Budget | Evidence | Actual B/frame | P95 B/frame | B/s (10Hz) | mAP | AMOTA | AMOTP | MOTA | IDS |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in periodic + adaptive:
        lines.append('| ' + ' | '.join([row['method'], row.get('budget_name') or 'N/A', row['origin']]
                                      + [display(row.get(key)) for key in ('mean_bytes_per_frame', 'p95_bytes_per_frame', 'bytes_per_second_at_10Hz') + METRICS]) + ' |')
    lines += ['', 'Every adaptive budget is compared with the nearest actual-byte Periodic point and both adjacent brackets. Missing brackets are N/A; every comparison is retained.', '',
              '| Adaptive | Budget | Role | Periodic | Adaptive bytes | Periodic bytes | Relative gap | ΔmAP | ΔAMOTA | Rate match |',
              '|---|---|---|---|---:|---:|---:|---:|---:|---|']
    for row in comparison_rows:
        lines.append('| ' + ' | '.join(display(row.get(key)) for key in ('method', 'budget_name', 'periodic_role', 'periodic_method',
                                                                       'adaptive_actual_bytes', 'periodic_actual_bytes', 'relative_actual_rate_difference',
                                                                       'delta_mAP', 'delta_AMOTA', 'rate_match')) + ' |')
    lines += ['', 'Full communication accounting, staleness summaries, six buckets and ten-bin optional p(update) histograms appear in the evidence below.',
              'Historical P1 retains its original measured 12161.645355850422 B/frame. Its serializer differs from new V2, and inherited P1/P2/P4 RNG placement differs.', '',
              'Seed 0 cannot establish statistical robustness. Favorable single-seed observations remain descriptive; unfavorable observations remain in the same tables.', '',
              '```json', json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), '```']
    for path in figures:
        lines += ['', '![' + Path(path).stem + '](../../experiments/v2/figures/' + Path(path).name + ')']
    output = root / 'reports/v2/V2_FINAL.md'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('V2_FORMAL_REPORT status=%s complete_new_val_runs=%s' % (status, len(fresh)), flush=True)
    return summary


def self_check():
    def point(method, cost, accuracy):
        return {'method': method, 'mean_bytes_per_frame': cost, 'mAP': accuracy, 'AMOTA': accuracy, 'origin': 'new_formal_v2'}
    references = [point('P32', 80, .5), point('P16', 100, .8), point('P8', 130, .1)]
    adaptive = point('mlp', 110, .3)
    neighbors = periodic_neighbors(adaptive, references)
    assert neighbors['nearest']['method'] == 'P16' and neighbors['lower_bracket']['method'] == 'P16'
    assert neighbors['upper_bracket']['method'] == 'P8'
    assert not comparison(adaptive, neighbors['nearest'])['single_seed_same_rate_advantage_observed']
    assert comparison(point('mlp', 100, .9), references[1])['single_seed_same_rate_advantage_observed']
    assert not comparison(point('mlp', 100, .9), references[1])['statistical_robustness_established']
    assert completion_status(['missing']) == 'INCOMPLETE'
    assert completion_status([]) == 'V2_FORMAL_EXPERIMENT_COMPLETE'
    merged = normalized_buckets({'elapsed=0': {'count': 1, 'update_count': 0, 'mean_probability': .1},
                                 '[1,2)': {'count': 3, 'update_count': 2, 'mean_probability': .5},
                                 'new_initialization': {'count': 99, 'update_count': 99}})
    assert merged['0-1']['count'] == 4 and math.isclose(merged['0-1']['mean_probability'], .4)
    assert merged['0-1']['optional_update_ratio'] == .5 and len(merged) == 1
    assert packet_issue({'identity': {}, 'communication': {}}, {'identity': {}, 'frames': []})
    frames = [dict(bytes=881, mandatory_bytes=101, queries=1, updates=1, new_init_count=1,
                   optional_queries=0, refresh_count=0, cooperative_token=str(index)) for index in range(3316)]
    result = {'identity': {}, 'communication': {'mean_bytes_per_frame': 881}}
    assert packet_issue(result, {'identity': {}, 'frames': frames}) is None
    result.update(method='mlp', metrics={key: .1 for key in METRICS},
                  communication=dict(mean_bytes_per_frame=881, optional_probability_histogram={'counts': [0] * 10, 'edges': [index / 10 for index in range(11)]}))
    assert run_row(result, Path('result.json'), frames)['optional_probability_histogram'] == [0] * 10
    assert local_path(Path('/workspace'), 'experiments/final.pth') == Path('/workspace/experiments/final.pth')
    frames[-1]['bytes'] = 882
    assert packet_issue(result, {'identity': {}, 'frames': frames})
    print('Formal V2 report actual-byte matching/status/bucket/packet checks PASS; no report artifacts generated')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/mnt/d/CoopTrack_Audit'))
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        runtime.initialize(args.root)
        report(args.root)
