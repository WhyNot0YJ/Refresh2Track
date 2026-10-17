"""Report only complete seed-zero evaluations; pending branches have no scores."""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runtime

LABELS = {
    'no_explicit_delta': 'w/o Explicit Delta',
    'no_receiver_state': 'w/o Receiver-State Conditioning',
    'no_communication_loss': 'w/o Communication Loss',
}
COLUMNS = ('Method', 'Status', 'Input dimensions (effective)', 'Feature BPS',
           'Serialized BPS', 'Optional UPDATE %', 'mAP', 'AMOTA', 'MOTA', 'IDS')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def metric_row(label, dimensions, result, status):
    row = dict.fromkeys(COLUMNS)
    row.update({'Method': label, 'Status': status, 'Input dimensions (effective)': dimensions})
    if result:
        keys = {'Feature BPS': 'feature_payload_BPS', 'Serialized BPS': 'serialized_BPS',
                'mAP': 'mAP', 'AMOTA': 'AMOTA', 'MOTA': 'MOTA', 'IDS': 'IDS'}
        row.update({column: result[key] for column, key in keys.items()})
        ratio = result.get('optional_update_ratio')
        row['Optional UPDATE %'] = 100 * ratio if ratio is not None else None
    return row


@contextmanager
def report_lock(root):
    """Serialize every shared snapshot and its fixed atomic-write names."""
    import fcntl
    path = Path(root) / 'experiments/v4_ablations/report.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def report(root):
    with report_lock(root):
        return _report(root)


def _report(root):
    root = Path(root).resolve()
    reference = runtime.verify_reference(root)
    baseline_path = runtime.reference(root) / 'final_validation.json'
    baseline = runtime.read(baseline_path)
    rows = [row for row in baseline['rows'] if row['method'] == 'V4' and row['seed'] == 0]
    if baseline.get('status') != 'PASS' or len(rows) != 1 or rows[0]['sample_count'] != 3316 or rows[0]['scene_count'] != 21:
        raise ValueError('The actual frozen V4 seed-zero baseline is incomplete')
    table = [metric_row('V4 Full (reference)', 2324, rows[0], 'COMPLETE')]
    experiments = root / 'experiments/v4_ablations'
    reports = root / 'reports/ablations'
    reports.mkdir(parents=True, exist_ok=True)
    branches = {}
    for mode in ('no_explicit_delta', 'no_receiver_state', 'no_communication_loss'):
        directory = experiments / mode
        state_path = directory / 'run_state.json'
        state = runtime.read(state_path) if state_path.exists() else {'status': 'PENDING', 'phase': 'pending'}
        selection_path = directory / 'metrics/best_selection.json'
        selection = runtime.read(selection_path) if selection_path.exists() else {}
        history_path = directory / 'metrics/train_history.json'
        history = runtime.read(history_path) if history_path.exists() else []
        manifest_path = directory / 'manifest.json'
        manifest = runtime.read(manifest_path) if manifest_path.exists() else None
        summary = {'mode': mode, 'method': LABELS[mode], 'status': state['status'],
                   'phase': state.get('phase'), 'completed_epoch': history[-1]['epoch'] if history else 0,
                   'coordinator_pid': state.get('coordinator_pid'), 'execution_mode': state.get('execution_mode'),
                   'last_state_utc': state.get('utc'), 'state': state,
                   'best_epoch': selection.get('best_epoch'), 'checkpoint_sha256': selection.get('checkpoint_sha256'),
                   'training_seed': 0, 'inference_seeds': [0], 'official_val_use':
                   'checkpoint selection, early stopping and final reporting; not an independent test',
                   'config': str(directory / 'config.json'), 'checkpoint': str(directory / 'checkpoints/best.pth'),
                   'train_log': str(root / 'logs/refresh2track_ablations' / mode / 'train.log'),
                   'updated_utc': timestamp()}
        final_path = directory / 'metrics/final_validation.json'
        result = None
        if final_path.exists():
            final = runtime.read(final_path)
            candidates = final.get('rows', [])
            frozen = runtime.read(directory / 'metrics/frozen_selection.json')
            if (final.get('status') != 'PASS' or len(candidates) != 1 or
                    candidates[0].get('method') != 'V4' or candidates[0].get('seed') != 0 or
                    candidates[0].get('sample_count') != 3316 or candidates[0].get('scene_count') != 21 or
                    final.get('checkpoint_sha256') != frozen['checkpoint_sha256'] or
                    candidates[0].get('checkpoint_sha256') != frozen['checkpoint_sha256'] or
                    final.get('identity') != frozen['identity'] or not manifest or
                    frozen['identity']['manifest_sha256'] != runtime.digest(manifest_path) or
                    runtime.digest(directory / 'checkpoints/best.pth') != frozen['checkpoint_sha256']):
                raise ValueError('Incomplete or incompatible final ablation result: ' + mode)
            result = candidates[0]
            summary.update(status='COMPLETE', phase='complete', result_file=str(final_path),
                           checkpoint_sha256=frozen['checkpoint_sha256'], best_epoch=frozen['best_epoch'],
                           results={key: result.get(key) for key in ('feature_payload_BPS', 'serialized_BPS',
                                    'optional_update_ratio', 'AMOTA', 'mAP', 'MOTA', 'IDS', 'FP', 'FN')})
        if manifest:
            summary.update(manifest_sha256=runtime.digest(manifest_path), settings=manifest['settings'])
        dimensions = {'no_explicit_delta': 1556, 'no_receiver_state': 787, 'no_communication_loss': 2324}[mode]
        table.append(metric_row(LABELS[mode], dimensions, result, summary['status']))
        runtime.write(directory / 'summary.json', summary)
        branches[mode] = summary
    overall_status = ('COMPLETE' if all(row['status'] == 'COMPLETE' for row in branches.values()) else
                      'FAILED' if any(row['status'] in ('FAILED', 'BLOCKED') for row in branches.values()) else 'RUNNING')
    execution_modes = {row['execution_mode'] for row in branches.values() if row['execution_mode']}
    execution = 'independent_branch_processes' if 'independent_branch_processes' in execution_modes else 'serial_B_A_C'
    manifest = {'status': overall_status, 'execution_mode': execution,
                'updated_utc': timestamp(), 'execution_order': list(runtime.MODES), 'branches': branches,
                'baseline': {'result': str(baseline_path), 'result_sha256': runtime.digest(baseline_path),
                             'best_epoch': reference['best_epoch'], 'checkpoint_sha256': reference['checkpoint_sha256']},
                'training_seed': 0, 'inference_seeds': [0],
                'selection': 'raw full-val AMOTA, strict improvement, first tie retained; every 3 epochs; patience 4 after epoch 12; max 36',
                'official_val_use': 'used for checkpoint selection and reporting; selection bias acknowledged',
                'statistical_scope': 'one training and inference seed; no cross-seed robustness or statistical significance claim'}
    runtime.write(reports / 'run_manifest.json', manifest)
    runtime.write(experiments / 'run_state.json', {
        'status': overall_status, 'phase': 'complete' if overall_status == 'COMPLETE' else 'branches',
        'utc': manifest['updated_utc'], 'execution_mode': execution,
        'branches': {mode: {key: branch.get(key) for key in
            ('status', 'phase', 'coordinator_pid', 'completed_epoch', 'best_epoch', 'last_state_utc')}
            for mode, branch in branches.items()},
        'completed_modes': [mode for mode, branch in branches.items() if branch['status'] == 'COMPLETE'],
        'table': str(reports / 'three_ablation_results.csv')})
    temporary = reports / 'three_ablation_results.csv.writing'
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(table)
    temporary.replace(reports / 'three_ablation_results.csv')
    text = '# Refresh2Track three ablations\n\n'
    text += 'Status: ' + manifest['status'] + '. Only completed full 21-scene / 3316-frame seed-zero evaluations appear as results. Pending branches have no final scores.\n\n'
    text += '| ' + ' | '.join(COLUMNS) + ' |\n| ' + ' | '.join('---' for _ in COLUMNS) + ' |\n'
    for row in table:
        cells = [str(row[column]) if row[column] is not None else 'PENDING' for column in COLUMNS]
        text += '| ' + ' | '.join(cells) + ' |\n'
    text += '\nFeature payload uses nominal 10 Hz × mean(52N + 780U); serialized BPS uses actual packet length. Optional UPDATE includes optional decisions only. MOTA/IDS/FP/FN use the inherited evaluator operating point.\n\n'
    text += 'Official validation participates in checkpoint selection, early stopping and reporting; these scores do not constitute an independent test. Only seed 0 is used, without claims about cross-seed robustness or statistical significance.\n\n'
    text += 'Explicit delta is computable from current and last received features: its ablation measures the value of an explicit representation, rather than new mathematical information. Receiver-state comparisons require joint assessment of communication and accuracy. Communication-loss outcomes, including extreme UPDATE/REUSE ratios, are recorded without threshold or regularization changes.\n\n'
    text += 'Frozen V4 reference checkpoint SHA256: `' + reference['checkpoint_sha256'] + '`. Source result: `' + str(baseline_path) + '`.\n\n'
    text += 'Evidence and branch paths are recorded in `run_manifest.json`; the source audit is `v4_preflight.md` and the input checks are `input_mask_tests.json`.\n'
    temporary = reports / 'three_ablation_summary.md.writing'
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(reports / 'three_ablation_summary.md')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps({'status': report(args.root)['status'], 'report': str(args.root / 'reports/ablations/three_ablation_results.csv')}))
