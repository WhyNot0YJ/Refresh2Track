"""Read-only recomputation of stored structure validation metrics and identities."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError((actual, expected))


def reaggregate(directory):
    result = read(directory / 'result.json')
    tracking = read(directory / 'track/metrics_summary.json')
    detection = read(directory / 'det/metrics_summary.json')
    track_details = read(directory / 'track/metrics_details.json')['car']
    det_details = read(directory / 'det/metrics_details.json')
    if set(tracking['label_metrics']['amota']) != {'car'}:
        raise AssertionError('Tracking classes differ from car-only protocol')
    if set(detection['label_aps']) != {'car'}:
        raise AssertionError('Detection classes differ from car-only protocol')
    if len(track_details['motar']) != 40 or tracking['cfg']['num_thresholds'] != 40:
        raise AssertionError('Tracking threshold count changed')
    values = [tracking['cfg']['metric_worst']['amota'] if math.isnan(value) else value
              for value in track_details['motar']]
    amota = sum(values) / len(values)
    close(amota, tracking['amota'])
    close(amota, result['AMOTA'])
    best_index = max((index for index, value in enumerate(track_details['mota'])
                      if not math.isnan(value)), key=lambda index: track_details['mota'][index])
    for name in ('mota', 'ids', 'fp', 'fn', 'gt'):
        close(track_details[name][best_index], tracking[name])
    close(tracking['ids'], result['IDS'])
    config = detection['cfg']
    aps = {}
    for distance in config['dist_ths']:
        precision = det_details['car:' + str(float(distance))]['precision']
        clipped = precision[round(100 * config['min_recall']) + 1:]
        ap = sum(max(value - config['min_precision'], 0.) for value in clipped)
        ap /= len(clipped) * (1. - config['min_precision'])
        close(ap, detection['label_aps']['car'][str(float(distance))])
        aps[str(distance)] = ap
    mean_ap = sum(aps.values()) / len(aps)
    close(mean_ap, detection['mean_ap'])
    close(mean_ap, result['mAP'])
    communication = read(directory / 'communication.json')
    frames = communication['frames']
    if len(frames) != result['sample_count'] or len(frames) != 3316:
        raise AssertionError('Incomplete validation')
    if len({row['cooperative_token'] for row in frames}) != 3316:
        raise AssertionError('Duplicate validation frames')
    for row in frames:
        if row['feature_payload_bytes'] != 52 * row['queries'] + 780 * row['updates']:
            raise AssertionError('Feature byte accounting differs')
        if row['updates'] != row['forced'] + row['optional_updates']:
            raise AssertionError('Update counts differ')
        if row['bytes'] != row['mandatory_bytes'] + 780 * row['updates']:
            raise AssertionError('Serialized byte accounting differs')
    close(sum(row['feature_payload_bytes'] for row in frames) / len(frames) * 10,
          result['feature_payload_BPS'])
    close(sum(row['bytes'] for row in frames) / len(frames) * 10, result['serialized_BPS'])
    inputs = read(directory / 'evaluator_inputs.json')
    input_hashes = {}
    for name in ('predictions', 'tracking', 'detection'):
        path = directory / Path(inputs[name]['path']).name
        actual = digest(path)
        if actual != inputs[name]['sha256']:
            raise AssertionError('Prediction input changed: ' + name)
        input_hashes[name] = actual
    return {'status': 'PASS', 'epoch': result['epoch'], 'seed': result['seed'],
            'AMOTA_recomputed': amota, 'mAP_recomputed': mean_ap,
            'AP_by_distance': aps, 'traditional_metrics_threshold_index': best_index,
            'tracking_GT_after_filtering_and_interpolation': tracking['gt'],
            'frames': len(frames), 'metric_and_byte_reaggregation': 'PASS',
            'input_sha256': input_hashes, 'checkpoint_sha256': result['checkpoint_sha256'],
            'tracking_config': tracking['cfg'], 'detection_config': config}, result


def audit(root):
    directories = {
        'V4_Full_epoch012_seed0': root / 'experiments/v4_budget_free/validation/final/V4_seed0',
        'C_GBA_GATE_epoch006_seed0': root / 'experiments/structure_v4_joint/C_GBA_GATE/validation/epoch006',
    }
    protected = [path for directory in directories.values()
                 for path in (directory / 'result.json', directory / 'communication.json',
                              directory / 'track/metrics_summary.json',
                              directory / 'det/metrics_summary.json')]
    before = {str(path.relative_to(root)): digest(path) for path in protected}
    runs, results = {}, {}
    for name, directory in directories.items():
        runs[name], results[name] = reaggregate(directory)
    old, current = results.values()
    pairing_differences = [key for key in set(old['pairing_identity']) | set(current['pairing_identity'])
                          if key != 'protocol' and old['pairing_identity'].get(key)
                          != current['pairing_identity'].get(key)]
    if pairing_differences:
        raise AssertionError(('Pairing identities differ', pairing_differences))
    for key in ('tracking_config', 'detection_config'):
        values = [row[key] for row in runs.values()]
        if values[0] != values[1]:
            raise AssertionError('Metric protocol changed: ' + key)
    checkpoint = root / 'experiments/structure_v4_joint/C_GBA_GATE/checkpoints/best.pth'
    if digest(checkpoint) != current['checkpoint_sha256']:
        raise AssertionError('Current C best checkpoint no longer represents epoch 6')
    if old['prediction_sha256'] == current['prediction_sha256']:
        raise AssertionError('New model prediction content equals historical predictions')
    after = {str(path.relative_to(root)): digest(path) for path in protected}
    if before != after:
        raise AssertionError('Protected evaluation files changed during audit')
    return {'status': 'PASS', 'checked_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'existing full-val prediction hashes, stored metric curves and byte ledger',
            'GPU_used': False, 'model_inference_rerun': False, 'matching_rerun': False,
            'runs': runs, 'pairing_differences_except_protocol': pairing_differences,
            'protected_files_unchanged': before, 'C_epoch006_retained_checkpoint': str(checkpoint),
            'limitations': ['Metric curves are reaggregated, not rematched from raw boxes.',
                            'No claim of independent test-set performance or isolated module contribution.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    evidence = audit(args.root.resolve())
    target = Path(__file__).with_name('metric_reaggregation.json')
    target.write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': evidence['status'], 'output': str(target),
                      'runs': {name: {key: row[key] for key in ('epoch', 'AMOTA_recomputed', 'mAP_recomputed', 'frames')}
                               for name, row in evidence['runs'].items()}}))
