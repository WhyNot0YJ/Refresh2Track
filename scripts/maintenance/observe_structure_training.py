"""Observe training logs and whole-device GPU activity without loading a model."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from benchmark_structure_performance import GPUSampler


def parse_rows(text):
    rows = {}
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (isinstance(row, dict) and 'phase' not in row
                and all(key in row for key in ('epoch', 'step', 'gradient_norms'))):
            rows[(int(row['epoch']), int(row['step']))] = row
    return rows


def read_rows(path):
    return parse_rows(path.read_text(encoding='utf-8', errors='replace')) if path.exists() else {}


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def observe(args):
    root, output = args.root.resolve(), args.output.resolve()
    log = root / 'experiments/structure_v4_joint/C_GBA_GATE/logs/train.log'
    reference_path = root / 'experiments/structure_v4_joint/performance_20261016/new1.train.jsonl'
    if output in (log, reference_path):
        raise ValueError('Output must be separate from the read-only training evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.perf_counter() + 900
    before = read_rows(log)
    while len(before) < args.minimum_steps:
        if time.perf_counter() >= deadline:
            raise TimeoutError('Training did not reach the minimum logged step count within 15 minutes')
        time.sleep(1.)
        before = read_rows(log)
    first = max(before)
    sampler = GPUSampler(output.with_suffix('.gpu.jsonl'), 1., 0)
    progress = []
    sampler.thread.start()
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    after = before
    try:
        while time.perf_counter() - started < args.seconds:
            time.sleep(min(1., max(0., args.seconds - (time.perf_counter() - started))))
            after = read_rows(log)
            latest = max(after) if after else None
            progress.append({'elapsed_seconds': time.perf_counter() - started,
                             'epoch_step': list(latest) if latest else None})
        elapsed = time.perf_counter() - started
        ended_utc = datetime.now(timezone.utc).isoformat()
    finally:
        sampler.stop.set()
        sampler.thread.join(timeout=6)
    last = max(after) if after else None
    completed = len(after.keys() - before.keys())
    reference = read_rows(reference_path)
    common = sorted(after.keys() & reference.keys())
    differences = [{'epoch': key[0], 'step': key[1],
                    'fields': sorted(name for name in after[key].keys() | reference[key].keys()
                                     if after[key].get(name) != reference[key].get(name))}
                   for key in common if after[key] != reference[key]]
    health = {'progress_observed': completed > 0, 'all_logged_scalars_finite': finite(list(after.values())),
              'log_not_replaced': before.keys() <= after.keys()}
    result = {'status': 'PASS' if all(health.values()) and common and not differences else 'FAIL',
              'started_utc': started_utc, 'ended_utc': ended_utc, 'seconds': elapsed,
              'start_epoch_step': list(first), 'end_epoch_step': list(last) if last else None,
              'step_difference': last[1] - first[1] if last and first[0] == last[0] else None,
              'completed_batches': completed, 'batches_per_second': completed / elapsed,
              'seconds_per_batch': elapsed / completed if completed else None,
              'health': health, 'gpu': sampler.result(), 'progress': progress,
              'scalar_reference_comparison': {'status': 'FAIL' if differences else 'PASS' if common else 'PENDING',
                  'reference': str(reference_path), 'common_steps': len(common),
                  'differences': len(differences), 'first_differences': differences[:20],
                  'comparison': 'complete scalar log row exact equality; no excluded keys or tolerance'},
              'scope': 'passive wall-time observation of completed logged batches; includes real training and logging, does not load CUDA or a model; GPU samples are whole-device nvidia-smi readings',
              'training_log': str(log)}
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--minimum-steps', type=int, default=12)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or args.minimum_steps < 1 or args.output.suffix != '.json':
        parser.error('Use seconds > 0, minimum-steps >= 1 and a .json output')
    result = observe(args)
    print(json.dumps({key: result[key] for key in (
        'status', 'seconds', 'completed_batches', 'batches_per_second', 'health', 'scalar_reference_comparison')},
        allow_nan=False), flush=True)
