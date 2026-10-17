"""Compare isolated structure-training helpers using identical real seed0 batches."""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback

sys.dont_write_bytecode = True


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def summary(values):
    values = sorted(float(value) for value in values if value is not None)
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Nonfinite benchmark measurements')
    if not values:
        return dict(count=0, mean=None, median=None, p50=None, p95=None, minimum=None, maximum=None)

    def quantile(fraction):
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p50=quantile(.5), p95=quantile(.95), minimum=values[0], maximum=values[-1])


def gpu_row(line):
    fields = next(csv.reader([line], skipinitialspace=True))
    if len(fields) != 8:
        raise ValueError('Unexpected nvidia-smi field count')

    def number(value):
        return None if value.strip() in ('N/A', '[N/A]', '[Not Supported]') else float(value)

    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), nvidia_timestamp=fields[0],
                device_index=int(fields[1]), device_name=fields[2],
                gpu_utilization_percent=number(fields[3]), memory_utilization_percent=number(fields[4]),
                memory_used_MiB=number(fields[5]), memory_total_MiB=number(fields[6]),
                power_W=number(fields[7]))


class GPUSampler:
    def __init__(self, output, interval, device):
        self.output, self.interval, self.device = output, interval, device
        self.rows, self.errors = [], []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        command = ['nvidia-smi', '--id=' + str(self.device),
                   '--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw',
                   '--format=csv,noheader,nounits']
        with self.output.open('w', encoding='utf-8') as stream:
            while not self.stop.is_set():
                started = time.perf_counter()
                try:
                    result = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, universal_newlines=True, timeout=5)
                    row = gpu_row(result.stdout.strip())
                    self.rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    stream.flush()
                except Exception as error:
                    self.errors.append(str(error))
                self.stop.wait(max(0., self.interval - (time.perf_counter() - started)))

    def result(self):
        keys = ('gpu_utilization_percent', 'memory_utilization_percent', 'memory_used_MiB',
                'memory_total_MiB', 'power_W')
        return {'status': 'PASS' if self.rows else 'UNAVAILABLE', 'samples': len(self.rows),
                'interval_seconds': self.interval, 'device_index': self.device,
                'scope': 'whole-device nvidia-smi utilization during the timed window; includes other GPU users',
                'statistics': {key: summary(row[key] for row in self.rows) for key in keys},
                'errors': self.errors[:10], 'raw_samples': str(self.output)}


def sources(root, source_dir):
    paths = list(source_dir.glob('*.py')) + [Path(__file__).resolve()]
    paths += [root / 'scripts' / name for name in (
        'main_runtime.py', 'main_message.py', 'main_train.py', 'v2_runtime.py',
        'v2_train.py', 'v2_message.py', 'v2_scheduler.py', 'v4_budget_free/train.py',
        'v4_budget_free/scheduler.py', 'v4_budget_free/native_hooks.py')]
    return {str(path): digest(path) for path in sorted(set(paths))}


def load_helpers(root, source_dir, state_dir, normalization):
    if any(name in sys.modules for name in ('runtime', 'modules', 'receiver', 'train')):
        raise ValueError('Each source must be benchmarked in its own fresh Python process')
    sys.path[:0] = [str(source_dir), str(root / 'scripts')]

    def load(name):
        spec = importlib.util.spec_from_file_location(name, str(source_dir / (name + '.py')))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        sys.path[:0] = [str(source_dir), str(root / 'scripts')]
        return module

    runtime = load('runtime')
    runtime.experiment = lambda root_arg, model=None: state_dir
    runtime.write(state_dir / 'normalization.json', normalization)

    def config(root_arg, role, split='val', training=False, batch_size=1):
        tree = runtime.config_tree()
        replacements = 0
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant)
                    and node.value == 'experiments/structure_v4_joint/C_GBA_GATE/configs/generated'):
                node.value = str(state_dir / 'configs/generated')
                replacements += 1
        if replacements != 1:
            raise ValueError('Benchmark config output binding changed')
        namespace = dict(runtime.inherited.make_config.__globals__)
        exec(compile(ast.fix_missing_locations(tree), '<benchmark-output-config>', 'exec'), namespace)
        return namespace['make_config'](root_arg, role, split, training, batch_size)

    runtime.make_config = config
    load('modules')
    receiver = load('receiver')
    receiver.SCRIPTS = root / 'scripts'
    training = load('train')
    runtime.initialize(root)
    return runtime, training


def benchmark(args):
    root, source_dir, output = args.root.resolve(), args.source_dir.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('Benchmark output already exists: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    source_before = sources(root, source_dir)
    norm_path = root / 'experiments/structure_v4_joint/C_GBA_GATE/normalization.json'
    normalization_sha256 = digest(norm_path)
    normalization = json.loads(norm_path.read_text(encoding='utf-8-sig'))
    if normalization.get('status') != 'PASS':
        raise ValueError('The same passed train-only normalization is required for both sources')
    runtime, training = load_helpers(root, source_dir, output.with_suffix('.state'), normalization)
    base = training.canonical(root)
    torch, modules = runtime.native.torch, training.modules
    settings = runtime.load_settings(root)
    if settings['batch_size'] != 4 or settings['precision'] != 'FP32' or settings['training_seed'] != 0:
        raise ValueError('Benchmark requires the actual seed0 FP32 batch4 protocol')
    cfg, dataset, scenes, sender_rows = base.load_training(root)
    batches = training.entries_for(base, scenes, 1)
    if args.warmup + args.batches > len(batches):
        raise ValueError('Benchmark window exceeds the first real training epoch')
    model, scheduler, optimizer = training.build(root, base, cfg, dataset, settings, old_scheduler=False)
    initial = {'scheduler_sha256': modules.state_hash(scheduler.state_dict()),
               'native_trainable_sha256': modules.state_hash(modules.native_state(model))}
    frozen = modules.frozen_snapshot(model)
    context, streams = {}, {}
    parallel, remove, _ = training.attach(base, model, context)
    capture_supported = 'capture_evidence' in inspect.signature(training.run_batch).parameters
    extra = {'capture_evidence': False} if capture_supported else {}
    batch_path, log_path = output.with_suffix('.batches.jsonl'), output.with_suffix('.train.jsonl')
    sampler = GPUSampler(output.with_suffix('.gpu.jsonl'), args.sample_interval, args.device)
    rows, records, processed = [], [], []
    durations, frames, measured_order = [], 0, []
    measured_seconds = None

    def run_batch(step, entries):
        progress = (step - 1) / max(1, len(batches) - 1)
        learning_rate = base.training.learning_rate(progress, settings)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate * group['initial_lr'] / settings['optimizer']['peak_lr']
        value = training.run_batch(root, base, 1, step, entries, dataset, streams, sender_rows,
                                   model, scheduler, parallel, optimizer, settings, context,
                                   optimize=True, diagnose=step in (2, 10), **extra)
        modules.assert_frozen(model, frozen)
        processed.extend(entry['index'] for entry in entries)
        records.extend(value['records'])
        row = {key: item for key, item in value.items() if key not in ('snapshots', 'features', 'records')}
        rows.append(row)
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(epoch=1, **row), allow_nan=False) + '\n')
        return value

    try:
        model.reset_memory()
        for step, entries in enumerate(batches[:args.warmup], 1):
            value = run_batch(step, entries)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        sampler.thread.start()
        with batch_path.open('w', encoding='utf-8') as stream:
            torch.cuda.synchronize()
            started = time.perf_counter()
            for step in range(args.warmup + 1, args.warmup + args.batches + 1):
                entries = batches[step - 1]
                torch.cuda.synchronize()
                batch_started = time.perf_counter()
                value = run_batch(step, entries)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - batch_started
                durations.append(seconds)
                frames += len(entries)
                measured_order.extend(int(entry['index']) for entry in entries)
                row = {'step': step, 'frames': len(entries), 'seconds': seconds,
                       'diagnostic': step in (2, 10), 'task_loss': value['task_loss'],
                       'comm_loss': value['comm_loss'], 'total_loss': value['total_loss'],
                       'learning_rates': {group['name']: group['lr'] for group in optimizer.param_groups}}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
            torch.cuda.synchronize()
            measured_seconds = time.perf_counter() - started
        sampler.stop.set()
        sampler.thread.join(timeout=6)
        modules.assert_frozen(model, frozen, full=True)
        if (sources(root, source_dir) != source_before or digest(norm_path) != normalization_sha256
                or runtime.read(output.with_suffix('.state') / 'normalization.json') != normalization):
            raise ValueError('Benchmark source or normalization changed while running')
        result = {'status': 'PASS', 'label': args.label, 'source_dir': str(source_dir),
                  'source_hashes': source_before, 'initialization': initial,
                  'normalization_sha256': normalization_sha256, 'settings': settings,
                  'data_identity': runtime.read(root / 'experiments/structure_v4_joint/precheck.json')['splits']['train'],
                  'environment': {'python': sys.version, 'torch': torch.__version__,
                                  'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                                  'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                                  'tf32_cudnn': torch.backends.cudnn.allow_tf32},
                  'warmup_batches': args.warmup, 'measured_batches': args.batches,
                  'first_measured_step': args.warmup + 1, 'last_measured_step': args.warmup + args.batches,
                  'epoch_total_batches': len(batches), 'measured_frames': frames,
                  'seconds': measured_seconds, 'batches_per_second': args.batches / measured_seconds,
                  'frames_per_second': frames / measured_seconds, 'batch_seconds': summary(durations),
                  'per_batch_seconds_sum': sum(durations),
                  'sample_order_sha256': hashlib.sha256(json.dumps(measured_order).encode()).hexdigest(),
                  'sample_indices': measured_order, 'capture_evidence': False if capture_supported else 'legacy default',
                  'diagnostic_steps': [2, 10], 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'gpu': sampler.result(),
                  'timing_scope': 'real prepare, native forward, backward, optimizer, frozen checks and per-batch training log; CUDA synchronized; excludes model construction, warmup, final full frozen hash and checkpoint saving',
                  'output_files': {'batches': str(batch_path), 'training_log': str(log_path)},
                  'completed_utc': datetime.now(timezone.utc).isoformat()}
        runtime.write(output, result)
        return result
    finally:
        sampler.stop.set()
        if sampler.thread.is_alive():
            sampler.thread.join(timeout=6)
        remove()
        streams.clear()


def self_check():
    result = summary([4, 1, 3, 2])
    assert result['mean'] == 2.5 and result['median'] == 2.5
    assert math.isclose(result['p95'], 3.85)
    assert summary([None])['count'] == 0
    assert summary([2])['p95'] == 2
    row = gpu_row('2026/10/16 12:00:00.000, 0, NVIDIA GeForce RTX 4060 Ti, 84, 53, 8192, 16380, 120.5')
    assert row['gpu_utilization_percent'] == 84 and row['power_W'] == 120.5
    assert gpu_row('time, 0, gpu, 0, 0, 1, 2, [N/A]')['power_W'] is None
    return {'status': 'PASS', 'scope': 'CPU-only measurement summaries and nvidia-smi CSV parsing'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--label')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--warmup', type=int, default=16)
    parser.add_argument('--batches', type=int, default=64)
    parser.add_argument('--sample-interval', type=float, default=1.)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), sort_keys=True))
        return
    if any(value is None for value in (args.root, args.source_dir, args.label, args.output)):
        parser.error('--source-dir, --label, --root and --output are required')
    if args.warmup < 0 or args.batches < 1 or not .5 <= args.sample_interval <= 60:
        parser.error('Use warmup >= 0, batches >= 1 and sample interval between 0.5 and 60 seconds')
    try:
        result = benchmark(args)
        print(json.dumps({key: result[key] for key in (
            'status', 'label', 'seconds', 'batches_per_second', 'frames_per_second', 'batch_seconds', 'gpu')},
            allow_nan=False), flush=True)
    except Exception:
        failure = {'status': 'FAIL', 'label': args.label, 'traceback': traceback.format_exc()}
        if not args.output.exists():
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(failure, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
