"""Execute canonical V4 training/evaluation with one explicit ablation adapter."""
import argparse
from functools import partial
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime


def bind_modules(root, mode):
    runtime.bind(root, mode)
    import input_modes
    base = HERE.parent / 'v4_budget_free'
    train = runtime.load_module('refresh2track_canonical_train', base / 'train.py')
    evaluate = runtime.load_module('refresh2track_canonical_evaluate', base / 'evaluate.py')
    train.Scheduler = evaluate.Scheduler = partial(input_modes.Scheduler, mode=mode)
    return train, evaluate, input_modes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE.parents[1])
    parser.add_argument('--mode', choices=runtime.MODES, required=True)
    parser.add_argument('--action', choices=('audit', 'cpu', 'correctness', 'train', 'evaluate'), required=True)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--method', default='V4', choices=('V4',))
    parser.add_argument('--seed', type=int, choices=(0,), default=0)
    parser.add_argument('--selection', action='store_true')
    args = parser.parse_args()
    train, evaluate, masks = bind_modules(args.root, args.mode)
    try:
        if args.action == 'audit':
            from audit import audit
            result = audit(args.root)
        elif args.action == 'cpu':
            result = masks.self_check(args.root)
            from storage import self_check
            result['storage_checks'] = self_check()
            runtime.write(Path(args.root) / 'reports/ablations/input_mask_tests.json', result)
        elif args.action == 'correctness':
            result = train.correctness_check(args.root, args.mode)
            result['peak_allocated_bytes'] = max(result.get('peak_allocated_bytes', 0),
                                                 runtime.native.torch.cuda.max_memory_allocated())
            runtime.write(runtime.experiment(args.root) / 'metrics/correctness.json', result)
        elif args.action == 'train':
            result = train.train_epoch(args.root, args.mode, args.epoch)
            settings = runtime.load_settings(args.root)
            scalar_log = dict(result, lambda_comm=settings['lambda_comm'], comm_loss_raw=result['comm_loss'],
                              hard_optional_update_count=result['optional_updates'],
                              optional_query_count=result['optional_queries'], forced_update_count=result['forced_init'],
                              reuse_count=result['queries'] - result['optional_updates'] - result['forced_init'],
                              samples_per_second=result['samples'] / result['seconds'],
                              checkpoint_sha256=runtime.digest(runtime.experiment(args.root) / 'checkpoints/latest.pth'))
            runtime.write(runtime.experiment(args.root) / 'metrics' / ('epoch%03d_summary.json' % args.epoch), scalar_log)
            print(json.dumps({'phase': 'epoch_scalars', **scalar_log}), flush=True)
        else:
            from storage import restore, retain_archives
            directory = runtime.experiment(args.root) / 'validation'
            directory = directory / ('epoch%03d' % args.epoch) if args.selection else directory / 'final/V4_seed0'
            existing = directory / 'result.json'
            if existing.exists():
                restore(runtime.read(existing), directory)
            result = evaluate.evaluate(args.root, 'V4', args.checkpoint, args.epoch, 0, args.selection)
            reference = runtime.read(runtime.reference(args.root) / 'final_validation.json')
            baseline = next(row for row in reference['rows'] if row['method'] == 'V4' and row['seed'] == 0)
            for key in ('native_checkpoint_sha256', 'cache_sha256', 'cache_file_count', 'val_order_sha256',
                        'native_source_sha256', 'dataset_source_sha256', 'fields', 'filter_stage',
                        'serializer', 'payload', 'frequency_hz', 'rng_reset'):
                if result['pairing_identity'][key] != baseline['pairing_identity'][key]:
                    raise ValueError('Formal evaluation differs from V4 baseline: ' + key)
            if result['sample_count'] != 3316 or result['scene_count'] != 21:
                raise ValueError('Formal evaluation coverage is incomplete')
            freed = retain_archives(result, directory)
            runtime.write(directory / 'storage.json', {'status': 'PASS', 'raw_duplicates_removed_bytes': freed,
                                                       'retained': 'SHA256-verified lossless gzip evaluator inputs'})
        print(json.dumps({'status': 'PASS', 'mode': args.mode, 'action': args.action}), flush=True)
    except Exception:
        torch = getattr(runtime.native, 'torch', None)
        runtime.write(runtime.experiment(args.root) / 'audit/worker_failure.json', {
            'status': 'FAILED', 'action': args.action, 'mode': args.mode,
            'traceback': traceback.format_exc(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch and torch.cuda.is_available() else None})
        raise


if __name__ == '__main__':
    main()
