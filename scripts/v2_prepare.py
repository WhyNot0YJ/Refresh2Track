"""Lock the full-train, fixed 12-epoch V2 formal protocol; preserve V1 evidence."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import main_runtime as runtime


SEED = 0
BUDGETS = [3474, 6382, 9290, 12198]
RATE_NORMALIZER = 12840.189523169913
P1_HISTORICAL_BYTES = 12161.645355850422
PROTOCOL = 'V2-FORMAL-20261007-FULL46-12E'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode('utf-8')).hexdigest()


def source_identity(root):
    files = ('scripts/v2_prepare.py', 'scripts/v2_runtime.py',
             'scripts/main_runtime.py', 'scripts/main_message.py')
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}


def lock_json(path, value):
    """Reject different protocol, data or source identities on resume."""
    if path.is_file():
        previous = runtime.read(path)
        if previous != value:
            raise ValueError('Existing formal V2 lock differs; preserved unchanged: ' + str(path))
        return previous
    runtime.write(path, value)
    return value


def prepare_train_manifest(root):
    infos = runtime.load_infos(root, 'coop', 'train')
    rows = runtime.native_manifest(root, 'train')
    infrastructure = {row['token']: row for row in rows}
    counts = Counter(str(info['scene_token']) for info in infos)
    official = runtime.read(root / 'code/CoopTrack/data/split_datas/cooperative-split-data-spd.json')['batch_split']
    if len(infos) != 7362 or len(counts) != 46 or set(counts) != set(official['train']):
        raise ValueError('Formal V2 requires all 7362 samples / 46 official train scenes')
    tokens = [str(info['token']) for info in infos]
    if len(set(tokens)) != len(tokens):
        raise ValueError('Duplicate official cooperative token')
    mapping = {}
    for info in infos:
        token, token_inf = str(info['token']), str(info['token_inf'])
        native = infrastructure[token_inf]
        if (native['scene'] != str(info['scene_token']) or
                native['timestamp_us'] != float(info['timestamp_inf'])):
            raise ValueError('Inherited cooperative/cache mapping differs: ' + token)
        cache_file = 'query_cache/train/' + token_inf + '.pkl'
        if not (root / cache_file).is_file():
            raise FileNotFoundError('Missing inherited query cache: ' + token_inf)
        mapping[token] = dict(native, token_inf=token_inf, cache_file=cache_file)
    scenes = sorted(counts)
    result = {'status': 'LOCKED', 'protocol_version': PROTOCOL,
              'prepare_source_identity': source_identity(root),
              'train_scenes': scenes, 'all_train_scenes': scenes,
              'train_samples': 7362, 'total_cooperative_samples': 7362,
              'train_tokens': tokens, 'scene_sample_counts': dict(sorted(counts.items())),
              'cooperative_to_infrastructure': mapping,
              'infrastructure_cache_frames': len(rows), 'native_manifest_identity': digest(rows),
              'official_split_identity': digest(official),
              'cache_manifest_inherited_identity': runtime.read(root / 'query_cache/train/manifest.json')['identity'],
              'training_scope': 'all official train scenes; no internal validation split'}
    result['manifest_identity'] = digest(result)
    return lock_json(root / 'configs/main_experiment_v2/train_manifest.json', result)


def training_config(root, manifest):
    np, torch = runtime.np, runtime.torch
    inherited = runtime.read(root / 'configs/main_experiment/budgets.json')
    if inherited['budgets_bytes'] != BUDGETS or inherited['full_int8_mean_bytes'] != RATE_NORMALIZER:
        raise ValueError('Inherited full-train budgets/rate normalizer conflict with formal V2')
    scenes = manifest['all_train_scenes']
    offsets = (np.arange(len(scenes)) % 4)[np.random.RandomState(SEED).permutation(len(scenes))]
    result = {
        'status': 'LOCKED_FORMAL_PROTOCOL', 'protocol_version': PROTOCOL,
        'train_manifest_identity': manifest['manifest_identity'], 'prepare_source_identity': source_identity(root),
        'train_scenes': 46, 'train_samples': 7362, 'epochs': 12, 'max_epochs': 12,
        'checkpoint_rule': 'final epoch 12 only; intermediate checkpoints are for crash resume',
        'backbone_commit': runtime.COMMIT,
        'checkpoints': {key: {'file': value[0], 'inherited_md5': value[1]}
                        for key, value in runtime.CHECKPOINTS.items() if key in ('inf', 'coop')},
        'budgets_bytes': BUDGETS, 'budget_names': ['B1', 'B2', 'B3', 'B4'],
        'full_int8_mean_bytes': RATE_NORMALIZER, 'budget_input': 'B / full_int8_mean_bytes',
        'scene_budget_offset_seed': SEED, 'initial_offset': dict(zip(scenes, map(int, offsets))),
        'budget_rotation': '(initial_offset[scene] + zero_based_epoch_index) % 4; fixed for whole scene',
        'budget_coverage': 'each scene receives each budget exactly three times in 12 epochs',
        'training_seed': SEED, 'primary_inference_seed': SEED, 'inference_seed_repeats': [],
        'native_batch_size': 4, 'requested_native_batch_size': 4, 'actual_native_batch_size': 4,
        'effective_optimizer_batch_size': 4, 'batch_measurement': None,
        'batch_registration': 'fixed protocol batch 4; measured correctness is recorded separately',
        'correctness_requirement': 'one real batch forward/loss/backward/optimizer step per variant; finite loss and nonzero finite scheduler gradient; no OOM',
        'batch_state': 'four independent scene streams; native batched forward; no gradient accumulation',
        'batch_loss_reduction': 'mean over actual valid streams, including tail batches',
        'optimizer': {'type': 'AdamW', 'peak_lr': 1e-4, 'weight_decay': 1e-4, 'grad_clip_norm': 5.0},
        'lr_schedule': {'warmup_epochs': 1, 'warmup_start_lr': 1e-5, 'min_lr': 1e-6,
                        'cosine_total_horizon_epochs': 12,
                        'formula': 'u<=1:1e-5+(1e-4-1e-5)*u; u>1:1e-6+0.5*(1e-4-1e-6)*(1+cos(pi*(u-1)/11))'},
        'precision': 'FP32', 'lambda_rate': 1.0, 'lambda_extra_association_preserve': 0.0,
        'task_loss': 'native return_loss=True detection/tracking/GT association with original weights',
        'rate_loss': '(relu(M+780*F+780*sum(optional_p)-max(B,M+780*F))/R)^2',
        'gate': {'threshold': 0.5, 'forced_update': 'no receiver cache only', 'staleness_cap': None,
                 'budget_top_k': False, 'ranking_noise': False, 'hard_forward': 'decoded.detach()+(soft-soft.detach())'},
        'cache': {'key': ['scene', 'native_reset_session', 'infrastructure_predicted_track_id'],
                  'ttl': None, 'last_received_update': 'actual successfully decoded UPDATE only',
                  'last_observed_update': 'every observed query including REUSE',
                  'reset': 'scene change; native timestamp reversal; dt>500000us; independent stream reset'},
        'serializer': {'name': 'main_message.pack / ReceiverCache', 'magic': 'APTU0001',
                       'version': 1, 'mandatory': 'current IDs,flags,ref_pts,pred_boxes,UTF8 frame metadata,CRC',
                       'latent_update_bytes': 780, 'actual_cost': 'len(packet)',
                       'quantization': 'INT8 per-query per-field symmetric absmax/127; FP32 scales; np.rint',
                       'network': 'reliable ordered no loss; excludes TCP/IP/radio/retransmission'},
        'features': {'dimension': 2324, 'latent_dimension': 768, 'state_dimension': 20,
                     'normalization': 'inherit existing feature builder; no staleness clipping'},
        'models': {'common': {'latent_projection': [768, 64], 'query_projection': [212, 64],
                              'query_activation': 'LayerNorm+ReLU', 'budget_embedding': [1, 64, 64],
                              'fresh_initialization': True},
                   'mlp': {'layers': [128, 64, 64, 1], 'activation': 'ReLU', 'cross_query_pooling': False,
                           'budget_injection': 'concat query token and budget embedding'},
                   'transformer': {'layers': 2, 'd_model': 64, 'heads': 4, 'dim_feedforward': 128,
                                   'dropout': 0, 'activation': 'ReLU', 'output_head': [64, 1],
                                   'budget_injection': 'budget token with same-scene query tokens',
                                   'query_position_encoding': False}},
        'periodic': {'points': ['P1', 'P2', 'P4', 'P6', 'P8', 'P10', 'P12', 'P16', 'P24', 'P32', 'INIT_ONLY'],
                     'new_official_val_points': ['P6', 'P8', 'P10', 'P12', 'P16', 'P24', 'P32', 'INIT_ONLY'],
                     'reuse_existing_val_periods': [1, 2, 4], 'conditional_extension': False,
                     'index': 'native infrastructure frame index; only registered period or no-cache initialization'},
        'primary_comparison': {'selection': 'actual measured bytes only; nearest periodic point and both adjacent periodic points for every V2 budget',
                               'metric_based_pair_selection': False},
        'evaluation': {'frequency_hz': 10, 'category': 'car-only', 'official_val_samples': 3316,
                       'official_val_scenes': 21, 'only_after_both_final_checkpoints': True,
                       'validation_during_training': False, 'early_stopping': False,
                       'independent_clean_state_per_run': True, 'native_batch_eval': 1,
                       'inference_seed_reset': 'after model/scheduler construction+load, before data flow'},
        'randomness': {'python_seed': SEED, 'numpy_seed': SEED, 'torch_cpu_seed': SEED, 'torch_cuda_seed': SEED,
                       'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
                       'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
                       'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
                       'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
                       'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
                       'guarantee': 'seeded RNG; inherited native kernels may remain nondeterministic'},
        'resume': {'boundary': 'complete epoch', 'required': ['scheduler', 'optimizer', 'LR progress',
                   'epoch', 'Python/NumPy/torch CPU/CUDA RNG', 'variant', 'train manifest identity',
                   'budget map', 'config identity', 'full V2 source identity'],
                   'old_protocol_checkpoint_accepted': False}}
    result['protocol_identity'] = digest(result)
    return lock_json(root / 'configs/main_experiment_v2/training_v2.json', result)


def reuse_val(root):
    identity = {'prepare_source_identity': source_identity(root), 'official_val_manifest_identity':
                digest(runtime.native_manifest(root, 'val')), 'backbone_commit': runtime.COMMIT,
                'cooperative_checkpoint_md5': runtime.CHECKPOINTS['coop'][1]}
    methods = {}
    for period in (2, 4):
        label = 'periodic' + str(period)
        metric_path = root / 'experiments/main/metrics' / (label + '.json')
        communication_path = root / 'experiments/main/communication' / (label + '.json')
        metrics, communication = runtime.read(metric_path), runtime.read(communication_path)
        prediction_path = root / 'experiments/main/predictions' / label / 'predictions.pkl'
        if (metrics['status'] != 'PASS' or metrics['samples'] != 3316 or metrics['scenes'] != 21 or
                not prediction_path.is_file() or len(communication['frames']) != 3316):
            raise ValueError('Incomplete inherited Periodic artifacts: ' + label)
        if metrics['identity']['cooperative_checkpoint_md5'] != runtime.CHECKPOINTS['coop'][1]:
            raise ValueError('Inherited Periodic checkpoint differs')
        methods['P' + str(period)] = {'status': 'REUSED_EXISTING_OFFICIAL_VAL', 'period_frames': period,
            'sample_count': 3316, 'scene_count': 21, 'metric_path': str(metric_path),
            'communication_path': str(communication_path), 'prediction_path': str(prediction_path),
            'metrics': metrics['metrics'], 'communication': metrics['communication'],
            'inherited_identity': metrics['identity'], 'new_model_forward_executed': False,
            'limitation': 'historical V1 RNG placement; forced_update_ratio uses the original V1 force rule'}
    legacy_path = root / 'artifacts/core/communication_baselines/FEAT_INT8_seed0_val/result.json'
    car_path = root / 'artifacts/core/r50_alignment/communication_car_only/FEAT_INT8_seed0_val/result.json'
    prediction_path = root / 'artifacts/core/communication_baselines/FEAT_INT8_seed0_val/evaluation/results_nusc.json'
    legacy, car = runtime.read(legacy_path), runtime.read(car_path)
    if (legacy['status'] != 'PASS' or car['status'] != 'PASS' or legacy['sample_count'] != 3316 or
            car['sample_count'] != 3316 or car['scene_count'] != 21 or not prediction_path.is_file() or
            legacy['summary']['mean_bytes_per_frame'] != P1_HISTORICAL_BYTES):
        raise ValueError('Incomplete or conflicting inherited FEAT_INT8/P1 reference')
    metrics = dict(car['metrics'])
    metrics['MOTA'] = runtime.read(car_path.parent / 'track/metrics_summary.json')['mota']
    methods['P1'] = {'status': 'REUSED_EXISTING_FEAT_INT8_OFFICIAL_VAL_REFERENCE',
        'period_frames': 1, 'sample_count': 3316, 'scene_count': 21, 'metric_path': str(car_path),
        'legacy_run_path': str(legacy_path), 'prediction_path': str(prediction_path), 'metrics': metrics,
        'communication': {'mean_bytes_per_frame': P1_HISTORICAL_BYTES,
            'p95_bytes_per_frame': legacy['summary']['p95_message_bytes'],
            'bytes_per_second': P1_HISTORICAL_BYTES * 10, 'update_ratio': 1.0},
        'historical_mean_bytes_per_frame': P1_HISTORICAL_BYTES, 'new_model_forward_executed': False,
        'serializer_difference': 'historical CTKMSG01 cost retained; new formal runs use APTU0001; no unified byte rebill',
        'limitation': 'P1-equivalent historical high-rate reference; original serializer and RNG placement are inherited'}
    result = {'status': 'REUSED_VAL_EVIDENCE_COMPLETE', 'protocol_version': PROTOCOL,
              'identity': identity, 'methods': methods, 'scope': 'existing official val references only',
              'user_instruction': 'do not rerun existing P1/P2/P4; retain historical P1 byte cost'}
    return lock_json(root / 'configs/main_experiment_v2/reused_val.json', result)


def prepare(root):
    """Call after main_runtime.initialize(root); create only formal full-train locks."""
    root = Path(root)
    if (root / 'configs/main_experiment_v2/dev_split.json').exists():
        raise ValueError('Old internal split remains; complete the authorized reset before preparing formal V2')
    manifest = prepare_train_manifest(root)
    config = training_config(root, manifest)
    reused = reuse_val(root)
    print('V2_FORMAL_PREPARE_PASS train_scenes=46 train_samples=7362 epochs=12 batch=4 source=' +
          digest(source_identity(root)), flush=True)
    return {'train_manifest': manifest, 'training': config, 'reused_val': reused}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    runtime.initialize(args.root)
    prepare(args.root)
