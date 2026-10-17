"""Read historical evidence and independently reconstruct comparable field payloads."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import runtime


def historical_references(root):
    root = Path(root).resolve()
    rows = []

    def add(method, result_path, communication_path=None, mode='temporal', seed=0, note=None):
        path = root / result_path
        row = {'method': method, 'seed': seed, 'mAP': None, 'AMOTA': None,
               'feature_payload_BPS': None, 'serialized_message_BPS': None,
               'source_path': result_path, 'source_sha256': None,
               'comparability_note': note or 'Historical reference only; fresh V4/P16 results supply paired inference evidence.',
               'status': 'NOT_FOUND', 'query_count': None, 'update_count': None,
               'sample_count': None, 'scene_count': None, 'payload_reconstruction': 'UNAVAILABLE'}
        if not path.is_file():
            rows.append(row)
            return
        result = runtime.read(path)
        row.update(status=result.get('status'), source_sha256=runtime.digest(path),
                   sample_count=result.get('sample_count'), scene_count=result.get('scene_count'))
        metrics = result.get('metrics', result)
        row.update(mAP=metrics.get('mAP'), AMOTA=metrics.get('AMOTA'))
        communication = result.get('communication', result.get('summary', {}))
        frames = result.get('frames', [])
        if communication_path:
            cp = root / communication_path
            row.update(communication_source_path=communication_path,
                       communication_source_sha256=runtime.digest(cp) if cp.is_file() else None)
            if cp.is_file():
                evidence = runtime.read(cp)
                communication, frames = evidence.get('summary', {}), evidence.get('frames', [])
        if frames and len(frames) == row['sample_count'] and all('queries' in frame for frame in frames):
            count = len(frames)
            queries = sum(frame['queries'] for frame in frames)
            updates = queries if mode in ('fp32', 'int8') else sum(frame.get('updates', 0) for frame in frames)
            complete_updates = mode in ('fp32', 'int8') or all('updates' in frame for frame in frames)
            row.update(query_count=queries, update_count=updates if complete_updates else None,
                       geometry_BPS=52 * queries / count * 10,
                       serialized_message_BPS=sum(frame['bytes'] for frame in frames) / count * 10)
            if complete_updates:
                cost = 3072 if mode == 'fp32' else 780
                row.update(feature_payload_BPS=(52 * queries + cost * updates) / count * 10,
                           feature_payload_latent_BPS=cost * updates / count * 10,
                           payload_reconstruction='3124*N' if mode == 'fp32' else '832*N' if mode == 'int8' else '52*N+780*U')
                row['data_scope'] = 'all original cached sender query rows before vehicle spatial filtering'
        if row['serialized_message_BPS'] is None:
            mean = communication.get('mean_bytes_per_frame', communication.get('mean_bytes', result.get('mean_bytes')))
            if mean is not None:
                row['serialized_message_BPS'] = mean * 10
        row['direct_paired_comparison'] = False
        rows.append(row)

    base = 'experiments/v2/robustness/mlp_b1_vs_full_fp32_seed3'
    add('FULL_FP32', base + '/metrics/FULL_FP32_seed0/result.json',
        base + '/communication/FULL_FP32_seed0.json', mode='fp32',
        note='Complete car-only val; five FP32 fields, payload 3124*N. Original CTKFP321 envelope differs from APTU0001; inference protocol recorded but not a fresh V4 paired run.')
    add('FULL_INT8', 'artifacts/core/r50_alignment/communication_car_only/FEAT_INT8_seed0_val/result.json',
        'artifacts/core/communication_baselines/FEAT_INT8_seed0_val/result.json', mode='int8',
        note='Car-only reevaluation of unchanged historical predictions, paired communication counts from original all-query INT8 run. Payload 832*N; original native RNG/envelope differ, descriptive only.')
    for method in ('P16', 'P32', 'INIT_ONLY', 'mlp_B1'):
        add('V2_MLP_B1' if method == 'mlp_B1' else method,
            'experiments/v2/metrics/official_val/' + method + '_seed0/result.json',
            'experiments/v2/communication/official_val/' + method + '_seed0.json')
    if not runtime.retirement(root):
        add('V3_FIXED_4K_best15', 'experiments/v3_budget_specialization/fixed_4k/validation/epoch015/4000/result.json',
            'experiments/v3_budget_specialization/fixed_4k/validation/epoch015/4000/communication.json',
            note='Paused, incomplete V3 training; selected best epoch 15 from complete official val. Budget-conditioned model, soft nominal 4000B constraint; historical seed0 reference only.')
    runtime.write(runtime.experiment(root) / 'metrics/reference_results.json', rows)
    return rows
