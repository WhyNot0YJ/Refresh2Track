"""Write the final table and curves only from completed, measured full-data results."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline_rows(root):
    final = read(root / 'artifacts/core/r50_alignment/final.json')
    rows = []
    for case in final['communication_evaluations']:
        source = read(root / 'artifacts/core/communication_baselines' / case['case'] / 'result.json')
        summary = source['summary']
        track = source['metrics']['track']['label_metrics']
        rows.append(dict(method=case['condition'], budget='seed%d' % case['seed'], reused=True,
                         metrics=dict(case['metrics'], MOTA=track['mota']['car']),
                         communication={'mean_bytes_per_frame': summary['mean_bytes_per_frame'],
                                        'p95_bytes_per_frame': summary['p95_message_bytes'],
                                        'bytes_per_second': 10 * summary['mean_bytes_per_frame'],
                                        'update_ratio': None},
                         latency={'scheduler_ms': None, 'pack_ms': summary['mean_pack_ms'],
                                  'unpack_ms': summary['mean_unpack_ms'],
                                  'cooperative_ms': summary['mean_inference_ms'],
                                  'pipeline_ms': summary['mean_pipeline_total_ms']},
                         peak_allocated_bytes=max(frame.get('peak_allocated_bytes', 0)
                                                  for frame in source['frames'])))
    return rows


def nondominated(row, candidates):
    values = (row['communication']['mean_bytes_per_frame'], -row['metrics']['mAP'], -row['metrics']['AMOTA'])
    for candidate in candidates:
        other = (candidate['communication']['mean_bytes_per_frame'], -candidate['metrics']['mAP'], -candidate['metrics']['AMOTA'])
        if all(a <= b for a, b in zip(other, values)) and any(a < b for a, b in zip(other, values)):
            return False
    return True


def number(value, digits=6):
    return '—' if value is None else ('%.*f' % (digits, value))


def best_main_result(rows):
    selected = max((row for row in rows if row['method'] == 'aptu'),
                   key=lambda row: (row['metrics']['AMOTA'], row['metrics']['mAP'],
                                    -row['communication']['mean_bytes_per_frame']))
    return dict(budget=selected['budget'],
                bytes_per_frame=selected['communication']['mean_bytes_per_frame'],
                bytes_per_second=selected['communication']['bytes_per_second'],
                **{key: selected['metrics'][key] for key in ('mAP', 'AMOTA', 'IDS')})


def self_check():
    rows = [dict(method='aptu', budget=budget, metrics={'AMOTA': amota, 'mAP': map_value, 'IDS': ids},
                 communication={'mean_bytes_per_frame': payload, 'bytes_per_second': payload * 10})
            for budget, amota, map_value, payload, ids in
            [('B1', .4, .3, 20, 1), ('B2', .5, .1, 50, 2),
             ('B3', .5, .2, 80, 3), ('B4', .5, .2, 70, 4)]]
    rows.append(dict(method='FEAT_INT8', metrics={'AMOTA': .9, 'mAP': .9}))
    assert best_main_result(rows) == dict(budget='B4', bytes_per_frame=70, bytes_per_second=700,
                                          mAP=.2, AMOTA=.5, IDS=4)
    print('APTU result summary ordering PASS')


def write_blocked(root, state):
    phase = state.get('phase', 'unknown')
    log = root / 'logs/main' / ('eval.log' if phase.startswith('eval_') or phase == 'final_report' else 'train.log')
    tail = '\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-30:]) if log.is_file() else ''
    action = ('Use the requested RTX 5090 32GB and resume the same full-data experiment.'
              if 'out of memory' in tail.lower() else 'Resolve the recorded error and resume the same full-data experiment.')
    lines = ['# 主实验执行状态', '', '**MAIN_EXPERIMENT_BLOCKED**', '',
             'Phase: `' + phase + '`', '', 'Exact blocker: ' + state['error'], '',
             'Minimal next action: ' + action, '',
             '未完成的缓存、训练或评测不作为正式主实验结果。已有 car-only INT8 基线仍为 Bytes/frame=12161.645355850422，mAP=0.36110747718997416，AMOTA=0.35334327871736654。']
    if tail:
        lines += ['', '```text', tail, '```']
    (root / 'reports').mkdir(parents=True, exist_ok=True)
    (root / 'reports/MAIN_EXPERIMENT_FINAL.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def finalize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = baseline_rows(root)
    cache = read(root / 'experiments/main/metrics/train_cache.json')
    if cache['status'] != 'FULL_TRAIN_CACHE_PASS' or cache['frames'] != 7445 or cache['scenes'] != 46:
        raise ValueError('Final report requires the complete official infrastructure train cache')
    budget_hash = sha(root / 'configs/main_experiment/budgets.json')
    checkpoint_hashes = {variant: sha(root / 'experiments/main/checkpoints' / variant / 'latest.pth')
                         for variant in ('aptu', 'mlp', 'no_assoc')}
    training = {}
    for variant in ('aptu', 'mlp', 'no_assoc'):
        result = read(root / 'experiments/main/metrics' / (variant + '_train.json'))
        if result['status'] != 'PASS' or result['last_stable_epoch'] != 12 or len(result['epochs']) != 12:
            raise ValueError('Final report requires complete stable 12-epoch training: ' + variant)
        if any(epoch['frames'] != 7362 or epoch['scenes'] != 46 for epoch in result['epochs']):
            raise ValueError('Training did not cover the complete official split')
        training[variant] = result
    cases = [(method, None) for method in ('periodic2', 'periodic4')]
    cases += [(method, budget) for method in ('delta', 'mlp', 'no_assoc', 'aptu')
              for budget in ('B1', 'B2', 'B3', 'B4')]
    for method, budget in cases:
        name = method + ('_' + budget if budget else '')
        result = read(root / 'experiments/main/metrics' / (name + '.json'))
        if result['status'] != 'PASS' or result['samples'] != 3316 or result['scenes'] != 21:
            raise ValueError('Final report requires complete official val: ' + name)
        if result['identity']['budgets_sha256'] != budget_hash:
            raise ValueError('Evaluation uses different locked budgets: ' + name)
        if method in checkpoint_hashes and result['identity']['scheduler_checkpoint_sha256'] != checkpoint_hashes[method]:
            raise ValueError('Evaluation uses a different final scheduler checkpoint: ' + name)
        rows.append(result)
    comparisons = [row for row in rows if row['method'] in ('FEAT_INT8', 'periodic2', 'periodic4', 'delta', 'mlp', 'aptu')]
    frontier = [row['budget'] for row in comparisons if row['method'] == 'aptu' and nondominated(row, comparisons)]
    best = best_main_result(rows)
    conclusion = 'APTU ' + ('达到' if len(frontier) >= 2 else '未达到') + '既定的“至少两个固定预算点位于联合 Pareto 前沿”标准。'
    artifact = root / 'experiments/main/metrics'
    columns = ['method', 'budget', 'bytes_per_frame', 'p95_bytes_per_frame', 'bytes_per_second',
               'mAP', 'AMOTA', 'AMOTP', 'IDS', 'MOTA', 'update_ratio', 'scheduler_ms',
               'pack_ms', 'unpack_ms', 'cooperative_ms', 'pipeline_ms', 'peak_allocated_bytes']
    with (artifact / 'main_results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            values = dict(method=row['method'], budget=row.get('budget'),
                          bytes_per_frame=row['communication']['mean_bytes_per_frame'],
                          peak_allocated_bytes=row.get('peak_allocated_bytes'))
            values.update({key: row['metrics'][key] for key in ('mAP', 'AMOTA', 'AMOTP', 'IDS', 'MOTA')})
            values.update({key: row['communication'].get(key) for key in ('p95_bytes_per_frame', 'bytes_per_second', 'update_ratio')})
            values.update({key: row['latency'].get(key) for key in ('scheduler_ms', 'pack_ms', 'unpack_ms', 'cooperative_ms', 'pipeline_ms')})
            writer.writerow(values)
    for metric in ('mAP', 'AMOTA'):
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for method in ('FULL_FP32', 'FEAT_INT8', 'periodic2', 'periodic4', 'delta', 'mlp', 'no_assoc', 'aptu'):
            selected = sorted((row for row in rows if row['method'] == method),
                              key=lambda row: row['communication']['mean_bytes_per_frame'])
            axis.plot([row['communication']['mean_bytes_per_frame'] for row in selected],
                      [row['metrics'][metric] for row in selected], 'o-', label=method,
                      linewidth=2 if method == 'aptu' else 1)
        axis.set(xlabel='Serialized bytes / frame', ylabel=metric)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        for extension in ('png', 'pdf'):
            figure.savefig(artifact / ('bytes_vs_' + metric.lower() + '.' + extension), dpi=300)
        plt.close(figure)
    final = {'status': 'MAIN_EXPERIMENT_COMPLETE', 'full_train_cache': cache,
             'training': training, 'rows': rows, 'aptu_joint_pareto_budgets': frontier,
             'method_success_minimum_met': len(frontier) >= 2,
             'BEST_MAIN_RESULT': best,
             'BEST_MAIN_RESULT_selection': 'Among the four fixed APTU budgets: highest AMOTA, then highest mAP, then lowest serialized bytes/frame; summary only, not used for model, budget or threshold selection.',
             'conclusion': conclusion,
             'completed_local': datetime.now(timezone(timedelta(hours=8))).isoformat()}
    (artifact / 'final.json').write_text(json.dumps(final, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    lines = ['# 主实验最终结果', '', '**MAIN_EXPERIMENT_COMPLETE**；' + final['completed_local'], '',
             '官方 train：46 scenes / 7362 cooperative samples；路侧缓存7445帧。官方 val：21 scenes / 3316 samples；10Hz、R50、car-only。', '',
             '既有 FP32/FP16/INT8/Top-K/Random 基线结果直接复用；Periodic、Delta和MLP为新增完整实验。新调度器以完整 train 锁定的四个预算训练一个模型，每个训练方法用最后一个稳定的第12个epoch评测；未使用val选预算、阈值或epoch。', '',
             '**BEST_MAIN_RESULT**', '',
             'budget=%s；Bytes/frame=%s；Bytes/s=%s；mAP=%s；AMOTA=%s；IDS=%s。' %
             (best['budget'], number(best['bytes_per_frame'], 3), number(best['bytes_per_second'], 3),
              number(best['mAP']), number(best['AMOTA']), number(best['IDS'], 0)), '',
             '此行在四个固定APTU预算结果中按AMOTA最高、mAP最高、Bytes/frame最低的顺序选取，仅用于结果摘要，不用于模型、预算或阈值选择。', '',
             '| Method | Budget | Bytes/frame | Bytes/s | mAP | AMOTA | IDS | Scheduler ms |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' %
                     (row['method'], row.get('budget', '—'), number(row['communication']['mean_bytes_per_frame'], 3),
                      number(row['communication']['bytes_per_second'], 3), number(row['metrics']['mAP']),
                      number(row['metrics']['AMOTA']), number(row['metrics']['IDS'], 0),
                      number(row['latency']['scheduler_ms'], 3)))
    lines += ['', 'APTU 三目标 Pareto 前沿预算（真实通信量越小、mAP和AMOTA越大）：' + ', '.join(frontier) + '。',
              '“至少两个固定预算点”条件：' + ('PASS' if len(frontier) >= 2 else 'FAIL') + '。' + conclusion + '结果不用于重新调整val上的阈值或模型。', '',
              '完整 AMOTP、MOTA、P95字节数、update ratio、各阶段延迟与GPU内存：[main_results.csv](../experiments/main/metrics/main_results.csv)。', '',
              '![Bytes/frame vs mAP](../experiments/main/metrics/bytes_vs_map.png)', '',
              '![Bytes/frame vs AMOTA](../experiments/main/metrics/bytes_vs_amota.png)', '',
              '消息字节数为实际序列化长度。新方法包含ID、UPDATE标记、场景/reset身份、几何和INT8scale；继承INT8基线保留原消息协议及原测量值。',
              '强制更新的新轨迹或陈旧轨迹可能使单帧超出预算，逐帧实际payload与overflow记录保留在communication及训练结果中。',
              '端到端延迟按缓存回放管线测量，完整路侧缓存生成开销单独记录；没有将其当作双端在线总延迟。']
    (root / 'reports/MAIN_EXPERIMENT_FINAL.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    finalize(parser.parse_args().root.resolve())
