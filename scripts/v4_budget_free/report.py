"""Report only observed V4 evidence; pending runs retain empty result cells."""
import argparse
import json
from pathlib import Path
import statistics
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import runtime


def optional_read(path, default=None):
    return runtime.read(path) if path.is_file() else default


def cell(value):
    if value is None:
        return 'PENDING / 未发现'
    if isinstance(value, float):
        return '%.6f' % value
    return str(value)


def report(root):
    root = Path(root).resolve()
    directory = runtime.experiment(root)
    runtime.load_settings(root)
    state = optional_read(directory / 'run_state.json', {'status': 'NOT_STARTED'})
    preflight = optional_read(directory / 'audit/preflight.json', {})
    correctness = optional_read(directory / 'metrics/correctness.json', {})
    cpu = optional_read(directory / 'audit/cpu_preflight.json', {})
    selection = optional_read(directory / 'metrics/best_selection.json', {})
    history = optional_read(directory / 'metrics/train_history.json', [])
    archive = root / 'archives/v3_fixed_budget_20261012'
    migration = runtime.retirement(root)
    archive_manifest = ({'status': migration['status'], 'file_count': 0,
                         'protected_file_count': len(migration['protected_shared_files'])}
                        if migration else runtime.read(archive / 'manifest.json'))
    from references import historical_references
    references = historical_references(root)
    if isinstance(references, dict):
        references = references.get('rows', references.get('results', []))
    runtime.write(directory / 'metrics/reference_results.json', references)
    observed, pairs = [], []
    for seed in (0, 2025, 2026):
        group = {}
        for method in ('V4', 'P16'):
            path = directory / 'validation/final' / ('%s_seed%d' % (method, seed)) / 'result.json'
            result = optional_read(path)
            if result and result.get('status') == 'PASS':
                group[method] = result
                observed.append(result)
        if len(group) == 2:
            left, right = group['V4'], group['P16']
            matched = bool(left.get('pairing')) and left.get('pairing') == right.get('pairing')
            pairs.append({'seed': seed, 'pairing_verified': matched,
                          'AMOTA_difference': left['AMOTA'] - right['AMOTA'],
                          'mAP_difference': left['mAP'] - right['mAP'],
                          'feature_payload_BPS_difference': left['feature_payload_BPS'] - right['feature_payload_BPS'],
                          'feature_payload_cost_ratio': left['feature_payload_BPS'] / right['feature_payload_BPS']})
    paired_summary = None
    if len(pairs) == 3 and all(row['pairing_verified'] for row in pairs):
        paired_summary = {name: {'mean': statistics.mean(row[name] for row in pairs),
                                'sample_standard_deviation': statistics.stdev(row[name] for row in pairs)}
                          for name in ('AMOTA_difference', 'mAP_difference', 'feature_payload_BPS_difference')}
    # Reporting convention fixed before training; no inference interval is tuned.
    nearby_cost = paired_summary is not None and all(abs(row['feature_payload_cost_ratio'] - 1) <= .1 for row in pairs)
    positive_each_seed = paired_summary is not None and all(row['AMOTA_difference'] > 0 for row in pairs)
    v2_reference = next((row for row in references if row.get('method') == 'V2_MLP_B1'), {})
    if paired_summary is None:
        advantage = 'PENDING: all three verified inference pairs are required.'
        paper_choice = 'Retain verified V2 MLP as the supported main result while V4 training/evaluation is incomplete.'
    else:
        advantage = ('All three inference seeds have a positive AMOTA difference at nearby observed cost.' if nearby_cost and positive_each_seed else
                     'No demonstrated positive AMOTA difference for every seed at nearby cost; inspect the actual paired differences.')
        if not nearby_cost:
            advantage += ' These runs have unequal cost (outside the preregistered descriptive ±10% payload convention).'
        v4_mean_amota = statistics.mean(row['AMOTA'] for row in observed if row['method'] == 'V4')
        if nearby_cost and positive_each_seed and v2_reference.get('AMOTA') is not None and v4_mean_amota >= v2_reference['AMOTA']:
            paper_choice = 'V4 is supported as a main-method candidate by this completed official-val comparison. Retain V2 as a historical reference; its seed0 row is not a fresh paired V4/V2 experiment.'
        else:
            paper_choice = 'Retain verified V2 MLP as the supported main result; this candidate has not established the requested nearby-cost advantage and a stronger case for replacing V2. Preserve V4 results as observed evidence.'
    snapshot = {'status': state.get('status'), 'experiment_id': runtime.MODEL,
                'preflight': preflight.get('status'), 'training_epochs_completed': len(history),
                'best_epoch': selection.get('best_epoch'), 'best_selection': selection,
                'final_results': observed, 'paired_differences': pairs,
                'paired_three_seed_summary': paired_summary, 'historical_references': references,
                'updated_utc': datetime.now(timezone.utc).isoformat()}
    runtime.write(directory / 'metrics/report_summary.json', snapshot)
    destination = root / 'reports/v4_budget_free'
    destination.mkdir(parents=True, exist_ok=True)
    method = '''# V4 budget-free method and audit

Experiment: `%s`; configuration: `configs/v4_budget_free/common.json`.

V3 archive: `archives/v3_fixed_budget_20261012`; status **%s**. It contains %d copied files and protects %d originals. Every copied source-before/archive/source-after SHA matched; protected originals retained SHA, size and mtime. Runtime evidence found no live V3 writer or scheduled writer. Fixed-4K is PARTIAL: completed epoch21, interrupted22, best epoch15 AMOTA0.36890589976964766. Fixed-8K, Fixed-12K and Shared have correctness checks only, no trained checkpoints. The original best checkpoint's embedded older selection metadata is retained and documented in the archive. V3 originals remain unchanged and are retired as the method-development mainline.

The actual upstream Git checkout is `code/CoopTrack`, pinned at `29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399`. The workspace root has no Git repository. Dirty upstream binary diffs and actual shared source are in the archive; V4 source/config hashes, checkpoint identities, train/val cache manifests and official data-info hashes are locked by `experiments/v4_budget_free/manifest.json`. No new model, dataset or dependency was installed.

Actual V3 scheduler state was already N×2324: current, last-received and delta, each768, plus20 state values. The budget arrived separately through a 1→64→64 embedding; it was not an additional raw state column. V4 removes that embedding and its concatenation, changing the first relation layer128→64 to64→64. Shared latent projection768→64, state projection212→64 with LayerNorm/ReLU, relation64→64→64 and output64→1 remain. Trainable parameters fall from79745 to71361. `Scheduler.forward(tokens, valid_mask=None)` accepts no budget. There is no budget scalar/token/FiLM, Top-K filling, stochastic gate, TTL forcing, ID/row input or future input.

For optional queries O_t with real receiver history, p_i=sigmoid(z_i); hard UPDATE is p_i≥0.5. Missing receiver history forces UPDATE and never contributes to optional regularization. The original straight-through helper returns the exact decoded packet in forward and supplies the equivalent hard-gate surrogate gradient through detached current-minus-last values. The receiver consumes only decoded `query_feats`, `query_embeds`, `cache_motion_feats`, `ref_pts`, `pred_boxes`. Sender mirror commits actual receiver-decoded values after successful unpack. Caches use predicted IDs internally for identity, reset at native scene/session boundaries and stay isolated by batch lane. Predicted IDs do not enter learned state numerically.

L_comm,t = (780/R) × Σ_{i∈O_t} p_i.

L_total = L_native_task + λ_comm × mean_real_frames(L_comm,t).

λ_comm=0.05 is the sole preregistered engineering candidate. R=12840.189523169913B/frame comes from `configs/main_experiment/budgets.json`, LOCKED_FROM_FULL_TRAIN, val_used=false:7445 infrastructure frames,7362 paired cooperative samples,46 scenes. Its source cache manifest SHA is fb5b85106cf245a8c9c1e7d22ab51131cc01c782651da757f2453e1bb577e1ad. R includes old full-INT8 serialization overhead and is only a fixed scale; it is never a budget ceiling or neural-network input. No official-val adjustment of R or λ occurs.

Both losses use the actual active frame lanes and their mean, excluding padding. No second division by optional-query count occurs. The inherited native loss definitions and per-loss normalizers remain; the previously verified outer reducer averages real streams. Instrumentation captures per-frame native task loss and asserts that its mean reproduces the native parsed batch loss. The V3 batch reducer already had this frame-mean convention, so no batch-scaling repair was needed. CPU evidence records old/new single-frame values and gradients, including repeated-frame batch invariance. Example with forced=[true,false,false], p=[.2,.3,.4], mandatory1000, old budget1000: V3 excess penalty0.0018081831479037057, derivative0.005166237565439158 per optional p; V4 comm0.042522736834588924, derivative0.0607467669065556; λ-weighted derivative0.00303733834532778. Forced derivatives are zero.

Communication keeps soft estimates and hard counts separate. Estimated optional bytes=780Σp_optional; actual optional bytes=780×hard optional UPDATE count. Each frame logs their signed gap, near-threshold probabilities[.45,.55], probability distribution, forced/optional updates and actual staleness. Geometry52N and forced initialization780F are logged but omitted from scheduler regularization.

Primary field-payload bytes/frame=52N+780U, with U=F+optional hard updates. Nominal10Hz feature_payload_BPS=10Σ(52N+780U)/T. Latent BPS=10Σ780U/T; geometry BPS=10Σ52N/T. Full FP32 is3124N and full INT8 is832N for the same three latent and two geometry fields before the vehicle spatial mask. Packet BPS separately uses actual len(packet), including JSON/header, IDs, flags and prefix. The original paper's tensor accounting and quoted285000B/s are distinct accounting conventions; no field is dropped to force agreement, and no vehicle-side filter is retroactively applied to sender counts. Neither nominal BPS is measured network throughput.

Training starts with a fresh V4 MLP, seed0, all46/7362 official train samples, R50/car-only/10Hz and frozen original CoopTrack weights. Batch4; AdamW peak1e-4, weight decay1e-4, clip5; one warmup epoch from1e-5 then cosine to1e-6 over36 epochs. Official val is evaluated every3 epochs; strict AMOTA improvement selects best; ties count as non-improvements; early stop at epoch≥12 after4 consecutive non-improvements. Epoch checkpoints retain optimizer, all RNG states, LR progress, experiment/source/config/weight identity. Preflight optimizer weights are discarded. No V3 MLP checkpoint initializes V4.

Final V4 and P16 use the same frozen infrastructure cache, native model construction, backend settings, data order, receiver fields and RNG reset after model/load/iter(loader), before first next. Seeds0,2025,2026 are inference seeds of one trained model. Raw predictions and evaluator inputs are bound to SHA before evaluation. Historical baselines remain unchanged and appear as labelled references; fresh paired P16 runs fulfill the requested controlled comparison. Large observed BPS gaps are reported as unequal cost, without official-val interval searches. A single λ gives one learned point, not a complete rate–utility curve. Official val is used for selection/early stopping/reporting; it is not an untouched independent test set.

Current preflight: **%s**. Actual commands and assertions are recorded in `audit/preflight.json`, CPU evidence in `audit/cpu_preflight.json`, native2–8frames-per-lane evidence in `metrics/correctness.json`, and logs in `logs/v4_budget_free`. Protection-before/after reports verify old sources, checkpoints and results. Failed gates prevent formal training.

CPU evidence status: %s. Native evidence status: %s. Task gradient norm: %s; comm gradient norm: %s. These are short-snippet proofs, not final performance evidence.
''' % (runtime.MODEL, archive_manifest['status'], archive_manifest['file_count'], archive_manifest['protected_file_count'], cell(preflight.get('status')), cell(cpu.get('status')), cell(correctness.get('status')), cell(correctness.get('task_gradient_norm')), cell(correctness.get('comm_gradient_norm')))
    if migration:
        start = method.index('V3 archive:')
        end = method.index('Actual V3 scheduler state')
        method = (method[:start] + 'V3 Fixed/Shared 已于 2026-10-12 按用户要求废止并清理，仅保留思路文档：`reports/v3/V3_FIXED_SHARED_FINAL.md`。V3 原实验、checkpoint、日志及含环境/源码的归档副本均已删除。V4 使用的 native_hooks 已按字节迁入 V4，共享数据、缓存、官方/V2/V4 权重和 Docker 环境保留。`audit/v3_retirement_transition.json` 记录此次维护前后的源码散列映射和保留资产核验；V4 原 manifest、checkpoint 身份与训练计算保持一致。\n\n'
                  + method[end:])
    (destination / 'V4_METHOD_AND_AUDIT.md').write_text(method, encoding='utf-8')
    lines = ['# V4 results and research status', '', '**%s**; completed training epochs: %d; best epoch: %s.' % (state.get('status'), len(history), cell(selection.get('best_epoch'))), '',
             'The table includes only actual completed evaluations. Pending cells are not results. Full official val is21scenes/3316samples; the three requested seeds measure inference stability of one training seed.', '',
             '| Method | Inference seed | mAP | AMOTA | AMOTP | MOTA | IDS | Payload B/s | Serialized B/s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    lookup = {(row['method'], row['seed']): row for row in observed}
    for seed in (0, 2025, 2026):
        for name in ('V4', 'P16'):
            row = lookup.get((name, seed), {})
            lines.append('| %s | %d | %s |' % (name, seed, ' | '.join(cell(row.get(key)) for key in ('mAP', 'AMOTA', 'AMOTP', 'MOTA', 'IDS', 'feature_payload_BPS', 'serialized_message_BPS'))))
    lines += ['', 'Historical references (unchanged; pairing and protocol caveats are stored with each source SHA in `metrics/reference_results.json`):', '',
              '| Method | Seed | mAP | AMOTA | Reconstructed payload B/s | Serialized B/s |', '|---|---:|---:|---:|---:|---:|']
    for row in references:
        lines.append('| %s | %s | %s |' % (row.get('method'), cell(row.get('seed')), ' | '.join(cell(row.get(key)) for key in ('mAP', 'AMOTA', 'feature_payload_BPS', 'serialized_message_BPS'))))
    lines += ['', 'Paired differences:', '', '```json', json.dumps({'pairs': pairs, 'three_seed_mean_and_sample_std': paired_summary}, indent=2), '```', '',
              'Cost ratios above are descriptive. Nearby cost uses a ±10% payload convention fixed in this reporting code before training; raw ratios are always reported. Unequal payload rates do not establish superiority at the same cost. No statistical significance or cross-training-seed stability is claimed.', '']
    latest = history[-1] if history else {}
    lines += ['1. Budget removal: the separate embedding and excess-budget penalty are removed; the only objective is native task plus0.05×optional communication. Preflight status: %s.' % cell(preflight.get('status')),
              '2. Learned optional UPDATE and actual payload: latest completed training optional rate=%s; payload=%sB/s. Final inference values are in the table; pending training values must not be presented as val results.' % (cell(latest.get('optional_update_ratio')), cell(latest.get('feature_payload_BPS'))),
              '3. Latest completed training soft/hard gap: signed mean=%sB/frame, absolute mean=%sB/frame, near-threshold fraction=%s. Final per-frame data are retained separately.' % (cell(latest.get('soft_hard_gap_mean_B_per_frame')), cell(latest.get('soft_hard_gap_absolute_mean_B_per_frame')), cell(latest.get('near_threshold_fraction'))),
              '4. Short-snippet native task gradient=%s; communication gradient=%s. Cold forced-only frames correctly have no optional gradient. Every formal epoch separately verifies a nonzero native task gradient and freezes native weights/BN.' % (cell(correctness.get('task_gradient_norm')), cell(correctness.get('comm_gradient_norm'))),
              '5. INIT_ONLY-style degeneration: %s. The coordinator records/stops on a completed-epoch optional update rate≤0.001 or≥0.999, finite-loss/gradient failures, and performs no second λ search.' % (state.get('error', 'Not established until completed training and full val; inspect actual optional rate and cache staleness.')),
              '6. Stable advantage versus paired P16 at nearby cost: %s The claim requires all three matched pairs and their observed costs; no result is inferred from the design.' % advantage,
              '7. Evidence supports archive preservation, implemented method and explicitly passed gates/observed rows. Pending performance claims remain hypotheses; completed table entries describe official-val observations only. Independent test generalization and a full rate–utility curve are not established. A single best checkpoint is not a proven Pareto optimum.',
              '8. Paper choice: %s' % paper_choice, '',
              'Official val was used for checkpoint selection/early stopping and performance reporting. These are not independent test-set results. Best checkpoint selection record:', '', '```json', json.dumps(selection, indent=2), '```', '',
              'Run state:', '', '```json', json.dumps(state, indent=2), '```', '',
              'Minimal next action: follow `experiments/v4_budget_free/run_state.json` and `logs/v4_budget_free/coordinator.log`; resume an interrupted compatible run with `python -B scripts/v4_budget_free/run.py --launch --resume`.']
    (destination / 'V4_FINAL.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': snapshot['status'], 'reports': str(destination), 'final_results': len(observed), 'paired_seeds': len(pairs)}), flush=True)
    return snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE.parents[1])
    report(parser.parse_args().root)
