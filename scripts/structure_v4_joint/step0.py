"""Compare native and zero-initialized matching on four real train batches."""
import argparse
import gc
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import modules
import runtime
import train


def equivalent(root):
    root = Path(root).resolve()
    runtime.initialize(root)
    base = train.canonical(root)
    settings = runtime.load_settings(root)
    cfg, dataset, scenes, sender_rows = base.load_training(root)
    model, scheduler, optimizer = train.build(root, base, cfg, dataset, settings, old_scheduler=True)
    torch = runtime.native.torch
    context, streams = {}, {}
    reasoner = model.STReasoner
    remove_structure, structure_proof = modules.install(model, context)
    modified_match = reasoner._learn_matching
    parallel, remove_native, native_proof = train.attach(base, model, context, structured=False)
    wrapped_match = reasoner._learn_matching
    wrapped_fusion = reasoner._query_fusion
    original_rule = reasoner._query_matching
    evidence, batch_rows, active_rows = [], [], {}
    tolerances = {'atol': 2e-5, 'rtol': 2e-5}

    def compare_indices(reference, actual, stage):
        reference = torch.as_tensor(reference, dtype=torch.long).detach().cpu()
        actual = torch.as_tensor(actual, dtype=torch.long).detach().cpu()
        if reference.shape != actual.shape or not torch.equal(reference, actual):
            raise ValueError('Step0 matching indices differ: ' + stage)
        return actual.tolist()

    def compare_logits(reference, actual, stage):
        if reference.shape != actual.shape or not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
            raise ValueError('Step0 matching logits shape or finite values differ: ' + stage)
        if not torch.allclose(reference, actual, **tolerances):
            raise ValueError('Step0 native matching logits differ: ' + stage)
        return float((reference - actual).abs().max()) if reference.numel() else 0.

    def checked_match(*args, **kwargs):
        lane = context['matching_cursor']
        if not reasoner.training:
            raise ValueError('Real training matching unexpectedly entered inference mode')
        reference = type(reasoner)._learn_matching(reasoner, *args, **kwargs)
        actual = wrapped_match(*args, **kwargs)
        if context['matching_cursor'] != lane + 1:
            raise ValueError('Native lane cursor did not advance exactly once')
        sample = context['samples'][lane]
        row = {'lane': lane, 'scene': sample['scene'], 'token': sample['token'],
               'train_logits_shape': list(actual[2].shape),
               'train_logits_max_abs_delta': compare_logits(reference[2], actual[2], 'training'),
               'train_vehicle_indices': compare_indices(reference[0], actual[0], 'training vehicle'),
               'train_infrastructure_indices': compare_indices(reference[1], actual[1], 'training infrastructure'),
               'association_positive_pairs': sample.get('association_positive_pairs', 0),
               'association_negative_pairs': sample.get('association_negative_pairs', 0)}
        active_rows[lane] = row
        previous_training = reasoner.training
        try:
            reasoner.training = False
            context['matching_cursor'] = lane
            inference_reference = type(reasoner)._learn_matching(reasoner, *args, **kwargs)
            inference_actual = modified_match(*args, **kwargs)
            row.update(
                inference_logits_shape=list(inference_actual[2].shape),
                inference_logits_max_abs_delta=compare_logits(
                    inference_reference[2], inference_actual[2], 'inference'),
                inference_vehicle_indices=compare_indices(
                    inference_reference[0], inference_actual[0], 'inference Hungarian vehicle'),
                inference_infrastructure_indices=compare_indices(
                    inference_reference[1], inference_actual[1], 'inference Hungarian infrastructure'))
        finally:
            reasoner.training = previous_training
            context['matching_cursor'] = lane + 1
            context['active_sample'] = lane
        return actual

    def checked_rule(*args, **kwargs):
        lane = context['active_sample']
        reference = type(reasoner)._query_matching(reasoner, *args, **kwargs)
        actual = original_rule(*args, **kwargs)
        row = active_rows[lane]
        row['geometric_vehicle_indices'] = compare_indices(reference[0], actual[0], 'training geometric vehicle')
        row['geometric_infrastructure_indices'] = compare_indices(
            reference[1], actual[1], 'training geometric infrastructure')
        return actual

    def checked_fusion(inf, veh, inf_idx, veh_idx):
        lane = context['active_sample']
        row = active_rows[lane]
        compare_indices(row['geometric_vehicle_indices'], veh_idx, 'actual fusion vehicle input')
        compare_indices(row['geometric_infrastructure_indices'], inf_idx, 'actual fusion infrastructure input')
        reference = type(reasoner)._query_fusion(reasoner, inf, veh, inf_idx, veh_idx)
        actual = wrapped_fusion(inf, veh, inf_idx, veh_idx)
        deltas = {}
        for name in ('cache_query_feats', 'cache_motion_feats', 'cache_query_embeds', 'cache_ref_pts'):
            deltas[name] = compare_logits(getattr(reference, name), getattr(actual, name), 'fusion ' + name)
        row['fusion_max_abs_deltas'] = deltas
        return actual

    reasoner._learn_matching = checked_match
    reasoner._query_matching = checked_rule
    reasoner._query_fusion = checked_fusion
    try:
        runtime.inherited.seed(0)
        model.reset_memory()
        batches = train.entries_for(base, scenes, 1)[:4]
        if len(batches) != 4 or any(len(entries) != 4 for entries in batches):
            raise ValueError('Step0 matching equivalence requires four real batch-four train batches')
        for step, entries in enumerate(batches, 1):
            active_rows.clear()
            with torch.no_grad():
                value = train.run_batch(root, base, 1, step, entries, dataset, streams, sender_rows,
                                        model, scheduler, parallel, optimizer, settings, context,
                                        optimize=False, diagnose=False)
            if value['gradient_norms'] or optimizer.state:
                raise ValueError('Step0 matching equivalence unexpectedly updated optimizer state')
            if len(active_rows) != len(entries) or any('fusion_max_abs_deltas' not in row for row in active_rows.values()):
                raise ValueError('Step0 matching did not cover every native lane and actual fusion')
            evidence.extend(dict(row, batch=step) for lane, row in sorted(active_rows.items()))
            batch_rows.append({'batch': step, 'frames': len(entries), 'task_loss': value['task_loss'],
                               'comm_loss': value['comm_loss'],
                               'packet_sha256': [row['packet_sha256'] for row in value['records']]})
            train.emit(phase='step0_matching_equivalence', batch=step, real_frames=len(evidence))
        train_pairs = sum(row['train_logits_shape'][0] * row['train_logits_shape'][1] for row in evidence)
        geometric_pairs = sum(len(row['geometric_vehicle_indices']) for row in evidence)
        inference_pairs = sum(len(row['inference_vehicle_indices']) for row in evidence)
        if not train_pairs or not geometric_pairs or not inference_pairs:
            raise ValueError('Step0 matching equivalence has no real nonempty train, geometric or Hungarian pairs')
        checkpoint = root / 'experiments/v4_budget_free/checkpoints/best.pth'
        result = {'status': 'PASS', 'real_batches': 4, 'batch_size': 4, 'real_frames': len(evidence),
                  'split': 'official train', 'optimizer_steps': 0, 'new_training_runs': 0,
                  'scheduler_checkpoint': {'path': str(checkpoint), 'sha256': runtime.digest(checkpoint),
                                           'purpose': 'historical V4 best for step0 only; never formal training'},
                  'scheduler_sha256': modules.state_hash(scheduler.state_dict()),
                  'normalization': train.normalization(root), 'tolerances': tolerances,
                  'training_candidate_pairs': train_pairs, 'training_geometric_pairs': geometric_pairs,
                  'inference_hungarian_pairs': inference_pairs,
                  'association_positive_pairs': sum(row['association_positive_pairs'] for row in evidence),
                  'association_negative_pairs': sum(row['association_negative_pairs'] for row in evidence),
                  'matching_indices_exact': True, 'actual_fusion_indices_exact': True,
                  'temporary_inference_only': 'same current real train instances, original inference Hungarian; no val GT',
                  'native_hook_proof': native_proof, 'structure_hook_proof': structure_proof,
                  'batches': batch_rows, 'frames': evidence}
        runtime.write(runtime.experiment(root) / 'preflight/step0_matching_equivalence.json', result)
        return result
    finally:
        reasoner._learn_matching = wrapped_match
        reasoner._query_matching = original_rule
        reasoner._query_fusion = wrapped_fusion
        remove_native()
        remove_structure()
        streams.clear()
        context.clear()
        del parallel, model, scheduler, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    arguments = parser.parse_args()
    result = equivalent(arguments.root)
    print(json.dumps({key: result[key] for key in ('status', 'real_batches', 'real_frames',
                                                  'training_candidate_pairs', 'training_geometric_pairs',
                                                  'inference_hungarian_pairs')}, allow_nan=False))


if __name__ == '__main__':
    main()
