"""Check actual native target-loop outputs on empty, negative and duplicate IDs."""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def verify(source, source_dir):
    sys.path.insert(0, str(source_dir))
    import native_targets
    import torch

    class Tracks(SimpleNamespace):
        def __len__(self):
            return len(self.obj_idxes)

    tree = ast.parse(source.read_text(encoding='utf-8'))
    native_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ClipMatcher')
    cases = [([], []), ([-2, -1, 10], []), ([-2, -1, 10, 11, 10, 12, 13], [10, 11, 14]),
             ([10, 10, 11], [10, 10, 11]), ([-2, -1, 0], [-2, -1, 0])]
    checks = []
    for function in native_class.body:
        if not isinstance(function, ast.FunctionDef) or function.name not in ('match_for_single_frame', 'loss_mem_bank'):
            continue
        transformed, count = native_targets.transform(copy.deepcopy(function))
        indices = [index for index, (before, after) in enumerate(zip(function.body, transformed.body))
                   if native_targets.structural(before) != native_targets.structural(after)]
        assert len(indices) == count
        variants = [compile(ast.fix_missing_locations(ast.Module(
            body=[copy.deepcopy(variant.body[index]) for index in indices], type_ignores=[])),
            '<actual-native-target-loops>', 'exec') for variant in (function, transformed)]
        for device in ('cpu', 'cuda'):
            for case_index, (track_ids, gt_ids) in enumerate(cases):
                results = []
                for variant in variants:
                    tracks = Tracks(obj_idxes=torch.tensor(track_ids, dtype=torch.long, device=device),
                                    matched_gt_idxes=torch.full((len(track_ids),), 99, dtype=torch.long, device=device))
                    scope = {'track_instances': tracks, 'obj_idx_to_gt_idx': {identity: index for index, identity in enumerate(gt_ids)},
                             'num_disappear_track': 0, 'num_pos': 0, 'num_neg': 0, 'pos_bbox_num': 0,
                             'matched_labels': torch.full((len(track_ids),), 3, dtype=torch.long, device=device),
                             'matched_bbox_targets': torch.zeros((len(track_ids), 9), device=device),
                             'matched_bbox_weights': torch.zeros((len(track_ids), 10), device=device),
                             'gt_labels_list': [torch.arange(len(gt_ids), dtype=torch.float32, device=device)],
                             'gt_bboxes_list': [torch.arange(len(gt_ids) * 9, dtype=torch.float64, device=device).reshape(-1, 9)]}
                    exec(variant, scope)
                    results.append({name: scope[name] for name in ('num_disappear_track', 'num_pos', 'num_neg', 'pos_bbox_num',
                                                                   'matched_labels', 'matched_bbox_targets', 'matched_bbox_weights')})
                    results[-1]['matched_gt_idxes'] = tracks.matched_gt_idxes
                    assert tracks.obj_idxes.tolist() == track_ids
                for name in results[0]:
                    before, after = results[0][name], results[1][name]
                    assert torch.equal(before, after) if torch.is_tensor(before) else before == after, name
                checks.append({'method': function.name, 'device': device, 'case': case_index, 'status': 'PASS'})
    assert len(checks) == 20
    return {'status': 'PASS', 'comparison': 'exact native loop targets and counters; CPU and CUDA',
            'cases': ['empty', 'all_unmatched', 'mixed', 'duplicate_gt_last_wins', 'negative_gt'], 'checks': checks}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.source, args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
