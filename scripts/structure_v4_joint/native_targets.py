"""Read native target IDs in bulk while preserving each loop body and tensor write."""
import ast
import copy
import hashlib
import inspect
from pathlib import Path
import textwrap
from types import MethodType


def structural(value):
    if isinstance(value, list):
        return [structural(item) for item in value]
    return ast.dump(value, include_attributes=False)


def transform(function):
    original = copy.deepcopy(function)
    if function.name == 'match_for_single_frame':
        template = ast.parse('for j in range(len(track_instances)):\n    obj_id = track_instances.obj_idxes[j].item()').body[0]
        replacement = ast.parse('for j, obj_id in enumerate(track_instances.obj_idxes.detach().cpu().tolist()):\n    pass').body[0]
        expected = 1
    elif function.name == 'loss_mem_bank':
        template = ast.parse('for track_idx, id in enumerate(track_instances.obj_idxes):\n    cpu_id = id.cpu().numpy().tolist()').body[0]
        replacement = ast.parse('for track_idx, cpu_id in enumerate(track_instances.obj_idxes.cpu().numpy().tolist()):\n    pass').body[0]
        expected = 2
    else:
        raise ValueError('Unsupported native target method: ' + function.name)
    changed = []
    for index, statement in enumerate(function.body):
        if (isinstance(statement, ast.For) and structural(statement.target) == structural(template.target)
                and structural(statement.iter) == structural(template.iter)
                and statement.body and structural(statement.body[0]) == structural(template.body[0])):
            changed.append((index, copy.deepcopy(statement)))
            statement.target = ast.copy_location(copy.deepcopy(replacement.target), statement.target)
            statement.iter = ast.copy_location(copy.deepcopy(replacement.iter), statement.iter)
            statement.body = statement.body[1:]
    if len(changed) != expected:
        raise ValueError('Native ID loop layout changed: ' + function.name)
    restored = copy.deepcopy(function)
    for index, prior in changed:
        current = restored.body[index]
        if structural(current.body) != structural(prior.body[1:]):
            raise ValueError('Native target loop body changed')
        current.target, current.iter = prior.target, prior.iter
        current.body.insert(0, prior.body[0])
    if structural(restored) != structural(original):
        raise ValueError('Native arithmetic or control flow changed outside ID reads')
    return ast.fix_missing_locations(function), len(changed)


def install(criterion):
    proof = {}
    if getattr(criterion, '_structure_bulk_target_reads', False):
        raise ValueError('Native target reads already installed')
    for name in ('match_for_single_frame', 'loss_mem_bank'):
        original = inspect.unwrap(getattr(criterion, name).__func__)
        source = textwrap.dedent(inspect.getsource(original))
        tree = ast.parse(source)
        if len(tree.body) != 1 or tree.body[0].decorator_list:
            raise ValueError('Unexpected native target function definition')
        tree.body[0], count = transform(tree.body[0])
        namespace = dict(original.__globals__)
        exec(compile(ast.fix_missing_locations(tree), '<structure-bulk-target-reads>', 'exec'), namespace)
        setattr(criterion, name, MethodType(namespace[name], criterion))
        proof[name] = {'id_read_loops': count, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                       'all_other_ast_exact': True}
    criterion._structure_bulk_target_reads = True
    return proof


def self_check(source):
    tree = ast.parse(Path(source).read_text(encoding='utf-8'))
    native_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ClipMatcher')
    counts = {}
    for function in native_class.body:
        if isinstance(function, ast.FunctionDef) and function.name in ('match_for_single_frame', 'loss_mem_bank'):
            transformed, count = transform(copy.deepcopy(function))
            compile(ast.fix_missing_locations(ast.Module(body=[transformed], type_ignores=[])), '<target-self-check>', 'exec')
            counts[function.name] = count
    if counts != {'match_for_single_frame': 1, 'loss_mem_bank': 2}:
        raise AssertionError('Native target methods were not checked')
    return {'status': 'PASS', 'id_read_loops': counts, 'loss_and_matching_body_ast_unchanged': True}


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(self_check(args.source)))
