"""Separate-process, first-frame GT deletion audit; never invokes metrics or training."""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
from types import MethodType
sys.dont_write_bytecode = True
os.environ['REFRESH2TRACK_STRUCTURE_MODEL'] = 'C_GBA_GATE'

def annotation(key):
    key = str(key).lower()
    return (key.startswith('gt_') or key in ('instance_inds', 'ann_info', 'annotations')
            or ('future' in key and any(part in key for part in ('label', 'gt', 'mask', 'bbox'))))

def scrub(value, path='', found=None):
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            name = path + '/' + str(key)
            if annotation(key):
                found.append({'path': name, 'elements': count(item)})
            else:
                output[key] = scrub(item, name, found)
        return output
    if isinstance(value, (list, tuple)):
        return type(value)(scrub(item, path + '/' + str(index), found) for index, item in enumerate(value))
    return value

def count(value):
    if hasattr(value, 'tensor'):
        return count(value.tensor)
    if hasattr(value, 'numel'):
        return int(value.numel())
    if isinstance(value, dict):
        return sum(count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(count(item) for item in value)
    return 0

def guarded_read(mapping, key, *default):
    if annotation(key):
        raise AssertionError('Inference accessed an annotation: ' + str(key))
    return mapping.get(key, *default) if default else mapping[key]

class GuardReads(ast.NodeTransformer):
    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'kwargs' and isinstance(node.ctx, ast.Load):
            key = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
            return ast.copy_location(ast.Call(ast.Name('_audit_read', ast.Load()), [node.value, key], []), node)
        return node
    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'kwargs' and node.func.attr == 'get'):
            return ast.copy_location(ast.Call(ast.Name('_audit_get', ast.Load()), [node.func.value] + node.args, node.keywords), node)
        return node


def guard_model(model, mode, observations):
    for name in ('forward_test', 'simple_test_track', '_forward_single_frame_inference'):
        original = inspect.unwrap(getattr(model, name).__func__)
        tree = GuardReads().visit(ast.parse(textwrap.dedent(inspect.getsource(original))))
        tree.body[0].decorator_list = []
        namespace = dict(original.__globals__, _audit_read=guarded_read,
                         _audit_get=lambda mapping, key, default=None: guarded_read(mapping, key, default))
        exec(compile(ast.fix_missing_locations(tree), '<audit-kwargs-read-guard>', 'exec'), namespace)
        setattr(model, name, MethodType(namespace[original.__name__], model))
    original = model.forward_test

    def forward_test(self, *args, **kwargs):
        found = []
        clean, clean_args = scrub(kwargs, found=found), scrub(args, path='/args', found=found)
        observations.append(found)
        return original(*(clean_args if mode == 'drop_gt' else args), **(clean if mode == 'drop_gt' else kwargs))

    def forbidden(*args, **kwargs):
        raise AssertionError('Training GT/criterion path entered during inference')

    model.forward_test = MethodType(forward_test, model)
    model.loss_single_batch = forbidden
    model.STReasoner._gen_asso_label = forbidden
    for name in ('initialize_for_single_clip', 'match_for_single_frame', 'loss_prediction', 'loss_mem_bank'):
        if hasattr(model.criterion, name):
            setattr(model.criterion, name, forbidden)
    return model

def same(node, expression):
    return ast.dump(node) == ast.dump(ast.parse(expression, mode='eval').body)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sha256', default='8715d2b620f6dd772cb8c57a739811fc3890c5a83d01d97a7c4004f669fadcdf')
    parser.add_argument('--mode', required=True, choices=('normal', 'drop_gt'))
    parser.add_argument('--attempt', type=int, default=1, choices=range(1, 100))
    parser.add_argument('--frames', type=int, default=16, choices=range(8, 129))
    args = parser.parse_args()
    root, checkpoint = Path(args.root).resolve(), Path(args.checkpoint).resolve()
    audit = root / 'audits/structure_accuracy_20261017'
    if audit not in checkpoint.parents:
        raise ValueError('Only the temporary audit checkpoint is permitted')
    scratch = audit / ('gt_probe_' + args.mode + '_attempt%d' % args.attempt)
    if scratch.exists():
        raise ValueError('Audit output already exists; preserve it and choose a fresh audit directory')
    sys.path.insert(0, str(root / 'scripts/structure_v4_joint'))
    import runtime
    import evaluate
    import torch
    if runtime.digest(checkpoint) != args.sha256:
        raise ValueError('Uploaded checkpoint SHA256 mismatch')
    state = torch.load(str(checkpoint), map_location='cpu')
    scratch.mkdir(parents=True)
    runtime.experiment = lambda root, model=None: scratch
    runtime.identity = lambda root, model=None: state['identity']
    runtime.write(scratch / 'normalization.json', state['normalization'])
    snapshot = root / 'experiments/structure_v4_joint/source/CoopTrack'
    for name in runtime.git(root, 'ls-files', '-z').decode().split('\0'):
        source = root / 'code/CoopTrack' / name
        if name and source.is_file() and not source.is_symlink() and not (snapshot / name).is_file():
            raise ValueError('Existing immutable source snapshot is incomplete: ' + name)
    config_tree = runtime.config_tree

    def audit_config_tree():
        tree, replaced = config_tree(), 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == 'experiments/structure_v4_joint/C_GBA_GATE/configs/generated':
                node.value = (scratch.relative_to(root) / 'configs').as_posix()
                replaced += 1
        assert replaced == 1
        return tree

    runtime.config_tree = audit_config_tree
    runtime.initialize(root)
    context, observations = {}, []
    module, namespace = evaluate.canonical(root, context)
    tree = ast.parse(textwrap.dedent(inspect.getsource(module.evaluate)))
    changed = dict(context=0, checkpoint=0, loop=0, length=0, metrics=0)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'context':
            node.value = ast.Name('_audit_context', ast.Load()); changed['context'] += 1
        if isinstance(node, ast.If) and same(node.test, "checkpoint.parent != experiment(root) / 'checkpoints'"):
            node.test.comparators[0] = ast.Name('_audit_checkpoint_parent', ast.Load()); changed['checkpoint'] += 1
        if isinstance(node, ast.For) and ast.dump(node.target) == ast.dump(ast.parse('position, info = ()').body[0].targets[0]):
            node.body[:0] = ast.parse('if position >= _audit_frames: break').body; changed['loop'] += 1
        if isinstance(node, ast.If) and same(node.test, 'len(outputs) != 3316'):
            node.test.comparators[0] = ast.Name('_audit_frames', ast.Load()); changed['length'] += 1
        if isinstance(node, ast.Try):
            for index, statement in enumerate(node.body):
                if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name) and statement.value.func.id == 'bind_and_evaluate':
                    node.body[index:] = ast.parse('return _audit_finish(outputs, records, pair_identity, after_inference)').body
                    changed['metrics'] += 1
                    break
    assert changed == dict.fromkeys(changed, 1), changed

    def fingerprint(value):
        if hasattr(value, 'tensor'):
            return fingerprint(value.tensor)
        if torch.is_tensor(value):
            value = value.detach().contiguous().cpu().numpy()
        if hasattr(value, 'dtype') and hasattr(value, 'tobytes'):
            return {'dtype': str(value.dtype), 'shape': list(value.shape), 'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
        if isinstance(value, dict):
            return {key: fingerprint(item) for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [fingerprint(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError('Unhashed prediction type: ' + str(type(value)))

    def finish(outputs, records, pairing, rng):
        assert len(observations) == len(outputs) == args.frames
        assert any(item['elements'] > 0 for frame in observations for item in frame), 'No nonempty GT was intervened on'
        return dict(status='PASS', mode=args.mode, frames=args.frames, checkpoint_sha256=args.sha256,
                    epoch=state['epoch'], prediction_fingerprints=fingerprint(outputs), communication=records,
                    pairing_identity=pairing, rng_after_inference=rng, annotations_at_model_entry=observations,
                    scope='First contiguous official val frames only; audit path/identity adapter, not a full formal preflight; no metrics or optimization')

    namespace.update(_audit_context=context, _audit_checkpoint_parent=checkpoint.parent,
                     _audit_frames=args.frames, _audit_finish=finish)
    exec(compile(ast.fix_missing_locations(tree), '<audit-first-frames-no-metrics>', 'exec'), namespace)
    with evaluate.inference_bindings(module, namespace, root, state, context) as (loaded, frames):
        original_build = module.native.build_model
        module.native.build_model = lambda *a, **kw: guard_model(original_build(*a, **kw), args.mode, observations)
        try:
            result = namespace['evaluate'](root, 'V4', checkpoint, state['epoch'], 0, False)
        finally:
            module.native.build_model = original_build
        result['loaded_structure_weights'] = dict(loaded)
        result['structure_frames'] = frames
    runtime.write(scratch / 'result.json', result)
    print(json.dumps({'status': 'PASS', 'mode': args.mode, 'result': str(scratch / 'result.json')}), flush=True)


if __name__ == '__main__':
    main()
