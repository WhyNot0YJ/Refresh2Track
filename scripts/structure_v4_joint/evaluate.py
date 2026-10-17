"""Full official 3316-frame validation with all joint checkpoint weights loaded."""
import argparse
import ast
import csv
from contextlib import contextmanager
import inspect
from pathlib import Path
import sys
import textwrap

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import runtime


def canonical(root, context, measurements=None):
    runtime.bind(root)
    module = runtime.load_module('structure_canonical_evaluate',
                                 Path(root) / 'scripts/v4_budget_free/evaluate.py')
    tree = ast.parse(textwrap.dedent(inspect.getsource(module.evaluate)))
    replacements = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'context'
                and isinstance(node.value, ast.Dict) and not node.value.keys):
            node.value = ast.Name(id='structure_context', ctx=ast.Load())
            replacements += 1
    if replacements != 1:
        raise ValueError('Canonical evaluator context binding changed')
    timed_loops = 0
    for block in tree.body[0].body:
        if not isinstance(block, ast.Try):
            continue
        statements = []
        for statement in block.body:
            if isinstance(statement, ast.With):
                statements.extend(ast.parse('''
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
structure_inference_started = time.perf_counter()
''').body)
                statements.append(statement)
                statements.extend(ast.parse('''
torch.cuda.synchronize()
structure_inference_seconds = time.perf_counter() - structure_inference_started
structure_measurements.update(
    inference_seconds=structure_inference_seconds,
    inference_frames=len(outputs),
    inference_FPS=len(outputs) / structure_inference_seconds,
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    scope='wall time for the official validation loop: data loading, receiver, inference and frame capture; excludes construction and official metric evaluation')
''').body)
                timed_loops += 1
            else:
                statements.append(statement)
        block.body = statements
    if timed_loops != 1:
        raise ValueError('Expected exactly one complete canonical inference loop')
    namespace = dict(module.__dict__, structure_context=context,
                     structure_measurements=measurements if measurements is not None else {}, SEEDS=(0,))
    exec(compile(ast.fix_missing_locations(tree), '<structure-canonical-val>', 'exec'), namespace)
    return module, namespace


def checkpoint_state(root, checkpoint, epoch):
    root, checkpoint = Path(root).resolve(), Path(checkpoint).resolve()
    if checkpoint.parent != runtime.experiment(root) / 'checkpoints':
        raise ValueError('Validation requires an isolated C_GBA_GATE checkpoint')
    runtime.initialize(root)
    state = runtime.native.torch.load(str(checkpoint), map_location='cpu')
    if state.get('identity') != runtime.identity(root) or state.get('epoch') != epoch:
        raise ValueError('Structure checkpoint identity or completed epoch differs')
    if epoch < 1 or epoch > 36 or not state.get('native_trainable') or 'scheduler' not in state:
        raise ValueError('Checkpoint must contain Scheduler and all updated native/new structure parameters')
    if state.get('normalization') != runtime.read(runtime.experiment(root) / 'normalization.json'):
        raise ValueError('Checkpoint normalization differs from the frozen train-only calibration')
    return state


@contextmanager
def inference_bindings(module, namespace, root, state, context):
    import modules
    from receiver import ReceiverCache
    torch = runtime.native.torch
    original_build = module.native.build_model
    removers, frames = [], []
    loaded = {}

    class EvaluationReceiver(ReceiverCache):
        def unpack(self, packet, meta):
            decoded, header = super().unpack(packet, meta)
            context.update(samples=[{'receiver_state': self.receiver_state.clone(),
                                     'receiver_ids': self.receiver_ids.tolist()}],
                           matching_cursor=0, frame=dict(meta), receiver=self)
            context.pop('active_sample', None)
            return decoded, header

    def build(root_arg, cfg, role):
        if role != 'coop' or Path(root_arg).resolve() != Path(root).resolve():
            raise ValueError('Structure evaluator can only construct the registered cooperative model')
        model = original_build(root_arg, cfg, role)
        modules.add_structures(model, state['normalization'], experiment=runtime.MODEL)
        modules.load_native_state(model, state['native_trainable'], experiment=runtime.MODEL)
        actual = modules.state_hash(modules.native_state(model, experiment=runtime.MODEL))
        expected = modules.state_hash(state['native_trainable'])
        if actual != expected:
            raise ValueError('Actual joint native/new weights do not match the evaluated checkpoint')
        model.eval().requires_grad_(False)
        remove, proof = modules.install(model, context)
        removers.append(remove)
        original_pts = model.crossview_alignment.transform_pts

        def transform_pts(points, transform):
            output, keep = original_pts(points, transform)
            sample = context['samples'][0]
            sidecar = context['receiver'].receiver_state.to(device=points.device, dtype=points.dtype)
            if len(sidecar) != len(points):
                raise ValueError('Evaluation receiver sidecar and CAA input row counts differ')
            sample['receiver_state'] = sidecar[keep]
            sample['caa_keep_indices'] = keep.nonzero().flatten().detach().cpu().tolist()
            return output, keep

        model.crossview_alignment.transform_pts = transform_pts
        removers.append(lambda: setattr(model.crossview_alignment, 'transform_pts', original_pts))

        def capture(native_model, incoming, output):
            sample = context['samples'][0]
            gains = sample.get('gate_values', torch.empty(0, 3)).detach().cpu()
            state_values = sample['receiver_state'].detach().cpu()
            frames.append(dict(context['frame'], receiver_ids=sample['receiver_ids'],
                               caa_keep_indices=sample.get('caa_keep_indices', []),
                               aligned_queries=len(state_values),
                               matched_pairs=sample.get('matched_pairs', 0),
                               gate_values=gains.tolist(),
                               receiver_state=state_values.tolist()))

        handle = model.register_forward_hook(capture)
        removers.append(handle.remove)
        loaded.update(native_trainable_sha256=actual, strict_native_allowlist_load=True,
                      native_initializer_sha256=runtime.digest(Path(root) / 'checkpoints/cooptrack_r50_coop.pth'),
                      normalization_sha256=runtime.digest(runtime.experiment(root) / 'normalization.json'),
                      native_and_reasoner_eval=True, hook=proof)
        return model

    module.native.build_model = build
    namespace['ReceiverCache'] = EvaluationReceiver
    try:
        yield loaded, frames
    finally:
        module.native.build_model = original_build
        for remove in reversed(removers):
            remove()


def diagnostics(frames):
    np = runtime.native.np
    values = [value for frame in frames for value in frame['gate_values']]
    gates = np.asarray(values, dtype=float).reshape(-1, 3)
    bins = {'0': [], '1-3': [], '4-7': [], '8+': []}
    for frame in frames:
        for state in frame['receiver_state']:
            age = int(round(float(np.expm1(state[0]))))
            bins['0' if age == 0 else '1-3' if age < 4 else '4-7' if age < 8 else '8+'].append(age)
    return {'status': 'PASS', 'frames': len(frames), 'matched_pairs': sum(frame['matched_pairs'] for frame in frames),
            'gate_branch_names': ['features', 'motion', 'embedding'],
            'gate_branch_mean': gates.mean(0).tolist() if len(gates) else None,
            'gate_branch_quantiles': {str(quantile): np.quantile(gates, quantile, axis=0).tolist()
                                      for quantile in (0., .05, .5, .95, 1.)} if len(gates) else {},
            'aligned_receiver_age_bins': {name: len(ages) for name, ages in bins.items()},
            'association_gt_analysis': 'not available in official unlabeled inference; no validation GT used by structures',
            'receiver_metadata_sent_online': False, 'caa_mask_shared_with_native': True}


def evaluate(root, checkpoint, epoch, selection=False):
    root, checkpoint = Path(root).resolve(), Path(checkpoint).resolve()
    state = checkpoint_state(root, checkpoint, epoch)
    directory = runtime.experiment(root) / 'validation'
    directory = directory / ('epoch%03d' % epoch) if selection else directory / 'final/V4_seed0'
    path = directory / 'result.json'
    if path.exists():
        result = runtime.read(path)
        proof = result.get('loaded_structure_weights', {})
        import modules
        if (result.get('status') != 'PASS' or result.get('sample_count') != 3316
                or result.get('scene_count') != 21 or result['identity']['protocol'] != runtime.identity(root)
                or result['checkpoint_sha256'] != runtime.digest(checkpoint)
                or proof.get('native_trainable_sha256') != modules.state_hash(state['native_trainable'])
                or proof.get('strict_native_allowlist_load') is not True):
            raise ValueError('Existing structure evaluation is incomplete or incompatible')
        manifest = runtime.read(directory / 'run_manifest.json')
        if (manifest.get('structure_result_sha256') != runtime.digest(path)
                or manifest.get('communication_sha256') != runtime.digest(directory / 'communication.json')):
            raise ValueError('Existing structure evaluation evidence changed')
        for kind, expected in manifest['metrics_sha256'].items():
            if runtime.digest(directory / kind / 'metrics_summary.json') != expected:
                raise ValueError('Existing official evaluator metric changed: ' + kind)
        return result
    context, measurements = {}, {}
    module, namespace = canonical(root, context, measurements)
    with inference_bindings(module, namespace, root, state, context) as (loaded, frames):
        result = namespace['evaluate'](root, 'V4', checkpoint, epoch, 0, selection)
    if len(frames) != 3316 or not loaded:
        raise ValueError('Structure evaluation did not apply hooks to all official validation frames')
    if measurements.get('inference_frames') != 3316 or measurements.get('inference_seconds', 0.) <= 0:
        raise ValueError('Full validation inference timing is missing or incomplete')
    result['loaded_structure_weights'] = dict(loaded)
    result['measured_inference'] = measurements
    result.update({name: measurements[name] for name in (
        'inference_seconds', 'inference_FPS', 'peak_allocated_bytes', 'peak_reserved_bytes')})
    result['checkpoint_role'] = 'base CoopTrack + complete updated GBA/fusion/additions + trained fresh Scheduler'
    result['structure_diagnostics'] = diagnostics(frames)
    runtime.write(directory / 'structure_frames.json', frames)
    runtime.write(directory / 'structure_diagnostics.json', result['structure_diagnostics'])
    runtime.write(path, result)
    manifest = runtime.read(directory / 'run_manifest.json')
    manifest.update(loaded_structure_weights=dict(loaded),
                    measured_inference=measurements,
                    structure_result_sha256=runtime.digest(path),
                    communication_sha256=runtime.digest(directory / 'communication.json'),
                    structure_frames_sha256=runtime.digest(directory / 'structure_frames.json'))
    runtime.write(directory / 'run_manifest.json', manifest)
    communication = runtime.read(directory / 'communication.json')
    target = runtime.experiment(root) / 'communication' / ('val_epoch%03d.csv' % epoch)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = communication['frames']
    keys = [key for key, value in records[0].items() if not isinstance(value, (list, dict))]
    with target.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    runtime.write(runtime.experiment(root) / 'metrics/association_or_gate_diagnostics.json',
                  result['structure_diagnostics'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE.parents[1])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--selection', action='store_true')
    args = parser.parse_args()
    evaluate(args.root, args.checkpoint, args.epoch, selection=args.selection)
