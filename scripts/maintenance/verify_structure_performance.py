"""Compare independent real-data training processes with exact recursive byte hashes."""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import shutil
import struct
import sys


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value, torch, numpy):
    leaves = {}
    active = set()

    def visit(item, path):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            payload = tensor.numpy().tobytes()
            prefix = 'tensor:' + str(tensor.dtype) + ':' + str(tuple(tensor.shape))
        elif isinstance(item, numpy.ndarray):
            if item.dtype.hasobject:
                raise TypeError('Object NumPy arrays cannot be byte-hashed: ' + path)
            payload = numpy.ascontiguousarray(item).tobytes()
            prefix = 'numpy:' + item.dtype.str + ':' + str(item.shape)
        elif isinstance(item, numpy.generic):
            payload, prefix = item.tobytes(), 'numpy_scalar:' + item.dtype.str
        elif item is None or isinstance(item, (str, bytes, bool, int, float)):
            prefix = type(item).__name__
            if isinstance(item, float):
                payload = struct.pack('>d', item)
            elif isinstance(item, bytes):
                payload = item
            else:
                payload = str(item).encode('utf-8')
        elif callable(item):
            prefix, payload = 'callable', item.__qualname__.encode('utf-8')
        else:
            identity = id(item)
            if identity in active:
                raise ValueError('Cyclic state cannot be verified: ' + path)
            active.add(identity)
            try:
                if isinstance(item, dict):
                    leaves[path + '/@type'] = 'dict:' + str(len(item))
                    for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                        visit(item[key], path + '/' + repr(key))
                elif isinstance(item, (tuple, list)):
                    leaves[path + '/@type'] = type(item).__name__ + ':' + str(len(item))
                    for index, child in enumerate(item):
                        visit(child, path + '/' + str(index))
                elif hasattr(item, 'get_fields'):
                    leaves[path + '/@type'] = type(item).__name__
                    visit(item.get_fields(), path + '/fields')
                    visit({name: child for name, child in vars(item).items() if name != '_fields'},
                          path + '/attributes')
                elif hasattr(item, '__dict__'):
                    leaves[path + '/@type'] = type(item).__name__
                    visit(vars(item), path + '/attributes')
                else:
                    raise TypeError('Unverified state type at ' + path + ': ' + str(type(item)))
            finally:
                active.remove(identity)
            return
        leaves[path] = prefix + ':' + hashlib.sha256(payload).hexdigest()

    visit(value, '$')
    encoded = json.dumps(leaves, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {'sha256': hashlib.sha256(encoded).hexdigest(), 'leaves': leaves}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def compare(first_path, second_path, output):
    first = json.loads(Path(first_path).read_text(encoding='utf-8'))
    second = json.loads(Path(second_path).read_text(encoding='utf-8'))
    differences = []

    def walk(before, after, path):
        if type(before) is not type(after):
            differences.append({'path': path, 'before': before, 'after': after})
        elif isinstance(before, dict):
            for key in sorted(set(before) | set(after)):
                if key not in before or key not in after:
                    differences.append({'path': path + '/' + key, 'missing_in': 'first' if key not in before else 'second'})
                else:
                    walk(before[key], after[key], path + '/' + key)
        elif isinstance(before, list):
            if len(before) != len(after):
                differences.append({'path': path + '/length', 'before': len(before), 'after': len(after)})
            for index, (left, right) in enumerate(zip(before, after)):
                walk(left, right, path + '/' + str(index))
        elif before != after:
            differences.append({'path': path, 'before': before, 'after': after})

    for key in ('initial', 'steps', 'final', 'coverage'):
        walk(first[key], second[key], key)
    result = {'status': 'PASS' if not differences else 'FAIL', 'comparison': 'exact bytes; zero tolerance',
              'first': str(first_path), 'second': str(second_path),
              'first_label': first['label'], 'second_label': second['label'],
              'differences': len(differences), 'first_differences': differences[:80],
              'first_source_sha256': first['source_sha256'], 'second_source_sha256': second['source_sha256'],
              'coverage': first['coverage']}
    if output:
        write(output, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if not differences else 1


def run(args):
    root, source, output = args.root.resolve(), args.source_dir.resolve(), args.output.resolve()
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(source), str(root / 'scripts')]
    runtime = importlib.import_module('runtime')
    train = importlib.import_module('train')
    modules = importlib.import_module('modules')
    receiver = importlib.import_module('receiver')
    for module in (runtime, train, modules, receiver):
        if Path(module.__file__).resolve().parent != source:
            raise RuntimeError('Wrong source import: ' + str(module.__file__))
    receiver.SCRIPTS = root / 'scripts'
    artifacts = output.parent / (args.label + '_artifacts')
    artifacts.mkdir(parents=True, exist_ok=True)
    normalization = root / 'experiments/structure_v4_joint/performance_20261016/normalization.json'
    shutil.copy2(normalization, artifacts / 'normalization.json')
    runtime.experiment = lambda root, model=None: artifacts
    source_hashes = {path.name: file_hash(path) for path in sorted(source.glob('*.py'))}
    source_sha = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode('utf-8')).hexdigest()
    runtime.initialize(root)
    base = train.canonical(root)
    torch, numpy = runtime.native.torch, runtime.native.np
    capture = lambda value: fingerprint(value, torch, numpy)
    settings = runtime.load_settings(root)
    cfg, dataset, scenes, sender_rows = base.load_training(root)
    model, scheduler, optimizer = train.build(root, base, cfg, dataset, settings)
    named = modules.named_trainables(model, scheduler)
    names = {id(parameter): name for name, parameter in named}
    context, streams = {}, {}
    parallel, remove, instrumentation = train.attach(base, model, context)
    frozen = modules.frozen_snapshot(model)
    batches = train.entries_for(base, scenes, 1)
    model.reset_memory()
    result = {'label': args.label, 'source_dir': str(source), 'source_hashes': source_hashes,
              'source_sha256': source_sha, 'comparison': 'exact contiguous dtype/shape/bytes; no pickle hashes',
              'initial': {'settings': capture(settings), 'parameters': capture(dict(named)),
                          'normalization': capture(train.normalization(root)),
                          'frozen_weights': modules.frozen_hash(model),
                          'buffers': capture(dict(model.named_buffers())),
                          'rng': capture(runtime.inherited.rng_state()),
                          'optimizer': capture(optimizer.state_dict()),
                          'parameter_proof': modules.parameter_proof(model, scheduler, optimizer),
                          'instrumentation': instrumentation}, 'steps': []}
    state = {}
    original_prepare, original_pack = base.prepare_batch, base.pack
    original_save = base.training.save_batch_state
    original_parse, original_clip = model._parse_losses, torch.nn.utils.clip_grad_norm_
    original_backward = torch.Tensor.backward
    original_geometry = modules.geometry_features
    original_alignment = model.crossview_alignment.transform_pts

    def prepare(*incoming, **kwargs):
        value = original_prepare(*incoming, **kwargs)
        state['input_batch'] = capture(value[0])
        state['prepared_samples'] = capture(value[1])
        return value

    def pack(*incoming, **kwargs):
        packet = original_pack(*incoming, **kwargs)
        flags = incoming[2] if len(incoming) > 2 else kwargs.get('updates', kwargs.get('flags'))
        state['packets'].append({'bytes': len(packet), 'sha256': hashlib.sha256(packet).hexdigest(),
                                 'flags': capture(flags)})
        return packet

    def parse(losses):
        state['raw_losses'] = capture(losses)
        value = original_parse(losses)
        state['parsed_task'] = capture(value[0])
        return value

    def backward(tensor, *incoming, **kwargs):
        state['backward_objective'] = capture(tensor)
        return original_backward(tensor, *incoming, **kwargs)

    def clip(parameters, *incoming, **kwargs):
        parameters = list(parameters)
        state['clip_parameter_order'] = [names[id(parameter)] for parameter in parameters]
        state['gradients_before_clip'] = capture({name: parameter.grad for name, parameter in named})
        value = original_clip(parameters, *incoming, **kwargs)
        state['clip_return'] = capture(value)
        state['gradients_after_clip'] = capture({name: parameter.grad for name, parameter in named})
        return value

    def save(model, entries, streams):
        excluded = {'raw_pair_features', 'raw_gate_features', 'gate_values', 'caa_keep_indices'}
        samples = [{key: value for key, value in sample.items() if key not in excluded}
                   for sample in context['samples']]
        state['forward_samples'] = capture(samples)
        state['native_temporal'] = capture(model.train_prev_infos)
        return original_save(model, entries, streams)

    def geometry(*incoming, **kwargs):
        value = original_geometry(*incoming, **kwargs)
        state['geometry'].append(capture(value))
        return value

    def alignment(*incoming, **kwargs):
        value = original_alignment(*incoming, **kwargs)
        state['alignment'].append(capture(value))
        return value

    def gate_output(module, incoming, output):
        state['gate_logits'].append(capture(output))

    base.prepare_batch, base.pack = prepare, pack
    base.training.save_batch_state = save
    model._parse_losses = parse
    torch.nn.utils.clip_grad_norm_, torch.Tensor.backward = clip, backward
    modules.geometry_features = geometry
    model.crossview_alignment.transform_pts = alignment
    gate_hook = model.STReasoner.fusion_branch_gate.register_forward_hook(gate_output)
    plan = [('continuous', index, batch) for index, batch in enumerate(batches[:args.batches], 1)]
    if args.reset_batches:
        used_scenes = {entry['scene'] for _, _, batch in plan for entry in batch}
        reset_scenes = {name: scenes[name] for name in sorted(scenes) if name not in used_scenes}
        reset_batches = list(base.training.scene_batches(reset_scenes, list(reset_scenes), 4))
        plan.extend(('scene_reset', index, batch)
                    for index, batch in enumerate(reset_batches[:args.reset_batches], 1))
    coverage = {'continuous_batches': args.batches, 'reset_batches': args.reset_batches,
                'real_frames': 0, 'positive_pairs': 0, 'negative_pairs': 0,
                'optional_queries': 0, 'optional_updates': 0, 'forced_init': 0}
    try:
        for phase, step, entries in plan:
            if phase == 'scene_reset' and step == 1:
                streams.clear()
                model.reset_memory()
            state.clear()
            state.update(phase=phase, step=step, entries=entries, packets=[], geometry=[], alignment=[], gate_logits=[])
            progress = (step - 1) / max(1, len(batches) - 1)
            learning_rate = base.training.learning_rate(progress, settings)
            for group in optimizer.param_groups:
                group['lr'] = learning_rate * group['initial_lr'] / settings['optimizer']['peak_lr']
            state['outer_rng_before'] = capture(runtime.inherited.rng_state())
            kwargs = {'optimize': True, 'diagnose': step in (2, 10)}
            if 'capture_evidence' in inspect.signature(train.run_batch).parameters:
                kwargs['capture_evidence'] = False
            value = train.run_batch(root, base, 1, step, entries, dataset, streams, sender_rows,
                                    model, scheduler, parallel, optimizer, settings, context, **kwargs)
            modules.assert_frozen(model, frozen)
            state['parameters_after_step'] = capture(dict(named))
            state['optimizer_after_step'] = capture(optimizer.state_dict())
            state['optimizer_parameter_order'] = [[names[id(parameter)] for parameter in group['params']]
                                                  for group in optimizer.param_groups]
            state['buffers_after_step'] = capture(dict(model.named_buffers()))
            state['streams_after_step'] = capture(streams)
            state['outer_rng_after'] = capture(runtime.inherited.rng_state())
            state['records'] = capture(value['records'])
            state['loss_values'] = {key: item for key, item in value.items()
                                    if key not in ('snapshots', 'features', 'records')}
            result['steps'].append(dict(state))
            coverage['real_frames'] += len(entries)
            coverage['positive_pairs'] += value['positive_pairs']
            coverage['negative_pairs'] += value['negative_pairs']
            for key in ('optional_queries', 'optional_updates', 'forced_init'):
                coverage[key] += sum(record[key] for record in value['records'])
            write(output, dict(result, status='RUNNING', coverage=coverage))
            print(json.dumps({'phase': 'exact_equivalence', 'label': args.label, 'scenario': phase,
                              'step': step, 'total_loss': value['total_loss'],
                              'positive_pairs': coverage['positive_pairs'],
                              'optional_queries': coverage['optional_queries']}), flush=True)
        modules.assert_frozen(model, frozen, full=True)
        result['final'] = {'frozen_weights': modules.frozen_hash(model),
                           'normalization_file_sha256': file_hash(normalization),
                           'rng': capture(runtime.inherited.rng_state())}
        result['coverage'] = coverage
        if args.batches < 16 or not all(coverage[key] > 0 for key in ('positive_pairs', 'negative_pairs', 'optional_queries')):
            raise ValueError('Insufficient real-data coverage for speed-only equivalence')
        if result['initial']['frozen_weights'] != result['final']['frozen_weights']:
            raise ValueError('Frozen parameters changed during verification')
        if source_hashes != {path.name: file_hash(path) for path in sorted(source.glob('*.py'))}:
            raise ValueError('Source changed during verification')
        result['status'] = 'COMPLETE_REQUIRES_COMPARISON'
        write(output, result)
    finally:
        gate_hook.remove()
        base.prepare_batch, base.pack = original_prepare, original_pack
        base.training.save_batch_state = original_save
        model._parse_losses = original_parse
        torch.nn.utils.clip_grad_norm_, torch.Tensor.backward = original_clip, original_backward
        modules.geometry_features = original_geometry
        model.crossview_alignment.transform_pts = original_alignment
        remove()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--label')
    parser.add_argument('--batches', type=int, default=16)
    parser.add_argument('--reset-batches', type=int, default=2)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compare', nargs=2, type=Path)
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare, args.output)
    if not all((args.root, args.source_dir, args.label, args.output)):
        parser.error('Run requires --root, --source-dir, --label and --output')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
