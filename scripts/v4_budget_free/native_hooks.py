"""Instance-only V3 observations and scene RNG; native source files stay unchanged."""
import ast
import hashlib
import inspect
import random
import textwrap
from contextlib import contextmanager
from types import MethodType

import main_runtime as native
import v2_runtime as inherited


def scene_rng(seed, scene, epoch):
    """Initialize once per scene rollout, without advancing the batch RNG."""
    saved = inherited.rng_state()
    value = int.from_bytes(hashlib.sha256(
        ('%d:%s:%d' % (seed, scene, epoch)).encode('utf-8')).digest()[:4], 'little')
    try:
        random.seed(value)
        native.np.random.seed(value)
        native.torch.manual_seed(value)
        native.torch.cuda.manual_seed_all(value)
        return inherited.rng_state()
    finally:
        inherited.restore_rng(saved)


@contextmanager
def stream_rng(stream):
    """A frame consumes its scene state; shared batched operations use outer RNG."""
    saved = inherited.rng_state()
    inherited.restore_rng(stream['rng'])
    try:
        yield
    finally:
        stream['rng'] = inherited.rng_state()
        inherited.restore_rng(saved)


def instrument(source):
    """Add two RNG contexts and one observer to the exact inherited method."""
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('Expected exactly one native training method')
    function = tree.body[0]
    if function.name != '_forward_single_frame_train_bs':
        raise ValueError('Unexpected native training method name')
    counts = {'prepare_loop': 0, 'task_loop': 0, 'loss_capture': 0}
    for statement in function.body:
        if not isinstance(statement, ast.For) or not isinstance(statement.target, ast.Name):
            continue
        lane = statement.target.id
        if lane not in ('i', 'j'):
            continue
        if ast.dump(statement.iter) != ast.dump(ast.parse('range(self.batch_size)', mode='eval').body):
            raise ValueError('Native lane loop changed')
        stage = 'prepare' if lane == 'i' else 'task'
        counts[stage + '_loop'] += 1
        if lane == 'j':
            observed = []
            for inner in statement.body:
                if (isinstance(inner, ast.Assign) and isinstance(inner.value, ast.Call)
                        and isinstance(inner.value.func, ast.Attribute)
                        and isinstance(inner.value.func.value, ast.Name)
                        and inner.value.func.value.id == 'self'
                        and inner.value.func.attr == 'frame_summarization'):
                    callback = ast.parse('self._v3_capture_loss(j, cur_loss)').body[0]
                    observed.append(ast.copy_location(callback, inner))
                    counts['loss_capture'] += 1
                observed.append(inner)
            statement.body = observed
        context = ast.parse('self._v3_lane_rng(%s, %r)' % (lane, stage), mode='eval').body
        wrapped = ast.With(items=[ast.withitem(context_expr=context, optional_vars=None)],
                           body=statement.body, type_comment=None)
        statement.body = [ast.copy_location(wrapped, statement)]
    if counts != {'prepare_loop': 1, 'task_loop': 1, 'loss_capture': 1}:
        raise ValueError('Native method layout changed: ' + str(counts))
    return ast.fix_missing_locations(tree), counts


def install(model, context):
    """Observe original per-frame losses, and isolate every native lane loop."""
    original = model._forward_single_frame_train_bs
    unwrapped = inspect.unwrap(original.__func__)
    source = inspect.getsource(unwrapped)
    tree, counts = instrument(source)
    namespace = dict(unwrapped.__globals__)
    exec(compile(tree, '<v3-native-lane-instrumentation>', 'exec'), namespace)
    replacement = namespace['_forward_single_frame_train_bs']
    proof = {'native_method_source_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
             'instrumentation_ast_sha256': hashlib.sha256(ast.dump(tree).encode('utf-8')).hexdigest(),
             'insertions': counts,
             'rng': 'persistent per-scene Python/NumPy/Torch CPU/CUDA state for preprocessing and native lane loops; separate outer RNG for batched BEV/detection operations',
             'native_source_files_modified': False}

    @contextmanager
    def lane_rng(self, lane, stage):
        sample = context['samples'][lane]
        stream = context['streams'][sample['scene']]
        with stream_rng(stream):
            yield

    def capture_loss(self, lane, values):
        sample = context['samples'][lane]
        components = {}
        for name, value in values.items():
            if 'loss' in name:
                if not native.torch.is_tensor(value) or not native.torch.isfinite(value).all():
                    raise FloatingPointError('Nonfinite native per-frame task loss: ' + name)
                components[name] = float(value.detach().mean())
        if not components or 'task_loss' in sample:
            raise ValueError('Native task loss capture missing or duplicated')
        sample['task_components'] = components
        sample['task_loss'] = sum(components.values())
        sample['native_loss_captured'] = True

    model._v3_lane_rng = MethodType(lane_rng, model)
    model._v3_capture_loss = MethodType(capture_loss, model)
    model._forward_single_frame_train_bs = MethodType(replacement, model)
    handles = []

    def finite_loss(module, incoming, output):
        def check(value):
            if native.torch.is_tensor(value) and not native.torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite native loss before sanitizer: ' + type(module).__name__)
            if isinstance(value, dict):
                for item in value.values():
                    check(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    check(item)
        check(output)

    for name in ('loss_cls', 'loss_bboxes', 'loss_traj'):
        module = getattr(model.criterion, name, None)
        if isinstance(module, native.torch.nn.Module):
            handles.append(module.register_forward_hook(finite_loss))

    def remove():
        for handle in handles:
            handle.remove()
        model._forward_single_frame_train_bs = original
        del model._v3_lane_rng
        del model._v3_capture_loss
    return remove, proof


def self_check():
    """CPU-only: instrumentation keeps arithmetic and captures before summarizing."""
    sample = '''
def _forward_single_frame_train_bs(self):
    result = []
    for i in range(self.batch_size):
        result.append(i * 2)
    for j in range(self.batch_size):
        cur_loss = {'task_loss': result[j] + 1}
        cur_track_instances = self.frame_summarization(result[j], tracking=False)
    return result
'''
    tree, counts = instrument(sample)
    namespace = {}
    exec(compile(tree, '<v3-instrumentation-check>', 'exec'), namespace)
    seen, events = [], []

    class Fake:
        batch_size = 3

        @contextmanager
        def _v3_lane_rng(self, lane, stage):
            events.append((stage, lane, 'enter'))
            yield
            events.append((stage, lane, 'exit'))

        def _v3_capture_loss(self, lane, loss):
            seen.append((lane, loss['task_loss']))

        def frame_summarization(self, value, tracking=False):
            assert len(seen) == value // 2 + 1
            return value
    assert namespace['_forward_single_frame_train_bs'](Fake()) == [0, 2, 4]
    assert seen == [(0, 1), (1, 3), (2, 5)]
    assert len(events) == 12 and counts['loss_capture'] == 1
    return {'status': 'PASS', 'scope': 'CPU AST preservation, exact callback timing, separate lane contexts'}
