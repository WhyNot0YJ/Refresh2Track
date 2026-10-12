"""Independent, read-only two-frame native receiver/downstream intervention.

Run in the inherited container with PYTHONDONTWRITEBYTECODE=1. No evaluator,
configuration writer, optimizer, checkpoint writer or native cache writer runs.
"""
import copy
import hashlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path('/mnt/d/CoopTrack_Audit')
sys.path.insert(0, str(ROOT / 'scripts'))
import main_runtime as runtime


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def protected_snapshot():
    paths = set((ROOT / 'scripts').glob('*.py'))
    paths.update((ROOT / 'configs/main_experiment_v2').glob('*'))
    paths.update((ROOT / 'configs/main_experiment').glob('*'))
    paths.update((ROOT / 'reports/v2').glob('*.md'))
    paths.update((ROOT / 'experiments/v2').rglob('*.json'))
    paths.update((ROOT / 'experiments/v2').rglob('*.csv'))
    paths.update((ROOT / 'experiments/v2').rglob('*.pth'))
    paths.update((ROOT / 'experiments/main/metrics').glob('*.csv'))
    for rel in ('checkpoints/cooptrack_r50_coop.pth',
                'query_cache/val/000902.pkl', 'query_cache/val/000903.pkl'):
        paths.add(ROOT / rel)
    native = ROOT / 'code/CoopTrack/projects/mmdet3d_plugin/cooptrack'
    paths.update(native.rglob('*.py'))
    paths = sorted(path for path in paths if path.is_file()
                   and path.name != 'V2_BUG_AUDIT.md')
    return {str(path.relative_to(ROOT)): file_sha(path) for path in paths}


def run():
    before = protected_snapshot()
    runtime.initialize(ROOT)
    torch, np, mmcv = runtime.torch, runtime.np, runtime.mmcv
    from mmcv.parallel import MMDataParallel
    from mmdet.datasets.builder import PIPELINES
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from mmdet.apis import set_random_seed
    from main_message import FIELDS, FEATURES, ReceiverCache, int8_roundtrip, pack, message_sizes
    from v2_message import TemporalCache

    def signature(value):
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
            return {'shape': list(array.shape), 'dtype': str(array.dtype),
                    'sha256': hashlib.sha256(array.tobytes()).hexdigest()}
        if hasattr(value, 'get_fields'):
            return signature(value.get_fields())
        if hasattr(value, 'tensor'):
            return signature(value.tensor)
        if isinstance(value, dict):
            return {str(key): signature(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [signature(item) for item in value]
        if isinstance(value, np.ndarray):
            return {'shape': list(value.shape), 'dtype': str(value.dtype),
                    'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(type(value))

    cfg = mmcv.Config.fromfile(str(ROOT / 'configs/main_experiment_v2/coop_val.py'))
    assert cfg.model.img_backbone.type == 'ResNet' and cfg.model.img_backbone.depth == 50
    assert not cfg.model.get('save_track_query', False)
    manifest = {row['token']: row for row in runtime.native_manifest(ROOT, 'val')}
    # Select by predicted geometry only. The initial scene prefix can be outside
    # the vehicle range; an UPDATE control must actually reach native fusion.
    previous, selection = None, []
    for info in runtime.load_infos(ROOT, 'coop', 'val'):
        token = str(info['token_inf'])
        fields = runtime.fields(ROOT / 'query_cache/val' / (token + '.pkl'))
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = info['VehLidar2InfLidar_rotation']
        transform[:3, 3] = info['VehLidar2InfLidar_translation']
        inverse = torch.tensor(np.linalg.inv(transform), dtype=torch.float32)
        points = fields['ref_pts'].clone()
        points *= points.new_tensor([102.4, 102.4, 8.0])
        points += points.new_tensor([0.0, -51.2, -5.0])
        points = torch.cat([points, torch.ones_like(points[:, :1])], -1) @ inverse.T
        mask = (points[:, 0].abs() <= 51.2) & (points[:, 1].abs() <= 51.2)
        retained = set(fields['obj_idxes'][mask].tolist())
        row = {'cooperative_token': str(info['token']), 'infrastructure_token': token,
               'scene': str(info['scene_token']), 'session': manifest[token]['session'],
               'retained_ids': sorted(retained)}
        selection.append(row)
        if previous is not None:
            previous_info, previous_fields, previous_row = previous
            common = retained & set(previous_row['retained_ids'])
            if common and (row['scene'], row['session']) == (previous_row['scene'], previous_row['session']):
                selected_infos = [previous_info, info]
                selected_tokens = [previous_row['infrastructure_token'], token]
                target_id = 869 if 869 in common else sorted(common)[0]
                raw = {previous_row['infrastructure_token']: previous_fields, token: fields}
                break
        previous = (info, fields, row)
    else:
        raise AssertionError('No real adjacent continuous query survives native geometry alignment')
    cooperative_tokens = [str(info['token']) for info in selected_infos]
    selected_hashes_before = {token: file_sha(ROOT / 'query_cache/val' / (token + '.pkl')) for token in selected_tokens}
    raw = {token: {key: val.detach().cpu().clone() for key, val in fields.items()
                   if torch.is_tensor(val)} for token, fields in raw.items()}
    continuous = set(raw[selected_tokens[0]]['obj_idxes'].tolist()) & set(raw[selected_tokens[1]]['obj_idxes'].tolist())
    control = {'branch': None, 'target_id': target_id, 'frame': None}
    branches = {}

    @PIPELINES.register_module(force=True)
    class AuditReceiver:
        def __init__(self, keys, load_file_root):
            assert tuple(keys) == FIELDS
            self.mirror, self.receiver = TemporalCache(), ReceiverCache()

        def __call__(self, results):
            token = str(results['sample_idx_inf'])
            assert token in raw
            metadata = manifest[token]
            assert str(results['scene_token']) == metadata['scene']
            current = {key: val.clone() for key, val in raw[token].items()}
            if token == selected_tokens[1] and control['branch'].startswith('perturb'):
                row = current['obj_idxes'].tolist().index(control['target_id'])
                for offset, key in enumerate(FEATURES):
                    # Distinct, bounded and unmistakable current sender values.
                    current[key][row] = torch.linspace(-7.0, 9.0, 256) + offset * 3.0
            context = self.mirror.prepare(int8_roundtrip(current), metadata)
            flags = context['force'].clone()
            if token == selected_tokens[1] and 'update' in control['branch']:
                flags[current['obj_idxes'].tolist().index(control['target_id'])] = True
            packet = pack(current, metadata, flags)
            decoded, header = self.receiver.unpack(packet, metadata)
            self.mirror.commit(current, flags, metadata, decoded)
            mandatory, cost, full = message_sizes(current, metadata)
            assert cost == 780 and len(packet) == mandatory + 780 * int(flags.sum())
            record = {'metadata': metadata, 'ids': current['obj_idxes'].tolist(),
                      'flags': flags.tolist(), 'force': context['force'].tolist(),
                      'packet_sha256': hashlib.sha256(packet).hexdigest(),
                      'packet_bytes': len(packet), 'mandatory_bytes': mandatory,
                      'decoded': signature({key: decoded[key] for key in FIELDS}),
                      'sender_current': signature({key: current[key] for key in FIELDS}),
                      'receiver_cache': signature(self.receiver.cache),
                      'hooks': {}}
            record['_decoded'] = {key: decoded[key].clone() for key in FIELDS}
            record['_current'] = {key: current[key].clone() for key in FIELDS}
            branches[control['branch']].append(record)
            control['frame'] = record
            results.update(decoded)
            return results

    def replace(pipeline):
        count = 0
        for transform in pipeline:
            if transform['type'] == 'LoadInfInformation':
                transform['type'] = 'AuditReceiver'
                count += 1
            if 'transforms' in transform:
                count += replace(transform['transforms'])
        return count

    assert replace(cfg.data.test.pipeline) == 1
    model = runtime.build_model(ROOT, cfg, 'coop').cuda()
    assert not model.training and not model.save_track_query
    assert not any(p.requires_grad for p in model.parameters())
    frozen_before = signature(model.state_dict())
    state_keys = ('scene_token', 'timestamp', 'prev_bev', 'test_track_instances',
                  'l2g_r_mat', 'l2g_t', 'prev_pos', 'prev_angle', 'runtime_tracker',
                  'test_flag')
    initial_state = {key: copy.deepcopy(getattr(model, key))
                     for key in state_keys if hasattr(model, key)}

    def alignment_before(module, args):
        data, calibration = args
        record = control['frame']
        record['hooks']['alignment_input'] = signature(data)
        record['hooks']['alignment_calibration'] = signature(calibration)
        inverse = data['ref_pts'].new_tensor(np.linalg.inv(calibration.cpu().numpy().T))
        _, mask = module.transform_pts(data['ref_pts'], inverse)
        record['alignment_retained_ids'] = [track for track, keep in zip(record['ids'], mask.tolist()) if keep]

    def alignment_after(module, args, output):
        control['frame']['hooks']['alignment_output'] = signature(output)

    def reasoner_before(module, args):
        control['frame']['hooks']['reasoner_input'] = signature(args)

    model.crossview_alignment.register_forward_pre_hook(alignment_before)
    model.crossview_alignment.register_forward_hook(alignment_after)
    model.STReasoner.register_forward_pre_hook(reasoner_before)
    original_init = model._init_inf_tracks
    def wrapped_init(*args, **kwargs):
        output = original_init(*args, **kwargs)
        control['frame']['hooks']['init_inf_output'] = signature(output)
        return output
    model._init_inf_tracks = wrapped_init
    original_aggregation = model.STReasoner.aggregation
    def wrapped_aggregation(*args, **kwargs):
        control['frame']['hooks']['association_input'] = signature(args)
        return original_aggregation(*args, **kwargs)
    model.STReasoner.aggregation = wrapped_aggregation
    original_fusion = model.STReasoner._query_fusion
    def wrapped_fusion(*args, **kwargs):
        control['frame']['hooks']['fusion_input'] = signature(args)
        return original_fusion(*args, **kwargs)
    model.STReasoner._query_fusion = wrapped_fusion
    parallel = MMDataParallel(model, device_ids=[0])

    def intervene_gt(value):
        if isinstance(value, runtime.DataContainer):
            intervene_gt(value.data)
        elif isinstance(value, (list, tuple)):
            for item in value:
                intervene_gt(item)
        elif torch.is_tensor(value):
            value.fill_(123 if not value.is_floating_point() else 12345.0)
        elif hasattr(value, 'tensor'):
            value.tensor.fill_(12345.0)

    for branch in ('baseline_reuse', 'perturb_reuse', 'baseline_update', 'perturb_update', 'gt_reuse'):
        control['branch'] = branch
        branches[branch] = []
        for key, value in initial_state.items():
            setattr(model, key, copy.deepcopy(value))
        dataset = runtime.dataset_for(cfg)
        dataset.data_infos = [info for info in dataset.data_infos
                              if str(info['token']) in cooperative_tokens]
        assert [str(info['token']) for info in dataset.data_infos] == cooperative_tokens
        model.CLASSES = dataset.CLASSES
        loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                                  dist=False, shuffle=False,
                                  nonshuffler_sampler=cfg.data.nonshuffler_sampler)
        iterator = iter(loader)
        set_random_seed(0, deterministic=False)
        torch.backends.cudnn.benchmark = False
        for index, info in enumerate(dataset.data_infos):
            data = next(iterator)
            runtime.assert_loaded_queries(data, control['frame']['_decoded'])
            if branch == 'gt_reuse':
                for key in ('gt_bboxes_3d', 'gt_labels_3d', 'gt_inds'):
                    assert key in data
                    intervene_gt(data[key])
            with torch.no_grad():
                output = parallel(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            assert len(output) == 1 and str(output[0]['token']) == str(info['token'])
            control['frame']['prediction'] = signature(output)
            assert set(control['frame']['hooks']) == {
                'alignment_input', 'alignment_calibration', 'alignment_output',
                'reasoner_input', 'init_inf_output', 'association_input', 'fusion_input'}
            print('AUDIT_NATIVE branch=%s frame=%d/2 packet_bytes=%d' %
                  (branch, index + 1, control['frame']['packet_bytes']), flush=True)
        if branch == 'baseline_reuse':
            retained = set(branches[branch][1]['alignment_retained_ids']) & continuous
            assert retained, 'No continuous query survives actual geometric alignment'
            if control['target_id'] not in retained:
                control['target_id'] = sorted(retained)[0]
            assert control['target_id'] in retained

    baseline, perturbed = branches['baseline_reuse'][1], branches['perturb_reuse'][1]
    target_row = baseline['ids'].index(control['target_id'])
    first_row = branches['baseline_reuse'][0]['ids'].index(control['target_id'])
    checks = {}
    def check(name, condition, details):
        checks[name] = {'status': 'PASS' if condition else 'FAIL', 'details': details}
    check('reuse_flags_fixed', baseline['flags'] == perturbed['flags'] and not baseline['flags'][target_row],
          {'target_id': control['target_id'], 'frame1_flags': baseline['flags']})
    check('perturbation_really_changes_all_current_latents', all(
          baseline['sender_current'][key] != perturbed['sender_current'][key] for key in FEATURES), list(FEATURES))
    check('reuse_packet_identical', baseline['packet_sha256'] == perturbed['packet_sha256'],
          baseline['packet_sha256'])
    check('reuse_receiver_is_last_received_all_three', all(torch.equal(
          branches['baseline_reuse'][0]['_decoded'][key][first_row],
          perturbed['_decoded'][key][target_row]) for key in FEATURES), list(FEATURES))
    check('reuse_all_native_downstream_inputs_identical', baseline['hooks'] == perturbed['hooks'],
          {key: baseline['hooks'][key] == perturbed['hooks'][key] for key in baseline['hooks']})
    check('reuse_native_prediction_identical', baseline['prediction'] == perturbed['prediction'],
          {'tokens': cooperative_tokens, 'rng_seed': 0})
    check('current_geometry_is_current_not_cached', all(torch.equal(
          perturbed['_decoded'][key], raw[selected_tokens[1]][key]) for key in FIELDS[3:]), list(FIELDS[3:]))
    update_before, update_after = branches['baseline_update'][1], branches['perturb_update'][1]
    quantization = {}
    for key in FEATURES:
        actual = update_after['_decoded'][key][target_row]
        sender = update_after['_current'][key][target_row]
        rounded = int8_roundtrip(update_after['_current'])[key][target_row]
        scale = float(sender.abs().max()) / 127.0
        error = float((actual - sender).abs().max())
        quantization[key] = {'max_abs_error': error, 'half_scale_bound': scale / 2.0,
                             'decoder_equals_roundtrip': bool(torch.equal(actual, rounded)),
                             'differs_from_baseline': not torch.equal(actual, update_before['_decoded'][key][target_row])}
    check('update_control_current_latents_quantization_correct', all(
          value['decoder_equals_roundtrip'] and value['differs_from_baseline'] and
          value['max_abs_error'] <= value['half_scale_bound'] + 2e-6
          for value in quantization.values()), quantization)
    check('update_control_reaches_real_alignment_init_association_fusion',
          update_before['packet_sha256'] != update_after['packet_sha256'] and all(
          update_before['hooks'][key] != update_after['hooks'][key]
          for key in ('alignment_input', 'alignment_output', 'init_inf_output', 'reasoner_input',
                      'association_input', 'fusion_input')),
          {key: update_before['hooks'][key] != update_after['hooks'][key]
           for key in update_before['hooks']})
    check('gt_perturbation_has_no_native_inference_effect', all(
          branches['baseline_reuse'][i]['prediction'] == branches['gt_reuse'][i]['prediction'] and
          branches['baseline_reuse'][i]['hooks'] == branches['gt_reuse'][i]['hooks']
          for i in range(2)), {'perturbed_keys': ['gt_bboxes_3d', 'gt_labels_3d', 'gt_inds'],
                              'replacement_float': 12345, 'replacement_integer': 123})
    check('native_weights_and_buffers_unchanged', frozen_before == signature(model.state_dict()),
          {'parameter_requires_grad': False, 'optimizer_constructed': False})
    after = protected_snapshot()
    check('protected_files_unchanged', before == after, {'file_count': len(before),
          'changed': [key for key in before if before[key] != after.get(key)]})
    selected_hashes_after = {token: file_sha(ROOT / 'query_cache/val' / (token + '.pkl')) for token in selected_tokens}
    check('selected_raw_cache_unchanged', selected_hashes_before == selected_hashes_after,
          {'before': selected_hashes_before, 'after': selected_hashes_after})
    for records in branches.values():
        for record in records:
            record.pop('_decoded')
            record.pop('_current')
    result = {'status': 'PASS' if all(value['status'] == 'PASS' for value in checks.values()) else 'FAIL',
              'scope': 'five controlled two-frame native forwards; no official evaluation or optimization',
              'source': 'actual frozen serializer, receiver and native R50 vehicle downstream',
              'target_id': control['target_id'], 'selected_cooperative_tokens': cooperative_tokens,
              'selected_infrastructure_tokens': selected_tokens, 'selection_by_current_geometry': selection,
              'checks': checks, 'branches': branches,
              'protected_sha256_before': before, 'protected_sha256_after': after}
    return result


if __name__ == '__main__':
    output = ROOT / 'audits/v2_bug_audit/downstream_evidence.json'
    try:
        result = run()
    except Exception:
        result = {'status': 'UNVERIFIED', 'traceback': traceback.format_exc()}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('AUDIT_DOWNSTREAM status=' + result['status'], flush=True)
    if result['status'] != 'PASS':
        print(json.dumps(result.get('checks', result), ensure_ascii=False), flush=True)
        sys.exit(1)
