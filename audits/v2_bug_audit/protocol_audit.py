"""Read-only CPU protocol replay; writes only protocol_evidence.json beside this file."""
import os
import sys
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def audit(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root / 'scripts'))
    import main_runtime as rt
    rt.initialize(root)
    import numpy as np
    import torch
    from v2_message import TemporalCache, ReceiverCache, periodic_updates
    from main_message import FIELDS, FEATURES, PREFIX, baseline_int8_pack, int8_roundtrip, pack, message_sizes

    torch.manual_seed(41)
    results = []

    def check(name, source, inputs, expected, actual, ok):
        results.append(dict(item=name, source=source, input=inputs,
                            expected=expected, actual=actual, status='PASS' if ok else 'FAIL'))

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def same(a, b):
        return np.isclose(a, b, rtol=1e-12, atol=1e-12)

    logs = {p.stem: json.loads(p.read_text()) for p in
            (root / 'experiments/v2/communication/official_val').glob('*.json')}
    p32 = logs['P32_seed0']['frames']
    init = logs['INIT_ONLY_seed0']['frames']
    meta_keys = ('token', 'scene', 'session', 'frame_idx', 'native_frame_idx', 'timestamp_us')
    segment = p32[:70]
    sample_paths = [root / 'query_cache/val' / (row['token'] + '.pkl') for row in segment]
    reset_positions = []
    for same_scene in (False, True):
        reset_positions.append(next(i for i, row in enumerate(p32) if i and
                                    (row['scene'] == p32[i-1]['scene']) == same_scene and
                                    row['session'] != p32[i-1]['session']))
    reset_paths = [root / 'query_cache/val' / (p32[i]['token'] + '.pkl')
                   for pos in reset_positions for i in (pos-1, pos)]
    protected = list(sample_paths) + list((root / 'scripts').glob('v2_*.py')) + [
        root / 'scripts/main_message.py', root / 'scripts/main_runtime.py',
        root / 'configs/main_experiment_v2/training_v2.json',
        root / 'configs/main_experiment_v2/train_manifest.json',
        root / 'experiments/v2/metrics/FINAL_PROTOCOL_MANIFEST.json']
    protected += list((root / 'experiments/v2/communication/official_val').glob('*.json'))
    protected += list((root / 'experiments/v2/metrics/official_val').glob('*/result.json'))
    protected += reset_paths
    before = {str(p.relative_to(root)): digest(p) for p in protected}
    currents = [{k: v[k].detach().cpu().clone() for k in FIELDS + ('obj_idxes', 'scores')}
                for v in (rt.fields(p) for p in sample_paths)]
    metas = [{k: row[k] for k in meta_keys} for row in segment]

    trace = {}
    for policy in ('P32', 'INIT_ONLY'):
        sender, receiver = TemporalCache(), ReceiverCache()
        expected_rows = p32 if policy == 'P32' else init
        updates_869, bytes_seen, exact = [], [], True
        geometry_exact, reuse_exact, update_exact = True, True, True
        for i, (raw, meta) in enumerate(zip(currents, metas)):
            current = int8_roundtrip(raw)
            ctx = sender.prepare(current, meta)
            flags = periodic_updates(ctx, 32, init_only=(policy == 'INIT_ONLY'))
            mandatory, cost, full = message_sizes(raw, meta)
            packet = pack(raw, meta, flags)
            decoded, header = receiver.unpack(packet, meta)
            for key in FEATURES:
                reuse_exact &= torch.equal(decoded[key][~flags], ctx['last'][key][~flags])
                update_exact &= torch.equal(decoded[key][flags], current[key][flags])
            geometry_exact &= all(torch.equal(decoded[key], raw[key]) for key in FIELDS[3:])
            actual = {'bytes': len(packet), 'mandatory_bytes': mandatory, 'full_temporal_bytes': full,
                      'queries': len(flags), 'updates': int(flags.sum()),
                      'new_init_count': int(ctx['force'].sum()),
                      'refresh_count': int((flags & ~ctx['force']).sum())}
            exact &= cost == 780 and header['updates'] == actual['updates']
            exact &= all(actual[k] == expected_rows[i][k] for k in actual)
            row_869 = raw['obj_idxes'].tolist().index(869)
            if flags[row_869]:
                updates_869.append(meta['frame_idx'])
            sender.commit(raw, flags, meta, decoded)
            bytes_seen.append(actual['bytes'])
        expected = [0, 32, 64] if policy == 'P32' else [0]
        trace[policy] = {'persistent_id': 869, 'update_frames': updates_869,
                         'rows_exact_vs_existing_log': exact, 'bytes': bytes_seen,
                         'geometry_current': geometry_exact, 'all_three_reuse_old': reuse_exact,
                         'all_three_updates_decoded_int8': update_exact}
        check(policy + '_real_70_frame_replay',
              ['scripts/v2_message.py:10', 'scripts/v2_message.py:57',
               'scripts/main_message.py:117', 'scripts/v2_eval.py:173'],
              {'scene': '0003', 'session': 0, 'frames': [0, 69], 'tokens': [metas[0]['token'], metas[-1]['token']],
               'cache_shape_frame0': {k: list(currents[0][k].shape) for k in FIELDS}},
              {'id869_update_frames': expected, 'all_existing_log_counts_and_bytes_equal': True,
               'REUSE_latents': 'last actual decoded UPDATE', 'geometry': 'current frame FP32'},
              trace[policy], updates_869 == expected and exact and reuse_exact and update_exact and geometry_exact)

    # Fixed flags: change only REUSE rows in independent copies, preserving new-ID initialization.
    receiver = ReceiverCache()
    first = currents[0]
    receiver.unpack(pack(first, metas[0], torch.ones(len(first['scores']), dtype=torch.bool)), metas[0])
    raw = currents[1]
    flags = torch.tensor([track not in receiver.cache for track in raw['obj_idxes'].tolist()])
    altered = {k: v.clone() for k, v in raw.items()}
    for key in FEATURES:
        altered[key][~flags] = altered[key][~flags] * -3 + 123
    packet, changed_packet = pack(raw, metas[1], flags), pack(altered, metas[1], flags)
    baseline, changed = copy.deepcopy(receiver), copy.deepcopy(receiver)
    decoded_a, _ = baseline.unpack(packet, metas[1])
    decoded_b, _ = changed.unpack(changed_packet, metas[1])
    i869 = raw['obj_idxes'].tolist().index(869)
    all_unchanged = packet == changed_packet and all(torch.equal(decoded_a[k], decoded_b[k]) for k in FIELDS)
    control_flags = flags.clone()
    control_flags[i869] = True
    update_receiver = copy.deepcopy(receiver)
    update_packet = pack(altered, metas[1], control_flags)
    update_decoded, _ = update_receiver.unpack(update_packet, metas[1])
    rounded = int8_roundtrip(altered)
    quant_error = {k: float((update_decoded[k][i869] - altered[k][i869]).abs().max()) for k in FEATURES}
    quant_bound = {k: float(altered[k][i869].abs().max()) / 127 / 2 + 1e-5 for k in FEATURES}
    control_ok = len(update_packet) == len(packet) + 780 and all(
        torch.equal(update_decoded[k][i869], rounded[k][i869]) and
        not torch.equal(update_decoded[k][i869], decoded_a[k][i869]) and
        quant_error[k] <= quant_bound[k] for k in FEATURES)
    check('fixed_flags_REUSE_mutation_and_UPDATE_control',
          ['scripts/main_message.py:48', 'scripts/main_message.py:68', 'scripts/main_message.py:177'],
          {'token': metas[1]['token'], 'known_id': 869, 'new_id': 946,
           'mutation': 'all three latent fields on REUSE rows = original * -3 + 123',
           'flags': flags.tolist()},
          'REUSE packet/decoded fields identical; enabling one UPDATE adds exactly 780 B and reflects INT8-mutated latents',
          {'reuse_packet_equal': packet == changed_packet, 'all_five_decoded_fields_equal': all_unchanged,
           'packet_bytes': len(packet), 'control_packet_bytes': len(update_packet),
           'max_absolute_quantization_errors': quant_error, 'half_scale_error_bounds': quant_bound,
           'update_control': control_ok},
          all_unchanged and control_ok)
    _, header_len, _ = PREFIX.unpack_from(packet)
    parsed_header = json.loads(packet[PREFIX.size:PREFIX.size + header_len])
    augmented = dict(altered, matched_gt_idxes=torch.full((len(flags),), 987654321),
                     fut_xyz=torch.full((len(flags), 8, 3), 99999.),
                     cache_query_feats=torch.full_like(altered['query_feats'], 77777.))
    independent_formula = PREFIX.size + header_len + 61 * len(flags) + 780 * int(flags.sum())
    body_ok = len(packet) == independent_formula and pack(augmented, metas[1], flags) == changed_packet
    check('independent_wire_layout_and_extra_fields_ignored',
          ['scripts/main_message.py:53', 'scripts/main_message.py:64', 'scripts/main_message.py:66',
           'scripts/main_message.py:68', 'scripts/main_message.py:175'],
          {'queries': len(flags), 'updates': int(flags.sum()), 'extra_fields_mutated': ['matched_gt_idxes', 'fut_xyz', 'cache_query_feats']},
          'length = prefix16 + actual JSON header + 61*N + 3*(256+4)*U; extra cache/GT/future fields not transmitted',
          {'packet_bytes': len(packet), 'independent_length': independent_formula,
           'JSON_header_bytes': header_len, 'per_query_mandatory_body_bytes': 61,
           'latent_bundle_bytes': 780, 'fields': [x['name'] for x in parsed_header['fields']],
           'augmented_packet_unchanged': pack(augmented, metas[1], flags) == changed_packet}, body_ok)

    # Compare old-format helper and new full-UPDATE decoding for the same real input,
    # without claiming that historical inference inputs/RNG were identical.
    legacy = baseline_int8_pack(first, metas[0])
    old_magic, old_header_length, _ = PREFIX.unpack_from(legacy)
    old_header = json.loads(legacy[PREFIX.size:PREFIX.size+old_header_length])
    old_body = legacy[PREFIX.size+old_header_length:]
    old_decoded = {}
    for entry in old_header['fields']:
        array = np.frombuffer(old_body, dtype=entry['dtype'], count=int(np.prod(entry['shape'])),
                              offset=entry['offset']).copy().reshape(entry['shape'])
        if entry['encoding'] == 'int8_per_query_symmetric':
            scale = np.frombuffer(old_body, dtype='<f4', count=entry['shape'][0],
                                  offset=entry['scale_offset']).copy().reshape(-1, 1)
            array = array.astype(np.float32) * scale
        old_decoded[entry['name']] = torch.from_numpy(array)
    new_decoded, _ = ReceiverCache().unpack(pack(first, metas[0], torch.ones(len(first['scores']), dtype=torch.bool)), metas[0])
    equivalent = old_magic == b'CTKMSG01' and all(torch.equal(old_decoded[k], new_decoded[k]) for k in FIELDS)
    check('legacy_helper_vs_APTU_full_UPDATE_same_input_decoding',
          ['scripts/main_message.py:88', 'scripts/main_message.py:107', 'scripts/main_message.py:117'],
          {'real_token': metas[0]['token'], 'same_five_FP32_fields': True},
          'all five decoded fields exactly equal for identical input; wire length may differ due to ID/flags/header',
          {'decoded_equal': equivalent, 'legacy_bytes': len(legacy),
           'APTU_bytes': len(pack(first, metas[0], torch.ones(len(first['scores']), dtype=torch.bool))),
           'scope': 'current inherited-format helper, not a replay of historical model/RNG'}, equivalent)

    # Each stage owns tensors; mutate independent returned tensors to test alias boundaries.
    saved = {k: baseline.cache[869][k].clone() for k in FEATURES}
    for k in FIELDS:
        decoded_a[k].add_(333)
    alias_ok = all(torch.equal(baseline.cache[869][k], saved[k]) for k in FEATURES)
    sender = TemporalCache()
    ctx = sender.prepare(int8_roundtrip(first), metas[0])
    d0, _ = ReceiverCache().unpack(pack(first, metas[0], ctx['force']), metas[0])
    sender.commit(first, ctx['force'], metas[0], d0)
    snapshot = copy.deepcopy(sender.cache)
    for k in FEATURES:
        d0[k].add_(222)
    ctx = sender.prepare(int8_roundtrip(raw), metas[1])
    for k in FEATURES:
        ctx['last'][k].add_(111)
    alias_ok &= all(torch.equal(sender.cache[t][k], snapshot[t][k]) for t in snapshot for k in FEATURES)
    check('cache_reference_alias', ['scripts/main_message.py:150', 'scripts/main_message.py:178',
                                  'scripts/main_message.py:182', 'scripts/v2_message.py:35'],
          'mutate decoded output and prepare.last independent tensors', 'sender/receiver cached tensors unchanged',
          {'cached_tensors_unchanged': alias_ok}, alias_ok)

    def one(ids, value=1.):
        n = len(ids)
        x = {k: torch.full((n, w), value, dtype=torch.float32)
             for k, w in zip(FIELDS, (256, 256, 256, 3, 10))}
        return dict(x, obj_idxes=torch.tensor(ids, dtype=torch.int64), scores=torch.full((n,), .8))

    def meta(frame, scene='A', session=0):
        return dict(scene=scene, session=session, frame_idx=frame, token=str(frame), timestamp_us=frame * 100000.)

    def step(stream, x, m, init_only=True):
        a, b = stream
        ctx = a.prepare(int8_roundtrip(x), m)
        f = periodic_updates(ctx, 32, init_only=init_only)
        received, _ = b.unpack(pack(x, m, f), m)
        a.commit(x, f, m, received)
        return ctx, f, received

    identity_cases = {}
    for init_only in (True, False):
        stream = (TemporalCache(), ReceiverCache())
        step(stream, one([7, 8]), meta(0), init_only)
        step(stream, one([8]), meta(1), init_only)
        c, f, out = step(stream, one([7, 9], 5), meta(40), init_only)
        policy = 'INIT_ONLY' if init_only else 'P32'
        identity_cases[policy] = {'reappeared_id7': bool(f[0]), 'new_id9': bool(f[1]),
                                  'id7_elapsed': float(c['elapsed'][0])}
        expected_flags = [False, True] if init_only else [True, True]
        identity_cases[policy]['ok'] = f.tolist() == expected_flags and c['elapsed'].tolist() == [40., 0.]
    stream = (TemporalCache(), ReceiverCache())
    step(stream, one([7]), meta(0))
    c, f, _ = step(stream, one([7], 2), meta(0, session=1))
    session_ok = f.tolist() == [True] and c['elapsed'].tolist() == [0.]
    c, f, _ = step(stream, one([7], 3), meta(0, scene='B', session=1))
    scene_ok = f.tolist() == [True] and c['elapsed'].tolist() == [0.]
    lanes = [(TemporalCache(), ReceiverCache()) for _ in range(2)]
    step(lanes[0], one([7], 1), meta(0, scene='A'))
    step(lanes[1], one([7], 8), meta(0, scene='B'))
    _, _, a = step(lanes[0], one([7], 100), meta(1, scene='A'))
    _, _, b = step(lanes[1], one([7], 100), meta(1, scene='B'))
    lane_ok = all(torch.equal(a[k], torch.ones_like(a[k])) and torch.equal(b[k], torch.full_like(b[k], 8)) for k in FEATURES)
    rejected = 0
    for payload, expected in [(pack(one([7]), meta(0, scene='A'), [False]), meta(0, scene='A')),
                              (pack(one([7]), meta(2, scene='A'), [False]), meta(2, scene='WRONG')),
                              (pack(one([7]), meta(0, scene='NEW'), [False]), meta(0, scene='NEW'))]:
        try:
            copy.deepcopy(lanes[0][1]).unpack(payload, expected)
        except ValueError:
            rejected += 1
    check('scene_session_predicted_ID_and_batch_lane_identity',
          ['scripts/main_message.py:132', 'scripts/main_message.py:180', 'scripts/main_message.py:206',
           'scripts/v2_message.py:14', 'scripts/v2_train.py:263'],
          'same ID across two scenes, session reset, ID7 absent then reappears at frame40, new ID9, interleaved lanes',
          'new/reset cache UPDATE; absent ID retained; P32 elapsed40 UPDATE, INIT_ONLY REUSE; independent lanes; reject wrong/reversed/no-cache packets',
          {'disappearance_reappearance': identity_cases, 'scene_reset': scene_ok,
           'session_reset': session_ok, 'batch_lanes': lane_ok, 'invalid_cases_rejected': rejected},
          all(x['ok'] for x in identity_cases.values()) and scene_ok and session_ok and lane_ok and rejected == 3)
    real_resets = []
    for pos in reset_positions:
        stream = (TemporalCache(), ReceiverCache())
        old_row, row = p32[pos-1], p32[pos]
        prior = rt.fields(root / 'query_cache/val' / (old_row['token'] + '.pkl'))
        current = rt.fields(root / 'query_cache/val' / (row['token'] + '.pkl'))
        prior = {k: prior[k] for k in FIELDS + ('obj_idxes', 'scores')}
        current = {k: current[k] for k in FIELDS + ('obj_idxes', 'scores')}
        step(stream, prior, {k: old_row[k] for k in meta_keys})
        ctx, flags, received = step(stream, current, {k: row[k] for k in meta_keys})
        ok = bool(flags.all()) and not bool(ctx['elapsed'].any()) and all(
            torch.equal(received[k], int8_roundtrip(current)[k]) for k in FEATURES)
        ok &= len(pack(current, {k: row[k] for k in meta_keys}, flags)) == row['bytes']
        real_resets.append({'before_token': old_row['token'], 'after_token': row['token'],
                            'before_identity': [old_row['scene'], old_row['session']],
                            'after_identity': [row['scene'], row['session']],
                            'timestamp_gap_seconds': (row['timestamp_us']-old_row['timestamp_us'])/1e6,
                            'same_IDs_in_both_frames': sorted(set(prior['obj_idxes'].tolist()) & set(current['obj_idxes'].tolist())),
                            'all_init_after_reset': bool(flags.all()), 'frame_idx': row['frame_idx'],
                            'native_frame_idx': row['native_frame_idx'], 'PASS': bool(ok)})
    check('real_scene_and_intra_scene_session_reset',
          ['scripts/main_runtime.py:164', 'scripts/main_message.py:132', 'scripts/main_message.py:209',
           'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/detectors/cooptrack.py:1266'],
          'two adjacent-frame reset boundaries loaded from real val caches',
          'all current predicted IDs freshly initialize; no old latent inherited; actual logged bytes identical',
          real_resets, all(x['PASS'] for x in real_resets))

    full_log_evidence = []
    layouts = []
    all_logs_ok = True
    for label, log in sorted(logs.items()):
        frames, summary = log['frames'], log['summary']
        actual = json.loads((root / 'experiments/v2/metrics/official_val' / label / 'result.json').read_text())
        sizes = np.array([x['bytes'] for x in frames], dtype=np.int64)
        mandatory = sum(x['mandatory_bytes'] for x in frames)
        updates = sum(x['updates'] for x in frames)
        queries = sum(x['queries'] for x in frames)
        new_init = sum(x['new_init_count'] for x in frames)
        refresh = sum(x['refresh_count'] for x in frames)
        optional = sum(x['optional_queries'] for x in frames)
        total = int(sizes.sum())
        formula = all(x['bytes'] == x['mandatory_bytes'] + 780*x['updates'] and
                      x['full_temporal_bytes'] == x['mandatory_bytes'] + 780*x['queries'] and
                      x['updates'] == x['new_init_count'] + x['refresh_count'] and
                      x['queries'] == x['new_init_count'] + x['optional_queries'] for x in frames)
        reset_inits = all(x['new_init_count'] == x['queries'] and x['updates'] == x['queries']
                          and x['optional_queries'] == 0 for i, x in enumerate(frames)
                          if not i or (x['scene'], x['session']) != (frames[i-1]['scene'], frames[i-1]['session']))
        ok = (len(frames) == 3316 and queries == 44329 and new_init == 1557 and optional == 42772 and
              formula and reset_inits and total == mandatory + 780*updates and summary == actual['communication'] and
              same(sizes.mean(), summary['mean_bytes_per_frame']) and
              same(np.percentile(sizes, 95), summary['p95_bytes_per_frame']) and
              same(sizes.mean()*10, summary['bytes_per_second_at_10Hz']) and
              same(mandatory/len(frames), summary['mean_mandatory_bytes']) and
              same(updates*780/len(frames), summary['mean_latent_update_bytes']))
        if label == 'INIT_ONLY_seed0':
            ok &= refresh == 0 and updates == 1557
        all_logs_ok &= ok
        layouts.append([(x['token'], x['cooperative_token'], x['scene'], x['session'], x['frame_idx'],
                         x['native_frame_idx'], x['timestamp_us'], x['queries'], x['mandatory_bytes']) for x in frames])
        full_log_evidence.append({'run': label, 'frames': len(frames), 'queries': queries,
                                  'init': new_init, 'optional': optional, 'refresh': refresh,
                                  'total_bytes': total, 'mandatory_total_bytes': mandatory,
                                  'latent_update_total_bytes': updates*780, 'mean_bytes': float(sizes.mean()),
                                  'p95_bytes': float(np.percentile(sizes, 95)), 'all_real_resets_fully_initialized': reset_inits, 'PASS': bool(ok)})
    consistent_layout = len(logs) == 16 and all(x == layouts[0] for x in layouts)
    check('all_16_existing_communication_logs_and_results',
          ['scripts/main_message.py:78', 'scripts/v2_eval.py:207', 'scripts/v2_eval.py:295', 'scripts/v2_eval.py:308'],
          '16 complete existing application payload logs, no new model inference',
          '3316 frames/44329 queries/1557 init/42772 optional each; total=M+780U; exact mean/P95/nominal10Hz rate; same mandatory layout',
          {'all_logs': full_log_evidence, 'same_frame_query_mandatory_layout': consistent_layout},
          all_logs_ok and consistent_layout)

    manifest = rt.native_manifest(root, 'val')
    manifest_map = {x['token']: x for x in manifest}
    coop_infos = rt.load_infos(root, 'coop', 'val')
    coop_map = {str(x['token']): x for x in coop_infos}
    aligned = all(all(row[k] == manifest_map[row['token']][k] for k in meta_keys) and
                  row['token'] == str(coop_map[row['cooperative_token']]['token_inf']) and
                  row['scene'] == str(coop_map[row['cooperative_token']]['scene_token']) for row in p32)
    diffs = []
    duration, interval_bytes = 0., 0
    offset = {}
    native_index_ok = True
    for i, row in enumerate(p32):
        identity = (row['scene'], row['session'])
        off = row['native_frame_idx'] - row['frame_idx']
        native_index_ok &= off == offset.setdefault(identity, off)
        if i and identity == (p32[i-1]['scene'], p32[i-1]['session']):
            dt = (row['timestamp_us'] - p32[i-1]['timestamp_us'])/1e6
            diffs.append(dt)
            duration += dt
            interval_bytes += row['bytes']
    dt_stats = {'interval_count': len(diffs), 'mean_seconds': float(np.mean(diffs)),
                'median_seconds': float(np.median(diffs)), 'min_seconds': min(diffs),
                'max_seconds': max(diffs), 'exact_100ms_count': sum(x == .1 for x in diffs),
                'within_session_timestamp_seconds': duration,
                'P32_payload_bytes_after_each_session_first_packet': interval_bytes,
                'P32_payload_timestamp_rate_Bps_excluding_session_first_packets': interval_bytes/duration,
                'P32_nominal_10Hz_Bps': logs['P32_seed0']['summary']['bytes_per_second_at_10Hz']}
    check('elapsed_native_frame_identity_and_actual_timestamp_rate',
          ['scripts/main_runtime.py:157', 'scripts/v2_message.py:63', 'scripts/v2_eval.py:203', 'scripts/v2_eval.py:298'],
          'all existing val frame metadata versus native infra/coop infos; measured within-session timestamp differences',
          'scene/session/frame identities align; native frame offset constant within session; 10Hz rate explicitly nominal',
          {'metadata_aligned': aligned, 'native_elapsed_same_difference': native_index_ok,
           'scene_count': len({x['scene'] for x in manifest}), 'session_count': len(offset), 'timestamps': dt_stats},
          aligned and native_index_ok and len({x['scene'] for x in manifest}) == 21 and min(diffs) > 0)

    check('current_geometry_prediction_source_and_whitelisted_wire_fields',
          ['scripts/v2_eval.py:169', 'scripts/v2_eval.py:173', 'scripts/v2_eval.py:228',
           'scripts/main_message.py:66', 'scripts/main_message.py:167',
           'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/detectors/cooptrack.py:1179',
           'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/detectors/cooptrack.py:1230',
           'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/detectors/cooptrack.py:1327',
           'code/CoopTrack/projects/mmdet3d_plugin/datasets/spd_dataset.py:498'],
          'token-specific cached native inference predictions; current calibration from same cooperative info',
          'geometry always current FP32; predicted IDs are native runtime counters; GT/future cache fields never selected/serialized',
          {'real_70_geometry_exact': all(x['geometry_current'] for x in trace.values()),
           'wire_fields': list(FIELDS), 'sender_model_source': 'native _forward_single_frame_inference -> runtime tracker -> token.pkl',
           'source_review': 'native current detector output and frame_summarization; no GT/future-frame read in V2 receiver/pack'},
          all(x['geometry_current'] for x in trace.values()))

    # Existing-cache geometry only: no network forward, evaluator, GT, or checkpoint.
    from v2_runtime import make_config
    from projects.mmdet3d_plugin.cooptrack.modules.latent_transformation import LatentTransformation
    cfg = make_config(root, 'coop', 'val')
    geometry = LatentTransformation(pc_range=cfg.model.pc_range, inf_pc_range=cfg.model.inf_pc_range)
    coverage = {'frame_count': len(coop_infos), 'raw_query_observations': 0,
                'surviving_query_observations': 0, 'empty_frames': 0, 'nonempty_frames': 0,
                'pc_range': list(cfg.model.pc_range), 'inf_pc_range': list(cfg.model.inf_pc_range),
                'by_scene': {}, 'first_continuous_surviving_pairs': []}
    geometry_ok, geometry_cache_unchanged = True, True
    cache_hash_before, cache_hash_after = hashlib.sha256(), hashlib.sha256()
    previous = None
    for index, info in enumerate(coop_infos):
        token, scene = str(info['token_inf']), str(info['scene_token'])
        path = root / 'query_cache/val' / (token + '.pkl')
        initial_hash = digest(path)
        cache_hash_before.update((token + ':' + initial_hash + '\n').encode())
        raw = rt.fields(path)
        matrix = np.eye(4)
        matrix[:3, :3] = info['VehLidar2InfLidar_rotation']
        matrix[:3, 3] = info['VehLidar2InfLidar_translation']
        veh2inf = matrix.T.astype(np.float32)
        calib_inf2veh = raw['ref_pts'].new_tensor(np.linalg.inv(veh2inf.T))
        reference, mask = geometry.transform_pts(raw['ref_pts'], calib_inf2veh)
        ids = set(raw['obj_idxes'][mask].tolist())
        count, kept = len(mask), int(mask.sum())
        stats = coverage['by_scene'].setdefault(scene, dict(frames=0, raw_queries=0, retained_queries=0, empty_frames=0))
        stats['frames'] += 1
        stats['raw_queries'] += count
        stats['retained_queries'] += kept
        stats['empty_frames'] += kept == 0
        coverage['raw_query_observations'] += count
        coverage['surviving_query_observations'] += kept
        coverage['empty_frames'] += kept == 0
        coverage['nonempty_frames'] += kept > 0
        geometry_ok &= count == p32[index]['queries'] and len(reference) == kept and bool(torch.isfinite(reference).all())
        native_meta = manifest_map[token]
        if previous and previous['identity'] == (scene, native_meta['session']):
            common = sorted(ids & previous['ids'])
            if common and len(coverage['first_continuous_surviving_pairs']) < 6:
                coverage['first_continuous_surviving_pairs'].append({
                    'before_cooperative_token': previous['coop'], 'before_infrastructure_token': previous['inf'],
                    'after_cooperative_token': str(info['token']), 'after_infrastructure_token': token,
                    'scene': scene, 'session': native_meta['session'], 'predicted_ids': common,
                    'before_retained': len(previous['ids']), 'after_retained': kept})
        previous = {'identity': (scene, native_meta['session']), 'ids': ids,
                    'coop': str(info['token']), 'inf': token}
        final_hash = digest(path)
        cache_hash_after.update((token + ':' + final_hash + '\n').encode())
        geometry_cache_unchanged &= initial_hash == final_hash
        if index % 500 == 0:
            print('CPU_GEOMETRY_CACHE_AUDIT %d/%d' % (index + 1, len(coop_infos)), flush=True)
    coverage['empty_frame_fraction'] = coverage['empty_frames'] / len(coop_infos)
    coverage['surviving_query_fraction'] = coverage['surviving_query_observations'] / coverage['raw_query_observations']
    coverage['all_consumed_val_cache_files_SHA256_unchanged'] = geometry_cache_unchanged
    coverage['combined_cache_SHA256_before'] = cache_hash_before.hexdigest()
    coverage['combined_cache_SHA256_after'] = cache_hash_after.hexdigest()
    coverage['interpretation'] = ('Actual native predicted-geometry range filtering coverage, a design limitation; '
                                  'early empty frames do not imply all caches are unused. '
                                  'This does not prove a cause of any task-metric difference.')
    check('full_val_native_geometry_mask_coverage_CPU_only',
          ['code/CoopTrack/projects/mmdet3d_plugin/datasets/spd_dataset.py:498',
           'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/modules/latent_transformation.py:104',
           'code/CoopTrack/projects/mmdet3d_plugin/cooptrack/modules/latent_transformation.py:143'],
          '3316 existing predicted ref_pts caches + same-frame VehLidar2InfLidar calibration; no inference/evaluation/GT',
          'native mask shape/count/finite coordinates and query counts valid; all cache hashes unchanged; report actual coverage without causal inference',
          coverage, geometry_ok and geometry_cache_unchanged and coverage['raw_query_observations'] == 44329)

    after = {str(p.relative_to(root)): digest(p) for p in protected}
    check('read_only_hash_protection', ['audits/v2_bug_audit/protocol_audit.py'],
          {'protected_file_count': len(before), 'includes': '70 contiguous + 4 real reset-boundary caches, formal sources/configs/manifest, all16 raw results/logs'},
          'SHA256 before/after equal; no formal writes, bytecode disabled',
          {'all_equal': before == after, 'before': before, 'after': after}, before == after)
    evidence = {'status': 'PASS_SCOPED_PROTOCOL_AUDIT' if all(x['status'] == 'PASS' for x in results) else 'FAIL',
                'utc': datetime.now(timezone.utc).isoformat(), 'checks': results,
                'scope': 'CPU protocol replay and complete existing communication-log audit; downstream model controlled tests owned by root',
                'limitations': ['Only first 70 real caches fully protocol-replayed; 4 reset-boundary caches replayed; all 3316 caches read for geometry-only coverage; all16 full communication logs audited.',
                                'Real disappearance/reappearance not observed in selected70; deterministic disappearance and ID-switch cases tested.',
                                'Application payload bytes exclude network transport overhead; 10Hz B/s is nominal, actual timestamp-derived diagnostic explicitly separated.',
                                'No full-val or training performed; no whole-audit no-Bug conclusion made here.'],
                'representative_real_case': {'scene': '0003', 'session': 0, 'persistent_id': 869,
                   'sender_tokens': [metas[0]['token'], metas[1]['token']],
                   'cooperative_tokens': [segment[0]['cooperative_token'], segment[1]['cooperative_token']],
                   'cache_paths': [str(x) for x in sample_paths[:2]], 'P32_update_frames': [0, 32, 64]},
                'historical_protocol_note': {'P1_mean_bytes': 12161.645355850422,
                   'new_APTU_all_update_counterfactual_mean_from_existing_layout': float(np.mean([x['full_temporal_bytes'] for x in p32])),
                   'old_magic': 'CTKMSG01', 'new_magic': 'APTU0001',
                   'source': ['scripts/main_message.py:88', 'scripts/main_message.py:107', 'configs/main_experiment_v2/reused_val.json'],
                   'note': 'No historical P1/P2/P4 rerun or rebilling; inherited RNG placement differs.'}}
    output = root / 'audits/v2_bug_audit/protocol_evidence.json'
    output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': evidence['status'], 'checks': [(x['item'], x['status']) for x in results],
                      'dt': dt_stats, 'output': str(output)}, ensure_ascii=False), flush=True)
    return evidence


if __name__ == '__main__':
    result = audit(sys.argv[1] if len(sys.argv) > 1 else '/mnt/d/CoopTrack_Audit')
    if result['status'] != 'PASS_SCOPED_PROTOCOL_AUDIT':
        raise SystemExit(1)
