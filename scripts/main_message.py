"""APTU reliable ordered UPDATE/REUSE wire protocol and sender mirror."""
import json
import struct
import zlib
import numpy as np
import torch

FIELDS = ('query_feats', 'query_embeds', 'cache_motion_feats', 'ref_pts', 'pred_boxes')
FEATURES = FIELDS[:3]
MAGIC = b'APTU0001'
PREFIX = struct.Struct('<8sII')
MAX_BYTES = 64 * 1024 * 1024


def validate(current):
    n = current['query_feats'].shape[0]
    if not 0 <= n <= 16384:
        raise ValueError('Invalid query count')
    for key, width in zip(FIELDS, (256, 256, 256, 3, 10)):
        value = current[key]
        if value.dtype != torch.float32 or tuple(value.shape) != (n, width) or not torch.isfinite(value).all():
            raise ValueError('Invalid native CPU FP32 communication field: ' + key)
    ids = current['obj_idxes']
    if ids.shape != (n,) or ids.dtype != torch.int64 or (ids < 0).any() or len(ids.unique()) != n:
        raise ValueError('Predicted track IDs must be unique nonnegative int64')
    scores = current.get('scores')
    if not torch.is_tensor(scores) or scores.dtype != torch.float32 or scores.shape != (n,) or not torch.isfinite(scores).all():
        raise ValueError('Sender confidence must be finite native FP32 scores')
    return n


def quantized(value):
    array = value.detach().cpu().contiguous().numpy()
    scale = np.max(np.abs(array), axis=1, keepdims=True) / 127 if len(array) else np.empty((0, 1), dtype=np.float32)
    scale = np.where(scale == 0, 1, scale).astype('<f4')
    encoded = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
    return encoded, scale


def int8_roundtrip(current):
    output = dict(current)
    for key in FEATURES:
        values, scale = quantized(current[key])
        output[key] = torch.from_numpy(values.astype(np.float32) * scale).to(current[key].device)
    return output


def pack(current, metadata, updates):
    n = validate(current)
    flags = torch.as_tensor(updates, dtype=torch.bool).detach().cpu()
    if flags.shape != (n,):
        raise ValueError('UPDATE flags differ from query count')
    required = {'scene', 'session', 'token', 'timestamp_us', 'frame_idx'}
    if not required <= set(metadata) or not isinstance(metadata['session'], int) or not isinstance(metadata['frame_idx'], int):
        raise ValueError('Native scene/reset-session/frame identity is required')
    header = {'version': 1, 'frame': metadata, 'queries': n, 'byte_order': 'little',
              'identity': {'name': 'predicted_track_id', 'dtype': '<i8', 'shape': [n]},
              'update_flags': {'dtype': '|u1', 'shape': [n]},
              'fields': [{'name': key, 'shape': list(current[key].shape), 'dtype': '|i1' if key in FEATURES else '<f4',
                          'encoding': 'int8_per_query_symmetric' if key in FEATURES else 'fp32',
                          'updated_rows_only': key in FEATURES,
                          'scale_dtype': '<f4' if key in FEATURES else None} for key in FIELDS]}
    encoded_header = json.dumps(header, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    body = bytearray(current['obj_idxes'].detach().cpu().numpy().astype('<i8').tobytes())
    body.extend(flags.numpy().astype(np.uint8).tobytes())
    for key in FIELDS[3:]:
        body.extend(current[key].detach().cpu().numpy().astype('<f4').tobytes())
    for key in FEATURES:
        values, scale = quantized(current[key][flags.to(current[key].device)])
        body.extend(scale.tobytes())
        body.extend(values.tobytes())
    content = encoded_header + bytes(body)
    if len(encoded_header) > 1048576 or PREFIX.size + len(content) > MAX_BYTES:
        raise ValueError('Communication message exceeds protocol limits')
    return PREFIX.pack(MAGIC, len(encoded_header), zlib.crc32(content)) + content


def message_sizes(current, metadata):
    n = validate(current)
    mandatory = len(pack(current, metadata, torch.zeros(n, dtype=torch.bool)))
    full = len(pack(current, metadata, torch.ones(n, dtype=torch.bool)))
    per_update = sum(current[key].shape[1] + 4 for key in FEATURES)
    if full != mandatory + n * per_update:
        raise ValueError('Serialized UPDATE cost is not additive')
    return mandatory, per_update, full


def baseline_int8_pack(current, metadata):
    """Exact inherited Stage3 FEAT_INT8 serializer, for train budget statistics."""
    n = validate(current)
    header = {'version': 1, 'condition': 'FEAT_INT8', 'frame': metadata,
              'original_queries': n, 'queries': n, 'selected_rows': list(range(n)),
              'byte_order': 'little', 'fields': []}
    body = bytearray()
    for key in FIELDS:
        array = current[key].detach().cpu().numpy().astype('<f4', copy=False)
        entry = {'name': key, 'shape': list(array.shape), 'original_dtype': 'float32', 'encoding': 'fp32'}
        if key in FEATURES:
            array, scale = quantized(current[key])
            entry.update(encoding='int8_per_query_symmetric', scale_offset=len(body), scale_bytes=scale.nbytes)
            body.extend(scale.tobytes())
        entry.update(dtype=array.dtype.str, offset=len(body), bytes=array.nbytes)
        body.extend(array.tobytes())
        header['fields'].append(entry)
    encoded = json.dumps(header, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    content = encoded + bytes(body)
    return PREFIX.pack(b'CTKMSG01', len(encoded), zlib.crc32(content)) + content


class ReceiverCache:
    """Receiver reconstructs all five model inputs from packet + received history."""
    def __init__(self):
        self.identity = None
        self.previous = None
        self.cache = {}

    def unpack(self, payload, expected_frame=None):
        if not PREFIX.size <= len(payload) <= MAX_BYTES:
            raise ValueError('Truncated or oversized message')
        magic, header_len, crc = PREFIX.unpack_from(payload)
        content = payload[PREFIX.size:]
        if magic != MAGIC or not 0 < header_len <= min(len(content), 1048576) or zlib.crc32(content) != crc:
            raise ValueError('Invalid message prefix or CRC')
        header = json.loads(content[:header_len].decode('utf-8'))
        if header['version'] != 1 or header['byte_order'] != 'little':
            raise ValueError('Unsupported communication protocol')
        metadata = header['frame']
        if not {'scene', 'session', 'token', 'timestamp_us', 'frame_idx'} <= set(metadata) or not isinstance(metadata['session'], int) or not isinstance(metadata['frame_idx'], int):
            raise ValueError('Invalid received native frame identity')
        if expected_frame is not None and metadata != expected_frame:
            raise ValueError('Receiver frame identity mismatch')
        identity = (metadata['scene'], metadata['session'])
        cache = {} if identity != self.identity else self.cache.copy()
        if identity == self.identity and self.previous is not None and metadata['frame_idx'] <= self.previous['frame_idx']:
            raise ValueError('Reliable ordered protocol received duplicate/reversed sender frame')
        n = header['queries']
        if not isinstance(n, int) or not 0 <= n <= 16384:
            raise ValueError('Invalid received query count')
        if header['identity'] != {'name': 'predicted_track_id', 'dtype': '<i8', 'shape': [n]} or header['update_flags'] != {'dtype': '|u1', 'shape': [n]}:
            raise ValueError('Invalid identity/flag shape or dtype')
        if [row['name'] for row in header['fields']] != list(FIELDS):
            raise ValueError('Native field order differs')
        body, cursor = content[header_len:], 0

        def take(dtype, shape):
            nonlocal cursor
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if cursor + size > len(body):
                raise ValueError('Truncated tensor body')
            output = np.frombuffer(body, dtype=dtype, count=int(np.prod(shape)), offset=cursor).copy().reshape(shape)
            cursor += size
            return output

        ids = take('<i8', (n,))
        flags = take('|u1', (n,))
        if (ids < 0).any() or len(set(ids.tolist())) != n or (flags > 1).any():
            raise ValueError('Invalid predicted identity or UPDATE flag')
        indices = np.flatnonzero(flags)
        output = {}
        for key, width in zip(FIELDS, (256, 256, 256, 3, 10)):
            entry = header['fields'][FIELDS.index(key)]
            expected = {'name': key, 'shape': [n, width], 'dtype': '|i1' if key in FEATURES else '<f4',
                        'encoding': 'int8_per_query_symmetric' if key in FEATURES else 'fp32',
                        'updated_rows_only': key in FEATURES, 'scale_dtype': '<f4' if key in FEATURES else None}
            if entry != expected:
                raise ValueError('Invalid field shape/encoding metadata')
        for key, width in zip(FIELDS[3:], (3, 10)):
            output[key] = torch.from_numpy(take('<f4', (n, width)))
        updates = {}
        for key in FEATURES:
            scale = take('<f4', (len(indices), 1))
            if not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError('Invalid INT8 scale')
            updates[key] = torch.from_numpy(take('|i1', (len(indices), 256)).astype(np.float32) * scale)
        if cursor != len(body):
            raise ValueError('Unexpected tensor body tail')
        for position, index in enumerate(indices.tolist()):
            cache[int(ids[index])] = {key: updates[key][position].clone() for key in FEATURES}
        for key in FEATURES:
            if any(int(track_id) not in cache for track_id in ids.tolist()):
                raise ValueError('REUSE without a received latent in this native session')
            output[key] = torch.stack([cache[int(track_id)][key] for track_id in ids]) if n else torch.empty((0, 256), dtype=torch.float32)
        if not all(torch.isfinite(output[key]).all() for key in FIELDS):
            raise ValueError('Nonfinite decoded state')
        self.identity, self.previous, self.cache = identity, metadata.copy(), cache
        header['updates'] = len(indices)
        return output, header


def scheduler_features(current, last, elapsed, score_delta, ref_delta, track_age):
    latent = torch.cat([current[key] for key in FEATURES], dim=-1)
    previous = torch.cat([last[key] for key in FEATURES], dim=-1)
    summary = torch.cat([elapsed[:, None], current['scores'][:, None], score_delta[:, None],
                         current['ref_pts'], ref_delta, current['pred_boxes'], track_age[:, None]], dim=-1)
    return torch.cat([latent, previous, latent - previous, summary], dim=-1)


class TemporalCache:
    """Sender mirror advances only on actual UPDATE, keyed by native predicted ID."""
    def __init__(self):
        self.identity = None
        self.cache = {}
        self.observed = {}
        self.first_seen = {}

    def prepare(self, current, metadata):
        n = validate(current)
        identity = (metadata['scene'], metadata['session'])
        if identity != self.identity:
            self.cache, self.observed, self.first_seen = {}, {}, {}
            self.identity = identity
        device = current['query_feats'].device
        elapsed = torch.zeros(n, dtype=torch.float32, device=device)
        score_delta = torch.zeros_like(elapsed)
        track_age = torch.zeros_like(elapsed)
        ref_delta = torch.zeros((n, 3), dtype=torch.float32, device=device)
        force = torch.zeros(n, dtype=torch.bool, device=device)
        last = {key: torch.zeros_like(current[key]) for key in FEATURES}
        frame = metadata['frame_idx']
        for row, track_id in enumerate(current['obj_idxes'].tolist()):
            self.first_seen.setdefault(track_id, frame)
            track_age[row] = frame - self.first_seen[track_id]
            saved = self.cache.get(track_id)
            if saved is None:
                force[row] = True
            else:
                elapsed[row] = frame - saved['frame_idx']
                force[row] = elapsed[row] >= 4
                for key in FEATURES:
                    last[key][row] = saved[key].to(device)
            observed = self.observed.get(track_id)
            if observed is not None:
                score_delta[row] = current['scores'][row] - observed['score']
                ref_delta[row] = current['ref_pts'][row] - observed['ref'].to(device)
        context = dict(current=current, last=last, elapsed=elapsed, score_delta=score_delta,
                       ref_delta=ref_delta, track_age=track_age, force=force)
        context['tokens'] = scheduler_features(current, last, elapsed, score_delta, ref_delta, track_age)
        return context

    def commit(self, current, flags, metadata):
        roundtrip = int8_roundtrip(current)
        for row, track_id in enumerate(current['obj_idxes'].tolist()):
            if bool(flags[row]):
                self.cache[track_id] = {key: roundtrip[key][row].detach().cpu().clone() for key in FEATURES}
                self.cache[track_id]['frame_idx'] = metadata['frame_idx']
            self.observed[track_id] = {'score': float(current['scores'][row]), 'ref': current['ref_pts'][row].detach().cpu().clone()}


def select_updates(scores, force, mandatory, per_update, budget):
    """Forced new/stale updates may exceed budget; caller records actual overflow."""
    if scores.shape != force.shape or scores.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError('Invalid scheduler score or forced-update shape')
    flags = force.detach().clone()
    slots = max(0, int((float(budget) - mandatory) // per_update) - int(force.sum()))
    if slots:
        order = np.lexsort((np.arange(len(scores)), -scores.detach().cpu().numpy()))
        chosen = [int(index) for index in order if not bool(force[index])][:slots]
        flags[torch.as_tensor(chosen, dtype=torch.long, device=flags.device)] = True
    return flags


def self_check():
    current = {key: torch.arange(3 * width, dtype=torch.float32).reshape(3, width) / 100
               for key, width in zip(FIELDS, (256, 256, 256, 3, 10))}
    current.update(obj_idxes=torch.tensor([13, 2, 81]), scores=torch.tensor([.8, .7, .6]))
    metadata = {'scene': 'formal_contract', 'session': 0, 'frame_idx': 0, 'token': 'a', 'timestamp_us': 0.0}
    mirror, receiver = TemporalCache(), ReceiverCache()
    context = mirror.prepare(current, metadata)
    assert context['tokens'].shape == (3, 2324) and context['force'].all()
    flags = context['force']
    decoded, _ = receiver.unpack(pack(current, metadata, flags), metadata)
    mirror.commit(current, flags, metadata)
    assert all(torch.equal(decoded[key], int8_roundtrip(current)[key]) for key in FEATURES)
    current = {key: value[torch.tensor([2, 0, 1])] if value.ndim else value for key, value in current.items()}
    current['query_feats'] = current['query_feats'] + 8
    metadata.update(frame_idx=1, token='b', timestamp_us=100000.0)
    context = mirror.prepare(current, metadata)
    flags = torch.zeros(3, dtype=torch.bool)
    mandatory, cost, full = message_sizes(current, metadata)
    packet = pack(current, metadata, flags)
    decoded, _ = receiver.unpack(packet, metadata)
    assert len(packet) == mandatory and full == mandatory + 3 * cost
    assert torch.equal(decoded['query_feats'], context['last']['query_feats'])
    mirror.commit(current, flags, metadata)
    metadata.update(frame_idx=4, token='e', timestamp_us=400000.0)
    assert mirror.prepare(current, metadata)['force'].all()
    metadata.update(session=1, frame_idx=0, token='new')
    assert mirror.prepare(current, metadata)['force'].all()
    try:
        receiver.unpack(pack(current, metadata, flags), metadata)
    except ValueError:
        pass
    else:
        raise AssertionError('A native reset must reject old-session REUSE')
    print('APTU temporal wire contract PASS')


if __name__ == '__main__':
    self_check()
