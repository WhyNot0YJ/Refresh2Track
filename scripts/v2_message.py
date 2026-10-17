"""V2 UPDATE/REUSE decisions over the inherited application wire protocol."""
import torch

from main_message import (FIELDS, FEATURES, ReceiverCache, TemporalCache as V1TemporalCache,
                          int8_roundtrip, message_sizes, pack, scheduler_features, validate)


class TemporalCache(V1TemporalCache):
    """Keep actual received latents until UPDATE or an explicit native reset."""
    def prepare(self, current, metadata):
        context = super().prepare(current, metadata)
        ids = current['obj_idxes'].tolist()
        saved = [self.cache.get(track_id) for track_id in ids]
        force = torch.tensor([item is None for item in saved], dtype=torch.bool,
                             device=current['query_feats'].device)
        if (context['elapsed'] < 0).any():
            raise ValueError('Native frame index precedes the last actual UPDATE')
        context.update(force=force, no_cache=force, has_cache=~force,
                       last_update_frame_idx_before=[None if item is None else item['frame_idx'] for item in saved],
                       last_update_timestamp_us_before=[None if item is None else item['timestamp_us'] for item in saved])
        context['tokens'] = context['tokens'].detach()
        return context

    def commit(self, current, flags, metadata, decoded):
        """Commit only after successful real unpack, using its detached values."""
        n = validate(current)
        flags = torch.as_tensor(flags, dtype=torch.bool).detach().cpu()
        if flags.shape != (n,) or self.identity != (metadata['scene'], metadata['session']):
            raise ValueError('Sender commit differs from the prepared native session')
        for key in FEATURES:
            if decoded[key].shape != current[key].shape or not torch.isfinite(decoded[key]).all():
                raise ValueError('Actual decoded latent shape/value differs: ' + key)
        for row, track_id in enumerate(current['obj_idxes'].tolist()):
            if bool(flags[row]):
                self.cache[track_id] = {key: decoded[key][row].detach().cpu().clone() for key in FEATURES}
                self.cache[track_id].update(frame_idx=metadata['frame_idx'], timestamp_us=metadata['timestamp_us'])
            elif track_id not in self.cache:
                raise ValueError('Sender REUSE without a received latent')
            elif any(not torch.equal(decoded[key][row].detach().cpu(), self.cache[track_id][key]) for key in FEATURES):
                raise ValueError('Sender mirror differs from real REUSE decode')
            self.observed[track_id] = {'score': float(current['scores'][row].detach()),
                                       'ref': current['ref_pts'][row].detach().cpu().clone()}


def select_updates(probability, force, valid_mask=None):
    """Deterministic threshold gate; optional UPDATEs are never budget filled."""
    if probability.ndim != 1 or probability.shape != force.shape or force.dtype != torch.bool:
        raise ValueError('Invalid probability or no-cache mask shape')
    if not torch.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Gate probability must lie in [0, 1]')
    valid = torch.ones_like(force) if valid_mask is None else valid_mask
    if valid.shape != force.shape or valid.dtype != torch.bool:
        raise ValueError('Invalid valid-query mask')
    return (valid & (force | (probability >= .5))).detach()


def periodic_updates(context, period_frames=None, init_only=False):
    """Periodic elapsed uses the infrastructure frame index of last UPDATE."""
    if init_only:
        return context['force'].detach().clone()
    if not isinstance(period_frames, int) or isinstance(period_frames, bool) or period_frames < 1:
        raise ValueError('A positive integer Periodic interval is required')
    return (context['force'] | (context['elapsed'] >= period_frames)).detach()


learned_updates = select_updates


def self_check():
    torch.manual_seed(19)
    raw = {key: torch.randn(3, width) for key, width in zip(FIELDS, (256, 256, 256, 3, 10))}
    raw.update(obj_idxes=torch.tensor([13, 2, 81]), scores=torch.tensor([.8, .7, .6]))

    def metadata(frame, scene='contract', session=0):
        return dict(scene=scene, session=session, frame_idx=frame, token=str(frame), timestamp_us=frame * 100000.)

    mirror, receiver = TemporalCache(), ReceiverCache()
    context = mirror.prepare(int8_roundtrip(raw), metadata(0))
    assert context['tokens'].shape == (3, 2324) and context['force'].all()
    flags = select_updates(torch.zeros(3), context['force'])
    assert flags.all()  # p=0 still initializes missing received history.
    decoded, _ = receiver.unpack(pack(raw, metadata(0), flags), metadata(0))
    mirror.commit(raw, flags, metadata(0), decoded)
    for elapsed in (1, 4, 8, 16):
        context = mirror.prepare(int8_roundtrip(raw), metadata(elapsed))
        assert not context['force'].any() and not select_updates(torch.full((3,), .49), context['force']).any()
        assert select_updates(torch.full((3,), .50), context['force']).all()
        assert context['elapsed'].tolist() == [float(elapsed)] * 3
    # A permissive B4 cannot add updates; gate has no budget capacity argument.
    assert not select_updates(torch.full((3,), .49), context['force']).any()

    for init_only in (False, True):
        mirror, receiver = TemporalCache(), ReceiverCache()
        actual = []
        for frame in range(17):
            current = dict(raw)
            current['query_feats'] = raw['query_feats'] + frame
            current['scores'] = raw['scores'] + frame / 100
            current['ref_pts'] = raw['ref_pts'] + frame / 10
            meta = metadata(frame)
            context = mirror.prepare(int8_roundtrip(current), meta)
            flags = periodic_updates(context, 8, init_only)
            actual.append(bool(flags[0]))
            mandatory, cost, full = message_sizes(current, meta)
            assert cost == 780 and full == mandatory + 3 * cost
            for mask in (torch.zeros(3, dtype=torch.bool), torch.tensor([True, False, True]), torch.ones(3, dtype=torch.bool)):
                assert len(pack(current, meta, mask)) == mandatory + cost * int(mask.sum())
            packet = pack(current, meta, flags)
            decoded, _ = receiver.unpack(packet, meta)
            before = {track: item['frame_idx'] for track, item in mirror.cache.items()}
            mirror.commit(current, flags, meta, decoded)
            for position, track in enumerate(current['obj_idxes'].tolist()):
                assert all(torch.equal(mirror.cache[track][key], decoded[key][position]) for key in FEATURES)
                assert not any(mirror.cache[track][key].requires_grad for key in FEATURES)
                assert mirror.observed[track]['score'] == float(current['scores'][position])
                if not flags[position]:
                    assert mirror.cache[track]['frame_idx'] == before[track]
        assert actual == ([frame in (0, 8, 16) for frame in range(17)] if not init_only else [True] + [False] * 16)
        permutation = torch.tensor([2, 0, 1])
        reordered = {key: value[permutation] for key, value in current.items()}
        context = mirror.prepare(int8_roundtrip(reordered), metadata(17))
        assert all(torch.equal(context['last'][key], decoded[key][permutation]) for key in FEATURES)
        for meta in (metadata(0, session=1), metadata(0, scene='other', session=0)):
            context = mirror.prepare(int8_roundtrip(reordered), meta)
            assert context['force'].all()
            try:
                receiver.unpack(pack(reordered, meta, torch.zeros(3, dtype=torch.bool)), meta)
            except ValueError:
                pass
            else:
                raise AssertionError('Native reset must reject old-session REUSE')
            decoded, _ = receiver.unpack(pack(reordered, meta, context['force']), meta)
            mirror.commit(reordered, context['force'], meta, decoded)
    empty = {key: value[:0] for key, value in raw.items()}
    mirror, receiver = TemporalCache(), ReceiverCache()
    context = mirror.prepare(int8_roundtrip(empty), metadata(0))
    decoded, _ = receiver.unpack(pack(empty, metadata(0), context['force']), metadata(0))
    mirror.commit(empty, context['force'], metadata(0), decoded)
    assert context['tokens'].shape == (0, 2324)
    assert not select_updates(torch.ones(3), torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)).any()
    print('V2 constructed gate/cache/serializer checks PASS (not a dataset experiment)')
    return {'status': 'PASS', 'scope': 'constructed CPU checks; no dataset experiment',
            'gate_no_stale_force_no_budget_fill': True, 'p8_and_init_only': True,
            'session_scene_reset': True, 'received_vs_observed_history': True,
            'serializer_actual_780_bytes_per_update': True, 'id_permutation_cache_alignment': True,
            'empty_and_padded_queries': True}


if __name__ == '__main__':
    self_check()
