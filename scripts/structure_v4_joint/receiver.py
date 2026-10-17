"""Receiver-only physical state sidecar over the unchanged V4 wire protocol."""
import importlib.util
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from main_message import PREFIX
from v2_message import ReceiverCache as V4ReceiverCache


@lru_cache(maxsize=1)
def official_box_decoder():
    source = SCRIPTS.parent / 'code/CoopTrack/projects/mmdet3d_plugin/core/bbox/util.py'
    spec = importlib.util.spec_from_file_location('structure_joint_official_bbox_util', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.denormalize_bbox


class ReceiverCache(V4ReceiverCache):
    """Expose [log1p(age), displacement, abs wrapped yaw delta, velocity delta]."""

    def __init__(self, inf_pc_range=(0., -51.2, -5., 102.4, 51.2, 3.), box_decoder=None):
        super().__init__()
        self.inf_pc_range = torch.as_tensor(inf_pc_range, dtype=torch.float32).clone()
        if (self.inf_pc_range.shape != (6,) or not torch.isfinite(self.inf_pc_range).all()
                or not (self.inf_pc_range[3:] > self.inf_pc_range[:3]).all()):
            raise ValueError('Infrastructure point cloud range must contain three finite increasing bounds')
        self.box_decoder = box_decoder or official_box_decoder()
        self.anchors = {}
        self.receiver_state = torch.empty((0, 4), dtype=torch.float32)
        self.receiver_ids = torch.empty(0, dtype=torch.int64)

    def unpack(self, payload, expected_frame=None):
        previous_identity = self.identity
        decoded, header = super().unpack(payload, expected_frame)
        if self.identity != previous_identity:
            self.anchors = {}
        metadata = header['frame']
        count = header['queries']
        _, header_len, _ = PREFIX.unpack_from(payload)
        body_offset = PREFIX.size + header_len
        ids = np.frombuffer(payload, dtype='<i8', count=count, offset=body_offset).copy()
        flags = np.frombuffer(payload, dtype=np.uint8, count=count,
                              offset=body_offset + 8 * count)
        self.receiver_ids = torch.from_numpy(ids)
        if not count:
            self.receiver_state = torch.empty((0, 4), dtype=torch.float32)
            return decoded, header
        anchors = []
        for row, track_id in enumerate(ids.tolist()):
            key = (metadata['scene'], metadata['session'], track_id)
            if flags[row]:
                self.anchors[key] = {
                    'frame_idx': metadata['frame_idx'],
                    'ref_pts': decoded['ref_pts'][row].detach().clone(),
                    'pred_boxes': decoded['pred_boxes'][row].detach().clone(),
                }
            anchors.append(self.anchors[key])
        elapsed = decoded['ref_pts'].new_tensor([
            metadata['frame_idx'] - anchor['frame_idx'] for anchor in anchors])
        if (elapsed < 0).any():
            raise ValueError('Received frame precedes its last high-dimensional UPDATE')
        anchor_refs = torch.stack([anchor['ref_pts'] for anchor in anchors])
        scale = self.inf_pc_range[3:] - self.inf_pc_range[:3]
        current_positions = decoded['ref_pts'] * scale + self.inf_pc_range[:3]
        anchor_positions = anchor_refs * scale + self.inf_pc_range[:3]
        current_boxes = self.box_decoder(decoded['pred_boxes'], self.inf_pc_range.tolist())
        anchor_boxes = self.box_decoder(torch.stack([anchor['pred_boxes'] for anchor in anchors]),
                                        self.inf_pc_range.tolist())
        yaw_delta = current_boxes[:, 6] - anchor_boxes[:, 6]
        self.receiver_state = torch.stack((
            torch.log1p(elapsed),
            (current_positions - anchor_positions).norm(dim=-1),
            torch.atan2(yaw_delta.sin(), yaw_delta.cos()).abs(),
            (current_boxes[:, 7:9] - anchor_boxes[:, 7:9]).norm(dim=-1),
        ), dim=-1).detach()
        if not torch.isfinite(self.receiver_state).all():
            raise ValueError('Nonfinite physical receiver sidecar state')
        return decoded, header


def self_check():
    from main_message import FIELDS, FEATURES, message_sizes, pack

    torch.manual_seed(0)
    current = {name: torch.randn(2, width) for name, width in
               zip(FIELDS, (256, 256, 256, 3, 10))}
    current['ref_pts'] = torch.tensor([[.1, .2, .5], [.4, .5, .6]])
    current['pred_boxes'] = torch.zeros(2, 10)
    yaw = torch.tensor([math.pi - .1, .5])
    current['pred_boxes'][:, 6] = yaw.sin()
    current['pred_boxes'][:, 7] = yaw.cos()
    current.update(obj_idxes=torch.tensor([19, 2]), scores=torch.tensor([.8, .7]))

    def metadata(frame, scene='contract', session=0):
        return dict(scene=scene, session=session, frame_idx=frame,
                    token=str(frame), timestamp_us=frame * 100000.)

    receiver, baseline = ReceiverCache(), V4ReceiverCache()

    def receive(values, meta, updates):
        packet = pack(values, meta, updates)
        actual, header = receiver.unpack(packet, meta)
        expected, expected_header = baseline.unpack(packet, meta)
        assert header == expected_header
        assert all(torch.equal(actual[name], expected[name]) for name in FIELDS)
        assert tuple(receiver.receiver_state.shape) == (len(values['obj_idxes']), 4)
        assert receiver.receiver_state.dtype == torch.float32
        return actual

    decoded = receive(current, metadata(0), torch.ones(2, dtype=torch.bool))
    assert torch.equal(receiver.receiver_state, torch.zeros(2, 4))
    anchor = receiver.anchors[('contract', 0, 19)]
    changed = {name: value.clone() for name, value in current.items()}
    changed['ref_pts'][0, 0] += 3 / 102.4
    changed['ref_pts'][0, 1] += 4 / 102.4
    changed_yaw = torch.tensor(-math.pi + .1)
    changed['pred_boxes'][0, 6] = changed_yaw.sin()
    changed['pred_boxes'][0, 7] = changed_yaw.cos()
    changed['pred_boxes'][0, 8:10] = torch.tensor([3., 4.])
    for name in FEATURES:
        changed[name] += 100
    reused = receive(changed, metadata(3), torch.zeros(2, dtype=torch.bool))
    assert all(torch.equal(reused[name], decoded[name]) for name in FEATURES)
    assert torch.allclose(receiver.receiver_state[0],
                          torch.tensor([math.log1p(3), 5., .2, 5.]), atol=2e-5)
    assert receiver.anchors[('contract', 0, 19)] is anchor
    receive(changed, metadata(6), torch.zeros(2, dtype=torch.bool))
    assert receiver.receiver_state[0, 0] == math.log1p(6)
    assert receiver.anchors[('contract', 0, 19)]['frame_idx'] == 0
    order = torch.tensor([1, 0])
    reordered = {name: value[order] for name, value in changed.items()}
    previous_state = receiver.receiver_state.clone()
    receive(reordered, metadata(7), torch.zeros(2, dtype=torch.bool))
    assert receiver.receiver_ids.tolist() == [2, 19]
    assert torch.allclose(receiver.receiver_state[:, 1:], previous_state[order, 1:])
    rebuilt = {name: value.clone() for name, value in reordered.items()}
    rebuilt['obj_idxes'][1] = 77
    missing = pack(rebuilt, metadata(8), torch.zeros(2, dtype=torch.bool))
    try:
        receiver.unpack(missing, metadata(8))
    except ValueError as error:
        assert 'REUSE' in str(error)
    else:
        raise AssertionError('Rebuilt predicted IDs must initialize their own receiver anchor')
    receive(rebuilt, metadata(8), torch.tensor([False, True]))
    assert torch.equal(receiver.receiver_state[1], torch.zeros(4))
    receive(rebuilt, metadata(9), torch.ones(2, dtype=torch.bool))
    assert torch.equal(receiver.receiver_state, torch.zeros(2, 4))
    for meta in (metadata(0, session=1), metadata(0, scene='other')):
        try:
            receiver.unpack(pack(rebuilt, meta, torch.zeros(2, dtype=torch.bool)), meta)
        except ValueError as error:
            assert 'REUSE' in str(error)
        else:
            raise AssertionError('Native reset must reject old-session REUSE')
        receive(rebuilt, meta, torch.ones(2, dtype=torch.bool))
        assert torch.equal(receiver.receiver_state, torch.zeros(2, 4))
        assert all(key[:2] == (meta['scene'], meta['session']) for key in receiver.anchors)
    empty = {name: value[:0] for name, value in rebuilt.items()}
    receive(empty, metadata(1, scene='other'), torch.zeros(0, dtype=torch.bool))
    assert receiver.receiver_state.shape == (0, 4)
    mandatory, cost, full = message_sizes(current, metadata(0))
    assert cost == 780 and full == mandatory + 2 * 780
    for flags in (torch.zeros(2, dtype=torch.bool), torch.tensor([False, True]),
                  torch.ones(2, dtype=torch.bool)):
        assert len(pack(current, metadata(0), flags)) == mandatory + 780 * int(flags.sum())
    result = dict(status='PASS', scope='constructed CPU packet checks, not a dataset experiment',
                  baseline_decode_exact=True, receiver_only_state=True,
                  physical_position_yaw_velocity=True, reuse_preserves_update_anchor=True,
                  predicted_id_permutation_and_rebuild=True, scene_session_reset=True,
                  empty_queries=True, serialized_update_bytes=780)
    print(result)
    return result


if __name__ == '__main__':
    self_check()
