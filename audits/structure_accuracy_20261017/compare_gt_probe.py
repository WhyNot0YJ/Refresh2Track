"""Require exact prediction equivalence and nonempty GT and fusion coverage."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def compare(normal_path, stripped_path):
    normal = json.loads(normal_path.read_text(encoding='utf-8'))
    stripped = json.loads(stripped_path.read_text(encoding='utf-8'))
    keys = ('prediction_fingerprints', 'communication', 'structure_frames',
            'rng_after_inference', 'loaded_structure_weights', 'pairing_identity',
            'annotations_at_model_entry')
    checks = {key: normal[key] == stripped[key] for key in keys}
    if not all(checks.values()):
        raise AssertionError([key for key, passed in checks.items() if not passed])
    annotations = normal['annotations_at_model_entry']
    elements = sum(row['elements'] for frame in annotations for row in frame)
    pairs = sum(frame.get('matched_pairs', 0) for frame in normal['structure_frames'])
    if elements <= 0 or pairs <= 0:
        raise AssertionError('Probe lacks nonempty annotation or GBA/fusion coverage')
    if normal['frames'] != stripped['frames'] or normal['epoch'] != stripped['epoch']:
        raise AssertionError('Probe scope differs')
    return {'status': 'PASS', 'checked_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'same checkpoint and seed; normal inputs versus annotation deletion at model entry',
            'frames_per_run': normal['frames'], 'checkpoint_epoch': normal['epoch'],
            'checkpoint_sha256': normal['checkpoint_sha256'],
            'compared_fields_equal': checks, 'annotation_elements_removed': elements,
            'annotation_paths': sorted({row['path'] for frame in annotations for row in frame}),
            'matched_pairs': pairs,
            'roadside_queries': sum(frame['queries'] for frame in normal['communication']),
            'native_trainable_sha256': normal['loaded_structure_weights']['native_trainable_sha256'],
            'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (normal_path, stripped_path)},
            'limitations': ['Bounded contiguous frames, not all 3316 validation frames.',
                            'This is a GT-dependency probe, not a full accuracy rerun.',
                            'Audit directory and checkpoint identity are adapted for cross-host execution.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('normal', type=Path)
    parser.add_argument('stripped', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.normal, args.stripped)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'frames_per_run', 'annotation_elements_removed', 'matched_pairs')}))
