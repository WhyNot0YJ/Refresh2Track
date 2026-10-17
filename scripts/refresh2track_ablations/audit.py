"""Read-only V4 baseline, complete split and feature-contract audit."""
import json
from pathlib import Path
import subprocess
import runtime


def audit(root):
    root = Path(root)
    runtime.initialize(root)
    torch = runtime.native.torch
    reference = runtime.verify_reference(root)
    checkpoint = torch.load(str(runtime.reference(root) / 'best.pth'), map_location='cpu')
    assert checkpoint['epoch'] == reference['best_epoch']
    assert checkpoint['identity']['settings'] == reference['settings']
    assert checkpoint['lr_progress'] == float(checkpoint['epoch'])
    results = runtime.read(runtime.reference(root) / 'final_validation.json')
    baseline = next(row for row in results['rows'] if row['method'] == 'V4' and row['seed'] == 0)
    assert baseline['sample_count'] == 3316 and baseline['scene_count'] == 21
    frozen_manifest = runtime.read(runtime.reference(root) / 'manifest.json')
    asset_checks = {}
    for relative, expected in frozen_manifest['asset_locks'].items():
        actual = runtime.digest(root / relative)
        assert actual == expected, 'Frozen shared asset differs: ' + relative
        asset_checks[relative] = actual
    for asset in frozen_manifest['shared_assets']['assets']:
        relative = asset['source_relative_path']
        if asset['kind'] == 'file' and relative.startswith('data_infos/'):
            actual = runtime.digest(root / relative)
            assert actual == asset['sha256'], 'Frozen official infos differ: ' + relative
            assert (root / relative).stat().st_size == asset['size_bytes']
            asset_checks[relative] = actual
    assert asset_checks['checkpoints/cooptrack_r50_coop.pth'] == baseline['pairing_identity']['native_checkpoint_sha256']
    official = runtime.read(root / 'code/CoopTrack/data/split_datas/cooperative-split-data-spd.json')['batch_split']
    splits = {}
    val_cache_digest = None
    for split, count, scenes in (('train', 7362, 46), ('val', 3316, 21)):
        infos = runtime.native.load_infos(root, 'coop', split)
        actual_scenes = {str(row['scene_token']) for row in infos}
        assert len(infos) == count and len(actual_scenes) == scenes and actual_scenes == set(official[split])
        sender = {row['token']: row for row in runtime.native.native_manifest(root, split)}
        assert all(str(row['token_inf']) in sender for row in infos)
        assert all((root / 'query_cache' / split / (token + '.pkl')).is_file() for token in sender)
        if split == 'val':
            cache_hashes = {token: runtime.digest(root / 'query_cache/val' / (token + '.pkl')) for token in sorted(sender)}
            val_cache_digest = __import__('hashlib').sha256(json.dumps(cache_hashes, sort_keys=True).encode()).hexdigest()
            assert val_cache_digest == baseline['pairing_identity']['cache_sha256'], 'Val cache differs from frozen V4'
        splits[split] = {'samples': count, 'scenes': scenes, 'sender_cache_files': len(sender), 'status': 'PASS'}
    import torch
    from v2_message import TemporalCache, int8_roundtrip
    from main_message import FIELDS
    row = runtime.native.native_manifest(root, 'train')[0]
    raw = runtime.native.fields(root / 'query_cache/train' / (row['token'] + '.pkl'))
    raw = {name: raw[name] for name in FIELDS + ('obj_idxes', 'scores')}
    current = int8_roundtrip(raw)
    context = TemporalCache().prepare(current, row)
    assert context['tokens'].shape[1] == 2324 and context['tokens'].dtype == torch.float32
    assert context['force'].all() and not context['tokens'][:, 768:1536].count_nonzero()
    assert torch.equal(context['tokens'][:, 1536:2304], context['tokens'][:, :768])
    report = {'status': 'PASS', 'V4_FULL_PRESERVED': 'PASS', 'CONFIG_PARITY': 'PASS',
              'baseline': {key: baseline[key] for key in ('epoch', 'AMOTA', 'mAP', 'MOTA', 'IDS', 'feature_payload_BPS', 'serialized_BPS', 'optional_update_ratio')},
              'best_checkpoint_sha256': reference['checkpoint_sha256'], 'completed_epoch': reference['completed_epoch'],
              'settings': reference['settings'], 'splits': splits,
              'feature_dtype': str(context['tokens'].dtype), 'feature_shape': list(context['tokens'].shape),
              'environment': {'python': __import__('sys').version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                              'gpu': torch.cuda.get_device_name(0), 'gpu_bytes': torch.cuda.get_device_properties(0).total_memory},
              'source_commits': reference['source_commits'],
              'git_status': {name: subprocess.check_output(['git','-C',str(root/'code'/name),'status','--short'],universal_newlines=True).strip() for name in reference['source_commits']},
              'identity_files_checked': len(reference['canonical_source_hashes']),
              'shared_asset_sha256_checks': asset_checks,
              'val_cache_sha256': val_cache_digest,
              'official_val_use': 'checkpoint selection, early stopping and final reporting; selection bias acknowledged'}
    directory = root / 'reports/ablations'
    runtime.write(directory / 'v4_preflight.json', report)
    table = '\n'.join('| ' + a + ' | ' + b + ' |' for a,b in (
        ('0:768', 'current three INT8-dequantized 256-D latents'),
        ('768:1536', 'last actually received three latents'),
        ('1536:2304', 'current minus last actually received'),
        ('2304', 'elapsed infrastructure frames since last actual UPDATE'),
        ('2305', 'current confidence'), ('2306', 'confidence change since last observed'),
        ('2307:2310', 'current ref_pts'), ('2310:2313', 'ref_pts change since last observed'),
        ('2313:2323', 'current pred_boxes'), ('2323', 'track lifecycle age')))
    text = '# Refresh2Track V4 Full preflight\n\nPASS. This is an audit, not new experiment results.\n\n'
    text += 'V4 Full was completed at epoch %s; best epoch %s. Frozen checkpoint: `%s/best.pth`; SHA256 `%s`.\n\n' % (reference['completed_epoch'], reference['best_epoch'], runtime.reference(root), reference['checkpoint_sha256'])
    text += '| Slice | Meaning |\n|---|---|\n' + table + '\n\n'
    text += 'Input is N×2324 FP32. Missing received latents and elapsed are zero; explicit delta then equals current. V2 TemporalCache forces UPDATE only when no received cache exists. ReceiverCache and sender mirror stay active for all variants. IDs are native predicted track IDs.\n\n'
    text += 'Hard gate: no-cache OR p≥0.5. Training receives actual packet-decoded values; STE is decoded + surrogate − stop_gradient(surrogate), with surrogate p×current+(1−p)×last and forced p=1. Native task gradients reach the MLP through these latents and association affinity. Communication loss is 780×sum(optional p)/12840.189523169913 per real frame, then mean across real lanes. C keeps this diagnostic but sets its total-loss coefficient to zero.\n\n'
    text += 'Wire protocol APTU0001: 16-byte prefix + variable JSON header H + 61N + 780U; actual len(packet) is checked. Payload at nominal10Hz is10×mean(52N+780U); serialized BPS is10×mean(len(packet)).\n\n'
    text += 'Training: complete46scenes/7362samples; validation: complete21scenes/3316samples. Sender caches and pairing tokens are present; data inventory and nine raw-data links were verified at transfer completion. No missing-data downloads or regeneration.\n\n'
    text += 'MLP71361parameters; frozen R50. Batch4; seed0; FP32; AdamW peak LR1e-4, weight decay1e-4, gradient clip5; one warmup epoch1e-5→1e-4, cosine horizon36 with floor1e-6. Validation every3epochs; raw AMOTA strict improvement; patience4, minimum stopping epoch12. Official val is used for selection and reporting and is not an independent test.\n\n'
    text += 'Canonical execution and six V4 reference files have matching SHA256. New adapters/configurations/outputs are isolated under scripts/refresh2track_ablations, configs/refresh2track_ablations and experiments/v4_ablations. The server uses its AutoDL container plus the exact restored Conda prefix; environment and CUDA custom-op checks passed.\n\n'
    text += 'Exact audit, GPU details, Git commits/status, settings and baseline metrics:\n\n```json\n' + json.dumps(report, indent=2) + '\n```\n'
    (directory / 'v4_preflight.md').write_text(text)
    return report
