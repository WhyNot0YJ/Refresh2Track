"""Audit the local preregistered structure experiment before changing the model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def audit(root):
    output = root / 'experiments/structure_v4_joint'
    output.mkdir(parents=True, exist_ok=True)
    result = {'status': 'RUNNING', 'utc': datetime.now(timezone.utc).isoformat(),
              'root': str(root), 'remote_A_B': 'PENDING: check host-specific preflight and queue state',
              'local_experiment': 'C_GBA_GATE', 'blockers': []}
    try:
        result['git_head'] = subprocess.check_output(
            ['git', '-c', 'safe.directory=*', '-C', str(root / 'code/CoopTrack'),
             'rev-parse', 'HEAD'], universal_newlines=True).strip()
        result['git_status'] = subprocess.check_output(
            ['git', '-c', 'safe.directory=*', '-C', str(root / 'code/CoopTrack'),
             'status', '--short'], universal_newlines=True)
        result['official_reference_commit'] = '29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399'
        result['official_commit_matches'] = result['git_head'] == result['official_reference_commit']
        result['disk_free_bytes'] = shutil.disk_usage(root).free
        result['processes'] = subprocess.check_output(
            ['ps', '-eo', 'pid,ppid,args'], universal_newlines=True)
        result['checkpoints'] = {}
        for name, relative in (
                ('coop', 'checkpoints/cooptrack_r50_coop.pth'),
                ('inf', 'checkpoints/cooptrack_r50_inf.pth'),
                ('old_v4_equivalence_only', 'experiments/v4_budget_free/checkpoints/best.pth')):
            path = root / relative
            result['checkpoints'][name] = {'path': relative, 'sha256': digest(path),
                                            'bytes': path.stat().st_size}
        print(json.dumps({'phase': 'checkpoint_identities', 'checkpoints': result['checkpoints']}), flush=True)
        sys.path.insert(0, str(root / 'code/CoopTrack'))
        sys.path.insert(0, str(root / 'scripts'))
        import torch
        import mmcv
        import mmdet3d
        import projects.mmdet3d_plugin
        import main_runtime as native
        result['environment'] = {'python': sys.version, 'torch': torch.__version__,
                                 'cuda': torch.version.cuda, 'mmcv': mmcv.__version__,
                                 'mmdet3d': mmdet3d.__version__,
                                 'cuda_available': torch.cuda.is_available()}
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable inside the fixed image')
        result['environment']['gpu'] = torch.cuda.get_device_name(0)
        result['environment']['gpu_memory_bytes'] = torch.cuda.get_device_properties(0).total_memory
        assert (torch.ones(2).cuda() + 1).sum().item() == 4
        result['environment']['cuda_tensor_check'] = 'PASS'
        native.mmcv = mmcv
        native.torch = torch
        native.DataContainer = mmcv.parallel.DataContainer
        native.np = __import__('numpy')
        result['splits'] = {}
        scene_sets = {}
        for split, expected_samples, expected_scenes in (('train', 7362, 46), ('val', 3316, 21)):
            infos = native.load_infos(root, 'coop', split)
            scenes = {str(info['scene_token']) for info in infos}
            scene_sets[split] = scenes
            assert len(infos) == expected_samples and len(scenes) == expected_scenes
            cache = root / 'query_cache' / split
            missing = [str(info['token_inf']) for info in infos
                       if not (cache / (str(info['token_inf']) + '.pkl')).is_file()]
            assert not missing, 'Missing cooperative cache: ' + str(missing[:10])
            files = sorted(cache.glob('*.pkl'))
            identity = hashlib.sha256()
            for index, path in enumerate(files, 1):
                identity.update(path.name.encode())
                identity.update(bytes.fromhex(digest(path)))
                if index % 1000 == 0:
                    print(json.dumps({'phase': 'cache_hash', 'split': split,
                                      'files': index, 'total': len(files)}), flush=True)
            result['splits'][split] = {'samples': len(infos), 'scenes': len(scenes),
                                      'cache_files': len(files), 'missing_cache_files': missing,
                                      'cache_identity_sha256': identity.hexdigest(),
                                      'info_sha256': digest(root / 'data_infos/V2X-Seq-SPD-CoopTrack-full/cooperative' /
                                                            ('spd_infos_temporal_' + split + '.pkl'))}
        assert not scene_sets['train'].intersection(scene_sets['val'])
        result['train_val_scene_disjoint'] = True
        result['source_hashes'] = {str(path.relative_to(root)): digest(path)
                                   for path in sorted((root / 'code/CoopTrack/projects').rglob('*.py'))}
        result['parameter_allowlist'] = {
            'scheduler': 'all; from scratch seed0; 71361 expected',
            'gba': ['get_node_veh', 'get_node_inf', 'get_edge', 'wq', 'wk', 'we', 'ffn'],
            'fusion': ['fuse_feats', 'fuse_motion', 'fuse_embed', 'cross_domain_query', 'cross_domain_motion'],
            'new': ['gba_state_adapter', 'fusion_branch_gate'],
            'all_other_parameters_and_BN': 'frozen'}
        result['protocol'] = {'batch_size': 4, 'precision': 'FP32', 'seed': 0,
                              'lambda_comm': .05, 'max_epochs': 36, 'validation_interval': 3,
                              'new_peak_lr': 1e-4, 'original_peak_lr': 1e-5,
                              'scheduler_initialization': 'from scratch; never load old V4 for training'}
        result['status'] = 'PASS'
    except Exception as error:
        result['status'] = 'BLOCKED'
        result['blockers'].append(str(error))
        result['traceback'] = traceback.format_exc()
    (output / 'precheck.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    text = '# Structure V4 joint local precheck\n\nPRECHECK: ' + result['status'] + '\n\n'
    text += 'Local: C_GBA_GATE. Remote: A_GBA/B_GATE, subject to independent real-data preflight.\n\n'
    text += '```json\n' + json.dumps(result, ensure_ascii=False, indent=2) + '\n```\n'
    (output / 'PRECHECK.md').write_text(text, encoding='utf-8')
    print(json.dumps({'phase': 'precheck_complete', 'status': result['status'],
                      'blockers': result['blockers']}), flush=True)
    return result['status'] == 'PASS'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(0 if audit(args.root.resolve()) else 1)
