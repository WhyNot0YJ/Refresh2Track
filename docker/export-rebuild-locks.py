import pathlib,subprocess,json,re,hashlib,email.parser
root=pathlib.Path('/mnt/d/CoopTrack_Audit');out=root/'docker/locks';out.mkdir(exist_ok=True);prefix='/home/yujie/miniconda/envs/cooptrack-smoke'
conda=subprocess.run(['/home/yujie/miniconda/bin/conda','list','-p',prefix,'--explicit','--sha256'],capture_output=True,text=True,check=True).stdout
(out/'conda-explicit-host-original.lock').write_text(conda)
conda=conda.replace('file:///opt/emiff/mamba/pkgs/https/conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/','https://conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/')
assert 'file://' not in conda
assert all(re.search(r'#[0-9a-f]{64}$',line) for line in conda.splitlines() if line and not line.startswith(('#','@')))
(out/'conda-explicit-linux-64.lock').write_text(conda)
freeze=subprocess.run([prefix+'/bin/python','-m','pip','freeze','--all'],capture_output=True,text=True,check=True).stdout
(out/'pip-freeze-host-original.txt').write_text(freeze)
installed=json.loads(subprocess.run([prefix+'/bin/python','-m','pip','list','--format=json'],capture_output=True,text=True,check=True).stdout)
(out/'pip-installed-host-original.json').write_text(json.dumps(installed,indent=2))
records=[]
for p in pathlib.Path(prefix+'/conda-meta').glob('*.json'):
 d=json.loads(p.read_text());records.append({k:d.get(k) for k in ['name','version','build','fn','url','sha256','md5','size']})
(out/'conda-installed-host-original.json').write_text(json.dumps(records,indent=2))
versions=[];requirements=[]
for item in installed:
 name=item['name'];version=item['version'];canonical=re.sub(r'[-_.]+','-',name).lower()
 if canonical=='mmdet3d':continue
 if canonical=='pip':continue
 if canonical=='ninja':version='1.11.1.3'
 versions.append(name+'=='+version)
 if canonical=='torch':requirements.append('torch @ file:///opt/rebuild/wheels/torch-1.9.1%2Bcu111-cp38-cp38-linux_x86_64.whl#sha256=2546dcaae81ac74f3f88dd4b29f5ead28852a5e8e66ca85b4f46f2eea306033c')
 elif canonical=='open3d':requirements.append('open3d @ file:///opt/rebuild/wheels/open3d-0.13.0-cp38-cp38-manylinux2014_x86_64.whl#sha256=da23d8aef00701a8351dc7948d4c0c10088cdb93308b288ba2ae6f9bdd7e4694')
 elif canonical=='ninja':
  p=root/'data/wheels/ninja-1.11.1.3-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl';sha=hashlib.sha256(p.read_bytes()).hexdigest();requirements.append('ninja @ file:///opt/rebuild/wheels/'+p.name+'#sha256='+sha)
 else:requirements.append(name+'=='+version)
(out/'pip-exact-versions.txt').write_text('\n'.join(versions)+'\n')
(out/'pip-rebuild-requirements.txt').write_text('\n'.join(requirements)+'\n')
(out/'pip-normalization.txt').write_text('pip is already locked as conda pip24.3.1. mmdet3d is built separately from the recorded official source commit. ninja is intentionally corrected from host1.11.1.1 to the verified Docker1.11.1.3 wheel. Torch/Open3D/ninja use retained local wheels with verified hashes, not stale private build paths. All other pip package versions match the captured host list exactly.\n')
source={}
for name in ['CoopTrack','mmdetection3d']:
 repo=root/'code'/name;sha=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip();diff=subprocess.run(['git','-c','core.autocrlf=true','-C',str(repo),'diff','--binary'],capture_output=True,text=True,check=True).stdout
 source[name]={'commit':sha,'diff_sha256':hashlib.sha256(diff.encode()).hexdigest(),'has_tracked_changes':bool(diff)};(out/(name+'.patch')).write_text(diff)
(out/'official-source-lock.json').write_text(json.dumps(source,indent=2))
print(json.dumps({'conda_packages':len(records),'pip_packages':len(installed),'normalized_pip_requirements':len(requirements),'source':source,'locks_path':str(out)},indent=2))
