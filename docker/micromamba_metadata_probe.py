import urllib.request,json,pathlib
url='https://api.github.com/repos/mamba-org/micromamba-releases/releases/latest'
request=urllib.request.Request(url,headers={'Accept':'application/vnd.github+json','User-Agent':'CoopTrack-Stage1-Audit'})
d=json.load(urllib.request.urlopen(request,timeout=45))
selected=[{k:item.get(k) for k in ['name','browser_download_url','size','digest']} for item in d['assets'] if 'linux-64' in item['name']]
out={'api_url':url,'tag_name':d['tag_name'],'html_url':d['html_url'],'assets':selected}
p=pathlib.Path('/mnt/d/CoopTrack_Audit/docker/locks/micromamba_bootstrap.json');p.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))