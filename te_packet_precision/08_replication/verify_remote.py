"""After uploading, compare each release file to GitHub's public Git tree.
No write access or credentials needed. Nonzero exit means mismatch or inaccessible API.
"""
from pathlib import Path
import urllib.request,json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
URL='https://api.github.com/repos/ShimonoMLab/NetworkTM_Analysis/git/trees/main?recursive=1'
with urllib.request.urlopen(urllib.request.Request(URL,headers={'Accept':'application/vnd.github+json','User-Agent':'te-reproducibility-check'}),timeout=30) as f: t=json.load(f)
assert not t.get('truncated'), 'Git tree truncated'
prefix='te_packet_precision/'
remote={x['path'][len(prefix):]:x['sha'] for x in t['tree'] if x['type']=='blob' and x['path'].startswith(prefix)}
local={}
for p in ROOT.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts and '.venv' not in p.parts:
        b=p.read_bytes();local[p.relative_to(ROOT).as_posix()]=hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
missing=sorted(local.keys()-remote.keys());extra=sorted(remote.keys()-local.keys());changed=sorted(k for k in local.keys()&remote.keys() if local[k]!=remote[k]);matched=len(local)-len(missing)-len(changed)
print(json.dumps({'tree':t['sha'],'matched':matched,'local':len(local),'remote':len(remote),'missing':missing,'extra':extra,'changed':changed},indent=2));sys.exit(1 if missing or extra or changed else 0)
