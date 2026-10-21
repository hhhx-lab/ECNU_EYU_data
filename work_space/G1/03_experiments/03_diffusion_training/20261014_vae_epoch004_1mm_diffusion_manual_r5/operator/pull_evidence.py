"""Read back immutable light launch evidence; no additional live job snapshot."""
import json
import shlex
from common import ROOT, REMOTE, PARENT_REMOTE, m, transport


def main():
    access = transport('pull_launch_evidence')
    code = r'''
import base64,hashlib,json,pathlib,re
root=pathlib.Path(__ROOT__);parent=pathlib.Path(__PARENT__)
result={}
for prefix,base in [('r5',root),('r4',parent)]:
 files=[p for p in (base/'evidence').iterdir() if p.suffix in ('.json','.log') and 'PAYLOAD' not in p.name]
 for p in sorted(files):
  assert p.stat().st_size<2500000
  raw=p.read_bytes();assert b'password:' not in raw.lower(),'unexpected authentication prompt in evidence'
  result[prefix+'/'+p.name]={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode(),'bytes':len(raw)}
print(json.dumps(result,sort_keys=True))
'''
    for key, value in {'__ROOT__':str(REMOTE), '__PARENT__':str(PARENT_REMOTE)}.items():
        code = code.replace(key, repr(value))
    try:
        access.start_vpn()
        data = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=100).strip().splitlines()[-1])
        import base64
        import hashlib
        for name, item in data.items():
            raw = base64.b64decode(item['base64'])
            assert hashlib.sha256(raw).hexdigest() == item['sha256']
            path = access.audit / 'pulled' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as h:
                h.write(raw)
            assert m.sha(path) == item['sha256']
        m.exclusive(access.audit / 'PULL_RECEIPT.json', {**m.META,'status':'sha_verified',
            'files':{n:{'sha256':i['sha256'],'bytes':i['bytes']} for n,i in data.items()},'live_queue_queries':0})
        print(json.dumps({'audit':str(access.audit),'files':len(data),'sha_verification':'pass','extra_snapshots':0}))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
