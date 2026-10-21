"""One raw-only SHA/header audit, no image arrays, tests or remote writes."""
import base64
import collections
import csv
import json

from common import ROOT, m, run_remote

CODE = r'''
import collections,concurrent.futures,datetime,hashlib,json,pathlib,subprocess,time,traceback
import nibabel as nib
import numpy as np
members=__MEMBERS__;raw=__RAW__;overlay=__OVERLAY__
deadline=time.monotonic()+540
base=pathlib.Path('/public/home/zqchen/data')
corrected=base/'MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels'
native=pathlib.Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_vae_epoch004_full_reconstruction_20261008_r2/dataset/cases')
result={'status':'pass','source_kind':'original_authentic_native','allowed_roots':[str(base)],'cases':{},'errors':[],
 'master_membership_sha256':__MASTER_SHA__,'raw_manifest_sha256':__RAW_SHA__,'overlay_sha256':__OVERLAY_SHA__,
 'scope':'original source byte SHA and header only; no get_fdata/test/quality scoring/remote writes'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):
   if time.monotonic()>deadline:raise TimeoutError('Raw source audit deadline; no retry')
   h.update(block)
 return h.hexdigest()
def process(member):
 case=member['source_case_id'];mods=['t1c','t1n','t2f']+(['t2w'] if member['t2w_status']=='authentic' else [])
 entry={k:member[k] for k in ['split','patient_group','t2w_status']};entry['files']={}
 try:
  manifest=None
  if member['t2w_status']=='authentic':
   mp=native/case/'case_manifest.json'
   assert mp.is_file(),'Original authentic provenance manifest missing'
   manifest=json.loads(mp.read_text())
   assert manifest['case_id']==case and manifest['split']==member['split'] and manifest['geometry']=='native'
   entry['historical_authentic_source_manifest']={'path':str(mp),'sha256':sha(mp)}
  for mod in mods+['seg']:
   p=corrected/(case+'-seg.nii.gz') if mod=='seg' and case in overlay else base/case/(mod+'.nii.gz')
   target=p.resolve(strict=True)
   assert target==p, 'Canonical original source path redirects: '+str(p)
   assert p.is_file()
   image=nib.load(str(p))
   assert len(image.shape)==3 and np.isfinite(image.affine).all()
   spacing=image.header.get_zooms()[:3]
   assert np.isfinite(spacing).all() and min(spacing)>0
   row=raw[case]
   expected_shape=[int(x) for x in row['shape_'+('seg' if mod=='seg' else mod)].split('x')]
   expected_spacing=[float(x) for x in row['spacing_'+('seg' if mod=='seg' else mod)].split(',')]
   assert list(image.shape)==expected_shape and np.allclose(spacing,expected_spacing,rtol=0,atol=1e-4), 'Archived original header mismatch'
   ref=entry['files'].get('t1c')
   if ref:assert image.shape==tuple(ref['shape']) and np.allclose(image.affine,ref['affine'],rtol=0,atol=1e-4), 'Native modality geometry mismatch'
   digest=sha(p)
   if manifest:assert digest==manifest['source_sha256'][mod], 'Historical authentic original SHA mismatch'
   entry['files'][mod]={'path':str(p),'resolved_path':str(target),'sha256':digest,'size':p.stat().st_size,
    'shape':list(image.shape),'spacing':[float(s) for s in spacing],'affine':image.affine.tolist(),
    'label_source':'corrected_overlay' if mod=='seg' and case in overlay else 'raw_original'}
  return case,entry,None
 except Exception as e:return case,None,{'case':case,'message':str(e),'traceback':traceback.format_exc()}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for case,entry,error in pool.map(process,members):
  if error:result['errors'].append(error)
  else:result['cases'][case]=entry
result['status']='failed' if result['errors'] else 'pass'
result['cases_ok']=len(result['cases']);result['cases_expected']=1165
result['split_counts']=dict(collections.Counter(r['split'] for r in members))
result['source_bytes']=sum(f['size'] for c in result['cases'].values() for f in c['files'].values())
result['remote_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
result['df']=subprocess.run(['df','-B1',str(base)],capture_output=True,text=True,timeout=25).stdout
print(json.dumps(result))
'''


def main():
    with (ROOT / 'plan/master_membership.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    selected = [r for r in rows if r['split'] in ('train', 'val')]
    assert len(selected) == 1165
    raw = {}
    for name in ('raw_manifest.csv', 'ucsd_manifest.csv'):
        with (ROOT / 'plan' / name).open(newline='') as f:
            for row in csv.DictReader(f):
                if row['case_id'] not in raw:
                    raw[row['case_id']] = row
    with (ROOT / 'plan/corrected_label_overlay.csv').open(newline='') as f:
        overlay = {r['case_id']: r for r in csv.DictReader(f) if r['applied'].lower() == 'true'}
    assert all(r['source_case_id'] in raw for r in selected)
    code = CODE
    selected = [{k: r[k] for k in ('source_case_id', 'split', 'patient_group', 't2w_status')} for r in selected]
    headers = {r['source_case_id']: {k: v for k, v in raw[r['source_case_id']].items()
                                   if k.startswith(('shape_', 'spacing_'))} for r in selected}
    for key, value in {'__MEMBERS__': selected, '__RAW__': headers,
                       '__OVERLAY__': sorted(overlay), '__MASTER_SHA__': m.sha(ROOT / 'plan/master_membership.csv'),
                       '__RAW_SHA__': m.sha(ROOT / 'plan/raw_manifest.csv'),
                       '__OVERLAY_SHA__': m.sha(ROOT / 'plan/corrected_label_overlay.csv')}.items():
        code = code.replace(key, repr(value))
    # Large metadata is transmitted as a single compressed argv, never an archive or data download.
    import zlib
    encoded = base64.b64encode(zlib.compress(code.encode())).decode()
    launch = 'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(encoded) + ')))'
    assert len(launch.encode()) < 60000, 'Bound SSH argv before connecting'
    result, audit = run_remote('raw_source_sha_header_audit', launch, timeout=600)
    assert result['cases_ok'] == 1165 and not result['errors']
    m.exclusive(ROOT / 'config/SOURCE_RECORDS.json', result)
    cfgpath = ROOT / 'config/RUN.json'
    cfg = json.loads(cfgpath.read_text())
    assert cfg['source_records_sha256'] == 'pending_source_audit'
    cfg['source_records_sha256'] = m.sha(ROOT / 'config/SOURCE_RECORDS.json')
    cfgpath.chmod(0o640)
    cfgpath.write_text(json.dumps(cfg, indent=2, sort_keys=True) + '\n')
    cfgpath.chmod(0o440)
    print(json.dumps({'status': 'pass', 'cases': 1165, 'source_bytes': result['source_bytes'], 'audit': str(audit)}))


if __name__ == '__main__':
    main()
