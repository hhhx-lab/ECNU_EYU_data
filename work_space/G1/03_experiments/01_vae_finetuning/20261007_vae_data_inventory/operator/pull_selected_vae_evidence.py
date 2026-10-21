"""Read-only, bounded pull of VAE selection and existing dataset evidence."""

import datetime
import hashlib
import importlib.util
import json
import pathlib
import shlex
import shutil


PROJECT = pathlib.Path(__file__).resolve().parents[6]
ACCESS_SOURCE = PROJECT / (
    "work_space/S2/03_experiments/12_k1245_rc_aware/"
    "s2_met_aug_k1245_20260923_r3_fallback/operator/access.py"
)
spec = importlib.util.spec_from_file_location("ecnu_access", ACCESS_SOURCE)
access_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(access_module)
access_module.ROOT = pathlib.Path(__file__).resolve().parents[1]

REMOTE_CODE = r'''
import pathlib,json,hashlib,datetime,os,csv
G=pathlib.Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1')
E=G/'results/experimental_unvalidated'
out={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'hostname':os.uname().nodename,'files':[],'directories':[],
     'dataset_candidates':[],'excluded_large_files':[]}
roots=[E/n for n in ('g1_vae_extended_finetune_20260901_r3_attempt03',
    'g1_vae_threeway_fixed103_validation_20260910_r1',
    'g1_vae_threeway_fixed103_validation_20260911_attempt02',
    'g1_vae_threeway_fixed103_validation_20260914_attempt03')]
budget=0
for root in roots:
    for base,dirs,files in os.walk(root):
        p=pathlib.Path(base); rel=p.relative_to(root)
        dirs[:]=[n for n in dirs if n not in {'runtime','checkpoints','__pycache__',
            'montages','comparison_samples','reconstructions','predictions'}
            and len(rel.parts)<5]
        out['directories'].append(str(p))
        for n in sorted(files):
            f=p/n
            if f.suffix.lower() not in {'.json','.csv','.md','.ok','.out','.err','.txt'}:
                continue
            size=f.stat().st_size
            if size>2000000 or budget+size>24000000:
                out['excluded_large_files'].append({'path':str(f),'bytes':size});continue
            raw=f.read_bytes();budget+=len(raw)
            out['files'].append({'root':str(root),'relative_path':str(f.relative_to(root)),
                'path':str(f),'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                'text':raw.decode('utf-8')})
ck=roots[0]/'training/checkpoints/checkpoint_epoch_004.pt'
h=hashlib.sha256()
with ck.open('rb') as handle:
    for block in iter(lambda:handle.read(1048576),b''):h.update(block)
out['selected_checkpoint']={'path':str(ck),'bytes':ck.stat().st_size,'sha256':h.hexdigest()}
assert out['selected_checkpoint']['sha256']=='5d86a0b3cbc5df34d64d5a0ff376eae453634a609bd24dec1a1191fde800b13b'
for top in (G/'data',E):
    for base,dirs,files in os.walk(top):
        p=pathlib.Path(base);depth=len(p.relative_to(top).parts)
        dirs[:]=[n for n in dirs if n not in {'input','input_inference','runtime',
            'training','predictions','reconstructions','samples','__pycache__'} and depth<3]
        if top==G/'data':out['directories'].append(str(p))
        if any(n in p.name.lower() for n in ('latent','encod','reencod')):
            data=list(p.rglob('*_latent.npy'))
            out['dataset_candidates'].append({'path':str(p),'files':len(data),
                'bytes':sum(f.stat().st_size for f in data)})
print('G1_EVIDENCE_BEGIN')
print(json.dumps(out,ensure_ascii=True))
print('G1_EVIDENCE_END')
'''


def main():
    access = access_module.Access("selected_evidence_pull")
    try:
        access.start_vpn()
        raw = access.ssh(
            "env PYTHONDONTWRITEBYTECODE=1 " + access_module.PYTHON
            + " -B -c " + shlex.quote(REMOTE_CODE), timeout=300
        )
        payload = json.loads(raw.split("G1_EVIDENCE_BEGIN", 1)[1]
                             .split("G1_EVIDENCE_END", 1)[0].strip())
        access_module.exclusive(access.audit / "REMOTE_EVIDENCE_INDEX.json", {
            **access_module.META, **{k: v for k, v in payload.items() if k != "files"},
            "files": [{k: v for k, v in f.items() if k != "text"}
                      for f in payload["files"]],
        })
        for item in payload["files"]:
            rel = pathlib.Path(item["relative_path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe source evidence path")
            dest = access.audit / "pulled" / pathlib.Path(item["root"]).name / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as handle:
                handle.write(item["text"].encode("utf-8"))
            if access_module.sha(dest) != item["sha256"]:
                raise ValueError("Pulled evidence SHA mismatch: " + str(rel))
            if dest.suffix == ".json":
                json.loads(dest.read_text())
        access_module.exclusive(access.audit / "LOCAL_OPERATION_RECEIPT.json", {
            **access_module.META,
            "local_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "remote_utc": payload["remote_utc"],
            "action": "read_only_selected_vae_evidence_pull",
            "source_immutable": True, "no_job_submission": True,
            "files_verified": len(payload["files"]),
            "json_and_sha_validation": "pass",
            "checkpoint_sha_verified": payload["selected_checkpoint"],
            "dataset_candidates": payload["dataset_candidates"],
            "local_disk_free_bytes": shutil.disk_usage(PROJECT).free,
        })
        print(json.dumps({"audit": str(access.audit),
            "remote_utc": payload["remote_utc"],
            "files_verified": len(payload["files"]),
            "dataset_candidates": payload["dataset_candidates"],
            "selected_checkpoint": payload["selected_checkpoint"],
            "g1_data_directories": [p for p in payload["directories"]
                                    if "/G1/data" in p]}, ensure_ascii=False, indent=2))
    finally:
        access.close_vpn()
        print("VPN_THIS_OPERATION_CLOSED_OR_PREEXISTING_PRESERVED")


if __name__ == "__main__":
    main()
