"""One bounded S0 capability snapshot; no remote files, downloads or jobs."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[4]
spec = importlib.util.spec_from_file_location(
    "external_mets_access",
    PROJECT / "work_space/S2/03_experiments/12_k1245_rc_aware/"
    "s2_met_aug_k1245_20260923_r3_fallback/operator/access.py",
)
access_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(access_module)
access_module.ROOT = ROOT

CODE = r'''
import importlib.metadata,json,os,shutil,subprocess,hashlib
from datetime import datetime,timezone
from pathlib import Path

root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
k1=Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_k1_selected103_official_met_20261008_r1')
def command(argv,timeout=25):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
        return dict(returncode=p.returncode,stdout=p.stdout[:24000],stderr=p.stderr[:8000])
    except (OSError,subprocess.TimeoutExpired) as e:
        return dict(returncode=124 if isinstance(e,subprocess.TimeoutExpired) else 127,error=type(e).__name__)
def file(p):
    if not p.is_file():return dict(exists=False)
    return dict(exists=True,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def listing(p):
    return [dict(name=x.name,is_directory=x.is_dir(),bytes=x.stat().st_size) for x in sorted(p.iterdir())][:60] if p.is_dir() else []
out=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed',
    status='snapshot_collected_not_runtime_pass',remote_started_utc=datetime.now(timezone.utc).isoformat(),hostname=os.uname().nodename,
    result_root=str(root),result_root_exists=root.exists(),result_parent_exists=root.parent.is_dir(),result_parent_writable=os.access(root.parent,os.W_OK),
    disk=command(['df','-B1',str(root.parent)]),quota=command(['quota','-s']),
    tools={x:shutil.which(x) for x in ['singularity','apptainer','docker','skopeo','unsquashfs','conda','sbatch','squeue','sacct','curl','wget']},
    container_modules=command(['bash','-lc','module -t avail singularity apptainer docker skopeo 2>&1']),
    container_versions={x:command([x,'--version']) for x in ['singularity','apptainer','docker','skopeo'] if shutil.which(x)},
    subuid_current_user=command(['bash','-lc','id; if test -r /etc/subuid; then awk -F: -v u="$(id -un)" \'$1==u {print}\' /etc/subuid; fi']),
    user_namespace_metadata=command(['bash','-lc','for p in /proc/sys/user/max_user_namespaces /proc/sys/kernel/unprivileged_userns_clone; do if test -r "$p"; then printf "%s=" "$p"; head -c 64 "$p"; fi; done']),
    account_queue=command(['squeue','-h','-u','zqchen','-o','%i|%T|%P|%j|%r|%b']),
    partitions=command(['sinfo','-h','-p','a100,cpu_192','-o','%P|%a|%l|%D|%G']),
    associations=command(['sacctmgr','-n','-P','show','assoc','where','user=zqchen','format=User,Account,Partition,QOS,MaxJobs,MaxSubmitJobs']),
    conda_environment_names=listing(Path('/public/home/zqchen/.conda/envs')),
    k1_root_exists=k1.is_dir(),k1_root_inventory=listing(k1),k1_contract={
        n:file(k1/n) for n in ['manifests/FIXED103_REAL_T2W_CASES.csv','manifests/INPUT_REFERENCE_SHA256.json','evidence/K1_VALIDATION.json',
        'cohort/fixed103/models/K1/evaluation/leaderboard_metrics.csv']},
    k1_model_inventory=listing(k1/'cohort/fixed103/models/K1'),
    k1_predictions_inventory=listing(k1/'cohort/fixed103/models/K1/predictions'),
    k1_snapshot=file(k1/'model_snapshots/K1/fold_0/checkpoint_final.pth'),
    new_root_inventory=listing(root),
    scope=dict(remote_writes=False,models_downloaded=False,images_read=False,test_read=False,dependencies_installed=False,jobs_created=False))
modules=['torch','nnunetv2','brats','transformers','timm','torchvision','nibabel','numpy','scipy','SimpleITK','einops','PyYAML','safetensors']
probe='import importlib.metadata as m,importlib.util as u,json; names='+repr(modules)+'; result={};\nfor n in names:\n try: result[n]=m.version(n)\n except m.PackageNotFoundError: result[n]=None\nprint(json.dumps(result))'
out['runtime_inventories']={}
for name in ['segmamba','brats_eval',*[x.name for x in Path('/public/home/zqchen/.conda/envs').iterdir() if x.is_dir() and any(t in x.name.lower() for t in ['bio','brats','external'])]]:
    py=Path('/public/home/zqchen/.conda/envs')/name/'bin/python'
    if py.is_file():out['runtime_inventories'][name]=command([str(py),'-B','-c',probe],timeout=20)
out['remote_utc']=datetime.now(timezone.utc).isoformat()
print(json.dumps(out))
'''


def main():
    for key in ("ECNU_SERVER_ENV_FILE", "ECNU_VPN_SCRIPT", "ECNU_REMOTE_PYTHON", "EXTERNAL_MET_REMOTE_ROOT"):
        if not os.environ.get(key):
            raise RuntimeError(f"Required env configuration missing: {key}")
    access = access_module.Access("s0_preflight")
    stage = "vpn_start"
    result = None
    started = access_module.datetime.now(access_module.timezone.utc).isoformat()
    stop_ok = False
    try:
        proc = subprocess.run([os.environ["ECNU_VPN_SCRIPT"], "start"], env=access.env, capture_output=True, timeout=100)
        flags = re.findall(rb"VPN_SESSION_OWNED=([01])", proc.stdout)
        if len(flags) != 1:
            raise RuntimeError("VPN ownership unconfirmed; private startup output suppressed; no SSH")
        access.owns_vpn = flags[0] == b"1"
        if proc.returncode:
            raise RuntimeError("VPN start failed; private startup output suppressed; no SSH")
        stage = "vpn_check"
        checked = subprocess.run([os.environ["ECNU_VPN_SCRIPT"], "check"], env=access.env, capture_output=True, timeout=40)
        if checked.returncode:
            raise RuntimeError("VPN check failed; no SSH")
        stage = "remote_read_only"
        command = "env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT=" + shlex.quote(os.environ["EXTERNAL_MET_REMOTE_ROOT"])
        command += " " + shlex.quote(os.environ["ECNU_REMOTE_PYTHON"]) + " -B -c " + shlex.quote(CODE)
        output = access.ssh(command, timeout=240)
        result = json.loads(output.strip().splitlines()[-1])
        result["local_received_utc"] = access_module.datetime.now(access_module.timezone.utc).isoformat()
        access_module.exclusive(access.audit / "RESULT.json", result)
        print(json.dumps(dict(audit=str(access.audit),status=result["status"],remote_utc=result["remote_utc"])))
    except BaseException as error:
        access_module.exclusive(access.audit / "OPERATION_STOPPED.json", dict(access_module.META,stage=stage,
            message=str(error),utc=access_module.datetime.now(access_module.timezone.utc).isoformat(),
            remote_attempted=stage=="remote_read_only",no_automatic_retry=True))
        raise
    finally:
        if access.owns_vpn:
            stopped = subprocess.run([os.environ["ECNU_VPN_SCRIPT"],"stop"],env=access.env,capture_output=True,timeout=40)
            stop_ok = stopped.returncode == 0
            if not stop_ok:
                status = subprocess.run([os.environ["ECNU_VPN_SCRIPT"],"status"],env=access.env,capture_output=True,text=True,timeout=40)
                states = [json.loads(line[line.index("{"):]) for line in status.stdout.splitlines() if "{" in line]
                stop_ok = status.returncode==0 and len(states)==1 and states[0].get("status")=="logout" and states[0].get("tunnel_status")=="tunnelUnavailable"
                access_module.exclusive(access.audit / "INDEPENDENT_VPN_STATUS.json",dict(access_module.META,
                    original_stop_returncode=stopped.returncode,confirmed_logout=stop_ok,status_returncode=status.returncode))
        access_module.exclusive(access.audit / "LOCAL_OPERATION_RECEIPT.json",dict(access_module.META,
            started_utc=started,completed_utc=access_module.datetime.now(access_module.timezone.utc).isoformat(),
            vpn_owned=access.owns_vpn,own_vpn_stop_completed=stop_ok,preexisting_vpn_preserved=not access.owns_vpn,
            stage=stage,result_status=result.get("status") if result else "unconfirmed"))
        if access.owns_vpn and not stop_ok:
            raise RuntimeError("Own VPN shutdown not confirmed")


if __name__ == "__main__":
    main()
