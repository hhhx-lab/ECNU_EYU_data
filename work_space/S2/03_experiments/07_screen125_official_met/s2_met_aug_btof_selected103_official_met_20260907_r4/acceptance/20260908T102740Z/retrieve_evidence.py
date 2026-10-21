"""Run the read-only verifier through bounded SSH and verify every downloaded byte."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
LOCAL = Path("/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated")
ROOTS = {
    "extend": LOCAL / "s2_met_aug_bef_extend175_20260907_r4",
    "met_r2": LOCAL / "s2_met_aug_btof_selected103_official_met_20260907_r2",
    "met_r4": LOCAL / "s2_met_aug_btof_selected103_official_met_20260907_r4",
}
VPN = "/Users/hwaigc/比赛+课题/服务器登陆/ecnu-vpn-split.sh"
META = {"artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed"}
started = datetime.now(timezone.utc).isoformat()
code = base64.b64encode((HERE / "verify_remote.py").read_bytes()).decode("ascii")
remote = "env PYTHONDONTWRITEBYTECODE=1 /public/home/zqchen/.conda/envs/segmamba/bin/python -c " + shlex.quote("import base64;exec(compile(base64.b64decode(" + repr(code) + "),'<read-only-acceptance>','exec'))")
env = dict(os.environ)
env["AUDIT_REMOTE_COMMAND"] = remote
for key, default in (("HPC_LOGIN_HOST","59.78.189.132"),("HPC_LOGIN_PORT","2323"),("HPC_SSH_USER","zqchen")):
    env.setdefault(key,default)
if not env.get("HPC_SSH_PASSWORD"):
    raise RuntimeError("SSH password was not loaded from the authorized env")
expect_script = r"""
log_user 0
set timeout 540
spawn ssh -p $env(HPC_LOGIN_PORT) -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$env(HPC_SSH_USER)@$env(HPC_LOGIN_HOST)" $env(AUDIT_REMOTE_COMMAND)
expect {
  -re "(P|p)assword:" { send -- "$env(HPC_SSH_PASSWORD)\r"; log_user 1; exp_continue }
  eof { exit [lindex [wait] 3] }
  timeout { close; exit 124 }
}
"""
try:
    start = subprocess.run([VPN,"start"],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=100)
    if start.returncode:
        raise RuntimeError("VPN startup failed; private output suppressed")
    response = subprocess.run(["expect"],input=expect_script,text=True,env=env,capture_output=True,timeout=570)
    if response.returncode:
        raise RuntimeError("Bounded SSH audit failed with exit " + str(response.returncode))
finally:
    stop = subprocess.run([VPN,"stop"],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    if stop.returncode:
        raise RuntimeError("VPN cleanup did not confirm success")
text = response.stdout
if "ECNU_AUDIT_JSON_BEGIN" not in text:
    raise RuntimeError("No audit payload received")
payload = json.loads(text.split("ECNU_AUDIT_JSON_BEGIN",1)[1].split("ECNU_AUDIT_JSON_END",1)[0].strip())
with (HERE/"REMOTE_ACCEPTANCE.json").open("x") as f:
    json.dump(payload["report"],f,indent=2,sort_keys=True,allow_nan=False)
if payload["report"]["status"] != "pass":
    print(json.dumps(payload["report"]))
    raise SystemExit(2)
manifest = []
created = reused = 0
for item in payload["files"]:
    rel = Path(item["relative_path"])
    if rel.is_absolute() or ".." in rel.parts or item["root_key"] not in ROOTS:
        raise RuntimeError("Unsafe artifact path")
    target = ROOTS[item["root_key"]] / rel
    data = base64.b64decode(item["content_base64"],validate=True)
    if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise RuntimeError("Transport SHA mismatch: " + str(rel))
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("Existing local artifact differs; refusing overwrite: " + str(target))
        reused += 1
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("xb") as f:
            f.write(data)
        created += 1
    if target.suffix == ".json":
        json.loads(target.read_text())
    manifest.append({k:v for k,v in item.items() if k != "content_base64"})
receipt = {**META,"status":"pass","local_retrieval_started_utc":started,"local_retrieval_completed_utc":datetime.now(timezone.utc).isoformat(),"remote_verified_at_utc":payload["report"]["remote_verified_at_utc"],"files":manifest,"created":created,"reused_identical":reused,"total_bytes":sum(x["size"] for x in manifest)}
with (HERE/"LOCAL_RETRIEVAL_VERIFICATION.json").open("x") as f:
    json.dump(receipt,f,indent=2,sort_keys=True)
print(json.dumps({k:v for k,v in receipt.items() if k != "files"}))
