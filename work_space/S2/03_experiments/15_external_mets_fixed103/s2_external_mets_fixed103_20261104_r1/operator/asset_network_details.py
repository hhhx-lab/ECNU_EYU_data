"""Diagnose direct IPv4 asset access once; no proxy, TLS bypass or downloads."""
import preflight

preflight.CODE = r'''
from datetime import datetime,timezone
import json,os,socket,subprocess

def command(argv):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=18)
        return dict(returncode=p.returncode,stdout=p.stdout[:4000],stderr=p.stderr[:2000])
    except (OSError,subprocess.TimeoutExpired) as e:
        return dict(returncode=124 if isinstance(e,subprocess.TimeoutExpired) else 127,error_type=type(e).__name__)
out=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed',
    status='direct_ipv4_asset_diagnosis',remote_started_utc=datetime.now(timezone.utc).isoformat(),dns={},probes={})
urls={
    'bio':'https://huggingface.co/keramik05/MICCAI-METS_challenge_bi-intel_model/resolve/4a8291024dafb79b4c0cc381438e976a36072201/weights.zip',
    'brainiac':'https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AMblt6reQVvlSrORTB3_2lE/BrainIAC.ckpt?rlkey=9w55le6tslwxlfz6c0viylmjb&dl=1',
    'docker':'https://auth.docker.io/token?service=registry.docker.io&scope=repository:brainles/brats25_met_gli-team:pull',
}
for host in ['huggingface.co','www.dropbox.com','auth.docker.io']:
    try:out['dns'][host]=sorted({item[4][0] for item in socket.getaddrinfo(host,443,type=socket.SOCK_STREAM)})
    except OSError as error:out['dns'][host]=dict(error_type=type(error).__name__)
for name,url in urls.items():
    out['probes'][name]=command(['curl','-4','-I','--connect-timeout','8','--max-time','12','--noproxy','*',url])
out['proxy_environment_present']={key:key in os.environ for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']}
out['scope']=dict(remote_writes=False,binary_download=False,tls_validation_disabled=False,proxy_created=False,
    namespace_or_firewall_changed=False,jobs_created=False)
out['remote_utc']=datetime.now(timezone.utc).isoformat()
print(json.dumps(out))
'''

if __name__ == '__main__':
    preflight.main()
