"""One explicit user retry after proven pre-SSH access failure; no sbatch."""
import inspect
import json
import re
import shlex
import subprocess
from pathlib import Path

from common import ROOT, REMOTE, m, transport
from release_existing_full import JOBS, PACKAGE_SHA, REMOTE_CODE, dependency_ids


def main():
    blocked=ROOT/'operator/release_existing_full_20261020T031009Z/CONNECTION_BLOCKED.json'
    prior=json.loads(blocked.read_text())
    assert prior['status']=='connection_blocked_before_ssh'
    assert prior['remote_release_attempted'] is False
    assert prior['remote_recovery_reservation_created'] is False
    original=json.loads((ROOT/'operator/full_20261020T030545Z/SUBMISSION_RECEIPT.json').read_text())['evidence']['FAILED.json']
    assert original['jobs_already_created']==JOBS
    code=REMOTE_CODE.replace('__PARSER__',inspect.getsource(dependency_ids))
    for key,value in dict(__ROOT__=str(REMOTE),__JOBS__=JOBS,__PACKAGE_SHA__=PACKAGE_SHA,
                          __ORIGINAL_FAILURE__=original,__OPERATOR_SHA__=m.sha(Path(__file__))).items():
        code=code.replace(key,repr(value))
    compile(code,'<authorized-existing-held-release>','exec')
    access=transport('retry_release_existing_full')
    reservation=ROOT/'EXISTING_FULL_RELEASE_USER_RETRY_RESERVATION_20261020.json'
    m.exclusive(reservation,dict(m.META,jobs=JOBS,utc=m.datetime.now(m.timezone.utc).isoformat(),
        authorization='2026-10-20 user: try again after pre-SSH VPN failure; original five jobs only',
        prior_blocked_sha256=m.sha(blocked),no_sbatch=True,no_automatic_retry=True))
    started=m.datetime.now(m.timezone.utc).isoformat()
    result=None
    access_stage='vpn_start'
    stop_completed=False
    start_result=None
    try:
        start_result=subprocess.run([m.VPN,'start'],env=access.env,capture_output=True,timeout=100)
        flags=re.findall(rb'VPN_SESSION_OWNED=([01])',start_result.stdout)
        assert len(flags)==1,'VPN ownership marker missing or ambiguous; no SSH'
        access.owns_vpn=flags[0]==b'1'
        assert start_result.returncode==0,'VPN start failed; private output suppressed'
        access_stage='vpn_check'
        check=subprocess.run([m.VPN,'check'],env=access.env,capture_output=True,timeout=40)
        assert check.returncode==0,'VPN split check failed; private output suppressed'
        access_stage='remote_audit_release'
        out=access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(code),timeout=120)
        result=json.loads(out.strip().splitlines()[-1])
        result['local_utc']=m.datetime.now(m.timezone.utc).isoformat()
        m.exclusive(access.audit/'SUBMISSION_RECEIPT.json',result)
        print(json.dumps(dict(audit=str(access.audit),**{k:v for k,v in result.items() if k!='evidence'})))
        assert result['status']=='released','Remote audit/release stopped; preserve evidence, no retry'
    except BaseException as error:
        m.exclusive(access.audit/'OPERATION_STOPPED.json',dict(m.META,stage=access_stage,
            local_utc=m.datetime.now(m.timezone.utc).isoformat(),message=str(error),
            remote_operation_attempted=access_stage=='remote_audit_release',
            result_status=result.get('status') if result else 'unconfirmed',
            start_returncode=start_result.returncode if start_result else None,
            vpn_owned_confirmed=access.owns_vpn,no_automatic_retry=True))
        raise
    finally:
        if access.owns_vpn:
            stop=subprocess.run([m.VPN,'stop'],env=access.env,capture_output=True,timeout=40)
            stop_completed=stop.returncode==0
        m.exclusive(access.audit/'LOCAL_OPERATION_RECEIPT.json',dict(m.META,started_utc=started,
            local_completed_utc=m.datetime.now(m.timezone.utc).isoformat(),vpn_owned=access.owns_vpn,
            own_vpn_stop_completed=stop_completed,preexisting_vpn_left_unchanged=not access.owns_vpn,
            result_status=result['status'] if result else 'unconfirmed',stage=access_stage))
        assert not access.owns_vpn or stop_completed,'Shutdown of this operation VPN not confirmed'


if __name__=='__main__':
    main()
