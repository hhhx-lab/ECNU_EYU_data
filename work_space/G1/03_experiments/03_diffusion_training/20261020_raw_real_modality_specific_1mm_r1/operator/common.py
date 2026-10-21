"""Reuse strict SSH transport; preserve VPN ownership per explicit start marker."""
import importlib.util
import json
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[4]
REMOTE = Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_raw_real_modality_specific_1mm_20261020_r1')
spec = importlib.util.spec_from_file_location('raw_access', PROJECT / 'work_space/S2/03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/access.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.ROOT = ROOT


def run_remote(action, code, timeout=120, setup=None):
    access = m.Access(action)
    result = None
    stage = 'vpn_start'
    started = m.datetime.now(m.timezone.utc).isoformat()
    try:
        start = subprocess.run([m.VPN, 'start'], env=access.env, capture_output=True, timeout=100)
        flags = re.findall(rb'VPN_SESSION_OWNED=([01])', start.stdout)
        assert len(flags) == 1, 'VPN ownership unconfirmed; no SSH'
        access.owns_vpn = flags[0] == b'1'
        assert start.returncode == 0, 'VPN start failed; private output suppressed'
        stage = 'vpn_check'
        check = subprocess.run([m.VPN, 'check'], env=access.env, capture_output=True, timeout=40)
        assert check.returncode == 0, 'VPN check failed; no SSH'
        stage = 'remote'
        if setup is not None:
            setup(access)
        out = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=timeout)
        result = json.loads(out.strip().splitlines()[-1])
        result['local_received_utc'] = m.datetime.now(m.timezone.utc).isoformat()
        m.exclusive(access.audit / 'RESULT.json', result)
        print(json.dumps(dict(audit=str(access.audit), status=result.get('status'), remote_utc=result.get('remote_utc'))))
        assert result['status'] == 'pass', 'Remote operation stopped; preserve evidence, no automatic retry'
        return result, access.audit
    except BaseException as error:
        m.exclusive(access.audit / 'OPERATION_STOPPED.json', dict(m.META, stage=stage, message=str(error),
                    utc=m.datetime.now(m.timezone.utc).isoformat(), remote_attempted=stage == 'remote', no_automatic_retry=True))
        raise
    finally:
        stopped = False
        if access.owns_vpn:
            stop = subprocess.run([m.VPN, 'stop'], env=access.env, capture_output=True, timeout=40)
            stopped = stop.returncode == 0
            if not stopped:
                status = subprocess.run([m.VPN, 'status'], env=access.env, capture_output=True, text=True, timeout=40)
                states = [json.loads(line[line.index('{'):]) for line in status.stdout.splitlines() if '{' in line]
                stopped = (status.returncode == 0 and len(states) == 1 and states[0].get('status') == 'logout'
                           and states[0].get('tunnel_status') == 'tunnelUnavailable')
                m.exclusive(access.audit / 'INDEPENDENT_VPN_STATUS.json', dict(m.META,
                            utc=m.datetime.now(m.timezone.utc).isoformat(), original_stop_returncode=stop.returncode,
                            confirmed_logout=stopped, status_returncode=status.returncode))
        m.exclusive(access.audit / 'LOCAL_OPERATION_RECEIPT.json', dict(m.META, started_utc=started,
                    completed_utc=m.datetime.now(m.timezone.utc).isoformat(), vpn_owned=access.owns_vpn,
                    own_vpn_stop_completed=stopped, preexisting_vpn_preserved=not access.owns_vpn,
                    stage=stage, result_status=result.get('status') if result else 'unconfirmed'))
        assert not access.owns_vpn or stopped, 'Own VPN shutdown not confirmed'
