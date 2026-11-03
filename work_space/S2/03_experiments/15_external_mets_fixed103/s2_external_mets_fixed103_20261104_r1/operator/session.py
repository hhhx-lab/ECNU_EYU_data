"""Owned aTrust session for bounded, audited authorized S0 writes/transfers."""
from contextlib import contextmanager
import re
import subprocess
import os
import preflight

access_module=preflight.access_module


@contextmanager
def session(action):
    access=access_module.Access(action)
    started=access_module.datetime.now(access_module.timezone.utc).isoformat()
    stop_ok=False
    try:
        p=subprocess.run([os.environ['ECNU_VPN_SCRIPT'],'start'],env=access.env,capture_output=True,timeout=100)
        flags=re.findall(rb'VPN_SESSION_OWNED=([01])',p.stdout)
        if len(flags)!=1:raise RuntimeError('VPN ownership not confirmed; private output suppressed')
        access.owns_vpn=flags[0]==b'1'
        if p.returncode:raise RuntimeError('VPN start failed; no SSH; private output suppressed')
        p=subprocess.run([os.environ['ECNU_VPN_SCRIPT'],'check'],env=access.env,capture_output=True,timeout=40)
        if p.returncode:raise RuntimeError('VPN check failed; no SSH')
        yield access
    except BaseException as error:
        access_module.exclusive(access.audit/'OPERATION_STOPPED.json',dict(access_module.META,
            error_type=type(error).__name__,message=str(error),utc=access_module.datetime.now(access_module.timezone.utc).isoformat(),no_automatic_retry=True))
        raise
    finally:
        if access.owns_vpn:
            p=subprocess.run([os.environ['ECNU_VPN_SCRIPT'],'stop'],env=access.env,capture_output=True,timeout=40)
            stop_ok=p.returncode==0
            if not stop_ok:raise RuntimeError('Own VPN stop not confirmed; do not stop any unrelated session')
        access_module.exclusive(access.audit/'LOCAL_OPERATION_RECEIPT.json',dict(access_module.META,started_utc=started,
            completed_utc=access_module.datetime.now(access_module.timezone.utc).isoformat(),vpn_owned=access.owns_vpn,
            own_vpn_stop_completed=stop_ok,preexisting_vpn_preserved=not access.owns_vpn))
