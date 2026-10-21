"""Bounded VPN-owned access; never repeat submissions or poll in a loop."""

import argparse
import importlib.util
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCESS = ROOT.parent / 's2_met_aug_k1245_20260923_r3_fallback/operator/access.py'
REMOTE = '/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/' + ROOT.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('stage', 'submit-after-timeout'))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('k1245_existing_access', ACCESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    access = mod.Access('k1_resume_' + args.action.replace('-', '_'))
    started = datetime.now(timezone.utc).isoformat()
    succeeded = False
    try:
        access.start_vpn()
        if args.action == 'stage':
            access.ssh('set -e; test ! -e ' + shlex.quote(REMOTE) +
                       '; mkdir -m 700 ' + shlex.quote(REMOTE))
            for name in ('EXECUTION_PLAN.json', 'PACKAGE_SHA256.json', 'README.md'):
                access.scp(ROOT / name, REMOTE + '/')
            for name in ('runtime', 'slurm', 'operator'):
                access.call(['scp', '-r', '-P', access.env['HPC_LOGIN_PORT'], '-o',
                             'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15', str(ROOT / name),
                             f'{access.env["HPC_SSH_USER"]}@{access.env["HPC_LOGIN_HOST"]}:{REMOTE}/'])
            access.ssh('mkdir -m 700 ' + shlex.quote(REMOTE + '/evidence') + ' ' + shlex.quote(REMOTE + '/logs'))
        command = 'set -e; export RESUME_ROOT=' + shlex.quote(REMOTE) + '; source "$RESUME_ROOT/runtime/research_environment.sh"; '
        command += shlex.quote(mod.PYTHON) + ' -B "$RESUME_ROOT/runtime/control_resume.py" ' + shlex.quote(args.action)
        raw = access.ssh(command, timeout=360)
        prefix = 'K1_RESUME_OPERATION_JSON='
        lines = [line for line in raw.splitlines() if line.startswith(prefix)]
        if len(lines) != 1:
            raise RuntimeError('missing structured remote result')
        report = json.loads(lines[0][len(prefix):])
        report['local_utc'] = datetime.now(timezone.utc).isoformat()
        mod.exclusive(access.audit / 'OPERATION_RESULT.json', report)
        # Only small JSON evidence is transferred; never pull checkpoints or source events.
        collect = 'import json,hashlib,pathlib; root=pathlib.Path(' + repr(REMOTE) + '); '
        collect += 'print(json.dumps({str(p.relative_to(root)):{"text":p.read_text(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in (root/"evidence").glob("*.json")}))'
        raw = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + mod.PYTHON + ' -B -c ' + shlex.quote(collect))
        files = json.loads(raw[raw.find('{'):])
        import hashlib
        for relative, entry in files.items():
            payload = entry['text'].encode()
            if hashlib.sha256(payload).hexdigest() != entry['sha256']:
                raise RuntimeError('pulled evidence SHA mismatch')
            json.loads(entry['text'])
            path = access.audit / 'pulled' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                handle.write(payload)
        mod.exclusive(access.audit / 'PULLED_SHA256.json', dict(mod.META,
                      files={relative: value['sha256'] for relative, value in files.items()}))
        succeeded = True
        print(json.dumps(dict(audit=str(access.audit), result=report), ensure_ascii=False))
    finally:
        owned = access.owns_vpn
        access.close_vpn()
        mod.exclusive(access.audit / 'LOCAL_OPERATION_RECEIPT.json', dict(mod.META,
                      local_started_utc=started, local_completed_utc=datetime.now(timezone.utc).isoformat(),
                      operation_succeeded=succeeded, vpn_started_by_this_operation=owned,
                      vpn_closed=owned, preexisting_vpn_left_unchanged=not owned))


if __name__ == '__main__':
    main()
