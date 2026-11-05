"""Bounded bootstrap/CLI review of failed R2; no jobs, installs or asset writes."""
import hashlib
import json
import os
import re
import shlex

from session import session, access_module
from s0_cpu_r2_sources import ROOT, runtime_files

SLURM_SHA = 'd879a1addf9756b3bfa8561b96e33f600deb87c1b14f9483e79217d8506620ff'
CONTRACT_SHA = '8a2246326f58d7351712785e2395c54fb81949ad7a4390f19e18e80980329980'
RECEIPT_SHA = '592c93f24b50993446ff38210c5e0ef352a554beb7c0c3dea147b41ee4f71eda'


def corrected_initialization(script):
    assert script.count('set -euo pipefail\n') == 1
    assert script.count('module load singularity-ce/4.1.1\n') == 1
    assert script.index('set -euo pipefail') < script.index('source /etc/profile')
    result = script.replace('set -euo pipefail\n', 'set -eo pipefail\n', 1)
    return result.replace('module load singularity-ce/4.1.1\n',
                          'module load singularity-ce/4.1.1\nset -u\n', 1)


def bootstrap_only(script, probe):
    lines = script.splitlines()
    payloads = [i for i, line in enumerate(lines) if line.startswith('exec "$CONDA_PREFIX/bin/python" -B ')]
    assert len(payloads) == 1 and payloads[0] == len(lines)-1
    return '\n'.join(lines[:payloads[0]]+[
        'exec "$CONDA_PREFIX/bin/python" -B -c '+shlex.quote(probe), ''])


def assess_observations(observations):
    original, corrected = [observations[name] for name in ('original', 'corrected')]
    optional = re.search(
        r'/etc/(?:profile|profile\.d/256term\.sh): line \d+: '
        r'(HISTCONTROL|COLORTERM|XTERM_VERSION|ROXTERM_ID|KONSOLE_DBUS_SESSION): unbound variable',
        original['stderr'],
    )
    probe = corrected.get('probe')
    valid_probe = isinstance(probe, dict) and probe.get('status') == 'bootstrap_import_compiler_CLI_probe_pass_not_artifact_pass'
    if valid_probe:
        valid_probe = all(probe.get(name) is False for name in
                          ('images_read', 'models_loaded', 'packages_installed', 'assets_written'))
        commands = probe.get('commands', {})
        expected = {'import_'+name: 0 for name in
                    ('numpy', 'pip', 'setuptools', 'wheel', 'Cython', 'tomli', 'packaging')}
        expected.update({name: 0 for name in ('pip_wheel_cli', 'libc', 'CC_version', 'CC_syntax',
                        'CXX_version', 'CXX_syntax', 'singularity_version', 'singularity_build_help',
                        'singularity_sif_add_help', 'singularity_sif_new_help', 'readelf_version')})
        valid_probe = valid_probe and all(commands.get(name, {}).get('returncode') == code
                                         for name, code in expected.items())
        valid_probe = valid_probe and commands.get('mksquashfs_help', {}).get('returncode') in (0, 1)
    return dict(pass_bootstrap_only=bool(original['returncode'] != 0 and optional
                                        and corrected['returncode'] == 0 and valid_probe),
                original_optional_variable=optional.group(1) if optional else None,
                full_preparation_pass=False, GPU_pass=False)


PROBE = r'''
import importlib.metadata as metadata,json,os,shlex,shutil,subprocess,sys
from pathlib import Path
prefix=Path(os.environ['CONDA_PREFIX']).resolve()
assert prefix.name=='bio_build_glibc217_r2'
assert Path(sys.executable).resolve()==(prefix/'bin/python').resolve()
assert sys.version_info[:2]==(3,10)
versions={n:metadata.version(n) for n in ('numpy','pip','setuptools','wheel','Cython','tomli','packaging')}
assert versions['numpy']=='2.2.6'
assert os.environ['PIP_NO_INDEX']=='1' and os.environ['PIP_CONFIG_FILE']=='/dev/null'
assert not os.environ.get('CUDA_VISIBLE_DEVICES')
commands={}
def call(name,argv,expected=(0,),input=None):
 p=subprocess.run(argv,input=input,capture_output=True,text=True,timeout=30)
 assert len(p.stdout.encode())+len(p.stderr.encode())<65536
 commands[name]=dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 assert p.returncode in expected, 'bootstrap probe failed: '+name
 return p.stdout+p.stderr
for name in ('numpy','pip','setuptools','wheel','Cython','tomli','packaging'):
 call('import_'+name,[sys.executable,'-B','-c','import '+name])
call('pip_wheel_cli',[sys.executable,'-B','-m','pip','wheel','--help'])
assert call('libc',['getconf','GNU_LIBC_VERSION']).strip()=='glibc 2.17'
for variable,lang,source in [('CC','c','int main(void){return 0;}'),
                            ('CXX','c++','#include <vector>\nint main(){std::vector<int> x;return x.size();}')]:
 args=shlex.split(os.environ[variable])
 assert len(args)==1
 binary=shutil.which(args[0])
 assert binary and Path(binary).resolve().is_relative_to(prefix)
 text=call(variable+'_version',[binary,'--version'])
 assert '13.' in text
 call(variable+'_syntax',[binary,'-x',lang,'-fsyntax-only','-'],input=source)
backend='/public/software/singularity-ce/4.1.1/bin/singularity'
assert call('singularity_version',[backend,'version']).strip()=='4.1.1'
build=call('singularity_build_help',[backend,'build','--help'])
assert all(x in build for x in ('--sandbox','--disable-cache','--no-cleanup'))
add=call('singularity_sif_add_help',[backend,'sif','add','--help'])
assert all(x in add for x in ('--datatype','--parttype','--partfs','--partarch'))
call('singularity_sif_new_help',[backend,'sif','new','--help'])
squash=call('mksquashfs_help',['/usr/sbin/mksquashfs','-help'],expected=(0,1))
assert all(x in squash for x in ('-read-queue','-write-queue','-fragment-queue','-processors','-noappend','-all-root'))
readelf=prefix/'bin/x86_64-conda-linux-gnu-readelf'
assert readelf.is_file()
call('readelf_version',[str(readelf),'--version'])
print('S0_BOOTSTRAP_PROBE='+json.dumps(dict(status='bootstrap_import_compiler_CLI_probe_pass_not_artifact_pass',
 interpreter=str(Path(sys.executable).resolve()),versions=versions,commands=commands,
 images_read=False,models_loaded=False,packages_installed=False,assets_written=False)))
'''


CODE = r'''
import os,subprocess
from datetime import datetime,timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
attempt=root/'attempts/s0_cpu_prepare_r2'
script=(attempt/'slurm/prepare.slurm').read_text()
require(sha(attempt/'slurm/prepare.slurm')==SLURM_SHA,'failed R2 Slurm drift')
require(sha(attempt/'contract/PREPARATION_CONTRACT.json')==CONTRACT_SHA,'failed R2 contract drift')
require(sha(attempt/'operator/submit_usage_r2/SUBMISSION_RECEIPT.json')==RECEIPT_SHA,'failed R2 receipt drift')
contract=json.loads((attempt/'contract/PREPARATION_CONTRACT.json').read_text())
before=prior_guard(root)
parent_before=parent_staging_guard(root)
require(before==contract['prior_guard'] and parent_before==contract['parent_attempt_files'],'prior identity drift')
for name,digest in contract['build_env_files'].items():
 require(sha(Path(contract['build_env'])/name)==digest,'build environment drift: '+name)
terminal=subprocess.run(['sacct','-P','-j','4304740','--format=JobID,State,ExitCode,Elapsed'],
 capture_output=True,text=True,timeout=30)
require(terminal.returncode==0 and '4304740|FAILED|1:0|00:00:21' in terminal.stdout,'failed R2 terminal changed')
profile=Path('/etc/profile.d/256term.sh')
profile_lines=profile.read_text().splitlines()
profile_context=[dict(line=i+1,text=line) for i,line in enumerate(profile_lines) if 'COLORTERM' in line]
clean={'HOME':os.environ['HOME'],'USER':'zqchen','LOGNAME':'zqchen','SHELL':'/bin/bash',
       'PATH':'/usr/bin:/bin','LANG':'C','PYTHONDONTWRITEBYTECODE':'1'}
observations={}
for name,source in [('original',script),('corrected',corrected_initialization(script))]:
 bootstrap=bootstrap_only(source,PROBE)
 p=subprocess.run(['/bin/bash','-s'],input=bootstrap,env=clean,
                  capture_output=True,text=True,timeout=180)
 require(len(p.stdout.encode())+len(p.stderr.encode())<131072,'bootstrap review output budget')
 marker=[line[len('S0_BOOTSTRAP_PROBE='):] for line in p.stdout.splitlines() if line.startswith('S0_BOOTSTRAP_PROBE=')]
 observations[name]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
   probe=json.loads(marker[0]) if len(marker)==1 else None,
   bootstrap_sha256=hashlib.sha256(bootstrap.encode()).hexdigest())
require(prior_guard(root)==before and parent_staging_guard(root)==parent_before,'prior identities changed during probe')
require(sha(attempt/'slurm/prepare.slurm')==SLURM_SHA and sha(attempt/'contract/PREPARATION_CONTRACT.json')==CONTRACT_SHA,
 'failed R2 files changed during probe')
assessment=assess_observations(observations)
ok=assessment['pass_bootstrap_only']
out=dict(META,status='real_bootstrap_probe_pass_not_full_preparation_pass' if ok else 'real_bootstrap_probe_requires_review_no_submission',
 observations=observations,assessment=assessment,failed_job_terminal=terminal.stdout,profile_context=profile_context,
 slurm_sha256=SLURM_SHA,contract_sha256=CONTRACT_SHA,receipt_sha256=RECEIPT_SHA,
 prior_environment_and_failures_preserved=True,remote_utc=datetime.now(timezone.utc).isoformat(),
 jobs_created=False,environment_reinstalled=False,solver_rerun=False,assets_written=False,images_read=False)
print(json.dumps(out))
'''


def remote_code():
    import inspect
    source = runtime_files()['s0_build_contract.py']+'\n'
    source += inspect.getsource(corrected_initialization)+'\n'+inspect.getsource(bootstrap_only)+'\n'
    source += inspect.getsource(assess_observations)+'\n'
    source += 'import re,shlex\nSLURM_SHA='+repr(SLURM_SHA)+'\nCONTRACT_SHA='+repr(CONTRACT_SHA)
    source += '\nRECEIPT_SHA='+repr(RECEIPT_SHA)+'\nPROBE='+repr(PROBE)+'\n'+CODE
    compile(source,'bounded_CPU_bootstrap_review','exec')
    return source


def verify_saved_review(directory):
    from datetime import datetime, timezone
    from pathlib import Path
    directory = Path(directory).resolve()
    assert directory.parent == ROOT/'operator' and directory.name.startswith('review_s0_cpu_preparation_')
    path = directory/'RESULT.json'
    result = json.loads(path.read_text())
    assert result['slurm_sha256'] == SLURM_SHA and result['contract_sha256'] == CONTRACT_SHA
    assert result['receipt_sha256'] == RECEIPT_SHA
    assert result['prior_environment_and_failures_preserved'] is True
    assert all(result[name] is False for name in ('jobs_created', 'environment_reinstalled',
                                                'solver_rerun', 'assets_written', 'images_read'))
    assessment = assess_observations(result['observations'])
    payload = dict(access_module.META, **assessment, original_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   remote_utc=result['remote_utc'], local_utc=datetime.now(timezone.utc).isoformat(),
                   original_result_preserved=True, new_remote_execution=False, jobs_created=False,
                   explanation='Clean shell fails earlier at HISTCONTROL; actual Slurm failed at COLORTERM. '
                               'Both are optional profile variables, not a successful original bootstrap.')
    access_module.exclusive(directory/'BOOTSTRAP_VERIFICATION.json', payload)
    print(json.dumps(payload))
    assert assessment['pass_bootstrap_only'], 'saved bootstrap observations did not pass'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-saved')
    args = parser.parse_args()
    if args.verify_saved:
        verify_saved_review(args.verify_saved)
        return
    source = remote_code()
    with session('review_s0_cpu_preparation') as access:
        access_module.exclusive(access.audit/'SOURCE_IDENTITY.json',dict(
            remote_code_sha256=hashlib.sha256(source.encode()).hexdigest(),
            slurm_sha256=SLURM_SHA,contract_sha256=CONTRACT_SHA,receipt_sha256=RECEIPT_SHA))
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(source)
        result=json.loads(access.ssh(command,timeout=500).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],remote_utc=result['remote_utc'])))


if __name__=='__main__':
    main()
