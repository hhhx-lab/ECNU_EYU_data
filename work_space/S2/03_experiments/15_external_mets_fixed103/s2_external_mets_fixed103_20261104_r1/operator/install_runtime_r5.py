"""Authorized exact-base R5 installation, consuming the successful R4 plan."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shlex
from urllib.parse import urlsplit

import install_runtime_r4 as previous
from session import session, access_module


def official_base_record(record):
    parsed = urlsplit(record['url'])
    assert parsed.scheme == 'https' and not parsed.query and not parsed.fragment, 'base package URL invalid'
    assert '/conda-forge/' in parsed.path, 'base channel not previously approved conda-forge'
    filename = record['fn']
    assert Path(filename).name == filename and filename.endswith(('.conda','.tar.bz2')), 'base archive filename invalid'
    subdir = record['subdir']
    assert subdir in ('linux-64','noarch'), 'base platform drift'
    assert parsed.path.endswith('/' + subdir + '/' + filename), 'base URL filename drift'
    assert re.fullmatch('[a-f0-9]{64}', record['sha256']), 'original base SHA256 required'
    return dict(name=record['name'],version=record['version'],build=record['build'],filename=filename,
                sha256=record['sha256'],url='https://conda.anaconda.org/conda-forge/' + subdir + '/' + filename)


COPY_UNTRACKED = r'''
import hashlib,json,os,shutil,sys
from pathlib import Path
from conda.misc import untracked
old,new=map(Path,sys.argv[1:])
count=0
for name in sorted(untracked(str(old))):
 source,target=old/name,new/name
 assert target.resolve().is_relative_to(new.resolve()),'untracked clone destination escape'
 target.parent.mkdir(parents=True,exist_ok=True)
 if source.is_symlink():
  target.symlink_to(os.readlink(source))
 else:
  data=source.read_bytes()
  try:data=data.decode('utf-8').replace(str(old),str(new)).encode('utf-8')
  except UnicodeDecodeError:pass
  with target.open('xb') as f:f.write(data)
  shutil.copystat(source,target)
 count+=1
print(json.dumps(dict(copied_untracked_files=count,algorithm='Conda_clone_untracked_copy_and_text_prefix_relocation',old_environment_modified=False)))
'''

BASE_PREPARATION = r'''
    stage='exact_base_official_lock_and_downloads'
    from urllib.request import Request,urlopen
    import time
    records=[official_base_record(json.loads(p.read_text())) for p in sorted((old/'conda-meta').glob('*.json'))]
    require(len(records)>10 and len({r['filename'] for r in records})==len(records),'base lock incomplete')
    cache=audit/'base_packages';cache.mkdir(mode=0o700)
    write(audit/'BASE_PACKAGE_LOCK.json',dict(records=records,old_metadata_guard=old_guard,solver_rerun=False))
    deadline=time.monotonic()+1800
    for record in records:
        require(time.monotonic()<deadline,'exact base download deadline')
        target=cache/record['filename']
        with urlopen(Request(record['url'],headers={'User-Agent':'ECNU-approved-runtime-preparation'}),timeout=60) as response,target.open('xb') as output:
            require(response.status==200 and response.geturl().startswith('https://'),'official package response invalid')
            for block in iter(lambda:response.read(1048576),b''):
                require(time.monotonic()<deadline,'exact base download deadline');output.write(block)
        require(sha(target)==record['sha256'],'downloaded exact base SHA drift')
    lock=audit/'BASE_OFFLINE_EXPLICIT.txt'
    with lock.open('x') as f:f.write('@EXPLICIT\n'+''.join((cache/r['filename']).as_uri()+'#'+r['sha256']+'\n' for r in records))
    stage='exact_base_install_offline_no_solver'
    run([conda,'create','--prefix',str(environment),'--file',str(lock),'--copy','--offline','-y'],'base_explicit_install.log',1200,
        {'CONDA_PKGS_DIRS':str(audit/'conda_cache'),'CONDARC':'/dev/null'})
    stage='relocate_existing_pip_base_without_redownload'
    copy_code=COPY_UNTRACKED
    run([str(Path(conda).parent/'python'),'-B','-c',copy_code,str(old),str(environment)],'copy_untracked_base.log',1200)
    observed_base=json.loads(subprocess.check_output([str(environment/'bin/python'),'-B','-c',probe],env=env,text=True,timeout=90))
    require(observed_base==installed,'exact base package inventory changed during relocation')
'''


def remote_code():
    body=previous.BODY.replace("bio_intel_py310_r4", "bio_intel_py310_r5").replace("operator/environment_r4", "operator/environment_r5").replace("logs/environment_r4", "logs/environment_r5")
    body=body.replace('R4 already attempted','R5 already attempted').replace('runtime_R4','runtime_R5')
    body=body.replace("require(root.joinpath('operator/environment_r3/FAILED.json').is_file(),'R3 failed audit absent')",
        "require(root.joinpath('operator/environment_r3/FAILED.json').is_file(),'R3 failed audit absent')\n    require(root.joinpath('operator/environment_r4/FAILED.json').is_file(),'R4 failed audit absent')\n    failures['operator/environment_r4/FAILED.json']=sha(root/'operator/environment_r4/FAILED.json')")
    start=body.index("    stage='exact_dependency_plan_before_environment_creation'")
    end=body.index("    stage='consume_approved_dependencies_and_compiled_wheels'")
    replacement=r'''
    stage='consume_accepted_R4_plan_without_resolving_again'
    prior=root/'operator/environment_r4'
    report=prior/'PIP_INSTALL_PLAN.json'
    require(sha(report)=='f310957e4a49ad06fca73740590da52b4892b0bb9a2765cc719052614783c1da','accepted R4 pip plan identity drift')
    selected=validate_pip_plan(json.loads(report.read_text()),installed,wheels)
    resolved_constraints=audit/'RESOLVED_CONSTRAINTS.txt'
    with resolved_constraints.open('x') as f:f.write(text+'\n'.join(k+'=='+v for k,v in selected.items())+'\n')
    write(audit/'PLAN_ACCEPTANCE.json',dict(meta,selected=selected,installed_base=installed,plan_sha256=sha(report),
        compiled_wheels=wheels,old_guard=old_guard,prior_failure_guard=failures,remote_utc=datetime.now(timezone.utc).isoformat(),
        status='previous_R4_exact_dependency_plan_consumed_without_resolver'))
'''+BASE_PREPARATION
    body=body[:start]+replacement+body[end:]
    code='DIRECT='+repr(previous.DIRECT)+'\nCOPY_UNTRACKED='+repr(COPY_UNTRACKED)+'\nimport re\nfrom pathlib import Path\nfrom urllib.parse import urlsplit\n'
    code+=inspect.getsource(official_base_record)+'\n'+inspect.getsource(previous.validate_pip_plan)+'\n'+body
    compile(code,'approved_exact_runtime_R5','exec')
    return code


def main():
    code=remote_code()
    authorization=dict(consume_previously_approved_dependencies_and_wheels=True,independent_R5_user_authorized=True,
        exact_official_base_packages_only=True,consume_successful_R4_pip_plan=True,solver_rerun=False,
        preserve_failed_R4=True,no_separate_GPU_smoke=True,direct_full103_after_environment_acceptance=True)
    with session('install_runtime_r5') as access:
        access_module.exclusive(access.audit/'EXECUTED_SOURCE.json',dict(source=code,sha256=hashlib.sha256(code.encode()).hexdigest()))
        access_module.exclusive(access.audit/'AUTHORIZATION.json',authorization)
        command='env LC_ALL=C PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' R4_RUNTIME_AUTHORIZATION='+shlex.quote(json.dumps(authorization))+' R4_RUNTIME_SOURCE_SHA256='+hashlib.sha256(code.encode()).hexdigest()
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(code)
        result=json.loads(access.ssh(command,timeout=7800).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),result=result)),flush=True)
        if result['status']=='runtime_R5_preparation_failed':raise RuntimeError('R5 failed; preserve without automatic retry')


if __name__=='__main__':main()
