"""Single held CPU evaluation using accepted predictions, no inference or installs."""
import ast
import argparse
import csv
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
from session import session, access_module
from s0_slurm_fields_r2 import job_fields
from submit_direct_full103 import expanded_queue, encode, digest
from audit_local_comparison import K1, E, ROOT
from s0_qos_table import parse_qos_table


R1_PAYLOAD = ROOT/'operator/submit_fixed103_comparison_20261106T102839Z/SUBMISSION_PAYLOAD.json'
R1_FAILURE_SHA = '782e7998d1befb08976e3e4a715878b9cff95655b1382d416ee40df4636c3aa8'


def canonical_qos_names(rows, requested):
    # Same canonicalization as the immutable S0 audit, kept local to this attempt.
    canonical = {}
    for name in rows:
        assert name.lower() not in canonical, 'case-ambiguous QoS names'
        canonical[name.lower()] = name
    assert all(name and not name.startswith(('+', '-')) for name in requested), 'QoS inheritance not resolved'
    assert all(name.lower() in canonical for name in requested), 'applicable QoS absent'
    return list(dict.fromkeys(canonical[name.lower()] for name in requested))


def continuation_payload(original, remote):
    import copy
    payload = copy.deepcopy(original)
    contract = payload['contract']
    parent = remote+'/attempts/compare_2025_K1_E_r1'
    assert contract['attempt'] == parent and contract['remote_root'] == remote
    contract['attempt'] = remote+'/attempts/compare_2025_K1_E_r2'
    for entry in contract['historical'].values():
        entry['path'] = contract['attempt']+'/contract/'+Path(entry['path']).name
    contract['submission_parent'] = dict(attempt=parent,failed_sha256=R1_FAILURE_SHA,
        package_sha256=digest(original['files']['contract/PACKAGE_SHA256.json']),
        reason='QoS_name_canonicalization_only_before_first_sbatch',no_prior_sbatch_required=True)
    payload['files']['contract/COMPARISON_CONTRACT.json'] = encode(contract)
    return payload


def validate_unsubmitted_parent(parent, remote, sha):
    from pathlib import Path
    import json
    root = Path(remote)/'attempts/compare_2025_K1_E_r1'
    assert parent['attempt'] == str(root) and parent['no_prior_sbatch_required']
    failed_path = root/'operator/FAILED_hold.json'
    assert sha(failed_path) == parent['failed_sha256'], 'R1 failure drift'
    failed = json.loads(failed_path.read_text())
    assert failed['stage'] == 'fresh_cpu_resources' and failed['job_id'] is None
    assert failed['status'] == 'comparison_submission_stopped'
    assert not any((root/'operator'/name).exists() for name in (
        'SUBMISSION_RESERVED.json','SBATCH_RESULT.json','HELD_CREATED.json','HELD_AUDIT.json',
        'RELEASE_RESERVED.json','SUBMISSION_RECEIPT.json')), 'R1 submission evidence exists; inspect only'
    package = root/'contract/PACKAGE_SHA256.json'
    assert sha(package) == parent['package_sha256'], 'R1 package drift'
    for name, expected in json.loads(package.read_text())['files'].items():
        assert sha(root/name) == expected, 'R1 frozen source drift'
    return dict(parent_failed_preserved=True,prior_sbatch_evidence_absent=True)


def validate_cpu_held(raw, identity, attempt):
    fields = job_fields(raw)
    expected = dict(JobId=identity,JobName='s2ext25k1e',JobState='PENDING',Reason='JobHeldUser',Priority='0',
        Partition='cpu_192G',Account='zqchen',QOS='user_zqchen',NumCPUs='8',NumTasks='1',Requeue='0',
        Command=attempt+'/slurm/compare.slurm',WorkDir=attempt)
    for key,value in expected.items(): assert fields.get(key)==value,'CPU held identity drift: '+key
    assert fields.get('CPUs/Task')=='8' and fields.get('NumNodes') in ('1','1-1')
    assert fields.get('UserId','').split('(')[0]=='zqchen'
    assert fields.get('TimeLimit') in ('1-00:00:00','24:00:00')
    assert fields.get('MinMemoryNode') in ('64G','65536M','65536')
    assert fields.get('Dependency') in ('(null)','None')
    tres = dict(item.split('=',1) for item in fields.get('TRES',fields.get('ReqTRES','')).split(',') if '=' in item)
    assert not any('gpu' in key and value!='0' for key,value in tres.items()),'CPU comparison requested GPU'
    for key,ext in (('StdOut','out'),('StdErr','err')):
        assert fields.get(key) in (attempt+'/logs/compare_'+identity+'.'+ext,attempt+'/logs/compare_%j.'+ext)
    return fields


def remote_execute(action, payload_path, payload_sha, verified):
    import ast
    import csv
    import hashlib
    import io
    import json
    import os
    from pathlib import Path
    import re
    import shutil
    import subprocess
    from datetime import datetime,timezone
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda:f.read(1048576),b''):h.update(block)
        return h.hexdigest()
    def write(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    assert sha(payload_path)==payload_sha,'comparison payload drift'
    payload=json.loads(Path(payload_path).read_text());contract=payload['contract'];attempt=Path(contract['attempt'])
    stage='before_attempt'
    def command(argv):
        p=subprocess.run(argv,capture_output=True,text=True,timeout=90)
        assert len(p.stdout)+len(p.stderr)<262144,'command output budget'
        if p.returncode:
            write(attempt/('operator/COMMAND_FAILED_'+action+'_'+stage+'.json'),dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
            raise RuntimeError('comparison command failed: '+p.stderr)
        return p.stdout
    def static():
        for name,e in json.loads((attempt/'contract/PACKAGE_SHA256.json').read_text())['files'].items(): assert sha(attempt/name)==e,'comparison package drift'
        for files in list(contract['models'].values())+[contract['references']]:
            assert len(files)==103
            for name,e in files.items():
                p=Path(e['path']);assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],'comparison input drift: '+name
        assert shutil.disk_usage(attempt).free>2*1024**3,'comparison storage unavailable'
        for name,e in contract['evaluator_bin_sha256'].items():assert sha(Path(contract['evaluator_prefix'])/'bin'/name)==e,'evaluator bin drift'
    try:
        if action=='hold':
            assert not attempt.exists(),'comparison attempt exists; no retry'
            expected_name='compare_2025_K1_E_r2' if contract.get('submission_parent') else 'compare_2025_K1_E_r1'
            assert attempt.name==expected_name and attempt.parent==Path(contract['remote_root'])/'attempts'
            queue=command(['squeue','-r','-h','-u','zqchen','-o','%i|%T|%P|%j|%r|%b|%C'])
            jobs=expanded_queue(queue)
            assert not any(row[3]=='s2ext25k1e' for row in jobs),'comparison already queued'
            acct=command(['sacct','-n','-P','-u','zqchen','--name=s2ext25k1e','--starttime=2026-11-06T00:00:00','--format=JobID,JobName,State,ExitCode'])
            assert not acct.strip(),'comparison accounting appeared; inspect only'
            if contract.get('submission_parent'):
                validate_unsubmitted_parent(contract['submission_parent'],contract['remote_root'],sha)
            attempt.mkdir(mode=0o700)
            for folder in ('contract','runtime','slurm','operator','logs','reports','stage','evaluation'):(attempt/folder).mkdir()
            stage='freeze_comparison'
            for name,text in payload['files'].items():
                p=attempt/name;assert p.resolve().is_relative_to(attempt.resolve())
                with p.open('x') as f:f.write(text)
                p.chmod(0o550 if name.endswith('.slurm') else 0o440)
                if name.endswith('.py'):ast.parse(text,feature_version=(3,10))
            static();command(['/bin/bash','-n',str(attempt/'slurm/compare.slurm')])
            stage='fresh_cpu_resources'
            partition=job_fields(command(['scontrol','show','partition','cpu_192G','-o']))
            assert partition['State']=='UP' and (partition['AllowAccounts']=='ALL' or 'zqchen' in partition['AllowAccounts'].split(','))
            qos=parse_qos_table(command(['sacctmgr','-P','show','qos']))
            names=canonical_qos_names(qos,['user_zqchen',partition['QoS']])
            selected=[qos[name] for name in names]
            usage=sum(int(row[6]) for row in jobs)
            for q in selected:
                if q.get('MaxTRESPU'):
                    assert re.fullmatch(r'cpu=\d+',q['MaxTRESPU']) and usage+8<=int(q['MaxTRESPU'].split('=')[1])
                for key in ('MaxJobsPU','MaxSubmitPU','MaxSubmitJobsPU'):
                    if q.get(key):assert len(jobs)+1<=int(q[key]),'CPU comparison QoS job limit'
            write(attempt/'operator/RESOURCE_SNAPSHOT.json',dict(partition=partition,qos=selected,expanded_own_queue=queue,
                prospective_CPU_upper_bound=usage+8,free_bytes=shutil.disk_usage(attempt).free,quota_stdout=command(['quota','-s']),
                remote_utc=datetime.now(timezone.utc).isoformat()))
            stage='single_sbatch_held'
            write(attempt/'operator/SUBMISSION_RESERVED.json',dict(one_submission=True))
            raw=command(['sbatch','--hold','--parsable','--export=NONE','--chdir='+str(attempt),str(attempt/'slurm/compare.slurm')])
            write(attempt/'operator/SBATCH_RESULT.json',dict(stdout=raw,returncode=0))
            match=re.fullmatch(r'(\d+)(?:;[\w.-]+)?',raw.strip());assert match,'ambiguous CPU job ID'
            identity=match.group(1);write(attempt/'operator/HELD_CREATED.json',dict(job_id=identity))
            raw_job=command(['scontrol','show','job',identity,'-o']);fields=validate_cpu_held(raw_job,identity,str(attempt))
            result=dict(status='comparison_submitted_held',job_id=identity,raw_scontrol=raw_job,fields=fields,
                contract_sha256=sha(attempt/'contract/COMPARISON_CONTRACT.json'),remote_utc=datetime.now(timezone.utc).isoformat())
            write(attempt/'operator/HELD_AUDIT.json',result)
        elif action=='release':
            identity=json.loads((attempt/'operator/HELD_CREATED.json').read_text())['job_id']
            assert verified['job_id']==identity and verified['local_held_verified']
            assert verified['held_audit_sha256']==sha(attempt/'operator/HELD_AUDIT.json')
            assert not (attempt/'operator/RELEASE_RESERVED.json').exists(),'CPU release already attempted'
            if contract.get('submission_parent'):
                validate_unsubmitted_parent(contract['submission_parent'],contract['remote_root'],sha)
            static();validate_cpu_held(command(['scontrol','show','job',identity,'-o']),identity,str(attempt))
            write(attempt/'operator/LOCAL_HELD_VERIFICATION.json',verified)
            write(attempt/'operator/RELEASE_RESERVED.json',dict(job_id=identity))
            stage='same_ID_first_release';raw=command(['scontrol','release',identity])
            result=dict(status='comparison_released_not_scoring_complete',job_id=identity,release_returncode=0,stdout=raw,
                observed_job=job_fields(command(['scontrol','show','job',identity,'-o'])),remote_utc=datetime.now(timezone.utc).isoformat(),
                free_bytes=shutil.disk_usage(attempt).free,contract_sha256=sha(attempt/'contract/COMPARISON_CONTRACT.json'))
            write(attempt/'operator/SUBMISSION_RECEIPT.json',result)
        else:raise RuntimeError('invalid comparison action')
        return result
    except BaseException as error:
        result=dict(status='comparison_submission_stopped',stage=stage,job_id=locals().get('identity'),message=str(error),remote_utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False)
        if attempt.exists():write(attempt/('operator/FAILED_'+action+'.json'),result)
        return result


def latest(prefix, filename):
    paths=sorted((ROOT/'operator').glob(prefix+'*/'+filename))
    assert paths,'accepted evidence missing: '+prefix
    return paths[-1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-unsubmitted-r1',action='store_true')
    args=parser.parse_args()
    remote_result=latest('collect_comparison_inputs_r2_','RESULT.json'); accepted=json.loads(remote_result.read_text())
    local_result=latest('audit_local_comparison_','RESULT.json'); local=json.loads(local_result.read_text())
    assert accepted['status']=='comparison_inputs_and_environment_identity_pass'
    assert local['status']=='fixed103_local_identity_pass' and local['issues']==[]
    with session('submit_fixed103_comparison_r2' if args.resume_unsubmitted_r1 else 'submit_fixed103_comparison') as access:
        remote_audit=os.environ['EXTERNAL_MET_REMOTE_ROOT']+'/operator/'+access.audit.name
        access.ssh('mkdir -m 700 '+shlex.quote(remote_audit))
        # Only these already verified historical prediction masks are uploaded, never MRI.
        if not args.resume_unsubmitted_r1:
            access.call(['scp','-r','-P',access.env['HPC_LOGIN_PORT'],'-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
                str(E/'prediction'),access.env['HPC_SSH_USER']+'@'+access.env['HPC_LOGIN_HOST']+':'+remote_audit+'/original_E_prediction'],timeout=240)
        remote=os.environ['EXTERNAL_MET_REMOTE_ROOT'];attempt=remote+'/attempts/compare_2025_K1_E_r1'
        models=dict(EXT2025={name:dict(path=accepted['prediction2025']['path']+'/'+name,**entry) for name,entry in accepted['prediction2025']['files'].items()},
            K1_175=accepted['K1_predictions'],original_E_200={name:dict(path=remote_audit+'/original_E_prediction/'+name,bytes=entry['original_E']['bytes'],sha256=entry['original_E']['sha256']) for name,entry in local['files'].items()})
        rows=list(csv.DictReader((K1/'manifests/FIXED103_REAL_T2W_CASES.csv').open()))
        historical={'K1_175':dict(path=attempt+'/contract/HISTORICAL_K1.csv',sha256=access_module.sha(K1/'cohort/fixed103/models/K1/evaluation/leaderboard_metrics.csv')),
            'original_E_200':dict(path=attempt+'/contract/HISTORICAL_E.csv',sha256=access_module.sha(E/'leaderboard_metrics.csv'))}
        contract=dict(access_module.META,attempt=attempt,remote_root=remote,cases=103,models=models,references=accepted['references'],
            historical=historical,evaluator=accepted['evaluator'],evaluator_bin_sha256=accepted['evaluator_bin_sha256'],
            evaluator_prefix='/public/home/zqchen/.conda/envs/brats_eval',patient_groups={r['source_case_id']+'.nii.gz':r['source_case_id'].rsplit('-',1)[0] for r in rows},
            checkpoint_sha256=dict(K1_175='28e81950b675cd0f5ca26cc1509f9271b2fa827b524f379024b417c0a5402204',original_E_200='4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267'),
            evidence=dict(remote_inputs_sha256=access_module.sha(remote_result),local_E_sha256=access_module.sha(local_result)),
            user_authorization=json.loads((ROOT/'BIO_FULL103_R3_AUTHORIZATION_20261106.json').read_text()),
            bootstrap=dict(iterations=20000,seed=20260831,frame='69_patient_groups_common_finite_per_metric'),
            raw_MRI_read=False,locked_test_read=False,automatic_retry=False,scientific_superiority_pass=False)
        assert len(set(contract['patient_groups'].values()))==69
        files={'contract/COMPARISON_CONTRACT.json':encode(contract),'contract/HISTORICAL_K1.csv':(K1/'cohort/fixed103/models/K1/evaluation/leaderboard_metrics.csv').read_bytes().decode(),
            'contract/HISTORICAL_E.csv':(E/'leaderboard_metrics.csv').read_bytes().decode(),
            'runtime/compare_fixed103.py':(ROOT/'runtime/compare_fixed103.py').read_text(),
            'runtime/run_official_met_eval.sh':(K1/'runtime/run_official_met_eval.sh').read_text()}
        if args.resume_unsubmitted_r1:
            original=json.loads(R1_PAYLOAD.read_text())
            original_members=json.loads(original['files']['contract/PACKAGE_SHA256.json'])['files']
            assert all(digest(original['files'][name])==expected for name,expected in original_members.items()),'local R1 frozen payload drift'
            resumed=continuation_payload(original,remote)
            contract=resumed['contract'];attempt=contract['attempt'];files=resumed['files']
            files.pop('contract/PACKAGE_SHA256.json')
            assert files['runtime/compare_fixed103.py']==(ROOT/'runtime/compare_fixed103.py').read_text(),'runtime changed; not QoS-only continuation'
            assert files['runtime/run_official_met_eval.sh']==(K1/'runtime/run_official_met_eval.sh').read_text(),'official evaluator script drift'
        contract_sha=digest(files['contract/COMPARISON_CONTRACT.json'])
        script='\n'.join(['#!/bin/bash','#SBATCH --job-name=s2ext25k1e','#SBATCH --partition=cpu_192G','#SBATCH --account=zqchen','#SBATCH --qos=user_zqchen',
            '#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=8','#SBATCH --mem=64G','#SBATCH --time=24:00:00','#SBATCH --no-requeue','#SBATCH --export=NONE',
            '#SBATCH --output='+attempt+'/logs/compare_%j.out','#SBATCH --error='+attempt+'/logs/compare_%j.err',
            'set -e','set -o pipefail','umask 077','source /etc/profile','set -u',
            'unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY',
            'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 LC_ALL=C OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8',
            'export COMPARE_EXPECTED_CONTRACT_SHA256='+contract_sha,'cd '+shlex.quote(attempt),
            'exec /public/home/zqchen/.conda/envs/brats_eval/bin/python -B '+shlex.quote(attempt+'/runtime/compare_fixed103.py')+' --root '+shlex.quote(attempt),''])
        check=subprocess.run(['/bin/bash','-n'],input=script,text=True,capture_output=True);assert check.returncode==0,check.stderr
        files['slurm/compare.slurm']=script
        files['contract/PACKAGE_SHA256.json']=encode(dict(access_module.META,files={n:digest(t) for n,t in files.items()}))
        payload=dict(contract=contract,files=files)
        payload_path=access.audit/'SUBMISSION_PAYLOAD.json';access_module.exclusive(payload_path,payload);access.scp(payload_path,remote_audit+'/')
        def invoke(action,verified=None):
            code='import re,csv,io\n'+inspect.getsource(job_fields)+'\n'+inspect.getsource(expanded_queue)
            for helper in (parse_qos_table,canonical_qos_names,validate_unsubmitted_parent,validate_cpu_held,remote_execute):
                code+='\n'+inspect.getsource(helper)
            code+='\nimport json\nprint(json.dumps(remote_execute('+repr(action)+','+repr(remote_audit+'/SUBMISSION_PAYLOAD.json')+','+repr(access_module.sha(payload_path))+','+repr(verified)+')))\n'
            ast.parse(code,feature_version=(3,10));compile(code,'<comparison>','exec')
            access_module.exclusive(access.audit/(action.upper()+'_EXECUTED_SOURCE.json'),dict(source=code,sha256=digest(code)))
            result=json.loads(access.ssh('env LC_ALL=C PYTHONDONTWRITEBYTECODE=1 '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(code),timeout=300).strip().splitlines()[-1])
            access_module.exclusive(access.audit/(action.upper()+'_RESULT.json'),result)
            assert result['status']!='comparison_submission_stopped',result
            return result
        held=invoke('hold');validate_cpu_held(held['raw_scontrol'],held['job_id'],attempt)
        assert held['contract_sha256']==contract_sha
        verified=dict(local_held_verified=True,job_id=held['job_id'],held_audit_sha256=digest(encode(held)))
        access_module.exclusive(access.audit/'LOCAL_HELD_VERIFICATION.json',verified)
        released=invoke('release',verified)
        print(json.dumps(dict(audit=str(access.audit),result=released)),flush=True)


if __name__=='__main__':main()
