"""Offline command-order guards; no ECNU connection or scheduler mutation."""
import ast
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


class SubmissionTests(unittest.TestCase):
    def execute(self, scenario='pass'):
        local=Path(__file__).resolve().parents[1]
        tree=ast.parse((local/'operator/submit_scan.py').read_text())
        code=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'root';parent=Path(temp)/'parent'
            (root/'evidence').mkdir(parents=True);(root/'logs').mkdir()
            (parent/'plan').mkdir(parents=True);(parent/'dataset').mkdir()
            (parent/'plan/PACKAGE_MANIFEST.json').write_text(json.dumps({'files':{}}))
            (parent/'dataset/FAILED.json').write_text('{}')
            source=Path(temp)/'source';(source/'data').mkdir(parents=True)
            (source/'data/data_csv.csv').touch()
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            cfg={'parent_package_sha256':digest(parent/'plan/PACKAGE_MANIFEST.json'),
                 'source_code':str(source),'source_csv_sha256':hashlib.sha256(b'').hexdigest()}
            files={'config/SCAN.json':json.dumps(cfg).encode(),'slurm/scan.slurm':b'#!/bin/bash\n',
                   'runtime/scan_coverage.py':b'MODS=("t1n","t1c","t2f")\n',
                   'tests/test_support_audit.py':b''}
            package={'files':{p:hashlib.sha256(b).hexdigest() for p,b in files.items()}}
            files['plan/PACKAGE_MANIFEST.json']=json.dumps(package).encode()
            payload=root/'evidence/PAYLOAD.json'
            payload.write_text(json.dumps({'files_b64':{p:base64.b64encode(b).decode() for p,b in files.items()}}))
            for k,v in {'__ROOT__':str(root),'__PARENT__':str(parent),'__PAYLOAD_SHA__':digest(payload),
                        '__PACKAGE_SHA__':hashlib.sha256(files['plan/PACKAGE_MANIFEST.json']).hexdigest(),'__PYTHON__':'python'}.items():
                code=code.replace(k,repr(v))
            calls=[];held=[True]

            def fake_run(a,**kwargs):
                calls.append(a);out='';err='';rc=0
                if a[0]=='sacct':
                    out='4086739|FAILED|1:0|00:05:46\n'+''.join(str(j)+'|CANCELLED|0:0|00:00:00\n' for j in range(4069916,4069920))
                elif a[:3]==['scontrol','show','partition']:
                    out='State=UP ReqResv=NO MaxTime=UNLIMITED AllowAccounts=zqchen AllowGroups=ALL AllowQos=ALL Nodes=node201 QoS=partition_cpu_192G'
                elif a[:3]==['scontrol','show','hostnames']:out='node201\n'
                elif a[:3]==['scontrol','show','node']:
                    out='NodeName=node201 RealMemory='+('95051' if scenario=='capacity' else '190913')+' MemSpecLimit=10240 CPUTot=28 State=MIXED'
                elif a[0]=='sacctmgr':out='user_zqchen|||||1000|\npartition_cpu_192g||||||\n'
                elif a[0]=='bash':out='G1_MASK_SCAN_R6_BASH_U_PASS\n'
                elif a[0]=='sbatch':
                    if scenario=='qos':rc=1;err='QOSMaxSubmitJobPerUserLimit'
                    else:out='999002\n'
                elif a[:3]==['scontrol','show','job']:
                    out=('JobName=g1dmaskr6 JobState=PENDING Reason='+('JobHeldUser' if held[0] else 'Priority')+
                         ' Partition=cpu_192G NumCPUs=8 CPUs/Task=8 MinMemoryNode=90G TimeLimit=1-00:00:00 '
                         'Dependency=(null) NumTasks=1 Account=zqchen Command='+str(root/'slurm/scan.slurm')+
                         ' StdOut='+str(root/'logs/g1dmaskr6_999002.out')+' StdErr='+str(root/'logs/g1dmaskr6_999002.err'))
                    if scenario=='held_drift':out=out.replace('MinMemoryNode=90G','MinMemoryNode=32G')
                elif a[:2]==['scontrol','release']:
                    if scenario=='release_denied':rc=1;err='Access denied'
                    else:held[0]=False
                return subprocess.CompletedProcess(a,rc,stdout=out,stderr=err)

            disk=type('Disk',(),{'f_bavail':223836372992,'f_frsize':1})()
            with patch('subprocess.run',side_effect=fake_run),patch('os.statvfs',return_value=disk):
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    exec(compile(code,'offline_scan_submit','exec'),{})
            for p in (root/'evidence').glob('*.json'):json.loads(p.read_text())
            return json.loads(out.getvalue().strip().splitlines()[-1]),calls,held[0]

    def test_unique_cpu_held_then_release(self):
        r,calls,held=self.execute()
        self.assertEqual(r['status'],'scan_submitted_released');self.assertFalse(held)
        self.assertEqual(r['training_jobs_submitted'],[])
        self.assertEqual(sum(a[0]=='sbatch' for a in calls),1)
        self.assertEqual(sum(a[:2]==['scontrol','release'] for a in calls),1)
        self.assertFalse(any(a[0]=='scancel' for a in calls))

    def test_capacity_before_submit(self):
        r,calls,_=self.execute('capacity');self.assertEqual(r['stage'],'precheck')
        self.assertFalse(any(a[0]=='sbatch' for a in calls))

    def test_qos_once_no_release(self):
        r,calls,_=self.execute('qos');self.assertEqual(r['stage'],'submit')
        self.assertEqual(sum(a[0]=='sbatch' for a in calls),1)
        self.assertFalse(any(a[:2]==['scontrol','release'] for a in calls))

    def test_held_drift_no_release(self):
        r,calls,held=self.execute('held_drift');self.assertEqual(r['status'],'stopped');self.assertTrue(held)
        self.assertFalse(any(a[:2]==['scontrol','release'] for a in calls))

    def test_release_once_on_rejection(self):
        r,calls,held=self.execute('release_denied');self.assertEqual(r['stage'],'release');self.assertTrue(held)
        self.assertEqual(sum(a[:2]==['scontrol','release'] for a in calls),1)


if __name__=='__main__':unittest.main(verbosity=2)
