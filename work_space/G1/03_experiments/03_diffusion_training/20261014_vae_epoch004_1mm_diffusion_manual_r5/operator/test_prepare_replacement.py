"""Offline replacement ordering and failure guards; no server connection."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


class ReplacementTests(unittest.TestCase):
    def execute(self, scenario='pass'):
        tree = ast.parse(Path(__file__).with_name('replace_prepare_cpu192.py').read_text())
        code = next(n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'code' for t in n.targets))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'root'
            source = Path(temp) / 'source'
            for name in ('plan', 'slurm', 'logs', 'evidence'):
                (root / name).mkdir(parents=True)
            (source / 'data').mkdir(parents=True)
            (source / 'data/data_csv.csv').touch()
            files = {}
            for i in range(57):
                name = 'slurm/prepare.slurm' if i == 0 else 'plan/file'+str(i)
                content = '--partition=cpu_96G g1d1m5p1' if i == 0 else ''
                (root / name).write_text(content)
                files[name] = hashlib.sha256(content.encode()).hexdigest()
            (root / 'plan/PACKAGE_MANIFEST.json').write_text(json.dumps({'files': files}))
            auth = {'artifact_status': 'experimental_unvalidated', 'operator_approved': False,
                    'formal_gate_status': 'not_run_not_passed', 'old_prepare_job': '4069915',
                    'gpu_jobs': ['4069916', '4069917', '4069918', '4069919'],
                    'original_package_sha256': hashlib.sha256((root / 'plan/PACKAGE_MANIFEST.json').read_bytes()).hexdigest(),
                    'script': '--partition=cpu_192G g1d1m5p2',
                    'resource': {'partition': 'cpu_192G', 'cpus': 8, 'mem': '90G', 'walltime': '24:00:00'}}
            for key, value in {'__ROOT__': str(root), '__SOURCE__': str(source), '__AUDIT_NAME__': 'mock'}.items():
                code = code.replace(key, repr(value))
            code = code.replace('__AUTH__', repr(auth)).replace(
                '0d1765e12cdad7d8a5df57a5d2c46c79414ed99a2015a4d8fed57cd4c4fc836b', hashlib.sha256(b'').hexdigest())
            state = {'held': True, 'cancelled': False, 'deps': {j: '4069915' for j in auth['gpu_jobs']}}
            calls = []

            def job_text(j):
                if j in state['deps']:
                    producer = '99' if scenario == 'dependency_drift' else state['deps'][j]
                    return ('JobId='+j+' JobState=PENDING RunTime=00:00:00 Account=zqchen Partition=a100 '
                            'Dependency=afterok:'+producer+'(unfulfilled) NumCPUs=8 MinMemoryNode=64G '
                            'TimeLimit=2-00:00:00 TRES=cpu=8,mem=64G,gres/gpu=1 KillOInInvalidDependent=Yes '
                            'Command='+str(root / 'slurm/t1c.slurm'))
                old = j == '4069915'
                prefix = 'g1d1m5p1' if old else 'g1d1m5p2'
                path = root / 'slurm/prepare.slurm' if old else root / 'evidence/mock/prepare_cpu192.slurm'
                return ('JobId='+j+' JobName='+prefix+' Account=zqchen Partition=cpu_192G NumCPUs=8 '
                        'NumTasks=1 CPUs/Task=8 MinMemoryNode=90G TimeLimit=1-00:00:00 NumNodes=1-1 '
                        'Dependency=(null) Features=(null) RunTime=00:00:00 Command='+str(path)+
                        ' StdOut='+str(root / 'logs' / (prefix+'_'+j+'.out'))+
                        ' StdErr='+str(root / 'logs' / (prefix+'_'+j+'.err'))+
                        ' JobState='+('CANCELLED' if old and state['cancelled'] else 'PENDING')+
                        ' Reason='+('JobHeldAdmin' if old else 'JobHeldUser' if state['held'] else 'Priority'))

            def fake_run(args, **kwargs):
                calls.append(args)
                out, err, rc = '', '', 0
                if args[:3] == ['scontrol', 'show', 'job']:
                    out = job_text(args[3])
                elif args[:3] == ['scontrol', 'show', 'partition']:
                    out = ('State=UP MaxTime=UNLIMITED ReqResv=NO AllowAccounts=zqchen AllowGroups=ALL '
                           'AllowQos=ALL Nodes=node201 QoS=partition_cpu_192g')
                elif args[:3] == ['scontrol', 'show', 'hostnames']:
                    out = 'node201\n'
                elif args[:3] == ['scontrol', 'show', 'node']:
                    memory = 95051 if scenario == 'capacity' else 190913
                    out = 'NodeName=node201 RealMemory='+str(memory)+' MemSpecLimit=10240 CPUTot=28 State=MIXED\n'
                elif args[0] == 'sacctmgr':
                    out = 'user_zqchen||||||1000|\npartition_cpu_192g|||||||\n'
                elif args[0] == 'bash':
                    out = 'G1_R5_BASH_U_DRY_RUN_PASS\n'
                elif args[0] == 'sbatch':
                    if scenario == 'qos':
                        rc, err = 1, 'QOSMaxSubmitJobPerUserLimit'
                    else:
                        out = '999001\n'
                elif args[:2] == ['scontrol', 'update']:
                    j = args[2].split('=')[1]
                    if scenario == 'partial_dependency' and j == '4069917':
                        rc, err = 1, 'Access denied'
                    else:
                        state['deps'][j] = args[3].split(':')[1]
                elif args[0] == 'scancel':
                    if scenario == 'cancel_denied':
                        rc, err = 1, 'Access denied'
                    else:
                        state['cancelled'] = True
                elif args[0] == 'sacct':
                    out = '4069915|CANCELLED by 101|0:0|00:00:00\n' if state['cancelled'] else ''
                elif args[:2] == ['scontrol', 'release']:
                    if scenario == 'release_denied':
                        rc, err = 1, 'Access denied'
                    else:
                        state['held'] = False
                return subprocess.CompletedProcess(args, rc, stdout=out, stderr=err)

            if scenario == 'package_drift':
                (root / 'slurm/prepare.slurm').write_text('drift')
            if scenario == 'existing_output':
                (root / 'dataset').mkdir()
            if scenario == 'reserved':
                (root / 'evidence/PREPARE_REPLACEMENT_RESERVATION.json').write_text('{}')
            disk = type('Disk', (), {'f_bavail': 224162480128, 'f_frsize': 1})()
            with patch('subprocess.run', side_effect=fake_run), patch('os.statvfs', return_value=disk):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    exec(compile(code, '<replacement_mock>', 'exec'), {})
            for p in (root / 'evidence').rglob('*.json'):
                json.loads(p.read_text())
            return json.loads(output.getvalue().strip().splitlines()[-1]), calls, state

    def test_success_order(self):
        result, calls, state = self.execute()
        self.assertEqual(result['status'], 'replacement_released_dependencies_migrated_old_retired')
        mutations = [a for a in calls if a[0] in ('sbatch', 'scancel') or a[:2] in
                     (['scontrol', 'update'], ['scontrol', 'release'])]
        self.assertEqual(mutations[0][:4], ['sbatch', '--parsable', '--hold', '--export=NONE'])
        self.assertEqual([a[2] for a in mutations[1:5]], ['JobId='+str(j) for j in range(4069916, 4069920)])
        self.assertEqual(mutations[5:], [['scancel', '4069915'], ['scontrol', 'release', '999001']])
        self.assertTrue(state['cancelled'])
        self.assertFalse(state['held'])

    def test_precheck_stops_without_submission(self):
        for scenario in ('capacity', 'dependency_drift', 'package_drift', 'existing_output', 'reserved'):
            with self.subTest(scenario=scenario):
                result, calls, _ = self.execute(scenario)
                self.assertEqual(result['stage'], 'precheck')
                self.assertFalse(any(a[0] == 'sbatch' for a in calls))

    def test_qos_rejection_once(self):
        result, calls, _ = self.execute('qos')
        self.assertEqual(result['stage'], 'submit')
        self.assertEqual(sum(a[0] == 'sbatch' for a in calls), 1)
        self.assertFalse(any(a[0] == 'scancel' for a in calls))

    def test_partial_dependency_no_cancel_or_release(self):
        result, calls, state = self.execute('partial_dependency')
        self.assertEqual(result['dependencies_already_migrated'], ['4069916'])
        self.assertTrue(state['held'])
        self.assertFalse(any(a[0] == 'scancel' or a[:2] == ['scontrol', 'release'] for a in calls))

    def test_cancel_denied_keeps_new_held(self):
        result, calls, state = self.execute('cancel_denied')
        self.assertEqual(result['stage'], 'retire_old')
        self.assertTrue(state['held'])
        self.assertFalse(any(a[:2] == ['scontrol', 'release'] for a in calls))

    def test_release_denied_no_retry(self):
        result, calls, state = self.execute('release_denied')
        self.assertEqual(result['stage'], 'release')
        self.assertTrue(state['held'])
        self.assertEqual(sum(a[:2] == ['scontrol', 'release'] for a in calls), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
