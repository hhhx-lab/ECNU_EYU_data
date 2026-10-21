"""Offline guard tests for the scheduler-only correction; never connect to ECNU."""
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


class CorrectionTests(unittest.TestCase):
    def run_correction(self, scenario='pass'):
        source = Path(__file__).with_name('change_prepare_partition.py').read_text()
        tree = ast.parse(source)
        code = next(n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'code' for t in n.targets))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'root'
            real_source = Path(temp) / 'source'
            for folder in ('plan', 'slurm', 'evidence', 'logs'):
                (root / folder).mkdir(parents=True)
            (real_source / 'data').mkdir(parents=True)
            (real_source / 'data/data_csv.csv').touch()
            files = {}
            for i in range(57):
                name = 'slurm/prepare.slurm' if i == 0 else 'plan/file_'+str(i)
                (root / name).touch()
                files[name] = hashlib.sha256(b'').hexdigest()
            (root / 'plan/PACKAGE_MANIFEST.json').write_text(json.dumps({'files': files}))
            (root / 'evidence/PREPARE_LAUNCH_AUDIT.json').write_text('{}')
            auth = {'artifact_status': 'experimental_unvalidated', 'operator_approved': False,
                    'formal_gate_status': 'not_run_not_passed', 'prepare_job': '4069915',
                    'dependent_jobs': ['4069916', '4069917', '4069918', '4069919'],
                    'package_sha256': hashlib.sha256((root / 'plan/PACKAGE_MANIFEST.json').read_bytes()).hexdigest()}
            code = code.replace('__ROOT__', repr(str(root))).replace('__SOURCE__', repr(str(real_source)))
            code = code.replace('__AUTH__', repr(auth)).replace('__AUDIT_NAME__', repr('test_correction'))
            code = code.replace('0d1765e12cdad7d8a5df57a5d2c46c79414ed99a2015a4d8fed57cd4c4fc836b', hashlib.sha256(b'').hexdigest())
            calls = []
            state = {'held': False, 'partition': 'cpu_96G'}

            def job_text(job_id):
                if job_id != '4069915':
                    dependency = 'afterok:999' if scenario == 'dependency_drift' else 'afterok:4069915(unfulfilled)'
                    return ('JobId='+job_id+' JobState=PENDING RunTime=00:00:00 Partition=a100 '
                            'Dependency='+dependency+' NumCPUs=8 MinMemoryNode=64G TimeLimit=2-00:00:00 '
                            'Command='+str(root / 'slurm/t1c.slurm'))
                return ('JobId=4069915 JobName=g1d1m5p1 JobState=PENDING RunTime=00:00:00 Restarts=0 '
                        'Partition='+state['partition']+' NumCPUs=8 NumTasks=1 CPUs/Task=8 NumNodes=1-1 '
                        'MinMemoryNode=90G TimeLimit=1-00:00:00 Account=zqchen QOS=user_zqchen Features=(null) '
                        'Command='+str(root / 'slurm/prepare.slurm')+' StdOut='+str(root / 'logs/g1d1m5p1_4069915.out')+
                        ' StdErr='+str(root / 'logs/g1d1m5p1_4069915.err')+' Dependency=(null) Reason='+
                        ('JobHeldAdmin' if state['held'] and scenario=='admin_hold' else 'JobHeldUser' if state['held'] else 'BadConstraints' if state['partition']=='cpu_96G' else 'Priority'))

            def fake_run(args, **kwargs):
                calls.append(args)
                out, err, status = '', '', 0
                if args[:3] == ['scontrol', 'show', 'job']:
                    out = job_text(args[3])
                elif args[:3] == ['scontrol', 'show', 'partition']:
                    out = ('PartitionName=cpu_192G State=UP AllowAccounts=zqchen AllowGroups=ALL '
                           'AllowQos=ALL MaxTime=UNLIMITED ReqResv=NO Nodes=node4 QoS=partition_cpu_192g')
                elif args[:3] == ['scontrol', 'show', 'hostnames']:
                    out = 'node4\n'
                elif args[:3] == ['scontrol', 'show', 'node']:
                    memory = 95051 if scenario == 'insufficient_node' else 190913
                    out = 'NodeName=node4 CPUTot=28 RealMemory='+str(memory)+' MemSpecLimit=4096 State=MIXED\n'
                elif args[0] == 'sacctmgr':
                    out = 'user_zqchen||||||1000|\npartition_cpu_192g|||||||\n'
                elif args[:2] == ['scontrol', 'hold']:
                    state['held'] = True
                elif args[:2] == ['scontrol', 'update']:
                    if scenario == 'update_denied':
                        status, err = 1, 'Access denied'
                    else:
                        state['partition'] = 'cpu_192G'
                elif args[:2] == ['scontrol', 'release']:
                    state['held'] = False
                return subprocess.CompletedProcess(args, status, stdout=out, stderr=err)

            fake_disk = type('Disk', (), {'f_bavail': 224162480128, 'f_frsize': 1})()
            if scenario == 'package_drift':
                (root / 'slurm/prepare.slurm').write_text('changed')
            with patch('subprocess.run', side_effect=fake_run), patch('os.statvfs', return_value=fake_disk):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    exec(compile(code, '<offline_partition_test>', 'exec'), {})
            return json.loads(output.getvalue().strip().splitlines()[-1]), calls, state

    def test_existing_id_and_dependencies_preserved(self):
        result, calls, state = self.run_correction()
        self.assertEqual(result['status'], 'existing_prepare_partition_corrected_and_released')
        self.assertEqual(result['dependency'], 'afterok:4069915')
        self.assertEqual(result['jobs_submitted'], [])
        self.assertEqual(result['jobs_cancelled'], [])
        self.assertEqual(state, {'held': False, 'partition': 'cpu_192G'})
        mutations = [a for a in calls if a[:2] in (['scontrol', 'hold'], ['scontrol', 'update'], ['scontrol', 'release'])]
        self.assertEqual(mutations, [['scontrol', 'hold', '4069915'],
                                   ['scontrol', 'update', 'JobId=4069915', 'Partition=cpu_192G'],
                                   ['scontrol', 'release', '4069915']])
        self.assertFalse(any(a[0] in ('sbatch', 'scancel') for a in calls))

    def test_memory_failure_is_before_hold(self):
        result, calls, _ = self.run_correction('insufficient_node')
        self.assertEqual(result['stage'], 'precheck')
        self.assertFalse(any(a[:2] == ['scontrol', 'hold'] for a in calls))

    def test_proven_successful_hold_admin_reason(self):
        result, _, state = self.run_correction('admin_hold')
        self.assertEqual(result['status'], 'existing_prepare_partition_corrected_and_released')
        self.assertFalse(state['held'])

    def test_dependency_drift_is_before_hold(self):
        result, calls, _ = self.run_correction('dependency_drift')
        self.assertEqual(result['stage'], 'precheck')
        self.assertFalse(any(a[:2] == ['scontrol', 'hold'] for a in calls))

    def test_package_drift_is_before_hold(self):
        result, calls, _ = self.run_correction('package_drift')
        self.assertEqual(result['stage'], 'precheck')
        self.assertFalse(any(a[:2] == ['scontrol', 'hold'] for a in calls))

    def test_update_rejection_keeps_job_held_without_retry(self):
        result, calls, state = self.run_correction('update_denied')
        self.assertEqual(result['stage'], 'partition_update')
        self.assertEqual(state, {'held': True, 'partition': 'cpu_96G'})
        self.assertEqual(sum(a[:2] == ['scontrol', 'update'] for a in calls), 1)
        self.assertFalse(any(a[:2] == ['scontrol', 'release'] for a in calls))


if __name__ == '__main__':
    unittest.main(verbosity=2)
