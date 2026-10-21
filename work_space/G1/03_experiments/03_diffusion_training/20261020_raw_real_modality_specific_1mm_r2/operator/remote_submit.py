"""Independent raw experiment: static gates, five held jobs, then audited release."""
import ast
import base64
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path


def dependency_ids(value):
    if value == '(null)':
        return set()
    matches = re.findall(r'afterok:([0-9]+)(?:\([^)]*\))?', value)
    remainder = re.sub(r'afterok:[0-9]+(?:\([^)]*\))?', '', value)
    assert matches and not remainder.strip(',: '), 'Unknown dependency expression'
    return set(matches)


def fields(text):
    return dict(re.findall(r'([A-Za-z][A-Za-z0-9_/:]*)=([^\s]+)', text))


def one_gpu(f):
    tres = fields(f.get('TRES', '').replace(',', ' '))
    if 'gres/gpu' in tres:
        return tres['gres/gpu'] == '1'
    entries = f.get('TresPerNode', '').split(',')
    return bool(entries) and all(re.fullmatch(r'(?:gres:)?gpu(?::a100)?:1', item) for item in entries)


def main():
    root = Path(__ROOT__)
    package_sha = __PACKAGE_SHA__
    payload_sha = __PAYLOAD_SHA__
    python = __PYTHON__
    meta = dict(artifact_status='experimental_unvalidated', operator_approved=False, formal_gate_status='not_run_not_passed')
    deadline = time.monotonic() + 540
    jobs, held, releases = {}, {}, {}
    stage = 'deploy'

    def now():
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def sha(p):
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        return h.hexdigest()

    def write(p, value):
        p.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o440), 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())

    def call(argv, timeout=30, checked=True, env=None):
        remaining = deadline - time.monotonic()
        assert remaining > 0, 'Bounded submission deadline exceeded'
        p = subprocess.run(argv, capture_output=True, text=True, cwd=root, env=env, timeout=min(timeout, remaining))
        r = dict(argv=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr, utc=now())
        if checked:
            assert p.returncode == 0, json.dumps(r)
        return r

    receipt = None
    try:
        payload_path = root / 'evidence/PAYLOAD.json'
        assert sha(payload_path) == payload_sha
        payload = json.loads(payload_path.read_text())
        for name, encoded in payload['files_b64'].items():
            p = root / name
            assert not Path(name).is_absolute() and p.resolve().is_relative_to(root)
            p.parent.mkdir(parents=True, exist_ok=True)
            with os.fdopen(os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o440), 'wb') as f:
                f.write(base64.b64decode(encoded))
        stage = 'static_preflight'
        assert sha(root / 'plan/PACKAGE_MANIFEST.json') == package_sha
        package = json.loads((root / 'plan/PACKAGE_MANIFEST.json').read_text())
        assert all(sha(root / p) == h for p, h in package['files'].items())
        cfg = json.loads((root / 'config/RUN.json').read_text())
        assert cfg['remote_root'] == str(root) and cfg['source_kind'] == 'original_authentic_native'
        assert cfg['vae_reconstruction'] is False and cfg['synthesized_t2w'] is False
        assert cfg['training']['num_steps'] == 200000 and cfg['training']['batch_size'] == 16
        assert cfg['training']['use_compile'] is False and cfg['training']['use_all_rows'] is False
        assert cfg['seed'] == 20261013
        upstream = json.loads((root / 'plan/UPSTREAM_SOURCE_MANIFEST.json').read_text())
        assert len(upstream['files']) == 37 and all(sha(root / 'runtime/upstream_source' / p) == h for p, h in upstream['files'].items())
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                   CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', G1_STATIC_DRY_RUN='1')
        for p in (root / 'runtime').glob('*.py'):
            ast.parse(p.read_text())
        for p in (root / 'slurm').glob('*.slurm'):
            call(['bash', '-n', str(p)])
            assert 'G1_RAW_R1_BASH_U_DRY_RUN_PASS' in call(['bash', '-u', str(p)], env=env)['stdout']
        tests = call([python, '-B', '-m', 'unittest', 'discover', '-s', str(root / 'tests'), '-p', 'test*.py', '-v'], timeout=180, env=env)
        assert 'Ran 0 tests' not in tests['stderr']
        probe = "import sys,json;sys.path.insert(0,'runtime');import prepare,contracts;cfg=json.loads((prepare.ROOT/'config/RUN.json').read_text());contracts.source_records();print(json.dumps(prepare.estimate(cfg)))"
        storage = json.loads(call([python, '-B', '-c', probe], timeout=120, env=env)['stdout'].strip().splitlines()[-1])
        assert storage['cases'] == 1165 and not storage['errors'] and storage['capacity_pass'] is True
        assert storage['required_bytes'] <= storage['available_bytes']
        network = "import sys,json;sys.path.insert(0,'runtime/upstream_source');from src.networks.DiffusionNetwork import get_diffusion_network;from types import SimpleNamespace;cfg=json.load(open('config/RUN.json'));n=get_diffusion_network(SimpleNamespace(**cfg['training']),n_steps=1000);p=sum(x.numel() for x in n.parameters());assert p==4475999;print(p)"
        imported = call([python, '-B', '-c', network], timeout=90, env=env)
        queue = call(['squeue', '-h', '-u', 'zqchen', '-o', '%i|%T|%j|%P|%r'])
        assert 'g1raw2' not in queue['stdout'], 'Duplicate independent raw job'
        assert len(queue['stdout'].splitlines()) + 5 < 100, 'Conservative partition submission count gate'
        partitions, suitable = {}, {}
        for partition, memory in [('cpu_192G', 92160), ('a100', 65536)]:
            p = call(['scontrol', 'show', 'partition', partition, '-o'])
            pf = fields(p['stdout'])
            assert pf['State'] == 'UP' and pf['ReqResv'] == 'NO' and pf['AllowGroups'] == 'ALL'
            assert pf['AllowAccounts'] == 'ALL' or 'zqchen' in pf['AllowAccounts'].split(',')
            assert pf['AllowQos'] == 'ALL' or 'user_zqchen' in pf['AllowQos'].split(',')
            names = call(['scontrol', 'show', 'hostnames', pf['Nodes']])['stdout'].splitlines()
            nodes = call(['scontrol', 'show', 'node', ','.join(names), '-o'])
            eligible = []
            for line in nodes['stdout'].splitlines():
                f = fields(line)
                capacity = int(f['RealMemory']) - int(f.get('MemSpecLimit', '0'))
                if capacity >= memory and int(f['CPUTot']) >= 8 and not any(x in f['State'] for x in ('DOWN', 'DRAIN', 'FAIL', 'NO_RESPOND', 'FUTURE')):
                    if partition != 'a100' or 'gpu' in f.get('Gres', ''):
                        eligible.append(dict(node=f['NodeName'], memory_mib=capacity, state=f['State'], gres=f.get('Gres')))
            assert eligible, 'No compatible node in ' + partition
            partitions[partition], suitable[partition] = p, eligible
        qos = call(['sacctmgr', '-n', '-P', 'show', 'qos', 'where', 'name=user_zqchen,partition_a100', 'format=Name,MaxWall,MaxJobsPU,MaxSubmitPU'])
        limits = {line.split('|')[0]: line.split('|')[1:4] for line in qos['stdout'].splitlines() if len(line.split('|')) >= 4}
        assert 'user_zqchen' in limits and 'partition_a100' in limits
        a100_queued = sum('|a100|' in line for line in queue['stdout'].splitlines())
        for name, count in [('user_zqchen', len(queue['stdout'].splitlines()) + 5), ('partition_a100', a100_queued + 4)]:
            max_submit = limits[name][2]
            if max_submit and max_submit != 'UNLIMITED':
                assert count <= int(max_submit), 'QoS max submission limit: ' + name
        static = dict(meta, status='pass', utc=now(), package_sha256=package_sha, tests=tests,
                      storage=storage, network_import=imported, queue=queue, partitions=partitions,
                      suitable_nodes=suitable, qos=qos, source_records_sha256=cfg['source_records_sha256'],
                      quality_review='not_run', scope='CPU synthetic and source/header/storage integrity, not A100 or scientific quality pass')
        write(root / 'evidence/STATIC_PREFLIGHT.json', static)
        for mod in ('prepare', 't1c', 't1n', 't2w', 't2f'):
            stage = 'submit_' + mod
            cpu = mod == 'prepare'
            script = root / 'slurm' / (mod + '.slurm')
            dependency = None if cpu else 'afterok:' + jobs['prepare']
            write(root / 'evidence' / (mod + '_RESERVATION.json'), dict(meta, utc=now(), dependency=dependency, script_sha256=sha(script)))
            argv = ['sbatch', '--parsable', '--hold', '--export=NONE']
            if dependency:
                argv += ['--dependency=' + dependency, '--kill-on-invalid-dep=yes']
            response = call(argv + [str(script)], checked=False)
            write(root / 'evidence' / (mod + '_SUBMISSION_RESPONSE.json'), response)
            assert response['returncode'] == 0, 'Submission rejected; no automatic retry'
            job = response['stdout'].strip().split(';')[0]
            assert re.fullmatch(r'[0-9]+', job)
            jobs[mod] = job
            write(root / 'evidence' / (mod + '_CANONICAL_JOB.json'), dict(meta, job_id=job, dependency=dependency))
            observed = call(['scontrol', 'show', 'job', job, '-o'])
            f = fields(observed['stdout'])
            assert f['JobName'] == 'g1raw2' + mod and f['UserId'].startswith('zqchen(')
            assert f['JobState'] == 'PENDING' and f['Reason'] == 'JobHeldUser'
            assert f['Partition'] == ('cpu_192G' if cpu else 'a100')
            assert f['NumCPUs'] == '8' and f['NumTasks'] == '1' and f['NumNodes'] in ('1', '1-1')
            assert f['MinMemoryNode'] == ('90G' if cpu else '64G')
            assert f['TimeLimit'] == ('1-00:00:00' if cpu else '2-00:00:00')
            assert f['Command'] == str(script) and f['WorkDir'] == str(root)
            assert f['StdOut'] == str(root / 'logs' / ('g1raw2' + mod + '_' + job + '.out'))
            assert f['StdErr'] == str(root / 'logs' / ('g1raw2' + mod + '_' + job + '.err'))
            assert dependency_ids(f['Dependency']) == (set() if cpu else {jobs['prepare']})
            if not cpu:
                assert one_gpu(f), 'Exactly one GPU required'
            assert sha(script) == package['files']['slurm/' + mod + '.slurm']
            held[mod] = observed
            write(root / 'evidence' / (mod + '_HELD_AUDIT.json'), dict(meta, status='held_audited', job=observed, dependency=dependency, utc=now()))
        assert len(jobs) == len(held) == 5
        # Consumers are released first, with the success dependency still holding them.
        for mod in ('t1c', 't1n', 't2w', 't2f', 'prepare'):
            stage = 'release_' + mod
            response = call(['scontrol', 'release', jobs[mod]], checked=False)
            releases[mod] = response
            write(root / 'evidence' / (mod + '_RELEASE_RECEIPT.json'), response)
            assert response['returncode'] == 0, 'Release rejected; retain evidence, no retry'
        receipt = dict(meta, status='pass', stage='all_five_released', jobs=jobs, held=held, releases=releases,
                       package_sha256=package_sha, static=static, remote_utc=now(), storage=storage,
                       queue_after_release=call(['squeue', '-h', '-j', ','.join(jobs.values()), '-o', '%i|%T|%j|%P|%r|%M']),
                       quality_review='not_run', heartbeat='paused_manual_queries_only')
        write(root / 'evidence/SUBMISSION_RECEIPT.json', receipt)
    except BaseException as e:
        receipt = dict(meta, status='failed', stage=stage, jobs=jobs, held=held, releases=releases,
                       remote_utc=now(), message=str(e), traceback=traceback.format_exc(), no_automatic_retry=True)
        write(root / 'evidence/OPERATION_FAILED.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
