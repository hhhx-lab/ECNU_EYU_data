"""Coherent own-job CPU upper bounds for the authorized per-user QoS gate."""
import hashlib
import re
import subprocess
import time

from s0_cpu_submission_contract import job_fields

STATES = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'STOPPED', 'RESIZING'}


def queue_records(text):
    records = {}
    for line in text.splitlines():
        if not line:
            continue
        values = line.split('|')
        assert len(values) == 6, 'CPU usage queue schema drift'
        identity, state, partition, name, _, _ = values
        assert re.fullmatch(r'[0-9]+', identity) and identity not in records, 'CPU usage JobID ambiguous'
        assert state in STATES and partition and name, 'CPU usage job state or identity unreviewed'
        records[identity] = dict(state=state, partition=partition, name=name)
    assert len(records) <= 100, 'CPU usage job metadata budget'
    return records


def job_cpu_record(fields, identity, expected):
    assert fields.get('JobId') == identity, 'CPU usage job identity drift'
    assert fields.get('JobState') == expected['state'], 'CPU usage job state changed'
    assert fields.get('Partition') == expected['partition'], 'CPU usage partition changed'
    assert fields.get('JobName') == expected['name'], 'CPU usage job name changed'
    assert fields.get('UserId', '').split('(')[0] == 'zqchen', 'CPU usage owner drift'
    assert fields.get('Account') == 'zqchen' and fields.get('QOS'), 'CPU usage account or QoS unreviewed'
    num_cpus = fields.get('NumCPUs', '')
    assert re.fullmatch(r'[1-9][0-9]*', num_cpus), 'CPU usage NumCPUs invalid'
    cpus = {'NumCPUs': int(num_cpus)}
    for name in ('ReqTRES', 'AllocTRES', 'TRES'):
        value = fields.get(name)
        if value in (None, '', '(null)', 'N/A', 'None'):
            continue
        tres = {}
        for item in value.split(','):
            key, count = item.split('=', 1)
            assert key not in tres, 'CPU usage duplicate TRES'
            tres[key] = count
        if 'cpu' in tres:
            assert re.fullmatch(r'[1-9][0-9]*', tres['cpu']), 'CPU usage TRES CPU invalid'
            cpus[name] = int(tres['cpu'])
    # Older Slurm may expose only TRES/NumCPUs; count their maximum, not an assumed allocation.
    return dict(job_id=identity, state=expected['state'], partition=expected['partition'],
                job_name=expected['name'], account=fields['Account'], qos=fields['QOS'],
                observed_cpu_fields=cpus, cpu_upper_bound=max(cpus.values()),
                explicit_allocated_cpu=cpus.get('AllocTRES'))


def collect_cpu_usage(queue_text):
    started = time.monotonic()
    before = queue_records(queue_text)
    jobs = []
    for identity, expected in before.items():
        p = subprocess.run(['scontrol', 'show', 'job', identity, '-o'], capture_output=True, text=True, timeout=30)
        assert p.returncode == 0 and len(p.stdout.encode()) + len(p.stderr.encode()) < 65536, 'CPU usage metadata query failed'
        jobs.append(job_cpu_record(job_fields(p.stdout), identity, expected))
    p = subprocess.run(['squeue', '-h', '-u', 'zqchen', '-o', '%i|%T|%P|%j|%r|%b'],
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0 and len(p.stdout.encode()) < 131072, 'CPU usage coherence query failed'
    assert queue_records(p.stdout) == before, 'CPU usage queue changed during snapshot'
    elapsed = time.monotonic() - started
    assert elapsed <= 120, 'CPU usage snapshot freshness budget exceeded'
    return dict(status='coherent_own_job_CPU_upper_bound_snapshot', jobs=jobs, elapsed_seconds=elapsed,
                queue_sha256=hashlib.sha256(queue_text.encode()).hexdigest(),
                pending_cpu_upper_bound=sum(x['cpu_upper_bound'] for x in jobs if x['state'] == 'PENDING'),
                active_cpu_upper_bound=sum(x['cpu_upper_bound'] for x in jobs if x['state'] != 'PENDING'),
                known_explicit_allocated_cpu=sum(x['explicit_allocated_cpu'] or 0 for x in jobs),
                all_active_allocations_explicit=all(x['explicit_allocated_cpu'] is not None for x in jobs if x['state'] != 'PENDING'),
                scope='all_own_jobs_conservative_including_pending_requests_not_measured_GPU_usage')


def validate_cpu_usage_limit(snapshot, row, key):
    assert key == 'MaxTRESPU', 'account aggregate TRES needs an independent account-wide audit'
    limits = {}
    for item in row[key].split(','):
        resource, value = item.split('=', 1)
        assert resource not in limits, 'aggregate QoS TRES duplicate'
        limits[resource] = value
    assert set(limits) == {'cpu'} and re.fullmatch(r'[1-9][0-9]*', limits['cpu']), 'aggregate non-CPU limit not reviewed'
    usage = snapshot['cpu_usage']
    assert usage['status'] == 'coherent_own_job_CPU_upper_bound_snapshot', 'CPU usage snapshot incomplete'
    queue = snapshot['scheduler']['own_queue']['stdout']
    assert usage['queue_sha256'] == hashlib.sha256(queue.encode()).hexdigest(), 'CPU usage queue binding drift'
    records = queue_records(queue)
    assert len(usage['jobs']) == len(records) and {x['job_id'] for x in usage['jobs']} == set(records), 'CPU usage job coverage drift'
    assert usage['elapsed_seconds'] <= 120, 'CPU usage snapshot stale'
    for job in usage['jobs']:
        assert job['state'] == records[job['job_id']]['state'] and job['cpu_upper_bound'] > 0, 'CPU usage record drift'
    pending = sum(x['cpu_upper_bound'] for x in usage['jobs'] if x['state'] == 'PENDING')
    active = sum(x['cpu_upper_bound'] for x in usage['jobs'] if x['state'] != 'PENDING')
    assert pending == usage['pending_cpu_upper_bound'] and active == usage['active_cpu_upper_bound'], 'CPU usage sum drift'
    total = pending + active + 8
    assert total <= int(limits['cpu']), 'aggregate CPU QoS capacity insufficient'
    usage.setdefault('checked_limits', []).append(dict(qos=row['Name'], per_user_cpu_limit=int(limits['cpu']),
        pending_cpu_upper_bound=pending, active_cpu_upper_bound=active, new_job_cpus=8, prospective_cpu_upper_bound=total))
