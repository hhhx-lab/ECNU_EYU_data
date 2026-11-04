"""Read headed Slurm QoS tables; never infer limits from column positions."""
import csv
import io
import re
from s0_cpu_submission_contract import memory_mib


def parse_qos_table(text):
    rows=list(csv.reader(io.StringIO(text),delimiter='|'))
    assert len(rows)>1, 'headed QoS table absent'
    header=[x.strip() for x in rows[0]]
    assert header.count('Name')==1 and len({x for x in header if x})==len([x for x in header if x]), 'QoS header ambiguous'
    result={}
    for values in rows[1:]:
        if not values:continue
        assert len(values)==len(header), 'truncated QoS row'
        row={k:v.strip() for k,v in zip(header,values) if k}
        assert row['Name'] and row['Name'] not in result, 'duplicate or empty QoS name'
        result[row['Name']]=row
    return result


def _limit(row,aliases):
    keys=[key for key in aliases if key in row]
    assert len(keys)==1, 'required QoS limit header absent or ambiguous: '+aliases[0]
    return row[keys[0]]


def wall_seconds(value):
    if value in ('','UNLIMITED','NONE'):return None
    match=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})',value)
    assert match, 'unsupported QoS wall time'
    days,hours,minutes,seconds=match.groups()
    return int(days or 0)*86400+int(hours)*3600+int(minutes)*60+int(seconds)


def validate_cpu_qos(rows,names,existing_jobs,cpus=8,mem_mib=65536,wall=14400):
    checked={}
    for name in names:
        assert name in rows, 'applicable QoS absent: '+name
        row=rows[name]
        maximum=wall_seconds(_limit(row,('MaxWall','MaxWallDurationPerJob')))
        assert maximum is None or maximum>=wall, 'CPU request exceeds QoS wall'
        tres=_limit(row,('MaxTRES','MaxTRESPJ','MaxTRESPerJob'))
        limits={}
        if tres:
            for item in tres.split(','):
                key,value=item.split('=',1)
                assert key not in limits, 'duplicate QoS TRES'
                limits[key]=value
        if limits.get('cpu'):assert int(limits['cpu'])>=cpus, 'CPU request exceeds QoS CPUs'
        if limits.get('mem'):assert memory_mib(limits['mem'])>=mem_mib, 'CPU request exceeds QoS memory'
        max_jobs=_limit(row,('MaxJobsPU','MaxJobsPerUser','MaxJobs'))
        max_submit=_limit(row,('MaxSubmitPU','MaxSubmitJobsPU','MaxSubmitJobsPerUser','MaxSubmitJobs'))
        if max_jobs:assert int(max_jobs)>=1, 'QoS disallows running jobs'
        if max_submit:assert int(max_submit)>existing_jobs, 'QoS submission limit reached'
        checked[name]=dict(max_wall_seconds=maximum,max_TRES_per_job=limits,
                           max_jobs_per_user=max_jobs,max_submit_jobs_per_user=max_submit)
    return checked


def validate_cpu_snapshot(snapshot):
    from s0_cpu_submission_contract import job_fields
    scheduler=snapshot['scheduler']
    for name,item in scheduler.items():
        if name!='quota':assert item['returncode']==0, 'scheduler command failed: '+name
    assert scheduler['libc']['stdout'].strip()=='glibc 2.17', 'real host libc drift'
    partition=job_fields(scheduler['partition']['stdout'])
    assert partition['PartitionName']=='cpu_192G' and partition['State']=='UP', 'CPU partition unavailable'
    assert partition['AllowGroups']=='ALL', 'CPU partition group restriction not reviewed'
    assert partition['RootOnly']=='NO' and partition['ReqResv']=='NO', 'CPU partition requires special access'
    limit=wall_seconds(partition['MaxTime'])
    assert limit is None or limit>=14400, 'CPU partition wall limit'
    if partition['MaxMemPerNode']!='UNLIMITED':
        assert memory_mib(partition['MaxMemPerNode'])>=65536, 'CPU partition memory limit'
    if partition['MaxCPUsPerNode']!='UNLIMITED':
        assert int(partition['MaxCPUsPerNode'])>=8, 'CPU partition CPU limit'
    if partition['MaxNodes']!='UNLIMITED':assert int(partition['MaxNodes'])>=1, 'CPU partition node limit'
    assert int(partition['MinNodes'])<=1, 'CPU partition minimum nodes'
    associations=list(csv.DictReader(io.StringIO(scheduler['association']['stdout']),delimiter='|'))
    eligible=[r for r in associations if r['User']=='zqchen' and r['Account']=='zqchen'
              and r['Partition'] in ('','cpu_192G')]
    assert len(eligible)==1, 'CPU association absent or ambiguous'
    association=eligible[0]
    assert partition['AllowAccounts']=='ALL' or 'zqchen' in partition['AllowAccounts'].split(','), 'CPU account disallowed'
    queue=[line.split('|') for line in scheduler['own_queue']['stdout'].splitlines() if line]
    assert all(len(row)==6 for row in queue), 'CPU queue schema drift'
    count=len(queue)
    for key in ('MaxJobs','MaxSubmit'):
        assert key in association, 'association limit header absent'
        if association[key]:assert int(association[key])>count, 'association job limit reached'
    rows=parse_qos_table(scheduler['qos']['stdout'])
    canonical={}
    for name in rows:
        assert name.lower() not in canonical, 'case-ambiguous QoS names'
        canonical[name.lower()]=name
    requested=[partition['QoS'],*association['QOS'].split(',')]
    assert all(name and not name.startswith(('+','-')) for name in requested), 'QoS inheritance not resolved'
    assert all(name.lower() in canonical for name in requested), 'applicable QoS absent'
    names=list(dict.fromkeys(canonical[name.lower()] for name in requested))
    allowed=partition['AllowQos']
    if allowed!='ALL':
        assert all(name.lower() in allowed.lower().split(',') for name in association['QOS'].split(',')), 'CPU QoS disallowed'
    checked=validate_cpu_qos(rows,names,count)
    for name in names:
        row=rows[name]
        for key in ('GrpTRES','GrpTRESMins','GrpTRESRunMins','GrpJobs','GrpSubmit','GrpWall','MaxTRESMins'):
            assert not row.get(key), 'shared or runtime QoS limit requires usage audit: '+key
        for key in ('MaxTRESPU','MaxTRESPA'):
            if row.get(key):
                assert count==0, 'aggregate QoS TRES requires current usage audit'
                limits=dict(item.split('=',1) for item in row[key].split(','))
                if 'cpu' in limits:assert int(limits['cpu'])>=8, 'aggregate CPU QoS limit'
                if 'mem' in limits:assert memory_mib(limits['mem'])>=65536, 'aggregate memory QoS limit'
                assert set(limits)<= {'cpu','mem','node'}, 'aggregate QoS resource not reviewed'
                if 'node' in limits:assert int(limits['node'])>=1, 'aggregate node QoS limit'
        for key in ('MaxJobsPA','MaxSubmitPA'):
            if row.get(key):assert int(row[key])>count, 'account QoS job limit reached'
    nodes=[line.split('|') for line in scheduler['nodes']['stdout'].splitlines() if line]
    assert nodes and all(len(row)==5 for row in nodes), 'CPU node table schema drift'
    suitable=[r[0] for r in nodes if r[1] in ('idle','mix','alloc') and int(r[2])>=8 and int(r[3])>=65536]
    assert suitable, 'no configured CPU node meets request'
    assert snapshot['free_bytes']>150*1024**3, 'CPU preparation filesystem capacity below reservation'
    return dict(status='CPU_scheduler_request_eligible_not_running_or_full_capacity_pass',
                account='zqchen',partition='cpu_192G',applicable_qos=names,checked_qos=checked,
                association=association,existing_jobs=count,suitable_configured_nodes=suitable,
                requested_resource=dict(cpus=8,memory_mib=65536,wall_seconds=14400,gpu=0),
                filesystem_free_bytes=snapshot['free_bytes'],CPU_reservation_bytes=150*1024**3,
                quota_query=scheduler['quota'],numeric_quota_verified=False,full_inference_capacity_pass=False)
