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
