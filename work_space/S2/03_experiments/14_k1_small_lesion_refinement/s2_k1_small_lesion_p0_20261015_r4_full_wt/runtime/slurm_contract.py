"""Validate held allocation semantically while retaining actual/expected fields."""
import re


def memory_mib(value):
    match=re.fullmatch(r'(\d+(?:\.\d+)?)([KMGTP]?)',value)
    if not match:raise ValueError('unrecognized Slurm memory representation')
    return float(match[1])*{'K':1/1024,'':1,'M':1,'G':1024,'T':1024**2,'P':1024**3}[match[2]]


def validate(raw,root,job_name,gpus=0,partition='cpu_192G',cpus=4,memory_gb=64,time='1-00:00:00',script='cpu_supplement.slurm'):
    fields=dict(s.split('=',1) for s in raw.split() if '=' in s)
    expected=dict(JobName=job_name,Partition=partition,NumCPUs=str(cpus),TimeLimit=time,
        JobState='PENDING',Reason='JobHeldUser',Command=root+'/slurm/'+script)
    checks={k:fields.get(k)==v for k,v in expected.items()}
    checks['NumNodes']=fields.get('NumNodes') in ('1','1-1')
    try:checks['memory']=memory_mib(fields.get('MinMemoryNode',''))==memory_gb*1024
    except ValueError:checks['memory']=False
    checks['stdout']=fields.get('StdOut','').startswith(root+'/logs/')
    checks['stderr']=fields.get('StdErr','').startswith(root+'/logs/')
    tres=fields.get('ReqTRES',fields.get('TRES',''))
    gpu_values=[int(x) for x in re.findall(r'(?:^|,)gres/gpu(?:\:[^=,]+)?=(\d+)',tres)]
    checks['gpu']=not gpu_values if gpus==0 else bool(gpu_values) and all(x==gpus for x in gpu_values)
    # Shared/OverSubscribe may describe partition policy; explicit request is audited from the script.
    checks['one_node']=checks['NumNodes']
    return dict(status='pass' if all(checks.values()) else 'failed',raw=raw,actual=fields,expected=expected,checks=checks)
