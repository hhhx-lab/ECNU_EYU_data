"""Render and audit the single authorized CPU job; never submits by itself."""
from pathlib import Path
import re
import shlex


def render_slurm(attempt, build_env, conda, contract_sha):
    attempt = Path(attempt)
    assert attempt.is_absolute() and attempt.name == 's0_cpu_prepare_r1'
    assert Path(build_env).is_absolute() and Path(build_env).name == 'bio_build_glibc217_r1'
    assert re.fullmatch('[0-9a-f]{64}', contract_sha)
    quote = shlex.quote
    return '\n'.join([
        '#!/bin/bash', '#SBATCH --job-name=s2exts0p1', '#SBATCH --partition=cpu_192G',
        '#SBATCH --nodes=1', '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=8',
        '#SBATCH --mem=64G', '#SBATCH --time=04:00:00', '#SBATCH --no-requeue', '#SBATCH --export=NONE',
        '#SBATCH --output='+str(attempt/'logs/s2exts0p1_%j.out'),
        '#SBATCH --error='+str(attempt/'logs/s2exts0p1_%j.err'),
        'set -euo pipefail', 'umask 077', 'source /etc/profile',
        'source '+quote(str(Path(conda).parent.parent/'etc/profile.d/conda.sh')),
        'conda activate '+quote(str(build_env)), 'module load singularity-ce/4.1.1',
        'unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY',
        'export PYTHONDONTWRITEBYTECODE=1 LC_ALL=C CUDA_VISIBLE_DEVICES=',
        'export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8',
        'export PIP_NO_INDEX=1 PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1',
        'export TMPDIR='+quote(str(attempt/'scratch')),
        'export S0_EXPECTED_CONTRACT_SHA256='+quote(contract_sha),
        'cd '+quote(str(attempt)),
        'exec "$CONDA_PREFIX/bin/python" -B '+quote(str(attempt/'runtime/prepare_s0_cpu_r1.py'))+' --attempt '+quote(str(attempt)),
        '',
    ])


def job_fields(text):
    tokens = list(re.finditer(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/]*)=', text))
    return {x.group(1):text[x.end():tokens[i+1].start() if i+1<len(tokens) else len(text)].strip()
            for i,x in enumerate(tokens)}


def memory_mib(value):
    match = re.fullmatch(r'(\d+)([KMGT]?)', value)
    assert match, 'unrecognized Slurm memory'
    n, suffix = match.groups()
    return int(n)*{'':1,'K':1/1024,'M':1,'G':1024,'T':1024**2}[suffix]


def validate_held(text, job_id, attempt, allowed_accounts):
    f = job_fields(text)
    attempt = Path(attempt)
    expected = {'JobId':job_id, 'JobName':'s2exts0p1', 'JobState':'PENDING', 'Reason':'JobHeldUser',
                'Priority':'0', 'Partition':'cpu_192G', 'NumCPUs':'8', 'NumTasks':'1', 'CPUs/Task':'8',
                'Requeue':'0', 'Command':str(attempt/'slurm/prepare.slurm'), 'WorkDir':str(attempt)}
    for key,value in expected.items():
        assert f.get(key) == value, 'CPU held identity drift: '+key
    assert f.get('Account') in allowed_accounts, 'CPU held account not authorized'
    assert f.get('UserId','').split('(')[0] == 'zqchen', 'CPU held owner drift'
    assert f.get('NumNodes') in ('1','1-1'), 'CPU held node count drift'
    assert f.get('TimeLimit') in ('04:00:00','4:00:00'), 'CPU held walltime drift'
    assert memory_mib(f.get('MinMemoryNode','')) == 65536, 'CPU held memory drift'
    assert f.get('Dependency') in ('(null)','None'), 'unexpected CPU dependency'
    assert not re.search('gpu', '|'.join(f.get(k,'') for k in ('TRES','ReqTRES','Gres','TresPerNode')), re.I), 'GPU in CPU held request'
    for key,suffix in [('StdOut','out'),('StdErr','err')]:
        assert f.get(key) in (str(attempt/f'logs/s2exts0p1_{job_id}.{suffix}'),
                              str(attempt/f'logs/s2exts0p1_%j.{suffix}')), 'CPU held log path drift'
    return f
