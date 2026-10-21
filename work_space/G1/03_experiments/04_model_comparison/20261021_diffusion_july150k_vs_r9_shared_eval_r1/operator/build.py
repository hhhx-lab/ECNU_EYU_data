"""Generate structured identities and Slurm templates before immutable freezing."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[4]
BASE=Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated')
REMOTE=BASE/'g1_diffusion_july150k_vs_r9_shared_eval_20261021_r1'
REFERENCE=BASE/'g1_r9_fixedval_selection_20261018_r1'
TRAIN=BASE/'g1_diffusion_1mm_vae_epoch004_20261016_resume_uniform_r9'
PYTHON='/public/home/zqchen/.conda/envs/segmamba/bin/python'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(json.dumps(v,indent=2,sort_keys=True)+'\n')


def main():
    assert not (ROOT/'RUN.json').exists() and not (ROOT/'PACKAGE_MANIFEST.json').exists()
    old=PROJECT/'work_space/G2/02_dataset_release/20260919_archived_local_release/qc/diffusion_checkpoint_full94_150000_a800_recovery_20260721/checkpoint_selection.json'
    selected=PROJECT/'work_space/G1/03_experiments/03_diffusion_training/20261018_r9_fixedval_selection_r1/operator/pull_best_20261021T043517Z/lightweight/evaluation/selection/SELECTION.json'
    ref=selected.parents[1]/'reference/REPORT.json'
    s=json.loads(selected.read_text());o=json.loads(old.read_text());models={'july150k':{},'r9':{}};local={'july150k':{},'r9':{}}
    for mod in ('t1c','t1n','t2w','t2f'):
        info=o['checkpoint_files'][mod]
        lp=PROJECT/'work_space/G1/04_models/20260720_diffusion_v3_canonical/brats2026_diffusion_v3_edm_zscore'/mod/'weights/diffusion_150000.pt'
        assert sha(lp)==info['sha256'] and lp.stat().st_size==info['bytes']
        models['july150k'][mod]=dict(path=str(REMOTE/'models/july150k'/mod/'diffusion_150000.pt'),modality=mod,step=150000,bytes=info['bytes'],sha256=info['sha256'])
        local['july150k'][mod]={**models['july150k'][mod],'path':str(lp)}
        chosen=s['modalities'][mod];step=chosen['selected_step']
        rp=PROJECT/'work_space/G1/03_experiments/03_diffusion_training/20261018_r9_fixedval_selection_r1/best_models_fixedval103'/mod/f'diffusion_{step}.pt'
        assert sha(rp)==chosen['selected_checkpoint_sha256']
        models['r9'][mod]=dict(path=chosen['selected_checkpoint_path'],modality=mod,step=step,bytes=rp.stat().st_size,sha256=sha(rp))
        local['r9'][mod]={**models['r9'][mod],'path':str(rp),'contract_path':str(selected.parents[2]/'training/training'/mod/'RUN_CONTRACT.json')}
    config=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed',
        remote_root=str(REMOTE),reference_root=str(REFERENCE),training_root=str(TRAIN),
        reference_package_sha256='3486d52cfa995aff8cd439932384d8393224b90630cb878793aba01f7b41d2d4',
        training_package_sha256='80f432b454ea80cf3d4142cb87d194c7913c0eabd2da473b3821b06d10df63f6',
        reference_report_sha256=sha(ref),selection_sha256=sha(selected),old_selection_sha256=sha(old),
        schedule=dict(name='edm',sigma_data=1.,sigma_min=.002,sigma_max=50.,rho=7.,snr_shift=0.,n_steps=1000),
        models=models,seeds=[20261013,20261014,20261015,20261016],scored_cases=94,total_cases=103,
        primary='real_foreground_center_ssim3d',secondary='reconstruction_same_generated_output',
        halo=5,tile_stride=32,window=7,data_range=6.,no_automatic_retry=True,test_images_read=False,
        authorization='2026-10-21 user: evaluate new and old models under one consistent scoring system',
        preregistration_is_after_prior_incompatible_scores=True)
    save(ROOT/'RUN.json',config);save(ROOT/'LOCAL_MODELS.json',local)
    phases=['audit',*['smoke_'+m for m in models['r9']],*['eval_'+m for m in models['r9']],'summarize']
    for phase in phases:
        gpu=phase.startswith(('smoke_','eval_'));mem='64G' if gpu else '90G' if phase=='audit' else '32G'
        limit='02:00:00' if phase.startswith('smoke_') else '48:00:00' if gpu else '04:00:00'
        command='runtime/audit.py' if phase=='audit' else 'runtime/summarize.py' if phase=='summarize' else 'runtime/evaluate.py --modality '+phase.split('_')[1]+(' --smoke' if phase.startswith('smoke_') else '')
        text=f'''#!/bin/bash
#SBATCH --job-name=g1pair_{phase}
#SBATCH --partition={'a100' if gpu else 'cpu_192G'}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --time={limit}
'''+('#SBATCH --gres=gpu:1\n' if gpu else '')+f'''#SBATCH --signal=B:USR1@300
#SBATCH --output={REMOTE}/logs/{phase}_%j.out
#SBATCH --error={REMOTE}/logs/{phase}_%j.err
set -euo pipefail
if [[ "${{G1_STATIC_DRY_RUN:-0}}" == "1" ]]; then
  printf 'G1_PAIR_DRY_RUN_PASS\\n'
  exit 0
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=20261013 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONPATH
cd {REMOTE}
printf 'START %s UTC ' "$SLURM_JOB_ID"; date -u +%FT%TZ
hostname
exec {PYTHON} -B -u {command}
'''
        p=ROOT/'slurm'/(phase+'.slurm');p.parent.mkdir(exist_ok=True)
        with p.open('x') as f:f.write(text)
    print(json.dumps(dict(status='generated',models=8,phases=phases,remote_root=str(REMOTE))))


if __name__=='__main__':main()
