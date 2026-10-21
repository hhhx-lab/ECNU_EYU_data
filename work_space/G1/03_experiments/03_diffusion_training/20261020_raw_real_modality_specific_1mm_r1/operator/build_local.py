"""Deterministically derive a new baseline without editing any historical package."""
import ast
import hashlib
import json
import shutil
from pathlib import Path

from common import PROJECT, REMOTE, ROOT, m


def copy(source, target):
    assert not target.exists(), 'Do not overwrite a prepared/frozen file'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


class RawTraining(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == 'data_identity':
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == 'CHECKPOINT_SHA256' for t in node.targets):
            return None
        if any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
               and t.value.id == 'args' and t.attr == 'csv_path' for t in node.targets):
            node.value = ast.parse("str(ROOT / ('dataset/lesions_' + options.modality + '.csv'))", mode='eval').body
        if any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
               and t.value.id == 'args' and t.attr == 'logdir' for t in node.targets):
            node.value = ast.Constant('raw_real_modality_specific_1mm_r1')
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if node.value == 'waived_by_user':
            return ast.copy_location(ast.Constant('runtime_identity_checks_required_quality_not_run'), node)
        return node


def main():
    parent = ROOT.parent / '20261016_vae_epoch004_1mm_resume_uniform_r9'
    assert m.sha(parent / 'plan/PACKAGE_MANIFEST.json') == '80f432b454ea80cf3d4142cb87d194c7913c0eabd2da473b3821b06d10df63f6'
    old = json.loads((parent / 'plan/PACKAGE_MANIFEST.json').read_text())
    assert all(m.sha(parent / p) == v for p, v in old['files'].items())
    source = json.loads((parent / 'plan/UPSTREAM_SOURCE_MANIFEST.json').read_text())
    assert len(source['files']) == 37
    for p, digest in source['files'].items():
        src = parent / 'runtime/upstream_source' / p
        assert m.sha(src) == digest
        copy(src, ROOT / 'runtime/upstream_source' / p)
    copy(parent / 'plan/UPSTREAM_SOURCE_MANIFEST.json', ROOT / 'plan/UPSTREAM_SOURCE_MANIFEST.json')
    copy(parent / 'runtime/train.py', ROOT / 'plan/BASELINE_TRAIN_REFERENCE.py')
    release = PROJECT / 'work_space/G2/02_dataset_release/20260919_archived_local_release'
    for src, dest in [('splits/splits_master_train_val_test_membership.csv', 'master_membership.csv'),
                      ('splits/splits_final_train_val_test_membership.csv', 'realonly_membership.csv'),
                      ('manifests/real_train_manifest_raw.csv', 'raw_manifest.csv'),
                      ('manifests/real_train_manifest_ucsd.csv', 'ucsd_manifest.csv'),
                      ('manifests/corrected_label_overlay.csv', 'corrected_label_overlay.csv')]:
        copy(release / src, ROOT / 'plan' / dest)
    assert m.sha(ROOT / 'plan/master_membership.csv') == '1c829e7f7cc77505cb487b072bb58b6e9e8ac26ea770f7f938404d6d779f5673'
    assert m.sha(ROOT / 'plan/realonly_membership.csv') == '0a533791617d44d1ffe1afefe04b1154db2e7f8cf6db9f3421a8c1ff1aec7f9a'
    tree = RawTraining().visit(ast.parse((parent / 'runtime/train.py').read_text()))
    tree.body.insert(0, ast.ImportFrom(module='identity', names=[ast.alias(name='data_identity')], level=0))
    ast.fix_missing_locations(tree)
    target = ROOT / 'runtime/train.py'
    assert not target.exists()
    target.write_text(ast.unparse(tree) + '\n')
    cfg = json.loads((parent / 'config/RUN.json').read_text())
    keep = ['seed', 'training', 'preprocess_workers', 'loss_policy', 'label_resampling_algorithm', 'support_contract']
    new = {k: cfg[k] for k in keep}
    new.update(m.META, attempt='raw_real_modality_specific_1mm_r1', remote_root=str(REMOTE),
               source_kind='original_authentic_native', vae_reconstruction=False, synthesized_t2w=False,
               independent_full_revalidation='mandatory_runtime_identity_checks', manual_quality_review='not_run',
               nnunet_equivalence='not_independently_verified', source_records_sha256='pending_source_audit',
               membership_counts={mod: {'train': 823, 'val': 103} if mod == 't2w' else {'train': 1035, 'val': 130}
                                  for mod in ('t1c', 't1n', 't2w', 't2f')})
    m.exclusive(ROOT / 'config/RUN.json', new)
    for mod in ('prepare', 't1c', 't1n', 't2w', 't2f'):
        cpu = mod == 'prepare'
        lines = ['#!/bin/bash', '#SBATCH --job-name=g1raw1' + mod,
                 '#SBATCH --partition=' + ('cpu_192G' if cpu else 'a100'),
                 '#SBATCH --nodes=1', '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=8',
                 '#SBATCH --mem=' + ('90G' if cpu else '64G'), '#SBATCH --time=' + ('24:00:00' if cpu else '48:00:00')]
        if not cpu:
            lines += ['#SBATCH --gres=gpu:1', '#SBATCH --signal=B:USR1@300']
        lines += ['#SBATCH --output=' + str(REMOTE / 'logs') + '/g1raw1' + mod + '_%j.out',
                  '#SBATCH --error=' + str(REMOTE / 'logs') + '/g1raw1' + mod + '_%j.err',
                  'set -euo pipefail', 'if [[ "${G1_STATIC_DRY_RUN:-0}" == "1" ]]; then',
                  '  printf "G1_RAW_R1_BASH_U_DRY_RUN_PASS\\n"', '  exit 0', 'fi',
                  'export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=20261013',
                  'unset PYTHONPATH', 'cd ' + str(REMOTE)]
        if cpu:
            lines += ["export CUDA_VISIBLE_DEVICES=''", 'exec ' + m.PYTHON + ' -B -u runtime/prepare.py']
        else:
            lines += ['exec ' + m.PYTHON + ' -B -u runtime/train.py --modality ' + mod]
        p = ROOT / 'slurm' / (mod + '.slurm')
        p.parent.mkdir(exist_ok=True)
        assert not p.exists()
        p.write_text('\n'.join(lines) + '\n')
    m.exclusive(ROOT / 'plan/BASELINE_DERIVATION.json', dict(m.META,
                baseline_package_sha256=m.sha(parent / 'plan/PACKAGE_MANIFEST.json'),
                baseline_train_sha256=m.sha(parent / 'runtime/train.py'), new_train_sha256=m.sha(target),
                method='AST remove VAE/reference data_identity; import original-source identity; per-modality CSV and logdir; no training-loop changes',
                upstream_files=37, upstream_bytes_unchanged=True, quality_pass=False))
    print(json.dumps({'status': 'pass', 'root': str(ROOT), 'upstream_files': 37}))


if __name__ == '__main__':
    main()
