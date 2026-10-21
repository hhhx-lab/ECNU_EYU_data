"""Target remaining evaluation identity checks and optimizer default compatibility on CPU."""
import ast
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "incoming_source"
if not SOURCE.exists():
    SOURCE = HERE.parents[4] / "work_space/G1/01_code/tumour_diffusion_1mm"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(SOURCE))
import torch
from src.train import tumour_main_diffusion as trainer
from src.infer import evaluate_generation as evaluation

torch.set_num_threads(2)
before = {p.relative_to(SOURCE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in SOURCE.rglob("*") if p.is_file()}

with tempfile.TemporaryDirectory(prefix="ecnu-dated-extra-") as directory:
    root = Path(directory)
    paths = {}
    csv = root / "current.csv"
    csv.write_text("patient_id,split\n00001-000,val\n")
    for mod in ("t1c", "t2w"):
        path = root / mod / "weights" / "diffusion_2.pt"
        path.parent.mkdir(parents=True)
        torch.save(dict(data_identity=dict(csv_sha256="different", csv_path="prior.csv"),
                        noise_schedule="edm", n_steps=1000,
                        schedule_config=dict(sigma_data=1., sigma_min=.002, sigma_max=50., rho=7.)), path)
        paths[mod] = path
    tree = ast.parse((SOURCE / "src/infer/evaluate_generation.py").read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "not args.self_test"
                  and any(isinstance(m, ast.Name) and m.id == "schedule_sources" for m in ast.walk(n)))
    executable = compile(ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])), "actual_eval_metadata_branch", "exec")
    touched = []
    original_load = torch.load
    def counted_load(*args, **kwargs):
        touched.append(str(args[0]))
        return original_load(*args, **kwargs)
    args = SimpleNamespace(self_test=False, modality="t1c,t2w", diffusion_ckpt_dir=str(root), ckpt_step="2",
                           csv_path=str(csv), noise_schedule="edm", n_steps=1000, sigma_data=1., sigma_min=.002,
                           sigma_max=50., rho=7., snr_shift=0.)
    env = dict(args=args, ckpt_metadata=None, schedule_sources={}, os=os, torch=torch,
               find_checkpoint=lambda wdir, step: str(Path(wdir) / f"diffusion_{step}.pt"),
               checkpoint_schedule=evaluation.checkpoint_schedule, reconcile_schedule=evaluation.reconcile_schedule)
    printed = io.StringIO()
    with patch.object(torch, "load", counted_load), contextlib.redirect_stdout(printed):
        exec(executable, env)
    identity = dict(status="fail", actual_branch_finished_without_rejection=True,
                    loaded_checkpoints=len(touched), csv_mismatch_warning_count=printed.getvalue().count("is not the CSV"),
                    both_checkpoints_csv_identity_mismatched=True, output=printed.getvalue(),
                    scope="actual_AST_metadata_branch_executed_no_sampling_or_GPU")

    tree = ast.parse((SOURCE / "src/train/tumour_main_diffusion.py").read_text())
    beta_parser = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and n.func.attr == "add_argument" and n.args and isinstance(n.args[0], ast.Constant)
                      and n.args[0].value == "--adam_beta1")
    default = next(k.value.value for k in beta_parser.keywords if k.arg == "default")
    opt_model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(opt_model.parameters(), betas=(.5, .999))
    weights = root / "Checkpoint/beta/t1c/weights"
    weights.mkdir(parents=True)
    trainer.save_ckp(dict(state_dict=opt_model.state_dict(), optimizer=optimizer.state_dict(), epoch=1,
                         global_step=2, step_semantics="completed_optimizer_updates_v2"), str(weights / "diffusion_2.pt"))
    resume = SimpleNamespace(logdir="beta", modality="t1c", resume_iter="2", adam_beta1=default)
    old = os.getcwd()
    mismatch = False
    try:
        os.chdir(root)
        try:
            trainer.load_ckp(resume, opt_model, optimizer)
        except ValueError as exc:
            mismatch = "beta1=0.5" in str(exc)
        resume.adam_beta1 = .5
        trainer.load_ckp(resume, opt_model, optimizer)
    finally:
        os.chdir(old)
    beta = dict(status="pass" if default == .9 and mismatch and optimizer.param_groups[0]["betas"][0] == .5 else "fail",
                parsed_new_default=default, prior_checkpoint_beta1=.5,
                default_rejected_legacy_beta1=True if mismatch else False,
                explicit_point5_resume_succeeded=optimizer.param_groups[0]["betas"][0] == .5,
                scope="actual_CPU_checkpoint_loader_without_GPU")

after = {p.relative_to(SOURCE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in SOURCE.rglob("*") if p.is_file()}
result = dict(artifact_status="experimental_unvalidated", operator_approved=False,
              formal_gate_status="not_run_not_passed", reviewed_at=datetime.now(timezone.utc).isoformat(),
              source_unchanged=before == after, checks=dict(evaluation_identity=identity, optimizer_beta1=beta))
with (HERE / "ADDITIONAL_CONTRACT_PROBES.json").open("x") as output:
    json.dump(result, output, ensure_ascii=False, indent=2)
    output.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
