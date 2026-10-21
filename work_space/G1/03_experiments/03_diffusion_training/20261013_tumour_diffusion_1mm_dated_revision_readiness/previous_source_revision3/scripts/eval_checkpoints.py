"""Evaluate several checkpoints of one modality and tabulate the metrics per step (Step 5).

Runs src/infer/evaluate_generation.py once per selected checkpoint (val split, fixed seed), then
writes <out_dir>/<modality>/summary.csv (+ summary.png if matplotlib is available) so the best
training step can be read off the curve. Finished steps are skipped on re-run.

Run from TumourDiffusion_1mm/, one GPU per modality:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_checkpoints.py --modality t1c
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_checkpoints.py --modality t1c --every 10000 --max_cases 300
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_checkpoints.py --modality t1c --steps 104000 150800 200000
Extra arguments after "--" are passed to evaluate_generation.py unchanged.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS = ("mse", "mae", "psnr", "ssim", "ssim_whole", "ssim_lesion")


def saved_steps(weights_dir):
    steps = []
    for name in os.listdir(weights_dir):
        m = re.fullmatch(r"diffusion_(\d+)\.pt", name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def select_steps(steps, every):
    """First checkpoint at or after each multiple of `every`, plus the last one."""
    chosen, next_mark = [], every
    for s in steps:
        if s >= next_mark:
            chosen.append(s)
            next_mark = (s // every + 1) * every
    if steps and steps[-1] not in chosen:
        chosen.append(steps[-1])
    return chosen


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", required=True, choices=["t1c", "t1n", "t2w", "t2f"])
    ap.add_argument("--logdir", default="brats_1mm")
    ap.add_argument("--every", type=int, default=20000, help="Evaluate about every N steps (plus the last)")
    ap.add_argument("--steps", type=int, nargs="*", default=None, help="Explicit steps (overrides --every)")
    ap.add_argument("--out_dir", default="eval_ckpts")
    ap.add_argument("--max_cases", type=int, default=0, help="Limit val rows per checkpoint (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    ckpt_dir = os.path.join("Checkpoint", args.logdir)
    weights_dir = os.path.join(ckpt_dir, args.modality, "weights")
    csv_path = os.path.join(ckpt_dir, f"{args.logdir}.csv")
    steps = saved_steps(weights_dir)
    if not steps:
        sys.exit(f"No diffusion_*.pt in {weights_dir}")
    todo = sorted(args.steps) if args.steps else select_steps(steps, args.every)
    missing = [s for s in todo if s not in steps]
    if missing:
        sys.exit(f"Checkpoints not found for steps {missing}; saved steps: {steps}")
    print(f"[{args.modality}] {len(steps)} checkpoints saved, evaluating {len(todo)}: {todo}")

    mod_dir = os.path.join(args.out_dir, args.modality)
    for step in todo:
        out = os.path.join(mod_dir, f"step_{step}")
        if os.path.isfile(os.path.join(out, "metrics.json")):
            print(f"[{args.modality}] step {step}: done already, skipping")
            continue
        cmd = [sys.executable, os.path.join(ROOT, "src", "infer", "evaluate_generation.py"),
               "--diffusion_ckpt_dir", ckpt_dir, "--csv_path", csv_path,
               "--modality", args.modality, "--ckpt_step", str(step), "--output_dir", out,
               "--dataset", "BRATS_2024", "--generator_type", "Unet_NnU", "--crop_size", "64",
               "--normalization", "nnunet", "--sigma_data", "1.0", "--noise_schedule", "edm",
               "--sampling_method", "edm_heun", "--sampling_steps", "18",
               "--max_cases", str(args.max_cases), "--seed", str(args.seed)] + extra
        print(f"[{args.modality}] step {step}: evaluating ...", flush=True)
        subprocess.run(cmd, check=True)

    # Collect every evaluated step of this modality (also ones from earlier runs)
    rows = []
    if os.path.isdir(mod_dir):
        for name in os.listdir(mod_dir):
            m = re.fullmatch(r"step_(\d+)", name)
            path = os.path.join(mod_dir, name, "metrics.json")
            if m and os.path.isfile(path):
                with open(path) as f:
                    summ = json.load(f).get("summary", {}).get(args.modality, {})
                if summ:
                    rows.append({"step": int(m.group(1)), **{k: summ[k] for k in METRICS if k in summ}})
    rows.sort(key=lambda r: r["step"])
    cols = ["step"] + [k for k in METRICS if any(k in r for r in rows)]
    summary_csv = os.path.join(mod_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"\n[{args.modality}] per-step metrics ({summary_csv}):")
    print("  ".join(f"{c:>10}" for c in cols))
    for r in rows:
        print("  ".join(f"{r.get(c, ''):>10}" for c in cols))
    if rows and "psnr" in cols:
        best = max(rows, key=lambda r: r["psnr"])
        print(f"best PSNR: step {best['step']} ({best['psnr']} dB)")
    ssim_key = next((k for k in ("ssim", "ssim_lesion") if k in cols), None)
    if rows and ssim_key:
        best = max(rows, key=lambda r: r[ssim_key])
        print(f"best {ssim_key.upper()}: step {best['step']} ({best[ssim_key]})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = [k for k in ("psnr", ssim_key) if k]
        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.5), squeeze=False)
        for ax, k in zip(axes[0], keys):
            ax.plot([r["step"] for r in rows], [r[k] for r in rows], marker="o")
            ax.set_xlabel("training step")
            ax.set_ylabel(k.upper())
            ax.set_title(f"{args.modality}: {k.upper()} vs step")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(mod_dir, "summary.png"), dpi=120)
        print(f"plot: {os.path.join(mod_dir, 'summary.png')}")
    except Exception as e:  # plotting is optional
        print(f"(no plot: {e})")


if __name__ == "__main__":
    main()
