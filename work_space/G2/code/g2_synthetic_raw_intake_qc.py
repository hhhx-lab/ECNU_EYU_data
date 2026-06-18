#!/usr/bin/env python3
"""G2 entry point for G1 synthetic raw intake and QC.

This is a thin, explicit wrapper around the shared G2 audit utilities. Use it
when G1 hands over either a legacy GliGAN-compatible diffusion run folder or a
completion-style `data/output/<case_id>/` folder. It creates the candidate
manifest, accepted/rejected manifests, QC tables, diffusion-quality tables, and
a batch report under work_space/G2/results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from g2_pretraining_audit import (
    build_synthetic_run_context,
    ensure_dirs,
    ingest_synthetic_run,
    write_progress_report,
    write_templates,
)


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intake one G1 synthetic raw run and generate G2 QC outputs."
    )
    parser.add_argument(
        "--synthetic-run-root",
        required=True,
        help="G1 synthetic raw run folder. It may contain generation_config.json, generation_log.jsonl, and raw case folders.",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="G2 results root. Defaults to the shared workspace results folder.",
    )
    parser.add_argument(
        "--synthetic-run-id",
        default="",
        help="Optional run id override. If omitted, G2 uses generation_config.json or the run folder name.",
    )
    parser.add_argument(
        "--refresh-templates",
        action="store_true",
        help="Rewrite current G2 manifest/QC/report templates before intake.",
    )
    args = parser.parse_args()

    run_root = Path(args.synthetic_run_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    if not run_root.exists() or not run_root.is_dir():
        raise SystemExit(f"synthetic run folder not found: {run_root}")

    dirs = ensure_dirs(results_root)
    if args.refresh_templates:
        write_templates(dirs)

    ctx = build_synthetic_run_context(run_root, results_root, args)
    run_id = str(ctx["generation_run_id"])
    outputs = ingest_synthetic_run(run_root, results_root, args, dirs)
    if not outputs:
        raise SystemExit(f"no synthetic NIfTI cases were found under: {run_root}")

    intake_index = [
        ("synthetic_generation_manifest", [dirs["manifests"] / f"synthetic_generation_manifest_{run_id}.csv"]),
        ("synthetic_candidate_manifest", [dirs["manifests"] / f"synthetic_candidate_manifest_{run_id}.csv"]),
        ("synthetic_accepted_manifest", [dirs["manifests"] / f"synthetic_accepted_manifest_{run_id}.csv"]),
        ("synthetic_rejected_manifest", [dirs["manifests"] / f"synthetic_rejected_manifest_{run_id}.csv"]),
        ("synthetic_normalized_mapping", [dirs["manifests"] / f"synthetic_normalized_mapping_{run_id}.csv"]),
        ("qc_metrics", [dirs["qc"] / f"qc_metrics_{run_id}.csv"]),
        ("diffusion_quality_metrics", [dirs["qc"] / f"diffusion_quality_metrics_{run_id}.csv"]),
        ("qc_case_review", [dirs["qc"] / f"qc_case_review_{run_id}.csv"]),
        ("qc_batch_summary", [dirs["qc"] / f"qc_batch_summary_{run_id}.json"]),
        ("quality_report", [dirs["reports"] / f"G2_synthetic_data_quality_report_{run_id}.md"]),
    ]
    write_progress_report(
        results_root,
        dirs["reports"] / "G2_synthetic_intake_progress_report.md",
        intake_outputs=outputs,
        intake_index=intake_index,
    )

    print("G2 synthetic intake finished.")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
