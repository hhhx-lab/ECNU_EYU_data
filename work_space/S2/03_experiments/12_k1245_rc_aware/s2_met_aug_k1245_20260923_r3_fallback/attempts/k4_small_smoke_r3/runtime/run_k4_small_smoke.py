#!/usr/bin/env python3
"""Run the fixed K4 small8 smoke with a contourpy-free montage renderer."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3] / "runtime/repository"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SHARED_PATH = REPO / "scripts/09_k1245/62_run_k1245_smoke.py"
spec = importlib.util.spec_from_file_location("k1245_smoke_shared", SHARED_PATH)
shared = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(shared)
from custom_nnunet.met_aug_data_loader import PreprocessedValidMaskStore
from custom_nnunet.met_aug_diffusion import G1FourModalityInpaintingBackend
from custom_nnunet.met_aug_k1245 import K1245Config, K1245Manifest, RCAwareFixV3CandidateProcessor
from custom_nnunet.met_aug_core import sha256_file

ROOT = Path(__file__).resolve().parents[1]
R2_PLAN = ROOT / "sources/K1245_RC_AWARE_EXECUTION_PLAN_20260923_R2.json"


def render_without_contour(before: np.ndarray, after: np.ndarray, label: np.ndarray, path: Path) -> None:
    """Render the required before/after evidence without importing contourpy."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    points = np.argwhere(label != 0)
    focus = tuple(int(round(value)) for value in points.mean(axis=0)) if len(points) else tuple(v // 2 for v in label.shape)
    figure, axes = plt.subplots(4, 6, figsize=(18, 12), constrained_layout=True)
    names = ("t1n", "t1c", "t2w", "t2f")
    for channel, name in enumerate(names):
        values = before[channel][np.isfinite(before[channel])]
        low, high = np.quantile(values, (0.01, 0.99)) if len(values) else (0.0, 1.0)
        planes_before = (before[channel][focus[0]], before[channel][:, focus[1]], before[channel][:, :, focus[2]])
        planes_after = (after[channel][focus[0]], after[channel][:, focus[1]], after[channel][:, :, focus[2]])
        label_planes = (label[focus[0]], label[:, focus[1]], label[:, :, focus[2]])
        for plane in range(3):
            for offset, volume in enumerate((planes_before[plane], planes_after[plane])):
                axis = axes[channel, plane * 2 + offset]
                axis.imshow(volume.T, cmap="gray", origin="lower", vmin=low, vmax=high)
                mask = label_planes[plane].T != 0
                if mask.any():
                    overlay = np.zeros(mask.shape + (4,), dtype=np.float32)
                    overlay[mask] = (1.0, 0.0, 0.0, 0.35)
                    axis.imshow(overlay, origin="lower", interpolation="nearest")
                axis.set_axis_off()
                if channel == 0:
                    axis.set_title(f"{('axial','coronal','sagittal')[plane]} {'before' if offset == 0 else 'after'}")
                if plane == 0 and offset == 0:
                    axis.set_ylabel(name)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-plan", required=True, type=Path)
    parser.add_argument("--pool-root", required=True, type=Path)
    parser.add_argument("--preprocessed-dir", required=True, type=Path)
    parser.add_argument("--valid-mask-manifest", required=True, type=Path)
    parser.add_argument("--legacy-manifest", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--g1-code-dir", required=True, type=Path)
    parser.add_argument("--g1-checkpoint-root", required=True, type=Path)
    parser.add_argument("--g1-checkpoint-selection", required=True, type=Path)
    parser.add_argument("--g2-parent-gate", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    smoke_plan = args.smoke_plan.resolve()
    payload = json.loads(smoke_plan.read_text())
    assert payload["route_id"] == "K4"
    assert payload["fallback_context"]["F_STRICT_PREREG_GATE"] is False
    assert payload["training_authorization"] is False and payload["k5_authorization"] is False
    assert sha256_file(R2_PLAN) == payload["parent_plan_sha256"]
    manifest = K1245Manifest.load(args.pool_root / "pools/small8/component_manifest.json")
    config = K1245Config.load(ROOT / "config/K1245_ROUTE_K4.json", manifest=manifest, plan_path=R2_PLAN)
    assert config.route_id == "K4" and config.use_x3 and not manifest.include_rc
    processor = RCAwareFixV3CandidateProcessor.load(
        args.calibration,
        expected_sha256=json.loads(R2_PLAN.read_text())["parent_contract"]["fix_v3_calibration_file_sha256"],
        expected_policy="label_only_qc_v1",
    )
    legacy_payload = json.loads(args.legacy_manifest.read_text())
    processor.bind_extended_manifest(
        legacy_manifest_sha256=legacy_payload["manifest_sha256"],
        extended_manifest_sha256=manifest.identity_sha256,
    )
    backend = G1FourModalityInpaintingBackend(
        g1_code_dir=args.g1_code_dir,
        checkpoint_root=args.g1_checkpoint_root,
        checkpoint_selection=args.g1_checkpoint_selection,
        qc_gate=args.g2_parent_gate,
        device=args.device,
    )
    masks = PreprocessedValidMaskStore(args.valid_mask_manifest)
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    dataset = infer_dataset_class(str(args.preprocessed_dir))(str(args.preprocessed_dir), sorted(masks.records))
    shared._render = render_without_contour
    output = ROOT / "smoke/k4_small24_r3"
    report = shared._run_set(kind="small24", manifest=manifest, config=config, backend=backend,
                             processor=processor, dataset=dataset, masks=masks, output=output)
    report.update({"route_id": "K4", "F_STRICT_PREREG_GATE": False,
                   "status": "hold_for_manual_review", "training_authorization": False,
                   "k5_authorization": False, "plan_sha256": sha256_file(smoke_plan),
                   "montage_renderer": "mask_overlay_without_contourpy",
                   "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")})
    report_path = ROOT / "evidence/K4_SMALL_SMOKE_AUTOMATIC.json"
    fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
