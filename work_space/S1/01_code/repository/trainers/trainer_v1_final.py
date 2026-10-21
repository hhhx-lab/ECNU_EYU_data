"""S1 multi-task trainer with full-volume SWI validation and metric-based selection.

Key design choices (see user optimization plan):
  - Train: lesion-balanced 96^3 patches, nonzero Z-score, light aug
  - Val: full-volume joint sliding-window inference (both heads in one pass)
  - Best checkpoint: BraTS-compatible region Dice / lesion / small-F1 proxy score
  - Memory: batch_size=1 + grad accumulation + AMP (BF16 on Ampere+, else FP16)
  - Convergence: LR scheduler + early stopping on full-val checkpoint score
  - Monitor RC uncertainty weights so the RC branch is not silently down-weighted
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "datasets"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "losses"))
sys.path.insert(0, str(ROOT / "metrics"))
sys.path.insert(0, str(ROOT / "inference"))

from brats_multitask_dataset import BraTSMultiTaskDataset  # noqa: E402
from brats_validation_metrics import (  # noqa: E402
    BraTSValidationMetrics,
    compose_label_map,
)
from multitask_loss import MultiTaskLoss  # noqa: E402
from multitask_unet import MultiTaskUNet, model_kwargs_from_config  # noqa: E402
from sliding_window_multitask import (  # noqa: E402
    logits_to_label_maps,
    sliding_window_multitask,
)


# ---------------------------------------------------------------------------
# Config / path helpers
# ---------------------------------------------------------------------------

def resolve_config_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return ROOT / path


def resolve_repo_path(path) -> str:
    if path is None or path == "":
        return ""
    path = Path(str(path)).expanduser()
    if path.is_absolute():
        return str(path)
    return str(ROOT / path)


def load_and_resolve_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("validation", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("augmentation", {})
    cfg.setdefault("checkpoint", {})
    cfg.setdefault("logging", {})
    cfg.setdefault("inference", {})

    cfg["data"]["data_root"] = os.environ.get(
        "BRATS_TRAIN_ROOT", cfg["data"]["data_root"]
    )
    split_dir = os.environ.get("BRATS_SPLIT_DIR")
    if split_dir:
        cfg["data"]["train_split"] = str(
            Path(split_dir) / Path(cfg["data"]["train_split"]).name
        )
        cfg["data"]["val_split"] = str(
            Path(split_dir) / Path(cfg["data"]["val_split"]).name
        )
    cfg["data"]["train_split"] = os.environ.get(
        "BRATS_TRAIN_SPLIT", cfg["data"]["train_split"]
    )
    cfg["data"]["val_split"] = os.environ.get(
        "BRATS_VAL_SPLIT", cfg["data"]["val_split"]
    )
    cfg["data"]["data_root"] = resolve_repo_path(cfg["data"]["data_root"])
    cfg["data"]["train_split"] = resolve_repo_path(cfg["data"]["train_split"])
    cfg["data"]["val_split"] = resolve_repo_path(cfg["data"]["val_split"])
    cfg["train"]["resume"] = resolve_repo_path(
        os.environ.get("S1_RESUME", cfg["train"].get("resume", ""))
    )
    cfg["checkpoint"]["save_dir"] = resolve_repo_path(
        os.environ.get("S1_CHECKPOINT_DIR", cfg["checkpoint"]["save_dir"])
    )
    cfg["logging"]["tensorboard_dir"] = resolve_repo_path(
        os.environ.get("S1_TENSORBOARD_DIR", cfg["logging"]["tensorboard_dir"])
    )
    if "output_dir" in cfg.get("inference", {}):
        cfg["inference"]["output_dir"] = resolve_repo_path(
            os.environ.get("S1_OUTPUT_DIR", cfg["inference"]["output_dir"])
        )
    return cfg


def read_case_list(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def select_amp_dtype(cfg: dict) -> torch.dtype | None:
    """BF16 on Ampere+/Hopper when available; otherwise FP16. None disables AMP."""
    if not cfg["train"].get("amp", True):
        return None
    if not torch.cuda.is_available():
        return None

    requested = str(cfg["train"].get("amp_dtype", "auto")).lower()
    bf16_ok = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()

    if requested in ("bf16", "bfloat16"):
        if not bf16_ok:
            print("WARNING: BF16 requested but not supported; falling back to FP16")
            return torch.float16
        return torch.bfloat16
    if requested in ("fp16", "float16"):
        return torch.float16
    # auto: prefer BF16 on modern GPUs
    if bf16_ok:
        return torch.bfloat16
    return torch.float16


def build_scheduler(optimizer, cfg: dict):
    train_cfg = cfg["train"]
    name = str(train_cfg.get("scheduler", "cosine")).lower()
    if name in ("none", "null", "off", ""):
        return None, "none"
    if name in ("plateau", "reducelronplateau"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(train_cfg.get("lr_factor", 0.5)),
            patience=int(train_cfg.get("lr_patience", 8)),
            min_lr=float(train_cfg.get("min_lr", 1e-6)),
        )
        return scheduler, "plateau"
    # default: cosine annealing over full epoch budget
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg.get("epochs", 300)),
        eta_min=float(train_cfg.get("min_lr", 1e-6)),
    )
    return scheduler, "cosine"


def save_checkpoint(
    path: Path,
    epoch: int,
    model,
    criterion,
    optimizer,
    scheduler,
    best_score: float,
    metrics: dict | None = None,
    amp_dtype_name: str | None = None,
):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_score": best_score,
        "metrics": metrics or {},
        "amp_dtype": amp_dtype_name,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Train / validate loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    criterion,
    optimizer,
    scaler,
    train_loader,
    device,
    epoch: int,
    amp_dtype: torch.dtype | None,
    grad_accum_steps: int,
    writer: SummaryWriter,
    log_every: int = 5,
):
    model.train()
    criterion.train()

    running_loss = 0.0
    running_tumor = 0.0
    running_rc = 0.0
    running_w_t = 0.0
    running_w_r = 0.0
    optimizer.zero_grad(set_to_none=True)
    steps_seen = 0

    for step, batch in enumerate(train_loader):
        image = batch["image"].to(device, non_blocking=True)
        tumor = batch["tumor"].to(device, non_blocking=True)
        rc = batch["rc"].to(device, non_blocking=True)

        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(image)
                loss_dict = criterion(
                    outputs["tumor"], outputs["rc"], tumor, rc
                )
                loss = loss_dict["loss"] / grad_accum_steps
        else:
            outputs = model(image)
            loss_dict = criterion(outputs["tumor"], outputs["rc"], tumor, rc)
            loss = loss_dict["loss"] / grad_accum_steps

        if torch.isnan(loss):
            print("\n===== NAN DETECTED =====")
            print(batch["case"])
            print("tumor unique =", torch.unique(tumor))
            print("rc unique =", torch.unique(rc))
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = ((step + 1) % grad_accum_steps == 0) or (
            step + 1 == len(train_loader)
        )
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        steps_seen += 1
        running_loss += loss_dict["loss"].item()
        running_tumor += float(loss_dict["tumor_loss"])
        running_rc += float(loss_dict["rc_loss"])
        running_w_t += float(loss_dict["weight_tumor"])
        running_w_r += float(loss_dict["weight_rc"])

        if step % log_every == 0:
            print(
                f"epoch {epoch} step {step} "
                f"loss {loss_dict['loss'].item():.4f} "
                f"tumor {float(loss_dict['tumor_loss']):.4f} "
                f"rc {float(loss_dict['rc_loss']):.4f} "
                f"w_t {float(loss_dict['weight_tumor']):.3f} "
                f"w_r {float(loss_dict['weight_rc']):.3f}"
            )

    n = max(steps_seen, 1)
    avg = {
        "loss": running_loss / n,
        "tumor_loss": running_tumor / n,
        "rc_loss": running_rc / n,
        "weight_tumor": running_w_t / n,
        "weight_rc": running_w_r / n,
    }
    writer.add_scalar("train/loss", avg["loss"], epoch)
    writer.add_scalar("train/tumor_loss", avg["tumor_loss"], epoch)
    writer.add_scalar("train/rc_loss", avg["rc_loss"], epoch)
    writer.add_scalar("train/weight_tumor", avg["weight_tumor"], epoch)
    writer.add_scalar("train/weight_rc", avg["weight_rc"], epoch)
    writer.add_scalar(
        "train/lr", optimizer.param_groups[0]["lr"], epoch
    )
    return avg


@torch.no_grad()
def validate_full_volume(
    model,
    val_loader,
    device,
    epoch: int,
    cfg: dict,
    amp_dtype: torch.dtype | None,
    writer: SummaryWriter,
):
    """Full-volume joint SWI validation; select checkpoints by official-compatible metrics."""
    model.eval()

    val_cfg = cfg.get("validation", {})
    train_cfg = cfg["train"]
    model_cfg = cfg.get("model", {})

    roi_size = tuple(
        val_cfg.get("roi_size", train_cfg.get("patch_size", [96, 96, 96]))
    )
    sw_batch_size = int(val_cfg.get("sw_batch_size", 1))
    overlap = float(val_cfg.get("overlap", 0.5))
    tumor_classes = int(model_cfg.get("tumor_classes", 4))
    rc_classes = int(model_cfg.get("rc_classes", 2))

    metric_tracker = BraTSValidationMetrics(
        small_lesion_volume_mm3=float(
            val_cfg.get("small_lesion_volume_mm3", 27.0)
        ),
        small_overlap_threshold=float(
            val_cfg.get("small_overlap_threshold", 0.2)
        ),
        selection_weights=val_cfg.get("selection_weights"),
    )

    for batch_idx, batch in enumerate(val_loader):
        image = batch["image"].to(device, non_blocking=True)
        tumor_gt = batch["tumor"][0, 0].cpu().numpy().astype(np.uint8)
        rc_gt = batch["rc"][0, 0].cpu().numpy().astype(np.uint8)
        spacing = batch["spacing"][0].cpu().numpy().tolist()
        case = batch["case"][0] if isinstance(batch["case"], (list, tuple)) else batch["case"]

        outputs = sliding_window_multitask(
            model=model,
            image=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            tumor_classes=tumor_classes,
            rc_classes=rc_classes,
            amp_dtype=amp_dtype,
        )
        tumor_pred, rc_pred = logits_to_label_maps(outputs)
        tumor_pred = tumor_pred[0].cpu().numpy().astype(np.uint8)
        rc_pred = rc_pred[0].cpu().numpy().astype(np.uint8)

        pred_map = compose_label_map(tumor_pred, rc_pred)
        ref_map = compose_label_map(tumor_gt, rc_gt)
        metric_tracker.update(pred_map, ref_map, spacing=spacing)

        if batch_idx % 10 == 0:
            print(
                f"  val epoch {epoch} case {batch_idx + 1}/{len(val_loader)}: {case}"
            )

    metrics = metric_tracker.compute()
    score = float(metrics["checkpoint_score"])

    writer.add_scalar("val/checkpoint_score", score, epoch)
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            writer.add_scalar(f"val/{key}", float(value), epoch)

    print(
        f"epoch {epoch} full-val "
        f"score {score:.4f} "
        f"region_dice_mean {metrics.get('region_dice_mean', float('nan')):.4f} "
        f"lesion_dice_proxy_mean {metrics.get('lesion_dice_proxy_mean', float('nan')):.4f} "
        f"small_f1_proxy_mean {metrics.get('small_f1_proxy_mean', float('nan')):.4f} "
        f"dice_et {metrics.get('dice_et', float('nan')):.4f} "
        f"dice_tc {metrics.get('dice_tc', float('nan')):.4f} "
        f"dice_wt {metrics.get('dice_wt', float('nan')):.4f} "
        f"dice_rc {metrics.get('dice_rc', float('nan')):.4f}"
    )
    return metrics, score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="S1 multi-task trainer")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "multitask_v1.yaml"),
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    cfg = load_and_resolve_config(config_path)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available; training on CPU")

    train_cases = read_case_list(cfg["data"]["train_split"])
    val_cases = read_case_list(cfg["data"]["val_split"])

    patch_size = tuple(cfg["train"].get("patch_size", [96, 96, 96]))
    lesion_prob = float(cfg["train"].get("lesion_probability", 0.8))
    normalize = bool(cfg["data"].get("normalize", True))
    aug_cfg = cfg.get("augmentation", {"enabled": True})

    train_ds = BraTSMultiTaskDataset(
        train_cases,
        cfg["data"]["data_root"],
        patch_size=patch_size,
        train=True,
        lesion_probability=lesion_prob,
        augmentation=aug_cfg,
        normalize=normalize,
    )
    val_ds = BraTSMultiTaskDataset(
        val_cases,
        cfg["data"]["data_root"],
        patch_size=patch_size,
        train=False,
        normalize=normalize,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    # Full volumes may differ slightly; always batch size 1 for val.
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["validation"].get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )

    print("train =", len(train_ds))
    print("val   =", len(val_ds))
    print("patch =", patch_size)
    print("lesion_probability =", lesion_prob)
    print("normalize =", normalize)

    model = MultiTaskUNet(**model_kwargs_from_config(cfg.get("model", {}))).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params = {n_params / 1e6:.2f}M")

    loss_cfg = cfg.get("loss", {})
    criterion = MultiTaskLoss(
        use_uncertainty=bool(loss_cfg.get("use_uncertainty", True)),
        fixed_tumor_weight=float(loss_cfg.get("fixed_tumor_weight", 1.0)),
        fixed_rc_weight=float(loss_cfg.get("fixed_rc_weight", 1.0)),
        max_log_sigma=float(loss_cfg.get("max_log_sigma", 2.0)),
        min_log_sigma=float(loss_cfg.get("min_log_sigma", -2.0)),
    ).to(device)

    # Include uncertainty log-sigma parameters in the optimizer.
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=float(cfg["train"].get("lr", 3e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-5)),
    )

    amp_dtype = select_amp_dtype(cfg) if device.type == "cuda" else None
    amp_dtype_name = None if amp_dtype is None else str(amp_dtype).replace("torch.", "")
    # GradScaler is only needed for FP16; BF16 is numerically stable without it.
    use_scaler = amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler) if use_scaler else None

    scheduler, scheduler_name = build_scheduler(optimizer, cfg)
    grad_accum = max(1, int(cfg["train"].get("gradient_accumulation", 2)))
    effective_batch = int(cfg["train"].get("batch_size", 1)) * grad_accum
    print(
        f"batch_size={cfg['train'].get('batch_size', 1)} "
        f"grad_accum={grad_accum} effective_batch={effective_batch} "
        f"amp={amp_dtype_name} scheduler={scheduler_name}"
    )

    checkpoint_dir = Path(cfg["checkpoint"]["save_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg["logging"]["tensorboard_dir"])

    # Persist resolved config for reproducibility.
    with open(checkpoint_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    best_score = -1e9
    start_epoch = 0
    epochs_without_improve = 0
    early_stop_patience = int(cfg["train"].get("early_stop_patience", 30))
    val_every = max(1, int(cfg["validation"].get("every_n_epochs", 1)))
    min_epochs = int(cfg["train"].get("min_epochs", 20))

    resume_path = cfg["train"].get("resume", "")
    if resume_path:
        print("loading checkpoint:", resume_path)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "criterion" in ckpt:
            criterion.load_state_dict(ckpt["criterion"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_score = float(ckpt.get("best_score", best_score))
        print(f"resume from epoch {start_epoch}, best_score={best_score:.4f}")

    print("trainer initialized")
    metrics_history = []

    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            amp_dtype=amp_dtype,
            grad_accum_steps=grad_accum,
            writer=writer,
            log_every=int(cfg["logging"].get("log_every", 5)),
        )
        print(
            f"epoch {epoch} train_loss {train_stats['loss']:.4f} "
            f"w_t {train_stats['weight_tumor']:.3f} "
            f"w_r {train_stats['weight_rc']:.3f}"
        )

        # Warn if RC is being heavily down-weighted.
        if train_stats["weight_rc"] < 0.25 * max(train_stats["weight_tumor"], 1e-6):
            print(
                "WARNING: RC uncertainty weight is much smaller than tumor weight "
                f"(w_rc={train_stats['weight_rc']:.3f}, "
                f"w_tumor={train_stats['weight_tumor']:.3f}). "
                "Check loss.use_uncertainty / max_log_sigma."
            )

        run_val = ((epoch + 1) % val_every == 0) or (
            epoch + 1 == int(cfg["train"]["epochs"])
        )
        metrics = {}
        score = best_score
        if run_val:
            metrics, score = validate_full_volume(
                model=model,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                cfg=cfg,
                amp_dtype=amp_dtype,
                writer=writer,
            )
            metrics_history.append({"epoch": epoch, **metrics})
            with open(checkpoint_dir / "val_metrics_history.json", "w") as f:
                json.dump(metrics_history, f, indent=2)

            if scheduler is not None and scheduler_name == "plateau":
                scheduler.step(score)
            elif scheduler is not None:
                scheduler.step()

            improved = score > best_score + float(
                cfg["train"].get("early_stop_min_delta", 1e-4)
            )
            if improved:
                best_score = score
                epochs_without_improve = 0
                best_path = checkpoint_dir / "best.pth"
                save_checkpoint(
                    best_path,
                    epoch=epoch,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_score=best_score,
                    metrics=metrics,
                    amp_dtype_name=amp_dtype_name,
                )
                print(f"new best model score={best_score:.4f} -> {best_path}")
            else:
                epochs_without_improve += 1
                print(
                    f"no improvement ({epochs_without_improve}/{early_stop_patience})"
                )
        else:
            if scheduler is not None and scheduler_name != "plateau":
                scheduler.step()

        latest_path = checkpoint_dir / "latest.pth"
        save_checkpoint(
            latest_path,
            epoch=epoch,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            best_score=best_score,
            metrics=metrics,
            amp_dtype_name=amp_dtype_name,
        )

        if (
            run_val
            and epoch + 1 >= min_epochs
            and early_stop_patience > 0
            and epochs_without_improve >= early_stop_patience
        ):
            print(
                f"Early stopping at epoch {epoch}: "
                f"no full-val improvement for {early_stop_patience} checks "
                f"(best_score={best_score:.4f})"
            )
            break

    writer.close()
    print("training complete")
    print(f"best_score={best_score:.4f}")
    print(f"checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()
