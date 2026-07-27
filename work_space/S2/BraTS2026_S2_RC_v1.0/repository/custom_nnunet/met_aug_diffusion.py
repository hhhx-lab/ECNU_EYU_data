"""Frozen four-modality G1 Diffusion V3 backend for the Route A transaction."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Iterator

import numpy as np
import torch

try:
    from .met_aug_core import (
        MetAugContractError,
        S2_MODALITIES,
        canonical_json_sha256,
        sha256_file,
    )
    from .online_diffusion_contract import g1_to_s2_layout, s2_to_g1_layout
except ImportError:
    from met_aug_core import (  # type: ignore
        MetAugContractError,
        S2_MODALITIES,
        canonical_json_sha256,
        sha256_file,
    )
    from online_diffusion_contract import g1_to_s2_layout, s2_to_g1_layout


G1_MODALITIES = ("t1c", "t1n", "t2w", "t2f")
G1_RUNTIME_FILE_KEYS = (
    "GliGAN/src/infer/diffusion_inference_utils.py",
    "GliGAN/src/networks/DiffusionNetwork.py",
    "repository/model.py",
)


def _g1_runtime_paths(g1_code_dir: str | Path) -> dict[str, Path]:
    root = Path(g1_code_dir).expanduser().resolve()
    repository_root = root.parent.parent
    return {
        G1_RUNTIME_FILE_KEYS[0]: root / "src" / "infer" / "diffusion_inference_utils.py",
        G1_RUNTIME_FILE_KEYS[1]: root / "src" / "networks" / "DiffusionNetwork.py",
        G1_RUNTIME_FILE_KEYS[2]: repository_root / "model.py",
    }


def g1_runtime_code_snapshot(g1_code_dir: str | Path) -> dict[str, object]:
    """Hash every G1 source file that participates in Route A inference."""
    files: dict[str, str] = {}
    for label, path in _g1_runtime_paths(g1_code_dir).items():
        if not path.is_file():
            raise FileNotFoundError(f"G1 runtime file is missing: {path}")
        files[label] = sha256_file(path)
    return {
        "files": files,
        "sha256": canonical_json_sha256(files),
    }


def _require_module_origin(module: ModuleType, expected_path: str | Path, *, label: str) -> None:
    observed = getattr(module, "__file__", None)
    expected = Path(expected_path).expanduser().resolve()
    if not observed or Path(observed).expanduser().resolve() != expected:
        raise MetAugContractError(
            f"{label} resolved outside the approved G1 tree: observed={observed!r}, expected={expected}"
        )


def resolve_selected_checkpoint(
    checkpoint_root: str | Path,
    metadata: dict,
    *,
    modality: str,
) -> Path:
    """Resolve only the immutable canonical path recorded by G1 selection."""
    root = Path(checkpoint_root).expanduser().resolve()
    relative_value = metadata.get("canonical_relative_path")
    if not isinstance(relative_value, str) or not relative_value.strip():
        raise MetAugContractError(f"{modality} selection lacks canonical_relative_path")
    relative = Path(relative_value)
    if relative.is_absolute():
        raise MetAugContractError(f"{modality} checkpoint path must be relative to the archive root")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise MetAugContractError(f"{modality} checkpoint path escapes the archive root")
    expected_name = f"diffusion_{int(metadata['step'])}.pt"
    if path.name != expected_name:
        raise MetAugContractError(
            f"{modality} canonical checkpoint name does not match the selected step"
        )
    return path


@contextmanager
def _temporary_rng(seed: int) -> Iterator[None]:
    """Use G1's NumPy/Torch samplers without perturbing nnU-Net RNG streams."""
    bounded_seed = int(seed % (2**63 - 1))
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np.random.seed(bounded_seed % (2**32 - 1))
    torch.manual_seed(bounded_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(bounded_seed)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class G1FourModalityInpaintingBackend:
    """Load the approved G1 150k models once and generate a 64^3 S2 crop."""

    def __init__(
        self,
        *,
        g1_code_dir: str | Path,
        checkpoint_root: str | Path,
        checkpoint_selection: str | Path,
        qc_gate: str | Path,
        device: str | torch.device,
    ) -> None:
        self.g1_code_dir = Path(g1_code_dir).expanduser().resolve()
        self.checkpoint_root = Path(checkpoint_root).expanduser().resolve()
        self.selection_path = Path(checkpoint_selection).expanduser().resolve()
        self.gate_path = Path(qc_gate).expanduser().resolve()
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise MetAugContractError("MET-AUG diffusion requires an available CUDA device")
        if not self.g1_code_dir.is_dir() or not self.checkpoint_root.is_dir():
            raise FileNotFoundError("G1 code or checkpoint root is missing")
        self.runtime_code = g1_runtime_code_snapshot(self.g1_code_dir)
        self._prepare_imports()
        self.selection = json.loads(self.selection_path.read_text(encoding="utf-8"))
        self.gate = json.loads(self.gate_path.read_text(encoding="utf-8"))
        self._validate_external_contract()
        self._load_models()

    def _prepare_imports(self) -> None:
        infer_dir = self.g1_code_dir / "src" / "infer"
        for path in (self.g1_code_dir, infer_dir):
            if not path.is_dir():
                raise FileNotFoundError(f"G1 import directory is missing: {path}")
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        module = importlib.import_module("diffusion_inference_utils")
        paths = _g1_runtime_paths(self.g1_code_dir)
        _require_module_origin(module, paths[G1_RUNTIME_FILE_KEYS[0]], label="G1 inference module")
        network_module = importlib.import_module("src.networks.DiffusionNetwork")
        _require_module_origin(
            network_module,
            paths[G1_RUNTIME_FILE_KEYS[1]],
            label="G1 DiffusionNetwork module",
        )
        diffusion_utils = getattr(module, "_diffusion_utils", None)
        if not isinstance(diffusion_utils, ModuleType):
            raise MetAugContractError("G1 inference module did not expose its diffusion utility module")
        _require_module_origin(
            diffusion_utils,
            paths[G1_RUNTIME_FILE_KEYS[2]],
            label="G1 diffusion utility module",
        )

        required = (
            "add_gaussian_noise_tumour_zscore",
            "load_diffusion_model",
            "make_diffusion_coefficients",
            "sample_tumour_diffusion_inpaint",
        )
        missing = [name for name in required if not callable(getattr(module, name, None))]
        if missing:
            raise MetAugContractError(f"G1 inference module lacks required callables: {missing}")
        for name in required:
            setattr(self, name, getattr(module, name))

    def _validate_external_contract(self) -> None:
        if self.gate.get("decision") != "approve":
            raise MetAugContractError("G2 diffusion QC gate is not approved")
        if self.gate.get("checkpoint_selection_sha256") != sha256_file(self.selection_path):
            raise MetAugContractError("G2 gate does not bind the selected G1 checkpoint file")
        if self.selection.get("status") != "frozen":
            raise MetAugContractError("G1 checkpoint selection is not frozen")
        if self.selection.get("normalization") != "zscore":
            raise MetAugContractError("MET-AUG requires G1 z-score checkpoints")
        if self.selection.get("sampling_method") != "edm_heun" or int(self.selection.get("sampling_steps", -1)) != 18:
            raise MetAugContractError("MET-AUG requires the approved EDM-Heun 18-step sampler")
        if int(self.selection.get("crop_size", -1)) != 64:
            raise MetAugContractError("MET-AUG requires the approved 64^3 G1 crop")
        files = self.selection.get("checkpoint_files", {})
        if set(files) != set(G1_MODALITIES):
            raise MetAugContractError("G1 selection must specify all four modality checkpoints")

    def _load_models(self) -> None:
        class Args:
            pass

        self.args = Args()
        self.args.generator_type = "Unet_NnU"
        self.args.feature_size = 48
        self.args.use_checkpoint = False
        self.args.in_channels = 5  # one image channel plus BraTS 2024 TC/WT/ET/RC labels
        self.args.out_channels = 1
        self.args.crop_size = 64
        self.models: dict[str, torch.nn.Module] = {}
        metadata_reference: dict | None = None
        for modality in G1_MODALITIES:
            metadata = self.selection["checkpoint_files"][modality]
            path = resolve_selected_checkpoint(
                self.checkpoint_root,
                metadata,
                modality=modality,
            )
            if not path.is_file():
                raise FileNotFoundError(f"selected G1 checkpoint is missing: {path}")
            if path.stat().st_size != int(metadata["bytes"]):
                raise MetAugContractError(f"G1 checkpoint size drifted: {path}")
            if sha256_file(path) != metadata["sha256"]:
                raise MetAugContractError(f"G1 checkpoint SHA256 drifted: {path}")
            model, observed = self.load_diffusion_model(str(path), self.args, self.device, False)
            if observed.get("normalization") not in (None, "zscore"):
                raise MetAugContractError(f"{modality} checkpoint has unexpected normalization")
            if metadata_reference is None:
                metadata_reference = observed
            else:
                for key in ("n_steps", "noise_schedule", "crop_size", "in_channels", "out_channels"):
                    expected = metadata_reference.get(key)
                    actual = observed.get(key)
                    if expected is not None and actual is not None and expected != actual:
                        raise MetAugContractError(f"G1 checkpoint metadata mismatch: {modality}/{key}")
            self.models[modality] = model
        if metadata_reference is None:
            raise MetAugContractError("no G1 checkpoint was loaded")
        schedule_values = metadata_reference.get("schedule_config") or {}
        self.schedule_cfg = self.make_diffusion_coefficients(
            n_steps=int(metadata_reference["n_steps"]),
            noise_schedule=str(metadata_reference["noise_schedule"]),
            device=self.device,
            sigma_data=float(schedule_values.get("sigma_data", 0.5)),
            sigma_max=float(schedule_values.get("sigma_max", 50.0)),
            sigma_min=float(schedule_values.get("sigma_min", 0.002)),
            rho=float(schedule_values.get("rho", 7.0)),
            gamma_max=float(schedule_values.get("gamma_max", 10.0)),
            gamma_min=float(schedule_values.get("gamma_min", -10.0)),
            snr_shift=float(schedule_values.get("snr_shift", 0.0)),
        )
        self.n_steps = int(metadata_reference["n_steps"])
        self.sampling_steps = int(self.selection["sampling_steps"])

    @staticmethod
    def _label_to_multichannel(label: np.ndarray) -> np.ndarray:
        result = np.zeros((4,) + label.shape, dtype=np.float32)
        result[0] = np.isin(label, (1, 3)).astype(np.float32)  # TC
        result[1] = np.isin(label, (1, 2, 3)).astype(np.float32)  # WT
        result[2] = (label == 3).astype(np.float32)  # ET
        result[3] = (label == 4).astype(np.float32)  # RC, always zero for Route A
        return result

    def generate(self, image_crop: np.ndarray, label_crop: np.ndarray, *, seed: int) -> np.ndarray:
        if image_crop.shape != (4, 64, 64, 64) or label_crop.shape != (64, 64, 64):
            raise MetAugContractError("G1 backend requires a 4x64x64x64 S2 crop and 64^3 label")
        if not np.all(np.isfinite(image_crop)):
            raise MetAugContractError("input crop contains non-finite values")
        g1_image, g1_seg = s2_to_g1_layout(image_crop, label_crop[None].astype(np.int16, copy=False))
        g1_label = g1_seg[0]
        if not set(int(value) for value in np.unique(g1_label)).issubset({0, 1, 2, 3}):
            raise MetAugContractError("Route A label contains RC or unsupported classes")
        support = g1_label != 0
        if not np.any(support):
            raise MetAugContractError("Route A backend received an empty support")
        condition = self._label_to_multichannel(g1_label)
        condition_tensor = torch.from_numpy(condition).unsqueeze(0).to(self.device)
        generated_g1 = g1_image.copy()
        for index, modality in enumerate(G1_MODALITIES):
            with _temporary_rng(seed + index):
                noisy, _ = self.add_gaussian_noise_tumour_zscore(g1_image[index], g1_label)
                noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    generated = self.sample_tumour_diffusion_inpaint(
                        model=self.models[modality],
                        noisy_scan=noisy_tensor,
                        label_cond=condition_tensor,
                        n_steps=self.n_steps,
                        betas=self.schedule_cfg.betas,
                        alphas_bar_sqrt=self.schedule_cfg.alphas_bar_sqrt,
                        one_minus_alphas_bar_sqrt=self.schedule_cfg.one_minus_alphas_bar_sqrt,
                        device=self.device,
                        method="edm_heun",
                        sampling_steps=self.sampling_steps,
                        alphas_bar=self.schedule_cfg.alphas_bar,
                        noise_schedule_cfg=self.schedule_cfg,
                        cfg_weight=1.0,
                    )
                rebuilt = generated.squeeze(0).squeeze(0).detach().cpu().numpy()
                if not np.all(np.isfinite(rebuilt)):
                    raise MetAugContractError(f"{modality} generation is non-finite")
                # Route A has explicit label support. Values outside it are
                # never committed, so G1's legacy ``healthy_crop == 0``
                # background sentinel would incorrectly erase valid in-brain
                # z-score voxels that happen to equal zero.
                generated_g1[index][support] = rebuilt[support]
        output_s2, _ = g1_to_s2_layout(generated_g1, g1_seg)
        if output_s2.shape != image_crop.shape or not np.all(np.isfinite(output_s2)):
            raise MetAugContractError("G1/S2 layout conversion produced invalid output")
        return output_s2.astype(np.float32, copy=False)
