"""Single-GPU trainer for the isolated K1/K2/K4/K5 exploratory study."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from .met_aug_core import JsonlAuditSink, MetAugContractError, sha256_file, write_or_validate_immutable_json
from .met_aug_data_loader import PreprocessedValidMaskStore
from .met_aug_diffusion import G1FourModalityInpaintingBackend
from .met_aug_fix_v3 import FixV3CandidateProcessor
from .met_aug_k1245 import (
    K1245Config,
    K1245Engine,
    K1245Manifest,
    RCAwareFixV3CandidateProcessor,
)
from .met_aug_paired_training import configure_paired_training_runtime, seed_paired_training_epoch
from .met_aug_transform import MetAugRouteATransform
from .nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune import (
    _required_env,
    nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune,
)
from .nnUNetTrainerBraTS2026RCFocalCompletionFineTune import (
    nnUNetTrainerBraTS2026RCFocalCompletionFineTune,
)


class nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune(
    nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune
):
    """Run one preregistered K route after RC/X3 smoke authorization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = int(os.environ.get("S2_K1245_EPOCHS", "125"))
        self.initial_lr = float(os.environ.get("S2_K1245_INITIAL_LR", "0.001"))
        self.save_every = int(os.environ.get("S2_K1245_SAVE_EVERY", "25"))
        if (self.num_epochs, self.initial_lr, self.save_every) != (125, 0.001, 25):
            raise MetAugContractError("K1245 requires epochs=125, LR=0.001, save_every=25")
        self._k1245_config = None

    def _initialize_met_aug(self) -> None:
        if self._met_aug_transform is not None:
            return
        if os.environ.get("S2_MET_AUG_ENABLE") != "1":
            raise MetAugContractError("K1245 is fail-closed without S2_MET_AUG_ENABLE=1")
        manifest_path = _required_env("S2_K1245_COMPONENT_MANIFEST")
        config_path = _required_env("S2_K1245_CONFIG")
        plan_path = _required_env("S2_K1245_PLAN")
        authorization_path = _required_env("S2_K1245_SMOKE_AUTHORIZATION")
        valid_mask_manifest = _required_env("S2_MET_AUG_VALID_MASK_MANIFEST")
        legacy_manifest_path = _required_env("S2_K1245_LEGACY_COMPONENT_MANIFEST")
        calibration_path = _required_env("S2_MET_AUG_FIX_V3_CALIBRATION")
        g1_code_dir = _required_env("S2_MET_AUG_G1_CODE_DIR", directory=True)
        checkpoint_root = _required_env("S2_MET_AUG_G1_CHECKPOINT_ROOT", directory=True)
        selection_path = _required_env("S2_MET_AUG_G1_CHECKPOINT_SELECTION")
        gate_path = _required_env("S2_MET_AUG_G2_QC_GATE")
        manifest = K1245Manifest.load(manifest_path)
        config = K1245Config.load(config_path, manifest=manifest, plan_path=plan_path)
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        required_status = "pass" if config.route_id != "K5" else "conditional_pass"
        if (
            authorization.get("status") != required_status
            or authorization.get("plan_sha256") != sha256_file(plan_path)
            or authorization.get("manifest_sha256") != manifest.identity_sha256
            or authorization.get("route_id") != config.route_id
        ):
            raise MetAugContractError("K1245 smoke/scientific authorization is absent or drifted")
        self._met_aug_valid_masks = PreprocessedValidMaskStore(valid_mask_manifest)
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        processor = RCAwareFixV3CandidateProcessor.load(
            calibration_path,
            expected_sha256=config_payload["legacy_fix_v3_calibration_sha256"],
            expected_policy="label_only_qc_v1",
        )
        processor.bind_extended_manifest(
            legacy_manifest_sha256=config_payload["legacy_component_manifest_identity_sha256"],
            extended_manifest_sha256=manifest.identity_sha256,
        )
        backend = G1FourModalityInpaintingBackend(
            g1_code_dir=g1_code_dir,
            checkpoint_root=checkpoint_root,
            checkpoint_selection=selection_path,
            qc_gate=gate_path,
            device=self.device,
        )
        audit_path = Path(os.environ.get("S2_MET_AUG_AUDIT_PATH", str(Path(self.output_folder) / "met_aug_events.jsonl"))).resolve()
        engine = K1245Engine(
            manifest=manifest,
            config=config,
            backend=backend,
            audit_sink=JsonlAuditSink(audit_path),
            candidate_processor=processor,
        )
        self._met_aug_transform = MetAugRouteATransform(engine)
        self._k1245_config = config
        write_or_validate_immutable_json(
            Path(self.output_folder) / "met_aug_k1245_provenance.json",
            {
                "artifact_status": "experimental_unvalidated",
                "operator_approved": False,
                "formal_gate_status": "not_run_not_passed",
                "route_id": config.route_id,
                "plan_sha256": sha256_file(plan_path),
                "config_sha256": sha256_file(config_path),
                "component_manifest_sha256": manifest.identity_sha256,
                "legacy_component_manifest_file_sha256": sha256_file(legacy_manifest_path),
                "smoke_authorization_sha256": sha256_file(authorization_path),
                "g1_checkpoint_selection_sha256": sha256_file(selection_path),
                "g2_qc_gate_sha256": sha256_file(gate_path),
                "epochs": 125,
                "checkpoint_epochs": [25, 50, 75, 100, 125],
                "seed": 20260724,
            },
            label="K1245 provenance",
        )
        self.print_to_log_file(f"MET_AUG_K1245_INITIALIZED route={config.route_id}")

    def on_train_start(self):
        self._paired_training_seed = configure_paired_training_runtime()
        if self.is_ddp or self._do_i_compile():
            raise MetAugContractError("K1245 requires one non-DDP GPU and compile disabled")
        if not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark:
            raise MetAugContractError("K1245 deterministic cuDNN contract drifted")
        if not self.was_initialized:
            self.initialize()
        self._prepare_paired_second_stage()
        self.print_to_log_file("MET_AUG_K1245_RUNTIME_PASS epochs=125 save_every=25")
        nnUNetTrainerBraTS2026RCFocalCompletionFineTune.on_train_start(self)

    def on_train_epoch_start(self):
        self._paired_epoch_seed = seed_paired_training_epoch(self._paired_training_seed, self.current_epoch)
        self._before_paired_epoch()
        nnUNetTrainerBraTS2026RCFocalCompletionFineTune.on_train_epoch_start(self)

    def on_epoch_end(self):
        super().on_epoch_end()
        completed_epoch = int(self.current_epoch)
        if completed_epoch % 25 == 0:
            self.current_epoch -= 1
            try:
                self.save_checkpoint(str(Path(self.output_folder) / f"checkpoint_epoch_{completed_epoch:04d}.pth"))
            finally:
                self.current_epoch += 1
