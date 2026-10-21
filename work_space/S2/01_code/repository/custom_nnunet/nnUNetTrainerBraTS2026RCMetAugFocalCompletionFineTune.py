"""Frozen E base plus the single-variable, train-only MET-AUG-A route."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from .met_aug_core import (
        ComponentManifest,
        JsonlAuditSink,
        MetAugContractError,
        MetAugEngine,
        RouteConfig,
        sha256_file,
        write_or_validate_immutable_json,
    )
    from .met_aug_data_loader import MetAugDataLoader, PreprocessedValidMaskStore
    from .met_aug_diffusion import G1FourModalityInpaintingBackend
    from .met_aug_fix_v2 import FixV2CandidateProcessor
    from .met_aug_fix_v3 import FixV3CandidateProcessor
    from .met_aug_gate import validate_route_a_approval
    from .met_aug_paired_training import MetAugPairedSecondStageMixin
    from .met_aug_transform import MetAugRouteATransform
    from .nnUNetTrainerBraTS2026RCFocalCompletionFineTune import (
        nnUNetTrainerBraTS2026RCFocalCompletionFineTune,
    )
except ImportError:
    from met_aug_core import (
        ComponentManifest,
        JsonlAuditSink,
        MetAugContractError,
        MetAugEngine,
        RouteConfig,
        sha256_file,
        write_or_validate_immutable_json,
    )
    from met_aug_data_loader import MetAugDataLoader, PreprocessedValidMaskStore
    from met_aug_diffusion import G1FourModalityInpaintingBackend
    from met_aug_fix_v2 import FixV2CandidateProcessor
    from met_aug_fix_v3 import FixV3CandidateProcessor
    from met_aug_gate import validate_route_a_approval
    from met_aug_paired_training import MetAugPairedSecondStageMixin
    from met_aug_transform import MetAugRouteATransform
    from nnUNetTrainerBraTS2026RCFocalCompletionFineTune import (
        nnUNetTrainerBraTS2026RCFocalCompletionFineTune,
    )


def _required_env(name: str, *, directory: bool = False) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise MetAugContractError(f"missing required MET-AUG environment variable: {name}")
    path = Path(value).expanduser().resolve()
    if (directory and not path.is_dir()) or (not directory and not path.is_file()):
        raise FileNotFoundError(f"invalid {name} path: {path}")
    return path


class nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune(
    MetAugPairedSecondStageMixin,
    nnUNetTrainerBraTS2026RCFocalCompletionFineTune,
):
    """Second-stage E/Focal adaptation with Route A after all contracts match."""

    authorization_env_name = "S2_MET_AUG_ROUTE_GATE"
    authorization_kind = "formal_gate_approval"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._met_aug_transform = None
        self._met_aug_valid_masks = None

    def _validate_met_aug_authorization(
        self,
        authorization_path: Path,
        *,
        manifest_path: Path,
        manifest: ComponentManifest,
        config: RouteConfig,
        config_path: Path,
        valid_mask_manifest: Path,
        calibration_path: Path | None,
        selection_path: Path,
        gate_path: Path,
        g1_code_dir: Path,
    ):
        del manifest_path, config, calibration_path
        return validate_route_a_approval(
            authorization_path,
            component_manifest=manifest,
            route_config_path=config_path,
            valid_mask_manifest_path=valid_mask_manifest,
            g1_checkpoint_selection_path=selection_path,
            g2_parent_gate_path=gate_path,
            g1_code_dir=g1_code_dir,
            code_dir=Path(__file__).resolve().parent,
        )

    def _initialize_met_aug(self) -> None:
        if self._met_aug_transform is not None:
            return
        if os.environ.get("S2_MET_AUG_ENABLE", "0") != "1":
            raise MetAugContractError(
                "MET-AUG trainer is fail-closed; set S2_MET_AUG_ENABLE=1 only after Gate 1/2 approval"
            )
        manifest_path = _required_env("S2_MET_AUG_COMPONENT_MANIFEST")
        config_path = _required_env("S2_MET_AUG_ROUTE_CONFIG")
        valid_mask_manifest = _required_env("S2_MET_AUG_VALID_MASK_MANIFEST")
        authorization_path = _required_env(self.authorization_env_name)
        g1_code_dir = _required_env("S2_MET_AUG_G1_CODE_DIR", directory=True)
        checkpoint_root = _required_env("S2_MET_AUG_G1_CHECKPOINT_ROOT", directory=True)
        selection_path = _required_env("S2_MET_AUG_G1_CHECKPOINT_SELECTION")
        gate_path = _required_env("S2_MET_AUG_G2_QC_GATE")
        manifest = ComponentManifest.load(manifest_path)
        config = RouteConfig.load(config_path, manifest)
        candidate_processor = None
        calibration_path = None
        if config.fix_v2 is not None:
            calibration_path = _required_env("S2_MET_AUG_FIX_V2_CALIBRATION")
            candidate_processor = FixV2CandidateProcessor.load(
                calibration_path,
                expected_sha256=config.fix_v2.calibration_sha256,
                expected_policy=config.fix_v2.boundary_policy,
            )
        if config.fix_v3 is not None:
            calibration_path = _required_env("S2_MET_AUG_FIX_V3_CALIBRATION")
            candidate_processor = FixV3CandidateProcessor.load(
                calibration_path,
                expected_sha256=config.fix_v3.calibration_sha256,
                expected_policy=config.fix_v3.boundary_policy,
            )
        self._met_aug_valid_masks = PreprocessedValidMaskStore(valid_mask_manifest)
        authorization = self._validate_met_aug_authorization(
            authorization_path,
            manifest_path=manifest_path,
            manifest=manifest,
            config=config,
            config_path=config_path,
            valid_mask_manifest=valid_mask_manifest,
            calibration_path=calibration_path,
            selection_path=selection_path,
            gate_path=gate_path,
            g1_code_dir=g1_code_dir,
        )
        backend = G1FourModalityInpaintingBackend(
            g1_code_dir=g1_code_dir,
            checkpoint_root=checkpoint_root,
            checkpoint_selection=selection_path,
            qc_gate=gate_path,
            device=self.device,
        )
        audit_path = Path(
            os.environ.get("S2_MET_AUG_AUDIT_PATH", str(Path(self.output_folder) / "met_aug_events.jsonl"))
        ).expanduser().resolve()
        engine = MetAugEngine(
            manifest=manifest,
            config=config,
            backend=backend,
            audit_sink=JsonlAuditSink(audit_path),
            candidate_processor=candidate_processor,
        )
        self._met_aug_transform = MetAugRouteATransform(engine)
        provenance = {
            "route_id": config.route_id,
            "component_manifest": str(manifest_path),
            "component_manifest_sha256": manifest.identity_sha256,
            "route_config": str(config_path),
            "route_config_sha256": sha256_file(config_path),
            "valid_mask_manifest": str(valid_mask_manifest),
            "valid_mask_manifest_sha256": self._met_aug_valid_masks.identity_sha256,
            "authorization_kind": self.authorization_kind,
            "authorization": str(authorization_path),
            "authorization_sha256": sha256_file(authorization_path),
            "authorization_decision": authorization["decision"],
            "g1_checkpoint_selection": str(selection_path),
            "g1_checkpoint_selection_sha256": sha256_file(selection_path),
            "g1_runtime_code_sha256": backend.runtime_code["sha256"],
            "g2_qc_gate": str(gate_path),
            "g2_qc_gate_sha256": sha256_file(gate_path),
            "audit_path": str(audit_path),
            "base_trainer": "nnUNetTrainerBraTS2026RCFocalCompletionFineTune",
            "training_contract": authorization["training_contract"],
        }
        if config.fix_v2 is not None:
            assert calibration_path is not None
            provenance["fix_v2"] = {
                "boundary_policy": config.fix_v2.boundary_policy,
                "calibration": str(calibration_path),
                "calibration_sha256": sha256_file(calibration_path),
            }
        if config.fix_v3 is not None:
            assert calibration_path is not None
            provenance["fix_v3"] = {
                "processor_policy": config.fix_v3.processor_policy,
                "boundary_policy": config.fix_v3.boundary_policy,
                "calibration": str(calibration_path),
                "calibration_sha256": sha256_file(calibration_path),
                "formal_validation_status": authorization["formal_validation_status"],
                "skipped_stages": authorization["skipped_stages"],
            }
        provenance_path = Path(self.output_folder) / "met_aug_provenance.json"
        provenance_state = write_or_validate_immutable_json(
            provenance_path,
            provenance,
            label="MET-AUG provenance",
        )
        self.print_to_log_file(
            "MET_AUG_ROUTE_A_INITIALIZED "
            f"manifest={manifest.identity_sha256} route={config_path} audit={audit_path} "
            f"provenance={provenance_state}"
        )

    def get_pre_spatial_training_transforms(self):
        if self._met_aug_transform is None:
            raise MetAugContractError("MET-AUG bridge was not initialized before data-loader creation")
        return [self._met_aug_transform]

    def build_training_dataloader(self, dataset_tr, initial_patch_size, transforms):
        if self._met_aug_valid_masks is None:
            raise MetAugContractError("MET-AUG valid masks are not initialized")
        return MetAugDataLoader(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            valid_mask_store=self._met_aug_valid_masks,
        )

    def _prepare_paired_second_stage(self):
        self._initialize_met_aug()

    def _before_paired_epoch(self):
        if self._met_aug_transform is not None:
            self._met_aug_transform.set_epoch(self.current_epoch, rank=self.local_rank, worker=0)
