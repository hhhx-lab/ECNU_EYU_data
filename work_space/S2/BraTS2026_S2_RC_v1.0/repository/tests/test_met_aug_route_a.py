from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from custom_nnunet.met_aug_core import (
    COMPACT_SUPPORT_ROUTE_CONFIG_SCHEMA,
    COMPONENT_MANIFEST_SCHEMA,
    VALID_MASK_MANIFEST_SCHEMA,
    CompactSupportEligibility,
    ComponentManifest,
    ComponentRecord,
    ComponentSampler,
    EventContext,
    MemoryAuditSink,
    MetAugAuditError,
    MetAugEngine,
    RouteConfig,
    canonical_json_sha256,
    make_route_a_config,
    patient_group,
    sha256_file,
    write_or_validate_immutable_json,
)
from custom_nnunet.met_aug_gate1 import run_gate1, split_event_ranges
from custom_nnunet.met_aug_transform import MetAugRouteATransform
from custom_nnunet.met_aug_diffusion import (
    G1_MODALITIES,
    G1FourModalityInpaintingBackend,
    _temporary_rng,
)


def _write_manifest(root: Path) -> tuple[Path, Path]:
    records: list[ComponentRecord] = []
    components = (
        ("component_a", "BraTS-MET-00001-000"),
        ("component_b", "BraTS-MET-00002-000"),
    )
    component_dir = root / "components"
    component_dir.mkdir()
    for component_id, source_case_id in components:
        path = component_dir / f"{component_id}.npz"
        np.savez_compressed(path, label=np.full((3, 3, 3), 3, dtype=np.int16))
        records.append(
            ComponentRecord(
                component_id=component_id,
                manifest_version="test",
                source_case_id=source_case_id,
                patient_group=source_case_id.rsplit("-", 1)[0],
                split="train",
                component_path=str(Path("components") / path.name),
                label_sha256=sha256_file(path),
                source_label_sha256="0" * 64,
                source_modalities_sha256={modality: "0" * 64 for modality in ("t1n", "t1c", "t2w", "t2f")},
                source_affine_sha256="0" * 64,
                spacing_mm=(1.0, 1.0, 1.0),
                core_volume_mm3=27.0,
                total_volume_mm3=27.0,
                bbox_mm=(3.0, 3.0, 3.0),
                bbox_voxels=(3, 3, 3),
                class_counts={"3": 27},
                classes_present=(3,),
                core_centroid_norm=(0.5, 0.5, 0.5),
            )
        )
    records_path = root / "components.jsonl"
    records_path.write_text(
        "".join(json.dumps(record.as_mapping(), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    groups_path = root / "target_case_groups.json"
    groups_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_to_patient_group": {
                    "target_a": "BraTS-MET-00001",
                    "target_b": "BraTS-MET-00002",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": COMPONENT_MANIFEST_SCHEMA,
        "manifest_version": "test",
        "coordinate_space": "nnUNetPlans_3d_fullres_preprocessed",
        "builder_code_sha256": "0" * 64,
        "component_core_sha256": "0" * 64,
        "nnunet_plans_sha256": "0" * 64,
        "train_file_sha256": "0" * 64,
        "mapping_csv_sha256": "0" * 64,
        "component_count": len(records),
        "records_file": records_path.name,
        "records_sha256": sha256_file(records_path),
        "target_groups_file": groups_path.name,
        "target_groups_sha256": sha256_file(groups_path),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest, exclude=("manifest_sha256",))
    manifest_path = root / "component_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    loaded = ComponentManifest.load(manifest_path)
    config = make_route_a_config(loaded, seed=17)
    config_path = root / "route_a.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return manifest_path, config_path


def _write_valid_mask_manifest(root: Path) -> Path:
    mask_dir = root / "valid_masks"
    mask_dir.mkdir()
    records = []
    for case_id in ("target_a", "target_b"):
        asset_path = mask_dir / f"{case_id}.npz"
        shape = (72, 72, 72)
        np.savez_compressed(
            asset_path,
            valid_mask=np.ones(shape, dtype=np.uint8),
            foreground_mask=np.zeros(shape, dtype=np.uint8),
        )
        records.append(
            {
                "case_id": case_id,
                "mask_path": str(Path("valid_masks") / asset_path.name),
                "sha256": sha256_file(asset_path),
                "shape": list(shape),
            }
        )
    records_path = root / "valid_mask_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    payload = {
        "schema_version": VALID_MASK_MANIFEST_SCHEMA,
        "records_file": records_path.name,
        "records_sha256": sha256_file(records_path),
        "train_count": len(records),
        "resampling_backend": "nnunet_configuration_resampling_fn_seg",
    }
    payload["manifest_sha256"] = canonical_json_sha256(
        payload, exclude=("manifest_sha256",)
    )
    path = root / "valid_mask_manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _replace_component_label(root: Path, component_id: str, label: np.ndarray) -> None:
    records_path = root / "components.jsonl"
    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    for row in rows:
        if row["component_id"] != component_id:
            continue
        component_path = root / row["component_path"]
        np.savez_compressed(component_path, label=label.astype(np.int16))
        support = label != 0
        core = np.isin(label, (1, 3))
        row.update({
            "label_sha256": sha256_file(component_path),
            "core_volume_mm3": float(np.count_nonzero(core)),
            "total_volume_mm3": float(np.count_nonzero(support)),
            "bbox_mm": [float(value) for value in label.shape],
            "bbox_voxels": list(label.shape),
            "class_counts": {
                str(value): int(np.count_nonzero(label == value))
                for value in sorted(int(item) for item in np.unique(label[support]))
            },
            "classes_present": sorted(int(item) for item in np.unique(label[support])),
        })
        break
    else:
        raise AssertionError(f"unknown fixture component: {component_id}")
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path = root / "component_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records_sha256"] = sha256_file(records_path)
    manifest["manifest_sha256"] = canonical_json_sha256(
        manifest, exclude=("manifest_sha256",)
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


class _ConstantBackend:
    def generate(self, image_crop: np.ndarray, label_crop: np.ndarray, *, seed: int) -> np.ndarray:
        del label_crop, seed
        return image_crop + 7.0


class _FailingBackend:
    def generate(self, image_crop: np.ndarray, label_crop: np.ndarray, *, seed: int) -> np.ndarray:
        del image_crop, label_crop, seed
        raise RuntimeError("forced modality failure")


class _FailingAuditSink:
    def append(self, event: dict) -> None:
        del event
        raise OSError("forced audit failure")


class MetAugRouteATests(unittest.TestCase):
    def test_patient_group_only_strips_three_digit_timepoint(self):
        self.assertEqual(patient_group("BraTS-MET-00001-000"), "BraTS-MET-00001")
        self.assertEqual(patient_group("BraTS-MET-00001-001"), "BraTS-MET-00001")
        self.assertEqual(patient_group("BraTS-MET-00001"), "BraTS-MET-00001")

    def test_compact_support_boundaries_are_inclusive(self):
        eligibility = CompactSupportEligibility(
            policy="compact_support_v1",
            max_total_support_voxels=4096,
            max_total_to_core_ratio=20.0,
            eligible_component_count=0,
            excluded_component_count=0,
            eligible_by_core_volume_bin={},
            eligible_patient_groups_by_core_volume_bin={},
        )

        self.assertTrue(
            eligibility.accepts_counts(total_support_voxels=4096, core_voxels=2048)
        )
        self.assertTrue(
            eligibility.accepts_counts(total_support_voxels=400, core_voxels=20)
        )
        self.assertFalse(
            eligibility.accepts_counts(total_support_voxels=4097, core_voxels=4097)
        )
        self.assertFalse(
            eligibility.accepts_counts(total_support_voxels=401, core_voxels=20)
        )

    def test_compact_support_config_filters_sampler_and_legacy_config_still_loads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, legacy_config_path = _write_manifest(root)
            legacy_manifest = ComponentManifest.load(manifest_path)
            legacy_config = RouteConfig.load(legacy_config_path, legacy_manifest)
            self.assertIsNone(legacy_config.donor_eligibility)

            _replace_component_label(
                root,
                "component_b",
                np.full((17, 17, 17), 3, dtype=np.int16),
            )
            manifest = ComponentManifest.load(manifest_path)
            compact_payload = make_route_a_config(
                manifest,
                seed=17,
                max_total_support_voxels=4096,
                max_total_to_core_ratio=20.0,
            )
            compact_path = root / "route_a_compact.json"
            compact_path.write_text(json.dumps(compact_payload, sort_keys=True))
            compact = RouteConfig.load(compact_path, manifest)
            sampler = ComponentSampler(manifest, compact)
            observed = {
                sampler.choose(np.random.default_rng(seed), "BraTS-MET-00002").component_id
                for seed in range(16)
            }

        self.assertEqual(compact.schema_version, COMPACT_SUPPORT_ROUTE_CONFIG_SCHEMA)
        self.assertEqual(compact.donor_eligibility.eligible_component_count, 1)
        self.assertEqual(compact.donor_eligibility.excluded_component_count, 1)
        self.assertEqual(observed, {"component_a"})

    def _engine(self, root: Path, backend) -> MetAugEngine:
        manifest_path, config_path = _write_manifest(root)
        manifest = ComponentManifest.load(manifest_path)
        return MetAugEngine(
            manifest=manifest,
            config=RouteConfig.load(config_path, manifest),
            backend=backend,
            audit_sink=MemoryAuditSink(),
        )

    @staticmethod
    def _context() -> EventContext:
        return EventContext(
            epoch=3,
            rank=0,
            worker=0,
            case_id="target_a",
            patch_index=4,
            patch_origin=(0, 0, 0),
            full_shape=(96, 96, 96),
        )

    def test_planner_excludes_same_patient_group_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = self._engine(Path(temporary), backend=None)
            segmentation = np.zeros((1, 96, 96, 96), dtype=np.int16)
            valid_mask = np.ones((96, 96, 96), dtype=bool)

            first = engine.plan(segmentation=segmentation, valid_mask=valid_mask, context=self._context())
            second = engine.plan(segmentation=segmentation, valid_mask=valid_mask, context=self._context())

        self.assertEqual(first.state, "PLACEMENT_VALID")
        self.assertEqual(first.event_id, second.event_id)
        self.assertEqual(first.event_seed, second.event_seed)
        self.assertIsNotNone(first.record)
        self.assertEqual(first.record.component_id, "component_b")
        self.assertEqual(first.placement.crop_start, second.placement.crop_start)

    def test_prevalidated_gate_input_has_identical_event_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = self._engine(Path(temporary), backend=None)
            segmentation = np.zeros((1, 96, 96, 96), dtype=np.int16)
            valid_mask = np.ones((96, 96, 96), dtype=bool)
            for patch_index in range(64):
                context = EventContext(
                    epoch=0,
                    rank=0,
                    worker=0,
                    case_id="target_a",
                    patch_index=patch_index,
                    full_shape=(96, 96, 96),
                )
                checked = engine.plan(
                    segmentation=segmentation,
                    valid_mask=valid_mask,
                    context=context,
                )
                prevalidated = engine.plan(
                    segmentation=segmentation,
                    valid_mask=valid_mask,
                    context=context,
                    inputs_prevalidated=True,
                )
                self.assertEqual(checked.audit_mapping(), prevalidated.audit_mapping())

    def test_gate1_parallel_output_is_byte_identical_to_serial(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, config_path = _write_manifest(root)
            manifest = ComponentManifest.load(manifest_path)
            config_path.write_text(
                json.dumps(
                    make_route_a_config(
                        manifest,
                        seed=17,
                        max_total_support_voxels=4096,
                        max_total_to_core_ratio=20.0,
                    ),
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            valid_mask_manifest = _write_valid_mask_manifest(root)
            common = {
                "component_manifest_path": manifest_path,
                "route_config_path": config_path,
                "valid_mask_manifest_path": valid_mask_manifest,
                "events": 256,
                "target_seed": 20260725,
                "minimum_events": 1,
                "enforce_acceptance": False,
            }
            serial_dir = root / "serial"
            parallel_dir = root / "parallel"
            serial_report = run_gate1(output_dir=serial_dir, workers=1, **common)
            parallel_report = run_gate1(output_dir=parallel_dir, workers=4, **common)

            self.assertEqual(serial_report, parallel_report)
            self.assertEqual(
                (serial_dir / "gate1_events.jsonl").read_bytes(),
                (parallel_dir / "gate1_events.jsonl").read_bytes(),
            )
            self.assertEqual(
                (serial_dir / "gate1_report.json").read_bytes(),
                (parallel_dir / "gate1_report.json").read_bytes(),
            )
            self.assertEqual(split_event_ranges(10, 3), [(0, 4), (4, 7), (7, 10)])

    def test_success_changes_only_final_support_and_writes_a_legal_segmentation(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = self._engine(Path(temporary), backend=_ConstantBackend())
            image = np.zeros((4, 96, 96, 96), dtype=np.float32)
            segmentation = np.zeros((1, 96, 96, 96), dtype=np.int16)
            result_image, result_segmentation, result = engine.apply(
                image=image,
                segmentation=segmentation,
                valid_mask=np.ones((96, 96, 96), dtype=bool),
                context=self._context(),
            )

        self.assertEqual(result.state, "COMMITTED")
        self.assertIsNotNone(result.placement)
        placement = result.placement
        start = placement.crop_start
        stop = tuple(value + 64 for value in start)
        support = np.zeros((96, 96, 96), dtype=bool)
        support[tuple(slice(begin, end) for begin, end in zip(start, stop))] = placement.support
        self.assertTrue(np.all(result_image[:, support] == 7.0))
        self.assertTrue(np.all(result_image[:, ~support] == 0.0))
        self.assertTrue(np.array_equal(result_segmentation[0][support], placement.label_cube[placement.support]))
        self.assertTrue(np.all(result_segmentation[0][~support] == 0))
        self.assertTrue(set(np.unique(result_segmentation)).issubset({0, 1, 2, 3, 4}))

    def test_backend_failure_returns_bit_identical_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = self._engine(Path(temporary), backend=_FailingBackend())
            image = np.random.default_rng(4).normal(size=(4, 96, 96, 96)).astype(np.float32)
            segmentation = np.zeros((1, 96, 96, 96), dtype=np.int16)
            image_before = image.copy()
            segmentation_before = segmentation.copy()
            result_image, result_segmentation, result = engine.apply(
                image=image,
                segmentation=segmentation,
                valid_mask=np.ones((96, 96, 96), dtype=bool),
                context=self._context(),
            )

        self.assertEqual(result.state, "NO_OP")
        self.assertEqual(result.reason, "MODALITY_QC_FAIL")
        self.assertTrue(np.array_equal(result_image, image_before))
        self.assertTrue(np.array_equal(result_segmentation, segmentation_before))

    def test_audit_failure_never_mutates_caller_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, config_path = _write_manifest(Path(temporary))
            manifest = ComponentManifest.load(manifest_path)
            engine = MetAugEngine(
                manifest=manifest,
                config=RouteConfig.load(config_path, manifest),
                backend=_ConstantBackend(),
                audit_sink=_FailingAuditSink(),
            )
            image = np.zeros((4, 96, 96, 96), dtype=np.float32)
            segmentation = np.zeros((1, 96, 96, 96), dtype=np.int16)
            with self.assertRaises(MetAugAuditError):
                engine.apply(
                    image=image,
                    segmentation=segmentation,
                    valid_mask=np.ones((96, 96, 96), dtype=bool),
                    context=self._context(),
                )

        self.assertTrue(np.array_equal(image, np.zeros_like(image)))
        self.assertTrue(np.array_equal(segmentation, np.zeros_like(segmentation)))

    def test_provenance_is_created_once_and_identical_resume_is_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "met_aug_provenance.json"
            payload = {"route_id": "MET-AUG-A", "focal_gamma": 2.0}

            self.assertEqual(
                write_or_validate_immutable_json(path, payload, label="MET-AUG provenance"),
                "created",
            )
            self.assertEqual(
                write_or_validate_immutable_json(path, payload, label="MET-AUG provenance"),
                "validated",
            )

    def test_provenance_drift_is_rejected_without_overwriting_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "met_aug_provenance.json"
            original = {"route_id": "MET-AUG-A", "route_config_sha256": "a" * 64}
            write_or_validate_immutable_json(path, original, label="MET-AUG provenance")
            before = path.read_bytes()

            with self.assertRaisesRegex(RuntimeError, "changed_keys=.*route_config_sha256"):
                write_or_validate_immutable_json(
                    path,
                    {**original, "route_config_sha256": "b" * 64},
                    label="MET-AUG provenance",
                )

            self.assertEqual(path.read_bytes(), before)

    def test_g1_backend_uses_explicit_support_for_zero_valued_brain_voxels(self):
        backend = G1FourModalityInpaintingBackend.__new__(G1FourModalityInpaintingBackend)
        backend.device = torch.device("cpu")
        backend.models = {modality: object() for modality in G1_MODALITIES}
        backend.n_steps = 1000
        backend.sampling_steps = 18
        backend.schedule_cfg = types.SimpleNamespace(
            betas=None,
            alphas_bar_sqrt=None,
            one_minus_alphas_bar_sqrt=None,
            alphas_bar=None,
        )
        backend.add_gaussian_noise_tumour_zscore = lambda image, label: (image.copy(), None)
        backend.sample_tumour_diffusion_inpaint = lambda **kwargs: torch.full(
            (1, 1, 64, 64, 64), 9.0, dtype=torch.float32
        )
        backend.correct_background_zscore = lambda *args: self.fail(
            "explicit support must not call G1's zero-value background sentinel"
        )
        image = np.zeros((4, 64, 64, 64), dtype=np.float32)
        label = np.zeros((64, 64, 64), dtype=np.int16)
        label[31, 32, 33] = 3

        generated = backend.generate(image, label, seed=123)

        support = label != 0
        self.assertTrue(np.all(generated[:, support] == 9.0))
        self.assertTrue(np.all(generated[:, ~support] == 0.0))

    def test_g1_temporary_rng_does_not_advance_nnunet_random_streams(self):
        np.random.seed(41)
        torch.manual_seed(41)
        expected = (np.random.random(), torch.rand(1).item())
        np.random.seed(41)
        torch.manual_seed(41)

        with _temporary_rng(999):
            _ = np.random.random(16)
            _ = torch.rand(16)

        observed = (np.random.random(), torch.rand(1).item())
        self.assertEqual(observed, expected)


class _CapturingEngine:
    def __init__(self) -> None:
        self.contexts: list[EventContext] = []

    def apply(self, *, image, segmentation, valid_mask, context):
        self.contexts.append(context)
        if image.shape != (4, 64, 64, 64) or segmentation.shape != (1, 64, 64, 64):
            raise AssertionError("transform did not receive one nnU-Net patch")
        if valid_mask.shape != (64, 64, 64):
            raise AssertionError("transform did not receive the aligned valid-mask patch")
        return image + 1.0, segmentation, object()


class MetAugTransformTests(unittest.TestCase):
    def test_transform_consumes_one_patch_and_removes_sidecars(self):
        engine = _CapturingEngine()
        transform = MetAugRouteATransform(engine)
        transform.set_epoch(9, rank=0, worker=0)
        result = transform(
            image=torch.zeros((4, 64, 64, 64), dtype=torch.float32),
            segmentation=torch.zeros((1, 64, 64, 64), dtype=torch.int16),
            met_aug_valid_mask=np.ones((64, 64, 64), dtype=np.uint8),
            met_aug_patch_origin=np.asarray((5, 6, 7), dtype=np.int32),
            met_aug_full_shape=np.asarray((100, 101, 102), dtype=np.int32),
            met_aug_case_id="target_a",
        )

        self.assertEqual(len(engine.contexts), 1)
        self.assertEqual(engine.contexts[0].case_id, "target_a")
        self.assertEqual(engine.contexts[0].epoch, 9)
        self.assertEqual(engine.contexts[0].patch_origin, (5, 6, 7))
        self.assertEqual(engine.contexts[0].full_shape, (100, 101, 102))
        self.assertTrue(torch.all(result["image"] == 1.0))
        self.assertEqual(
            set(result) & {
                "met_aug_valid_mask",
                "met_aug_patch_origin",
                "met_aug_full_shape",
                "met_aug_case_id",
            },
            set(),
        )


if __name__ == "__main__":
    unittest.main()
