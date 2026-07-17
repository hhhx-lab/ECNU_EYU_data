import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from synthesis.spatial import (
    SpatialTransform,
    assert_support_contained,
    build_spatial_transform,
    resample_labels_to_model,
    resample_to_model,
    restore_to_native,
)
from synthesis.utils import (
    load_image_in_model_space,
    load_segmentation_in_model_space,
    prepare_subject_space,
    resize_center_crop_pad,
)


class SpatialTransformTests(unittest.TestCase):
    def setUp(self):
        self.native_shape = (64, 64, 40)
        self.native_affine = np.diag([0.5, 0.5, 0.5, 1.0])

        self.image = np.zeros(self.native_shape, dtype=np.float32)
        self.image[4:60, 5:59, 3:37] = 0.5
        self.image[7:11, 28:34, 16:22] = 1.0

        self.seg = np.zeros(self.native_shape, dtype=np.int16)
        self.seg[7:11, 28:34, 16:22] = 3

    def test_physical_transform_keeps_lesion_lost_by_fixed_voxel_crop(self):
        old_seg, _ = resize_center_crop_pad(self.seg, (32, 32, 20))
        self.assertEqual(int(old_seg.sum()), 0)

        transform = build_spatial_transform(
            [self.image, self.image, self.image],
            self.native_affine,
            segmentation=self.seg,
            target_shape=(32, 32, 20),
            base_spacing_mm=1.0,
            margin_mm=1.0,
        )
        audit = assert_support_contained(self.seg > 0, transform, "lesion")
        model_seg = resample_to_model(self.seg, transform, order=0)

        self.assertEqual(audit["outside_voxel_count"], 0)
        self.assertGreater(int(np.count_nonzero(model_seg)), 0)
        self.assertEqual(set(np.unique(model_seg)), {0, 3})

    def test_model_space_roundtrip_restores_native_shape_affine_and_signal(self):
        transform = build_spatial_transform(
            [self.image, self.image, self.image],
            self.native_affine,
            segmentation=self.seg,
            target_shape=(32, 32, 20),
            base_spacing_mm=1.0,
            margin_mm=1.0,
        )
        model_image = resample_to_model(self.image, transform, order=1)
        restored = restore_to_native(model_image, transform, order=1)

        self.assertEqual(restored.shape, self.native_shape)
        np.testing.assert_allclose(transform.native_affine, self.native_affine)
        self.assertGreater(float(restored[self.seg > 0].mean()), 0.8)

    def test_restore_to_native_accepts_float16_model_output(self):
        transform = build_spatial_transform(
            [self.image, self.image, self.image],
            self.native_affine,
            segmentation=self.seg,
            target_shape=(32, 32, 20),
            base_spacing_mm=1.0,
            margin_mm=1.0,
        )

        model_image = resample_to_model(self.image, transform, order=1).astype(np.float16)
        restored = restore_to_native(model_image, transform, order=1)

        self.assertEqual(restored.shape, self.native_shape)
        self.assertEqual(restored.dtype, np.float32)
        self.assertTrue(np.isfinite(restored).all())
        self.assertGreater(float(restored[self.seg > 0].mean()), 0.8)

    def test_transform_metadata_roundtrip_is_lossless(self):
        transform = build_spatial_transform(
            [self.image, self.image, self.image],
            self.native_affine,
            segmentation=self.seg,
            target_shape=(32, 32, 20),
            base_spacing_mm=1.0,
            margin_mm=1.0,
        )

        restored = SpatialTransform.from_dict(
            json.loads(json.dumps(transform.to_dict()))
        )

        self.assertEqual(restored.native_shape, transform.native_shape)
        self.assertEqual(restored.target_shape, transform.target_shape)
        np.testing.assert_allclose(restored.native_affine, transform.native_affine)
        np.testing.assert_allclose(restored.target_affine, transform.target_affine)
        self.assertEqual(restored.foreground_voxel_count, transform.foreground_voxel_count)

    def test_label_resampling_preserves_a_submillimeter_single_voxel_lesion(self):
        tiny_seg = np.zeros(self.native_shape, dtype=np.int16)
        tiny_seg[7, 29, 17] = 4
        transform = build_spatial_transform(
            [self.image, self.image, self.image],
            self.native_affine,
            segmentation=tiny_seg,
            target_shape=(32, 32, 20),
            base_spacing_mm=1.0,
            margin_mm=1.0,
        )

        model_seg = resample_labels_to_model(tiny_seg, transform)

        self.assertIn(4, np.unique(model_seg))
        self.assertGreaterEqual(int(np.count_nonzero(model_seg == 4)), 1)

    def test_subject_preparation_applies_one_transform_to_all_modalities_and_seg(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            modality_paths = []
            for index, modality in enumerate(("t1n", "t1c", "t2f")):
                path = root / f"case-{modality}.nii.gz"
                nib.save(
                    nib.Nifti1Image(self.image + index * 0.05, self.native_affine),
                    path,
                )
                modality_paths.append(path)
            seg_path = root / "case-seg.nii.gz"
            nib.save(nib.Nifti1Image(self.seg, self.native_affine), seg_path)

            prepared = prepare_subject_space(
                modality_paths,
                seg_path=seg_path,
                target_shape=(32, 32, 20),
                base_spacing_mm=1.0,
                margin_mm=1.0,
            )

            self.assertEqual(len(prepared["images"]), 3)
            self.assertTrue(all(image.shape == (32, 32, 20) for image in prepared["images"]))
            self.assertEqual(prepared["segmentation"].shape, (32, 32, 20))
            self.assertGreater(int(np.count_nonzero(prepared["segmentation"])), 0)
            self.assertEqual(prepared["lesion_support_audit"]["outside_voxel_count"], 0)
            np.testing.assert_allclose(
                prepared["transform"].native_affine, self.native_affine
            )

    def test_subject_preparation_rejects_misaligned_modalities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "case-t1n.nii.gz"
            second = root / "case-t1c.nii.gz"
            nib.save(nib.Nifti1Image(self.image, self.native_affine), first)
            shifted_affine = self.native_affine.copy()
            shifted_affine[0, 3] += 2.0
            nib.save(nib.Nifti1Image(self.image, shifted_affine), second)

            with self.assertRaisesRegex(ValueError, "affine mismatch"):
                prepare_subject_space(
                    [first, second],
                    target_shape=(32, 32, 20),
                    base_spacing_mm=1.0,
                    margin_mm=1.0,
                )

    def test_saved_transform_drives_auxiliary_segmentation_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seg_path = root / "case-seg.nii.gz"
            nib.save(nib.Nifti1Image(self.seg, self.native_affine), seg_path)
            transform = build_spatial_transform(
                [self.image, self.image, self.image],
                self.native_affine,
                segmentation=self.seg,
                target_shape=(32, 32, 20),
                base_spacing_mm=1.0,
                margin_mm=1.0,
            )
            metadata_path = root / "spatial_transform.json"
            metadata_path.write_text(
                json.dumps({"transform": transform.to_dict()}) + "\n"
            )

            model_seg = load_segmentation_in_model_space(seg_path, metadata_path)
            image_path = root / "case-t1n.nii.gz"
            nib.save(nib.Nifti1Image(self.image, self.native_affine), image_path)
            model_image = load_image_in_model_space(image_path, metadata_path)

            self.assertEqual(model_seg.shape, (32, 32, 20))
            self.assertIn(3, np.unique(model_seg))
            self.assertEqual(model_image.shape, (32, 32, 20))
            self.assertGreater(float(model_image.max()), 0.9)


if __name__ == "__main__":
    unittest.main()
