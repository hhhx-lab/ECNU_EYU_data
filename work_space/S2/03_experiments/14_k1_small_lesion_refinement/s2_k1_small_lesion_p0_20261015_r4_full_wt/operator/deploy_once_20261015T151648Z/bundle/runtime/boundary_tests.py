"""Real adapter/official evaluator tests in the frozen brats_eval environment."""
import json
import unittest
from pathlib import Path

from cpu_roundtrip import ROOT, require, sha
from official_identity import match_arrays, stable_id, validate_reference_fingerprint


def run_suite(evaluator):
    import numpy as np
    import SimpleITK as sitk
    plan = json.loads((ROOT / "CPU_EXECUTION_PLAN.json").read_text())
    checked, real_checked = [], []

    class Boundaries(unittest.TestCase):
        def compare(self, name, prediction, reference, spacing=(1., 1., 1.)):
            rows, fp = match_arrays((prediction != 0).astype(np.uint8), (reference != 0).astype(np.uint8))
            pred_image, ref_image = sitk.GetImageFromArray(prediction), sitk.GetImageFromArray(reference)
            for image in (pred_image, ref_image):
                image.SetSpacing(spacing)
            existing = evaluator.evaluate(pred_image, ref_image)["rc"].to_dict(True)
            validate_reference_fingerprint(rows, fp, existing, float(np.prod(spacing)))
            official_small = [r for r in existing["reference_instances"] if r["volume"] < 27]
            own_small = [r for r in rows if r["voxel_count"] * np.prod(spacing) < 27]
            tp = sum(r["is_matched"] and r["sq_dsc"] >= .2 for r in own_small)
            expected_tp = sum(r.get("is_matched", 0) and r["sq_dsc"] >= .2 for r in official_small)
            self.assertEqual(tp, expected_tp)
            self.assertEqual(len(own_small)-tp, len(official_small)-expected_tp)
            checked.append(dict(name=name, reference_count=len(rows), fp=fp,
                                small_tp=int(tp), small_fn=len(own_small)-int(tp),
                                official_fingerprint_equal=True))
            return rows, fp, existing

        def empty(self):
            return np.zeros((20, 20, 20), dtype=np.uint8)

        def two(self):
            array = self.empty()
            array[6, 6, 6] = 4
            array[12, 12, 12] = 4
            return array

        def test_both_empty(self):
            rows, fp, result = self.compare("both_empty", self.empty(), self.empty())
            self.assertEqual((rows, fp), ([], 0))
            self.assertTrue(np.isnan(result["sq_dsc"]))

        def test_empty_reference_two_fp(self):
            rows, fp, _ = self.compare("empty_reference_two_fp", self.two(), self.empty())
            self.assertEqual((rows, fp), ([], 2))

        def test_empty_prediction_two_fn(self):
            rows, fp, _ = self.compare("empty_prediction_two_fn", self.empty(), self.two())
            self.assertEqual((len(rows), fp), (2, 0))
            self.assertTrue(all(not r["is_matched"] and r["sq_dsc"] is None for r in rows))
            self.assertEqual(sorted(tuple(r["coords"][0]) for r in rows), [(6, 6, 6), (12, 12, 12)])

        def test_label4_two_normal_and_crop(self):
            rows, fp, _ = self.compare("label4_two_normal_crop", self.two(), self.two())
            self.assertEqual((len(rows), fp), (2, 0))
            self.assertTrue(all(r["sq_dsc"] == 1 for r in rows))
            self.assertEqual(sorted(tuple(r["coords"][0]) for r in rows), [(6, 6, 6), (12, 12, 12)])

        def test_26_connectivity(self):
            label = self.empty()
            label[6, 6, 6] = label[7, 7, 7] = 4
            rows, _, _ = self.compare("26_connectivity", label, label)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["voxel_count"], 2)

        def test_disjoint(self):
            prediction, reference = self.empty(), self.empty()
            prediction[12, 12, 12] = reference[6, 6, 6] = 4
            rows, fp, _ = self.compare("disjoint_fp_fn", prediction, reference)
            self.assertEqual(fp, 1)
            self.assertEqual(rows[0]["is_matched"], 0)

        def test_detection_threshold(self):
            for voxels in (8, 9, 10):
                with self.subTest(voxels=voxels):
                    prediction, reference = self.empty(), self.empty()
                    prediction[6, 6, 6:6+voxels] = reference[6, 6, 6] = 4
                    rows, fp, _ = self.compare("dsc_boundary_pred"+str(voxels), prediction, reference)
                    self.assertEqual(fp, 0)
                    self.assertAlmostEqual(rows[0]["sq_dsc"], 2/(voxels+1))
                    self.assertEqual(rows[0]["sq_dsc"] >= .2, voxels <= 9)

        def test_physical_small_threshold(self):
            for voxels, spacing in ((26, (1., 1., 1.)), (27, (1., 1., 1.)),
                                    (27, (.5, .5, .5)), (2, (.5, .5, .5))):
                with self.subTest(voxels=voxels, spacing=spacing):
                    label = self.empty()
                    coords = list(np.ndindex(3, 3, 3))[:voxels]
                    for z, y, x in coords:
                        label[z+6, y+6, x+6] = 4
                    rows, _, result = self.compare("physical_volume"+str((voxels, spacing)), label, label, spacing)
                    self.assertEqual(len(rows), 1)
                    volume = result["reference_instances"][0]["volume"]
                    self.assertEqual(volume, voxels * np.prod(spacing))
                    self.assertEqual(volume < 27, voxels != 27 or spacing != (1., 1., 1.))

        def test_equal_volume_identity(self):
            rows, _, _ = self.compare("equal_volume_distinct_id", self.two(), self.two())
            keys = [stable_id("case", "rc", "a"*64, dict(shape=[20]*3), r["coords"]) for r in rows]
            self.assertEqual(len(set(keys)), 2)

        def test_invalid_inputs_fail_closed(self):
            bad = self.empty().astype(float)
            bad[0, 0, 0] = np.nan
            with self.assertRaises(RuntimeError):
                match_arrays(bad, self.empty())
            with self.assertRaises(RuntimeError):
                match_arrays(self.empty()[0], self.empty()[0])
            with self.assertRaises(RuntimeError):
                match_arrays(self.two(), self.two())

        def test_real_fixed103_empty_and_nonempty_rc(self):
            prior = {m: json.loads((ROOT/"sources"/m/"panoptica_evaluation_summary.json").read_text())["metrics"] for m in ("K1", "K4")}
            selected = [next(c for c in prior["K1"] if len(c["rc"]["reference_instances"]) == 0),
                        next(c for c in prior["K1"] if len(c["rc"]["reference_instances"]) > 0)]
            for case in selected:
                case_id = case["subject_name"].removesuffix(".nii.gz")
                reference_path = Path(plan["cohort_root"]) / "reference" / case["subject_name"]
                require(sha(reference_path) == plan["input_reference_sha256"][str(reference_path)], "real GT drift")
                ref_image = sitk.ReadImage(str(reference_path))
                ref = sitk.GetArrayFromImage(ref_image)
                for method in ("K1", "K4"):
                    prediction_path = Path(plan["prediction_roots"][method]) / case["subject_name"]
                    require(sha(prediction_path) == plan["prediction_sha256"][method][prediction_path.name], "real prediction drift")
                    pred_image = sitk.ReadImage(str(prediction_path))
                    for getter in ("GetSpacing", "GetOrigin", "GetDirection", "GetSize"):
                        require(np.allclose(getattr(pred_image, getter)(), getattr(ref_image, getter)(), atol=1e-6, rtol=0), "real geometry mismatch")
                    pred = sitk.GetArrayFromImage(pred_image)
                    require(set(np.unique(pred)) <= set(range(5)) and set(np.unique(ref)) <= set(range(5)), "real illegal label")
                    rows, fp = match_arrays((pred == 4).astype(np.uint8), (ref == 4).astype(np.uint8))
                    old = next(c for c in prior[method] if c["subject_name"] == case["subject_name"])["rc"]
                    validate_reference_fingerprint(rows, fp, old, float(np.prod(ref_image.GetSpacing())))
                    real_checked.append(dict(subject_id=case_id, method=method, region="rc",
                        reference_count=len(rows), fp=fp, reference_sha256=sha(reference_path),
                        prediction_sha256=sha(prediction_path), original_official_fingerprint_equal=True))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Boundaries)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(), "real official boundary tests failed")
    require(len(real_checked) == 4, "real cases not checked")
    return dict(status="pass", actual_official_import=True, unittest_count=result.testsRun,
                synthetic_checks=checked, real_fixed103_checks=real_checked,
                zero_region_policy="official component counts, no matcher on zero instances",
                no_old_evaluation_modified=True, scope="small real subset plus synthetic boundaries; not full103")
