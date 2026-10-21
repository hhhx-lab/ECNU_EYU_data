import math

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


# These groups match BraTS_evaluation config_mets.yaml. RC is evaluated separately
# and is intentionally not included in TC or WT.
REGION_LABELS = {
    "et": (3,),
    "rc": (4,),
    "tc": (1, 3),
    "wt": (1, 2, 3),
}


def compose_label_map(tumor, rc):
    tumor = np.asarray(tumor, dtype=np.uint8)
    rc = np.asarray(rc)
    if tumor.shape != rc.shape:
        raise ValueError(f"tumor/RC shape mismatch: {tumor.shape} != {rc.shape}")
    output = tumor.copy()
    output[rc > 0] = 4
    return output


def binary_dice(prediction, reference):
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    denominator = int(prediction.sum()) + int(reference.sum())
    if denominator == 0:
        return math.nan
    return 2.0 * float(np.logical_and(prediction, reference).sum()) / denominator


def _component_matching(prediction, reference):
    structure = ndimage.generate_binary_structure(3, 3)
    pred_components, pred_count = ndimage.label(prediction, structure=structure)
    ref_components, ref_count = ndimage.label(reference, structure=structure)
    pred_sizes = np.bincount(pred_components.ravel(), minlength=pred_count + 1)
    ref_sizes = np.bincount(ref_components.ravel(), minlength=ref_count + 1)

    dice_matrix = np.zeros((ref_count, pred_count), dtype=np.float64)
    for ref_id in range(1, ref_count + 1):
        overlapping, intersections = np.unique(
            pred_components[ref_components == ref_id],
            return_counts=True,
        )
        for pred_id, intersection in zip(overlapping, intersections):
            if pred_id == 0:
                continue
            dice_matrix[ref_id - 1, pred_id - 1] = (
                2.0 * float(intersection) / (ref_sizes[ref_id] + pred_sizes[pred_id])
            )

    matches = {}
    matched_predictions = set()
    if ref_count and pred_count:
        ref_indices, pred_indices = linear_sum_assignment(-dice_matrix)
        for ref_index, pred_index in zip(ref_indices, pred_indices):
            score = float(dice_matrix[ref_index, pred_index])
            if score > 0:
                matches[ref_index + 1] = (pred_index + 1, score)
                matched_predictions.add(pred_index + 1)

    return {
        "ref_count": ref_count,
        "pred_count": pred_count,
        "ref_sizes": ref_sizes,
        "matches": matches,
        "matched_predictions": matched_predictions,
    }


class BraTSValidationMetrics:
    """Full-volume checkpoint metrics aligned with the official MET regions.

    The lesion and small-instance values are training-time proxies. Final reporting
    must still be produced by the bundled official BraTS_evaluation package.
    """

    def __init__(
        self,
        small_lesion_volume_mm3=27.0,
        small_overlap_threshold=0.2,
        selection_weights=None,
    ):
        self.small_lesion_volume_mm3 = float(small_lesion_volume_mm3)
        self.small_overlap_threshold = float(small_overlap_threshold)
        self.selection_weights = selection_weights or {
            "region_dice_mean": 0.5,
            "lesion_dice_proxy_mean": 0.25,
            "small_f1_proxy_mean": 0.25,
        }
        self.region_dice = {region: [] for region in REGION_LABELS}
        self.lesion_dice = {region: [] for region in REGION_LABELS}
        self.small_counts = {
            region: {"tp": 0, "fp": 0, "fn": 0}
            for region in REGION_LABELS
        }
        self.case_count = 0

    def update(self, prediction, reference, spacing=(1.0, 1.0, 1.0)):
        prediction = np.asarray(prediction)
        reference = np.asarray(reference)
        if prediction.shape != reference.shape:
            raise ValueError(
                f"prediction/reference shape mismatch: {prediction.shape} != {reference.shape}"
            )
        if not set(np.unique(prediction)).issubset({0, 1, 2, 3, 4}):
            raise ValueError(f"prediction has illegal labels: {np.unique(prediction)}")
        if not set(np.unique(reference)).issubset({0, 1, 2, 3, 4}):
            raise ValueError(f"reference has illegal labels: {np.unique(reference)}")

        voxel_volume = float(np.prod(np.asarray(spacing, dtype=np.float64)))
        for region, labels in REGION_LABELS.items():
            pred_mask = np.isin(prediction, labels)
            ref_mask = np.isin(reference, labels)
            score = binary_dice(pred_mask, ref_mask)
            if math.isfinite(score):
                self.region_dice[region].append(score)

            matching = _component_matching(pred_mask, ref_mask)
            ref_count = matching["ref_count"]
            pred_count = matching["pred_count"]
            matches = matching["matches"]

            lesion_scores = [
                matches.get(ref_id, (None, 0.0))[1]
                for ref_id in range(1, ref_count + 1)
            ]
            lesion_scores.extend(
                [0.0] * (pred_count - len(matching["matched_predictions"]))
            )
            self.lesion_dice[region].extend(lesion_scores)

            small_ref_ids = [
                ref_id
                for ref_id in range(1, ref_count + 1)
                if matching["ref_sizes"][ref_id] * voxel_volume
                < self.small_lesion_volume_mm3
            ]
            small_tp = sum(
                1
                for ref_id in small_ref_ids
                if ref_id in matches
                and matches[ref_id][1] >= self.small_overlap_threshold
            )
            self.small_counts[region]["tp"] += small_tp
            self.small_counts[region]["fn"] += len(small_ref_ids) - small_tp
            self.small_counts[region]["fp"] += (
                pred_count - len(matching["matched_predictions"])
            )

        self.case_count += 1

    @staticmethod
    def _mean(values):
        return float(np.mean(values)) if values else math.nan

    def compute(self):
        metrics = {"case_count": self.case_count}
        for region in REGION_LABELS:
            metrics[f"dice_{region}"] = self._mean(self.region_dice[region])
            metrics[f"lesion_dice_proxy_{region}"] = self._mean(
                self.lesion_dice[region]
            )
            counts = self.small_counts[region]
            denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
            metrics[f"small_tp_{region}"] = counts["tp"]
            metrics[f"small_fp_{region}"] = counts["fp"]
            metrics[f"small_fn_{region}"] = counts["fn"]
            metrics[f"small_f1_proxy_{region}"] = (
                2.0 * counts["tp"] / denominator
                if denominator
                else math.nan
            )

        metrics["region_dice_mean"] = self._mean(
            [
                metrics[f"dice_{region}"]
                for region in REGION_LABELS
                if math.isfinite(metrics[f"dice_{region}"])
            ]
        )
        metrics["lesion_dice_proxy_mean"] = self._mean(
            [
                metrics[f"lesion_dice_proxy_{region}"]
                for region in REGION_LABELS
                if math.isfinite(metrics[f"lesion_dice_proxy_{region}"])
            ]
        )
        metrics["small_f1_proxy_mean"] = self._mean(
            [
                metrics[f"small_f1_proxy_{region}"]
                for region in REGION_LABELS
                if math.isfinite(metrics[f"small_f1_proxy_{region}"])
            ]
        )

        weighted = []
        for name, weight in self.selection_weights.items():
            value = metrics.get(name, math.nan)
            if weight > 0 and math.isfinite(value):
                weighted.append((float(weight), float(value)))
        if not weighted:
            raise RuntimeError("No finite validation metric is available for checkpoint selection")
        total_weight = sum(weight for weight, _ in weighted)
        metrics["checkpoint_score"] = sum(
            weight * value for weight, value in weighted
        ) / total_weight
        return metrics
