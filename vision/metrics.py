"""Metrics for binary text masks."""

from __future__ import annotations

import cv2
import numpy as np


def compute_mask_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    bubble_boundary: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute exact pixel metrics for equally sized binary masks."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")

    predicted = prediction > 0
    expected = target > 0
    true_positive = int(np.count_nonzero(predicted & expected))
    false_positive = int(np.count_nonzero(predicted & ~expected))
    false_negative = int(np.count_nonzero(~predicted & expected))

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    union = true_positive + false_positive + false_negative
    dice_denominator = 2 * true_positive + false_positive + false_negative

    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else float(recall_denominator == 0)
    )
    recall = true_positive / recall_denominator if recall_denominator else 1.0
    dice = 2 * true_positive / dice_denominator if dice_denominator else 1.0
    iou = true_positive / union if union else 1.0

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
        "true_positive_pixels": float(true_positive),
        "false_positive_pixels": float(false_positive),
        "false_negative_pixels": float(false_negative),
    }

    if bubble_boundary is not None:
        if bubble_boundary.shape != target.shape:
            raise ValueError("bubble_boundary and target must have the same shape")
        boundary = bubble_boundary > 0
        boundary_pixels = int(np.count_nonzero(boundary))
        overlap = int(np.count_nonzero(predicted & boundary))
        metrics["bubble_border_overlap"] = (
            float(overlap / boundary_pixels) if boundary_pixels else 0.0
        )

    return metrics



def _validate_spatial_shape(
    image_or_mask: np.ndarray,
    reference: np.ndarray,
    name: str,
) -> None:
    if image_or_mask.shape[:2] != reference.shape[:2]:
        raise ValueError(f"{name} must have the same spatial shape")


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask) > 0
    if binary.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary.astype(np.uint8), kernel, iterations=1) > 0
    return binary & ~eroded


def _dilate_binary(mask: np.ndarray, tolerance_px: int) -> np.ndarray:
    if tolerance_px == 0:
        return mask
    size = 2 * tolerance_px + 1
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def compute_boundary_f1(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance_px: int = 2,
) -> float:
    """Compare binary mask boundaries with a pixel tolerance."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if not isinstance(tolerance_px, (int, np.integer)) or tolerance_px < 0:
        raise ValueError("tolerance_px must be a non-negative integer")

    predicted_boundary = _mask_boundary(prediction)
    target_boundary = _mask_boundary(target)
    predicted_count = int(np.count_nonzero(predicted_boundary))
    target_count = int(np.count_nonzero(target_boundary))
    if predicted_count == 0 and target_count == 0:
        return 1.0
    if predicted_count == 0 or target_count == 0:
        return 0.0

    predicted_hits = np.count_nonzero(
        predicted_boundary & _dilate_binary(target_boundary, int(tolerance_px))
    )
    target_hits = np.count_nonzero(
        target_boundary & _dilate_binary(predicted_boundary, int(tolerance_px))
    )
    precision = float(predicted_hits / predicted_count)
    recall = float(target_hits / target_count)
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _changed_pixel_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    if before.shape != after.shape:
        raise ValueError("before and after must have the same shape")
    if before.ndim == 2:
        return before != after
    if before.ndim == 3:
        return np.any(before != after, axis=2)
    raise ValueError("images must be two- or three-dimensional arrays")


def compute_bubble_border_damage(
    before: np.ndarray,
    after: np.ndarray,
    bubble_boundary: np.ndarray,
) -> float:
    """Return the fraction of bubble-boundary pixels changed by erasure."""
    changed = _changed_pixel_mask(before, after)
    _validate_spatial_shape(bubble_boundary, before, "bubble_boundary")
    boundary = np.asarray(bubble_boundary) > 0
    boundary_pixels = int(np.count_nonzero(boundary))
    if not boundary_pixels:
        return 0.0
    return float(np.count_nonzero(changed & boundary) / boundary_pixels)


def compute_outside_mask_delta(
    before: np.ndarray,
    after: np.ndarray,
    mask: np.ndarray,
) -> int:
    """Count changed scalar values outside the supplied binary mask."""
    if before.shape != after.shape:
        raise ValueError("before and after must have the same shape")
    _validate_spatial_shape(mask, before, "mask")
    outside = ~(np.asarray(mask) > 0)
    if before.ndim == 2:
        return int(np.count_nonzero((before != after) & outside))
    if before.ndim != 3:
        raise ValueError("images must be two- or three-dimensional arrays")
    return int(np.count_nonzero((before != after) & outside[..., None]))


def _masked_mean_absolute_error(
    actual: np.ndarray,
    expected: np.ndarray,
    mask: np.ndarray,
) -> float:
    selected = np.asarray(mask) > 0
    if not np.any(selected):
        return 0.0
    delta = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    if delta.ndim == 3:
        delta = np.mean(delta, axis=2)
    return float(np.mean(delta[selected]))


def compute_inpainting_metrics(
    before: np.ndarray,
    after: np.ndarray,
    clean_target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | None]:
    """Measure restoration error and safety for a predicted erasure mask."""
    if before.shape != after.shape or after.shape != clean_target.shape:
        raise ValueError("before, after, and clean_target must have the same shape")
    _validate_spatial_shape(mask, before, "mask")
    if before.ndim != 3 or before.shape[2] != 3:
        raise ValueError("inpainting metrics require BGR color images")

    mask_array = np.asarray(mask) > 0
    before_lab = cv2.cvtColor(before, cv2.COLOR_BGR2LAB)
    after_lab = cv2.cvtColor(after, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(clean_target, cv2.COLOR_BGR2LAB)
    source_lab_mae = _masked_mean_absolute_error(before_lab, target_lab, mask_array)
    restored_lab_mae = _masked_mean_absolute_error(after_lab, target_lab, mask_array)
    source_rgb_mae = _masked_mean_absolute_error(before, clean_target, mask_array)
    restored_rgb_mae = _masked_mean_absolute_error(after, clean_target, mask_array)

    return {
        "masked_lab_mae": restored_lab_mae,
        "masked_rgb_mae": restored_rgb_mae,
        "source_masked_lab_mae": source_lab_mae,
        "source_masked_rgb_mae": source_rgb_mae,
        "residual_ratio": (
            float(restored_lab_mae / source_lab_mae)
            if source_lab_mae
            else None
        ),
        "outside_mask_delta": float(compute_outside_mask_delta(before, after, mask_array)),
        "ssim": None,
        "lpips": None,
    }
