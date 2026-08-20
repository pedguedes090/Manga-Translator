"""Metrics for binary text masks."""

from __future__ import annotations

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
