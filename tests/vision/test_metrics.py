import numpy as np

from vision.metrics import compute_mask_metrics


def test_mask_metrics_count_true_false_pixels_exactly():
    target = np.array([[255, 255], [0, 0]], np.uint8)
    prediction = np.array([[255, 0], [255, 0]], np.uint8)

    metrics = compute_mask_metrics(prediction, target)

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["dice"] == 0.5
    assert metrics["iou"] == 1 / 3
    assert metrics["false_positive_pixels"] == 1.0
