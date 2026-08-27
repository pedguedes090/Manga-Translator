import numpy as np

from vision.metrics import (
    compute_boundary_f1,
    compute_bubble_border_damage,
    compute_inpainting_metrics,
    compute_outside_mask_delta,
)


def test_outside_mask_delta_counts_changed_channel_values():
    before = np.zeros((3, 3, 3), dtype=np.uint8)
    after = before.copy()
    after[0, 0] = (1, 2, 3)
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 255

    assert compute_outside_mask_delta(before, after, mask) == 3


def test_boundary_f1_is_one_for_identical_masks():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255

    assert compute_boundary_f1(mask, mask, tolerance_px=2) == 1.0


def test_bubble_border_damage_is_changed_boundary_fraction():
    before = np.zeros((2, 2, 3), dtype=np.uint8)
    after = before.copy()
    after[0, 0] = 255
    boundary = np.array([[255, 255], [0, 0]], dtype=np.uint8)

    assert compute_bubble_border_damage(before, after, boundary) == 0.5


def test_inpainting_metrics_are_zero_for_clean_reconstruction():
    clean = np.full((4, 4, 3), 128, dtype=np.uint8)
    before = clean.copy()
    before[1:3, 1:3] = 0
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255

    metrics = compute_inpainting_metrics(before, clean, clean, mask)

    assert metrics["masked_lab_mae"] == 0.0
    assert metrics["outside_mask_delta"] == 0.0



def test_inpainting_ratio_is_undefined_when_clean_source_is_damaged():
    clean = np.full((4, 4, 3), 128, dtype=np.uint8)
    damaged = clean.copy()
    damaged[1, 1] = 0
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 255

    metrics = compute_inpainting_metrics(clean, damaged, clean, mask)

    assert metrics["masked_lab_mae"] > 0
    assert metrics["residual_ratio"] is None
