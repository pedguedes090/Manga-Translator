import json

import pytest

from vision.config import VisionConfig


def test_default_config_has_cuda_full_resolution_profile():
    config = VisionConfig.load("configs/vision.json")

    assert config.profile == "cuda"
    assert config.inpaint.lama_full_resolution is True
    assert config.inpaint.precision == "fp32"
    assert config.text_mask.prob_low < config.text_mask.prob_high
    assert len(config.config_hash()) == 64


def test_config_rejects_reversed_probability_thresholds(tmp_path):
    path = tmp_path / "vision.json"
    path.write_text(
        json.dumps(
            {
                "profile": "cuda",
                "mask_backend": "auto",
                "allow_cpu_fallback": True,
                "text_mask": {"prob_low": 0.8, "prob_high": 0.2},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prob_low must be less than prob_high"):
        VisionConfig.load(path)


def test_config_rejects_string_rollout_gates(tmp_path):
    path = tmp_path / "vision.json"
    path.write_text(
        json.dumps({"hybrid_gate_passed": "false"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="hybrid_gate_passed must be boolean"):
        VisionConfig.load(path)


def test_config_rejects_unknown_lama_precision(tmp_path):
    path = tmp_path / "vision.json"
    path.write_text(
        json.dumps({"inpaint": {"precision": "int8"}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="inpaint precision must be fp16 or fp32"):
        VisionConfig.load(path)
