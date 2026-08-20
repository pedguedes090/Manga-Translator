from dataclasses import replace

from vision.config import VisionConfig
from vision.maskers.heuristic import HeuristicTextMasker
from vision.maskers.hybrid import HybridTextMasker
from vision.pipeline import build_text_masker


def test_auto_backend_uses_hybrid_after_gate_is_enabled():
    config = replace(
        VisionConfig.load("configs/vision.json"), hybrid_gate_passed=True
    )

    assert isinstance(build_text_masker(config), HybridTextMasker)


def test_auto_backend_keeps_heuristic_before_gate():
    config = replace(
        VisionConfig.load("configs/vision.json"), hybrid_gate_passed=False
    )

    assert isinstance(build_text_masker(config), HeuristicTextMasker)


def test_explicit_hybrid_backend_remains_available_before_gate():
    config = replace(
        VisionConfig.load("configs/vision.json"),
        mask_backend="hybrid",
        hybrid_gate_passed=False,
    )

    assert isinstance(build_text_masker(config), HybridTextMasker)
