import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import tools.benchmark_vision_pipeline as benchmark_module
from tools.benchmark_vision_pipeline import benchmark_manifest


class FakePipeline:
    def __init__(self):
        self.prepare_calls = 0
        self.erase_calls = 0

    def prepare_page(self, image, blocks):
        self.prepare_calls += 1
        return []

    def erase_page(self, image, prepared):
        self.erase_calls += 1
        return image.copy(), []


class FailingPipeline:
    def prepare_page(self, image, blocks):
        raise RuntimeError("synthetic pipeline failure")


def _write_manifest(tmp_path):
    image = np.full((32, 32, 3), 240, dtype=np.uint8)
    clean = image.copy()
    text_mask = np.zeros((32, 32), dtype=np.uint8)
    text_mask[10:20, 10:22] = 255
    bubble_mask = np.zeros((32, 32), dtype=np.uint8)
    bubble_mask[4:28, 4:28] = 255
    image[10:20, 10:22] = 0

    cv2.imwrite(str(tmp_path / "image.png"), image)
    cv2.imwrite(str(tmp_path / "clean.png"), clean)
    cv2.imwrite(str(tmp_path / "text.png"), text_mask)
    cv2.imwrite(str(tmp_path / "bubble.png"), bubble_mask)
    (tmp_path / "vision.json").write_text(json.dumps({"mask_backend": "heuristic"}), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "case-0",
                "image": "image.png",
                "clean_target": "clean.png",
                "text_mask": "text.png",
                "bubble_mask": "bubble.png",
                "bbox": [8, 8, 24, 22],
                "text": "TEXT",
                "category": "white_bubble",
                "language": "ja",
            }
        ) + "\n",
        encoding="utf-8",
    )
    return manifest, tmp_path / "vision.json"


def test_benchmark_report_has_stage_timings_and_hashes(tmp_path):
    manifest, config = _write_manifest(tmp_path)
    pipeline = FakePipeline()

    report = benchmark_manifest(
        manifest,
        config,
        mode="prepared",
        backend="heuristic",
        indices=[0],
        pipeline=pipeline,
    )

    assert report["schema_version"] == 1
    assert report["dataset_hash"]
    assert report["config_hash"]
    assert "method_counts" in report
    row = report["rows"][0]
    assert {"decode_ms", "prepare_ms", "erase_ms", "render_ms", "total_ms"} <= row.keys()
    assert pipeline.prepare_calls == 1
    assert pipeline.erase_calls == 1


def test_benchmark_keeps_page_errors_as_json_rows(tmp_path):
    manifest, config = _write_manifest(tmp_path)

    report = benchmark_manifest(
        manifest,
        config,
        mode="prepared",
        backend="heuristic",
        indices=[0],
        pipeline=FailingPipeline(),
    )

    assert report["failed_pages"] == 1
    assert report["rows"][0]["status"] == "error"
    assert report["rows"][0]["error"] == "RuntimeError"


def test_legacy_benchmark_reports_clean_target_quality_metrics(tmp_path):
    manifest, config = _write_manifest(tmp_path)

    report = benchmark_manifest(
        manifest,
        config,
        mode="legacy",
        backend="heuristic",
        indices=[0],
    )

    row = report["rows"][0]
    assert row["quality_status"] == "measured"
    assert row["inpainting_metrics"]["outside_reference_mask_delta"] == 0.0
    assert row["inpainting_metrics"]["outside_predicted_mask_delta"] is None
    assert "mask_metrics" in row



def test_prepared_method_counts_report_actual_fallback(monkeypatch, tmp_path):
    manifest, config = _write_manifest(tmp_path)

    class FallbackPipeline:
        def prepare_page(self, image, blocks):
            return [SimpleNamespace(erase_method="lama_full_page")]

        def erase_page(self, image, prepared):
            return image.copy(), [
                SimpleNamespace(method="telea", warning="used Telea fallback")
            ]

    monkeypatch.setattr(
        benchmark_module,
        "_prepared_appearance",
        lambda prepared: {"text_color": (0, 0, 0), "should_skip": False},
    )
    monkeypatch.setattr(
        benchmark_module,
        "_quality_metrics",
        lambda **kwargs: {"quality_status": "not_available"},
    )

    report = benchmark_manifest(
        manifest,
        config,
        mode="prepared",
        backend="heuristic",
        indices=[0],
        pipeline=FallbackPipeline(),
    )

    assert report["rows"][0]["method_counts"] == {"telea": 1}



def test_benchmark_rejects_empty_or_duplicate_indices(tmp_path):
    manifest, config = _write_manifest(tmp_path)

    with pytest.raises(ValueError, match="indices must not be empty"):
        benchmark_manifest(
            manifest,
            config,
            mode="prepared",
            backend="heuristic",
            indices=[],
            pipeline=FakePipeline(),
        )
    with pytest.raises(ValueError, match="indices must be unique"):
        benchmark_manifest(
            manifest,
            config,
            mode="prepared",
            backend="heuristic",
            indices=[0, 0],
            pipeline=FakePipeline(),
        )
