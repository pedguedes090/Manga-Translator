import logging
from unittest.mock import Mock

import numpy as np

from vision.app_adapter import VisionPageAdapter, build_optional_vision_adapter


class CountingPipeline:
    def __init__(self):
        self.prepare_calls = 0
        self.erase_calls = 0

    def prepare_page(self, image, blocks):
        self.prepare_calls += 1
        return ["prepared"]

    def erase_page(self, image, prepared):
        self.erase_calls += 1
        return image.copy(), ["erased"]


def test_optional_adapter_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MANGA_VISION_PIPELINE", raising=False)

    assert build_optional_vision_adapter() is None


def test_process_page_prepares_and_erases_once():
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    pipeline = CountingPipeline()
    adapter = VisionPageAdapter(pipeline=pipeline)

    execution = adapter.process_page(
        image, [{"text": "TEXT", "bbox": [4, 4, 20, 20]}]
    )

    assert pipeline.prepare_calls == 1
    assert pipeline.erase_calls == 1
    assert execution.prepared == ["prepared"]
    assert execution.erase_results == ["erased"]


def test_factory_returns_none_and_logs_when_runtime_construction_fails(
    monkeypatch, caplog
):
    monkeypatch.setenv("MANGA_VISION_PIPELINE", "on")
    factory = Mock(side_effect=RuntimeError("runtime missing"))

    with caplog.at_level(logging.WARNING):
        adapter = build_optional_vision_adapter(pipeline_factory=factory)

    assert adapter is None
    assert "runtime missing" in caplog.text
