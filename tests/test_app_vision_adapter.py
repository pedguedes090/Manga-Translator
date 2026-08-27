from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import app as app_module


class CountingAdapter:
    def __init__(self):
        self.calls = []

    def process_page(self, image, blocks):
        self.calls.append([dict(block) for block in blocks])
        prepared = [SimpleNamespace(block_id=f"block-{i}") for i, _ in enumerate(blocks)]
        return SimpleNamespace(
            prepared=prepared,
            erased_image=image.copy(),
            erase_results=[],
        )


def _sample_results(bbox=None):
    image = np.full((40, 60, 3), 245, dtype=np.uint8)
    return [("page.png", image, [{"text": "TEXT", "bbox": bbox or [8, 8, 36, 28]}])]


def _patch_render_dependencies(monkeypatch):
    monkeypatch.setattr(app_module, "emit_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module, "get_font_path", lambda selected: None)
    monkeypatch.setattr(app_module, "render_all_blocks", lambda image, blocks, font: image)
    monkeypatch.setattr(app_module, "should_skip_ocr_artifact", lambda *args, **kwargs: False)


def test_adapter_none_keeps_legacy_erase(monkeypatch):
    _patch_render_dependencies(monkeypatch)
    legacy = Mock(
        side_effect=lambda image, bbox, source_lang="ja": (
            image,
            (0, 0, 0),
            {"should_skip": False},
        )
    )
    monkeypatch.setattr(app_module, "erase_text_region", legacy)

    result = app_module.translate_and_render(
        _sample_results(),
        SimpleNamespace(last_warning=None),
        "arial",
        "unknown",
        "ja",
        "vi",
        "default",
        vision_adapter=None,
    )

    assert legacy.call_count == 1
    assert set(result[0]) == {"name", "image"}


def test_adapter_enabled_processes_page_once_and_preserves_result_keys(monkeypatch):
    _patch_render_dependencies(monkeypatch)
    adapter = CountingAdapter()
    appearance = {
        "text_color": (0, 0, 0),
        "should_skip": False,
        "erase_method": "stroke-fill-sampled",
    }
    monkeypatch.setattr(app_module, "appearance_for_prepared", lambda prepared: dict(appearance))
    legacy = Mock(side_effect=AssertionError("legacy erase must not run"))
    monkeypatch.setattr(app_module, "erase_text_region", legacy)

    result = app_module.translate_and_render(
        _sample_results(),
        SimpleNamespace(last_warning=None),
        "arial",
        "unknown",
        "ja",
        "vi",
        "default",
        vision_adapter=adapter,
    )

    assert len(adapter.calls) == 1
    assert len(adapter.calls[0]) == 1
    assert legacy.call_count == 0
    assert set(result[0]) == {"name", "image"}


def test_manual_correction_rebuilds_prepared_page_from_new_bbox(monkeypatch):
    _patch_render_dependencies(monkeypatch)
    adapter = CountingAdapter()
    monkeypatch.setattr(
        app_module,
        "appearance_for_prepared",
        lambda prepared: {"text_color": (0, 0, 0), "should_skip": False},
    )

    for bbox in ([8, 8, 36, 28], [12, 10, 42, 30]):
        app_module.translate_and_render(
            _sample_results(bbox),
            SimpleNamespace(last_warning=None),
            "arial",
            "unknown",
            "ja",
            "vi",
            "default",
            vision_adapter=adapter,
        )

    assert [call[0]["bbox"] for call in adapter.calls] == [
        [8, 8, 36, 28],
        [12, 10, 42, 30],
    ]


def test_uncertain_prepared_block_is_not_re_erased_by_legacy(monkeypatch):
    _patch_render_dependencies(monkeypatch)
    adapter = CountingAdapter()
    monkeypatch.setattr(
        app_module,
        "appearance_for_prepared",
        lambda prepared: {"text_color": (0, 0, 0), "should_skip": True},
    )
    legacy = Mock(side_effect=AssertionError("unsafe prepared block must stay preserved"))
    monkeypatch.setattr(app_module, "erase_text_region", legacy)

    result = app_module.translate_and_render(
        _sample_results(),
        SimpleNamespace(last_warning=None),
        "arial",
        "unknown",
        "ja",
        "vi",
        "default",
        vision_adapter=adapter,
    )

    assert legacy.call_count == 0
    assert set(result[0]) == {"name", "image"}


def test_adapter_metadata_failure_falls_back_to_legacy(monkeypatch, capsys):
    _patch_render_dependencies(monkeypatch)
    adapter = CountingAdapter()
    monkeypatch.setattr(
        app_module,
        "appearance_for_prepared",
        Mock(
            side_effect=[
                {"text_color": (0, 0, 0), "should_skip": True},
                RuntimeError("metadata failed"),
            ]
        ),
    )
    legacy = Mock(
        side_effect=lambda image, bbox, source_lang="ja": (
            image,
            (0, 0, 0),
            {"should_skip": False},
        )
    )
    monkeypatch.setattr(app_module, "erase_text_region", legacy)

    image = np.full((40, 60, 3), 245, dtype=np.uint8)
    results = [(
        "page.png",
        image,
        [
            {"text": "FIRST", "bbox": [4, 4, 24, 18]},
            {"text": "SECOND", "bbox": [28, 20, 56, 36]},
        ],
    )]
    app_module.translate_and_render(
        results,
        SimpleNamespace(last_warning=None),
        "arial",
        "unknown",
        "ja",
        "vi",
        "default",
        vision_adapter=adapter,
    )

    assert legacy.call_count == 2
    assert "Skipped 1" not in capsys.readouterr().out
