"""Pytest suite for Manual Mode V2 backend: render_plan persistence, post-render
editor, /re-render-image, /re-render-all, /translate-result, and legacy-session
compatibility.

Contract under test: docs/manual-mode-v2-spec.md §4.1–4.4.
Key invariants:
  * re-render never calls OCR or any translator (A5.2) — enforced by monkeypatch
    guards below;
  * re-render always starts from the ORIGINAL page_<i>.jpg (idempotent);
  * invalid bbox -> 422, bad index/json -> 400, missing session -> 404;
  * old sessions without render_plan keep working (spec §6.7);
  * /continue-translate and /correction stay compatible.
"""
import base64
import io
import json
import os
import uuid

import cv2
import numpy as np
import pytest

import app as app_module
from app import app as flask_app


def _make_image(width=60, height=40, color=255):
    return np.full((height, width, 3), color, dtype=np.uint8)


def _make_session(monkeypatch, tmp_path, n_images=2, with_render_plan=True):
    """Create a fake session dir and return (session_id, ocr_images).

    monkeypatch: pytest fixture used to point app_module.TEMP_DIR at tmp_path.
    """
    monkeypatch.setattr(app_module, "TEMP_DIR", str(tmp_path))
    app_module.ocr_sessions.clear()

    session_id = str(uuid.uuid4())
    images = []
    all_ocr_results = []
    for i in range(n_images):
        image = _make_image(color=245 - i * 10)
        images.append(image)
        all_ocr_results.append((
            "page%d" % i,
            image,
            [{"text": "text-%d" % i, "bbox": [2, 2, 30, 12]}],
        ))
    session_data = {
        "all_ocr_results": all_ocr_results,
        "all_texts": ["text-%d" % i for i in range(n_images)],
        "selected_translator": "google",
        "selected_font": "animeace_",
        "source_lang": "ja",
        "target_lang": "vi",
        "style": "",
        "gemini_api_keys": [],
        "gemini_api_key": "",
        "gemini_model": "gemini-3.1-flash-lite",
        "copilot_server": "http://localhost:8080",
        "copilot_model": "gpt-4o",
        "translator_type": "google",
    }
    if with_render_plan:
        session_data["render_plan"] = [
            {
                "name": "page%d" % i,
                "erase_regions": [[2, 2, 30, 12]],
                "blocks": [
                    {"text": "text-%d" % i,
                     "translated": "dịch-%d" % i,
                     "bbox": [2, 2, 30, 12]},
                ],
            }
            for i in range(n_images)
        ]
    app_module._save_session(session_id, session_data)
    return session_id, images


@pytest.fixture()
def no_vision(monkeypatch):
    monkeypatch.setattr(app_module, "build_optional_vision_adapter", lambda: None)


@pytest.fixture()
def guard_translator(monkeypatch):
    """Fail the test if anything instantiates/uses a translator or OCR."""
    monkeypatch.setattr(
        app_module, "MangaTranslator",
        lambda **kwargs: pytest.fail("MangaTranslator called during re-render"),
    )
    monkeypatch.setattr(
        app_module, "ChromeLensOCR",
        lambda **kwargs: pytest.fail("ChromeLensOCR called during re-render"),
    )


# ── Unit: normalize_bbox_for_json / plan helpers ─────────────────────────────

def test_normalize_bbox_swaps_and_clamps():
    assert app_module.normalize_bbox_for_json([40, 30, 5, 2], (40, 60)) == [5, 2, 40, 30]
    assert app_module.normalize_bbox_for_json([-5, -5, 999, 999], (40, 60)) == [0, 0, 60, 40]
    assert app_module.normalize_bbox_for_json([10, 10, 10, 20], (40, 60)) is None
    assert app_module.normalize_bbox_for_json("garbage", (40, 60)) is None


def test_normalize_render_plan_entry_filters_bad_blocks():
    entry = {
        "name": "p1",
        "erase_regions": [[0, 0, 5, 5], "junk"],
        "blocks": [
            {"text": "a", "translated": "b", "bbox": [1, 1, 10, 10]},
            {"text": "bad", "translated": "x", "bbox": [9, 9, 2, 2]},
            {"text": "nonum", "translated": "x", "bbox": "nope"},
            {"text": "skip", "translated": "x", "bbox": [50, 50, 80, 80]},
        ],
    }
    cleaned = app_module._normalize_render_plan_entry(entry, image_shape=(40, 60))
    assert cleaned is not None
    # Block 2 has inverted coords but normalize_bbox_for_json swaps them, so it
    # survives; only truly invalid bboxes are dropped.
    assert len(cleaned["blocks"]) == 2
    assert cleaned["blocks"][0]["bbox"] == [1, 1, 10, 10]
    assert cleaned["blocks"][1]["bbox"] == [2, 2, 9, 9]
    assert cleaned["erase_regions"] == [[0, 0, 5, 5]]


# ── render_image_with_blocks ─────────────────────────────────────────────────

def test_render_image_with_blocks_never_touches_original(no_vision):
    image = _make_image()
    original = image.copy()
    blocks = [{"text": "a", "translated": "Xin chào", "bbox": [5, 5, 50, 20]}]
    rendered, entry = app_module.render_image_with_blocks(
        "p", image, blocks, "fonts/animeace_i.ttf", "ja",
        vision_adapter=None,
    )
    assert np.array_equal(image, original)
    assert rendered.shape == image.shape
    assert not np.array_equal(rendered, original)
    assert entry["blocks"][0]["translated"] == "Xin chào"


def test_render_image_with_blocks_extra_erase_regions(no_vision):
    image = _make_image()
    blocks = [{"text": "a", "translated": "Mới", "bbox": [40, 5, 55, 20]}]
    extra = [[5, 5, 30, 20]]  # old position of the original text
    rendered, entry = app_module.render_image_with_blocks(
        "p", image, blocks, "fonts/animeace_i.ttf", "ja",
        vision_adapter=None, extra_erase_regions=extra,
    )
    assert rendered.shape == image.shape
    assert entry["blocks"][0]["bbox"] == [40, 5, 55, 20]


# ── Endpoint: /re-render-image ───────────────────────────────────────────────

def test_rerender_image_success_persists_plan_and_jpeg(
    monkeypatch, tmp_path, no_vision, guard_translator
):
    session_id, images = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    payload = [
        {"text": "text-0", "translated": "Bản dịch mới", "bbox": [4, 4, 32, 14]},
        {"text": "text-1", "translated": "", "bbox": [2, 2, 30, 12]},
    ]
    response = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": json.dumps(payload),
        "deleted_regions_json": "[[10, 10, 20, 15]]",
    })
    assert response.status_code == 200
    body = response.get_json()
    assert body["name"] == "page0"
    assert body["data"]  # base64 JPEG
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(body["data"]), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None and decoded.shape == images[0].shape
    assert body["blocks"][0]["translated"] == "Bản dịch mới"

    # session.json render_plan[0] updated; page_0_rendered.jpg persisted
    session_data = app_module.load_session(session_id)
    assert session_data["render_plan"][0]["blocks"][0]["translated"] == "Bản dịch mới"
    rendered_path = os.path.join(
        str(tmp_path), session_id, "page_0_rendered.jpg"
    )
    assert os.path.exists(rendered_path)


def test_rerender_image_invalid_bbox_422(monkeypatch, tmp_path, no_vision, guard_translator):
    session_id, _ = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    response = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        # Degenerate bbox (zero width) → normalize returns None → 422.
        "blocks_json": json.dumps([
            {"text": "a", "translated": "x", "bbox": [10, 10, 10, 20]},
        ]),
        "deleted_regions_json": "[]",
    })
    assert response.status_code == 422
    assert response.get_json()["error"] == "invalid_bbox"


def test_rerender_image_index_out_of_range_400(monkeypatch, tmp_path, no_vision, guard_translator):
    session_id, _ = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    response = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "7",
        "blocks_json": "[]",
        "deleted_regions_json": "[]",
    })
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_image_idx"


def test_rerender_image_bad_json_400(monkeypatch, tmp_path, no_vision, guard_translator):
    session_id, _ = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    response = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": "not-json",
        "deleted_regions_json": "[]",
    })
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_blocks_json"


def test_rerender_image_missing_session_404(monkeypatch, tmp_path, no_vision, guard_translator):
    client = flask_app.test_client()
    response = client.post("/re-render-image", data={
        "session_id": str(uuid.uuid4()),
        "image_idx": "0",
        "blocks_json": "[]",
        "deleted_regions_json": "[]",
    })
    assert response.status_code == 404
    assert response.get_json()["error"] == "session_not_found"


def test_rerender_image_path_traversal_rejected(monkeypatch, tmp_path, no_vision, guard_translator):
    client = flask_app.test_client()
    response = client.post("/re-render-image", data={
        "session_id": "../../etc",
        "image_idx": "0",
        "blocks_json": "[]",
        "deleted_regions_json": "[]",
    })
    assert response.status_code == 404


def test_rerender_deleted_block_keeps_erase_regions_in_plan(
    monkeypatch, tmp_path, no_vision, guard_translator
):
    """Deleting a block must not shrink erase_regions (spec F5/R1): the original
    text position stays erased on every future re-render."""
    session_id, images = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    # Block 1 is removed from the submitted blocks and its bbox sent as deleted.
    response = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": json.dumps([]),
        "deleted_regions_json": json.dumps([[2, 2, 30, 12]]),
    })
    assert response.status_code == 200
    session_data = app_module.load_session(session_id)
    entry0 = session_data["render_plan"][0]
    # The original erase region is preserved (and the deleted region matches it).
    assert [2, 2, 30, 12] in entry0["erase_regions"]

    # A second re-render (with text again) must still erase the original spot.
    response2 = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": json.dumps([
            {"text": "text-0", "translated": "Trở lại", "bbox": [40, 4, 55, 18]},
        ]),
        "deleted_regions_json": "[]",
    })
    assert response2.status_code == 200
    entry0_after = app_module.load_session(session_id)["render_plan"][0]
    assert [2, 2, 30, 12] in entry0_after["erase_regions"]
    assert [40, 4, 55, 18] == entry0_after["blocks"][0]["bbox"]


def test_rerender_image_idempotent_double_submit(
    monkeypatch, tmp_path, no_vision, guard_translator
):
    session_id, images = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    data = {
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": json.dumps([
            {"text": "text-0", "translated": "Lần một", "bbox": [2, 2, 30, 12]},
        ]),
        "deleted_regions_json": "[]",
    }
    first = client.post("/re-render-image", data=data)
    second = client.post("/re-render-image", data=data)
    assert first.status_code == 200 and second.status_code == 200
    # Same input twice must produce identical output (starts from original).
    assert first.get_json()["data"] == second.get_json()["data"]


# ── Endpoint: /postrender/<sid> ──────────────────────────────────────────────

def test_postrender_page_returns_mode_and_translated_blocks(
    monkeypatch, tmp_path, no_vision
):
    session_id, images = _make_session(monkeypatch, tmp_path)
    app_module._save_rendered_jpeg(session_id, 0, images[0])
    client = flask_app.test_client()
    response = client.get("/postrender/%s?img=1" % session_id)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'mode: "postrender"' in html
    assert "Chỉnh sửa sau dịch" in html
    assert "btn-rerender-one" in html
    assert "btn-save-all" in html


def test_postrender_page_missing_session_redirects(monkeypatch, tmp_path, no_vision):
    client = flask_app.test_client()
    response = client.get("/postrender/%s" % uuid.uuid4())
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")


def test_postrender_page_legacy_session_falls_back_to_correction(
    monkeypatch, tmp_path, no_vision
):
    session_id, _ = _make_session(monkeypatch, tmp_path, with_render_plan=False)
    client = flask_app.test_client()
    response = client.get("/postrender/%s" % session_id)
    # Legacy fallback: redirect to the OCR correction page (spec risk R3).
    assert response.status_code == 302
    assert "/correction/" in response.headers["Location"]


# ── Endpoint: /re-render-all ─────────────────────────────────────────────────

def test_rerender_all_returns_translate_page(monkeypatch, tmp_path, no_vision, guard_translator):
    session_id, images = _make_session(monkeypatch, tmp_path)
    app_module._save_rendered_jpeg(session_id, 0, images[0])
    app_module._save_rendered_jpeg(session_id, 1, images[1])
    client = flask_app.test_client()
    response = client.post("/re-render-all", data={
        "session_id": session_id,
        "dirty_indices_json": "[0]",
    })
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Kết quả dịch" in html
    assert "Chỉnh sửa" in html  # edit button shown when correction_session_id set
    session_data = app_module.load_session(session_id)
    assert len(session_data["render_plan"]) == 2


def test_rerender_all_missing_session_redirects(monkeypatch, tmp_path, no_vision):
    client = flask_app.test_client()
    response = client.post("/re-render-all", data={
        "session_id": str(uuid.uuid4()),
        "dirty_indices_json": "[0]",
    })
    assert response.status_code == 302


# ── Endpoint: /translate-result/<sid> ────────────────────────────────────────

def test_translate_result_uses_persisted_rendered_images(monkeypatch, tmp_path, no_vision):
    session_id, images = _make_session(monkeypatch, tmp_path)
    app_module._save_rendered_jpeg(session_id, 0, images[0])
    app_module._save_rendered_jpeg(session_id, 1, images[1])
    client = flask_app.test_client()
    response = client.get("/translate-result/%s" % session_id)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Kết quả dịch" in html
    assert "/postrender/" + session_id in html


def test_translate_result_missing_session_redirects(monkeypatch, tmp_path, no_vision):
    client = flask_app.test_client()
    response = client.get("/translate-result/%s" % uuid.uuid4())
    assert response.status_code == 302


# ── Compatibility: existing flows ────────────────────────────────────────────

def test_correction_page_legacy_session_still_works(monkeypatch, tmp_path, no_vision):
    session_id, _ = _make_session(monkeypatch, tmp_path, with_render_plan=False)
    client = flask_app.test_client()
    response = client.get("/correction/%s" % session_id)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Tiếp tục dịch & Render" in html


def test_rerender_after_session_evicted_from_memory(monkeypatch, tmp_path, no_vision, guard_translator):
    """Session survives memory-cache eviction (loads from disk)."""
    session_id, _ = _make_session(monkeypatch, tmp_path)
    app_module.ocr_sessions.clear()  # simulate restart/cache eviction
    client = flask_app.test_client()
    response = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": json.dumps([
            {"text": "text-0", "translated": "Sau evict", "bbox": [2, 2, 30, 12]},
        ]),
        "deleted_regions_json": "[]",
    })
    assert response.status_code == 200
    assert response.get_json()["blocks"][0]["translated"] == "Sau evict"
