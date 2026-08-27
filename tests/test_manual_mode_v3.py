"""Pytest suite for Manual Mode V3 backend: per-block style, font list API,
style-editor prepare/page, erase-region accumulation (monotonic), erase mask,
and V2 backward compatibility.

Contract under test: docs/manual-mode-v3-spec.md sections 4.1-4.7.
Key invariants:
  * re-render never calls OCR or any translator (guarded by monkeypatch);
  * erase_regions never shrink (spec A4.5/A7.4);
  * style is persisted in render_plan and survives re-renders (A5.10/A7.8);
  * /api/fonts + /font-file are whitelist-only (no path traversal, A8.2);
  * V2 clients calling /re-render-image without style still work (A7.8).
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


def _make_image(width=60, height=40, color=245):
    return np.full((height, width, 3), color, dtype=np.uint8)


def _make_session(monkeypatch, tmp_path, n_images=2, with_render_plan=True,
                  with_text_bars=True):
    """Create a fake session dir and return (session_id, ocr_images).

    Each image carries two thin dark bars inside the bbox [2, 2, 30, 12] so
    erasure is observable in pixels (legacy erase path is deterministic).
    """
    monkeypatch.setattr(app_module, "TEMP_DIR", str(tmp_path))
    app_module.ocr_sessions.clear()

    session_id = str(uuid.uuid4())
    images = []
    all_ocr_results = []
    for i in range(n_images):
        image = _make_image(color=245 - i * 10)
        if with_text_bars:
            image[3:5, 3:29] = 30
            image[8:10, 3:29] = 30
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
                    {
                        "text": "text-%d" % i,
                        "translated": "dịch-%d" % i,
                        "bbox": [2, 2, 30, 12],
                    },
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


class _FakeTranslator:
    """Stand-in for MangaTranslator that returns deterministic translations."""

    def __init__(self, fail=False, **kwargs):
        self.fail = fail
        self.last_warning = None
        self.calls = []

    def translate_batch_google(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("translation exploded")
        return ["bản dịch %d" % i for i in range(len(texts))]


def _fake_translator_factory(fake):
    return lambda **kwargs: fake


# ── Unit: fonts / style normalization ───────────────────────────────────────

def test_list_available_fonts_has_base_and_yuki():
    fonts = app_module.list_available_fonts()
    names = [f["name"] for f in fonts]
    assert len(names) >= 23
    assert names[:3] == ["animeace_", "arial", "mangat"]
    assert "Yuki-Burobu" in names
    assert len(names) == len(set(names))
    # no non-TTF files leak in
    assert all(f["name"] == f["name"].strip() for f in fonts)


def test_normalize_block_style_valid_and_invalid():
    good = app_module.normalize_block_style(
        {"font": "Yuki-Burobu", "font_size": 40, "text_color": "#e53935",
         "bold": True, "italic": True, "align": "left"},
        "animeace_",
    )
    assert good["font"] == "Yuki-Burobu"
    assert good["font_size"] == 40
    assert good["text_color"] == "#E53935"
    assert good["bold"] is True and good["italic"] is True
    assert good["align"] == "left"

    bad = app_module.normalize_block_style(
        {"font": "../../etc/passwd", "font_size": 9999,
         "text_color": "not-a-color", "bold": "yes", "align": "top"},
        "animeace_",
    )
    assert bad["font"] == "animeace_"
    assert bad["font_size"] == 120
    assert bad["text_color"] is None
    assert bad["bold"] is True and bad["italic"] is False
    assert bad["align"] == "center"

    tiny = app_module.normalize_block_style({'font_size': -5}, 'animeace_')
    assert tiny["font_size"] == 0
    assert app_module.normalize_block_style(None, 'animeace_') is None
    assert app_module.normalize_block_style('junk', 'animeace_') is None


# ── Endpoint: /api/fonts + /font-file ───────────────────────────────────────

def test_api_fonts_route():
    client = flask_app.test_client()
    response = client.get("/api/fonts")
    assert response.status_code == 200
    fonts = response.get_json()["fonts"]
    names = [f["name"] for f in fonts]
    assert names[:3] == ["animeace_", "arial", "mangat"]
    assert "Yuki-Burobu" in names


def test_font_file_serves_ttf():
    client = flask_app.test_client()
    response = client.get("/font-file/animeace_")
    assert response.status_code == 200
    assert response.mimetype == "font/ttf"
    with open(os.path.join("fonts", "animeace_i.ttf"), "rb") as f:
        assert response.data == f.read()


def test_font_file_traversal_rejected():
    client = flask_app.test_client()
    for name in ["..%2Fapp.py", "..%2F..%2Fapp.py", "app.py", "Yuki-%2F..", "../app.py"]:
        response = client.get("/font-file/" + name)
        assert response.status_code == 404, name
    assert client.get("/font-file/Yuki-Burobu").status_code == 200


# ── Unit: shared helpers ────────────────────────────────────────────────────

def test_rebuild_ocr_from_modified_blocks(monkeypatch, tmp_path, no_vision):
    session_id, _ = _make_session(monkeypatch, tmp_path)
    session_data = app_module.load_session(session_id)
    modified = [
        {"image_idx": 0, "blocks": [
            {"text": "text-0", "bbox": [2, 2, 30, 12]},
            {"text": "", "bbox": [40, 40, 50, 50]},
        ]},
    ]
    results, texts = app_module.rebuild_ocr_from_modified_blocks(session_data, modified)
    assert len(results) == 1
    assert texts == ["text-0"]
    assert results[0][2][0]["_text_idx"] == 0
    # invalid index is skipped, not fatal
    results2, texts2 = app_module.rebuild_ocr_from_modified_blocks(
        session_data, [{"image_idx": 99, "blocks": []}]
    )
    assert results2 == [] and texts2 == []


def test_translate_texts_all_google_and_failure(monkeypatch):
    fake = _FakeTranslator()
    translated, warning = app_module.translate_texts_all(
        ["a", "b"], fake, "google", source_lang="ja", target_lang="vi"
    )
    assert translated == ["bản dịch 0", "bản dịch 1"]
    assert warning is None

    failing = _FakeTranslator(fail=True)
    translated2, warning2 = app_module.translate_texts_all(
        ["a"], failing, "google", source_lang="ja", target_lang="vi"
    )
    assert translated2 == ["a"]
    assert warning2 is not None


# ── Endpoint: /styleditor-prepare ───────────────────────────────────────────

def test_styleditor_prepare_writes_draft_and_erased(
    monkeypatch, tmp_path, no_vision
):
    fake = _FakeTranslator()
    monkeypatch.setattr(app_module, "MangaTranslator", _fake_translator_factory(fake))
    monkeypatch.setattr(
        app_module, "ChromeLensOCR",
        lambda **kwargs: pytest.fail("OCR called during prepare"),
    )
    session_id, images = _make_session(monkeypatch, tmp_path, with_render_plan=False)
    client = flask_app.test_client()
    modified = [
        {"image_idx": 0, "blocks": [{"text": "text-0", "bbox": [2, 2, 30, 12]}]},
        {"image_idx": 1, "blocks": [{"text": "text-1", "bbox": [2, 2, 30, 12]}]},
    ]
    response = client.post("/styleditor-prepare", data={
        "session_id": session_id,
        "modified_blocks": json.dumps(modified),
    })
    assert response.status_code == 302
    assert response.headers["Location"] == "/styleditor/%s?img=0" % session_id

    session_data = app_module.load_session(session_id)
    draft = session_data["v3_draft"]["images"]
    assert len(draft) == 2
    assert draft[0]["blocks"][0]["translated"] == "bản dịch 0"
    assert draft[0]["blocks"][0]["style"]["font"] == "animeace_"
    assert draft[0]["blocks"][0]["style"]["font_size"] == 0
    assert draft[0]["blocks"][0]["style"]["align"] == "center"
    # no render plan created yet (A1.4: no Phase 3)
    assert "render_plan" not in session_data

    erased_path = os.path.join(str(tmp_path), session_id, "page_0_erased.jpg")
    assert os.path.exists(erased_path)
    erased = cv2.imread(erased_path)
    assert erased is not None and erased.shape == images[0].shape
    # original text bars are gone from the erased background (A1.2/A2.2)
    assert float(erased[4:9, 5:28].mean()) > 200


def test_styleditor_prepare_no_text_returns_translate_page(
    monkeypatch, tmp_path, no_vision
):
    fake = _FakeTranslator()
    monkeypatch.setattr(app_module, "MangaTranslator", _fake_translator_factory(fake))
    session_id, _ = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    modified = [
        {"image_idx": 0, "blocks": [{"text": "", "bbox": [2, 2, 30, 12]}]},
    ]
    response = client.post("/styleditor-prepare", data={
        "session_id": session_id,
        "modified_blocks": json.dumps(modified),
    })
    assert response.status_code == 200
    assert "Kết quả dịch" in response.get_data(as_text=True)
    session_data = app_module.load_session(session_id)
    assert "v3_draft" not in session_data


def test_styleditor_prepare_missing_session_redirects(monkeypatch, tmp_path):
    client = flask_app.test_client()
    response = client.post("/styleditor-prepare", data={
        "session_id": str(uuid.uuid4()),
        "modified_blocks": "[]",
    })
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")


def test_styleditor_prepare_translator_error_keeps_original_text(
    monkeypatch, tmp_path, no_vision
):
    fake = _FakeTranslator(fail=True)
    monkeypatch.setattr(app_module, "MangaTranslator", _fake_translator_factory(fake))
    session_id, _ = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    modified = [
        {"image_idx": 0, "blocks": [{"text": "text-0", "bbox": [2, 2, 30, 12]}]},
    ]
    response = client.post("/styleditor-prepare", data={
        "session_id": session_id,
        "modified_blocks": json.dumps(modified),
    })
    assert response.status_code == 302
    session_data = app_module.load_session(session_id)
    draft = session_data["v3_draft"]["images"]
    assert draft[0]["blocks"][0]["translated"] == "text-0"
    assert session_data.get("v3_last_warning") is not None


# ── Endpoint: /styleditor/<sid> ─────────────────────────────────────────────

def test_styleditor_page_renders_editor(monkeypatch, tmp_path, no_vision):
    session_id, images = _make_session(monkeypatch, tmp_path)
    session_data = app_module.load_session(session_id)
    session_data["v3_draft"] = {"images": [
        {
            "name": "page0",
            "blocks": [{"text": "text-0", "translated": "bản dịch 0",
                        "bbox": [2, 2, 30, 12],
                        "style": {"font": "animeace_", "font_size": 0,
                                  "text_color": None, "bold": False,
                                  "italic": False, "align": "center"}}],
        },
    ]}
    app_module._save_session(session_id, session_data)
    app_module._save_erased_jpeg(session_id, 0, images[0])
    client = flask_app.test_client()
    response = client.get("/styleditor/%s?img=0" % session_id)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'mode: "styleditor"' in html
    assert "style" in html
    assert "Yuki" in html or "font" in html
    assert response.status_code == 200


def test_styleditor_page_missing_session_redirects(monkeypatch, tmp_path):
    client = flask_app.test_client()
    response = client.get("/styleditor/%s" % uuid.uuid4())
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")


def test_styleditor_page_no_draft_redirects_correction(
    monkeypatch, tmp_path, no_vision
):
    session_id, _ = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    response = client.get("/styleditor/%s?img=1" % session_id)
    assert response.status_code == 302
    assert "/correction/" in response.headers["Location"]


def test_styleditor_page_bad_img_clamped_to_zero(monkeypatch, tmp_path, no_vision):
    session_id, images = _make_session(monkeypatch, tmp_path)
    session_data = app_module.load_session(session_id)
    session_data["v3_draft"] = {"images": [
        {"name": "page0", "blocks": []},
    ]}
    app_module._save_session(session_id, session_data)
    app_module._save_erased_jpeg(session_id, 0, images[0])
    client = flask_app.test_client()
    response = client.get("/styleditor/%s?img=42" % session_id)
    assert response.status_code == 200
    assert 'mode: "styleditor"' in response.get_data(as_text=True)


# ── Endpoint: /re-render-image with style ───────────────────────────────────

def test_rerender_with_style_persists_plan_and_response(
    monkeypatch, tmp_path, no_vision, guard_translator
):
    session_id, images = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    payload = [
        {
            "text": "text-0",
            "translated": "Bản dịch mới",
            "bbox": [4, 4, 32, 14],
            "style": {"font": "Yuki-Burobu", "font_size": 18,
                      "text_color": "#E53935", "bold": True,
                      "italic": False, "align": "left"},
        },
    ]
    response = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": json.dumps(payload),
        "erase_regions_json": "[[10, 10, 20, 15]]",
    })
    assert response.status_code == 200
    body = response.get_json()
    assert body["blocks"][0]["style"]["font"] == "Yuki-Burobu"
    assert body["blocks"][0]["style"]["text_color"] == "#E53935"

    session_data = app_module.load_session(session_id)
    plan_block = session_data["render_plan"][0]["blocks"][0]
    assert plan_block["style"]["font_size"] == 18
    assert plan_block["style"]["align"] == "left"
    assert plan_block["style"]["bold"] is True


def test_rerender_without_style_reuses_plan_style(
    monkeypatch, tmp_path, no_vision, guard_translator
):
    session_id, _ = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    style = {"font": "Yuki-Burobu", "font_size": 22, "text_color": "#1E88E5",
            "bold": False, "italic": True, "align": "right"}
    first = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": json.dumps([
            {"text": "text-0", "translated": "Lần đầu", "bbox": [2, 2, 30, 12],
             "style": style},
        ]),
        "erase_regions_json": "[]",
    })
    assert first.status_code == 200
    # V2-style client: no style field -> server reuses the plan style (A7.8)
    second = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": json.dumps([
            {"text": "text-0", "translated": "Lần hai", "bbox": [2, 2, 30, 12]},
        ]),
        "erase_regions_json": "[]",
    })
    assert second.status_code == 200
    session_data = app_module.load_session(session_id)
    plan_style = session_data["render_plan"][0]["blocks"][0]["style"]
    assert plan_style["font"] == "Yuki-Burobu"
    assert plan_style["font_size"] == 22
    assert plan_style["align"] == "right"
    assert plan_style["italic"] is True


def test_rerender_invalid_style_is_normalized(
    monkeypatch, tmp_path, no_vision, guard_translator
):
    session_id, _ = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    response = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": json.dumps([
            {
                "text": "text-0", "translated": "X", "bbox": [2, 2, 30, 12],
                "style": {"font": "../../x", "font_size": 999,
                          "text_color": "red", "align": "sideways"},
            },
        ]),
        "erase_regions_json": "[]",
    })
    assert response.status_code == 200
    body = response.get_json()
    style = body["blocks"][0]["style"]
    assert style["font"] == "animeace_"
    assert style["font_size"] == 120
    assert style["text_color"] is None
    assert style["align"] == "center"


# ── Endpoint: erase regions accumulate (never shrink) ───────────────────────

def test_rerender_erase_regions_json_accumulates_monotonic(
    monkeypatch, tmp_path, no_vision, guard_translator
):
    session_id, _ = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    base = {
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": json.dumps([
            {"text": "text-0", "translated": "X", "bbox": [2, 2, 30, 12]},
        ]),
    }
    r1 = client.post("/re-render-image", data={**base, 
        "erase_regions_json": "[[10, 10, 20, 15]]"})
    assert r1.status_code == 200
    r2 = client.post("/re-render-image", data={**base, 
        "erase_regions_json": "[[40, 25, 50, 35]]"})
    assert r2.status_code == 200
    entry = app_module.load_session(session_id)["render_plan"][0]
    regions = [list(r) for r in entry["erase_regions"]]
    # original bbox + both user regions: nothing was ever removed
    assert [2, 2, 30, 12] in regions
    assert [10, 10, 20, 15] in regions
    assert [40, 25, 50, 35] in regions


def test_rerender_erase_regions_json_preferred_over_alias(
    monkeypatch, tmp_path, no_vision, guard_translator
):
    session_id, _ = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    response = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": json.dumps([
            {"text": "text-0", "translated": "X", "bbox": [2, 2, 30, 12]},
        ]),
        "erase_regions_json": "[[1, 1, 4, 4]]",
        "deleted_regions_json": "[[9, 9, 12, 12]]",
    })
    assert response.status_code == 200
    regions = [list(r) for r in app_module.load_session(session_id)["render_plan"][0]["erase_regions"]]
    assert [1, 1, 4, 4] in regions
    assert [9, 9, 12, 12] not in regions


def test_rerender_invalid_erase_regions_json_400(
    monkeypatch, tmp_path, no_vision, guard_translator
):
    session_id, _ = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    response = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": "[]",
        "erase_regions_json": "not-json",
    })
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_erase_regions_json"


# ── Endpoint: erase mask (P1) ───────────────────────────────────────────────

def _b64_png(array):
    ok, buf = cv2.imencode(".png", array)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def test_rerender_erase_mask_applied_and_accumulates(
    monkeypatch, tmp_path, no_vision, guard_translator
):
    session_id, images = _make_session(monkeypatch, tmp_path)
    # put a dark square outside the OCR bbox; the mask targets it
    session_data = app_module.load_session(session_id)
    image = session_data["all_ocr_results"][0][1]
    image[30:38, 40:50] = 0
    app_module._save_session(session_id, session_data)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[30:38, 40:50] = 255
    client = flask_app.test_client()
    payload = {
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": "[]",
        "erase_regions_json": "[]",
        "erase_mask": _b64_png(mask),
    }
    r1 = client.post("/re-render-image", data=payload)
    assert r1.status_code == 200
    decoded1 = cv2.imdecode(
        np.frombuffer(base64.b64decode(r1.get_json()["data"]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded1[32:36, 43:47].mean() > 150, "masked square must be inpainted"

    entry = app_module.load_session(session_id)["render_plan"][0]
    assert entry.get("erase_mask"), "mask is persisted for accumulation"

    # second render WITHOUT the mask must still erase the same spot (A7.4)
    payload.pop("erase_mask")
    r2 = client.post("/re-render-image", data=payload)
    assert r2.status_code == 200
    decoded2 = cv2.imdecode(
        np.frombuffer(base64.b64decode(r2.get_json()["data"]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded2[32:36, 43:47].mean() > 150


def test_rerender_invalid_erase_mask_400(
    monkeypatch, tmp_path, no_vision, guard_translator
):
    session_id, _ = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    response = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": "[]",
        "erase_regions_json": "[]",
        "erase_mask": "!!!not-base64!!!",
    })
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_erase_mask"


# ── Unit: erase_mask_region + fixed font size ───────────────────────────────

def test_erase_mask_region_unit():
    from add_text import erase_mask_region

    image = np.full((40, 60, 3), 245, dtype=np.uint8)
    image[10:20, 10:20] = 0
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    out = erase_mask_region(image, mask)
    assert out[12:18, 12:18].mean() > 150
    assert np.array_equal(image[0:5, 0:5], out[0:5, 0:5])
    # empty mask is a no-op
    empty = np.zeros((40, 60), dtype=np.uint8)
    assert np.array_equal(erase_mask_region(image, empty), image)


def test_compute_font_and_wrap_fixed_size_and_shrink(monkeypatch):
    from add_text import _compute_font_and_wrap

    font, lines, lh = _compute_font_and_wrap(
        "Xin chao ban", [5, 5, 200, 60], "fonts/animeace_i.ttf",
        style={"font": "Yuki-Burobu", "font_size": 20},
    )
    assert font.size == 20
    # oversized request shrinks to fit the bbox (A5.4)
    font2, _, _ = _compute_font_and_wrap(
        "Xin chao ban", [5, 5, 60, 20], "fonts/animeace_i.ttf",
        style={"font_size": 200},
    )
    assert font2.size < 200 and font2.size >= 12
    # tiny fixed size renders as-is
    font3, _, _ = _compute_font_and_wrap(
        "Xin chao ban", [5, 5, 200, 60], "fonts/animeace_i.ttf",
        style={"font_size": 8},
    )
    assert font3.size == 8
    # auto (no style) still works
    font4, _, _ = _compute_font_and_wrap(
        "Xin chao ban", [5, 5, 200, 60], "fonts/animeace_i.ttf"
    )
    assert font4.size >= 12


# ── Style actually changes rendered pixels ─────────────────────────────────

def test_style_rendering_changes_pixels(no_vision):
    """Bold/italic/color must visibly change the rendered text (A5.6/A5.7)."""
    plain = np.full((60, 130, 3), 245, dtype=np.uint8)
    app_module.render_all_blocks(
        plain,
        [{"text": "Test dam nghien", "bbox": [4, 4, 126, 56],
          "text_color": (0, 0, 0), "appearance": {"need_outline": False}}],
        "fonts/animeace_i.ttf",
    )
    styled = np.full((60, 130, 3), 245, dtype=np.uint8)
    app_module.render_all_blocks(
        styled,
        [{"text": "Test dam nghien", "bbox": [4, 4, 126, 56],
          "text_color": (0, 0, 0), "appearance": {"need_outline": False},
          "style": {"font": "arial", "font_size": 0, "text_color": "#E53935",
                    "bold": True, "italic": True, "align": "left"}}],
        "fonts/animeace_i.ttf",
    )
    assert not np.array_equal(plain, styled), "style must change the pixels"


def test_rerender_double_submit_with_style_is_idempotent(
    monkeypatch, tmp_path, no_vision, guard_translator
):
    session_id, _ = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    data = {
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": json.dumps([
            {"text": "text-0", "translated": "Lần một", "bbox": [2, 2, 30, 12],
             "style": {"font": "Yuki-Burobu", "font_size": 14,
                       "text_color": "#FFFFFF", "bold": True,
                       "italic": False, "align": "center"}},
        ]),
        "erase_regions_json": "[[10, 10, 20, 15]]",
    }
    first = client.post("/re-render-image", data=data)
    second = client.post("/re-render-image", data=data)
    assert first.status_code == 200 and second.status_code == 200
    assert first.get_json()["data"] == second.get_json()["data"]
    assert first.get_json()["blocks"][0]["style"]["font"] == "Yuki-Burobu"

# ── V2 compatibility ────────────────────────────────────────────────────────

def test_rerender_legacy_v2_payload_still_works(
    monkeypatch, tmp_path, no_vision, guard_translator
):
    session_id, _ = _make_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    response = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": json.dumps([
            {"text": "text-0", "translated": "Bản dịch", "bbox": [2, 2, 30, 12]},
        ]),
        "deleted_regions_json": "[[5, 5, 9, 9]]",
    })
    assert response.status_code == 200
    assert "style" not in response.get_json()["blocks"][0]


def test_rerender_all_keeps_style(monkeypatch, tmp_path, no_vision, guard_translator):
    session_id, _ = _make_session(monkeypatch, tmp_path)
    session_data = app_module.load_session(session_id)
    session_data["render_plan"][0]["blocks"][0]["style"] = {
        "font": "arial", "font_size": 16, "text_color": "#FFFFFF",
        "bold": False, "italic": False, "align": "center",
    }
    app_module._save_session(session_id, session_data)
    client = flask_app.test_client()
    response = client.post("/re-render-all", data={
        "session_id": session_id,
        "dirty_indices_json": "[0]",
    })
    assert response.status_code == 200
    style = app_module.load_session(session_id)["render_plan"][0]["blocks"][0]["style"]
    assert style["font"] == "arial"
    assert style["font_size"] == 16


# ── A5.10: reload /styleditor keeps styles/edits after render ─────────────

def _make_draft_session(monkeypatch, tmp_path, n_blocks=1):
    """Session with a v3_draft + erased jpeg so /styleditor can render."""
    session_id, images = _make_session(monkeypatch, tmp_path)
    session_data = app_module.load_session(session_id)
    blocks = [
        {
            "text": "text-%d" % i,
            "translated": "bản dịch %d" % i,
            "bbox": [2 + i * 8, 2, 30 + i * 8, 12],
            "style": {"font": "animeace_", "font_size": 0, "text_color": None,
                      "bold": False, "italic": False, "align": "center"},
        }
        for i in range(n_blocks)
    ]
    session_data["v3_draft"] = {"images": [{"name": "page0", "blocks": blocks}]}
    app_module._save_session(session_id, session_data)
    app_module._save_erased_jpeg(session_id, 0, images[0])
    return session_id, images


def test_rerender_then_reload_styleditor_keeps_style(
    monkeypatch, tmp_path, no_vision, guard_translator
):
    """A5.10: style set in the editor must survive a reload of /styleditor."""
    session_id, _ = _make_draft_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    style = {"font": "Yuki-Burobu", "font_size": 40, "text_color": "#1E88E5",
             "bold": True, "italic": False, "align": "left"}
    resp = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": json.dumps([
            {"text": "text-0", "translated": "bản dịch 0", "bbox": [2, 2, 30, 12],
             "style": style},
        ]),
        "erase_regions_json": "[]",
    })
    assert resp.status_code == 200

    # the draft itself was synced at render time
    draft_block = app_module.load_session(session_id)["v3_draft"]["images"][0]["blocks"][0]
    assert draft_block["style"]["font"] == "Yuki-Burobu"
    assert draft_block["style"]["font_size"] == 40

    # reloading the editor keeps the style (A5.10)
    page = client.get("/styleditor/%s?img=0" % session_id)
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "Yuki-Burobu" in html
    assert "#1E88E5" in html
    assert "40" in html


def test_rerender_deleted_block_stays_deleted_in_draft(
    monkeypatch, tmp_path, no_vision, guard_translator
):
    """A block removed in the editor must not resurrect on reload."""
    session_id, _ = _make_draft_session(monkeypatch, tmp_path, n_blocks=2)
    client = flask_app.test_client()
    # client sends only block 0; block 1 was deleted (bbox marked for erase)
    resp = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": json.dumps([
            {"text": "text-0", "translated": "bản dịch 0", "bbox": [2, 2, 30, 12]},
        ]),
        "erase_regions_json": "[[10, 2, 38, 12]]",
    })
    assert resp.status_code == 200
    draft_blocks = app_module.load_session(session_id)["v3_draft"]["images"][0]["blocks"]
    assert [b["text"] for b in draft_blocks] == ["text-0"]


def test_rerender_keeps_empty_translated_chip_in_draft(
    monkeypatch, tmp_path, no_vision, guard_translator
):
    """Blocks with an empty translation stay in the draft as "-" chips (A1.6)."""
    session_id, _ = _make_draft_session(monkeypatch, tmp_path, n_blocks=2)
    client = flask_app.test_client()
    resp = client.post("/re-render-image", data={
        "session_id": session_id,
        "image_idx": "0",
        "blocks_json": json.dumps([
            {"text": "text-0", "translated": "bản dịch 0", "bbox": [2, 2, 30, 12]},
            {"text": "text-1", "translated": "", "bbox": [10, 2, 38, 12]},
        ]),
        "erase_regions_json": "[]",
    })
    assert resp.status_code == 200
    draft_blocks = app_module.load_session(session_id)["v3_draft"]["images"][0]["blocks"]
    assert [b["text"] for b in draft_blocks] == ["text-0", "text-1"]
    assert draft_blocks[1]["translated"] == ""


# ── Spec 4.2 MERGE RULE: reload uses render_plan as truth + erase state ──


def test_styleditor_fresh_prepare_session_uses_draft(
    monkeypatch, tmp_path, no_vision
):
    """Fresh V3 session (prepared, never rendered) must serve the DRAFT.

    Regression for t9: _load_render_plan fabricates a legacy plan (empty
    translations, default style) for sessions without a raw render_plan —
    the MERGE RULE must not treat such an image as rendered.
    """
    session_id, _ = _make_draft_session(monkeypatch, tmp_path)
    session_data = app_module.load_session(session_id)
    # drop any fixture render_plan: this is a pristine prepare session
    session_data.pop("render_plan", None)
    session_data["v3_draft"]["images"][0]["blocks"][0]["translated"] = "HELLO-WORLD"
    session_data["v3_draft"]["images"][0]["blocks"][0]["style"] = {
        "font": "Yuki-Burobu", "font_size": 24, "text_color": "#E53935",
        "bold": False, "italic": False, "align": "center",
    }
    app_module._save_session(session_id, session_data)
    client = flask_app.test_client()
    page = client.get("/styleditor/%s?img=0" % session_id)
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    # draft translations and style are served (not empty/fabricated)
    assert "HELLO-WORLD" in html
    assert "Yuki-Burobu" in html
    assert "E53935" in html
    # no stale erase state on a never-rendered image
    assert '"erase_regions": []' in html


def test_styleditor_prepare_clears_old_render_state(
    monkeypatch, tmp_path, no_vision
):
    """Re-prepare starts a fresh generation: old plan/jpegs must not leak."""
    fake = _FakeTranslator()
    monkeypatch.setattr(app_module, "MangaTranslator", _fake_translator_factory(fake))
    session_id, images = _make_session(monkeypatch, tmp_path)
    # simulate a previous generation: plan + rendered jpeg exist
    session_data = app_module.load_session(session_id)
    session_data["render_plan"] = [
        {"name": "page0", "erase_regions": [[2, 2, 30, 12]],
         "blocks": [{"text": "text-0", "translated": "STALE", "bbox": [2, 2, 30, 12]}]},
    ]
    app_module._save_session(session_id, session_data)
    app_module._save_rendered_jpeg(session_id, 0, images[0])

    client = flask_app.test_client()
    modified = [
        {"image_idx": 0, "blocks": [{"text": "text-0", "bbox": [2, 2, 30, 12]}]},
    ]
    response = client.post("/styleditor-prepare", data={
        "session_id": session_id,
        "modified_blocks": json.dumps(modified),
    })
    assert response.status_code == 302
    fresh = app_module.load_session(session_id)
    assert "render_plan" not in fresh
    rendered_path = os.path.join(str(tmp_path), session_id, "page_0_rendered.jpg")
    assert not os.path.exists(rendered_path)
    draft_block = fresh["v3_draft"]["images"][0]["blocks"][0]
    assert draft_block["translated"] == "bản dịch 0"


def _make_rendered_draft_session(monkeypatch, tmp_path, n_images=1):
    """Session with v3_draft AND a render_plan entry carrying styles/erase."""
    session_id, images = _make_session(monkeypatch, tmp_path, n_images=n_images)
    session_data = app_module.load_session(session_id)
    draft_images = [
        {"name": "page%d" % i, "blocks": [
            {"text": "text-%d" % i, "translated": "DRAFT-OLD-%d" % i,
             "bbox": [2, 2, 30, 12],
             "style": {"font": "animeace_", "font_size": 0, "text_color": None,
                       "bold": False, "italic": False, "align": "center"}},
        ]}
        for i in range(n_images)
    ]
    session_data["v3_draft"] = {"images": draft_images}
    plan = [
        {
            "name": "page%d" % i,
            "erase_regions": [[2, 2, 30, 12], [40, 25, 50, 35]],
            "blocks": [
                {"text": "text-%d" % i, "translated": "PLAN-NEW-%d" % i,
                 "bbox": [2, 2, 30, 12],
                 "style": {"font": "Yuki-Burobu", "font_size": 40,
                           "text_color": "#1E88E5", "bold": True,
                           "italic": False, "align": "left"}},
            ],
        }
        for i in range(n_images)
    ]
    session_data["render_plan"] = plan
    app_module._save_session(session_id, session_data)
    for i in range(n_images):
        app_module._save_erased_jpeg(session_id, i, images[i])
        # the plan is only authoritative when a rendered jpeg exists (t9 gate)
        app_module._save_rendered_jpeg(session_id, i, images[i])
    return session_id, images


def test_styleditor_stale_jpeg_without_plan_uses_draft(
    monkeypatch, tmp_path, no_vision
):
    """A stale rendered jpeg with NO raw render_plan must still serve the draft.

    Regression: _load_render_plan fabricates a legacy plan (translated="",
    default style) when session_data has no render_plan — the MERGE RULE gate
    requires BOTH a real plan AND the rendered jpeg, so a stale jpeg alone can
    never override the draft with fabricated empty blocks.
    """
    session_id, images = _make_draft_session(monkeypatch, tmp_path)
    session_data = app_module.load_session(session_id)
    session_data.pop("render_plan", None)
    session_data["v3_draft"]["images"][0]["blocks"][0]["translated"] = "FRESH-DRAFT"
    app_module._save_session(session_id, session_data)
    # stale rendered jpeg from an old generation, no plan to back it
    app_module._save_rendered_jpeg(session_id, 0, images[0])
    client = flask_app.test_client()
    page = client.get("/styleditor/%s?img=0" % session_id)
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "FRESH-DRAFT" in html
    assert '"translated": ""' not in html


def test_styleditor_reload_plan_blocks_are_truth(
    monkeypatch, tmp_path, no_vision
):
    """MERGE RULE: rendered images load blocks from render_plan, not draft."""
    session_id, _ = _make_rendered_draft_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    page = client.get("/styleditor/%s?img=0" % session_id)
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    # plan state wins (A5.10)
    assert "PLAN-NEW-0" in html
    assert "Yuki-Burobu" in html
    assert "1E88E5" in html
    assert "DRAFT-OLD-0" not in html


def test_styleditor_reload_returns_erase_regions_and_mask(
    monkeypatch, tmp_path, no_vision
):
    """MERGE RULE: erase state is returned so the client preview restores (A4.10)."""
    session_id, _ = _make_rendered_draft_session(monkeypatch, tmp_path)
    client = flask_app.test_client()
    # attach a persisted erase mask to the plan entry
    session_data = app_module.load_session(session_id)
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[20:30, 20:30] = 255
    ok, buf = cv2.imencode(".png", mask)
    mask_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    session_data["render_plan"][0]["erase_mask"] = mask_b64
    app_module._save_session(session_id, session_data)

    page = client.get("/styleditor/%s?img=0" % session_id)
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "erase_regions" in html
    assert "[40, 25, 50, 35]" in html
    assert "erase_mask" in html
    assert mask_b64 in html


def test_styleditor_unrendered_image_uses_draft_and_empty_erase(
    monkeypatch, tmp_path, no_vision
):
    """Images never rendered still load from the draft with no erase state."""
    session_id, _ = _make_rendered_draft_session(monkeypatch, tmp_path, n_images=2)
    session_data = app_module.load_session(session_id)
    del session_data["render_plan"][1]
    app_module._save_session(session_id, session_data)
    client = flask_app.test_client()
    page = client.get("/styleditor/%s?img=1" % session_id)
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "DRAFT-OLD-1" in html
    assert "erase_regions" in html
    assert "40, 25, 50, 35" not in html


def test_postrender_page_style_pass_through(monkeypatch, tmp_path, no_vision):
    session_id, images = _make_session(monkeypatch, tmp_path)
    session_data = app_module.load_session(session_id)
    session_data["render_plan"][0]["blocks"][0]["style"] = {
        "font": "Yuki-Burobu", "font_size": 0, "text_color": None,
        "bold": False, "italic": False, "align": "center",
    }
    app_module._save_session(session_id, session_data)
    app_module._save_rendered_jpeg(session_id, 0, images[0])
    client = flask_app.test_client()
    response = client.get("/postrender/%s?img=0" % session_id)
    assert response.status_code == 200
    assert "Yuki-Burobu" in response.get_data(as_text=True)
