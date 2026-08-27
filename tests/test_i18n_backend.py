"""Pytest suite for the backend i18n layer (team task t3).

Covers, per the i18n spec (docs/i18n-v1-spec.md §3.3/§5.7/§5.8):
  * dictionaries vi/en exist with identical key sets (backend.* section);
  * i18n.translate()/t() fallback chain: locale dict -> vi -> raw key;
  * i18n.tp() plural selection (en: one/other; vi: always other);
  * resolve_locale: cookie mt_locale > Accept-Language > default vi;
  * before_request sets g.locale from cookie/header;
  * emit_progress payloads carry a valid message key + params + a localized
    message (no hardcoded Vietnamese in UI payloads);
  * POST /translate error responses carry error_key/error_params and localized
    text for both vi and en locales;
  * backend warnings are structured {"key", "params"} mappings and stringify
    to the localized message;
  * gemini_translator detects the all-keys-failed path by a flag, not by a
    Vietnamese substring;
  * no user-visible Vietnamese literals remain in app.py /
    translator/gemini_translator.py.
"""
import io
import json
import re
from pathlib import Path

import pytest

import app as app_module
import i18n
from app import app as flask_app
from i18n import WarningMessage, resolve_locale, t, tp

REPO_ROOT = Path(__file__).resolve().parent.parent

_VIETNAMESE_RE = re.compile(
    "[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]",
    re.IGNORECASE,
)


def _form_data(translator="google"):
    """Minimum form payload required by upload_file ([] indexing on
    selected_translator/selected_font)."""
    return {
        "selected_translator": translator,
        "selected_font": "animeace",
        "selected_source_lang": "ja",
        "selected_language": "vi",
        "selected_style": "default",
    }


# ── dictionaries ────────────────────────────────────────────────────────────

def test_dictionaries_exist_and_key_parity():
    vi = i18n.load_dict("vi")
    en = i18n.load_dict("en")
    assert vi, "i18n/vi.json must exist with backend.* keys"
    assert en, "i18n/en.json must exist with backend.* keys"
    assert set(vi) == set(en), "vi/en key sets must be identical (spec A0.1)"


def test_dictionaries_have_backend_section():
    vi = i18n.load_dict("vi")
    expected = {
        "backend.formats",
        "backend.progress.translating",
        "backend.progress.batchFallback",
        "backend.progress.unknownTranslator",
        "backend.progress.translated",
        "backend.progress.noText",
        "backend.progress.rendering",
        "backend.progress.renderImage",
        "backend.progress.done_one",
        "backend.progress.done_other",
        "backend.progress.ocrStart",
        "backend.progress.ocrDone",
        "backend.warn.geminiNotInit",
        "backend.warn.geminiFailed",
        "backend.warn.localLlmFailed",
        "backend.warn.googleFailed",
        "backend.warn.unknownTranslator",
        "backend.warn.geminiAllKeysFailed",
        "backend.error.noApiKey",
        "backend.error.noModel",
        "backend.error.noImages",
        "backend.error.unsupportedFormat",
        "backend.error.unreadableImage",
        "backend.error.translationFailed",
    }
    assert expected <= set(vi)


# ── translate / tp fallback chain ──────────────────────────────────────────

def test_translate_locale_chain():
    assert i18n.translate("backend.progress.ocrStart", "vi") == "Bắt đầu OCR toàn ảnh..."
    assert i18n.translate("backend.progress.ocrStart", "en") == "Starting full-image OCR..."
    assert i18n.translate("backend.progress.renderImage", "vi", name="p1.jpg") == "Render: p1.jpg"
    assert i18n.translate("backend.progress.renderImage", "en", name="p1.jpg") == "Render: p1.jpg"
    assert i18n.translate("backend.error.noImages", "en") == "Please select at least 1 image to translate."


def test_translate_fallback_raw_key_and_vi():
    # missing key -> raw key (both locales)
    assert i18n.translate("nope.missing", "en") == "nope.missing"
    assert i18n.translate("nope.missing", "vi") == "nope.missing"
    # unknown locale -> vi dict
    assert i18n.translate("backend.progress.ocrStart", "zz") == "Bắt đầu OCR toàn ảnh..."


def test_translate_safe_when_dictionary_missing(monkeypatch):
    # Simulate a missing dict file: cache miss + no file -> {} -> vi fallback.
    monkeypatch.setitem(i18n._dict_cache, "en", {})
    assert i18n.translate("backend.progress.ocrStart", "en") == "Bắt đầu OCR toàn ảnh..."
    assert i18n.translate("backend.progress.ocrStart", "en")  # never crashes


def test_tp_plural_selection():
    with flask_app.test_request_context("/", headers={"Accept-Language": "en-US"}):
        flask_app.preprocess_request()
        assert tp("backend.progress.done", 1) == "Done! 1 image"
        assert tp("backend.progress.done", 5) == "Done! 5 images"
    with flask_app.test_request_context("/", headers={"Accept-Language": "vi-VN"}):
        flask_app.preprocess_request()
        assert tp("backend.progress.done", 1) == "Hoàn tất! 1 ảnh"
        assert tp("backend.progress.done", 5) == "Hoàn tất! 5 ảnh"


def test_tp_fallback_missing_one(monkeypatch):
    monkeypatch.setitem(i18n._dict_cache, "en", {"fake.msg_other": "Other {n}"})
    monkeypatch.setitem(i18n._dict_cache, "vi", {})
    with flask_app.test_request_context("/", headers={"Accept-Language": "en-US"}):
        flask_app.preprocess_request()
        assert tp("fake.msg", 1) == "Other 1"


# ── locale resolution ──────────────────────────────────────────────────────

def test_resolve_locale():
    assert resolve_locale("en", "vi-VN,vi;q=0.9") == "en"          # cookie wins
    assert resolve_locale(None, "vi-VN,vi;q=0.9,en;q=0.5") == "vi"
    assert resolve_locale(None, "en-US,en;q=0.9") == "en"
    assert resolve_locale(None, "fr-FR,fr;q=0.9") == "vi"          # unsupported -> default
    assert resolve_locale(None, "") == "vi"
    assert resolve_locale("xx", "en") == "en"                      # invalid cookie -> header


def test_before_request_sets_locale_from_accept_language():
    with flask_app.test_request_context("/", headers={"Accept-Language": "en-US,en;q=0.9"}):
        flask_app.preprocess_request()
        assert i18n.get_locale() == "en"
    with flask_app.test_request_context("/", headers={"Accept-Language": "vi-VN,vi;q=0.9"}):
        flask_app.preprocess_request()
        assert i18n.get_locale() == "vi"
    # cookie wins over the header
    with flask_app.test_request_context(
        "/", headers={"Accept-Language": "vi-VN,vi;q=0.9"},
        environ_overrides={"HTTP_COOKIE": "mt_locale=en"},
    ):
        flask_app.preprocess_request()
        assert i18n.get_locale() == "en"


def test_get_locale_outside_request_is_default():
    assert i18n.get_locale() == "vi"


# ── i18n_json payload ──────────────────────────────────────────────────────

def test_i18n_json_safe_and_complete():
    raw = i18n.i18n_json()
    assert "</" not in raw  # safe inside a <script> tag
    payload = json.loads(raw)
    assert set(payload) == {"vi", "en"}
    assert set(payload["vi"]) == set(payload["en"])


# ── emit_progress payloads ─────────────────────────────────────────────────

def _capture_emit(monkeypatch):
    captured = {}

    def fake_emit(event, payload, **kwargs):
        captured["event"] = event
        captured["payload"] = payload

    monkeypatch.setattr(app_module.socketio, "emit", fake_emit)
    return captured


def test_emit_progress_carries_valid_key_and_localized_message(monkeypatch):
    captured = _capture_emit(monkeypatch)
    with flask_app.test_request_context("/", headers={"Accept-Language": "en-US"}):
        flask_app.preprocess_request()
        app_module.emit_progress("ocr", 2, 5, key="backend.progress.ocrDone", params={"n": 12})
    payload = captured["payload"]
    assert payload["key"] == "backend.progress.ocrDone"
    assert payload["params"] == {"n": 12}
    assert payload["message"] == "OCR complete: 12 text blocks"
    assert payload["phase"] == "ocr"
    assert payload["percent"] == 40


def test_emit_progress_vietnamese_locale(monkeypatch):
    captured = _capture_emit(monkeypatch)
    with flask_app.test_request_context("/", headers={"Accept-Language": "vi-VN"}):
        flask_app.preprocess_request()
        app_module.emit_progress("ocr", 0, 3, key="backend.progress.ocrStart")
    assert captured["payload"]["message"] == "Bắt đầu OCR toàn ảnh..."
    assert captured["payload"]["key"] == "backend.progress.ocrStart"


def test_emit_progress_legacy_message_still_supported(monkeypatch):
    captured = _capture_emit(monkeypatch)
    with flask_app.test_request_context("/"):
        app_module.emit_progress("done", 1, 1, "plain message")
    assert captured["payload"]["message"] == "plain message"
    assert captured["payload"]["key"] is None


# ── /translate error responses ─────────────────────────────────────────────

def test_upload_no_images_error_key_and_message(monkeypatch):
    captured = {}

    def fake_render(template_name, **kwargs):
        captured["template"] = template_name
        captured.update(kwargs)
        return "rendered"

    monkeypatch.setattr(app_module, "render_template", fake_render)
    client = flask_app.test_client()
    resp = client.post("/translate", data=_form_data(), content_type="multipart/form-data")
    assert resp.status_code == 200
    assert captured["template"] == "index.html"
    assert captured["error_key"] == "backend.error.noImages"
    assert captured["error_params"] == {}
    assert captured["error"] == "Vui lòng chọn ít nhất 1 ảnh để dịch."


def _form_error(html):
    """Extract the rendered .form-error div (visible UI, not the embedded
    i18n-data dictionaries which legitimately contain both locales)."""
    match = re.search(r'<div class="form-error"[^>]*>(.*?)</div>', html, re.S)
    return match.group(1) if match else ""


def test_upload_no_images_localized_response():
    client = flask_app.test_client()
    resp = client.post("/translate", data=_form_data(), content_type="multipart/form-data")
    assert "Vui lòng chọn ít nhất 1 ảnh để dịch." in _form_error(resp.get_data(as_text=True))

    client = flask_app.test_client()
    client.set_cookie("mt_locale", "en")
    resp = client.post("/translate", data=_form_data(), content_type="multipart/form-data")
    error_div = _form_error(resp.get_data(as_text=True))
    assert "Please select at least 1 image to translate." in error_div
    assert not _VIETNAMESE_RE.search(error_div)


def test_upload_gemini_errors_use_keys(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        app_module, "render_template",
        lambda template_name, **kwargs: captured.update(template=template_name, **kwargs) or "x",
    )
    client = flask_app.test_client()
    client.post("/translate", data=_form_data(translator="gemini"), content_type="multipart/form-data")
    assert captured["error_key"] == "backend.error.noApiKey"
    assert captured["error"] == "Vui lòng nhập ít nhất 1 Gemini API Key."

    captured.clear()
    data = _form_data(translator="gemini")
    data["gemini_api_key"] = "test-key-1"
    data["gemini_model_input"] = ""
    client.post("/translate", data=data, content_type="multipart/form-data")
    assert captured["error_key"] == "backend.error.noModel"
    assert captured["error"] == "Vui lòng nhập tên model Gemini."


def test_upload_unsupported_format_error_localized():
    client = flask_app.test_client()
    resp = client.post(
        "/translate",
        data=dict(_form_data(), files=(io.BytesIO(b"hello"), "photo.txt")),
        content_type="multipart/form-data",
    )
    assert "Chỉ hỗ trợ ảnh JPG, JPEG, PNG, WebP, BMP, TIFF hoặc AVIF." in _form_error(resp.get_data(as_text=True))

    client = flask_app.test_client()
    client.set_cookie("mt_locale", "en")
    resp = client.post(
        "/translate",
        data=dict(_form_data(), files=(io.BytesIO(b"hello"), "photo.txt")),
        content_type="multipart/form-data",
    )
    error_div = _form_error(resp.get_data(as_text=True))
    assert "Only JPG, JPEG, PNG, WebP, BMP, TIFF or AVIF images are supported." in error_div
    assert not _VIETNAMESE_RE.search(error_div)


def test_upload_unreadable_image_error_localized():
    client = flask_app.test_client()
    resp = client.post(
        "/translate",
        data=dict(_form_data(), files=(io.BytesIO(b"not-an-image"), "photo.png")),
        content_type="multipart/form-data",
    )
    html = resp.get_data(as_text=True)
    assert "Không đọc được ảnh. Hãy thử file JPG, JPEG, PNG, WebP, BMP, TIFF hoặc AVIF khác." in html


def test_upload_data_value_aliases_work(monkeypatch):
    # data-value payloads (spec §5.6) must map exactly like display texts.
    def fake_render(template_name, **kwargs):
        return "rendered"

    monkeypatch.setattr(app_module, "render_template", fake_render)
    client = flask_app.test_client()
    data = {
        "selected_translator": "copilot",
        "selected_font": "animeace_",
        "selected_source_lang": "ja",
        "selected_language": "vi",
        "selected_style": "keep_honorifics",
    }
    resp = client.post("/translate", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200  # no 500 from mapping


# ── structured warnings ────────────────────────────────────────────────────

def test_translate_texts_all_unknown_translator_warning(monkeypatch):
    monkeypatch.setattr(app_module, "emit_progress", lambda *a, **k: None)

    class Stub:
        pass

    stub = Stub()
    with flask_app.test_request_context("/", headers={"Accept-Language": "en-US"}):
        flask_app.preprocess_request()
        translated, warning = app_module.translate_texts_all(["a", "b"], stub, "unknown")
        assert translated == ["a", "b"]
        assert isinstance(warning, dict)
        assert warning["key"] == "backend.warn.unknownTranslator"
        assert warning["params"] == {}
        assert str(warning) == "Unknown translator, so the app kept the original text."


def test_translate_texts_all_gemini_not_initialized_warning(monkeypatch):
    monkeypatch.setattr(app_module, "emit_progress", lambda *a, **k: None)

    class Stub:
        _gemini_translator = None

    stub = Stub()
    with flask_app.test_request_context("/", headers={"Accept-Language": "vi-VN"}):
        flask_app.preprocess_request()
        translated, warning = app_module.translate_texts_all(["x"], stub, "gemini")
    assert translated == ["x"]
    assert warning["key"] == "backend.warn.geminiNotInit"
    assert str(warning) == "Gemini chưa được khởi tạo nên app giữ nguyên text gốc. Hãy kiểm tra API key rồi thử lại."


def test_warning_message_stringifies_per_locale():
    w = WarningMessage(key="backend.warn.googleFailed")
    with flask_app.test_request_context("/", headers={"Accept-Language": "en-US"}):
        flask_app.preprocess_request()
        assert str(w) == "Google Translate failed, so the app kept the original text."
    with flask_app.test_request_context("/", headers={"Accept-Language": "vi-VN"}):
        flask_app.preprocess_request()
        assert str(w) == "Google Translate lỗi nên app giữ nguyên text gốc."


def test_translation_failed_warning_structured():
    w = WarningMessage(key="backend.error.translationFailed", params={"error": "boom"})
    with flask_app.test_request_context("/", headers={"Accept-Language": "en-US"}):
        flask_app.preprocess_request()
        assert str(w) == "Translation error: boom"
    assert json.loads(json.dumps(w)) == {"key": "backend.error.translationFailed", "params": {"error": "boom"}}


# ── gemini_translator all-keys-failed flag ─────────────────────────────────

def test_gemini_all_keys_failed_detected_by_flag(monkeypatch):
    from translator.gemini_translator import GeminiTranslator

    translator = GeminiTranslator(api_keys=["k1"])

    def boom(self, prompt):
        self.last_warning = {"key": "backend.warn.geminiAllKeysFailed", "params": {}}
        self._all_keys_failed = True
        raise RuntimeError("All Gemini API keys failed")

    monkeypatch.setattr(GeminiTranslator, "_generate_content_with_rotation", boom)
    out = translator._translate_batch_internal(["x", "y"], "ja", "vi")
    assert out == ["x", "y"]
    assert translator.last_warning["key"] == "backend.warn.geminiAllKeysFailed"
    assert translator._all_keys_failed is True


def test_gemini_success_resets_flag(monkeypatch):
    from translator.gemini_translator import GeminiTranslator

    class FakeResponse:
        text = '["ok"]'  # valid JSON array, as the real API returns

    class FakeModels:
        def generate_content(self, model, contents):
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    translator = GeminiTranslator(api_keys=["k1"], client_factory=lambda key: FakeClient(key))
    # stale state from a previous failure must be cleared on success
    translator._all_keys_failed = True
    translator.last_warning = {"key": "backend.warn.geminiAllKeysFailed", "params": {}}
    assert translator._translate_batch_internal(["x"], "ja", "vi") == ["ok"]
    assert translator._all_keys_failed is False
    assert translator.last_warning is None


# ── key inventory vs source ────────────────────────────────────────────────

def test_all_referenced_backend_keys_exist_in_both_dicts():
    sources = (
        (REPO_ROOT / "app.py").read_text(encoding="utf-8")
        + (REPO_ROOT / "translator" / "gemini_translator.py").read_text(encoding="utf-8")
    )
    referenced = set(re.findall(r"(?:t|tp)\(\s*['\"](backend\.[a-z0-9_.]+)['\"]", sources))
    referenced |= set(re.findall(r"key=\s*['\"](backend\.[a-z0-9_.]+)['\"]", sources))
    vi = i18n.load_dict("vi")
    en = i18n.load_dict("en")
    assert referenced, "no backend keys referenced?"
    for key in referenced:
        if (key + "_one") in vi:  # plural base: both forms must ship
            assert (key + "_other") in vi, f"{key}_other missing from i18n/vi.json"
            assert (key + "_one") in en, f"{key}_one missing from i18n/en.json"
            assert (key + "_other") in en, f"{key}_other missing from i18n/en.json"
            continue
        assert key in vi, f"{key} missing from i18n/vi.json"
        assert key in en, f"{key} missing from i18n/en.json"


def test_no_hardcoded_vietnamese_in_ui_payload_sources():
    app_src = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    for literal in [
        "Đang dịch", "Vui lòng nhập", "Vui lòng chọn", "Không có text để dịch",
        "Đang render text", "Render: ", "Hoàn tất!", "Bắt đầu OCR", "OCR hoàn tất",
        "Cảnh báo: Translator", "Dịch hoàn tất", "Chỉ hỗ trợ ảnh", "Không đọc được ảnh",
        "Lỗi dịch:", "Gemini chưa được khởi tạo", "Gemini không dịch được",
        "Local LLM không dịch được", "Google Translate lỗi", "Translator không xác định",
        "Tất cả Gemini API key",
    ]:
        assert literal not in app_src, f"hardcoded UI string still in app.py: {literal}"

    gemini_src = (REPO_ROOT / "translator" / "gemini_translator.py").read_text(encoding="utf-8")
    assert "Tất cả Gemini API key" not in gemini_src
    assert "tất cả gemini api key" not in gemini_src


def test_scanner_clean_for_backend_files():
    """Spec A0.2 gate: tools/scan_hardcoded_strings.py must report zero
    findings for the backend sources owned by t3."""
    import importlib.util

    scanner_path = REPO_ROOT / "tools" / "scan_hardcoded_strings.py"
    if not scanner_path.exists():
        pytest.skip("scanner not present yet")
    spec = importlib.util.spec_from_file_location("scan_hardcoded_strings", scanner_path)
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    findings = scanner.scan_file(REPO_ROOT / "app.py") + scanner.scan_file(
        REPO_ROOT / "translator" / "gemini_translator.py"
    )
    assert not findings, f"scanner findings in backend sources: {findings}"


def test_all_progress_emit_sites_are_key_based():
    app_src = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    # Every emit_progress call site must pass key= (the definition itself is excluded).
    body = app_src[app_src.index("def emit_progress"):]
    next_def = body.find("\ndef ", 1)
    definition = body[:next_def] if next_def != -1 else body
    calls = app_src.replace(definition, "")
    matches = list(re.finditer(r"emit_progress\(", calls))
    assert matches, "no emit_progress call sites found?"
    for match in matches:
        start = match.start()
        line = calls[:start].count("\n") + 1
        segment = calls[start:start + 300]
        assert "key=" in segment, f"emit_progress call on app.py line {line} is not key-based"
