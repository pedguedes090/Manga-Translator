# Vietnamese/English UI and Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent Vietnamese/English interface switch and guarantee that Gemini/local-LLM prompt language follows the UI locale independently of the selected translation target.

**Architecture:** A small `localization.py` module owns locale resolution and translation catalogs, while `selection_options.py` converts stable form codes at the request boundary. A shared `translator/prompts.py` builds all Gemini and local-LLM prompts in Vietnamese or English; provider adapters only send and parse them. Flask injects locale helpers into all templates and emits semantic progress keys for browser-side localization.

**Tech Stack:** Python 3.11, Flask/Jinja, vanilla JavaScript, pytest, Node syntax checks.

## Global Constraints

- Supported UI locales are exactly `vi` and `en`.
- Locale precedence is explicit validated selection, `ui_language` cookie, browser `Accept-Language`, then English.
- The UI locale controls prompt language; the target-language code controls requested output language.
- Custom prompt text is inserted verbatim and is never machine-translated.
- Google Translate does not receive an LLM prompt.
- Form and local-storage selections use stable codes, never localized labels.
- Existing saved Vietnamese/English display values remain accepted during migration.
- Preserve unrelated existing working-tree deletions.

---

### Task 1: Locale resolution, catalogs, and Flask binding

**Files:**
- Create: `localization.py`
- Create: `tests/test_localization.py`
- Modify: `app.py:6-93,634-637`

**Interfaces:**
- Produces: `normalize_locale(value: str | None) -> str | None`.
- Produces: `resolve_locale(explicit: str | None, cookie: str | None, accept_language: str | None) -> str`.
- Produces: `translate(locale: str, key: str, **params) -> str`.
- Produces: `javascript_catalog(locale: str) -> dict[str, str]`.
- Produces: Flask `g.ui_language`, template globals `ui_language`, `t`, and `js_i18n`.
- Produces: `POST /set-language` with a validated local redirect and `ui_language` cookie.

- [ ] **Step 1: Write failing locale-resolution and route tests**

Create `tests/test_localization.py`:

```python
from localization import javascript_catalog, normalize_locale, resolve_locale, translate


def test_locale_resolution_precedence_and_fallback():
    assert normalize_locale("EN-us") == "en"
    assert normalize_locale("vi_VN") == "vi"
    assert normalize_locale("fr") is None
    assert resolve_locale("vi", "en", "en-US") == "vi"
    assert resolve_locale(None, "vi", "en-US") == "vi"
    assert resolve_locale(None, None, "vi-VN,vi;q=0.9,en;q=0.8") == "vi"
    assert resolve_locale(None, None, "fr-FR,fr;q=0.9") == "en"


def test_catalogs_have_identical_keys_and_safe_fallback():
    assert javascript_catalog("vi").keys() == javascript_catalog("en").keys()
    assert translate("vi", "common.language") == "Ngôn ngữ"
    assert translate("en", "common.language") == "Language"
    assert translate("vi", "missing.key") == "missing.key"


def test_set_language_sets_cookie_and_rejects_external_redirect():
    import app as app_module

    client = app_module.app.test_client()
    response = client.post(
        "/set-language",
        data={"language": "vi", "next": "https://example.com/steal"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")
    assert "ui_language=vi" in response.headers["Set-Cookie"]
    assert "SameSite=Lax" in response.headers["Set-Cookie"]


def test_home_uses_cookie_then_browser_locale():
    import app as app_module

    client = app_module.app.test_client()
    assert '<html lang="vi">' in client.get(
        "/", headers={"Accept-Language": "vi-VN,vi;q=0.9"}
    ).get_data(as_text=True)
    client.set_cookie("ui_language", "en")
    assert '<html lang="en">' in client.get(
        "/", headers={"Accept-Language": "vi-VN"}
    ).get_data(as_text=True)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_localization.py -q
```

Expected: collection fails because `localization.py` does not exist.

- [ ] **Step 3: Implement locale normalization and catalog lookup**

Create `localization.py` with this public structure:

```python
SUPPORTED_LOCALES = ("vi", "en")
DEFAULT_LOCALE = "en"

CATALOGS = {
    "vi": {
        "common.language": "Ngôn ngữ",
        "common.back": "Quay lại",
        "common.cancel": "Huỷ",
        "common.download": "Tải xuống",
        "common.images": "ảnh",
        "common.blocks": "bóng thoại",
        "language.vi": "Tiếng Việt",
        "language.en": "Tiếng Anh",
        "language.ja": "Tiếng Nhật (Manga)",
        "language.zh": "Tiếng Trung (Manhua)",
        "language.ko": "Tiếng Hàn (Manhwa)",
        "language.th": "Tiếng Thái",
        "language.id": "Tiếng Indonesia",
        "language.fr": "Tiếng Pháp",
        "language.de": "Tiếng Đức",
        "language.es": "Tiếng Tây Ban Nha",
        "language.ru": "Tiếng Nga",
        "index.source": "Ngôn ngữ gốc",
        "index.target": "Dịch sang",
        "index.engine": "Bộ máy dịch",
        "index.style": "Phong cách dịch",
        "index.font": "Phông chữ",
        "index.manual": "Chỉnh sửa thủ công",
        "index.manual_help": "Dừng sau OCR để thêm hoặc xoá bóng thoại",
        "index.choose_images": "📁 Chọn ảnh",
        "index.choose_images_many": "📁 Chọn ảnh (có thể chọn nhiều)",
        "index.selected_images": "📁 {count} ảnh đã chọn",
        "index.more_images": "... và {count} ảnh khác",
        "index.submit": "Dịch ảnh",
        "index.processing": "Đang xử lý... Vui lòng đợi!",
        "index.custom_prompt": "Yêu cầu tuỳ chỉnh",
        "index.custom_prompt_placeholder": "Ví dụ: Dịch theo phong cách light novel và giữ nguyên tên nhân vật...",
        "index.server_url": "URL máy chủ",
        "index.model_name": "Tên model",
        "index.local_model_help": "Nhập tên model đang chạy trên máy chủ.",
        "index.gemini_model_help": "Nhập tên model Gemini dùng cho bản dịch.",
        "index.gemini_keys": "Gemini API Keys",
        "index.gemini_keys_placeholder": "Dán mỗi API key một dòng. Ứng dụng sẽ chuyển sang key tiếp theo khi gặp lỗi quota hoặc xác thực.",
        "index.gemini_keys_help": "Keys chỉ được lưu trong localStorage của trình duyệt; có thể phân tách bằng dòng mới, dấu phẩy hoặc dấu chấm phẩy.",
        "engine.gemini": "Gemini",
        "engine.local_llm": "Local LLM",
        "engine.google": "Google",
        "style.default": "Mặc định",
        "style.casual": "Thân mật",
        "style.formal": "Trang trọng",
        "style.keep_honorifics": "Giữ kính ngữ (-san, senpai...)" ,
        "style.web_novel": "Phong cách web novel",
        "style.action": "Hành động (ngắn gọn)",
        "style.literal": "Sát nghĩa",
        "style.custom": "Tuỳ chỉnh...",
        "validation.files_required": "Vui lòng chọn ít nhất 1 ảnh.",
        "validation.gemini_keys_required": "Vui lòng nhập ít nhất 1 Gemini API Key.",
        "validation.gemini_model_required": "Vui lòng nhập tên model Gemini.",
        "validation.unsupported_image": "Chỉ hỗ trợ ảnh {formats}.",
        "validation.unreadable_image": "Không đọc được ảnh. Hãy thử file {formats} khác.",
        "validation.expired_job": "Phiên xử lý không tồn tại hoặc đã hết hạn. Vui lòng tải ảnh lại.",
        "progress.phase.ocr": "📖 OCR nhận dạng văn bản",
        "progress.phase.translation": "🌐 Dịch văn bản",
        "progress.phase.rendering": "✏️ Render chữ vào ảnh",
        "progress.phase.done": "✅ Hoàn tất",
        "progress.preparing": "Đang chuẩn bị",
        "progress.ocr_start": "Bắt đầu OCR toàn ảnh...",
        "progress.ocr_done": "OCR hoàn tất: {count} bóng thoại",
        "progress.ocr_image": "OCR: {name}",
        "progress.translation_start": "Đang dịch {count} đoạn văn bản...",
        "progress.translation_done": "Dịch hoàn tất",
        "progress.render_start": "Đang render chữ vào ảnh...",
        "progress.render_image": "Render: {name}",
        "progress.done": "Hoàn tất! {count} ảnh",
        "progress.no_text": "Không có văn bản để dịch",
        "correction.title": "Chỉnh sửa bóng thoại",
        "correction.select": "Chọn",
        "correction.add": "Thêm",
        "correction.delete": "Xoá",
        "correction.reset": "Reset",
        "correction.undo": "Hoàn tác",
        "correction.redo": "Làm lại",
        "correction.zoom_controls": "Điều khiển thu phóng",
        "correction.zoom_out": "Thu nhỏ (-)",
        "correction.zoom_in": "Phóng to (+)",
        "correction.zoom_fit": "Vừa màn hình",
        "correction.zoom_actual": "Kích thước thật",
        "correction.shortcuts": "S Chọn · A Thêm · D Xoá · Ctrl+Wheel Thu phóng · Space Kéo ảnh · ←→ Đổi ảnh",
        "correction.previous": "Trước",
        "correction.next": "Sau",
        "correction.properties": "Thuộc tính",
        "correction.no_selection": "Chọn bóng thoại để chỉnh sửa",
        "correction.text": "Nội dung văn bản",
        "correction.position": "Vị trí (x1,y1,x2,y2)",
        "correction.clean": "Làm sạch chữ",
        "correction.reocr": "OCR lại",
        "correction.continue": "Tiếp tục dịch và render",
        "correction.reset_confirm": "Reset tất cả bóng thoại về kết quả OCR gốc?",
        "correction.deleted": "Đã xoá bóng thoại",
        "correction.cleaned": "Đã làm sạch chữ",
        "correction.reset_done": "Đã reset về OCR gốc",
        "correction.ocr_running": "Đang OCR...",
        "correction.ocr_empty": "Không nhận được văn bản",
        "correction.ocr_error": "Lỗi OCR",
        "result.title": "Kết quả dịch ({count} ảnh)",
        "result.translated": "Đã dịch",
        "result.original": "Gốc",
        "result.compare": "So sánh ảnh {name}",
        "result.empty": "Không có ảnh nào được xử lý.",
        "result.download_zip": "Tải ZIP",
        "result.back_to_correction": "Quay lại chỉnh sửa",
        "result.zip_creating": "Đang tạo ZIP...",
        "result.zip_done": "Đã tải xong!",
        "result.zip_error": "Lỗi tạo ZIP",
        "warning.gemini_unavailable": "Gemini không dịch được nên ứng dụng giữ nguyên văn bản gốc. Hãy kiểm tra API key hoặc quota rồi thử lại.",
        "warning.local_llm_unavailable": "Local LLM không dịch được nên ứng dụng giữ nguyên văn bản gốc. Hãy kiểm tra URL máy chủ và model rồi thử lại.",
        "warning.google_unavailable": "Google Translate lỗi nên ứng dụng giữ nguyên văn bản gốc.",
        "warning.unknown_engine": "Không nhận diện được bộ máy dịch nên ứng dụng giữ nguyên văn bản gốc.",
    },
    "en": {
        "common.language": "Language",
        "common.back": "Back",
        "common.cancel": "Cancel",
        "common.download": "Download",
        "common.images": "images",
        "common.blocks": "speech bubbles",
        "language.vi": "Vietnamese",
        "language.en": "English",
        "language.ja": "Japanese (Manga)",
        "language.zh": "Chinese (Manhua)",
        "language.ko": "Korean (Manhwa)",
        "language.th": "Thai",
        "language.id": "Indonesian",
        "language.fr": "French",
        "language.de": "German",
        "language.es": "Spanish",
        "language.ru": "Russian",
        "index.source": "Source language",
        "index.target": "Translate to",
        "index.engine": "Translation engine",
        "index.style": "Translation style",
        "index.font": "Font",
        "index.manual": "Manual correction",
        "index.manual_help": "Pause after OCR to add or remove speech bubbles",
        "index.choose_images": "📁 Choose images",
        "index.choose_images_many": "📁 Choose one or more images",
        "index.selected_images": "📁 {count} images selected",
        "index.more_images": "... and {count} more images",
        "index.submit": "Translate images",
        "index.processing": "Processing... Please wait!",
        "index.custom_prompt": "Custom instruction",
        "index.custom_prompt_placeholder": "Example: Use a light-novel style and keep character names unchanged...",
        "index.server_url": "Server URL",
        "index.model_name": "Model name",
        "index.local_model_help": "Enter the model name running on the server.",
        "index.gemini_model_help": "Enter the Gemini model used for translation.",
        "index.gemini_keys": "Gemini API keys",
        "index.gemini_keys_placeholder": "Paste one API key per line. The app moves to the next key after a quota or authentication failure.",
        "index.gemini_keys_help": "Keys are stored only in browser localStorage; separate them with new lines, commas, or semicolons.",
        "engine.gemini": "Gemini",
        "engine.local_llm": "Local LLM",
        "engine.google": "Google",
        "style.default": "Default",
        "style.casual": "Casual",
        "style.formal": "Formal",
        "style.keep_honorifics": "Keep honorifics (-san, senpai...)" ,
        "style.web_novel": "Web novel style",
        "style.action": "Action (concise)",
        "style.literal": "Literal",
        "style.custom": "Custom...",
        "validation.files_required": "Select at least one image.",
        "validation.gemini_keys_required": "Enter at least one Gemini API key.",
        "validation.gemini_model_required": "Enter a Gemini model name.",
        "validation.unsupported_image": "Supported image formats: {formats}.",
        "validation.unreadable_image": "The image could not be read. Try another {formats} file.",
        "validation.expired_job": "This job does not exist or has expired. Upload the images again.",
        "progress.phase.ocr": "📖 Recognizing text",
        "progress.phase.translation": "🌐 Translating text",
        "progress.phase.rendering": "✏️ Rendering text into images",
        "progress.phase.done": "✅ Complete",
        "progress.preparing": "Preparing",
        "progress.ocr_start": "Starting full-image OCR...",
        "progress.ocr_done": "OCR complete: {count} speech bubbles",
        "progress.ocr_image": "OCR: {name}",
        "progress.translation_start": "Translating {count} text segments...",
        "progress.translation_done": "Translation complete",
        "progress.render_start": "Rendering text into images...",
        "progress.render_image": "Rendering: {name}",
        "progress.done": "Complete! {count} images",
        "progress.no_text": "No text to translate",
        "correction.title": "Edit speech bubbles",
        "correction.select": "Select",
        "correction.add": "Add",
        "correction.delete": "Delete",
        "correction.reset": "Reset",
        "correction.undo": "Undo",
        "correction.redo": "Redo",
        "correction.zoom_controls": "Zoom controls",
        "correction.zoom_out": "Zoom out (-)",
        "correction.zoom_in": "Zoom in (+)",
        "correction.zoom_fit": "Fit to screen",
        "correction.zoom_actual": "Actual size",
        "correction.shortcuts": "S Select · A Add · D Delete · Ctrl+Wheel Zoom · Space Pan · ←→ Change image",
        "correction.previous": "Previous",
        "correction.next": "Next",
        "correction.properties": "Properties",
        "correction.no_selection": "Select a speech bubble to edit",
        "correction.text": "Text content",
        "correction.position": "Position (x1,y1,x2,y2)",
        "correction.clean": "Clean text",
        "correction.reocr": "Run OCR again",
        "correction.continue": "Continue translation and rendering",
        "correction.reset_confirm": "Reset every speech bubble to the original OCR result?",
        "correction.deleted": "Speech bubble deleted",
        "correction.cleaned": "Text cleaned",
        "correction.reset_done": "Reset to the original OCR result",
        "correction.ocr_running": "Running OCR...",
        "correction.ocr_empty": "No text was detected",
        "correction.ocr_error": "OCR failed",
        "result.title": "Translation results ({count} images)",
        "result.translated": "Translated",
        "result.original": "Original",
        "result.compare": "Compare image {name}",
        "result.empty": "No images were processed.",
        "result.download_zip": "Download ZIP",
        "result.back_to_correction": "Back to correction",
        "result.zip_creating": "Creating ZIP...",
        "result.zip_done": "Download complete!",
        "result.zip_error": "Could not create ZIP",
        "warning.gemini_unavailable": "Gemini could not translate the text, so the original text was kept. Check the API key or quota and try again.",
        "warning.local_llm_unavailable": "The local LLM could not translate the text, so the original text was kept. Check the server URL and model and try again.",
        "warning.google_unavailable": "Google Translate failed, so the original text was kept.",
        "warning.unknown_engine": "The translation engine was not recognized, so the original text was kept.",
    },
}

JS_KEYS = tuple(CATALOGS["en"])


def normalize_locale(value):
    candidate = str(value or "").strip().lower().replace("_", "-").split("-", 1)[0]
    return candidate if candidate in SUPPORTED_LOCALES else None


def resolve_locale(explicit=None, cookie=None, accept_language=None):
    return (
        normalize_locale(explicit)
        or normalize_locale(cookie)
        or normalize_locale((accept_language or "").split(",", 1)[0])
        or DEFAULT_LOCALE
    )


def translate(locale, key, **params):
    selected = normalize_locale(locale) or DEFAULT_LOCALE
    template = CATALOGS[selected].get(key, CATALOGS[DEFAULT_LOCALE].get(key, key))
    class SafeParams(dict):
        def __missing__(self, name):
            return "{" + name + "}"
    return template.format_map(SafeParams(params))


def javascript_catalog(locale):
    return {key: translate(locale, key) for key in JS_KEYS}
```

- [ ] **Step 4: Bind the locale to Flask and add the switch endpoint**

In `app.py`, import `g`, `url_for`, and the localization helpers. Add:

```python
@app.before_request
def bind_ui_language():
    g.ui_language = resolve_locale(
        request.values.get("ui_language"),
        request.cookies.get("ui_language"),
        request.headers.get("Accept-Language"),
    )


@app.context_processor
def inject_localization():
    locale = getattr(g, "ui_language", "en")
    return {
        "ui_language": locale,
        "t": lambda key, **params: translate(locale, key, **params),
        "js_i18n": javascript_catalog(locale),
    }


def _safe_local_redirect(value):
    value = str(value or "")
    return value if value.startswith("/") and not value.startswith("//") else url_for("home")


@app.post("/set-language")
def set_language():
    locale = normalize_locale(request.form.get("language")) or "en"
    response = redirect(_safe_local_redirect(request.form.get("next")))
    response.set_cookie("ui_language", locale, max_age=31536000, samesite="Lax")
    return response
```

Change the root template to `<html lang="{{ ui_language }}">` so the route test can pass before the rest of the copy migration.

- [ ] **Step 5: Run focused tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_localization.py -q
git add -- localization.py tests/test_localization.py app.py templates/index.html
git commit -m "feat: add persistent UI locale resolution"
```

Expected: all four tests pass.

---

### Task 2: Stable semantic selection codes

**Files:**
- Create: `selection_options.py`
- Create: `tests/test_selection_options.py`
- Modify: `templates/index.html:20-215`
- Modify: `static/js/app.js:1-237`
- Modify: `app.py:639-699`

**Interfaces:**
- Produces: `normalize_source(value)`, `normalize_target(value)`, `normalize_engine(value)`, and `normalize_style(value)`.
- Produces: custom-select option attribute `data-value` and local-storage values containing codes.
- Consumes: `window.APP_I18N` and `window.t` added in Task 5; until then labels remain server-rendered.

- [ ] **Step 1: Write failing normalization tests**

Create `tests/test_selection_options.py`:

```python
from selection_options import normalize_engine, normalize_source, normalize_style, normalize_target


def test_stable_codes_pass_through():
    assert normalize_source("ja") == "ja"
    assert normalize_target("en") == "en"
    assert normalize_engine("local_llm") == "local_llm"
    assert normalize_style("keep_honorifics") == "keep_honorifics"


def test_known_legacy_labels_are_migrated():
    assert normalize_source("Japanese (Manga)") == "ja"
    assert normalize_target("Vietnamese") == "vi"
    assert normalize_engine("Local LLM") == "local_llm"
    assert normalize_engine("copilot") == "local_llm"
    assert normalize_style("Casual (thân mật)") == "casual"


def test_invalid_values_use_safe_defaults():
    assert normalize_source("xx") == "ja"
    assert normalize_target("xx") == "vi"
    assert normalize_engine("xx") == "gemini"
    assert normalize_style("xx") == "default"
```

- [ ] **Step 2: Run the test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_selection_options.py -q
```

Expected: import failure for `selection_options`.

- [ ] **Step 3: Implement the normalization boundary**

Create `selection_options.py` with these allowed values and legacy aliases:

```python
SOURCE_CODES = {"ja", "zh", "ko", "en"}
TARGET_CODES = {"vi", "en", "zh", "ko", "th", "id", "fr", "de", "es", "ru"}
ENGINE_CODES = {"gemini", "local_llm", "google"}
STYLE_CODES = {"default", "casual", "formal", "keep_honorifics", "web_novel", "action", "literal", "custom"}

LEGACY_SOURCE = {"japanese (manga)": "ja", "chinese (manhua)": "zh", "korean (manhwa)": "ko", "english (comic)": "en"}
LEGACY_TARGET = {"vietnamese": "vi", "english": "en", "chinese": "zh", "korean": "ko", "thai": "th", "indonesian": "id", "french": "fr", "german": "de", "spanish": "es", "russian": "ru"}
LEGACY_ENGINE = {"local llm": "local_llm", "copilot": "local_llm", "gemini": "gemini", "google": "google"}
LEGACY_STYLE = {
    "default": "default", "casual (thân mật)": "casual", "formal (trang trọng)": "formal",
    "keep honorifics (-san, senpai...)": "keep_honorifics", "web novel style": "web_novel",
    "action (ngắn gọn)": "action", "literal (sát nghĩa)": "literal", "custom...": "custom",
}


def _normalize(value, allowed, aliases, default):
    raw = str(value or "").strip()
    return raw if raw in allowed else aliases.get(raw.lower(), default)


def normalize_source(value): return _normalize(value, SOURCE_CODES, LEGACY_SOURCE, "ja")
def normalize_target(value): return _normalize(value, TARGET_CODES, LEGACY_TARGET, "vi")
def normalize_engine(value): return _normalize(value, ENGINE_CODES, LEGACY_ENGINE, "gemini")
def normalize_style(value): return _normalize(value, STYLE_CODES, LEGACY_STYLE, "default")
```

- [ ] **Step 4: Put stable values in the form and browser storage**

Add `data-value` to every `.option`, for example:

```html
<span class="option" data-value="ja">Japanese (Manga)</span>
<span class="option" data-value="vi">Vietnamese</span>
<span class="option" data-value="local_llm">Local LLM</span>
<span class="option" data-value="keep_honorifics">Keep Honorifics (-san, senpai...)</span>
```

Use the code for selection, visibility, persistence, and hidden inputs:

```javascript
const optionValue = option => option.dataset.value || option.textContent.trim();
const selectOption = (selectBox, option) => {
    selectBox.dataset.value = optionValue(option);
    selectBox.querySelector('.selected').textContent = option.textContent;
    selectBox.querySelectorAll('.option').forEach(item => item.classList.toggle('selected', item === option));
};
```

When restoring a saved preference, match `optionValue(option) === savedValue`; if no code matches, match the legacy text once and immediately replace local storage with `optionValue(option)`. Change style/engine visibility checks to `custom`, `local_llm`, and `gemini`. Change `updateHiddenInputs()` to read `document.getElementById(id).dataset.value`, not `.innerText`.

- [ ] **Step 5: Normalize form data in `upload_file()`**

Replace the localized dictionaries in `app.py` with:

```python
selected_translator = normalize_engine(request.form.get("selected_translator"))
source_lang = normalize_source(request.form.get("selected_source_lang"))
target_lang = normalize_target(request.form.get("selected_language"))
style_code = normalize_style(request.form.get("selected_style"))
custom_prompt = request.form.get("custom_prompt", "").strip() if style_code == "custom" else ""
```

Carry `style_code` and `custom_prompt` as separate fields through direct translation and correction-session metadata. Change internal engine branches from `copilot` to `local_llm`, while accepting `copilot` only at the normalization boundary.

- [ ] **Step 6: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_selection_options.py translator/test_translator.py -q
node --check static/js/app.js
git add -- selection_options.py tests/test_selection_options.py app.py templates/index.html static/js/app.js
git commit -m "refactor: use locale-independent selection codes"
```

Expected: tests pass and Node reports no syntax error.

---

### Task 3: Shared bilingual prompt builder

**Files:**
- Create: `translator/prompts.py`
- Create: `tests/test_prompts.py`
- Modify: `translator/base.py:7-61`
- Modify: `translator/gemini_translator.py:61-376`
- Modify: `translator/local_llm_translator.py:16-232`
- Modify: `translator/translator.py:16-96`

**Interfaces:**
- Produces: `build_single_prompt(text, source, target, prompt_locale, style_code="default", custom_instruction="") -> str`.
- Produces: `build_batch_prompt(texts, source, target, prompt_locale, style_code="default", custom_instruction="") -> str`.
- Produces: `build_pages_prompt(pages, source, target, prompt_locale, style_code="default", custom_instruction="") -> str`.
- Adds optional `prompt_locale="en"` to `BaseTranslator`, `GeminiTranslator`, `LocalLLMTranslator`, and `MangaTranslator` constructors.
- Provider translate method signatures remain backward compatible.

- [ ] **Step 1: Write failing prompt contract tests**

Create `tests/test_prompts.py`:

```python
import json

from translator.prompts import build_batch_prompt, build_pages_prompt, build_single_prompt


def test_prompt_locale_is_independent_from_target_language():
    en_prompt = build_single_prompt("xin chào", "vi", "ja", "en")
    vi_prompt = build_single_prompt("hello", "en", "en", "vi")
    assert "Translate" in en_prompt and "Japanese" in en_prompt
    assert "Hãy dịch" in vi_prompt and "tiếng Anh" in vi_prompt


def test_custom_instruction_is_preserved_verbatim():
    instruction = "Keep   NAME\nunchanged: 山田"
    prompt = build_batch_prompt(["text"], "en", "vi", "en", custom_instruction=instruction)
    assert instruction in prompt
    assert "<custom-instruction>" in prompt
    assert "</custom-instruction>" in prompt


def test_style_preset_uses_prompt_locale():
    assert "formal, respectful language" in build_single_prompt("x", "ja", "en", "en", style_code="formal")
    assert "trang trọng, lịch sự" in build_single_prompt("x", "ja", "en", "vi", style_code="formal")


def test_batch_and_pages_payloads_are_valid_json_inside_prompt():
    batch = ["a", "b"]
    pages = {"page-1": ["a"], "page-2": ["b"]}
    assert json.dumps(batch, ensure_ascii=False) in build_batch_prompt(batch, "ja", "en", "en")
    assert json.dumps(pages, ensure_ascii=False, indent=2) in build_pages_prompt(pages, "ja", "vi", "vi")


def test_unknown_prompt_locale_falls_back_to_english():
    assert "Return only" in build_single_prompt("x", "ja", "vi", "fr")
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_prompts.py -q
```

Expected: import failure for `translator.prompts`.

- [ ] **Step 3: Implement the three prompt builders**

Create `translator/prompts.py`. Use localized language names and one delimiter helper:

```python
import json

LANGUAGE_NAMES = {
    "en": {"ja": "Japanese", "zh": "Chinese", "ko": "Korean", "en": "English", "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "fr": "French", "de": "German", "es": "Spanish", "ru": "Russian"},
    "vi": {"ja": "tiếng Nhật", "zh": "tiếng Trung", "ko": "tiếng Hàn", "en": "tiếng Anh", "vi": "tiếng Việt", "th": "tiếng Thái", "id": "tiếng Indonesia", "fr": "tiếng Pháp", "de": "tiếng Đức", "es": "tiếng Tây Ban Nha", "ru": "tiếng Nga"},
}

STYLE_GUIDANCE = {
    "en": {
        "default": "", "formal": "Use formal, respectful language.",
        "casual": "Use casual, natural everyday language.",
        "keep_honorifics": "Keep honorifics such as -san, -kun, senpai, and sensei untranslated.",
        "web_novel": "Use a dramatic web-novel voice while keeping dialogue concise.",
        "action": "Use short, punchy dialogue with quick pacing.",
        "literal": "Preserve the meaning closely while keeping spoken dialogue natural.",
    },
    "vi": {
        "default": "", "formal": "Dùng ngôn ngữ trang trọng, lịch sự.",
        "casual": "Dùng lời thoại thân mật, tự nhiên hằng ngày.",
        "keep_honorifics": "Giữ nguyên kính ngữ như -san, -kun, senpai và sensei.",
        "web_novel": "Dùng giọng văn web novel giàu cảm xúc nhưng lời thoại vẫn gọn.",
        "action": "Dùng câu thoại ngắn, mạnh và nhịp nhanh.",
        "literal": "Bám sát ý nghĩa nhưng vẫn giữ lời thoại tự nhiên.",
    },
}


def _locale(value):
    return "vi" if value == "vi" else "en"


def _name(code, locale):
    return LANGUAGE_NAMES[locale].get(code, code)


def _instructions(style_code, custom_instruction, locale):
    if custom_instruction:
        heading = "Additional instruction" if locale == "en" else "Yêu cầu bổ sung"
        return f"\n{heading}:\n<custom-instruction>\n{custom_instruction}\n</custom-instruction>"
    guidance = STYLE_GUIDANCE[locale].get(style_code, "")
    if not guidance:
        return ""
    heading = "Style instruction" if locale == "en" else "Hướng dẫn phong cách"
    return f"\n{heading}: {guidance}"


def build_single_prompt(text, source, target, prompt_locale, style_code="default", custom_instruction=""):
    locale = _locale(prompt_locale)
    source_name, target_name = _name(source, locale), _name(target, locale)
    if locale == "vi":
        return f"""Bạn là chuyên gia dịch manga/comic. Hãy dịch hội thoại từ {source_name} sang {target_name}.

Quy tắc:
- Viết như hội thoại tự nhiên khi đọc thành tiếng.
- Giữ giọng điệu, cảm xúc và tính cách nhân vật.
- Không dịch từng từ; dùng cấu trúc tự nhiên trong {target_name}.
- Chỉ trả về bản dịch, không giải thích hoặc định dạng.{_instructions(style_code, custom_instruction, locale)}

<source-text>
{text}
</source-text>"""
    return f"""You are an expert manga/comic translator. Translate spoken dialogue from {source_name} to {target_name}.

Rules:
- Write natural dialogue that sounds right aloud.
- Preserve tone, emotion, and character personality.
- Do not translate word by word; use natural {target_name} sentence structure.
- Return only the translation, without explanation or formatting.{_instructions(style_code, custom_instruction, locale)}

<source-text>
{text}
</source-text>"""


def build_batch_prompt(texts, source, target, prompt_locale, style_code="default", custom_instruction=""):
    locale = _locale(prompt_locale)
    source_name, target_name = _name(source, locale), _name(target, locale)
    payload = json.dumps(texts, ensure_ascii=False)
    if locale == "vi":
        return f"""Hãy dịch mảng JSON chứa hội thoại manga/comic từ {source_name} sang {target_name}. Giữ đúng thứ tự, giọng nhân vật và câu thoại tự nhiên. Chỉ trả về một mảng JSON có cùng số phần tử.{_instructions(style_code, custom_instruction, locale)}

<source-json>
{payload}
</source-json>"""
    return f"""Translate this JSON array of manga/comic dialogue from {source_name} to {target_name}. Preserve order, character voice, and natural spoken dialogue. Return only one JSON array with the same number of items.{_instructions(style_code, custom_instruction, locale)}

<source-json>
{payload}
</source-json>"""


def build_pages_prompt(pages, source, target, prompt_locale, style_code="default", custom_instruction=""):
    locale = _locale(prompt_locale)
    source_name, target_name = _name(source, locale), _name(target, locale)
    payload = json.dumps(pages, ensure_ascii=False, indent=2)
    if locale == "vi":
        return f"""Hãy dịch các trang manga/comic liên tiếp từ {source_name} sang {target_name}. Giữ nhất quán mạch truyện, giọng nhân vật, tên trang và thứ tự bóng thoại. Chỉ trả về JSON object có cùng cấu trúc.{_instructions(style_code, custom_instruction, locale)}

<source-pages-json>
{payload}
</source-pages-json>"""
    return f"""Translate these consecutive manga/comic pages from {source_name} to {target_name}. Keep story context, character voices, page names, and bubble order consistent. Return only a JSON object with the same structure.{_instructions(style_code, custom_instruction, locale)}

<source-pages-json>
{payload}
</source-pages-json>"""
```

- [ ] **Step 4: Route all Gemini and local-LLM prompt creation through the builder**

Add `prompt_locale="en"` to constructors and store `self.prompt_locale`. Replace inline prompt literals as follows:

```python
# Gemini methods, which expose a per-call custom_prompt argument:
custom_instruction = custom_prompt if custom_prompt is not None else self.custom_prompt
prompt = build_single_prompt(text, source, target, self.prompt_locale, self.style_code, custom_instruction)
prompt = build_batch_prompt(texts_to_translate, source, target, self.prompt_locale, self.style_code, custom_instruction)
prompt = build_pages_prompt(pages_texts, source, target, self.prompt_locale, self.style_code, custom_instruction)

# Local LLM methods:
prompt = build_single_prompt(text, source, target, self.prompt_locale, self.style_code, self.custom_prompt)
prompt = build_batch_prompt(texts_to_translate, source, target, self.prompt_locale, self.style_code, self.custom_prompt)
```

In `BaseTranslator`, replace preset expansion with:

```python
self.style_code = style if style in STYLE_PRESETS else "default"
self.custom_prompt = custom_prompt or ""
self.prompt_locale = "vi" if prompt_locale == "vi" else "en"
```

Keep `self.custom_prompt` verbatim and remove English-only `_build_style_instructions()` from provider prompt assembly. Pass `style=style_code`, `custom_prompt=custom_prompt`, and `prompt_locale` through `MangaTranslator` when it lazily constructs `GeminiTranslator`; do the same for `LocalLLMTranslator` in `app.py`.

- [ ] **Step 5: Add provider-level prompt capture tests**

Extend `tests/test_prompts.py`:

```python
from types import SimpleNamespace

from translator.gemini_translator import GeminiTranslator
from translator.local_llm_translator import LocalLLMTranslator


def test_gemini_submits_vietnamese_prompt_and_verbatim_custom_instruction():
    captured = []
    class Models:
        def generate_content(self, model, contents):
            captured.append(contents)
            return SimpleNamespace(text="bản dịch")
    translator = GeminiTranslator(
        api_key="test", prompt_locale="vi", custom_prompt="Keep  山田\nexact",
        client_factory=lambda key: SimpleNamespace(models=Models()),
    )
    assert translator.translate_single("hello", source="en", target="vi") == "bản dịch"
    assert "Hãy dịch" in captured[0]
    assert "Keep  山田\nexact" in captured[0]


def test_local_llm_submits_english_prompt_for_japanese_target(monkeypatch):
    captured = []
    translator = LocalLLMTranslator(prompt_locale="en", style="formal")
    monkeypatch.setattr(translator, "_post_chat", lambda prompt, timeout: captured.append(prompt) or "翻訳")
    assert translator.translate_single("hello", source="en", target="ja") == "翻訳"
    assert "Translate" in captured[0]
    assert "Japanese" in captured[0]
    assert "formal, respectful language" in captured[0]
```

- [ ] **Step 6: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_prompts.py translator/test_translator.py -q
git add -- translator/prompts.py tests/test_prompts.py translator/base.py translator/gemini_translator.py translator/local_llm_translator.py translator/translator.py app.py
git commit -m "feat: build LLM prompts in the UI language"
```

Expected: prompt and maintained translator tests pass.

---

### Task 4: Localized server validation, progress, and warnings

**Files:**
- Create: `tests/test_localized_routes.py`
- Modify: `app.py:285-294,476-632,639-1030`

**Interfaces:**
- Changes: `emit_progress(phase, current, total, message_key, **message_params) -> None`.
- Produces: socket payload keys `phase`, `current`, `total`, `percent`, `message_key`, `message_params`.
- Consumes: `g.ui_language`, `translate()`, and provider `prompt_locale` from prior tasks.
- Stores: `ui_language` in correction session/job metadata.

- [ ] **Step 1: Write failing progress and localized validation tests**

Create `tests/test_localized_routes.py`:

```python
def test_progress_event_contains_semantic_message(monkeypatch):
    import app as app_module

    captured = []
    monkeypatch.setattr(app_module.socketio, "emit", lambda event, payload: captured.append((event, payload)))
    app_module.emit_progress("ocr", 1, 4, "progress.ocr_done", count=8)
    assert captured == [("progress", {
        "phase": "ocr", "current": 1, "total": 4, "percent": 25,
        "message_key": "progress.ocr_done", "message_params": {"count": 8},
    })]


def test_missing_files_error_follows_ui_locale():
    import app as app_module

    client = app_module.app.test_client()
    response = client.post("/translate", data={
        "ui_language": "en", "selected_translator": "google",
        "selected_font": "animeace", "selected_source_lang": "ja",
        "selected_language": "vi", "selected_style": "default",
    })
    assert "Select at least one image." in response.get_data(as_text=True)
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_localized_routes.py -q
```

Expected: the progress payload still has `message`, and the route renders Vietnamese.

- [ ] **Step 3: Emit keys and localize server-rendered messages**

Implement:

```python
def emit_progress(phase, current, total, message_key, **message_params):
    try:
        socketio.emit("progress", {
            "phase": phase,
            "current": current,
            "total": total,
            "message_key": message_key,
            "message_params": message_params,
            "percent": int((current / max(total, 1)) * 100),
        })
    except Exception:
        pass
```

Replace every call with one of the exact catalog keys from Task 1. Render validation errors with `translate(g.ui_language, key, **params)`. Store warning keys (`warning.gemini_unavailable`, `warning.local_llm_unavailable`, `warning.google_unavailable`, `warning.unknown_engine`) and translate only while rendering the template.

- [ ] **Step 4: Propagate prompt locale through the pipeline**

Add `prompt_locale`, `style_code`, and `custom_prompt` to `_do_full_pipeline(...)` and `translate_and_render(...)`. Construct:

```python
translator_obj = MangaTranslator(source=source_lang, target=target_lang, prompt_locale=prompt_locale)
GeminiTranslator(..., prompt_locale=prompt_locale, style=style_code, custom_prompt=custom_prompt)
LocalLLMTranslator(..., prompt_locale=prompt_locale, style=style_code, custom_prompt=custom_prompt)
```

Save `'ui_language': g.ui_language` in manual-correction metadata and pass `session_data.get('ui_language', g.ui_language)` after correction.

- [ ] **Step 5: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_localized_routes.py tests/test_prompts.py translator/test_translator.py -q
git add -- app.py tests/test_localized_routes.py
git commit -m "feat: localize server progress and failures"
```

Expected: route tests pass and no provider prompt loses its locale.

---

### Task 5: Translate all three screens and dynamic JavaScript copy

**Files:**
- Create: `templates/_language_switch.html`
- Create: `static/js/i18n.js`
- Create: `tests/test_i18n_templates.py`
- Modify: `templates/index.html`
- Modify: `templates/correction.html`
- Modify: `templates/translate.html`
- Modify: `static/js/app.js`
- Modify: `static/js/correction.js`
- Modify: `static/css/style.css`
- Modify: `static/css/correction.css`

**Interfaces:**
- Consumes: template globals `ui_language`, `t`, and `js_i18n`.
- Produces: `window.APP_I18N` and `window.t(key, params)`.
- Produces: accessible `VI | EN` switch on upload, correction, and result screens.

- [ ] **Step 1: Write failing template coverage tests**

Create `tests/test_i18n_templates.py`:

```python
import re


def test_all_pages_render_active_language_and_switch(monkeypatch):
    import app as app_module

    client = app_module.app.test_client()
    home = client.get("/", headers={"Accept-Language": "en-US"}).get_data(as_text=True)
    assert '<html lang="en">' in home
    assert 'aria-label="Language"' in home
    assert 'data-value="local_llm"' in home


def test_frontend_sources_do_not_contain_dynamic_vietnamese_literals():
    sources = [
        open("static/js/app.js", encoding="utf-8").read(),
        open("static/js/correction.js", encoding="utf-8").read(),
    ]
    forbidden = ("Vui lòng", "Đang OCR", "Đã xoá", "Chọn bóng thoại", "Không nhận được")
    assert not any(text in source for source in sources for text in forbidden)


def test_templates_bind_html_language_dynamically():
    for path in ("templates/index.html", "templates/correction.html", "templates/translate.html"):
        source = open(path, encoding="utf-8").read()
        assert '<html lang="{{ ui_language }}">' in source
        assert "_language_switch.html" in source
        assert "js_i18n | tojson" in source
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_i18n_templates.py -q
```

Expected: correction/result HTML uses fixed languages and dynamic JavaScript still contains Vietnamese strings.

- [ ] **Step 3: Add the shared language switch and browser translator**

Create `templates/_language_switch.html`:

```html
<form class="language-switch" action="{{ url_for('set_language') }}" method="post" aria-label="{{ t('common.language') }}">
  <input type="hidden" name="next" value="{{ request.full_path if request.query_string else request.path }}">
  <button name="language" value="vi" aria-pressed="{{ 'true' if ui_language == 'vi' else 'false' }}">VI</button>
  <span aria-hidden="true">|</span>
  <button name="language" value="en" aria-pressed="{{ 'true' if ui_language == 'en' else 'false' }}">EN</button>
</form>
```

Create `static/js/i18n.js`:

```javascript
window.t = function (key, params = {}) {
    let value = (window.APP_I18N && window.APP_I18N[key]) || key;
    Object.entries(params).forEach(([name, replacement]) => {
        value = value.replaceAll(`{${name}}`, String(replacement));
    });
    return value;
};
```

Before page-specific JavaScript on each page, add:

```html
<script>window.APP_I18N = {{ js_i18n | tojson }};</script>
<script src="{{ url_for('static', filename='js/i18n.js') }}"></script>
```

- [ ] **Step 4: Replace visible template literals with catalog calls**

Set all three `<html lang>` attributes dynamically, include `_language_switch.html`, and replace every user-facing label from the Task 1 catalog with `{{ t('key') }}`. Add `<input type="hidden" name="ui_language" value="{{ ui_language }}">` to upload and correction forms. Keep product name, provider/model names, font filenames, keyboard keys, and numeric coordinates untranslated.

- [ ] **Step 5: Replace dynamic JavaScript literals with translation keys**

Use calls such as:

```javascript
fileText.textContent = t('index.selected_images', { count: files.length });
alert(t('validation.gemini_keys_required'));
blockProperties.innerHTML = `<p class="no-sel">${escapeHtml(t('correction.no_selection'))}</p>`;
showToast(t('correction.deleted'));
progressText.textContent = t(data.message_key, data.message_params || {});
progressPhase.textContent = t(`progress.phase.${data.phase}`);
```

For editor HTML, escape translated strings before interpolation. Use translation keys for reset confirmation, OCR states, thumbnail block counts, previous/next labels, and result comparison/download controls.

- [ ] **Step 6: Style the switch for both page layouts**

Add a compact inline-flex switch with a 44px minimum button hit area, visible focus outline, and `[aria-pressed="true"]` state. Place it in the existing top area without absolute positioning that can overlap long English headings. Add the equivalent rules to `correction.css` because the correction screen does not load `style.css`.

- [ ] **Step 7: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_localization.py tests/test_i18n_templates.py tests/test_localized_routes.py -q
node --check static/js/i18n.js
node --check static/js/app.js
node --check static/js/correction.js
git add -- templates/_language_switch.html static/js/i18n.js tests/test_i18n_templates.py templates/index.html templates/correction.html templates/translate.html static/js/app.js static/js/correction.js static/css/style.css static/css/correction.css
git commit -m "feat: translate the interface into Vietnamese and English"
```

Expected: tests pass and all three Node checks are silent.

---

### Task 6: Full i18n regression and responsive UI verification

**Files:**
- Modify only if a failure proves necessary: files changed in Tasks 1-5

**Interfaces:**
- Verifies: all public locale, selection, prompt, progress, and template contracts.

- [ ] **Step 1: Run all Python and JavaScript checks**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
node --check static/js/i18n.js
node --check static/js/app.js
node --check static/js/correction.js
git diff --check
```

Expected: all tests pass, JavaScript syntax checks are silent, and `git diff --check` reports nothing.

- [ ] **Step 2: Run the Flask smoke matrix**

Use the Flask test client to request `/` with `Accept-Language: vi-VN` and `en-US`, then with an overriding cookie. Submit stable source/target/engine/style codes with mocked OCR/translators. Confirm English UI + Japanese target produces an English prompt naming Japanese, and Vietnamese UI + English target produces a Vietnamese prompt naming tiếng Anh.

- [ ] **Step 3: Inspect desktop and narrow layouts**

Start the app with:

```powershell
.\run_app.ps1
```

At widths 1440px, 768px, and 390px, inspect upload, correction, and result screens in both locales. Confirm the switch is keyboard reachable, focus is visible, English text does not clip, and switching locale preserves the current local page path.

- [ ] **Step 4: Run the Impeccable frontend detector once**

```powershell
node C:\Users\dun\.agents\skills\impeccable\scripts\detect.mjs --json templates/index.html templates/correction.html templates/translate.html static/js/app.js static/js/correction.js static/js/i18n.js static/css/style.css static/css/correction.css
```

Expected: no unresolved high-confidence UI findings. Fix a reported issue only when it applies to these changed targets, then repeat Steps 1 and 4.

- [ ] **Step 5: Commit any verification fixes**

```powershell
git add -- localization.py selection_options.py translator/prompts.py translator/base.py translator/gemini_translator.py translator/local_llm_translator.py translator/translator.py app.py templates static/js static/css tests
git commit -m "fix: complete bilingual UI verification"
```

Skip this commit when verification required no changes.
