# Đặc tả i18n — Manga-Translator (vi/en)

> **Task:** t1 — i18n Spec. **Tác giả:** requirements-i18n (product-sprint-prioritizer).
> **Người tiêu thụ:** frontend-i18n (t2), backend-i18n (t3), verifier-i18n (t4), reviewer-i18n (t5/t6).
> **Trạng thái:** BẢN CHỐT cho vòng implement. Mọi thay đổi lớn phải được captain phê duyệt.

---

## 1. Mục tiêu

Đưa toàn bộ text cứng tiếng Việt của giao diện Manga-Translator vào kiến trúc i18n
2 ngôn ngữ **vi** (mặc định, giá trị = y hệt text hiện tại — **zero visual regression**) và **en**,
kèm:

1. **Auto-detect** ngôn ngữ trình duyệt ở lần truy cập đầu (navigator.language), lưu lựa chọn.
2. **Dropdown chuyển đổi** ngôn ngữ trên mọi trang + **localStorage** (lựa chọn người dùng thắng auto-detect).
3. **Fallback** an toàn: locale thiếu key → vi → raw key (không bao giờ crash).
4. Kiến trúc mở rộng được: thêm locale mới = thêm 1 file JSON + 1 option dropdown.
5. Giữ nguyên nhận diện thương hiệu **#5E1675 / Exo 2**.

---

## 2. Khảo sát hiện trạng (đã verify trên mã nguồn)

| File | Dung lượng | Dòng có text Việt (diacritics) | Ghi chú |
|---|---|---|---|
| templates/index.html | 272 dòng | 22 dòng (~33 chuỗi hiển thị) | labels, options, placeholders, hints, loading, phaseNames (inline script) |
| templates/translate.html | 177 dòng | 12 dòng (~16 chuỗi) | title, tabs, buttons, ZIP status (inline script) |
| templates/correction.html | 133 dòng | 37 dòng (~48 chuỗi) | toolbar, hints, badges, footer, aria-labels |
| static/js/app.js | 254 dòng | 6 chuỗi | alerts + file-list labels |
| static/js/correction.js | 2718 dòng | 76 dòng (~85 chuỗi) | toasts (22), hints (6), ocr status (4), shortcuts (3), style panel (~22), block editor (~14), aria/canvas (5), confirm (1) |
| app.py | 2414 dòng | 30 dòng (~24 chuỗi backend) | progress events, errors, warnings |
| translator/gemini_translator.py | — | 1 warning user-visible (dòng 152) + prompt LLM (ngoài phạm vi) | last_warning |
| static/css/*.css | — | 0 | chỉ content:"A" (swatch) — không cần i18n |
| static/qa/*, tests/, temp_sessions/, debug_outputs/, docs/ | — | có | **ngoài phạm vi** (fixture/test) |

**Ước lượng key dictionary:** ~160 key duy nhất (sau khi dedup shared key giữa template ↔ JS ↔ backend).

**Phát hiện quan trọng:**
- `app.js` alert text và `app.py` error text trùng ý nghĩa (API key, model, ảnh) → **dùng chung key** nơi giống hệt, tách key nơi khác dấu câu ("!" vs ".").
- `gemini_translator.py:152` set `last_warning` tiếng Việt; dòng 288 check `"tất cả gemini api key" in error_str.lower()` → phải xử lý khi chuyển structured (mục 5.7).
- Progress message đi qua Socket.IO `emit_progress(phase, current, total, message)` (app.py:358) → client index.html chỉ hiển thị `data.message` → **backend localize message ngay tại emit** (server-side t()).
- Options dropdown (source/target/translator/style) hiện gửi **display text** lên backend, backend map lowercase display→value (app.py:1035-1084) → nếu dịch display text thì phải có **data-value refactor** (mục 5.6), không được phép dịch trôi nổi.
- Font names, tên ngôn ngữ đích ("Vietnamese", "English"...), tên translator ("Gemini", "Local LLM"), tên font ("Animeace", "Yuki-*") là **proper nouns / giá trị**, không dịch.

---

## 3. Kiến trúc

### 3.1 Locales & dictionary files

- Locales: `["vi", "en"]`, **default = "vi"**.
- Dictionary: **single source of truth** tại:
  - `i18n/vi.json` — giá trị = y hệt chuỗi hiện tại (chép nguyên văn).
  - `i18n/en.json` — bản dịch tiếng Anh.
- Cả Python (Flask) và JS (trình duyệt) **cùng đọc 2 file này**:
  - Python: `json.load` + cache (mục 3.3).
  - JS: nhúng cả 2 dict inline vào mỗi trang qua `<script id="i18n-data" type="application/json">` (mục 3.4). Nhúng cả 2 locale (~30-40KB) để hỗ trợ live-switch (P2) và tránh fetch bất đồng bộ / FOUC.
- **Quy tắc bất biến:** key set của vi.json và en.json **phải giống hệt nhau** (script check, mục 8).

### 3.2 Key naming & placeholder convention

- Key phẳng, namespace dấu chấm:
  - `common.*` — dùng chung mọi trang (vd `common.back`).
  - `index.*` — trang chủ + app.js.
  - `results.*` — translate.html.
  - `corr.*` — correction.html + correction.js.
  - `backend.*` — app.py + gemini_translator.py.
- Tham số: `{name}`, `{n}`, `{i}`, `{formats}`, `{text}`, `{error}`, `{c}`, `{size}`, `{message}`.
- **HTML trong value:** chỉ key kết thúc bằng `_html` được chứa HTML (vd `corr.shortcuts.preview_html`, `corr.editor.noSelectionStyled_html`); mọi key khác phải là text thuần (render qua textContent / autoescape).
- **Plural:** cặp key `<base>_one` / `<base>_other`; helper `tp(base, n)` chọn theo plural rule của locale (mục 5.2).

### 3.3 Python layer — `i18n/__init__.py` (backend-i18n t3)

```python
SUPPORTED_LOCALES = ["vi", "en"]
DEFAULT_LOCALE = "vi"

# cache dicts: functools.lru_cache -> json.load(i18n/<loc>.json)
def load_dict(locale): ...
def resolve_locale(cookie_val=None, accept_lang=""): ...
    # cookie hợp lệ -> dùng; nếu không: parse Accept-Language tag đầu tiên,
    # prefix "vi" -> vi, "en" -> en, else DEFAULT_LOCALE
def t(key, **params): ...
    # chain: dict[locale] -> dict[vi] -> raw key. _html key -> Markup (Jinja safe).
def tp(key_base, n, **params): ...
    # chọn <base>_one/_other theo plural rule; nếu thiếu _one -> fallback _other
def i18n_json(): ...  # json.dumps({vi:..., en:...}, ensure_ascii=False).replace("</", "<\\/")
```

- `before_request`: `g.locale = resolve_locale(request.cookies.get("mt_locale"), request.headers.get("Accept-Language", ""))`.
- `teardown_request`: clear state.
- **Context processor** (app.py): inject `t`, `tp`, `current_locale`, `i18n_json` vào mọi template.
- `emit_progress` không đổi signature; **call sites** gọi `t('backend.progress.*', ...)` → message localize sẵn theo locale của request.
- `t()` phải an toàn khi dict file chưa tồn tại (fallback → raw key) để t2/t3 phát triển song song.

### 3.4 JS layer — `static/js/i18n.js` (frontend-i18n t2)

IIFE expose `window.I18N`:

```js
I18N = {
  locale: 'vi',
  dicts: { vi: {...}, en: {...} },   // parse từ <script id="i18n-data">
  t(key, params),                     // chain locale -> vi -> raw key; escape params (trừ _html)
  tp(base, n, params),                // plural (Intl.PluralRules cho en; vi luôn 'other')
  init(),                             // detect + dropdown + persist (mục 3.6)
  refreshCallbacks: [], onRefresh(fn) // P2: live-switch re-render hooks
}
```

- Load trong `<head>` (defer không được — cần chạy trước paint cho logic reload lần đầu).
- Nếu thiếu `#i18n-data` (vd QA harness): dict rỗng → `t` trả raw key; trang vẫn chạy (harness ngoài phạm vi).
- Escape: param luôn escapeHtml; value có `_html` được phép chứa HTML cố định (không param).

### 3.5 Jinja layer — templates

- Mọi text tĩnh: `{{ t('key') }}`, có tham số: `{{ t('key', n=images|length) }}`.
- Key `_html`: `{{ t('key') }}` trả Markup → không cần `| safe`.
- `<html lang="{{ current_locale }}">` (3 template).
- Head mỗi template thêm: `<script id="i18n-data" type="application/json">{{ i18n_json | safe }}</script>` + `<script src=".../js/i18n.js"></script>`.

### 3.6 Locale resolution & auto-detect

**Thứ tự ưu tiên (client):**
1. `localStorage['mt_locale']` — lựa chọn người dùng (thắng tất cả).
2. Lần đầu (chưa có localStorage): `navigator.language` prefix — `vi*` → vi, `en*` → en, khác → vi (default).
3. Server: cookie `mt_locale` nếu có, nếu không thì Accept-Language header.

**Luồng lần truy cập đầu (I18N.init):**
1. Đọc localStorage; nếu có → dùng, đồng bộ cookie (nếu thiếu cookie → set + reload 1 lần để server render đúng locale; guard `sessionStorage['mt_i18n_reloaded']` chống loop).
2. Chưa có → detect navigator.language:
   - Kết quả **khác** locale server đã render (lưu trong `I18N.serverLocale` từ dict hiện tại / data attribute) → set localStorage + cookie → `location.reload()` (tối đa 1 lần, có guard).
   - Kết quả **bằng** → chỉ set localStorage, không reload.
3. Cookie: `mt_locale=<loc>; path=/; max-age=31536000` (1 năm).

**Dropdown:** `<select id="locale-switch" class="locale-switch">` (vi: "Tiếng Việt", en: "English"), i18n.js tự inject vào `header` (index/translate) và `.corr-topbar` (correction) nếu chưa tồn tại. onChange → localStorage + cookie → `location.reload()`. P2: live-switch không reload (mục 6, P2-1). Style: Exo 2, viền/border 1px #5E1675, text #5E1675, nền trắng, border-radius 8px, padding 4px 8px, aria-label "Ngôn ngữ/Language".

---

## 4. Phạm vi bao phủ chi tiết (key inventory — implementer bám theo đây)

> vi = chép nguyên văn chuỗi hiện tại. en = bản dịch. Key có `_html` là HTML; `_one/_other` là cặp plural.

### 4.1 common
| Key | vi | en |
|---|---|---|
| common.back | ← Quay lại | ← Back |

### 4.2 index (index.html + app.js)
| Key | vi | en |
|---|---|---|
| index.label.sourceLang | Ngôn ngữ gốc | Source language |
| index.label.targetLang | Dịch sang | Translate to |
| index.label.translator | Translator | Translator |
| index.label.style | Phong cách dịch | Translation style |
| index.label.font | Font | Font |
| index.label.manualCorrection | Chỉnh sửa thủ công | Manual correction |
| index.hint.manualCorrection | Dừng sau OCR để thêm/xoá bóng thoại | Stop after OCR to add/remove speech bubbles |
| index.label.customPrompt | Custom Prompt | Custom prompt |
| index.placeholder.customPrompt | Ví dụ: Dịch theo phong cách light novel, giữ nguyên tên nhân vật... | e.g. Translate in light-novel style, keep character names... |
| index.label.serverUrl | Server URL | Server URL |
| index.hint.serverPorts | Ollama: 11434 | LM Studio: 1234 | LocalAI: 8080 | Ollama: 11434 | LM Studio: 1234 | LocalAI: 8080 |
| index.label.modelName | Model Name | Model name |
| index.hint.copilotModel | Nhập tên model đang chạy trên server | Enter the model name running on the server |
| index.hint.geminiModel | Nhập tên model Gemini muốn dùng cho bản dịch. | Enter the Gemini model name to use for translation. |
| index.label.geminiApiKeys | Gemini API Keys | Gemini API keys |
| index.placeholder.geminiKeys | Dán mỗi API key một dòng. App sẽ tự đổi sang key tiếp theo khi key trước lỗi quota/auth. | Paste one API key per line. The app switches to the next key when the previous one hits quota/auth errors. |
| index.hint.geminiKeysStorage | 🔒 Keys được lưu trong trình duyệt của bạn (localStorage), có thể phân tách bằng xuống dòng, dấu phẩy hoặc dấu chấm phẩy. | 🔒 Keys are stored in your browser (localStorage) and can be separated by newlines, commas, or semicolons. |
| index.fileLabel.choose | 📁 Chọn ảnh | 📁 Choose images |
| index.fileLabel.chooseMany | 📁 Chọn ảnh (có thể chọn nhiều) | 📁 Choose images (multiple allowed) |
| index.fileLabel.chosen | 📁 {n} ảnh đã chọn | 📁 {n} images selected |
| index.fileLabel.more | ... và {n} ảnh khác | ... and {n} more images |
| index.submit | Translate | Translate |
| index.loading | Đang xử lý... Vui lòng đợi! | Processing... Please wait! |
| index.phase.ocr | 📖 OCR nhận dạng text | 📖 OCR text recognition |
| index.phase.translation | 🌐 Dịch văn bản | 🌐 Translating text |
| index.phase.rendering | ✏️ Render text vào ảnh | ✏️ Rendering text onto images |
| index.phase.done | ✅ Hoàn tất | ✅ Done |
| index.phase.preparing | ⏳ Chuẩn bị | ⏳ Preparing |
| index.phase.init | Khởi tạo... | Initializing... |
| index.option.styleDefault | Default | Default |
| index.option.styleCasual | Casual (thân mật) | Casual (informal) |
| index.option.styleFormal | Formal (trang trọng) | Formal |
| index.option.styleKeepHonorifics | Keep Honorifics (-san, senpai...) | Keep Honorifics (-san, senpai...) |
| index.option.styleWebNovel | Web Novel Style | Web Novel Style |
| index.option.styleAction | Action (ngắn gọn) | Action (concise) |
| index.option.styleLiteral | Literal (sát nghĩa) | Literal (faithful) |
| index.option.styleCustom | Custom... | Custom... |
| index.error.noApiKey | Vui lòng nhập ít nhất 1 Gemini API Key! | Please enter at least 1 Gemini API key! |
| index.error.noModel | Vui lòng nhập tên model Gemini! | Please enter the Gemini model name! |
| index.error.noImages | Vui lòng chọn ít nhất 1 ảnh! | Please select at least 1 image! |

**Không dịch (giữ nguyên, không cần key):** tên font (Animeace...Yuki-*), tên ngôn ngữ nguồn/đích (Japanese (Manga)...Russian), "Gemini"/"Local LLM"/"Google", URL placeholder mẫu, brand "Manga Translator" (title/alt/logo).

### 4.3 results (translate.html)
| Key | vi | en |
|---|---|---|
| results.pageTitle | Manga Translator - Results | Manga Translator - Results |
| results.title | ✨ Kết quả dịch ({n} ảnh) | ✨ Translation results ({n} images) |
| results.editNote | Ảnh đã chỉnh sửa sau render sẽ tự cập nhật tại đây. | Images edited after rendering will update here automatically. |
| results.compareAria | So sánh ảnh {name} | Compare image {name} |
| results.tab.translated | Đã dịch | Translated |
| results.tab.original | Gốc | Original |
| results.edit | ✏️ Chỉnh sửa | ✏️ Edit |
| results.download | 💾 Download | 💾 Download |
| results.noImages | Không có ảnh nào được xử lý. | No images were processed. |
| results.downloadZip | 📦 Download ZIP | 📦 Download ZIP |
| results.backToOcr | ✏️ Quay lại chỉnh OCR | ✏️ Back to OCR editing |
| results.zipping | ⏳ Đang tạo ZIP... | ⏳ Creating ZIP... |
| results.zipDone | ✅ Đã tải xong! | ✅ Download complete! |
| results.zipError | ❌ Lỗi tạo ZIP | ❌ ZIP creation failed |

### 4.4 corr (correction.html + correction.js)
| Key | vi | en |
|---|---|---|
| corr.pageTitle | Chỉnh sửa thủ công - Manga Translator | Manual correction - Manga Translator |
| corr.title.postrender | ✏️ Chỉnh sửa sau dịch | ✏️ Post-translation edit |
| corr.title.styleditor | 🎨 Chỉnh sửa & Style | 🎨 Edit & Style |
| corr.title.preview | ✏️ Chỉnh sửa bóng thoại | ✏️ Edit speech bubbles |
| corr.stats.bubbles_one | {n} bóng thoại | {n} speech bubble |
| corr.stats.bubbles_other | {n} bóng thoại | {n} speech bubbles |
| corr.stats.images_one | {n} ảnh | {n} image |
| corr.stats.images_other | {n} ảnh | {n} images |
| corr.toolbar.toolsAria | Công cụ | Tools |
| corr.tool.select | 🖱️ Chọn | 🖱️ Select |
| corr.tool.add | ＋ Thêm | ＋ Add |
| corr.tool.delete | 🗑️ Xoá | 🗑️ Delete |
| corr.tool.undo | ↩ Undo | ↩ Undo |
| corr.tool.redo | ↪ Redo | ↪ Redo |
| corr.tool.reset | ↺ Reset | ↺ Reset |
| corr.tool.rect | ▭ Rect | ▭ Rect |
| corr.tool.brush | 🖌 Cọ | 🖌 Brush |
| corr.tool.brushSize | Cỡ cọ | Brush size |
| corr.tool.brushSizeAria | Cỡ cọ xoá nền | Background erase brush size |
| corr.tool.eraseRectTitle | Xoá vùng hình chữ nhật (phím E) | Erase rectangular area (E key) |
| corr.tool.eraseBrushTitle | Cọ xoá nền (phím E) | Background erase brush (E key) |
| corr.zoom.aria | Điều khiển zoom | Zoom controls |
| corr.zoom.outTitle | Thu nhỏ (-) | Zoom out (-) |
| corr.zoom.inTitle | Phóng to (+) | Zoom in (+) |
| corr.zoom.fitTitle | Vừa màn hình | Fit to screen |
| corr.zoom.fit | Fit | Fit |
| corr.shortcuts.preview_html | <kbd>S</kbd> Chọn <kbd>A</kbd> Thêm <kbd>D</kbd> Xoá <kbd>Ctrl+Wheel</kbd> Zoom <kbd>Space</kbd> Kéo ảnh <kbd>←→</kbd> Ảnh | <kbd>S</kbd> Select <kbd>A</kbd> Add <kbd>D</kbd> Delete <kbd>Ctrl+Wheel</kbd> Zoom <kbd>Space</kbd> Pan <kbd>←→</kbd> Image |
| corr.shortcuts.styleditor_html | (nối từ JS 116-119) | Select/Delete/Erase/Bold-Italic/Align/Pick bubble/Undo/Zoom |
| corr.shortcuts.postrender_html | (nối từ JS 130-132) | Select/Delete/Undo/Redo/Zoom/Pan |
| corr.thumb.goToAria | Chuyển đến ảnh {name} | Go to image {name} |
| corr.thumb.notRendered | chưa render | not rendered |
| corr.thumb.notSaved | chưa lưu | not saved |
| corr.thumb.blocks_one | {n} block | {n} block |
| corr.thumb.blocks_other | {n} blocks | {n} blocks |
| corr.editor.toggle | 📐 Thuộc tính | 📐 Properties |
| corr.editor.title | Thuộc tính | Properties |
| corr.editor.closeAria | Đóng bảng thuộc tính | Close properties panel |
| corr.editor.noSelection | Chọn bóng thoại để chỉnh sửa | Select a speech bubble to edit |
| corr.editor.noSelectionStyled_html | Chọn bóng thoại để chỉnh sửa — bấm <kbd>[</kbd>/<kbd>]</kbd> để chọn | Select a speech bubble to edit — press <kbd>[</kbd>/<kbd>]</kbd> to select |
| corr.canvas.aria | Ảnh chỉnh sửa | Editing image |
| corr.canvas.ariaStyled | Ảnh đã xoá text và vẽ chữ dịch {name} — {n} bóng thoại | Image with original text erased and translation drawn {name} — {n} speech bubbles |
| corr.canvas.ariaPostrender | Ảnh đã dịch {name} — {n} bóng thoại | Translated image {name} — {n} speech bubbles |
| corr.canvas.ariaPreview | Ảnh {name} — {n} bóng thoại | Image {name} — {n} speech bubbles |
| corr.canvas.ariaDirty | , có thay đổi chưa render | , has unrendered changes |
| corr.nav.prev | ← Trước | ← Prev |
| corr.nav.next | Sau → | Next → |
| corr.footer.backToResults | ↩ Về kết quả | ↩ Back to results |
| corr.footer.rerenderOne | 🔄 Re-render ảnh này | 🔄 Re-render this image |
| corr.footer.saveAllBack | ✅ Lưu tất cả & Về kết quả | ✅ Save all & back to results |
| corr.footer.backToOcr | ✏️ Quay lại chỉnh OCR | ✏️ Back to OCR editing |
| corr.footer.renderOne | 🔄 Render ảnh này | 🔄 Render this image |
| corr.footer.saveAllView | ✅ Lưu tất cả & Xem kết quả | ✅ Save all & view results |
| corr.footer.cancel | Huỷ | Cancel |
| corr.footer.continue | ✅ Tiếp tục dịch & Render | ✅ Continue translating & rendering |
| corr.toast.undone | Đã hoàn tác ↩ | Undone ↩ |
| corr.toast.redone | Đã làm lại ↪ | Redone ↪ |
| corr.toast.eraseTooSmall | Vùng xoá quá nhỏ (tối thiểu 4×4 px) | Erase area too small (minimum 4×4 px) |
| corr.toast.deleted | 🗑️ Đã xoá bóng thoại | 🗑️ Speech bubble deleted |
| corr.toast.undoAction | Hoàn tác | Undo |
| corr.toast.rendered | ✅ Đã render lại ảnh | ✅ Image re-rendered |
| corr.toast.sessionExpired | Phiên đã hết hạn | Session expired |
| corr.toast.bboxInvalid | Toạ độ bbox không hợp lệ | Invalid bbox coordinates |
| corr.toast.renderFailed | Không thể render lại ảnh | Could not re-render image |
| corr.toast.retryAction | Thử lại | Retry |
| corr.toast.rendering | ⏳ Đang render… | ⏳ Rendering… |
| corr.toast.renderingCount | ⏳ Đang render ảnh {i}/{n} | ⏳ Rendering image {i}/{n} |
| corr.toast.bboxInvalidAt | Toạ độ bbox không hợp lệ ở ảnh {n} | Invalid bbox coordinates in image {n} |
| corr.toast.renderFailedCount | Không thể render ảnh {i}/{n} | Could not render image {i}/{n} |
| corr.toast.resetDone | Đã reset về OCR gốc | Reset to original OCR |
| corr.toast.preparing | ⏳ Đang chuẩn bị… | ⏳ Preparing… |
| corr.toast.fontLoadFailed | ⚠️ Không tải được font {name} | ⚠️ Could not load font {name} |
| corr.toast.fontFallback | ⚠️ Không tải được font {name} — dùng font mặc định | ⚠️ Could not load font {name} — using the default font |
| corr.toast.fontListFailed | ⚠️ Không tải được danh sách phông chữ | ⚠️ Could not load the font list |
| corr.toast.styleAppliedAll | 📋 Đã áp dụng kiểu cho tất cả block ảnh này | 📋 Style applied to all blocks in this image |
| corr.toast.bboxInvalidRule | Bbox không hợp lệ (x2 phải > x1, y2 > y1) | Invalid bbox (x2 must be > x1, y2 > y1) |
| corr.toast.warningPrefix | ⚠️ {message} | ⚠️ {message} |
| corr.hint.eraseRect | ▭ Kéo trên ảnh để xoá vùng hình chữ nhật (text gốc/SFX còn sót) · vùng nhỏ hơn 4×4px bị bỏ qua · ↩ Undo chỉ khôi phục hiển thị, vùng xoá vẫn render sạch · Esc để thoát | ▭ Drag on the image to erase a rectangular area (leftover original text/SFX) · areas smaller than 4×4px are ignored · ↩ Undo only restores the preview; erased areas still render clean · Esc to exit |
| corr.hint.eraseBrush | 🖌 Vẽ tự do để xoá text gốc/SFX còn sót · Cỡ cọ {size}px (đổi ở toolbar) · ↩ Undo chỉ khôi phục hiển thị, vùng xoá vẫn render sạch · Esc để thoát | 🖌 Draw freely to erase leftover original text/SFX · Brush size {size}px (change in the toolbar) · ↩ Undo only restores the preview; erased areas still render clean · Esc to exit |
| corr.hint.deletePostrender | 🖱️ Nhấp vào bóng thoại để xoá (vùng đó sẽ được xoá nền, không render chữ) · kéo lướt qua sẽ không xoá · Esc để thoát | 🖱️ Click a speech bubble to delete it (that area is erased from the background, no text rendered) · dragging across does not delete · Esc to exit |
| corr.hint.deletePreview | 🖱️ Nhấp vào bóng thoại để xoá (text gốc sẽ giữ nguyên trên ảnh kết quả) · kéo lướt qua sẽ không xoá · Esc để thoát | 🖱️ Click a speech bubble to delete it (the original text stays on the result image) · dragging across does not delete · Esc to exit |
| corr.hint.move | ⌨️ Dùng ←→↑↓ để di chuyển 1px · Giữ Shift = 10px · Kéo cạnh/góc để resize | ⌨️ Use ←→↑↓ to move 1px · Hold Shift = 10px · Drag edges/corners to resize |
| corr.hint.add | ✏️ Kéo trên ảnh để vẽ bóng thoại mới · Esc để huỷ | ✏️ Drag on the image to draw a new speech bubble · Esc to cancel |
| corr.ocr.running | ⏳ Đang OCR... | ⏳ OCR running... |
| corr.ocr.done | ✅ {text} | ✅ {text} |
| corr.ocr.noText | ⚠️ Không nhận được text | ⚠️ No text recognized |
| corr.ocr.error | ❌ Lỗi OCR | ❌ OCR error |
| corr.confirm.reset | Reset tất cả bóng thoại về kết quả OCR gốc? | Reset all speech bubbles to the original OCR results? |
| corr.style.heading | Kiểu chữ | Text style |
| corr.style.font | Phông chữ | Font |
| corr.style.fontAria | Phông chữ | Font |
| corr.style.size | Cỡ chữ | Font size |
| corr.style.sizeAria | Cỡ chữ tính bằng px | Font size in px |
| corr.style.auto | Tự động | Auto |
| corr.style.sizeWarn | ⚠️ Cỡ chữ sẽ được thu nhỏ cho vừa khung | ⚠️ Font size will be scaled down to fit the frame |
| corr.style.color | Màu chữ | Text color |
| corr.style.colorSwatchesAria | Màu chữ mẫu | Text color swatches |
| corr.style.autoColorTitle | Tự động: đen/trắng theo nền | Auto: black/white based on the background |
| corr.style.autoColorAria | Màu tự động | Auto color |
| corr.style.colorAria | Màu {c} | Color {c} |
| corr.style.customColorAria | Chọn màu chữ tuỳ chỉnh | Choose a custom text color |
| corr.style.boldItalicAria | Kiểu đậm nghiêng | Bold/italic style |
| corr.style.boldTitle | In đậm (phím B) | Bold (B key) |
| corr.style.italicTitle | In nghiêng (phím I) | Italic (I key) |
| corr.style.alignAria | Căn lề chữ | Text alignment |
| corr.style.alignLeft | ⬅ Trái | ⬅ Left |
| corr.style.alignLeftTitle | Căn trái (phím L) | Align left (L key) |
| corr.style.alignCenter | ↔ Giữa | ↔ Center |
| corr.style.alignCenterTitle | Căn giữa (phím C) | Align center (C key) |
| corr.style.alignRight | ➡ Phải | ➡ Right |
| corr.style.alignRightTitle | Căn phải (phím R) | Align right (R key) |
| corr.style.applyAll | 📋 Áp dụng cho tất cả block ảnh này | 📋 Apply to all blocks in this image |
| corr.editor.translatedLabel | Bản dịch | Translation |
| corr.editor.textLabel | Nội dung text | Text content |
| corr.editor.original | Gốc: {text} | Original: {text} |
| corr.editor.position | Vị trí (x1,y1,x2,y2) | Position (x1,y1,x2,y2) |
| corr.editor.nudgeAria | Di chuyển bóng thoại | Move speech bubble |
| corr.editor.nudgeUp | Lên 1px | Up 1px |
| corr.editor.nudgeLeft | Trái 1px | Left 1px |
| corr.editor.nudgeRight | Phải 1px | Right 1px |
| corr.editor.nudgeDown | Xuống 1px | Down 1px |
| corr.editor.delete | 🗑️ Xoá bóng thoại | 🗑️ Delete speech bubble |

**Không dịch:** "100%", "Fit" (đã có key), tên font, "4px/12px/24px", "x1..y2" placeholders, "blocks" → có key.

### 4.5 backend (app.py + gemini_translator.py)
| Key | vi | en |
|---|---|---|
| backend.formats | JPG, JPEG, PNG, WebP, BMP, TIFF hoặc AVIF | JPG, JPEG, PNG, WebP, BMP, TIFF or AVIF |
| backend.progress.translating | Đang dịch {n} đoạn text... | Translating {n} text segments... |
| backend.progress.batchFallback | Batch failed, falling back to single translations... | Batch failed, falling back to single translations... |
| backend.progress.unknownTranslator | Cảnh báo: Translator không xác định, text không được dịch | Warning: Unknown translator, text was not translated |
| backend.progress.translated | Dịch hoàn tất | Translation complete |
| backend.progress.noText | Không có text để dịch | No text to translate |
| backend.progress.rendering | Đang render text vào ảnh... | Rendering text onto images... |
| backend.progress.renderImage | Render: {name} | Render: {name} |
| backend.progress.done_other | Hoàn tất! {n} ảnh | Done! {n} images |
| backend.progress.done_one | Hoàn tất! {n} ảnh | Done! {n} image |
| backend.progress.ocrStart | Bắt đầu OCR toàn ảnh... | Starting full-image OCR... |
| backend.progress.ocrDone | OCR hoàn tất: {n} text blocks | OCR complete: {n} text blocks |
| backend.warn.geminiNotInit | Gemini chưa được khởi tạo nên app giữ nguyên text gốc. Hãy kiểm tra API key rồi thử lại. | Gemini is not initialized, so the app kept the original text. Check your API key and try again. |
| backend.warn.geminiFailed | Gemini không dịch được nên app giữ nguyên text gốc. Hãy kiểm tra API key/quota rồi thử lại. | Gemini could not translate, so the app kept the original text. Check your API key/quota and try again. |
| backend.warn.localLlmFailed | Local LLM không dịch được nên app giữ nguyên text gốc. Hãy kiểm tra server URL/model rồi thử lại. | Local LLM could not translate, so the app kept the original text. Check your server URL/model and try again. |
| backend.warn.googleFailed | Google Translate lỗi nên app giữ nguyên text gốc. | Google Translate failed, so the app kept the original text. |
| backend.warn.unknownTranslator | Translator không xác định nên app giữ nguyên text gốc. | Unknown translator, so the app kept the original text. |
| backend.warn.geminiAllKeysFailed | Tất cả Gemini API key đều không dùng được hoặc request thất bại. | All Gemini API keys are unusable or the request failed. |
| backend.error.noApiKey | Vui lòng nhập ít nhất 1 Gemini API Key. | Please enter at least 1 Gemini API key. |
| backend.error.noModel | Vui lòng nhập tên model Gemini. | Please enter the Gemini model name. |
| backend.error.noImages | Vui lòng chọn ít nhất 1 ảnh để dịch. | Please select at least 1 image to translate. |
| backend.error.unsupportedFormat | Chỉ hỗ trợ ảnh {formats}. | Only {formats} images are supported. |
| backend.error.unreadableImage | Không đọc được ảnh. Hãy thử file {formats} khác. | Could not read the image. Try another {formats} file. |
| backend.error.translationFailed | Lỗi dịch: {error} | Translation error: {error} |

---

## 5. Các trường hợp khó & giải pháp

### 5.1 Chuỗi động có tham số
- `f'Đang dịch {len(all_texts)} đoạn text...'` → `t('backend.progress.translating', n=len(all_texts))`.
- `'📁 ' + files.length + ' ảnh đã chọn'` → `I18N.t('index.fileLabel.chosen', {n: files.length})`.
- `'Render: ' + name` → `t('backend.progress.renderImage', name=name)`.
- **Quy tắc:** không bao giờ nối chuỗi dịch được; luôn key + params. Params (tên file, text OCR) là dữ liệu người dùng → **escape**.

### 5.2 Pluralization (đơn giản)
- Helper `tp(key_base, n, **params)`: en → `Intl.PluralRules('en').select(n)` ("one" n=1 / "other"); vi → luôn "other" (vi không có plural).
- Python: `tp(key_base, n, **params)` — en: n==1 → one; vi → other.
- Dictionary bắt buộc đủ cả `_one` + `_other` ở **cả 2 locale** (vi ghi 2 giá trị giống nhau); fallback của `_one` → `_other` để an toàn.
- Áp dụng: `corr.stats.bubbles/images`, `corr.thumb.blocks`, `backend.progress.done`.

### 5.3 Toasts có biến
- `showToast('Không thể render ảnh ' + (p.idx + 1) + '/' + payloads.length, {...})` → `showToast(I18N.t('corr.toast.renderFailedCount', {i: p.idx+1, n: payloads.length}), {...})`.
- `actionLabel: 'Thử lại'` → `actionLabel: I18N.t('corr.toast.retryAction')`.
- `'⚠️ ' + DATA.warning` → nếu warning là mapping (P1) → `I18N.t(w.key, w.params)`; P0 (string) → `I18N.t('corr.toast.warningPrefix', {message: DATA.warning})`.

### 5.4 aria-labels & title
- Phải dịch cùng hàng với text hiển thị (a11y): `aria-label="So sánh ảnh {{ img.name }}"`, `aria-label="Chuyển đến ảnh {{ name }}"`, `title="Căn trái (phím L)"`, canvas aria (JS `updateCanvasAria` — dùng `corr.canvas.ariaStyled/Postrender/Preview` + suffix `corr.canvas.ariaDirty`).
- Button không có text (nudge ▲◀▶▼) đã có title → dịch title.

### 5.5 HTML trong dictionary
- Chỉ key `_html`: `corr.shortcuts.*_html`, `corr.editor.noSelectionStyled_html` (chứa `<kbd>`). Python trả Markup; JS render qua innerHTML.
- Các key còn lại render qua textContent / autoescape — không được chứa HTML.

### 5.6 Option values & data-value refactor (P0 — bắt buộc cho en index.html)
Vấn đề: backend map display-text → value (app.py 1035-1084). Nếu dịch display text, map vỡ.
Giải pháp (thay đổi nhỏ, backward-compatible):
1. **index.html**: mọi `.option` của source_lang/language/translator/style/font thêm `data-value`:
   - source_lang: ja / zh / ko / en
   - language: vi / en / zh / ko / th / id / fr / de / es / ru
   - translator: gemini / copilot / google ("Local LLM" → copilot)
   - style: default / casual / formal / keep_honorifics / web_novel / action / literal / custom
   - font: **giữ nguyên display text** (backend đang lowercase + passthrough Yuki-*; không đổi hành vi)
2. **app.js `updateHiddenInputs`**: `el.getAttribute('data-value') || el.innerText`.
3. **localStorage `select_<id>`**: lưu **data-value** (mới); khi restore so sánh saved với data-value trước, rồi text (migrate giá trị cũ).
4. **app.py maps — thêm alias thô (additive, giữ key cũ)**:
   - source_lang_map += `{"ja":"ja","zh":"zh","ko":"ko","en":"en"}`
   - target_lang_map += `{code:code}` cho 10 code
   - translator_map += `{"gemini":"gemini","google":"google","copilot":"copilot"}`
   - style_map += `{"default":"","casual":"casual","formal":"formal","keep_honorifics":"keep_honorifics","web_novel":"web_novel","action":"action","literal":"literal","custom":""}`
5. **Option labels** → key `index.option.style*` (dịch được); các option proper-noun giữ nguyên text (không key).

### 5.7 Backend warnings (last_warning)
- **P0:** localize tại thời điểm tạo: `translator_obj.last_warning = t('backend.warn.geminiNotInit')` — `translate_texts_all` chạy trong request thread nên `g.locale` có sẵn. `gemini_translator.py:152` cũng set key `backend.warn.geminiAllKeysFailed`.
- **P1 (structured):** `last_warning = {"key": "...", "params": {...}}`; hiển thị lúc render: translate.html `{% if warning is mapping %}{{ t(warning.key, **warning.params) }}{% else %}{{ warning }}{% endif %}`; JS `DATA.warning` xử lý mapping tương tự.
- **Bắt buộc kèm P1:** sửa check `gemini_translator.py:288` (`"tất cả gemini api key" in error_str.lower()`) → check theo key/flag (vd `self.last_warning.get('key') == 'backend.warn.geminiAllKeysFailed'` hoặc cờ boolean `self._all_keys_failed`).
- `v3_last_warning` (session) lưu cùng cấu trúc để không lỗi thời ngôn ngữ khi đổi locale giữa session.

### 5.8 Progress events qua Socket.IO
- `emit_progress` giữ nguyên signature; call sites truyền `t(...)` → message localize server-side theo locale của request (cookie > Accept-Language > vi).
- Client index.html hiển thị `data.message` không đổi; `phaseNames` inline chuyển sang `I18N.t('index.phase.*')`.
- Ghi chú chấp nhận: nếu user đổi locale giữa chừng, progress in-flight giữ locale cũ (trang sẽ reload khi đổi qua dropdown).

### 5.9 localStorage cũ & migration
- Giữ nguyên key cũ: `select_*`, `gemini_api_keys`, `gemini_api_key`, `gemini_model`, `copilot_server`, `copilot_model`, `custom_prompt`, `styleditor_dirty_*`, `styleditor_draft_*` — **không đổi tên, không đổi format** (trừ select_* lưu data-value như 5.6, có migrate khi restore).
- Key mới: `mt_locale` (localStorage), cookie `mt_locale`, sessionStorage `mt_i18n_reloaded` (guard).

---

## 6. Acceptance criteria

### P0 (bắt buộc — hoàn thành t2 + t3)
- **A0.1** `i18n/vi.json` + `i18n/en.json` tồn tại; key set 2 file **giống hệt** (script `tools/check_i18n_keys.py`); giá trị vi = chép nguyên văn chuỗi hiện tại.
- **A0.2** Không còn text cứng user-visible ngoài dictionary: scan `tools/scan_hardcoded_strings.py` trên templates/, static/js/, app.py, translator/gemini_translator.py (loại trừ i18n/, static/qa/, tests/, temp_sessions/, docs/, debug_outputs/, comment lines) → 0 phát hiện user-visible.
- **A0.3** Auto-detect: profile mới, `navigator.language = en-US` → UI tiếng Anh (≤ 1 reload); `vi-VN` → tiếng Việt, không reload; lựa chọn lưu (localStorage) thắng browser language.
- **A0.4** Dropdown có trên cả 3 trang (index/translate/correction, đủ 3 mode correction); đổi → reload hiển thị đúng; lưu localStorage + cookie; `<html lang>` đúng.
- **A0.5** Fallback: xoá 1 key khỏi en.json → hiển thị vi; xoá khỏi cả vi → hiện raw key; **không** lỗi JS/Python.
- **A0.6** Tham số động đúng: progress "Translating N text segments...", "Render: name", "Done! N images"; toast "{i}/{n}"; aria "{name}" — không thấy "{...}" chưa thay.
- **A0.7** Plural en đúng: "1 speech bubble" / "5 speech bubbles"; "1 image" / "3 images"; "1 block" / "4 blocks".
- **A0.8** Backend error localize: submit không ảnh / thiếu API key / format sai / ảnh hỏng → tiếng Anh khi locale en.
- **A0.9** Warning localize: gemini init/fail, local LLM fail, google fail, unknown translator.
- **A0.10** Nhận diện giữ nguyên: #5E1675, Exo 2, gradient progress; layout en không tràn tại 1440px và 390px (kiểm tra bằng ảnh/screenshot).
- **A0.11** Locale vi: toàn bộ text = hệt bản hiện tại (không regression), flow chức năng không đổi.
- **A0.12** Progress phase localize: `index.phase.*` hiển thị đúng locale.

### P1
- **A1.1** Warnings structured (key+params) qua toàn bộ pipeline + `gemini_translator.py:288` check theo key/flag.
- **A1.2** Page title dịch theo locale (results.pageTitle, corr.pageTitle).
- **A1.3** Live-switch không reload cho phần JS động: `I18N.onRefresh` hook → `updateHints`, `updateBlockEditor`, `refreshStylePanel`, `updateCanvasAria`, `updateThumbnails` badges, phaseNames, file labels chạy lại khi `i18n:changed`.
- **A1.4** Migrate select_* localStorage cũ (display text → data-value).
- **A1.5** `tools/check_i18n_keys.py` chạy như một phần verify (key parity + placeholder parity: mọi `{x}` trong value vi phải có trong en và ngược lại).

### P2
- **A2.1** Live-switch toàn phần (cả text tĩnh) qua `data-i18n` attributes — không cần reload khi đổi dropdown.
- **A2.2** Thêm locale mới (zh/ja/ko...) chỉ bằng: file JSON mới + 1 option dropdown.
- **A2.3** Deep-link `?lang=en` override.
- **A2.4** QA harness pages (static/qa/*) — mặc định ngoài phạm vi; chỉ làm nếu còn thời gian.

---

## 7. Phân công & giao diện giữa t2/t3

| Hạng mục | Chủ | Files |
|---|---|---|
| Dictionaries (vi/en, key inventory mục 4) | **t2** frontend-i18n | i18n/vi.json, i18n/en.json |
| JS runtime + dropdown + detect | **t2** | static/js/i18n.js |
| Templates refactor | **t2** | templates/index.html, translate.html, correction.html |
| JS refactor (app.js, correction.js) + data-value (index.html + app.js) | **t2** | static/js/app.js, static/js/correction.js |
| Python tầng i18n | **t3** backend-i18n | i18n/__init__.py |
| app.py messages + maps aliases + before_request/context processor | **t3** | app.py |
| gemini_translator last_warning + check 288 | **t3** | translator/gemini_translator.py |

**Giao diện:** t3 đọc `i18n/*.json` do t2 tạo; nếu file chưa tồn tại lúc t3 code → `t()` fallback raw key, không block. Key names cố định trong mục 4 — **không tự ý đổi tên key**; nếu cần key mới: thêm vào cả 2 file + báo captain.

---

## 8. Kế hoạch verify (cho verifier-i18n, t4)

1. **Key parity:** `python tools/check_i18n_keys.py` → so sánh key set vi/en + placeholder set.
2. **Scan sót:** `python tools/scan_hardcoded_strings.py` → danh sách ứng viên residual; phân loại: comment (allowlist), string user-visible (FAIL).
3. **Auto-detect:** CDP/browser: set `navigator.language=en-US`, clear localStorage → load / → assert text tiếng Anh; lặp vi-VN; lặp localStorage='vi' + browser en → vi (choice thắng).
4. **Dropdown + persist:** đổi sang en → reload → en; kiểm tra localStorage `mt_locale` + cookie; đổi về vi.
5. **Fallback:** xoá tạm 1 key en → trang hiện vi value; xoá cả vi → raw key; không console error.
6. **Params/plural:** chạy pipeline thật (hoặc mock) → progress en đúng param; correction thumbnail count plural; toast {i}/{n}.
7. **Backend errors:** submit không ảnh, thiếu API key, file sai format, ảnh hỏng → error en đúng.
8. **Layout:** screenshot 1440x900 và 390x844 cho 3 trang ở en + vi; không tràn; màu/font không đổi.
9. **Regression vi:** diff text giữa bản cũ và bản mới ở locale vi (so sánh ảnh hoặc DOM text) → không khác.
10. **a11y spot-check:** aria-labels, titles dịch đủ ở en.

---

## 9. Out of scope (chốt)

- static/qa/*, tests/, temp_sessions/, debug_outputs/, docs/* (trừ spec này).
- CLI main.py, print() logs, filenames (`_translated.jpg`, `manga_translated`), image assets, brand "Manga Translator".
- **LLM translation prompts** (gemini_translator.py:176-356, local_llm_translator.py:210-229): là nội dung hướng dẫn dịch thuật cho model, không phải UI — **không i18n** (ghi chú: nếu sau này cần prompt theo ngôn ngữ đích, làm task riêng).
- Thêm ngôn ngữ mới ngoài vi/en (P2 architecture ready).
- Đổi nội dung tiếng Việt hiện tại (vd submit "Translate" → "Dịch"): **ngoài phạm vi** — vi giữ nguyên bản.

---

## 10. Rủi ro & quyết định mở

1. **data-value refactor** là thay đổi duy nhất chạm logic submit form (P0). Giảm rủi ro: alias additive ở backend (submission cũ vẫn hoạt động), localStorage migrate khi restore.
2. **Reload 1 lần ở lần đầu** (browser en, server chưa có cookie): chấp nhận; guard chống loop. P1 tối ưu bằng Accept-Language (đã có trong resolve_locale).
3. **Warning language**: P0 localize-at-creation (locale cố định theo session), P1 structured render-at-display. Không làm P1 thì đổi locale giữa session có thể thấy warning ngôn ngữ cũ — chấp nhận tạm.
4. **Payload nhúng cả 2 dict** (~30-40KB/trang): chấp nhận cho tool local; nếu cần tối ưu → fetch-on-demand (P2).
5. **i18n.json escape**: `json.dumps(...).replace("</", "<\\/")` để an toàn trong `<script>` tag.
6. Quyết định mở: tên hiển thị dropdown ("Tiếng Việt"/"English" + có cần cờ 🌐 không) — mặc định có 🌐, chờ UX gate t5 góp ý. **Đã chốt:** t5 yêu cầu 🌐 prefix → đã implement (i18n.js LOCALE_LABELS = "🌐 Tiếng Việt"/"🌐 English", repair t7).
7. **CHỐT CAPTAIN (namespace keys):** giữ nguyên namespace hiện tại — `common.*` / `index.*` / `results.*` / `corr.*` / `backend.*` theo spec §3.2/§4 đã duyệt (205 keys, parity 100%). **Không** thêm alias `frontend.*` / `js.*` (không code nào reference), **không** đổi tên key (tránh chạm verify/review in-flight).
8. **CHỐT CAPTAIN (repair t7 — UI gate t5):** (a) change handler `#locale-switch` luôn `sessionStorage.removeItem('mt_i18n_reloaded')` trước reload — guard chỉ bảo vệ auto-detect (init/reloadTo); (b) `corr.canvas.ariaStyled/Postrender/Preview` → cặp plural `_one`/`_other` (updateCanvasAria dùng tp); (c) 🌐 prefix dropdown (mục 6).
