# Manual Mode V2 — Đặc tả sản phẩm & kỹ thuật (Agency Product + Impeccable Shape)

> Trạng thái: **Chốt cho sprint** (chờ captain/engineers đánh giá)
> Phạm vi: nâng cấp chế độ chỉnh sửa thủ công của Manga-Translator.
> Nguyên tắc số 1: **giữ nhận diện hiện tại** — màu tím #5E1675, font Exo 2, layout toolbar trắng, ngôn ngữ UI tiếng Việt. Không redesign ngoài phạm vi.
> Nguyên tắc số 2: mọi surface được mô tả theo **mode Operate** — đủ ID phần tử, trạng thái, tương tác để frontend-engineer build trực tiếp từ spec này.

---

## 1. Mục tiêu & phạm vi

### 1.1 Goal
1. **Xoá block rõ ràng nhưng không xoá nhầm** — thao tác xoá dễ khám phá, có phòng hộ chống xoá nhầm, luôn hoàn tác được.
2. **Resize/nudge bbox trực quan** — 8 handle resize hoạt động thật (hiện tại chỉ vẽ, chưa nối logic), kéo di chuyển block, phím mũi tên nudge.
3. **Post-render editing** — sửa text + vị trí bbox **sau khi đã dịch/render** ngay trên trang kết quả, không cần chạy lại toàn bộ pipeline.
4. **Re-render một ảnh** — chỉ xoá nền + render lại đúng 1 ảnh (không OCR, không dịch lại).

### 1.2 Non-goals (out of scope — không làm trong sprint này)
- Không redesign trang chủ, không đổi bảng màu/font, không đổi layout ngoài các vùng nêu ở mục 2.
- Không multi-select, không copy/paste block, không rotate block, không snap grid.
- Không "OCR lại" trong chế độ post-render, không thêm block mới trong post-render.
- Không i18n, không thay font cho từng block, không server-side undo của kết quả re-render.
- Không chạm flow "không manual" (đường /translate → render thẳng vẫn y nguyên).

### 1.3 Hiện trạng tóm tắt (đã đọc code, đây là ground truth cho spec)
| Thành phần | Hiện trạng | Khoảng trống |
|---|---|---|
| app.py | Session = temp_sessions/<uuid>/ (session.json + page_*.jpg); /continue-translate build lại block từ form rồi chạy **cả pipeline** (dịch + render tất cả ảnh); /ocr-region OCR 1 vùng | Chưa có endpoint re-render 1 ảnh; sau render không lưu lại bản dịch/block để sửa |
| translate.html | Gallery kết quả + tab Đã dịch/Gốc + download/ZIP + nút "Quay lại chỉnh OCR" | Chưa có nút "Chỉnh sửa sau render" từng ảnh |
| correction.html | Toolbar (S/A/D/Undo/Redo/Reset + zoom), sidebar thumbnails, canvas, panel thuộc tính (textarea + 4 ô toạ độ + Clean/OCR/Xoá) | Toolbar chưa có trạng thái/hint cho chế độ xoá; chưa có chế độ post-render |
| correction.js | Draw 8 handle resize (hàm drawHandles) **nhưng không có hit-test/resize logic**; xoá = click **ngay trên mousedown** (rủi ro xoá nhầm khi kéo); kéo di chuyển block, undo/redo 60 bước, zoom/pan, phím tắt | Handle chưa nối; xoá thiếu phòng hộ; chưa có nudge; chưa có mode post-render |

### 1.4 Kiến trúc dữ liệu đề xuất (session state mở rộng)
Session hiện có giữ all_ocr_results (ảnh gốc + block OCR). **Bổ sung** sau khi render lần đầu (chỉ khi correction_session_id tồn tại):

~~~
// thêm vào session.json (chỉ với session đi qua manual mode)
{
  "render_plan": [
    {
      "name": "page1",
      "erase_regions": [[x1,y1,x2,y2], ...],   // bbox gốc của MỌI block đã render lần đầu (để re-render xoá đúng chỗ text gốc)
      "blocks": [                               // bản dịch đã render
        { "text": "こんにちは", "translated": "Xin chào", "bbox": [x1,y1,x2,y2] }
      ]
    }
  ]
}
// + ảnh đã render lưu file page_<i>_rendered.jpg (JPEG q92)
~~~

Luồng dữ liệu post-render:

~~~
/translate → OCR → correction.html (sửa pre-render, như cũ)
   → POST /continue-translate → dịch + render → _do_full_pipeline
   → [MỚI] lưu render_plan + page_*_rendered.jpg vào session → translate.html
translate.html → [MỚI] nút "✏️ Chỉnh sửa" mỗi card → GET /postrender/<sid>?img=<idx>
postrender editor (tái sử dụng correction.html + correction.js, mode=postrender)
   → sửa translated/bbox/xoá → [MỚI] POST /re-render-image (1 ảnh) hoặc POST /re-render-all
   → trả về translate.html với ảnh mới (không OCR, không dịch lại)
~~~

---

## 2. Surface map (theo mode Operate)

### 2.1 Màn hình A — correction.html (chế độ OCR/pre-render) — giữ như cũ + 3 cải tiến
**Bố cục hiện tại giữ nguyên** (topbar tím, toolbar trắng, sidebar trái, canvas giữa, editor phải, footer).

| Vùng | ID phần tử | Trạng thái mới cần có |
|---|---|---|
| Toolbar xoá | #tool-delete | Khi active: button tím + **hint bar** .delete-hint hiện dưới toolbar: "🖱️ Nhấp vào bóng thoại để xoá · Esc để thoát chế độ" |
| Toolbar select | #tool-select | Khi block được chọn: hiện hint nudge "←→↑↓ di chuyển 1px · Shift = 10px" |
| Canvas | #main-canvas | Handle resize hoạt động (xem F2); con trỏ đổi theo handle |
| Editor | #block-properties | Thêm hàng nút nudge (4 mũi tên + giá trị bước), hiển thị W×H block |
| Toast | #toast | Thêm biến thể toast--error, toast--undo (kèm nút "Hoàn tác") |

**Luồng xoá (F1)** — trạng thái máy:
~~~
[Chế độ D] → hover block: viền đỏ #ff1744 + fill 0.2 → mousedown ghi nhớ → mouseup với dịch chuyển ≤ 6px: XOÁ + toast undo
                                                       → mouseup với dịch chuyển > 6px: không xoá (coi là kéo nhầm)
[Esc] → thoát chế độ D → tool trở về select
~~~

### 2.2 Màn hình B — translate.html (trang kết quả) — thêm 1 nút/card
| Vùng | Thay đổi |
|---|---|
| Mỗi .image-card | Thêm nút <a class="edit-btn" href="/postrender/<sid>?img=<idx>">✏️ Chỉnh sửa</a> cạnh Download — **chỉ render khi correction_session_id tồn tại** |
| Tiêu đề | Thêm dòng phụ "Ảnh đã chỉnh sửa sau render sẽ tự cập nhật tại đây" |

### 2.3 Màn hình C — Post-render editor (MỚI, tái sử dụng correction.html)
Đây là **cùng template** correction.html, bật bởi window.CORRECTION_DATA.mode = "postrender":

| Phần tử | Khác biệt so với chế độ OCR |
|---|---|
| Nền canvas | Ảnh **đã render** (page_<i>_rendered.jpg), không phải ảnh gốc |
| Block hiển thị | Label = translated; **viền nét đứt màu cam = dirty (chưa re-render), viền liền xanh = đã render** |
| Toolbar | Chỉ giữ: Select, Xoá, Undo, Redo, Zoom, Pan. **Ẩn**: Thêm, Reset (Reset = "Khôi phục bản render cuối", P1), OCR lại, Clean |
| Editor panel | Textarea sửa **bản dịch** (label "Bản dịch"), 4 ô toạ độ, nút Xoá. Không có Clean/OCR lại |
| Footer | #btn-cancel "↩ Về kết quả" · #btn-rerender-one "🔄 Re-render ảnh này" · #btn-save-all "✅ Lưu tất cả & Về kết quả" |
| Sidebar thumb | Có badge cam "dirty" trên ảnh có thay đổi chưa render |

---

## 3. Feature spec

### F1 — Xoá block an toàn (P0)
**Mục tiêu:** dễ khám phá, không xoá nhầm, luôn hoàn tác được.

**Hành vi:**
1. Xoá chỉ kích hoạt trên **mouseup** với ngưỡng dịch chuyển ≤ 6px màn hình (click thật, không phải drag). Kéo lướt qua block trong chế độ D không xoá.
2. Giữ hover đỏ + label; giữ nút Xoá trong editor panel; giữ phím Del xoá block đang chọn khi tool = select (khớp code hiện tại correction.js dòng 489: `if (selectedBlockIdx >= 0 && currentTool === 'select')`).
3. Sau khi xoá: toast .toast--undo hiện 4s: "🗑️ Đã xoá bóng thoại — [Hoàn tác]" (nút bấm được, gọi undo()).
4. Chế độ D hiện hint bar + con trỏ crosshair; **Esc** thoát chế độ D.
5. Ngữ nghĩa pre-render: block bị xoá ⇒ vùng đó **không bị pipeline đụng tới** (text gốc còn nguyên trên ảnh kết quả) — giữ đúng hành vi hiện tại, ghi rõ trong hint.
6. Ngữ nghĩa post-render: block bị xoá ⇒ vùng đó **xoá nền (inpaint), không render text** (xem F5).

**Acceptance:**
- A1.1: Nhấn D, hover block → viền đỏ; click chuột (không di chuyển) → block biến mất, toast "Hoàn tác" xuất hiện.
- A1.2: Bấm "Hoàn tác" trên toast → block trở lại nguyên trạng text+bbox.
- A1.3: Mousedown lên block rồi kéo > 6px rồi thả → block **không** bị xoá.
- A1.4: Esc trong chế độ D → tool về select, hint biến mất, hover đỏ tắt.
- A1.5: Del chỉ xoá khi có block đang chọn và tool = select (giữ hành vi cũ, không xoá ngoài ý muốn khi đang gõ textarea — keydown đã bỏ qua INPUT/TEXTAREA).

### F2 — Resize bằng 8 handle (P0)
**Mục tiêu:** handle đã vẽ sẵn (drawHandles) phải thao tác được.

**Hành vi:**
1. Hit-test 8 handle: **mouse** bán kính ≥ 10 CSS px (độc lập zoom); **touch** dùng vùng hit **vô hình** mục tiêu **≥ 44×44 CSS px** quanh mỗi handle (visual handle vẫn nhỏ). Cursor: nwse-resize (góc TL/BR), nesw-resize (TR/BL), ns-resize (cạnh trên/dưới), ew-resize (cạnh trái/phải). **[CHỐT CAPTAIN]** Nếu bbox nhỏ làm vùng hit chồng nhau: chọn handle **gần tâm con trỏ nhất**, ưu tiên **corner trước edge**; clamp/min-size vẫn giữ. 24×24 chỉ là **fallback tối thiểu** trong không gian cực hẹp — **không phải** acceptance mặc định.
2. Kéo góc: đổi cả 2 trục; kéo cạnh: 1 trục. Kích thước tối thiểu **8×8 px ảnh**; clamp trong [0, image bounds].
3. Redraw liên tục khi kéo; khi thả: **1 snapshot undo**, cập nhật 4 ô toạ độ + thumbnails.
4. Handle có viền trắng 1px quanh ô xanh #00e676 (nhìn rõ trên nền manga tối).
5. (P1) Giữ **Shift** khi kéo góc = giữ tỷ lệ khung hình.

**Acceptance:**
- A2.1: Chọn block → 8 handle xuất hiện; rê chuột tới handle → cursor đổi đúng loại.
- A2.2: Kéo handle góc TL sang phải 20px → x1 tăng 20px, y1 không đổi (khi kéo đúng trục); kéo cạnh phải → chỉ x2 đổi.
- A2.3: Kéo thu nhỏ dưới 8px → bbox không nhỏ hơn 8×8.
- A2.4: Kéo handle vượt mép ảnh → bbox bị clamp vào mép.
- A2.5: Ctrl+Z sau resize → bbox trở về kích thước trước khi resize (1 lần undo cho cả lượt kéo).
- A2.6 (touch, chốt captain): vùng hit ẩn quanh handle đạt **≥ 44×44 CSS px**; khi bbox nhỏ làm vùng hit chồng nhau → thao tác nhắm vào handle **gần tâm chạm nhất**, **corner ưu tiên trước edge**; clamp/min-size giữ nguyên.

### F3 — Di chuyển & nudge bbox (P0: move, P1: nudge)
**Mục tiêu:** điều chỉnh vị trí trực quan, chính xác tới từng pixel.

**Hành vi:**
1. Kéo di chuyển nguyên block trong tool Select (đã có) — giữ nguyên; đảm bảo kéo bắt đầu từ **bên trong** block, không phải từ handle (ưu tiên handle khi trùng).
2. **Nudge bằng phím** (P1 nhưng rẻ — khuyến nghị làm cùng P0): khi có block chọn, Arrow keys dịch block ±1 px ảnh; **Shift+Arrow = ±10 px**. Khi **không** có block chọn, ←/→ giữ hành vi chuyển ảnh cũ.
3. Editor panel thêm 4 nút nudge (▲▼◀▶) cho người không dùng phím + hiển thị W×H của block.
4. 4 ô toạ độ: làm tròn + clamp vào mép ảnh khi blur (P0); nhấn Enter commit (P1).

**Acceptance:**
- A3.1: Kéo block (không trúng handle) → block di chuyển, thả chuột = 1 undo step, toạ độ editor cập nhật.
- A3.2: Chọn block, nhấn → → x1,x2 cùng tăng 1px (clamp ở mép phải).
- A3.3: Shift+↑ → dịch −10px, 1 undo step.
- A3.4: Không chọn block nào, nhấn → → chuyển ảnh kế (hành vi cũ giữ nguyên).
- A3.5: Gõ x2 = 99999 rồi blur → bị clamp về width ảnh.

### F4 — Post-render editing (P0)
**Mục tiêu:** sửa text dịch + vị trí sau khi đã render, từ trang kết quả.

**Hành vi:**
1. translate.html: mỗi card có nút "✏️ Chỉnh sửa" → GET /postrender/<session_id>?img=<idx>.
2. Editor mở đúng ảnh idx; canvas nền = ảnh đã render; block hiển thị text dịch.
3. Sửa textarea = sửa translated; sửa 4 ô toạ độ / kéo / resize = sửa bbox. Mọi thay đổi đánh dấu ảnh **dirty** (viền cam nét đứt + badge thumb) cho tới khi re-render.
4. Undo/redo hoạt động với snapshot mở rộng {text, translated, bbox} (P0).
5. Nút "Re-render ảnh này" (F5) và "Lưu tất cả & Về kết quả" (re-render các ảnh dirty, ảnh sạch giữ nguyên, redirect translate.html).
6. Không có nút Thêm/OCR lại trong mode này (P0).

**Acceptance:**
- A4.1: Từ translate.html bấm "Chỉnh sửa" → mở editor đúng ảnh, đúng bản dịch.
- A4.2: Sửa text → nhấn Re-render → text mới xuất hiện trên ảnh, viền chuyển xanh (sạch).
- A4.3: Kéo bbox block → Re-render → text render đúng vị trí mới, **text gốc cũ không lộ ra** (nhờ erase_regions, F5).
- A4.4: Xoá block → Re-render → vùng đó sạch (không còn chữ), không ảnh hưởng block khác.
- A4.5: "Lưu tất cả" → về translate.html, ảnh hiển thị là bản mới; F5 trên browser → vẫn là bản mới (đã persist).
- A4.6: Session hết hạn (404) → toast lỗi + redirect "/" (không treo trang).

### F5 — Re-render một ảnh (P0) — lõi kỹ thuật
**Mục tiêu:** chỉ xoá nền + render lại 1 ảnh, **không** OCR, **không** dịch lại.

**Thuật toán server (pseudo-code trong spec):**
~~~
# Refactor: tách render_single_image trong translate_and_render thành hàm module-level:
def render_image_with_blocks(name, image, blocks, font_path, source_lang,
                             vision_adapter=None, extra_erase_regions=None):
    # blocks: [{text, translated, bbox}]
    # 1) Với mỗi block: erase_text_region(image, bbox) → lấy appearance (text_color...)
    # 2) Với extra_erase_regions (erase_regions gốc + deleted_regions) KHÔNG trùng bbox hiện tại:
    #    erase_text_region từng vùng → xoá text gốc còn sót khi user đã dời/resize bbox
    # 3) render_all_blocks(image, render_blocks, font_path)
    return image  # và danh sách blocks đã chuẩn hoá
~~~

- POST /re-render-image: input session_id, image_idx, blocks_json, deleted_regions_json. Server: load ảnh gốc page_<i>.jpg → chạy render_image_with_blocks với extra_erase_regions = render_plan[i].erase_regions + deleted_regions → ghi page_<i>_rendered.jpg → cập nhật render_plan[i].blocks → trả {name, data (b64 JPEG), blocks}.
- POST /re-render-all: lặp các ảnh dirty (client gửi danh sách) rồi trả toàn bộ processed_images cho translate.html (tái dùng build_result_images).
- **Contract cứng:** 2 endpoint này không được gọi ChromeLensOCR, không gọi translator. Verify bằng code review + test tích hợp đo thời gian (re-render 1 ảnh phải nhanh hơn rõ rệt so với pipeline đầy đủ — tiêu chí mềm < 15s/ảnh trên máy thường, không gọi mạng ngoài trừ vision adapter nếu bật).
- Double-submit guard: frontend khoá nút khi đang bay (isBusy flag dùng chung với isOcrPending), server không cần lock (P0) — ghi đè idempotent.

**Acceptance:**
- A5.1: Sửa 1 block trên 1 ảnh → chỉ ảnh đó được render lại; các ảnh khác không đổi (so sánh hash/pixel).
- A5.2: Re-render không gọi OCR/dịch: log server không có "Phase 1/Phase 2" khi gọi 2 endpoint này.
- A5.3: Dời bbox ra xa vị trí text gốc → sau re-render vùng cũ **sạch** (không còn chữ gốc), vùng mới có chữ dịch.
- A5.4: Xoá block → re-render → vùng xoá sạch nền, các block khác giữ nguyên nội dung.
- A5.5: Bấm Re-render 2 lần liên tục nhanh → không có 2 request song song (nút bị khoá trong lúc bay).
- A5.6: blocks gửi lên bị chuẩn hoá/clamp server-side (normalize_bbox_for_json với image_shape) — bbox ngoài mép không gây lỗi.

### F6 — Undo/Redo (P0: giữ + mở rộng)
- Snapshot mở rộng: {imageIdx, images: [{blocks: [{text, translated, bbox}], deletedRegions}]} — đảm bảo post-render undo khôi phục cả text dịch lẫn danh sách vùng đã xoá.
- Giữ MAX_UNDO = 60, Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y.
- Ghi chú UX: undo sau re-render **không** đổi ảnh nền ngay — block trở về trạng thái dirty (viền cam) và cần Re-render lại. Việc này đúng thiết kế (re-render là hành động tường minh, không tự động).

**Acceptance:** A6.1: chuỗi "sửa text → resize → xoá → Undo×3" khôi phục đúng từng bước; A6.2: undo trong post-render giữ nguyên translated cũ.

### F7 — Loading / Error / Trạng thái (P0)
| Tình huống | Hành vi |
|---|---|
| Đang re-render | Nút → disabled + label "⏳ Đang render…"; con trỏ canvas wait; chặn thao tác sửa tiếp (không bắt buộc P0 — tối thiểu chặn double submit) |
| Re-render lỗi mạng/5xx | Toast toast--error đỏ + nút "Thử lại" (giữ nguyên payload lần trước) |
| Session 404 | Toast lỗi "Phiên hết hạn" → redirect "/" sau 2s |
| Bbox không hợp lệ (x2≤x1...) | Server 422, frontend toast + bôi đỏ 4 ô toạ độ |
| /continue-translate đang submit | Nút "Tiếp tục dịch & Render" → disabled + "⏳ Đang chuẩn bị…" (chống double submit) |
| OCR vùng (/ocr-region) | Giữ hành vi ocr-status hiện tại |

**Acceptance:** A7.1: tắt mạng → Re-render → toast lỗi + nút Thử lại; A7.2: bật lại mạng → Thử lại thành công; A7.3: 2 cú click nhanh → chỉ 1 request; A7.4: xoá thư mục session → mọi action báo hết hạn + redirect, không exception JS.

### F8 — Bàn phím tắt (P0: giữ map cũ; P1: bổ sung)
| Phím | Hành vi | Trạng thái |
|---|---|---|
| S / A / D | Chọn / Thêm / Xoá (Thêm ẩn ở postrender) | giữ |
| Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y | Undo / Redo | giữ |
| Del | Xoá block đang chọn | giữ |
| Esc | Bỏ chọn / thoát chế độ D / huỷ đang vẽ | giữ |
| ←/→ | Chuyển ảnh (khi không chọn block) / nudge (khi có chọn) | **đổi nhẹ** |
| ↑/↓ + Shift | Nudge block ±1px / ±10px | **mới** |
| + / − / 0 / 1 | Zoom in/out / fit / 100% | giữ |
| Space + kéo | Pan | giữ |
| [ / ] | Chuyển block đang chọn (cycle) | P1 |
| F | Fit màn hình (alias của 0) | P1 |

**Acceptance:** A8.1: map trên hoạt động đúng; A8.2: gõ phím trong textarea/input không kích hoạt shortcut (giữ guard hiện có).

### F9 — Responsive & Touch (**TRONG SPRINT** — captain chốt; plan đã duyệt có Impeccable audit)
- ≥ 1100px: layout 3 cột như hiện tại.
- 700–1100px: sidebar thumbnails chuyển thành **dải ngang** cuộn được trên đầu canvas; editor panel thu gọn thành nút "Thuộc tính" mở **drawer phải**.
- < 700px: toolbar wrap (đã có flex-wrap), editor = bottom sheet; canvas chiếm toàn bộ.
- Touch: vùng bắt handle resize **≥ 44×44 CSS px vô hình** (xem F2.1 — chốt captain); ngưỡng xoá F1 giữ nguyên ≤ 6px vì tap không phải drag.
- Giữ nguyên màu/font/khoảng cách hiện có — chỉ đổi cách bố trí vùng chứa, không đổi style thành phần.
**Acceptance:** A9.1: 3 breakpoint hiển thị đúng (kiểm tra bằng DevTools), không tràn ngang, mọi nút bấm được. A9.2 (touch): handle resize thao tác được bằng ngón tay nhờ vùng hit ≥ 44×44px; bbox nhỏ/chồng vùng hit vẫn chọn đúng handle (gần tâm, corner trước).

### F10 — Accessibility (P1)
- Canvas: role="img" + aria-label mô tả ảnh hiện tại và số block; trạng thái tool công bố qua aria-pressed trên các tool-btn.
- Toast: role="status" (thường) / role="alert" (lỗi) + aria-live.
- Focus: các tool-btn có focus-visible ring rõ; block đang chọn thấy được qua outline khi focus editor (editor scrollIntoView block đang sửa).
- Contrast: handle xanh có viền trắng (F2); viền xoá đỏ #ff1744 trên nền sáng dùng thêm viền đen 1px (P1).
- prefers-reduced-motion: tắt animation toast/pan mượt.
- Mọi thao tác chuột đều có đường phím tương đương (nudge, Del, cycle block).
**Acceptance:** A10.1: chạy Lighthouse a11y ≥ 90 trên 2 màn hình editor; A10.2: hoàn thành luồng "chọn block → resize → xoá → undo" chỉ bằng bàn phím.

---

## 4. API contract (backend — giao cho backend-engineer)

### 4.1 POST /re-render-image (mới)
~~~
form-data:
  session_id: str (uuid)
  image_idx: int
  blocks_json: '[{"text": "...", "translated": "...", "bbox": [x1,y1,x2,y2]}, ...]'
  deleted_regions_json: '[[x1,y1,x2,y2], ...]'
200 → { "name": str, "data": "<b64 jpeg>", "blocks": [{text, translated, bbox}] }
400 → {"error": "..."} (bad json / idx ngoài phạm vi)
404 → {"error": "session_not_found"}
422 → {"error": "invalid_bbox"}
~~~
Side effect: ghi page_<i>_rendered.jpg, cập nhật session.json.render_plan[i].blocks; **không** đổi all_ocr_results (ảnh gốc bất biến — mọi re-render đều xuất phát từ ảnh gốc, nên render lại luôn idempotent).

### 4.2 POST /re-render-all (mới)
~~~
form-data: session_id, dirty_indices_json: "[0,2]"
200 → render translate.html với processed_images (tái dùng build_result_images)
~~~

### 4.3 GET /postrender/<session_id>?img=<idx> (mới)
Render correction.html với mode="postrender", images = render_plan (data = b64 của ảnh đã render, blocks kèm translated). Session không tồn tại → redirect "/".

### 4.4 Thay đổi hàm hiện có
- _do_full_pipeline: khi correction_session_id → sau translate_and_render, lưu render_plan + page_*_rendered.jpg vào session.
- translate_and_render: tách render_single_image → render_image_with_blocks (module-level, thêm param extra_erase_regions) — dùng chung cho cả pipeline lẫn re-render.
- translate.html template: nút "Chỉnh sửa" khi có correction_session_id.

---

## 5. Ưu tiên P0 / P1 / P2

### P0 — MVP (phải có trong sprint này)
1. **F1** xoá an toàn (mouseup+ngưỡng, toast undo, hint bar, Esc).
2. **F2** resize 8 handle thật (hit-test, clamp, min-size, undo).
3. **F4** post-render editing (entry từ translate.html, editor chế độ postrender, dirty state).
4. **F5** re-render 1 ảnh + /re-render-image + persist render_plan (không OCR/dịch lại).
5. **F6** undo/redo mở rộng cho post-render.
6. **F7** loading/error/double-submit guard cho các nút mới.
7. **F3** move (giữ nguyên) + clamp toạ độ khi blur.

### P1 — Nên có (làm ngay sau P0 nếu sprint còn dư)
*Ghi chú scope (đồng bộ với chốt cuối của captain):* toast lỗi + nút "Thử lại" = **P0** (F7); badge dirty thumbnail = **P0** (F4 + surface map 2.3); điều kiện Delete (tool=select, F1.2/A1.5) **đã chốt đúng**; F9 responsive/touch = **trong sprint** (plan đã duyệt có Impeccable audit). Dưới đây là phần P1 thực sự:
- Nudge phím mũi tên + nút nudge + hiển thị W×H; Shift giữ tỷ lệ khi resize.
- [ / ] cycle block; F = fit; Enter commit toạ độ.
- **F10** a11y (aria, role, focus-visible, reduced-motion, contrast).
- Reset trong postrender = "Khôi phục bản render cuối" (undo mọi thay đổi chưa lưu của ảnh hiện tại).

### P2 — Sau này (ghi nhận, không làm sprint này)
- Multi-select, copy/paste block, snap grid, rotate.
- "OCR lại" trong post-render; thêm block mới trong post-render.
- Server-side undo re-render (giữ N bản JPEG gần nhất/ảnh).
- Chọn font/text-color cho từng block; merge/split block; panel lịch sử (history list UI).

---

## 6. Acceptance criteria tổng (điều kiện hoàn thành sprint)
1. Luồng đầy đủ E2E: upload 2 ảnh → manual correction → xoá 1 block, resize 1 block, nudge 1 block → Tiếp tục → kết quả đúng → "Chỉnh sửa" ảnh 1 → sửa text + kéo bbox + xoá 1 block → "Re-render ảnh này" → ảnh 1 đúng mới, ảnh 2 không đổi → "Lưu tất cả" → translate.html hiển thị bản mới → F5 browser vẫn mới.
2. Không có hành vi xoá nhầm (A1.3, A1.5 pass).
3. Re-render không gọi OCR/dịch (A5.2 pass) và nhanh hơn hẳn pipeline đầy đủ.
4. Toàn bộ acceptance A1–A10 (P0 bắt buộc, P1 theo dư địa sprint) pass manual test theo checklist.
5. UI giữ nhận diện: màu #5E1675, font Exo 2, toolbar trắng, text tiếng Việt; diff template chỉ thêm phần tử mới, không vẽ lại style cũ.
6. Không regression flow không-manual (upload → render thẳng) — test lại bằng 1 ảnh không bật manual correction.
7. Session lưu/đọc mới tương thích ngược: session cũ (không có render_plan) vẫn mở correction.html bình thường.

## 7. Rủi ro & câu hỏi mở
| # | Rủi ro | Giảm thiểu đã chốt |
|---|---|---|
| R1 | Dời bbox làm text gốc cũ lộ ra khi re-render | erase_regions lưu mọi bbox gốc, luôn xoá trước khi render (F5) |
| R2 | erase_text_region đè lên vùng đã xoá → appearance/text_color lệch | Thứ tự: erase từng block hiện tại trước (lấy appearance), rồi mới xoá extra regions (F5 pseudo-code) |
| R3 | Re-render trùng session cũ thiếu render_plan (user vào thẳng /postrender) | Nếu thiếu render_plan → redirect về correction.html chế độ OCR (fallback rõ ràng) |
| R4 | Session phình to (thêm N ảnh rendered) | TTL 6h có sẵn; JPEG q92 ≈ ảnh gốc; chấp nhận, P2 mới tối ưu |
| R5 | Vision adapter (nếu bật) không deterministic giữa 2 lần render | Chấp nhận khác biệt nhỏ; vẫn đúng chữ/vị trí. Ghi note trong code |
| R6 | Ảnh rất lớn → canvas chậm khi kéo handle | Giữ cơ chế hiện tại (canvas full-res); P2 mới cần tile/layer |
| R7 | Xung đột phím ←/→ giữa nudge và chuyển ảnh | Quy tắc: có block chọn = nudge; không = chuyển ảnh; Esc để bỏ chọn (F3/F8) |

## 8. Câu hỏi mở cho captain (không chặn sprint)
1. Post-render "xoá block" = xoá nền (chốt hiện tại) hay khôi phục text gốc? (Đã chốt: xoá nền, vì text gốc đã mất sau render lần đầu; đổi ý chỉ cần sửa F5.4.)
2. Có cần giới hạn dung lượng render_plan khi session nhiều ảnh? (P2)
