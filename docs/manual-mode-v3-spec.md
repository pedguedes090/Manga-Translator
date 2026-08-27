# Manual Mode V3 — Đặc tả sản phẩm & kỹ thuật (Editor thủ công WYSIWYG)

> Trạng thái: **Chốt cho sprint** (chờ captain/engineers đánh giá)
> Phạm vi: nâng cấp Manual Mode V2 → editor thủ công WYSIWYG toàn diện.
> Nguyên tắc số 1: **giữ nhận diện hiện tại** — màu tím #5E1675, font Exo 2, layout toolbar trắng, ngôn ngữ UI tiếng Việt. Không redesign ngoài phạm vi.
> Nguyên tắc số 2: mọi surface được mô tả theo **mode Operate** — đủ ID phần tử, trạng thái, tương tác để frontend-engineer build trực tiếp từ spec này.
> Nền móng: **tái sử dụng V2** — render_plan, /re-render-image, canvas engine correction.js, erase_text_region/render_all_blocks (add_text.py). Không viết lại, chỉ mở rộng.

---

## 1. Mục tiêu & phạm vi

### 1.1 Goal
1. **Luồng correction → editor WYSIWYG**: sau nút "Tiếp tục dịch & Render" ở trang correction, thay vì chạy thẳng pipeline, mở trang editor với **NỀN ẢNH ĐÃ XOÁ TEXT** (text gốc của mọi block đã bị xoá khỏi ảnh nền).
2. **Chữ dịch hiện sẵn trên ảnh**: bản dịch của từng block được vẽ live trên canvas (WYSIWYG), sửa được nội dung (textarea), kéo vị trí, resize bbox (kế thừa F2/F3 V2).
3. **Công cụ xoá nền**: brush (cọ, đường kính đổi được) + rect (khung chữ nhật) để dọn text gốc còn sót / SFX / vùng bẩn; **erase_regions mở rộng, không bao giờ co lại** (server-side guarantee).
4. **Panel style per-block**: font (danh sách từ fonts/ qua API), cỡ chữ (px hoặc Tự động), màu chữ, đậm/nghiêng, căn lề (trái/giữa/phải).
5. **Re-render WYSIWYG không gọi lại OCR/dịch**: /re-render-image mở rộng nhận style + erase regions/mask; ảnh kết quả khớp preview.
6. Giữ nhận diện #5E1675 / Exo 2 / tiếng Việt ở mọi surface mới; không regression V2 và flow không-manual.

### 1.2 Non-goals (out of scope — không làm trong sprint này)
- Không "OCR lại" trong style editor; **không thêm block mới** trong style editor — **[CHỐT CAPTAIN mục 8.1] giữ non-goal P2**: editor V3 chỉ chỉnh sửa các block ĐÃ DỊCH (nội dung, vị trí, style, xoá); không có nút Thêm / OCR lại trong sprint này (chống scope creep).
- Không multi-select, copy/paste, rotate, snap grid.
- Không i18n; không style theo đoạn text trong block (P2).
- Không server-side undo của kết quả render (giữ N bản JPEG/ảnh — P2).
- Không đổi bảng màu/font/layout ngoài các vùng nêu ở mục 2.
- Không chạm flow không-manual (/translate → render thẳng).

### 1.3 Hiện trạng tóm tắt (đã đọc code — ground truth cho spec)
| Thành phần | V2 hiện có | Khoảng trống cho V3 |
|---|---|---|
| app.py | render_plan trong session.json; /continue-translate → _do_full_pipeline (dịch + render tất cả); /postrender; /re-render-image (erase_regions accumulate, không co lại); /re-render-all; /translate-result | Chưa có endpoint "dịch xong rồi dừng, vào editor ảnh đã xoá text"; chưa có font list API; /re-render-image chưa nhận style/mask |
| add_text.py | erase_text_region (inpaint); _compute_font_and_wrap (auto-size); _draw_text_on_pil (outline, căn giữa); render_all_blocks; get_cached_font; resolve_font_path_for_text | Chưa hỗ trợ per-block: font khác, cỡ cố định, màu, đậm/nghiêng, căn lề; chưa có erase theo mask |
| correction.html/js | Engine canvas: modes preview/postrender, 8-handle resize, nudge, undo/redo mở rộng, dirty state, double-submit guard, keyboard map, responsive, a11y | Chưa có mode "styleditor": vẽ text live bằng font thật, tool brush/rect xoá nền, panel style per-block |
| Fonts | fonts/*.ttf (23 font: animeace_i, ariali, mangati, Yuki-*); selected_font trong session; get_font_path map tên→file | Không có API danh sách font; font không được serve cho trình duyệt (không FontFace → không preview chữ đúng font) |

### 1.4 Kiến trúc dữ liệu đề xuất (session state mở rộng, backward compatible)
~~~
session.json (thêm, chỉ khi đi qua style editor):
{
  "v3_draft": {                          // bản dịch + style CHƯA render (editor V3 đọc đây)
    "images": [
      { "name": "page1",
        "blocks": [ { "text": "...", "translated": "...",
                      "bbox": [x1,y1,x2,y2],
                      "style": { "font": "Yuki-Burobu", "font_size": 0,
                                 "text_color": "#000000", "bold": false,
                                 "italic": false, "align": "center" } } ] }
    ]
  }
  // + ảnh đã xoá text: page_<i>_erased.jpg (JPEG q92) — nền của editor
}
// render_plan blocks: thêm field KHÔNG bắt buộc "style" (đồng cấu trúc như trên)
// — /postrender (V2) đọc/ghi được, bỏ qua nếu không có.
~~~

Luồng dữ liệu V3:
~~~
/translate → OCR → correction.html (sửa pre-render, như cũ)
   → POST /styleditor-prepare (mới): rebuild blocks + DỊCH (Phase 2) + xoá text gốc
     → ghi v3_draft + page_*_erased.jpg → redirect /styleditor/<sid>?img=0
/styleditor/<sid>?img=<i> (mới): editor V3 — nền ảnh erased, text dịch vẽ live, panel style,
   tool xoá nền (brush/rect)
   → POST /re-render-image (mở rộng): nhận style + erase_regions + erase_mask
     → ghi page_<i>_rendered.jpg + cập nhật render_plan (kèm style) — KHÔNG OCR/dịch
   → "Lưu tất cả & Xem kết quả" → /translate-result/<sid> (translate.html, ảnh mới)
~~~

### 1.5 Quy ước chung
- **Dirty**: mọi thay đổi (text, bbox, style, erase) đánh dấu ảnh dirty — viền cam nét đứt + badge thumb, tới khi render thành công (kế thừa V2 F4).
- **Monotonic erase**: vùng xoá (delete block, rect, brush) **không bao giờ bị gỡ khỏi danh sách xoá** — kể cả khi undo (undo chỉ khôi phục preview, xem F4.5). Server tự accumulate (đã có ở V2 /re-render-image), client gửi lại toàn bộ vùng của mình.
- **Preview vs Render**: canvas preview = ảnh erased + text vẽ live (khớp tới ~1–2px, xem R1). Ảnh render cuối do server tạo (erase thật + render thật) là **nguồn chân lý** cho translate.html. Editor V3 **luôn** giữ nền erased + layer text live (không chuyển sang ảnh baked) để tiếp tục chỉnh sửa.

---

## 2. Surface map (theo mode Operate)

### 2.1 Màn hình A — correction.html (chế độ OCR/pre-render) — thay đổi tối thiểu
| Vùng | ID phần tử | Thay đổi |
|---|---|---|
| Footer | #btn-continue "✅ Tiếp tục dịch & Render" | Giữ nguyên văn bản. Hành vi đổi: submit → POST /styleditor-prepare (không còn /continue-translate). Chỉ đổi **khi session có ≥1 block có text**; nếu không có text → vẫn qua translate.html (không lỗi) |
| Footer | (không đổi) | Không thêm nút mới ở màn hình A |

### 2.2 Màn hình B — Style editor V3 (MỚI; tái sử dụng correction.html + correction.js, mode="styleditor")
Đây là **cùng template correction.html**, bật bởi `window.CORRECTION_DATA.mode = "styleditor"`; mỗi trang load **1 ảnh** (?img=<idx>), sidebar liệt kê **tất cả ảnh** (thumb = ảnh erased) để chuyển trang.

| Phần tử | Khác biệt so với mode preview/postrender |
|---|---|
| Nền canvas | Ảnh **đã xoá text gốc** (page_<i>_erased.jpg), không phải ảnh gốc, không phải ảnh đã render |
| Block hiển thị | **Text dịch vẽ live** với style của block (font thật, cỡ, màu, đậm/nghiêng, căn lề). Block có translated rỗng → chip "—" nét đứt (render sẽ bỏ qua). Block dirty = viền cam nét đứt; sạch = viền xanh #00e676 |
| Toolbar | Nhóm mới **"Xoá nền"**: `#tool-erase-rect` (▭ Rect) + `#tool-erase-brush` (🖌 Cọ) + `#brush-size` (select 4/12/24px). **Ẩn**: Thêm, Reset. Giữ: Chọn, Xoá block, Undo, Redo, Zoom, Pan |
| Editor panel | Textarea "Bản dịch" (như postrender) + 4 ô toạ độ + nudge + W×H + **khối style mới** (mục 2.3) + nút Xoá block. Không có Clean/OCR lại |
| Footer | `#btn-back-correction` "✏️ Quay lại chỉnh OCR" (→ /correction/<sid>) · `#btn-cancel` "↩ Về kết quả" (→ /translate-result/<sid>) · `#btn-render-one` "🔄 Render ảnh này" · `#btn-save-all` "✅ Lưu tất cả & Xem kết quả" (render các ảnh dirty → translate.html) |
| Sidebar | Thumb tất cả ảnh (ảnh erased); click → /styleditor/<sid>?img=<n> (reload trang). Badge cam "chưa render" khi dirty (P0 cho ảnh hiện tại; P1 cho toàn session qua localStorage) |
| Hint bar | #corr-hints: theo tool (xoá nền rect/brush, select) — văn bản tiếng Việt, mục 3.4 |

### 2.3 Panel style per-block (trong #block-properties, mode styleditor)
Khi chọn block, thêm khối `#style-group` (sau prop-meta, trước prop-actions):
| Control | ID | Giá trị / trạng thái |
|---|---|---|
| Font | `#style-font` (select) | Danh sách từ GET /api/fonts; mặc định = session selected_font; mỗi option label = tên font |
| Cỡ chữ | `#style-size` (input number, 0–120) + checkbox `#style-size-auto` (mặc định ON) | Auto ON → cỡ tự động (V2); OFF → số px cố định (server shrink-to-fit, F5.3) |
| Màu chữ | `#style-color` (input type=color) + 8 swatch preset `#style-swatches` | Mặc định: "Tự động" (appearance V2: đen/trắng theo nền). Swatch: #000000, #ffffff, #5E1675, #E53935, #1E88E5, #43A047, #FDD835, #8E24AA |
| Đậm | `#style-bold` (toggle button "B") | aria-pressed; synthesize client+server |
| Nghiêng | `#style-italic` (toggle button "I") | aria-pressed; synthesize client+server |
| Căn lề | `#style-align` (3 nút: Trái `#align-left` / Giữa `#align-center` / Phải `#align-right`) | aria-pressed; mặc định Giữa |
| Nút phụ (P1) | `#style-apply-all` "Áp dụng cho tất cả block ảnh này" | copy style hiện tại sang mọi block của ảnh |

---

## 3. Feature spec

### F1 — Luồng correction → style editor (P0)
**Hành vi:**
1. correction.html bấm "Tiếp tục dịch & Render" → POST /styleditor-prepare (payload giống /continue-translate cũ: session_id + modified_blocks).
2. Server: rebuild all_ocr_results (tách hàm dùng chung với continue_translate) → **dịch Phase 2** (dùng lại toàn bộ nhánh gemini/copilot/google — tách thành helper `translate_texts_all`) → xoá text gốc mọi bbox (erase_text_region) → ghi v3_draft + page_*_erased.jpg → redirect /styleditor/<sid>?img=0.
3. /styleditor/<sid>?img=<i>: load ảnh erased + blocks (text/translated/bbox/style từ draft; style mặc định nếu thiếu). Session không tồn tại → redirect "/"; không có v3_draft → redirect /correction/<sid> (fallback, risk R6).
4. "Quay lại chỉnh OCR" → /correction/<sid> (dữ liệu OCR gốc bất biến — sửa ở editor không ảnh hưởng correction).
5. Lỗi dịch (translator exception): vẫn vào editor với translated = text gốc + toast cảnh báo "⚠️ Không dịch được, đang hiển thị text gốc" (giữ hành vi V2 last_warning).

**Acceptance:**
- A1.1: [P0] Từ correction.html (có ≥1 block text) bấm "Tiếp tục dịch & Render" → mở style editor (URL /styleditor/<sid>?img=0), không phải translate.html.
- A1.2: [P0] Nền canvas = ảnh gốc đã xoá toàn bộ text trong bbox (không còn chữ gốc; kiểm tra vùng bbox đầu tiên).
- A1.3: [P0] Bản dịch của từng block hiện sẵn trên ảnh đúng vị trí bbox.
- A1.4: [P0] Vào editor KHÔNG gọi render lại: log server không có "Phase 3"/render khi prepare (chỉ Phase 2 dịch).
- A1.5: [P0] "Quay lại chỉnh OCR" → correction.html mở đúng session, blocks OCR nguyên trạng.
- A1.6: [P0] Block có translated rỗng → canvas hiện chip "—" nét đứt; render bỏ qua block đó (không lỗi, không crash).
- A1.7: [P0] Session không có v3_draft (vào thẳng /styleditor) → redirect /correction/<sid> không 500.
- A1.8: [P1] Không có block nào có text → prepare trả translate.html (ảnh gốc) như V2, không lỗi.

### F2 — Nền ảnh đã xoá text (P0)
**Hành vi:**
1. Prepare: với mỗi ảnh, copy ảnh gốc rồi erase_text_region theo từng bbox (thứ tự như render_image_with_blocks: từng block trước, không cần extra regions vì chưa render lần nào) → lưu page_<i>_erased.jpg (q92) + dùng làm canvas.
2. Khi render lần đầu từ editor: extra_erase_regions = vùng xoá của user (rect/brush/delete) — bbox gốc đã nằm trong nền erased, nhưng render_image_with_blocks vẫn xoá lại (idempotent, R1 V2).
3. erase_regions lưu trong render_plan[i].erase_regions **không bao giờ co lại**: server merge bbox gốc + mọi vùng xoá user (kế thừa V2, giữ nguyên logic `merged_erase_regions`).

**Acceptance:**
- A2.1: [P0] Mỗi ảnh có page_<i>_erased.jpg (JPEG q92) khi mở editor; ảnh xem được, đúng kích thước ảnh gốc.
- A2.2: [P0] Vùng bbox OCR không còn chữ gốc trên nền erased (so sánh pixel vùng bbox trước/sau).
- A2.3: [P0] Nền erased = nền ảnh render lần đầu ở vùng ngoài bbox (sai khác ≤ q92 artifacts).
- A2.4: [P0] Render 2 lần liên tiếp không đổi kết quả ở vùng đã erase (idempotent).

### F3 — Text dịch editable + WYSIWYG live (P0; kế thừa V2 move/resize/nudge)
**Hành vi:**
1. Canvas vẽ text dịch của từng block **live** với style (không chỉ vẽ khung) — hàm `drawStyledBlocks()` (mục F6).
2. Sửa textarea "Bản dịch" → block.text/translated cập nhật + vẽ lại ngay + dirty (kế thừa V2).
3. Kéo / resize 8 handle / nudge → bbox đổi + vẽ lại text ngay (kế thừa F2/F3 V2, không thay đổi).
4. Undo/redo snapshot mở rộng {text, translated, bbox, style, deletedRegions, eraseRegions} (F9).

**Acceptance:**
- A3.1: [P0] Chọn block → text dịch vẽ live trên ảnh (không chỉ khung bbox).
- A3.2: [P0] Gõ textarea → canvas cập nhật từng ký tự, block chuyển dirty (viền cam đứt).
- A3.3: [P0] Kéo block → text vẽ lại đúng vị trí mới ngay trong lúc kéo; thả = 1 undo step (kế thừa A3.1 V2).
- A3.4: [P0] Resize block → text re-wrap theo bbox mới ngay (cùng luật wrap, F6.2).
- A3.5: [P0] Ctrl+Z sau khi sửa text → text cũ + bbox cũ + dirty khôi phục đúng (1 bước).

### F4 — Công cụ xoá nền: brush + rect (P0), mask (P1)
**Hành vi chung:** 2 tool trong nhóm "Xoá nền" (toolbar). Khi active: hint bar + con trỏ crosshair; con trỏ cọ = vòng tròn hiển thị đường kính (brush). Thao tác vẽ trực tiếp lên layer xoá (offscreen canvas `eraseLayer` cùng kích thước ảnh), canvas chính composite: erased bg → áp layer xoá (vùng xoá hiện ra ngay). **Preview xoá = flat fill màu nền lấy mẫu từ biên vùng (approximation); render thật = server inpaint (R3).**

1. **Rect** (`#tool-erase-rect`): kéo chuột → vẽ rect trên eraseLayer; pointerup (kích thước ≥ 4×4 px ảnh) → thêm `[x1,y1,x2,y2]` (clamp mép ảnh) vào `img.eraseRegions` + 1 undo step.
2. **Brush** (`#tool-erase-brush` + `#brush-size` 4/12/24px): vẽ tự do; stroke = chuỗi điểm; pointerup → thêm **bbox của stroke** (clamp, ≥2px) vào `img.eraseRegions` + 1 undo step; đồng thời vẽ mask stroke vào `eraseLayer` (P0: vẽ tròn đặc; P1: mềm cạnh).
3. **Monotonic**: `img.eraseRegions` chỉ thêm, không bao giờ xoá phần tử (kể cả undo — F4.5); server tự accumulate thêm một lần nữa (V2 logic) → bất biến. Khi reload editor, client nạp lại eraseRegions từ render_plan[i] (mục 4.2 MERGE RULE) để preview khôi phục đúng vùng đã xoá.
4. **Undo erase**: khôi phục preview (vẽ lại eraseLayer từ danh sách vùng còn lại) **nhưng vẫn giữ region trong eraseRegions** → re-render vùng đó vẫn sạch. Hint bar ghi rõ: "↩ Undo xoá nền chỉ khôi phục hiển thị; vùng đã xoá sẽ được render sạch (an toàn hơn)".
5. **Delete block** (D/tool Xoá): thêm bbox block vào eraseRegions (hợp nhất với deletedRegions V2 về 1 danh sách `img.eraseRegions`).
6. **Mask (P1)**: ngoài rects, client gửi `erase_mask` (PNG b64, grayscale, white=erase, downscale cạnh dài ≤ 2048px) — server inpaint theo mask (F7).
7. **[CHỐT CAPTAIN mục 8.2 — brush & screentone]**: brush/rect chỉ MỞ RỘNG erase region/mask; vùng nào được mask phủ thì server inpaint đúng vùng đó (monotonic, không co lại, idempotent từ ảnh gốc). **Không heuristic riêng cho screentone** — vẽ đâu xoá đó; screentone do inpaint tự xử lý.

**Acceptance:**
- A4.1: [P0] Chọn Rect, kéo 1 vùng trên ảnh → vùng đó biến mất khỏi preview ngay (fill nền); thả chuột → 1 undo step; vùng xuất hiện trong eraseRegions.
- A4.2: [P0] Chọn Cọ (4px), vẽ nguệch ngoạc → vệt cọ mất text ngay trên preview; pointerup → 1 undo step; bbox stroke có trong eraseRegions.
- A4.3: [P0] Đổi `#brush-size` 12/24px → vệt cọ to tương ứng (con trỏ vòng tròn đúng cỡ).
- A4.4: [P0] Sau "Render ảnh này": vùng rect/brush đã xoá **sạch hẳn** (không còn text gốc/screentone) trên ảnh kết quả (kiểm tra pixel vùng).
- A4.5: [P0] Vẽ xoá → Undo → preview khôi phục nhưng **re-render vẫn xoá vùng đó** (monotonic; kiểm tra pixel sau render).
- A4.6: [P0] Rect/brush kéo ra ngoài mép ảnh → clamp vào mép, không lỗi; stroke < 2px bỏ qua.
- A4.7: [P0] Delete block → bbox của nó nằm trong eraseRegions; render xong vùng sạch (kế thừa V2 A4.4).
- A4.8: [P1] Gửi erase_mask → server inpaint theo mask (vùng tròn cọ sạch đúng hình stroke, không phải hình chữ nhật).
- A4.9: [P1] Rect nhỏ hơn 4×4 px → không thêm region (chống vô tình xoá).
- A4.10: [P0] Vẽ rect/brush xoá → Render → tải lại editor (?img=<i>) → các vùng đã xoá vẫn hiện đã xoá trên preview (eraseRegions nạp lại từ render_plan[i] — mục 4.2 MERGE RULE; không mất trạng thái xoá khi reload).

### F5 — Panel style per-block (P0)
**Hành vi:**
1. Style của block = `{font, font_size, text_color, bold, italic, align}`; mặc định: font = session selected_font, font_size = 0 (Tự động), text_color = null (Tự động theo appearance), bold/italic = false, align = "center".
2. Đổi bất kỳ style nào → cập nhật block.style + vẽ lại ngay + dirty (nếu giá trị thực sự khác trước).
3. **Font**: select từ GET /api/fonts; trình duyệt tải TTF qua FontFace (F8) trước khi vẽ; đang tải → giữ font cũ + spinner nhỏ ở label (P1).
4. **Cỡ chữ**: Auto = giữ nguyên hành vi V2 (binary search 12–60px theo bbox). Số px cố định (8–120, clamp): server **shrink-to-fit** — dùng cỡ yêu cầu nếu wrap vừa bbox; nếu không vừa, giảm dần tới cỡ lớn nhất vừa (tối thiểu 12px). Client preview mô phỏng đúng luật (F6.2).
5. **Màu**: null → màu từ appearance V2 (tự động theo nền); hex → ép dùng màu đó (bỏ appearance.text_color). Swatch bấm 1 chạm; input color cho tuỳ biến.
6. **Đậm**: client canvas stroke quanh glyph + fillText; server PIL `stroke_width=2, stroke_fill=text_color` (không đụng outline V2 — nếu appearance.need_outline vẫn vẽ outline như cũ, bold = thêm stroke đồng màu bên dưới).
7. **Nghiêng**: client ctx.transform shear 0.18; server PIL: vẽ từng dòng vào strip RGBA rồi affine shear (data=(1,0.18,0,0,1,0)) paste về vị trí cũ.
8. **Căn lề**: trái = x1 + padding; giữa = hiện tại; phải = x2 − padding − chiều rộng dòng (padding = PADDING_RATIO 0.12 × bbox).
9. Style đi kèm blocks khi render (F7) và persist vào render_plan.blocks[].style (mở lại editor → style giữ nguyên).
10. (P1) "Áp dụng cho tất cả" — copy style hiện tại sang mọi block của ảnh (1 undo step chung).

**Acceptance:**
- A5.1: [P0] Chọn block → panel hiện đủ: Font, Cỡ (Auto ON mặc định), Màu (swatch + picker), B, I, Căn lề 3 nút — đúng giá trị block.
- A5.2: [P0] Đổi Font → text trên canvas vẽ lại bằng font đó ngay (đúng glyph, so sánh với ảnh render sau khi render — glyph giống nhau), dirty.
- A5.3: [P0] Tắt Auto, gõ cỡ 40 → text vẽ đúng 40px trên canvas; render xong ảnh cũng đúng 40px (đo pixel/so sánh vùng).
- A5.4: [P0] Gõ cỡ quá to so với bbox → cả preview lẫn render tự thu nhỏ tới cỡ vừa (shrink-to-fit), không tràn bbox.
- A5.5: [P0] Đổi màu swatch/picker → text đổi màu ngay trên canvas + đúng màu sau render.
- A5.6: [P0] Bật B → text đậm hơn trên canvas + render (so sánh độ dày glyph), tắt lại → về thường.
- A5.7: [P0] Bật I → text nghiêng trên canvas + render; tắt → thẳng.
- A5.8: [P0] Căn Trái/Giữa/Phải → dòng text dịch chuyển đúng trong bbox (cả preview lẫn render).
- A5.9: [P0] Block mới/chưa sửa → style mặc định đúng (font session, auto size, auto color, giữa); render = hành vi V2 (không đổi kết quả so với pipeline cũ cho block chưa style).
- A5.10: [P0] Render → tải lại editor (?img=<i>) → style vẫn giữ (persist qua render_plan; /styleditor merge render_plan[i] làm nguồn chân lý khi ảnh đã render — mục 4.2 MERGE RULE).
- A5.11: [P1] "Áp dụng cho tất cả" → mọi block ảnh đó cùng style, 1 undo step khôi phục cả lô.

### F6 — WYSIWYG preview canvas (P0)
**Hành vi:**
1. `drawStyledBlocks()`: với mỗi block có translated: tải font (FontFace, F8) → vẽ trong bbox: wrap theo luật giống server (F6.2), line-height 1.3, căn lề, đậm/nghiêng, màu; outline V2 nếu style.color = auto và appearance cần (client ước lượng: nếu bbox trung bình tối → vẽ viền đen mỏng — P1; P0 chỉ vẽ màu).
2. **Thuật toán wrap client = mirror server**: CJK (không space) → wrap ký tự; Latin → wrap từ; measure bằng `ctx.measureText` với font thật; usable = bbox × (1 − 2×0.12); auto-size: nhị phân 12→60 (client) / cố định (F5.4).
3. Vẽ lại toàn bộ block khi: load ảnh, đổi style, đổi text, đổi bbox, undo/redo.
4. Zoom 25%–800% giữ chất lượng (vẽ theo toạ độ ảnh, canvas full-res — cơ chế V2 giữ nguyên).

**Acceptance:**
- A6.1: [P0] Text hiển thị đúng font (so glyph với ảnh render sau khi render ở cùng tham số — sai khác ≤ 2px về vị trí dòng).
- A6.2: [P0] Text dài tự xuống dòng trong bbox giống render (cùng số dòng với ảnh render cho cùng text/bbox/style).
- A6.3: [P0] Zoom 25% / 100% / 400% → text không vỡ, không nhòe bất thường (re-draw).
- A6.4: [P1] Cỡ cố định không vừa bbox → hiện cảnh báo nhỏ "⚠️ Cỡ chữ sẽ được thu nhỏ cho vừa" (toast hoặc badge trên panel).

### F7 — Re-render không OCR/dịch (P0 — mở rộng /re-render-image)
**Hành vi:**
1. Payload /re-render-image mở rộng:
   - `blocks_json`: mỗi block thêm field không bắt buộc `"style"` (schema F5).
   - `erase_regions_json` (mới, chuẩn chính): rects của user (rect/brush/delete). Giữ `deleted_regions_json` làm **alias** (tương thích client V2).
   - `erase_mask` (P1): PNG b64 grayscale (white = erase), cạnh dài ≤ 2048.
2. Server (rerender_image):
   - Load ảnh gốc → render_image_with_blocks(blocks kèm style, extra_erase_regions = erase_regions entry hiện có + erase_regions_json + mask).
   - erase_regions accumulate **không co lại** (giữ logic V2; thêm merge vùng mới).
   - style từng block normalize: font ∈ danh sách fonts/ (không hợp lệ → session selected_font), font_size clamp 0–120, color hex hợp lệ (khác → null), align ∈ {left,center,right}, bold/italic bool.
   - Ghi page_<i>_rendered.jpg + render_plan (kèm style) → trả {name, data, blocks(kèm style)}.
3. Không gọi ChromeLensOCR, không gọi translator (contract cứng — giữ nguyên; kiểm chứng log + thời gian).
4. "Lưu tất cả & Xem kết quả" **[CHỐT CAPTAIN mục 8.3 — toast i/n]**: với mỗi ảnh dirty → gọi tuần tự /re-render-image (payload hiện tại của ảnh đó); trong lúc chạy: nút bị VÔ HIỆU HOÁ (double-submit guard) + toast tiến độ "⏳ Đang render ảnh i/n"; mỗi ảnh xong → cập nhật badge hết dirty cho ảnh đó; xong hết → redirect /translate-result/<sid> (tránh lệ thuộc /re-render-all vì nó đọc plan cũ thiếu vùng xoá của ảnh khác; P1: thêm variant re-render-all nhận blocks+erase từng ảnh). Không cần progress bar phức tạp.

**Acceptance:**
- A7.1: [P0] Render 1 ảnh với style + erase → ảnh mới khớp preview WYSIWYG (vùng chữ: nội dung, vị trí, màu, cỡ, căn lề; sai số vị trí ≤ 3px).
- A7.2: [P0] Log server không có "Phase 1/Phase 2" khi gọi /re-render-image (không OCR, không dịch).
- A7.3: [P0] Render ảnh 1 → hash ảnh khác (page_<j>_rendered.jpg, j≠1) không đổi.
- A7.4: [P0] Render → xoá thêm vùng → render lại → vùng xoá lần 1 vẫn sạch (accumulate, không co lại).
- A7.5: [P0] Bấm Render 2 lần nhanh → 1 request (nút khoá, isBusy giữ).
- A7.6: [P0] Bbox invalid → 422 + toast + bôi đỏ ô toạ độ (kế thừa V2); session mất → 404 → toast + redirect "/" sau 2s.
- A7.7: [P0] "Lưu tất cả" → translate.html hiển thị ảnh mới cho mọi ảnh dirty; F5 browser → vẫn mới (persist).
- A7.8: [P0] blocks gửi lên thiếu style → server dùng style cũ trong plan (nếu có) / mặc định — không lỗi (backward compat với client V2 cũ).
- A7.9: [P1] erase_mask được áp dụng (vùng mask sạch sau render).
- A7.10: [P0] "Lưu tất cả" với 2 ảnh dirty → toast "⏳ Đang render ảnh 1/2" rồi "…2/2", nút bị khoá trong lúc chạy; ảnh nào xong → badge dirty tắt ngay; xong hết → redirect /translate-result/<sid> (chốt captain 8.3).

### F8 — Fonts API + FontFace (P0)
**Hành vi:**
1. `GET /api/fonts` → {"fonts": [{name, label}]}: quét thư mục fonts/ (đuôi .ttf); map: animeace_i.ttf → "animeace_"/"Animeace", ariali.ttf → "arial"/"Arial", mangati.ttf → "mangat"/"Mangat", Yuki-*.ttf → tên không đuôi (label = name). Sắp xếp: 3 font cơ bản trước, Yuki theo chữ cái.
2. `GET /font-file/<name>`: serve file TTF (mimetype "font/ttf", inline). **Chống path traversal**: name phải khớp chính xác một entry của /api/fonts (không nhận đường dẫn, không "../").
3. Client: tải /api/fonts lúc init; `new FontFace(name, "url(/font-file/<name>)")` + document.fonts.add; lazy-load theo font của block đang vẽ (cache). Thất bại → fallback "sans-serif" + toast "⚠️ Không tải được font X" (không chặn chỉnh sửa).
4. Preview chọn font: option select hiển thị tên; (P1) ô preview nhỏ vẽ chữ mẫu "Aa" bằng font đó.

**Acceptance:**
- A8.1: [P0] GET /api/fonts trả đủ 23+ font trong fonts/ (3 cơ bản + Yuki-*), đúng name/label, không chứa file không phải TTF.
- A8.2: [P0] GET /font-file/Yuki-Burobu → 200 font/ttf, nội dung = file; `/font-file/..%2Fapp.py` → 404/400, không đọc file ngoài fonts/.
- A8.3: [P0] Canvas vẽ bằng font thật sau khi FontFace load (đo width khác sans-serif cho font khác biệt).
- A8.4: [P0] Font lỗi tải (đổi tên file tạm) → fallback sans-serif + toast cảnh báo, editor vẫn dùng được.
- A8.5: [P1] Option "Aa" preview trong select hiển thị đúng font.

### F9 — Undo/Redo mở rộng (P0)
**Hành vi:**
1. Snapshot: {imageIdx, images: [{blocks: [{text, translated, bbox, style}], deletedRegions, eraseRegions}]}. eraseLayer phục hồi: vì eraseRegions monotonic, undo erase chỉ cần **redraw eraseLayer từ danh sách vùng (không gồm vùng vừa undo)** — không cần snapshot pixel.
2. Giữ MAX_UNDO 60, Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y, guard INPUT/TEXTAREA.
3. Undo sau render: block về dirty (cần render lại) — giữ nguyên tắc V2 (F6 V2).

**Acceptance:**
- A9.1: [P0] Chuỗi "đổi style → vẽ rect xoá → sửa text → Undo×3" khôi phục đúng từng bước (style, vùng xoá preview, text).
- A9.2: [P0] Redo sau undo → trạng thái khôi phục đúng (kể cả style).
- A9.3: [P0] Undo erase khôi phục preview nhưng eraseRegions vẫn chứa vùng đó (A4.5).
- A9.4: [P0] 60 bước tối đa (giữ).

### F10 — Loading / Error / Trạng thái (P0)
| Tình huống | Hành vi |
|---|---|
| Prepare đang chạy | Nút "Tiếp tục dịch & Render" → disabled + "⏳ Đang dịch…"; (P1: thanh tiến trình qua socketio progress như index) |
| Dịch lỗi | Vào editor với text gốc + toast cảnh báo (F1.5) |
| Đang render 1 ảnh | Nút Render → disabled + "⏳ Đang render…"; isBusy chặn sửa (kế thừa) |
| Đang Lưu tất cả (nhiều ảnh) | Nút bị khoá + toast "⏳ Đang render ảnh i/n" (chốt captain 8.3); mỗi ảnh xong → badge hết dirty; xong hết → redirect kết quả |
| Render lỗi mạng/5xx | Toast đỏ + "Thử lại" (giữ payload) |
| Session 404 | Toast "Phiên hết hạn" → redirect "/" sau 2s |
| Bbox invalid | 422 + bôi đỏ ô toạ độ (kế thừa) |

**Acceptance:**
- A10.1: [P0] Click "Tiếp tục dịch & Render" 2 lần nhanh → 1 request (nút bị khoá).
- A10.2: [P0] Tắt mạng → Render → toast lỗi + "Thử lại"; bật mạng → Thử lại thành công.
- A10.3: [P0] Translator lỗi → editor vẫn mở, text gốc hiển thị, toast cảnh báo.
- A10.4: [P0] Xoá thư mục session giữa chừng → mọi action 404 → redirect "/", không exception JS.

### F11 — Bàn phím (P0: giữ map V2 + bổ sung)
| Phím | Hành vi | Trạng thái |
|---|---|---|
| S / D | Chọn / Xoá block | giữ |
| A (Thêm) | **KHÔNG kích hoạt trong styleditor** (chốt captain 8.1 — không thêm block mới) | **tắt** |
| E | Công cụ Xoá nền (bật rect nếu đang select; bấm lần nữa → brush; Esc thoát) | **mới** |
| B / I | Đậm / Nghiêng block đang chọn (khi focus ngoài input) | **mới** |
| L / C / R | Căn trái / giữa / phải (focus ngoài input) | **mới** |
| ←/→/↑/↓ + Shift | Nudge (giữ) / chuyển ảnh khi không chọn | giữ |
| Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y | Undo/Redo | giữ |
| Del | Xoá block chọn (tool select) | giữ |
| Esc | Thoát tool xoá nền → select; bỏ chọn | giữ |
| +/−/0/1, Space+kéo, [ / ], F | Zoom, pan, cycle block, fit | giữ |

**Acceptance:**
- A11.1: [P0] E → tool xoá nền active (rect); E lần nữa → brush; Esc → select. B/I/L/C/R áp style cho block chọn, canvas cập nhật ngay.
- A11.2: [P0] Gõ B/I trong textarea/input → không kích shortcut (guard giữ).

### F12 — Responsive & Touch (P0)
- ≥1100px: 3 cột (giữ). 700–1100px: sidebar dải ngang; style panel trong drawer phải (giữ cơ chế V2). <700px: toolbar wrap; editor = bottom sheet; brush/rect dùng được bằng ngón tay (vùng bấm ≥44×44 cho nút tool + brush-size).
- Pinch zoom (P1).

**Acceptance:**
- A12.1: [P0] 3 breakpoint hiển thị đúng (DevTools), không tràn ngang, mọi nút bấm được.
- A12.2: [P0] Touch: vẽ cọ/rect bằng ngón tay hoạt động (pointer events), undo được.
- A12.3: [P1] Pinch zoom 2 ngón.

### F13 — Accessibility (P1)
- aria-pressed cho tool mới (E/rect/brush, B/I/align), label cho `#brush-size`, `#style-*` controls.
- Toast role status/alert + aria-live (giữ V2).
- Keyboard-only: "chọn block → đổi style → render" hoàn thành không chuột (A11 + tab order).
- prefers-reduced-motion: tắt animation toast/pan (giữ V2).

**Acceptance:**
- A13.1: [P0] aria-pressed phản ánh đúng trạng thái tool/style; mọi control style có label.
- A13.2: [P0] Hoàn thành "chọn block → đổi font → căn lề → render" chỉ bằng bàn phím.
- A13.3: [P1] Lighthouse a11y ≥ 90 trên trang styleditor (spot-check).

### F14 — Nhận diện & Regression (P0)
**Acceptance:**
- A14.1: [P0] Mọi surface mới giữ #5E1675, Exo 2, tiếng Việt; không vẽ lại style cũ, chỉ thêm class mới.
- A14.2: [P0] Flow không-manual (upload → render thẳng) không đổi (chạy lại 1 ảnh không bật manual).
- A14.3: [P0] Session V2 (render_plan, không style) mở /styleditor (draft không style) không lỗi — style mặc định.
- A14.4: [P0] Spot-check V2: correction preview + postrender không regression (A1–A10 V2 chạy lại 3 case chính: xoá an toàn, resize, post-render edit).

---

## 4. API contract (backend)

### 4.1 POST /styleditor-prepare (mới)
~~~
form-data:
  session_id: str (uuid)
  modified_blocks: '[{"image_idx": 0, "blocks": [{"text": "...", "bbox": [...]}]}]'  // y hệt /continue-translate
Hành vi: rebuild all_ocr_results → translate Phase 2 (helper translate_texts_all)
  → với mỗi ảnh: erase_text_region từng bbox → page_<i>_erased.jpg (q92)
  → ghi session_data["v3_draft"] = {"images": [{name, blocks: [{text, translated, bbox, style(mặc định)}]}]}
  → 302 redirect /styleditor/<sid>?img=0
Lỗi: session không tồn tại → redirect "/"; JSON hỏng → redirect "/"
Không render text, không OCR lại. Nếu không có text nào: redirect translate.html (hành vi V2).
~~~

### 4.2 GET /styleditor/<session_id>?img=<idx> (mới)
~~~
Render correction.html mode="styleditor", images=[{name, data: b64 erased jpeg,
  blocks: [{text, translated, bbox, style}]}], postrender_image_idx=idx,
  all_images=[{name, idx}] cho sidebar.
Không có session → "/"; không có v3_draft → /correction/<sid>; idx ngoài phạm vi → idx=0.

[MERGE RULE — chốt sau review t2/A5.10]: ảnh nào ĐÃ render (render_plan[i] tồn tại)
→ render_plan[i] là NGUỒN CHÂN LÝ cho blocks (text/translated/bbox/style) thay cho
v3_draft (draft chỉ dùng cho ảnh CHƯA render lần nào). Trả thêm:
  erase_regions: render_plan[i].erase_regions (client nạp lại để preview xoá khôi
  phục đúng vùng rect/brush đã xoá — monotonic, không mất khi reload)
  erase_mask: render_plan[i].erase_mask nếu có (P1; client nạp lại layer xoá)
Lý do: /re-render-image persist style+erase vào render_plan; nếu editor chỉ đọc
draft thì reload mất trạng thái đã render (finding t2 → fix t3).
~~~

### 4.3 POST /re-render-image (mở rộng — giữ contract V2 cũ tương thích)
~~~
form-data (cũ + mới):
  session_id, image_idx, blocks_json
  blocks_json[i] thêm: "style": {"font": str, "font_size": int(0=auto), "text_color": "#RRGGBB"|null,
                                  "bold": bool, "italic": bool, "align": "left"|"center"|"right"}  // optional
  erase_regions_json: '[[x1,y1,x2,y2], ...]'   // MỚI — vùng xoá user (rect/brush/delete)
  deleted_regions_json: alias cũ (ưu tiên erase_regions_json nếu cả hai)
  erase_mask: '<b64 PNG grayscale, white=erase, cạnh dài ≤2048>'  // MỚI, P1
200 → { "name", "data": b64 jpeg, "blocks": [{text, translated, bbox, style}] }
400/404/422 → như V2
Side effect: page_<i>_rendered.jpg + render_plan[i].blocks (kèm style) + erase_regions accumulate.
Contract cứng: KHÔNG gọi ChromeLensOCR / translator (giữ V2; verify log).
~~~

### 4.4 GET /api/fonts (mới)
~~~
200 → {"fonts": [{"name": "animeace_", "label": "Animeace"}, ...]}
Nguồn: quét fonts/*.ttf; map tên theo get_font_path (animeace_i.ttf→animeace_, ariali.ttf→arial, mangati.ttf→mangat, Yuki-*.ttf→tên không đuôi).
~~~

### 4.5 GET /font-file/<name> (mới)
~~~
200 font/ttf, inline. Name phải ∈ danh sách /api/fonts (whitelist) — không path traversal.
~~~

### 4.6 Refactor backend (backend-engineer)
- Tách `rebuild_ocr_from_modified_blocks(session_data, modified_blocks)` (dùng chung continue_translate + styleditor_prepare).
- Tách `translate_texts_all(all_texts, translator_obj, translator_type)` khỏi translate_and_render (trả (translated_texts, last_warning)).
- add_text.py: `_compute_font_and_wrap(text, bbox, font_path, style=None)`; `_draw_text_on_pil(..., style=None)`; `render_all_blocks(image, blocks, font_path)` đọc `block.get("style")`; mới `erase_mask_region(image, mask_bgr, source_lang)` (cv2.inpaint TELEA r=3, fallback fill median biên).
- app.py: `normalize_block_style(raw, default_font)` (clamp 0–120, hex validate, enum align); `list_available_fonts()`; `_entry_erase_regions` giữ nguyên; rerender_image merge erase_regions_json + mask.
- /postrender (V2): plan blocks đã có style → trả kèm (correction.js postrender giữ nguyên UI nhưng payload blocks_json truyền style nếu có — P1, tránh mất style khi sửa post-render sau V3).

### 4.7 session.json schema mở rộng (backward compatible)
~~~
v3_draft: {"images": [{name, blocks: [{text, translated, bbox, style}]}]}   // mới
render_plan[i].blocks[j].style: {font, font_size, text_color, bold, italic, align}  // optional
page_<i>_erased.jpg  // mới
(_normalize_render_plan_entry: giữ style nếu hợp lệ, bỏ qua nếu không — session cũ không lỗi)
~~~

---

## 5. Ưu tiên P0 / P1 / P2

### P0 — MVP (phải có trong sprint này)
1. **F1** luồng correction → style editor (prepare + trang editor + fallback).
2. **F2** nền ảnh đã xoá text (page_*_erased.jpg).
3. **F3** text dịch editable + vẽ live (kế thừa V2 move/resize/nudge).
4. **F4** xoá nền: rect + brush (bbox regions, monotonic, undo preview-only) — **mask là P1**.
5. **F5** panel style per-block: font/size/color/bold/italic/align + render đúng.
6. **F6** WYSIWYG preview canvas (FontFace, wrap mirror).
7. **F7** /re-render-image mở rộng (style + erase_regions_json) + Lưu tất cả.
8. **F8** /api/fonts + /font-file + FontFace.
9. **F9** undo/redo mở rộng (style + erase).
10. **F10** loading/error/double-submit.
11. **F11** bàn phím (E, B/I, L/C/R + giữ map).
12. **F12** responsive/touch (không có pinch).
13. **F14** identity + regression.

### P1 — Nên có (làm ngay sau P0 nếu sprint còn dư)
- **F4.8/A4.8** erase_mask (brush chất lượng cao).
- **F5.11** "Áp dụng cho tất cả"; **F6.4** cảnh báo cỡ quá to; **F8.5** preview "Aa".
- **F7.9** re-render-all variant nhận blocks+erase từng ảnh; **A12.3** pinch zoom; **F13** a11y nâng cao (Lighthouse ≥90).
- Style pass-through trong V2 /postrender (4.6).
- Progress bar prepare (socketio progress).

### P2 — Sau này (ghi nhận, không làm sprint này)
- Style theo đoạn text trong block; kiểm soát outline/stroke riêng (màu viền, độ dày).
- Thêm block mới / OCR lại trong style editor; rotate; multi-select; copy/paste.
- Server-side undo render (giữ N bản JPEG/ảnh); history panel UI.
- Font preview grid (chọn font bằng hình); font subsetting.

---

## 6. Acceptance criteria tổng (điều kiện hoàn thành sprint)
1. E2E: upload 2 ảnh → correction → sửa 1 bbox → "Tiếp tục dịch & Render" → editor mở với ảnh đã xoá text + bản dịch hiện sẵn → đổi font + màu + căn lề block 1, vẽ cọ xoá 1 vùng, kéo block 2 → "Render ảnh này" → ảnh khớp preview → "Lưu tất cả & Xem kết quả" → translate.html đúng → F5 browser vẫn đúng.
2. Không gọi OCR/dịch khi re-render (A7.2) và nhanh hơn pipeline đầy đủ (mềm < 15s/ảnh như V2).
3. erase_regions không bao giờ co lại (A4.5, A7.4 pass).
4. Toàn bộ acceptance P0 (A1.1–A14.4) pass manual checklist; P1 theo dư địa.
5. UI giữ nhận diện #5E1675 / Exo 2 / tiếng Việt (A14.1); diff template chỉ thêm phần tử, không vẽ lại style cũ.
6. Không regression: flow không-manual + V2 preview/postrender spot-check (A14.2–A14.4).
7. Backward compatible: session V2 không style mở V3 editor không lỗi; client V2 gọi /re-render-image không style vẫn chạy (A7.8).

## 7. Rủi ro & câu hỏi mở
| # | Rủi ro | Giảm thiểu đã chốt |
|---|---|---|
| R1 | Preview canvas ≠ render server (font metrics, wrap) | Cùng TTF qua FontFace; mirror wrap; sai số ≤ 2–3px chấp nhận; render là nguồn chân lý (F6/F7) |
| R2 | erase_mask phình session (PNG b64 full-res) | Downscale cạnh dài ≤ 2048; P1 mới persist mask; TTL 6h có sẵn |
| R3 | Preview xoá (flat fill) kém chất lượng trên nền phức tạp | Chấp nhận approximation; render thật inpaint (A4.4); ghi chú trong hint |
| R4 | Bold/italic synthesize lệch giữa PIL và canvas | Chấp nhận sai khác nhỏ; kiểm chứng A5.6/A5.7 ở mức glyph tương đương |
| R5 | Cỡ chữ cố định tràn bbox | shrink-to-fit server + preview đồng luật (F5.4/A5.4) |
| R6 | Session cũ / thiếu draft | Fallback /correction/<sid> (F1.7); _normalize_render_plan_entry bỏ qua style hỏng (4.7) |
| R7 | Mất style khi sửa qua V2 postrender sau V3 | Pass-through style trong payload /postrender (4.6, P1) |
| R8 | Undo erase vs monotonic gây bối rối | Hint ghi rõ + acceptance A4.5 chốt hành vi |
| R9 | Prepare chậm với nhiều ảnh (dịch + erase) | Phase 2 đã có batch; erase nhanh; P1 progress bar |

## 8. Câu hỏi mở cho captain (không chặn sprint) — ĐÃ CHỐT
1. **Thêm block mới trong editor**: GIỮ non-goal P2 cho V3 — KHÔNG implement thêm block mới trong sprint này; editor chỉ chỉnh sửa block đã dịch (sửa nội dung, kéo vị trí, resize, style). Ghi rõ trong scope để tránh scope creep.
2. **Brush xoá & screentone**: ĐỒNG Ý theo mask — brush/rect chỉ MỞ RỘNG erase region/mask; vùng nào được mask phủ thì inpaint vùng đó (monotonic, không co lại, idempotent từ ảnh gốc). KHÔNG cần heuristic đặc biệt cho screentone.
3. **"Lưu tất cả" progress**: DÙNG toast tiến độ "Đang render ảnh i/n" + VÔ HIỆU HOÁ nút khi đang chạy (double-submit guard); mỗi ảnh xong cập nhật badge hết dirty. KHÔNG cần progress bar phức tạp.