# Verifier Onboarding Map + Verification Plan — Manual Mode V2
# Prepared by verifier (turn 1). Baseline captured BEFORE t2/t3 implementation lands.

## 1. Call-flow map (GitNexus-confirmed + source-read)

/correction/<sid> (GET) → correction.html (CORRECTION_DATA = preview images + blocks)
POST /continue-translate → continue_translate() app.py:940
  → rebuild all_ocr_results from modified_blocks (normalize_bbox_for_json, expand_ratio=0)
  → _do_full_pipeline() app.py:858
      → translate_and_render() app.py:478
          → render_single_image (nested) app.py:576-668
              → vision_adapter.process_page OR erase_text_region() [add_text.py]
              → render_all_blocks() [add_text.py:1413]
      → build_result_images() app.py:457 → translate.html
POST /ocr-region → ocr_region() app.py:1014 (OCR 1 vùng, giữ nguyên)

Session persistence: _save_session app.py:175 (JSON + page_<i>.jpg q92, mem-cache ocr_sessions)
                   load_session app.py:226 (mem-cache → disk fallback)
                   cleanup_old_sessions app.py:147 (TTL)

## 2. Regression surface (what the sprint MUST NOT break)
- Non-manual flow: upload_file → _do_full_pipeline (no correction_session_id) — translate.html render straight.
- continue_translate exact-bbox contract: test_continue_translate_keeps_correction_bboxes_exact
- OCR region join order: test_ocr_region_joins_blocks_in_reading_order
- Vision adapter paths: tests/test_app_vision_adapter.py (13 tests)
- Backward compat: old session (no render_plan) opens correction.html normally (spec §6.7)
- R3: /postrender on session without render_plan → redirect to correction.html OCR mode

## 3. Baseline evidence (captured BEFORE sprint changes)
- Git HEAD: b919428 "perf: release CUDA cache after LaMa inference"
- Pre-existing dirty: app.py, add_text.py (M), tests/vision/test_pipeline.py (M), vision/* (M), pycache noise
- Baseline hashes (SHA256): see git-tracked baseline in session logs:
  app.py 81087119... / correction.js 008F4ABE... / correction.html 8F871010... /
  translate.html D64C32BF... / correction.css 45694439... / style.css C2710A87...
- Targeted suite (tests/test_app_vision_adapter.py + translator/test_translator.py): 53 passed, 2 skipped (exit 0)
- Full suite (tests translator): 116 passed, 2 skipped, 4 FAILED pre-existing env issue
  (test_lama_inpainter.py: ModuleNotFoundError 'torch') — NOT sprint-related, exclude from gate.

## 4. Verification checklist (run after t2 + t3 both complete)

A. Static syntax (all changed files):
   - node --check static/js/correction.js (and any new JS files)
   - python -m py_compile app.py add_text.py (byte-compile check)
B. Baseline regression: targeted suite must stay 53 passed / 2 skipped (no NEW failures)
   - .venv\Scripts\python.exe -m pytest tests/test_app_vision_adapter.py translator/test_translator.py -q
C. New backend tests (if backend-engineer adds, e.g. test_re_render*): run them
D. Full suite: .venv\Scripts\python.exe -m pytest tests translator -q
   → expect same 4 torch-env failures, 0 new; compare against baseline 116 passed.
E. API contract spot-checks (Flask test client or live):
   1. GET /postrender/<sid>?img=<idx> — render_plan present → 200 correction.html mode=postrender;
      no render_plan → redirect to /correction/<sid> (R3); bad sid → redirect /
   2. POST /re-render-image — 200 {name,data,blocks}; bad json → 400; bad session → 404; invalid bbox → 422
   3. POST /re-render-all — renders translate.html; dirty list only
   4. No "Phase 1"/"Phase 2" in server logs for re-render endpoints (A5.2)
   5. page_<i>_rendered.jpg written; session.json gains render_plan (A4.5 persist)
F. Spec acceptance A1–A10 (P0 mandatory): per docs/manual-mode-v2-spec.md §6 (v2, 341 lines)
   - A1.1–A1.5 delete safety (A1.5 = Del chỉ khi tool=select, khớp correction.js:489)
   - A2.1–A2.6 resize handles (A2.6 touch: hit-area ≥ 44×44 CSS px vô hình, gần-tâm + corner-trước-edge khi chồng)
   - A3.1–A3.5 move/nudge/clamp | A4.1–A4.6 post-render edit | A5.1–A5.6 re-render
   - A6.1–A6.2 undo/redo | A7.1–A7.4 loading/error | A8.1–A8.2 keyboard
   - A9.1–A9.2 responsive+touch — F9 TRONG SPRINT (captain chốt); A10.1–A10.2 a11y (P1)
   - P0 scope v2: toast lỗi + nút "Thử lại" (F7) = P0; badge dirty thumbnail (F4/2.3) = P0
F-bis. Spec deltas đã đồng bộ (bản 342 dòng) — đều ĐÃ nằm trong checklist trên:
   A2.6 touch hit-zone ≥44×44px + disambiguation; A9.2 touch (F9 TRONG SPRINT);
   F1.2 Del cần tool=select (correction.js:489); toast 'Thử lại' + dirty badge = P0.
   Phương pháp verify touch: DevTools device emulation cho A2.6/A9.2 — PHẢI bật
   'Emulate touch' (Sensors/device toolbar) chứ không chỉ responsive viewport, vì
   pointer-type khác nhau (touch vs mouse) đi nhánh hit-test khác nhau;
   kiểm tra Del KHÔNG xoá khi tool != select (thủ công hoặc đọc guard trong code).
F-ter. F5 refactor contract (§4.4): render_single_image (nested, app.py:576-668) →
   render_image_with_blocks MODULE-LEVEL + param extra_erase_regions; dùng chung cho
   translate_and_render VÀ /re-render-image. Verify: (1) đọc code thấy hàm module-level;
   (2) translate_and_render vẫn gọi qua hàm mới với extra_erase_regions=None — không đổi
   hành vi pipeline; (3) GitNexus detect_changes/context để xác nhận không còn call
   site nào dùng symbol cũ. Regression surface: erase_text_region/render_all_blocks
   (add_text.py) không đổi contract khi bị gọi từ 2 đường.
G. Non-manual flow regression: 1 image, manual_correction OFF → translate.html unchanged behavior
H. Identity: #5E1675, Exo 2, Vietnamese UI retained; diff = additions only (spec §6.5)

## 5. Blocking gate rules for my completion
- FAIL if: any NEW pytest failure, any node --check/py_compile failure, P0 acceptance
  (A1–A9, incl. A2.6 touch + A9.2) evidence absent or contradicted, or non-manual flow regression.
- A10 (a11y) = P1: report findings, không chặn gate trừ khi regression a11y rõ ràng.
- torch ModuleNotFoundError (4 tests) is pre-existing env — exclude, but report.
