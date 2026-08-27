# V3 Verification Report — t4 (verifier-v3)

Team: manual-mode-v3 · Task: t4 V3 Verification · Attempt: 84611176-e994-40e2-96c0-5bb33b4e2908
Date: 2026-08-27 · Workspace: F:\duancanhan\newbiew (HEAD 09f9c33)

## 1. Static checks — PASS
- `node --check static/js/correction.js` → OK
- `python -m py_compile app.py add_text.py tests/test_manual_mode_v3.py` → OK

## 2. Targeted pytest — PASS (30/30)
- `python -m pytest tests/test_manual_mode_v3.py -v` → **30 passed in 1.70s**
- Covers: fonts API + traversal 404, normalize_block_style, rebuild_ocr_from_modified_blocks,
  translate_texts_all google/failure, styleditor-prepare (draft+erased, no-text→translate,
  missing session, translator error fallback), styleditor page (render, missing session,
  no-draft→correction, bad img clamp), re-render style persist/reuse/normalize,
  erase_regions_json accumulate monotonic + preferred over alias, erase mask decode/accumulate,
  fixed size + shrink-to-fit, style pixel change, double-submit idempotent, legacy V2 payload,
  re-render-all keeps style, postrender style pass-through.

## 3. Full suite — PASS (no new failures vs V2 baseline)
- Baseline (V2, b919428): 116 passed / 4 failed / 2 skipped — 4 fails = torch ModuleNotFoundError (env).
- Now (`pytest tests translator -q`): **167 passed / 4 failed / 2 skipped** — same 4 torch-env fails, 0 new.

## 4. E2E browser flow (Edge headless CDP, real mouse/keyboard events) — PASS
Server: dedicated Flask instance (port 5021, V3 code) with stdout/stderr captured; seeded fresh
session f0a1b2c3-4d5e-4f6a-8b9c-0d1e2f3a4b5c (post-OCR state, no draft).

### 4.1 Correction → continue → styleditor (A1.1, A1.2, A1.4, A2.x)
- /correction/<sid>: mode=preview, #btn-continue present, canvas present → 200
- Click "Tiếp tục dịch & Render" → POST /styleditor-prepare → **302 → /styleditor/<sid>?img=0** (editor, not translate.html)
- Server log during prepare: `[Phase 2] Translating 6 text segments...` then `[styleditor-prepare] 2 image(s), 6 text(s) translated; warning=False` — **no Phase 3/render** (A1.4 ✓)
- /styleditor/<sid>?img=0: mode=styleditor, 2 images, 4 blocks, canvas 1200×1600
- **A2.1**: page_0_erased.jpg exists, size = original (1200×1600), q92
- **A2.2 (pixel)**: bbox1 (120,150,520,310) dark px: page_0.jpg=834 → page_0_erased.jpg=**0** (original text fully erased)
- **A2.4**: re-render idempotent (pytest test_rerender_double_submit_with_style_is_idempotent + render twice 200)

### 4.2 Erase brush/rect + undo monotonic (A4.x, A7.4) — PASS
- Brush tool (24px) stroke over SFX area via real mouse → `eraseRegions=[[879,685,1015,693]]`, eraseStrokes=1, dirty=true
- Undo (#tool-undo) → preview restored (eraseStrokesPreview=0) **but eraseRegions unchanged** (monotonic, A4.5 ✓)
- Render → `erase_regions_json` sent; session render_plan page1 erase_regions = 4 OCR bboxes + brush + probe regions = **7 regions, all accumulated, none shrunk** (A7.4 ✓)
- Legacy V2 alias (deleted_regions_json only) still accepted → 200 (A7.8 backward compat ✓)

### 4.3 Style panel + WYSIWYG render (A5.x, A7.1) — PASS
- Select block0 (canvas click) → panel shows font/size-auto/color/bold/italic/align values
- Set font=Yuki-Burobu, size=40 (auto off), swatch #E53935, bold, align=left → block0.style = {font:"Yuki-Burobu", font_size:40, text_color:"#e53935", bold:true, align:"left"}, dirty=true
- Render → 200; block style persists in render_plan (A5.10 ✓)
- **Pixel proof**: rendered page_0_rendered.jpg block0 region has **787 red (#E53935) px vs 0 in erased control** (A5.5 ✓); rendered differs from original (abs-diff sum 17.6M, A7.3 ✓)
- Style without style field (block2) → server reuses plan style / default, no error (A7.8 ✓)
- Render plan block with empty translated → skipped in render (A1.6 render-bỏ-qua ✓; chip "—" code path verified in correction.js:529-558)

### 4.4 No OCR/translate on re-render (A7.2) — PASS
- Server stdout (unbuffered, -u) around POST /re-render-image (×3): **zero** "Phase 1"/"Phase 2"/"Phase 3"/"Translating"/OCR lines; only erase/render internals
- Tests: guard_translator monkeypatch fails any test that instantiates translator/ChromeLensOCR during re-render (test_manual_mode_v3.py:95-103)

### 4.5 Save all → results (A7.7) — PASS
- #btn-save-all → /translate-result/<sid> (200, "Manga Translator - Results")

## 5. V2 regression + fallbacks — PASS
- **A1.7**: /styleditor/<sid> without v3_draft → redirect /correction/<sid> (200, no 500)
- **R3 (V2)**: /postrender without render_plan → redirect /correction/<sid>
- **postrender with plan** → mode=postrender, #btn-rerender-one + #btn-save-all present, blocks from plan
- **A14.3**: V2-style session (draft without style fields) → styleditor opens, default style applied
- **A11.1**: keyboard E → erase-rect aria-pressed=true; E → brush; Esc → select
- **A1.6**: block with empty translated renders dashed "—" chip (code + data verified)
- **A8.1**: /api/fonts → 24 fonts (3 base + Yuki-*); traversal test → 404 (pytest)
- Non-manual flow: untouched code paths (translate_and_render/render_all_blocks only gained optional params); full suite green.

## 6. Findings
- None blocking. Note: env has no torch (4 pre-existing lama_inpainter failures, baseline-identical);
  google translator returns original text in this env (no external API) — prepare still completes
  (A10.3 fallback: editor opens with original text + warning path), which the E2E exercised.

## Evidence artifacts
- .agent-teams/manual-mode-v3/verifier-server-5021j.out (phase-marker proof)
- .agent-teams/manual-mode-v3/verifier-targeted-v3.log, verifier-full-v3-fullcmd.log
- debug_outputs/v3verify-*.png (editor, brushed, after-undo, styled, after-render, result)
- temp_sessions/v3_pixel_check.py / v3_pixel_check2.py / v3_e2e_attach.py / v3_phase_probe2.py
