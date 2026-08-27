# V3 Verification Supplement — A4.10/A5.10 fix (verifier-v3, follow-up)

Backend fix (sync draft sau render + merge render_plan safety net) re-verified.

## 1. Static + tests — PASS
- node --check static/js/correction.js OK; py_compile app.py add_text.py tests/test_manual_mode_v3.py OK
- Targeted: **39 passed** (was 30; +9 reload/persistence tests incl. the 3 new)
  - test_rerender_then_reload_styleditor_keeps_style PASS
  - test_rerender_deleted_block_stays_deleted_in_draft PASS
  - test_rerender_keeps_empty_translated_chip_in_draft PASS
  - test_styleditor_stale_jpeg_without_plan_uses_draft PASS
  - test_styleditor_reload_plan_blocks_are_truth PASS
  - test_styleditor_reload_returns_erase_regions_and_mask PASS
  - test_styleditor_fresh_prepare_session_uses_draft PASS
  - test_styleditor_prepare_clears_old_render_state PASS
  - test_styleditor_unrendered_image_uses_draft_and_empty_erase PASS
- Full suite: **176 passed / 4 failed / 2 skipped** — same 4 torch-env fails (pre-existing), 0 new.

## 2. E2E reload persistence (real browser, Edge CDP) — PASS
Session f0a1b2c3-4d5e-4f6a-8b9c-0d1e2f3a4b5c, server 5021 (captured logs).

### A5.10: style per-block persists after reload
- Set block0 style: font=Yuki-Burobu, size=36 (auto off), color=#1E88E5 (swatch), italic, align=right → dirty=true
- Render (#btn-rerender-one) → 200, dirty=false
- **Reload /styleditor/<sid>?img=0** → block0 style returned EXACTLY:
  {"font":"Yuki-Burobu","font_size":36,"text_color":"#1e88e5","bold":true,"italic":true,"align":"right"} ✓

### A4.10: erase preview persists after reload
- Brush stroke over SFX (image 860,760) → eraseRegions grew to 8, strokes=1, previewRects=7, dirty=true
- Render → 200
- **Reload** → eraseRegions all 8 returned (incl. brush bbox [860,760,964,768]), erase_mask present,
  erasePreviewRects=8 (preview rects rebuilt from server erase_regions) ✓

### A4.7: deleted block does NOT resurrect
- Select block "sayounara" (click center) → #edit-text present → #btn-delete-block click
- blocks: [konnichiwa, sayounara] → [konnichiwa] (sayounara gone), eraseRegions grew (bbox merged into erase set), dirty=true
- Render → 200 → **Reload** → blocks still [konnichiwa] only; eraseRegions kept (monotonic, 8 unique after merge with OCR bboxes) ✓

## 3. Server log (unbuffered) — no OCR/translate during all re-renders
- Only "[Phase 2]" line comes from prepare; re-render POSTs show no Phase markers (as before).

## Evidence
- debug_outputs/v3verify-reload-p1.png / -p2.png / -p3.png (screenshots)
- .agent-teams/manual-mode-v3/verifier-server-5021r.out (reload run), 5021u.err (delete run)
- temp_sessions/v3_reload_check.py, v3_delete_check2.py (drivers)
