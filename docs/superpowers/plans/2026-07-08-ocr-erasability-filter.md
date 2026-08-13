# OCR Erasability Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically process only OCR blocks whose original text can be erased cleanly by the current OpenCV/PIL pipeline.

**Architecture:** Add a non-mutating `assess_erasability()` helper in `add_text.py` that reuses existing region analysis and stroke-mask logic. Wire that helper into `app.py` after OCR artifact filtering so only safe blocks proceed into translation/rendering, while risky blocks are counted and logged.

**Tech Stack:** Python, OpenCV (`cv2`), NumPy, PIL, pytest.

## Global Constraints

- Preserve original pixels for skipped OCR blocks.
- Do not add external AI cleanup or new service dependencies.
- Keep render-time erase defenses in place; the new filter is an early gate.
- Keep changes focused to `add_text.py`, `app.py`, and `translator/test_translator.py`.
- Generate local debug/evaluation images under `debug_outputs/erasability_eval/`.

---

### Task 1: Add Erasability Unit Tests

**Files:**
- Modify: `translator/test_translator.py`

**Interfaces:**
- Consumes: `assess_erasability(image, bbox, text=None, source_lang='ja')` from `add_text.py`
- Produces: regression coverage for safe bubble, safe flat background, risky complex artwork, and no image mutation

- [ ] **Step 1: Write failing tests**

Add imports:

```python
from add_text import assess_erasability
```

Add tests:

```python
def test_assess_erasability_accepts_speech_bubble_text():
    image = np.full((140, 180, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (90, 70), (65, 42), 0, 0, 360, (0, 0, 0), 3)
    cv2.putText(image, "HEY", (52, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    result = assess_erasability(image, [48, 48, 125, 82], text="HEY", source_lang="en")

    assert result["safe"] is True
    assert result["reason"] in {"in_bubble", "uniform_background", "text_like_mask"}
    assert result["score"] >= 0.55


def test_assess_erasability_accepts_flat_background_text():
    image = np.full((90, 150, 3), 245, dtype=np.uint8)
    cv2.putText(image, "TEST", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    result = assess_erasability(image, [10, 20, 120, 65], text="TEST", source_lang="en")

    assert result["safe"] is True
    assert result["reason"] in {"uniform_background", "text_like_mask"}
    assert result["score"] >= 0.55


def test_assess_erasability_skips_text_on_complex_artwork():
    image = np.full((120, 180, 3), 180, dtype=np.uint8)
    for x in range(0, 180, 8):
        cv2.line(image, (x, 0), (179 - x // 2, 119), (30 + x % 80, 40, 90), 2)
    for y in range(0, 120, 10):
        cv2.line(image, (0, y), (179, 119 - y // 2), (220, 220 - y % 90, 40), 1)
    cv2.putText(image, "BOOM", (35, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

    result = assess_erasability(image, [25, 35, 145, 82], text="BOOM", source_lang="en")

    assert result["safe"] is False
    assert result["reason"] in {"complex_artwork", "sfx_on_artwork", "excessive_mask"}
    assert result["score"] < 0.55


def test_assess_erasability_does_not_mutate_image():
    image = np.full((90, 150, 3), 245, dtype=np.uint8)
    cv2.putText(image, "TEST", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    original = image.copy()

    assess_erasability(image, [10, 20, 120, 65], text="TEST", source_lang="en")

    assert np.array_equal(image, original)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest translator\test_translator.py -q`
Expected: FAIL because `assess_erasability` is not defined.

---

### Task 2: Implement Erasability Assessment

**Files:**
- Modify: `add_text.py`

**Interfaces:**
- Produces: `assess_erasability(image, bbox, text=None, source_lang='ja') -> dict`
- Uses: `_analyze_region`, `_decide_text_appearance`, `_build_text_stroke_mask`, `_remove_screentone_dots`, `_filter_components_outside_inner`, `_detect_sfx`

- [ ] **Step 1: Implement non-mutating assessment helper**

Add helper near `erase_text_region()`:

```python
def assess_erasability(image, bbox, text=None, source_lang='ja'):
    # Clamp bbox and return skip result for invalid boxes.
    # Analyze region and build a padded ROI stroke mask without modifying image.
    # Remove screentone dots and filter components using inner_rect.
    # Compute mask coverage, edge-touching evidence, bubble/artwork context,
    # SFX status, and a score.
    # Return {"safe": bool, "reason": str, "score": float, "analysis": {...}}.
```

Scoring requirements:

```python
score = 0.0
if bubble_context == 'in_bubble': score += 0.45
if uniformity == 'uniform': score += 0.30
elif uniformity == 'textured' and intensity_std < 28: score += 0.15
if edge_score < 25: score += 0.15
elif edge_score > 55 and bubble_context != 'in_bubble': score -= 0.35
if 0.01 <= mask_coverage <= 0.55: score += 0.15
if mask_coverage > 0.62: score -= 0.35
if sfx_info['is_sfx'] and bubble_context != 'in_bubble' and (edge_score > 35 or intensity_std > 30): score -= 0.40
safe = score >= 0.55
```

Reason requirements:

```python
if invalid: "invalid_bbox"
elif sfx_on_artwork: "sfx_on_artwork"
elif complex artwork: "complex_artwork"
elif excessive mask: "excessive_mask"
elif bubble: "in_bubble"
elif uniform: "uniform_background"
else: "text_like_mask" or "risky_background"
```

- [ ] **Step 2: Run unit tests**

Run: `python -m pytest translator\test_translator.py -q`
Expected: PASS for erasability tests and existing erase tests.

---

### Task 3: Wire Assessment Into OCR Filtering

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: `assess_erasability`
- Modifies: `filter_ocr_blocks(blocks, image_shape, source_lang)` to `filter_ocr_blocks(blocks, image, source_lang)`
- Produces: only safe blocks in `all_ocr_results`

- [ ] **Step 1: Update imports**

Add `assess_erasability` to the `from add_text import (...)` list.

- [ ] **Step 2: Update `filter_ocr_blocks`**

Change signature:

```python
def filter_ocr_blocks(blocks, image, source_lang):
    image_shape = image.shape
```

After bbox normalization, call:

```python
erasability = assess_erasability(image, expanded_bbox, block.get("text", ""), source_lang)
if not erasability.get("safe"):
    skipped += 1
    print(f"  [SKIP ERASE] '{_short_log_text(block.get('text', ''))}' reason={erasability.get('reason')} score={erasability.get('score', 0):.2f}")
    continue
```

For safe blocks, attach metadata:

```python
block["_erasability"] = {
    "reason": erasability.get("reason"),
    "score": erasability.get("score"),
}
```

- [ ] **Step 3: Add log text helper**

Add a private helper in `app.py`:

```python
def _short_log_text(text, max_len=36):
    cleaned = re.sub(r'\s+', ' ', str(text or '')).strip()
    return cleaned if len(cleaned) <= max_len else cleaned[:max_len - 1] + '…'
```

- [ ] **Step 4: Update call site**

Change:

```python
blocks, skipped_artifacts = filter_ocr_blocks(blocks, image.shape, source_lang)
```

to:

```python
blocks, skipped_artifacts = filter_ocr_blocks(blocks, image, source_lang)
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest translator\test_translator.py -q`
Expected: PASS.

---

### Task 4: Generate Evaluation Images

**Files:**
- Create generated outputs under `debug_outputs/erasability_eval/`

**Interfaces:**
- Consumes: `assess_erasability`, `erase_text_region`, `render_all_blocks`
- Produces: before/after/overlay images for manual inspection

- [ ] **Step 1: Generate synthetic cases**

Create a one-off Python script via shell pipeline, not committed, that renders:

- `bubble_safe`
- `flat_safe`
- `complex_artwork_skip`
- `textured_artwork_skip`
- `border_overlap_safe`

For each case, save:

- `<case>_before.png`
- `<case>_after.png`
- `<case>_overlay.png`

Only call `erase_text_region()` and `render_all_blocks()` when `assess_erasability()["safe"]` is true.

- [ ] **Step 2: Generate real sample output**

Use `temp_sessions/274bb997-145e-4494-87ad-dd9b99421b87/page_0.jpg` and its session JSON if present. Save a processed page and per-block log summary under `debug_outputs/erasability_eval/`.

- [ ] **Step 3: Verify visual invariants**

Programmatically print:

- safe/skip decision per synthetic case
- border pixel preservation for `border_overlap_safe`
- changed-pixel counts for skipped cases, expected `0`

---

### Task 5: Final Verification

**Files:**
- Inspect: `add_text.py`, `app.py`, `translator/test_translator.py`, `debug_outputs/erasability_eval/`

**Interfaces:**
- Verifies all previous tasks

- [ ] **Step 1: Run tests**

Run: `python -m pytest translator\test_translator.py -q`
Expected: all tests pass.

- [ ] **Step 2: Inspect git diff**

Run: `git diff -- add_text.py app.py translator/test_translator.py docs/superpowers/plans/2026-07-08-ocr-erasability-filter.md`
Expected: only intended implementation, tests, and plan changes.

- [ ] **Step 3: Report evidence**

Report:

- tests run and results
- evaluation image directory
- safe/skip decisions
- any known threshold tradeoffs
