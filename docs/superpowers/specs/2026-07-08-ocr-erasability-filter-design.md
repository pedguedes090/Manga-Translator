# OCR Erasability Filter Design

## Goal

Only automatically translate and erase OCR blocks that can be cleaned safely by
the current image-processing pipeline. Text on speech bubbles or flat backgrounds
should continue through translation and rendering. Text over complex artwork,
heavy texture, hair, clothing, detailed objects, or large SFX should be skipped
so the output image is not damaged.

## Non-Goals

- Do not improve OCR accuracy in this change.
- Do not build a new inpainting model or use external AI cleanup.
- Do not redesign the manual correction UI in this pass.
- Do not remove OCR text from the raw OCR result before it can be logged or
  inspected during debugging.

## Proposed Flow

```text
OCR full image
-> skip obvious OCR artifacts
-> assess erasability for each remaining block
-> keep only safe blocks for translate/render
-> skip risky blocks without modifying the image
```

## Erasability Assessment

Add an `assess_erasability(image, bbox, text=None, source_lang='ja')` helper in
`add_text.py`. It should analyze the block without mutating the image and return
a small structured result:

```python
{
    "safe": bool,
    "reason": str,
    "score": float,
    "analysis": {...},
}
```

The helper should reuse existing local primitives where possible:

- `_analyze_region()` for `bubble_context`, `uniformity`, `intensity_std`, and
  `edge_score`.
- `_decide_text_appearance()` for expected fill/text behavior.
- `_build_text_stroke_mask()` and the existing component filters to estimate
  whether the mask is likely text-only.

## Safe Criteria

A block should be considered safe when the mask checks do not show obvious
border/artwork capture and at least one of these is true:

- It is detected as `in_bubble`.
- It is on a uniform or near-uniform background with low texture.
- It has moderate edge complexity but a text-like mask that does not touch the
  erase-region border.

The assessment should prefer keeping normal dialogue in bubbles, including
slightly textured or compressed bubble interiors, because these are the main use
case.

## Skip Criteria

A block should be skipped when the background is too risky to erase cleanly:

- `bubble_context` is artwork rather than bubble and the region has high
  `edge_score`.
- `intensity_std` indicates heavy texture or complex shading.
- The stroke mask has excessive coverage, suggesting the detector is capturing
  artwork instead of just glyph strokes.
- The mask components touch the erase-region edge in a way that suggests line art
  or bubble borders are being captured.
- The text is likely SFX on complex artwork.

Skipped blocks should not be passed into translation or rendering in the
automatic flow. The original image pixels must remain unchanged for those blocks.

## Integration Points

`app.py` should apply the assessment in `filter_ocr_blocks()` after
`should_skip_ocr_artifact()` and bbox normalization. Safe blocks are returned as
before. Skipped blocks are counted and logged with the reason.

The render-time code may keep its existing defensive checks. The new assessment
is an early gate; it does not replace safety checks inside `erase_text_region()`.

## Logging

Add concise logs for visibility:

```text
[SAFE ERASE] text="..." reason=in_bubble score=0.82
[SKIP ERASE] text="..." reason=complex_artwork edge=61.4 std=44.2
```

Long OCR text should be truncated in logs.

## Testing

Add focused tests around the new helper:

- White speech bubble with black text is safe.
- Flat light background with text is safe.
- Text over a synthetic high-edge artwork background is skipped.
- Existing border-preservation erase tests still pass.

The test suite should prove that safe blocks continue to be processed and risky
blocks are excluded before translation/rendering.
