# Inpainting and application integration implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route prepared masks through flat fill, Telea, or one full-resolution LaMa pass per page, then reuse prepared results safely in Flask sessions and the CLI.

**Architecture:** Inpainters implement small injectable interfaces and write only inside their masks. `PageInpaintRouter` groups prepared blocks by strategy, unions complex masks for one native-resolution LaMa inference, and falls back through a context-aware retry to Telea when CUDA runs out of memory.

**Tech Stack:** Python 3.10/3.11, NumPy, OpenCV, PyTorch CUDA FP16, Flask, JavaScript, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-full-vision-pipeline-design.md`

## Global constraints

- LaMa runs at native full-page resolution first; do not resize the page in the primary path.
- Process one LaMa page at a time and combine all complex masks into one inference call.
- Every pixel outside the selected method's mask remains byte-identical.
- Retry CUDA out-of-memory once with an adaptive context crop, then use Telea for the affected mask.
- Unsafe or unconfirmed blocks preserve original pixels and do not receive translated rendering.
- Web session JSON stores references and scalar summaries, never NumPy arrays.
- Existing Flask, CLI, OCR, translation, and rendering behavior remains compatible for callers that do not opt into new flags.
- Complete each task in one commit and push directly to `origin/main` after focused and regression tests pass.
- Never force-push or stage unrelated working-tree changes.

## File map

| Path | Responsibility |
| --- | --- |
| `vision/inpainting/base.py` | Inpainter protocol and mask-only compositing helper. |
| `vision/inpainting/flat.py` | Component-local background sampling and fill. |
| `vision/inpainting/opencv.py` | Telea implementation. |
| `vision/inpainting/lama_arch.py` | Attributed Apache-2.0 LaMa generator architecture needed for inference. |
| `vision/inpainting/lama.py` | Full-resolution CUDA inference and native-size output. |
| `vision/inpainting/context.py` | Adaptive OOM fallback crop planning and mapping. |
| `vision/inpainting/router.py` | Page plan, union masks, fallback order, and erase results. |
| `vision/cache.py` | Compressed mask persistence within safe session roots. |
| `app.py` | Prepared-block creation, review state, cache references, and page erase orchestration. |
| `main.py` | Testable parser and the same vision pipeline used by the web flow. |
| `templates/correction.html` | Risk status and explicit erase confirmation control. |
| `static/js/correction.js` | Preserve and submit review metadata. |
| `static/css/correction.css` | Minimal risk-state styling. |

---

### Task 1: Implement mask-only flat fill and Telea inpainters

**Files:**
- Create: `vision/inpainting/__init__.py`
- Create: `vision/inpainting/base.py`
- Create: `vision/inpainting/flat.py`
- Create: `vision/inpainting/opencv.py`
- Create: `tests/vision/test_opencv_inpainters.py`

**Interfaces:**
- Produces: `composite_inside_mask(original, candidate, mask) -> np.ndarray`.
- Produces: `FlatInpainter.inpaint(image, mask, prepared_blocks) -> np.ndarray`.
- Produces: `TeleaInpainter.inpaint(image, mask, prepared_blocks) -> np.ndarray`.

- [ ] **Step 1: Write outside-mask invariance and removal tests**

```python
import cv2
import numpy as np

from vision.inpainting.flat import FlatInpainter
from vision.inpainting.opencv import TeleaInpainter


def assert_outside_mask_unchanged(before, after, mask):
    assert np.array_equal(before[mask == 0], after[mask == 0])


def test_flat_fill_changes_mask_only():
    image = np.full((80, 120, 3), 245, np.uint8)
    image[30:45, 40:80] = 0
    mask = np.zeros((80, 120), np.uint8)
    mask[30:45, 40:80] = 255
    result = FlatInpainter().inpaint(image, mask, [])
    assert_outside_mask_unchanged(image, result, mask)
    assert result[35, 60].mean() > 230


def test_telea_changes_mask_only():
    image = np.full((80, 120, 3), 220, np.uint8)
    cv2.line(image, (0, 40), (119, 40), (20, 20, 20), 2)
    mask = np.zeros((80, 120), np.uint8)
    mask[35:46, 50:70] = 255
    result = TeleaInpainter(radius=3).inpaint(image, mask, [])
    assert_outside_mask_unchanged(image, result, mask)
```

- [ ] **Step 2: Run the tests and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_opencv_inpainters.py -q`

Expected: FAIL because the inpainting package does not exist.

- [ ] **Step 3: Implement component-local fill and Telea**

For each connected mask component, sample a ring formed by dilating the component by five pixels and subtracting a two-pixel dilation; use the median BGR value. Build a complete flat candidate, then composite only within the mask. Telea calls `cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)` and also composites only within the mask.

- [ ] **Step 4: Run focused and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_opencv_inpainters.py translator\test_translator.py -q`

Expected: PASS and outside-mask comparisons are exact.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add vision\inpainting\__init__.py vision\inpainting\base.py vision\inpainting\flat.py vision\inpainting\opencv.py tests\vision\test_opencv_inpainters.py
git commit -m "feat: add mask-only OpenCV inpainters"
git push origin main
```

### Task 2: Add full-resolution LaMa CUDA inference

**Files:**
- Create: `vision/inpainting/lama_arch.py`
- Create: `vision/inpainting/lama.py`
- Create: `tools/import_lama.py`
- Create: `tools/benchmark_inpainting.py`
- Create: `tests/vision/test_lama_inpainter.py`
- Modify: `models/NOTICE.md`
- Modify: `models/manifest.json`

**Interfaces:**
- Produces: `LamaInpainter.inpaint(image, mask, prepared_blocks) -> np.ndarray`.
- Produces: `LamaInpainter.last_peak_vram_bytes: int` and `last_elapsed_ms: float`.
- Produces: `tools/benchmark_inpainting.py` single-image smoke mode; the rollout plan extends it with manifest matrix mode.
- Produces: release asset `lama-big-v1.0.0.pt` converted from the official Apache-2.0 checkpoint.

- [ ] **Step 1: Write native-resolution, single-call, and compositing tests**

```python
import numpy as np

from vision.inpainting.lama import LamaInpainter


class FakeLamaModel:
    def __init__(self):
        self.calls = []

    def __call__(self, image, mask):
        self.calls.append((tuple(image.shape), tuple(mask.shape)))
        return np.ones_like(image) * 0.5


def test_lama_preserves_native_size_and_writes_inside_mask_only():
    image = np.full((73, 119, 3), 200, np.uint8)
    mask = np.zeros((73, 119), np.uint8)
    mask[20:35, 40:70] = 255
    model = FakeLamaModel()
    result = LamaInpainter(model=model, device="cuda", precision="fp16").inpaint(image, mask, [])
    assert result.shape == image.shape
    assert len(model.calls) == 1
    assert np.array_equal(result[mask == 0], image[mask == 0])
```

- [ ] **Step 2: Run the tests and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_lama_inpainter.py -q`

Expected: FAIL because `vision.inpainting.lama` does not exist.

- [ ] **Step 3: Implement padding, FP16 inference, and attribution**

Normalize native BGR to RGB `[0, 1]`, pad image and mask by reflection/zero respectively to a multiple of eight, run one model call under `torch.inference_mode()` and CUDA autocast FP16, unpad without resizing, convert to uint8 BGR, and composite only inside mask. Track CUDA peak memory and wall time. Adapt only the generator/loading code required from the official LaMa project in `lama_arch.py`; retain its Apache-2.0 notice in `models/NOTICE.md`. Add a single-image benchmark command accepting `--image`, `--mask`, `--methods`, `--config`, and `--output` so this task can verify the real model before the rollout plan adds dataset-manifest mode.

- [ ] **Step 4: Import, publish, and register the trusted artifact**

Run:

```powershell
.\.venv\Scripts\python.exe -m gdown --folder https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips --output training_runs\lama-source
.\.venv\Scripts\python.exe tools\import_lama.py --source training_runs\lama-source --output training_runs\lama-big-v1.0.0.pt
gh release upload vision-models-v1.0.0 training_runs\lama-big-v1.0.0.pt --repo pedguedes090/Manga-Translator
.\.venv\Scripts\python.exe tools\register_model.py --manifest models\manifest.json --artifact training_runs\lama-big-v1.0.0.pt --name lama-big --version 1.0.0 --url https://github.com/pedguedes090/Manga-Translator/releases/download/vision-models-v1.0.0/lama-big-v1.0.0.pt --license Apache-2.0 --source advimman-lama-official --input-size 0 --layout NCHW
```

Expected: import tool loads source weights safely, writes inference-only state, and registration computes exact size/hash.

- [ ] **Step 5: Run a CUDA smoke test, regression tests, commit, and push**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\vision\test_lama_inpainter.py translator\test_translator.py -q
.\.venv\Scripts\python.exe tools\benchmark_inpainting.py --image debug_outputs\vision_baseline\image_0000.png --mask debug_outputs\vision_baseline\text_mask_0000.png --methods lama --config configs\vision.json --output reports\vision\lama-smoke.json
```

Expected: tests PASS; smoke report records full native dimensions, elapsed time, and peak VRAM.

```powershell
git fetch origin
git add vision\inpainting\lama_arch.py vision\inpainting\lama.py tools\import_lama.py tools\benchmark_inpainting.py tests\vision\test_lama_inpainter.py models\NOTICE.md models\manifest.json
git commit -m "feat: add full-resolution LaMa inpainting"
git push origin main
```

### Task 3: Retry LaMa out-of-memory with an adaptive context crop

**Files:**
- Create: `vision/inpainting/context.py`
- Create: `tests/vision/test_lama_context_fallback.py`
- Modify: `vision/inpainting/lama.py`

**Interfaces:**
- Produces: `plan_context_crop(mask, image_shape, min_context_px=256, max_mask_ratio=0.08) -> BBox`.
- Produces: `LamaOutOfMemory` for a normalized CUDA OOM signal.
- Changes: `LamaInpainter.inpaint()` retries exactly once with the planned crop.

- [ ] **Step 1: Write crop geometry and retry tests**

```python
import numpy as np

from vision.inpainting.context import plan_context_crop
from vision.inpainting.lama import LamaInpainter, LamaOutOfMemory


def test_context_crop_contains_mask_and_minimum_available_context():
    mask = np.zeros((1200, 1600), np.uint8)
    mask[500:540, 700:760] = 255
    assert plan_context_crop(mask, mask.shape, 256, 0.08) == (444, 244, 1016, 796)


def test_lama_retries_once_after_full_page_oom():
    calls = []

    def model(image, mask):
        calls.append(tuple(image.shape))
        if len(calls) == 1:
            raise LamaOutOfMemory("synthetic OOM")
        return np.zeros_like(image)

    image = np.full((1200, 1600, 3), 200, np.uint8)
    mask = np.zeros((1200, 1600), np.uint8)
    mask[500:540, 700:760] = 255
    result = LamaInpainter(model=model).inpaint(image, mask, [])
    assert len(calls) == 2
    assert result.shape == image.shape
```

- [ ] **Step 2: Run the tests and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_lama_context_fallback.py -q`

Expected: FAIL because context planning and normalized OOM do not exist.

- [ ] **Step 3: Implement one bounded retry**

Find the union bbox of nonzero mask pixels, expand each side by at least 256 pixels, and continue symmetric growth until mask coverage is at most eight percent or image bounds stop growth. Normalize only `torch.cuda.OutOfMemoryError` to `LamaOutOfMemory`, clear CUDA cache, retry once, map candidate pixels back to page coordinates, and re-raise after the second failure for router-level Telea fallback.

- [ ] **Step 4: Run focused and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_lama_context_fallback.py tests\vision\test_lama_inpainter.py translator\test_translator.py -q`

Expected: PASS; fake model call count is exactly two.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add vision\inpainting\context.py vision\inpainting\lama.py tests\vision\test_lama_context_fallback.py
git commit -m "feat: retry LaMa with contextual crop"
git push origin main
```

### Task 4: Route a page through flat, Telea, and one LaMa union mask

**Files:**
- Create: `vision/inpainting/router.py`
- Create: `tests/vision/test_inpaint_router.py`
- Modify: `vision/pipeline.py`

**Interfaces:**
- Produces: `PageInpaintRouter.erase(image, prepared_blocks) -> tuple[np.ndarray, list[EraseResult]]`.
- Changes: `VisionPipeline.erase_page()` delegates to the router.

- [ ] **Step 1: Write grouping, priority, and fallback tests**

```python
from unittest.mock import Mock

import numpy as np

from vision.inpainting.lama import LamaOutOfMemory
from vision.inpainting.router import PageInpaintRouter


def test_router_unions_complex_masks_into_one_lama_call(prepared_block_factory):
    blocks = [
        prepared_block_factory("a", (10, 10, 30, 30), "lama_full_page"),
        prepared_block_factory("b", (60, 60, 80, 80), "lama_full_page"),
    ]
    lama = Mock()
    lama.inpaint.side_effect = lambda image, mask, blocks: image.copy()
    router = PageInpaintRouter(lama=lama)
    router.erase(np.full((100, 100, 3), 200, np.uint8), blocks)
    assert lama.inpaint.call_count == 1
    assert np.count_nonzero(lama.inpaint.call_args.args[1]) == 800


def test_router_falls_back_to_telea_after_lama_retry_fails(prepared_block_factory):
    block = prepared_block_factory("a", (10, 10, 30, 30), "lama_full_page")
    lama = Mock()
    lama.inpaint.side_effect = LamaOutOfMemory("still OOM")
    telea = Mock()
    telea.inpaint.side_effect = lambda image, mask, blocks: image.copy()
    PageInpaintRouter(lama=lama, telea=telea).erase(
        np.full((50, 50, 3), 200, np.uint8), [block],
    )
    assert telea.inpaint.call_count == 1
```

- [ ] **Step 2: Run the test and observe import/fixture failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_inpaint_router.py -q`

Expected: FAIL because router and `prepared_block_factory` do not exist.

- [ ] **Step 3: Implement the fixture and router**

Add the factory to `tests/vision/conftest.py` using the exact dataclasses from plan one. Build non-overlapping strategy masks with priority LaMa, Telea, flat; skip `preserve` and unconfirmed review blocks. Call LaMa once with the complex union. Convert a final LaMa OOM into Telea for that union and attach the warning to each affected `EraseResult`.

- [ ] **Step 4: Run all vision and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision translator\test_translator.py -q`

Expected: PASS; no outside-mask delta occurs in any backend.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add vision\inpainting\router.py vision\pipeline.py tests\vision\conftest.py tests\vision\test_inpaint_router.py
git commit -m "feat: route page inpainting by complexity"
git push origin main
```

### Task 5: Persist compressed masks inside safe session roots

**Files:**
- Create: `vision/cache.py`
- Create: `tests/vision/test_mask_cache.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `MaskCache.save(session_root: Path, prepared: PreparedBlock) -> Path`.
- Produces: `MaskCache.load(session_root: Path, mask_ref: str, prepared_summary: dict) -> MaskResult`.
- Produces: `MaskCache.invalidate(session_root: Path, mask_ref: str) -> None`.

- [ ] **Step 1: Write roundtrip and traversal tests**

```python
import numpy as np
import pytest

from vision.cache import MaskCache


def test_mask_cache_roundtrips_arrays(tmp_path, prepared_block_factory):
    prepared = prepared_block_factory("block-1", (10, 10, 30, 30), "telea")
    ref = MaskCache().save(tmp_path, prepared)
    loaded = MaskCache().load(tmp_path, str(ref.relative_to(tmp_path)), prepared.to_summary())
    assert np.array_equal(loaded.mask, prepared.mask_result.mask)
    assert loaded.backend == prepared.mask_result.backend


def test_mask_cache_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="outside session root"):
        MaskCache().load(tmp_path, "../outside.npz", {})
```

- [ ] **Step 2: Run the test and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_mask_cache.py -q`

Expected: FAIL because `vision.cache` does not exist.

- [ ] **Step 3: Implement compressed persistence and safe resolution**

Store mask, optional probability, optional bubble mask, ROI bbox, and scalar metadata with `np.savez_compressed()`. Resolve references with `Path.resolve()` and require the result to remain under `<session_root>/vision`. Write to a temporary NPZ and replace atomically. Add `temp_sessions/*/vision/` to ignore rules.

- [ ] **Step 4: Run focused and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_mask_cache.py translator\test_translator.py -q`

Expected: PASS and traversal is rejected.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add .gitignore vision\cache.py tests\vision\test_mask_cache.py
git commit -m "feat: cache prepared masks per session"
git push origin main
```

### Task 6: Integrate prepared blocks into Flask automatic rendering

**Files:**
- Create: `tests/vision/test_app_vision_integration.py`
- Modify: `app.py:343-633`
- Modify: `app.py:807-959`

**Interfaces:**
- Produces: `prepare_ocr_blocks(blocks, image, source_lang, pipeline, include_review) -> tuple[list[dict], int]`.
- Preserves: `filter_ocr_blocks(blocks, image, source_lang)` public behavior for legacy callers.
- Changes: `translate_and_render()` loads cached `PreparedBlock` state and calls `erase_page()` once per page.

- [ ] **Step 1: Write automatic-flow call-count and preservation tests**

```python
from unittest.mock import Mock

import numpy as np

import app as app_module


def test_flask_render_erases_page_once_with_prepared_blocks(monkeypatch, prepared_block_factory):
    pipeline = Mock()
    prepared = prepared_block_factory("block-1", (10, 10, 30, 30), "flat")
    pipeline.prepare_page.return_value = [prepared]
    pipeline.erase_page.return_value = (np.full((40, 40, 3), 245, np.uint8), [])
    monkeypatch.setattr(app_module, "get_vision_pipeline", lambda: pipeline)
    blocks, skipped = app_module.prepare_ocr_blocks(
        [{"text": "hello", "bbox": [10, 10, 30, 30]}],
        np.full((40, 40, 3), 245, np.uint8), "en", pipeline, False,
    )
    assert skipped == 0
    assert blocks[0]["_vision"]["block_id"] == "block-1"
    assert pipeline.prepare_page.call_count == 1
```

- [ ] **Step 2: Run the test and observe missing integration**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_app_vision_integration.py -q`

Expected: FAIL because `prepare_ocr_blocks()` and `get_vision_pipeline()` do not exist.

- [ ] **Step 3: Implement lazy pipeline construction and page erasure**

Create one lazily initialized pipeline per process, but keep model inference serialized through its existing page loop. Attach only `PreparedBlock.to_summary()` and mask references to block dictionaries. Automatic mode excludes unsafe blocks from translation/rendering but preserves them in review-capable results. Replace per-block erasure with one `erase_page()` call before `render_all_blocks()`.

- [ ] **Step 4: Run integration and full regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_app_vision_integration.py translator\test_translator.py -q`

Expected: PASS; current endpoints and translation fallbacks remain compatible.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add app.py tests\vision\test_app_vision_integration.py
git commit -m "feat: reuse prepared masks in Flask rendering"
git push origin main
```

### Task 7: Preserve risky blocks for explicit manual review

**Files:**
- Create: `tests/vision/test_manual_review_flow.py`
- Modify: `app.py:414-447`
- Modify: `app.py:867-959`
- Modify: `templates/correction.html`
- Modify: `static/js/correction.js`
- Modify: `static/css/correction.css`

**Interfaces:**
- Produces JSON fields: `requires_review: bool`, `review_reason: str`, and `erase_confirmed: bool`.
- Changes: corrected bbox invalidates `mask_ref`; text-only edits retain it.

- [ ] **Step 1: Write endpoint tests for review confirmation and bbox invalidation**

```python
import app as app_module


def test_unconfirmed_risky_block_is_preserved(client, saved_review_session, monkeypatch):
    captured = {}
    monkeypatch.setattr(app_module, "translate_and_render", lambda results, *args, **kwargs: captured.setdefault("results", results) or [])
    response = client.post("/continue_translate", json={
        "session_id": saved_review_session,
        "blocks": [{
            "text": "BOOM", "bbox": [10, 10, 80, 50],
            "requires_review": True, "erase_confirmed": False,
        }],
    })
    assert response.status_code == 200
    assert captured["results"][0][2][0]["_preserve_original"] is True


def test_changed_bbox_invalidates_cached_mask():
    block = {"bbox": [10, 10, 80, 50], "_vision": {"bbox": [10, 10, 80, 50], "mask_ref": "vision/a.npz"}}
    updated = app_module.apply_manual_block_update(block, {"text": "BOOM", "bbox": [12, 10, 82, 50]})
    assert updated["_vision"]["mask_ref"] is None
```

- [ ] **Step 2: Run the test and observe missing behavior/fixtures**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_manual_review_flow.py -q`

Expected: FAIL until fixtures, update helper, and review behavior exist.

- [ ] **Step 3: Implement minimal review UI and server enforcement**

Before editing frontend files, use the `impeccable` skill. Render a visible `Cần kiểm tra` badge and an unchecked `Cho phép xóa vùng này` checkbox for risky blocks. JavaScript submits the three exact fields above. The server, not JavaScript, enforces preservation when confirmation is false. Compare normalized old/new bbox values to decide cache invalidation.

- [ ] **Step 4: Run focused, endpoint, and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_manual_review_flow.py translator\test_translator.py -q`

Expected: PASS; unconfirmed risky blocks retain original pixels.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add app.py templates\correction.html static\js\correction.js static\css\correction.css tests\vision\test_manual_review_flow.py
git commit -m "feat: require review for risky text masks"
git push origin main
```

### Task 8: Use the shared vision pipeline from the CLI

**Files:**
- Create: `tests/vision/test_cli_vision.py`
- Modify: `main.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `build_parser() -> argparse.ArgumentParser` and `run(args: argparse.Namespace) -> int`.
- Adds: `--vision-config`, `--masker`, `--model-dir`, `--inpainter`, `--device`, and `--debug-masks`.

- [ ] **Step 1: Write parser and pipeline-use tests**

```python
from unittest.mock import Mock

from main import build_parser, run


def test_cli_cuda_vision_flags_parse():
    args = build_parser().parse_args([
        "-i", "examples/0.png", "-s", "output", "--device", "cuda",
        "--masker", "auto", "--inpainter", "auto",
        "--vision-config", "configs/vision.json",
    ])
    assert args.device == "cuda"
    assert args.masker == "auto"
    assert args.inpainter == "auto"


def test_cli_prepares_and_erases_once_per_page(monkeypatch, tmp_path):
    pipeline = Mock()
    monkeypatch.setattr("main.get_vision_pipeline", lambda args: pipeline)
    args = build_parser().parse_args([
        "-i", "examples/0.png", "-s", str(tmp_path), "--translator", "google",
    ])
    run(args)
    assert pipeline.prepare_page.call_count == 1
    assert pipeline.erase_page.call_count == 1
```

- [ ] **Step 2: Run the test and observe import/behavior failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_cli_vision.py -q`

Expected: FAIL because parsing is currently inside `if __name__ == "__main__"` and the CLI uses per-block erasure.

- [ ] **Step 3: Extract parser/run and document CUDA use**

Keep all existing options and defaults, add the six exact flags, prepare all OCR blocks before translation, erase the page once, and render accepted blocks afterward. README commands must install base dependencies first and `requirements-vision-cuda.txt` only for the CUDA profile; document explicit CPU fallback warnings and model cache location.

- [ ] **Step 4: Run CLI, vision, and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision translator\test_translator.py -q`

Expected: PASS.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add main.py README.md tests\vision\test_cli_vision.py
git commit -m "feat: expose shared vision pipeline in CLI"
git push origin main
```
