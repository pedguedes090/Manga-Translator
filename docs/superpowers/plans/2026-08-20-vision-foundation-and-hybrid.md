# Vision foundation and hybrid masking implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a behavior-compatible vision foundation, freeze measurable heuristic output, and add a benchmark-gated hybrid text masker that generates one reusable mask per OCR block.

**Architecture:** Introduce typed result objects and a focused `vision` package while keeping `add_text.py` wrappers intact. Extract current analysis/masking without behavior changes, then layer local-background hysteresis and OCR-anchored postprocessing behind the same `TextMasker` interface.

**Tech Stack:** Python 3.10/3.11, NumPy 1.24.2, OpenCV 4.9.0, dataclasses, JSON, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-full-vision-pipeline-design.md`

## Global constraints

- Develop and deliver directly on `main`.
- Complete each independently testable task in one commit and push it to `origin/main` after its tests pass.
- Fetch before each task; pull with rebase when `origin/main` is ahead; never force-push.
- Do not stage unrelated working-tree changes under `.commandcode/`, `.jules/`, or `__pycache__/`.
- Keep Chrome Lens OCR, translators, and rendering behavior unchanged.
- Keep `add_text.py` compatibility wrappers until the rollout plan proves replacement gates.
- Do not add CUDA or training dependencies in this plan.
- Do not commit generated datasets or benchmark output.

## File map

| Path | Responsibility |
| --- | --- |
| `vision/types.py` | Shared bbox, analysis, mask, decision, prepared-block, and erase-result dataclasses. |
| `vision/config.py` | Validated JSON configuration and stable config hashing. |
| `vision/region_analysis.py` | Non-mutating background and bubble-context analysis extracted from `add_text.py`. |
| `vision/metrics.py` | Pixel mask metrics used by baseline and later model gates. |
| `vision/postprocess.py` | Hysteresis, OCR anchoring, component cleanup, bubble gating, and stroke-width dilation. |
| `vision/maskers/base.py` | `TextMasker` protocol. |
| `vision/maskers/heuristic.py` | Behavior-compatible wrapper around the current threshold masker. |
| `vision/maskers/hybrid.py` | Local-background residual masker with postprocessing. |
| `vision/pipeline.py` | Page preparation and erasability decisions using one generated mask. |
| `tools/generate_synthetic_dataset.py` | Deterministic baseline images and exact masks. |
| `tools/evaluate_masks.py` | Run a masker over a JSONL manifest and emit aggregate JSON metrics. |
| `tests/vision/` | Focused unit and integration tests for the new package. |

---

### Task 1: Define stable vision result types

**Files:**
- Create: `vision/__init__.py`
- Create: `vision/types.py`
- Create: `tests/vision/__init__.py`
- Create: `tests/vision/test_types.py`

**Interfaces:**
- Produces: `BBox`, `RegionAnalysis`, `BubbleInstance`, `MaskResult`, `ErasabilityDecision`, `PreparedBlock`, `EraseResult`, and `EraseMethod` with the exact fields defined in the approved spec.
- Consumes: NumPy arrays and standard-library dataclasses only.

- [ ] **Step 1: Write serialization-boundary tests**

```python
from pathlib import Path

import numpy as np

from vision.types import ErasabilityDecision, MaskResult, PreparedBlock, RegionAnalysis


def test_prepared_block_keeps_runtime_array_outside_serialized_summary():
    region = RegionAnalysis(
        mean_bgr=(245, 245, 245), mean_intensity=245.0,
        intensity_std=2.0, edge_score=4.0, texture_std=3.0,
        dominant_tone="light", uniformity="uniform",
        bubble_context="in_bubble",
    )
    mask = MaskResult(
        roi_bbox=(10, 20, 50, 60), mask=np.zeros((40, 40), np.uint8),
        probability=None, bubble_mask=None, coverage=0.0,
        confidence=0.9, edge_touch_ratio=0.0, backend="heuristic",
    )
    block = PreparedBlock(
        block_id="page-0-block-0", text="hello", bbox=(10, 20, 50, 60),
        region=region, mask_result=mask,
        decision=ErasabilityDecision(True, "uniform_background", 0.9, False),
        erase_method="flat", mask_ref=Path("vision/page-0-block-0.npz"),
    )

    summary = block.to_summary()

    assert summary["block_id"] == "page-0-block-0"
    assert summary["bbox"] == [10, 20, 50, 60]
    assert summary["mask_ref"] == "vision/page-0-block-0.npz"
    assert "mask" not in summary
```

- [ ] **Step 2: Run the focused test and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_types.py -q`

Expected: FAIL because `vision.types` does not exist.

- [ ] **Step 3: Implement the dataclasses and summary method**

Copy the field definitions from the spec exactly. Add this method to `PreparedBlock`:

```python
def to_summary(self) -> dict[str, object]:
    return {
        "block_id": self.block_id,
        "text": self.text,
        "bbox": list(self.bbox),
        "mask_ref": str(self.mask_ref) if self.mask_ref else None,
        "decision": {
            "safe": self.decision.safe,
            "reason": self.decision.reason,
            "score": self.decision.score,
            "requires_review": self.decision.requires_review,
        },
        "backend": self.mask_result.backend,
        "coverage": self.mask_result.coverage,
        "confidence": self.mask_result.confidence,
        "erase_method": self.erase_method,
    }
```

- [ ] **Step 4: Run focused and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_types.py translator\test_translator.py -q`

Expected: PASS with 49 tests or more.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add vision\__init__.py vision\types.py tests\vision\__init__.py tests\vision\test_types.py
git commit -m "refactor: add vision result types"
git push origin main
```

### Task 2: Load and hash validated vision configuration

**Files:**
- Create: `configs/vision.json`
- Create: `vision/config.py`
- Create: `tests/vision/test_config.py`

**Interfaces:**
- Produces: `VisionConfig.load(path: str | Path) -> VisionConfig` and `VisionConfig.config_hash() -> str`.
- Consumes: the exact approved JSON keys under `text_mask`, `bubble`, `inpaint`, `safety`, and `debug`.

- [ ] **Step 1: Write config validation tests**

```python
import json

import pytest

from vision.config import VisionConfig


def test_default_config_has_cuda_full_resolution_profile():
    config = VisionConfig.load("configs/vision.json")
    assert config.profile == "cuda"
    assert config.inpaint.lama_full_resolution is True
    assert config.text_mask.prob_low < config.text_mask.prob_high
    assert len(config.config_hash()) == 64


def test_config_rejects_reversed_probability_thresholds(tmp_path):
    path = tmp_path / "vision.json"
    path.write_text(json.dumps({
        "profile": "cuda", "mask_backend": "auto", "allow_cpu_fallback": True,
        "text_mask": {"prob_low": 0.8, "prob_high": 0.2},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="prob_low must be less than prob_high"):
        VisionConfig.load(path)
```

- [ ] **Step 2: Run the tests and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_config.py -q`

Expected: FAIL because `vision.config` does not exist.

- [ ] **Step 3: Implement immutable nested config dataclasses**

Use `json`, `hashlib`, and dataclasses only. Merge user JSON over the approved defaults, reject unknown top-level keys, require `0 <= prob_low < prob_high <= 1`, and hash canonical JSON using `sort_keys=True` and compact separators. Write the approved JSON block from the spec verbatim to `configs/vision.json`.

- [ ] **Step 4: Run focused and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_config.py translator\test_translator.py -q`

Expected: PASS.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add configs\vision.json vision\config.py tests\vision\test_config.py
git commit -m "feat: add validated vision configuration"
git push origin main
```

### Task 3: Freeze mask metrics and deterministic synthetic fixtures

**Files:**
- Create: `vision/metrics.py`
- Create: `tools/generate_synthetic_dataset.py`
- Create: `tools/evaluate_masks.py`
- Create: `tests/vision/test_metrics.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `compute_mask_metrics(prediction: np.ndarray, target: np.ndarray, bubble_boundary: np.ndarray | None = None) -> dict[str, float]`.
- Produces: CLI manifests under `debug_outputs/vision_baseline/manifest.jsonl`, which remains ignored by Git.

- [ ] **Step 1: Write exact metric tests**

```python
import numpy as np

from vision.metrics import compute_mask_metrics


def test_mask_metrics_count_true_false_pixels_exactly():
    target = np.array([[255, 255], [0, 0]], np.uint8)
    prediction = np.array([[255, 0], [255, 0]], np.uint8)
    metrics = compute_mask_metrics(prediction, target)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["dice"] == 0.5
    assert metrics["iou"] == 1 / 3
    assert metrics["false_positive_pixels"] == 1.0
```

- [ ] **Step 2: Run the focused test and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_metrics.py -q`

Expected: FAIL because `vision.metrics` does not exist.

- [ ] **Step 3: Implement metrics and deterministic generation**

`compute_mask_metrics()` must binarize at `> 0`, use exact confusion counts, and define precision/recall as `1.0` when both prediction and target are empty. The generator must use `np.random.default_rng(seed)` and emit the eight approved categories with image, text mask, bubble mask, bbox, language, and category fields. Name artifacts `image_0000.png`, `text_mask_0000.png`, and `bubble_mask_0000.png`, incrementing the numeric suffix per sample. The evaluator imports a backend by name, processes manifest rows, and writes aggregate JSON containing per-category means and the config hash.

Add these ignore rules:

```gitignore
datasets/generated/
reports/vision/
models/cache/
```

- [ ] **Step 4: Run tests and generate a ten-sample smoke dataset**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\vision\test_metrics.py translator\test_translator.py -q
.\.venv\Scripts\python.exe tools\generate_synthetic_dataset.py --output debug_outputs\vision_baseline --samples 10 --seed 20260820
```

Expected: tests PASS and the manifest contains exactly ten JSON lines.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add .gitignore vision\metrics.py tools\generate_synthetic_dataset.py tools\evaluate_masks.py tests\vision\test_metrics.py
git commit -m "test: add vision mask benchmark harness"
git push origin main
```

### Task 4: Extract region analysis without changing behavior

**Files:**
- Create: `vision/region_analysis.py`
- Create: `tests/vision/test_region_analysis.py`
- Modify: `add_text.py:572-808`

**Interfaces:**
- Produces: `analyze_region(image: np.ndarray, bbox: BBox) -> RegionAnalysis | None`.
- Preserves: `add_text._analyze_region(image, bbox) -> dict | None` as a compatibility wrapper.

- [ ] **Step 1: Write parity and non-mutation tests**

```python
import cv2
import numpy as np

from add_text import _analyze_region
from vision.region_analysis import analyze_region


def test_extracted_region_analysis_matches_legacy_wrapper():
    image = np.full((140, 180, 3), 255, np.uint8)
    cv2.ellipse(image, (90, 70), (65, 42), 0, 0, 360, (0, 0, 0), 3)
    original = image.copy()
    typed = analyze_region(image, (48, 48, 125, 82))
    legacy = _analyze_region(image, [48, 48, 125, 82])
    assert typed is not None
    assert legacy["bubble_context"] == typed.bubble_context
    assert legacy["uniformity"] == typed.uniformity
    assert np.array_equal(image, original)
```

- [ ] **Step 2: Run the test and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_region_analysis.py -q`

Expected: FAIL because `vision.region_analysis` does not exist.

- [ ] **Step 3: Move analysis into the typed module**

Move the existing calculations without changing thresholds. Return `RegionAnalysis`; make `_analyze_region()` convert the dataclass with `dataclasses.asdict()` and add legacy aliases `mean_bgr` and `is_bubble` expected by current callers.

- [ ] **Step 4: Run focused and full regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_region_analysis.py translator\test_translator.py -q`

Expected: PASS, including current erasability and bubble-border tests.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add vision\region_analysis.py tests\vision\test_region_analysis.py add_text.py
git commit -m "refactor: extract typed region analysis"
git push origin main
```

### Task 5: Wrap the current masker behind `TextMasker`

**Files:**
- Create: `vision/maskers/__init__.py`
- Create: `vision/maskers/base.py`
- Create: `vision/maskers/heuristic.py`
- Create: `tests/vision/test_heuristic_masker.py`
- Modify: `add_text.py:932-1145`

**Interfaces:**
- Produces: `TextMasker.generate(image: np.ndarray, bbox: BBox, text: str, region: RegionAnalysis, bubble: BubbleInstance | None) -> MaskResult`.
- Produces: `HeuristicTextMasker.generate(image: np.ndarray, bbox: BBox, text: str, region: RegionAnalysis, bubble: BubbleInstance | None) -> MaskResult`.
- Preserves: `_build_text_stroke_mask()`, `_filter_components_outside_inner()`, and `_remove_screentone_dots()` wrappers in `add_text.py`.

- [ ] **Step 1: Write legacy parity tests**

```python
import cv2
import numpy as np

from add_text import _build_text_stroke_mask
from vision.maskers.heuristic import HeuristicTextMasker
from vision.region_analysis import analyze_region


def test_heuristic_backend_matches_legacy_mask_pixels():
    image = np.full((90, 150, 3), 245, np.uint8)
    cv2.putText(image, "TEST", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    bbox = (10, 20, 120, 65)
    region = analyze_region(image, bbox)
    result = HeuristicTextMasker().generate(image, bbox, "TEST", region, None)
    roi = image[20:65, 10:120]
    legacy = _build_text_stroke_mask(roi, region.mean_bgr, {"text_color": (0, 0, 0)})
    assert np.array_equal(result.mask, legacy)
```

- [ ] **Step 2: Run the test and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_heuristic_masker.py -q`

Expected: FAIL because the masker package does not exist.

- [ ] **Step 3: Implement the protocol and adapter**

Define `TextMasker` with the exact signature from the spec. Move mask/component functions into `vision/maskers/heuristic.py`, retain forwarding wrappers in `add_text.py`, and populate `MaskResult` with ROI-local mask, coverage, edge-touch ratio, backend name `heuristic`, and no probability map.

- [ ] **Step 4: Run focused and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_heuristic_masker.py translator\test_translator.py -q`

Expected: PASS with byte-identical parity fixtures.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add vision\maskers\__init__.py vision\maskers\base.py vision\maskers\heuristic.py tests\vision\test_heuristic_masker.py add_text.py
git commit -m "refactor: wrap heuristic text masker"
git push origin main
```

### Task 6: Implement reusable hybrid postprocessing

**Files:**
- Create: `vision/postprocess.py`
- Create: `vision/maskers/hybrid.py`
- Create: `tests/vision/test_hybrid_masker.py`

**Interfaces:**
- Produces: `hysteresis_mask(probability, low, high) -> np.ndarray`.
- Produces: `keep_ocr_anchored_components(mask, inner_rect, min_overlap=0.30) -> np.ndarray`.
- Produces: `apply_bubble_gate(mask, bubble_mask, border_px) -> np.ndarray`.
- Produces: `HybridTextMasker.generate(image: np.ndarray, bbox: BBox, text: str, region: RegionAnalysis, bubble: BubbleInstance | None) -> MaskResult`.

- [ ] **Step 1: Write postprocessing and border-safety tests**

```python
import cv2
import numpy as np

from vision.maskers.hybrid import HybridTextMasker
from vision.postprocess import hysteresis_mask
from vision.region_analysis import analyze_region


def test_hysteresis_keeps_weak_pixels_connected_to_strong_seed_only():
    probability = np.array([[0.7, 0.4, 0.0, 0.4], [0.0, 0.4, 0.0, 0.4]], np.float32)
    mask = hysteresis_mask(probability, low=0.34, high=0.62)
    assert mask[:, :2].sum() == 3 * 255
    assert mask[:, 3].sum() == 0


def test_hybrid_masker_preserves_bubble_outline():
    image = np.full((140, 180, 3), 255, np.uint8)
    cv2.ellipse(image, (90, 70), (65, 42), 0, 0, 360, (0, 0, 0), 3)
    cv2.putText(image, "HEY", (52, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    bbox = (40, 35, 140, 95)
    region = analyze_region(image, bbox)
    result = HybridTextMasker().generate(image, bbox, "HEY", region, None)
    assert result.mask[35, 0] == 0
    assert result.coverage < 0.65
```

- [ ] **Step 2: Run the test and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_hybrid_masker.py -q`

Expected: FAIL because `vision.postprocess` and the hybrid backend do not exist.

- [ ] **Step 3: Implement local-background residual masking**

Convert ROI to grayscale and Lab, estimate local background with an odd median kernel of nine, normalize absolute residual to `[0, 1]`, build strong/weak seeds from config, retain OCR-anchored components, remove screentone after anchoring, apply bubble erosion when present, estimate stroke width from distance transform, and dilate between configured one and four pixels. Store raw residual, component counts, and dilation radius in `MaskResult.debug`.

- [ ] **Step 4: Run focused and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_hybrid_masker.py translator\test_translator.py -q`

Expected: PASS and bubble-outline assertions remain unchanged.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add vision\postprocess.py vision\maskers\hybrid.py tests\vision\test_hybrid_masker.py
git commit -m "feat: add hybrid text masking backend"
git push origin main
```

### Task 7: Prepare each OCR block once and reuse its mask

**Files:**
- Create: `vision/pipeline.py`
- Create: `tests/vision/test_pipeline.py`
- Modify: `add_text.py:1234-1505`

**Interfaces:**
- Produces: `VisionPipeline(masker: TextMasker | None = None, bubble_segmenter: object | None = None, config: VisionConfig | None = None)`; a missing masker is built from config.
- Produces: `VisionPipeline.prepare_page(image, blocks) -> list[PreparedBlock]`.
- Produces: `score_erasability(region, mask_result, text) -> ErasabilityDecision`.
- Changes: `assess_erasability(image, bbox, text=None, source_lang='ja', prepared=None)` accepts an existing `PreparedBlock`.
- Changes: `erase_text_region(image, bbox, source_lang='ja', prepared=None)` consumes its existing `MaskResult` when supplied.

- [ ] **Step 1: Write the single-generation integration test**

```python
from unittest.mock import Mock

import cv2
import numpy as np

from vision.maskers.hybrid import HybridTextMasker
from vision.pipeline import VisionPipeline


def test_prepare_then_assess_and_erase_generates_one_mask():
    image = np.full((90, 150, 3), 245, np.uint8)
    cv2.putText(image, "TEST", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    masker = Mock(wraps=HybridTextMasker())
    pipeline = VisionPipeline(masker=masker, bubble_segmenter=None)
    prepared = pipeline.prepare_page(image, [{"text": "TEST", "bbox": [10, 20, 120, 65]}])
    pipeline.assess(prepared[0])
    pipeline.erase_block(image.copy(), prepared[0])
    assert masker.generate.call_count == 1
```

- [ ] **Step 2: Run the test and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_pipeline.py -q`

Expected: FAIL because `vision.pipeline` does not exist.

- [ ] **Step 3: Implement preparation, scoring, and compatibility paths**

Generate stable block IDs from page-local index and normalized bbox. Run page bubble segmentation once when a segmenter exists. Analyze each region, generate one mask, score it using the current safety rules expressed over `RegionAnalysis` and `MaskResult`, and choose `preserve`, `flat`, `telea`, or provisional `lama_full_page`. Compatibility wrappers create a temporary prepared block only when callers do not supply one.

- [ ] **Step 4: Run all foundation tests and the current suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision translator\test_translator.py -q`

Expected: PASS, including the call-count assertion and all existing erase tests.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add vision\pipeline.py tests\vision\test_pipeline.py add_text.py
git commit -m "refactor: reuse prepared text masks"
git push origin main
```

### Task 8: Gate hybrid activation with the frozen baseline

**Files:**
- Create: `reports/vision/.gitkeep`
- Create: `tests/vision/test_backend_selection.py`
- Modify: `vision/pipeline.py`
- Modify: `vision/config.py`
- Modify: `configs/vision.json`

**Interfaces:**
- Produces: `build_text_masker(config: VisionConfig) -> TextMasker`.
- Consumes: aggregate reports emitted by `tools/evaluate_masks.py`.

- [ ] **Step 1: Write backend-selection tests**

```python
from dataclasses import replace

from vision.config import VisionConfig
from vision.maskers.heuristic import HeuristicTextMasker
from vision.maskers.hybrid import HybridTextMasker
from vision.pipeline import build_text_masker


def test_auto_backend_uses_hybrid_after_gate_is_enabled():
    config = replace(VisionConfig.load("configs/vision.json"), hybrid_gate_passed=True)
    assert isinstance(build_text_masker(config), HybridTextMasker)


def test_auto_backend_keeps_heuristic_before_gate():
    config = replace(VisionConfig.load("configs/vision.json"), hybrid_gate_passed=False)
    assert isinstance(build_text_masker(config), HeuristicTextMasker)
```

- [ ] **Step 2: Run the test and observe missing selection behavior**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_backend_selection.py -q`

Expected: FAIL because `build_text_masker()` and `hybrid_gate_passed` do not exist.

- [ ] **Step 3: Implement explicit gate selection**

Add top-level booleans `hybrid_gate_passed`, `neural_gate_passed`, `bubble_gate_passed`, and `production_ready` to `VisionConfig` and the shipped JSON, all initially `false`. Run both backends on the generated and curated manifest, write reports under `reports/vision/`, and change `hybrid_gate_passed` to `true` only when hard-mask Dice improves by at least 15 percent. If the gate does not pass, commit the selection mechanism with `false` and keep hybrid selectable through `mask_backend="hybrid"` for continued experiments.

- [ ] **Step 4: Run the benchmark and full test suite**

Run:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_masks.py --manifest debug_outputs\vision_baseline\manifest.jsonl --backend heuristic --config configs\vision.json --output reports\vision\heuristic.json
.\.venv\Scripts\python.exe tools\evaluate_masks.py --manifest debug_outputs\vision_baseline\manifest.jsonl --backend hybrid --config configs\vision.json --output reports\vision\hybrid.json
.\.venv\Scripts\python.exe -m pytest tests\vision translator\test_translator.py -q
```

Expected: reports identify config hashes and per-category metrics; tests PASS. Generated JSON reports remain ignored.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add reports\vision\.gitkeep tests\vision\test_backend_selection.py vision\pipeline.py vision\config.py configs\vision.json
git commit -m "feat: gate hybrid masker rollout"
git push origin main
```
