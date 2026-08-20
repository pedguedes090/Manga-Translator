# Vision evaluation and rollout implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the complete vision pipeline on reproducible data, calibrate configuration from evidence, document CUDA operation, and remove superseded code only after all quality gates pass.

**Architecture:** Evaluation remains outside request-time Flask code. Deterministic tools produce machine-readable reports keyed by dataset, config, and model hashes; a calibration tool recommends a profile, and production defaults change only when the approved gates are present in a signed-off rollout report.

**Tech Stack:** Python 3.10/3.11, NumPy, OpenCV, scikit-image, LPIPS/PyTorch for optional perceptual metrics, pytest, Markdown, JSON/JSONL.

**Spec:** `docs/superpowers/specs/2026-08-20-full-vision-pipeline-design.md`

## Global constraints

- Dataset images, model weights, generated overlays, and benchmark JSON remain outside Git.
- Every report records dataset split hash, config hash, model versions, runtime, and peak VRAM.
- Outside-mask pixel delta must equal zero before rollout.
- Bubble-border damage must remain below one percent.
- Hybrid requires at least 15 percent hard-mask Dice improvement over heuristic.
- Neural requires higher Dice than hybrid without lower artwork-preservation precision.
- Simple uniform bubbles must never invoke LaMa.
- Legacy helpers are removed only when all relevant gates pass and current regression tests remain green.
- Complete each task in one commit and push directly to `origin/main` after focused and regression tests pass.
- Never force-push or stage unrelated working-tree changes.

## File map

| Path | Responsibility |
| --- | --- |
| `vision/metrics.py` | Boundary, border-damage, residual, and inpainting metrics. |
| `tools/validate_vision_dataset.py` | Manifest schema, path, license, split, and category checks. |
| `tools/benchmark_inpainting.py` | Fixed flat/Telea/LaMa evaluation matrix. |
| `tools/calibrate_vision.py` | Evidence-based threshold and backend recommendation. |
| `tools/run_vision_regression.py` | End-to-end sample run and debug artifact tree. |
| `tests/vision/test_advanced_metrics.py` | Exact metric behavior. |
| `tests/vision/test_dataset_validation.py` | Dataset contract and stratification. |
| `tests/vision/test_calibration.py` | Rollout gates and recommendation behavior. |
| `docs/vision/CUDA.md` | CUDA install, verification, fallback, and troubleshooting. |
| `docs/vision/MODELS.md` | Model provenance, versions, cache, and replacement workflow. |
| `docs/vision/BENCHMARKS.md` | Reproducible commands and accepted rollout evidence. |

---

### Task 1: Add advanced mask and inpainting metrics

**Files:**
- Create: `tests/vision/test_advanced_metrics.py`
- Modify: `vision/metrics.py`
- Modify: `requirements-vision-cuda.txt`

**Interfaces:**
- Produces: `compute_boundary_f1(prediction, target, tolerance_px=2) -> float`.
- Produces: `compute_bubble_border_damage(before, after, bubble_boundary) -> float`.
- Produces: `compute_outside_mask_delta(before, after, mask) -> int`.
- Produces: `compute_inpainting_metrics(before, after, clean_target, mask) -> dict[str, float]`.

- [ ] **Step 1: Write exact invariance and boundary tests**

```python
import numpy as np

from vision.metrics import (
    compute_boundary_f1,
    compute_bubble_border_damage,
    compute_outside_mask_delta,
)


def test_outside_mask_delta_counts_changed_channel_values():
    before = np.zeros((3, 3, 3), np.uint8)
    after = before.copy()
    after[0, 0] = (1, 2, 3)
    mask = np.zeros((3, 3), np.uint8)
    mask[1, 1] = 255
    assert compute_outside_mask_delta(before, after, mask) == 3


def test_identical_boundaries_have_perfect_f1():
    mask = np.zeros((20, 20), np.uint8)
    mask[5:15, 5:15] = 255
    assert compute_boundary_f1(mask, mask, tolerance_px=2) == 1.0


def test_bubble_border_damage_reports_changed_fraction():
    before = np.zeros((2, 2, 3), np.uint8)
    after = before.copy()
    after[0, 0] = 255
    boundary = np.array([[255, 255], [0, 0]], np.uint8)
    assert compute_bubble_border_damage(before, after, boundary) == 0.5
```

- [ ] **Step 2: Run the focused test and observe missing metrics**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_advanced_metrics.py -q`

Expected: FAIL because advanced metric functions do not exist.

- [ ] **Step 3: Implement local-region metrics**

Compute morphological boundaries and tolerance matching for boundary F1. Count outside-mask changed channel values exactly. Compute border damage per changed boundary pixel, not per channel. Compute PSNR, SSIM, MAE, and LPIPS on mask bounding regions dilated by 0, 5, and 20 pixels; return `lpips=null` in report serialization when the optional dependency is unavailable.

- [ ] **Step 4: Run focused and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_advanced_metrics.py translator\test_translator.py -q`

Expected: PASS.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add vision\metrics.py requirements-vision-cuda.txt tests\vision\test_advanced_metrics.py
git commit -m "feat: add vision quality metrics"
git push origin main
```

### Task 2: Validate and freeze a stratified evaluation manifest

**Files:**
- Create: `tools/validate_vision_dataset.py`
- Create: `tests/vision/test_dataset_validation.py`
- Create: `datasets/manifests/README.md`

**Interfaces:**
- Produces: `validate_manifest(path: Path, root: Path) -> DatasetSummary`.
- Requires: image, text mask, bubble mask, bbox, category, language, source, license, split for every JSONL row.
- Enforces: the eight approved category shares within two percentage points for a 500-to-1000 sample evaluation set.

- [ ] **Step 1: Write schema, path, and quota tests**

```python
import json

import pytest

from tools.validate_vision_dataset import validate_manifest


def test_manifest_rejects_missing_license(tmp_path):
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(json.dumps({
        "image": "a.png", "text_mask": "a-mask.png", "bubble_mask": "a-bubble.png",
        "bbox": [0, 0, 10, 10], "category": "white_bubble", "language": "ja",
        "source": "generated", "split": "test",
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="license"):
        validate_manifest(manifest, tmp_path)


def test_manifest_rejects_non_test_rows(tmp_path, complete_manifest_factory):
    manifest = complete_manifest_factory(tmp_path, split="train")
    with pytest.raises(ValueError, match="evaluation manifest must use test split"):
        validate_manifest(manifest, tmp_path)
```

- [ ] **Step 2: Run the test and observe import/fixture failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_dataset_validation.py -q`

Expected: FAIL because validator and fixture do not exist.

- [ ] **Step 3: Implement validation and document local assembly**

Add `complete_manifest_factory` to `tests/vision/conftest.py`. Resolve all paths under the provided root, decode images/masks with OpenCV, require equal spatial dimensions, clamp-free valid bboxes, recognized categories, non-empty source/license, unique sample IDs, and test-only split. README gives exact generation and validation commands and states that raw restricted data remains under ignored `datasets/local/`.

- [ ] **Step 4: Generate and validate the version-one set**

Run:

```powershell
.\.venv\Scripts\python.exe tools\generate_synthetic_dataset.py --output datasets\local\vision-eval-v1 --samples 1000 --seed 20260820
.\.venv\Scripts\python.exe tools\validate_vision_dataset.py --manifest datasets\local\vision-eval-v1\manifest.jsonl --root datasets\local\vision-eval-v1 --output reports\vision\dataset-v1-summary.json
.\.venv\Scripts\python.exe -m pytest tests\vision\test_dataset_validation.py translator\test_translator.py -q
```

Expected: exactly 1000 valid test samples; category distribution matches the approved 20/10/10/15/15/15/10/5 percentages.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add tools\validate_vision_dataset.py tests\vision\test_dataset_validation.py tests\vision\conftest.py datasets\manifests\README.md
git commit -m "test: validate stratified vision datasets"
git push origin main
```

### Task 3: Benchmark the fixed masker and inpainting matrix

**Files:**
- Modify: `tools/benchmark_inpainting.py`
- Create: `tests/vision/test_benchmark_tools.py`
- Modify: `tools/evaluate_masks.py`

**Interfaces:**
- Produces masker rows for `heuristic`, `hybrid`, and `neural`.
- Produces inpainting rows for `flat`, `telea`, and `lama`.
- Every row includes category, metrics, runtime, peak VRAM, config hash, dataset hash, and model versions.

- [ ] **Step 1: Write report-schema and method-isolation tests**

```python
from tools.benchmark_inpainting import benchmark_samples


def test_benchmark_report_contains_reproducibility_fields(clean_sample, fake_inpainters):
    report = benchmark_samples([clean_sample], fake_inpainters, config_hash="a" * 64, dataset_hash="b" * 64)
    row = report["rows"][0]
    assert row["config_hash"] == "a" * 64
    assert row["dataset_hash"] == "b" * 64
    assert row["method"] in {"flat", "telea", "lama"}
    assert row["outside_mask_delta"] == 0
```

- [ ] **Step 2: Run the focused test and observe missing fixtures/tool**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_benchmark_tools.py -q`

Expected: FAIL because benchmark helpers and fixtures do not exist.

- [ ] **Step 3: Implement the shared report schema and fixed matrix**

Add `clean_sample` and injected `fake_inpainters` fixtures. The CLI loads one manifest once, evaluates each selected method against identical samples, catches per-sample failures without losing other rows, and writes summary plus raw rows. `evaluate_masks.py` adopts the same metadata keys and includes boundary/border metrics.

- [ ] **Step 4: Run the complete benchmark matrix**

Run:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_masks.py --manifest datasets\local\vision-eval-v1\manifest.jsonl --backend heuristic --config configs\vision.json --output reports\vision\mask-heuristic-v1.json
.\.venv\Scripts\python.exe tools\evaluate_masks.py --manifest datasets\local\vision-eval-v1\manifest.jsonl --backend hybrid --config configs\vision.json --output reports\vision\mask-hybrid-v1.json
.\.venv\Scripts\python.exe tools\evaluate_masks.py --manifest datasets\local\vision-eval-v1\manifest.jsonl --backend neural --config configs\vision.json --output reports\vision\mask-neural-v1.json
.\.venv\Scripts\python.exe tools\benchmark_inpainting.py --manifest datasets\local\vision-eval-v1\manifest.jsonl --methods flat telea lama --config configs\vision.json --output reports\vision\inpainting-v1.json
```

Expected: all reports share the same dataset hash; each method has 1000 rows or explicit per-sample errors.

- [ ] **Step 5: Run tests, commit, and push**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_benchmark_tools.py translator\test_translator.py -q`

Expected: PASS.

```powershell
git fetch origin
git add tools\benchmark_inpainting.py tools\evaluate_masks.py tests\vision\test_benchmark_tools.py
git commit -m "test: benchmark vision backend matrix"
git push origin main
```

### Task 4: Calibrate production gates from benchmark evidence

**Files:**
- Create: `tools/calibrate_vision.py`
- Create: `tests/vision/test_calibration.py`
- Modify: `configs/vision.json`

**Interfaces:**
- Produces: `calibrate(heuristic_report, hybrid_report, neural_report, inpaint_report) -> CalibrationResult`.
- Produces: explicit backend gates and selected probability/coverage/border thresholds.

- [ ] **Step 1: Write gate pass/fail tests**

```python
from tools.calibrate_vision import calibrate


def test_calibration_enables_backends_only_when_all_gates_pass(report_factory):
    result = calibrate(
        report_factory(dice=0.50, precision=0.96),
        report_factory(dice=0.60, precision=0.97, hard_dice_gain=0.16),
        report_factory(dice=0.67, precision=0.97, bubble_border_damage=0.005),
        report_factory(outside_mask_delta=0, simple_lama_calls=0),
    )
    assert result.hybrid_gate_passed is True
    assert result.neural_gate_passed is True
    assert result.bubble_gate_passed is True


def test_calibration_rejects_any_outside_mask_change(report_factory):
    result = calibrate(
        report_factory(dice=0.50), report_factory(dice=0.60, hard_dice_gain=0.16),
        report_factory(dice=0.67), report_factory(outside_mask_delta=1),
    )
    assert result.production_ready is False
```

- [ ] **Step 2: Run the focused test and observe import/fixture failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_calibration.py -q`

Expected: FAIL because calibration tool and fixture do not exist.

- [ ] **Step 3: Implement bounded threshold search and immutable evidence output**

Search `prob_high` over `0.55, 0.60, 0.62, 0.65, 0.70`, `prob_low` over `0.25, 0.30, 0.34, 0.38, 0.42` with low below high, coverage over `0.55, 0.60, 0.65`, and border overlap over `0.01, 0.02, 0.03`. Select highest Dice among candidates that meet precision and border gates. Emit chosen values, rejected candidates, input hashes, and gate booleans.

- [ ] **Step 4: Calibrate and apply only the recorded passing profile**

Run:

```powershell
.\.venv\Scripts\python.exe tools\calibrate_vision.py --heuristic reports\vision\mask-heuristic-v1.json --hybrid reports\vision\mask-hybrid-v1.json --neural reports\vision\mask-neural-v1.json --inpainting reports\vision\inpainting-v1.json --output reports\vision\calibration-v1.json
.\.venv\Scripts\python.exe tools\calibrate_vision.py --apply reports\vision\calibration-v1.json --config configs\vision.json
.\.venv\Scripts\python.exe -m pytest tests\vision\test_calibration.py tests\vision\test_backend_selection.py translator\test_translator.py -q
```

Expected: config changes only when `production_ready=true`; otherwise command exits nonzero without modifying config.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add tools\calibrate_vision.py tests\vision\test_calibration.py tests\vision\conftest.py configs\vision.json
git commit -m "feat: calibrate vision rollout gates"
git push origin main
```

### Task 5: Produce end-to-end visual regression artifacts

**Files:**
- Create: `tools/run_vision_regression.py`
- Create: `tests/vision/test_end_to_end_vision.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces per-sample `roi.png`, `raw_mask.png`, `final_mask.png`, `overlay_before.png`, `inpainted.png`, `overlay_after.png`, and `metrics.json`.
- Produces page summary containing accepted, reviewed, preserved, and fallback block counts.

- [ ] **Step 1: Write artifact and preservation tests**

```python
from tools.run_vision_regression import run_sample


def test_regression_sample_writes_complete_debug_bundle(tmp_path, end_to_end_sample, fake_pipeline):
    summary = run_sample(end_to_end_sample, fake_pipeline, tmp_path)
    sample_dir = tmp_path / end_to_end_sample["id"]
    assert {path.name for path in sample_dir.iterdir()} == {
        "roi.png", "raw_mask.png", "final_mask.png", "overlay_before.png",
        "inpainted.png", "overlay_after.png", "metrics.json",
    }
    assert summary["outside_mask_delta"] == 0
```

- [ ] **Step 2: Run the test and observe missing tool/fixtures**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_end_to_end_vision.py -q`

Expected: FAIL because regression runner and fixtures do not exist.

- [ ] **Step 3: Implement deterministic artifact rendering**

Use PNG for all lossless intermediates. Draw mask overlays with fixed blue/red alpha colors, serialize metrics with sorted JSON keys, and never include source text or filesystem paths beyond sample IDs in shared summaries. Add `debug_outputs/vision_regression/` to ignore rules.

- [ ] **Step 4: Run representative pages and full tests**

Run:

```powershell
.\.venv\Scripts\python.exe tools\run_vision_regression.py --manifest datasets\local\vision-eval-v1\manifest.jsonl --config configs\vision.json --output debug_outputs\vision_regression --limit 100
.\.venv\Scripts\python.exe -m pytest tests\vision translator\test_translator.py -q
```

Expected: 100 complete artifact directories; all tests PASS.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add .gitignore tools\run_vision_regression.py tests\vision\test_end_to_end_vision.py tests\vision\conftest.py
git commit -m "test: add end-to-end vision regression runner"
git push origin main
```

### Task 6: Document CUDA, models, benchmarks, and operational failures

**Files:**
- Create: `docs/vision/CUDA.md`
- Create: `docs/vision/MODELS.md`
- Create: `docs/vision/BENCHMARKS.md`
- Create: `tests/vision/test_vision_docs.py`
- Modify: `README.md`

**Interfaces:**
- Documents exact setup, provider verification, model download/cache, benchmark commands, gate results, OOM fallback, and recovery.

- [ ] **Step 1: Write documentation assertions**

Create `tests/vision/test_vision_docs.py` with:

```python
from pathlib import Path


def test_vision_docs_include_required_operational_commands():
    cuda = Path("docs/vision/CUDA.md").read_text(encoding="utf-8")
    models = Path("docs/vision/MODELS.md").read_text(encoding="utf-8")
    benchmarks = Path("docs/vision/BENCHMARKS.md").read_text(encoding="utf-8")
    assert "onnxruntime.get_available_providers" in cuda
    assert "SHA-256" in models
    assert "tools\\calibrate_vision.py" in benchmarks
    assert "outside-mask pixel delta" in benchmarks
```

- [ ] **Step 2: Run the test and observe missing files**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_vision_docs.py -q`

Expected: FAIL because the documentation files do not exist.

- [ ] **Step 3: Write task-oriented documentation**

Use the `docs-generator` skill. Lead CUDA docs with driver/package compatibility and the exact provider probe. Explain cache deletion and redownload. Record actual model versions/hashes from the manifest. Copy reproducible benchmark commands from Tasks 2 through 5 and summarize measured results without embedding restricted images.

- [ ] **Step 4: Run documentation and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_vision_docs.py translator\test_translator.py -q`

Expected: PASS.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add docs\vision\CUDA.md docs\vision\MODELS.md docs\vision\BENCHMARKS.md README.md tests\vision\test_vision_docs.py
git commit -m "docs: document CUDA vision pipeline"
git push origin main
```

### Task 7: Deprecate superseded vision code behind evidence gates

**Files:**
- Create: `tests/vision/test_legacy_deprecation.py`
- Modify: `add_text.py`
- Modify: `detect_bubbles.py`
- Modify: `process_bubble.py`
- Modify: `docs/vision/BENCHMARKS.md`

**Interfaces:**
- Preserves public wrappers: `assess_erasability()` and `erase_text_region()`.
- Removes duplicate private implementations only after calibrated production gates are true.

- [ ] **Step 1: Write compatibility and duplicate-call tests**

```python
import ast
from pathlib import Path

from add_text import assess_erasability, erase_text_region


def test_public_erase_wrappers_remain_callable():
    assert callable(assess_erasability)
    assert callable(erase_text_region)


def test_add_text_no_longer_defines_private_mask_engine():
    tree = ast.parse(Path("add_text.py").read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    assert len(functions["_build_text_stroke_mask"].body) <= 2
    assert len(functions["_analyze_region"].body) <= 2
```

- [ ] **Step 2: Run the test before cleanup**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_legacy_deprecation.py -q`

Expected: the public wrapper test passes; duplicate-definition assertion fails while legacy implementations remain.

- [ ] **Step 3: Remove only code proven redundant**

Require `production_ready=true` in `reports/vision/calibration-v1.json` before editing. Keep forwarding wrappers and imports used by external callers. Replace `detect_bubbles.py` active entry points with documented calls into `vision.bubbles`; replace useful `process_bubble.py` entry points with `vision.inpainting` wrappers. Remove private duplicate bodies and update benchmark docs with the calibration evidence hash.

- [ ] **Step 4: Run the entire test suite and smoke commands**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py -i examples\0.png -s debug_outputs\final-cli-smoke --source-lang ja --target-lang vi -t google --vision-config configs\vision.json --device cuda
```

Expected: all tests PASS; CLI writes `output_image.jpg`; no removed private implementation is called.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add add_text.py detect_bubbles.py process_bubble.py docs\vision\BENCHMARKS.md tests\vision\test_legacy_deprecation.py
git commit -m "refactor: retire superseded vision heuristics"
git push origin main
```

### Task 8: Record final rollout evidence

**Files:**
- Create: `docs/vision/ROLLOUT-v1.0.0.md`
- Modify: `README.md`

**Interfaces:**
- Records final commit, model versions, dataset/config hashes, test count, benchmark gates, known fallbacks, and rollback command.

- [ ] **Step 1: Generate a release evidence draft**

Run:

```powershell
git fetch origin
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools\calibrate_vision.py --heuristic reports\vision\mask-heuristic-v1.json --hybrid reports\vision\mask-hybrid-v1.json --neural reports\vision\mask-neural-v1.json --inpainting reports\vision\inpainting-v1.json --output reports\vision\calibration-v1.json
git rev-parse HEAD
git rev-list --left-right --count main...origin/main
```

Expected: zero test failures, `production_ready=true`, and local/remote counts `0 0` before the documentation commit.

- [ ] **Step 2: Write the evidence document**

Use `docs-generator`. Include exact command outputs and report hashes, explicitly state full-resolution LaMa behavior and OOM fallback, list any samples preserved for manual review, and define rollback as reverting rollout commits in reverse order without force-pushing.

- [ ] **Step 3: Validate document links and absence of generated artifacts**

Run:

```powershell
git diff --check
git status --short
.\.venv\Scripts\python.exe -m pytest tests\vision\test_vision_docs.py -q
```

Expected: only `docs/vision/ROLLOUT-v1.0.0.md` and the README update are intended staged candidates; generated datasets/reports/models remain ignored.

- [ ] **Step 4: Commit, push, and verify remote equality**

```powershell
git add docs\vision\ROLLOUT-v1.0.0.md README.md
git commit -m "docs: record vision pipeline rollout"
git push origin main
git fetch origin
git rev-list --left-right --count main...origin/main
```

Expected: final count is `0 0`.
