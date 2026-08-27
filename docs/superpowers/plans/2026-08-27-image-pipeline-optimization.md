# Image Pipeline Speed and Text Erasure Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpower-subagent-driven-development (recommended) or superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up the Manga-Translator image pipeline and improve text erasure, with reproducible paired evidence for latency, safety, mask quality, and restoration quality.

**Architecture:** Keep the current legacy Heuristic + flat/Telea path as the default and fail-safe. Add a lazy, opt-in PreparedPage adapter that prepares each page once and uses the existing page-level batching; keep masks/model objects ephemeral and rebuild them after manual correction. Add a deterministic benchmark that compares the legacy and prepared paths on identical samples and refuses to promote a backend without measured gates.

**Tech Stack:** Python 3.11, NumPy, OpenCV, Pillow, Flask/SocketIO, pytest, existing VisionPipeline/LaMa abstractions, JSON/JSONL benchmark reports.

**Spec:** `docs/superpowers/specs/2026-08-20-full-vision-pipeline-design.md` and the approved research constraints in `docs/superpowers/plans/2026-08-20-vision-evaluation-and-rollout.md`.

## Global constraints

- Production default remains legacy Heuristic + flat/Telea until the new path passes the paired safety and quality gates.
- The opt-in adapter is lazy and must not import or initialize torch at module import time.
- Chrome Lens OCR, translation providers, font rendering, route responses, Socket.IO progress keys, session JSON, and `name/data/original_data` result keys remain unchanged.
- Prepared masks and model objects stay in memory; never serialize NumPy arrays or model instances into a session.
- Manual correction always rebuilds PreparedBlock objects from the corrected source image and current bboxes.
- Any CUDA/model failure falls back to the configured CPU/Telea path with a visible warning; no silent unsafe erase is allowed.
- Do not modify `build_text_stroke_mask`, `_analyze_region`, or `remove_screentone_dots`; GitNexus marked them CRITICAL-risk shared symbols.
- Any edit to the nested `app.py::render_single_image` path must preserve its tuple input `(idx, name, image, blocks)` and result `{name, image}`; GitNexus marked it HIGH risk.
- Do not use COMIX detector-proposal boxes as text-erasure quality ground truth; use them only for operational timing, routing, fallback, and crash evidence.
- Generated datasets, model weights, reports, and debug images remain outside Git and must not stage pre-existing `.commandcode/`, `.jules/`, `__pycache__/`, or other unrelated changes.
- Every benchmark report records dataset hash, config hash, backend/mode, runtime, sample counts, failures, and unavailable hardware capabilities.
- A claim of completion requires a fresh test command and a fresh paired benchmark; existing historical reports alone are insufficient.

---

### Task 1: Add exact erasure-quality metrics

**Files:**
- Modify: `vision/metrics.py`
- Create: `tests/vision/test_advanced_metrics.py`

**Interfaces:**
- Add `compute_boundary_f1(prediction: np.ndarray, target: np.ndarray, tolerance_px: int = 2) -> float`.
- Add `compute_bubble_border_damage(before: np.ndarray, after: np.ndarray, bubble_boundary: np.ndarray) -> float`.
- Add `compute_outside_mask_delta(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> int`.
- Add `compute_inpainting_metrics(before: np.ndarray, after: np.ndarray, clean_target: np.ndarray, mask: np.ndarray) -> dict[str, float | None]`.

- [ ] **Step 1: Write the failing metric tests**

Add tests with these exact behaviors before changing production code:

```python
def test_outside_mask_delta_counts_changed_channel_values():
    before = np.zeros((3, 3, 3), dtype=np.uint8)
    after = before.copy()
    after[0, 0] = (1, 2, 3)
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 255
    assert compute_outside_mask_delta(before, after, mask) == 3


def test_boundary_f1_is_one_for_identical_masks():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    assert compute_boundary_f1(mask, mask, tolerance_px=2) == 1.0


def test_bubble_border_damage_is_changed_boundary_fraction():
    before = np.zeros((2, 2, 3), dtype=np.uint8)
    after = before.copy()
    after[0, 0] = 255
    boundary = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    assert compute_bubble_border_damage(before, after, boundary) == 0.5


def test_inpainting_metrics_are_zero_for_clean_reconstruction():
    clean = np.full((4, 4, 3), 128, dtype=np.uint8)
    before = clean.copy()
    before[1:3, 1:3] = 0
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    metrics = compute_inpainting_metrics(before, clean, clean, mask)
    assert metrics["masked_lab_mae"] == 0.0
    assert metrics["outside_mask_delta"] == 0.0
```

Import the four new functions at the top of the test file. The test specifies behavior, not implementation details.

- [ ] **Step 2: Run the focused tests and confirm the expected red failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\vision\test_advanced_metrics.py -q
```

Expected result: collection/import failure because the four functions do not yet exist. Do not edit production code before observing this failure.

- [ ] **Step 3: Implement the smallest metric functions**

In `vision/metrics.py`:

1. Convert masks to boolean arrays and validate equal shapes.
2. Build boundaries with a one-pixel morphological erosion and compare predicted/target boundary pixels with a bounded dilation tolerance.
3. Count changed channel values only outside `mask > 0`; return an integer.
4. Count changed pixels on `bubble_boundary > 0` divided by boundary pixels, returning `0.0` for an empty boundary.
5. For the mask and its bounding ROI, return masked Lab MAE, masked RGB MAE, outside-mask delta, and optional SSIM/LPIPS fields as `None` when unavailable. Use clean target as the restoration reference and never alter input arrays.

- [ ] **Step 4: Run focused and existing metric tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\vision\test_advanced_metrics.py tests\vision\test_metrics.py -q
```

Expected result: all focused metric tests pass.

- [ ] **Step 5: Commit only the metric change**

Run `git diff --check`, verify only the two task files are intended, then commit with `feat: add erasure quality metrics`. Do not stage existing workspace changes.

---

### Task 2: Build a deterministic paired benchmark

**Files:**
- Create: `tools/benchmark_vision_pipeline.py`
- Create: `tests/vision/test_benchmark_vision_pipeline.py`
- Modify: `tools/evaluate_masks.py` only if a shared hash/stat helper is required

**Interfaces:**
- Add `benchmark_manifest(manifest_path: str | Path, config_path: str | Path, mode: str, backend: str, *, indices: list[int] | None = None, warmup: int = 0, pipeline: object | None = None) -> dict[str, object]`.
- Accept `mode` values "legacy" and "prepared", and `backend` values "heuristic" and "hybrid".
- Return JSON-safe `schema_version=1`, dataset/config hashes, runtime capabilities, one row per selected page, and summary p50/p95/p99 plus total/mean for `decode_ms`, `prepare_ms`, `erase_ms`, `render_ms`, and `total_ms`.
- Each row includes `id`, `status`, `block_count`, `method_counts`, `warning_count`, and error text without absolute paths when a page fails. Synthetic rows also include `mask_metrics` and `inpainting_metrics`.

- [ ] **Step 1: Write failing benchmark contract tests**

Use a temporary JSONL manifest containing one 32x32 PNG, one clean target, one text mask, one bubble mask, and one bbox. Inject a fake page adapter/pipeline so the test never requires torch or network access:

```python
def test_benchmark_report_has_stage_timings_and_hashes(tmp_path):
    report = benchmark_manifest(
        tmp_path / "manifest.jsonl",
        tmp_path / "vision.json",
        mode="prepared",
        backend="heuristic",
        indices=[0],
        pipeline=FakePipeline(),
    )
    assert report["schema_version"] == 1
    assert report["dataset_hash"]
    assert report["config_hash"]
    row = report["rows"][0]
    assert set(("decode_ms", "prepare_ms", "erase_ms", "render_ms", "total_ms")) <= row.keys()


def test_benchmark_keeps_page_errors_as_json_rows(tmp_path):
    report = benchmark_manifest(
        tmp_path / "manifest.jsonl",
        tmp_path / "vision.json",
        mode="prepared",
        backend="heuristic",
        indices=[0],
        pipeline=FailingPipeline(),
    )
    assert report["failed_pages"] == 1
    assert report["rows"][0]["status"] == "error"
    assert "tmp_path" not in report["rows"][0]["error"]
```

The test fixture must create its input files and use injected fakes; it must not call a real external service.

- [ ] **Step 2: Run the benchmark tests and confirm the expected red failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\vision\test_benchmark_vision_pipeline.py -q
```

Expected result: import/signature failure because `tools/benchmark_vision_pipeline.py` does not exist.

- [ ] **Step 3: Implement the benchmark runner**

Implement these concrete rules:

1. Read JSONL rows and hash the exact manifest bytes with SHA-256.
2. Decode once per row, start separate timers around page preparation, page erasure, and rendering/compositing; include decode in total but never in prepare/erase.
3. In "legacy" mode, use the existing `erase_text_region` call for each non-empty test block and retain its appearance/render behavior.
4. In "prepared" mode, call the adapter once for the page, call `erase_page` once, and use the returned prepared-block metadata without erasing a block twice.
5. Use `time.perf_counter`; compute percentile values only when the observation count supports them and emit "insufficient_evidence" for unsupported p99.
6. If torch/CUDA is present, synchronize around GPU work and record device/capability; otherwise record `cuda_available=false` and continue with the configured fallback.
7. For synthetic rows, place predicted masks in page coordinates, call the new metrics against `clean_target`, and record outside-mask delta. For COMIX rows, omit quality fields and mark `annotation_semantics=detector-proposal`.
8. Catch page exceptions, increment `failed_pages`, and continue; sanitize absolute paths in error text.

- [ ] **Step 4: Run the benchmark contract tests and existing evaluator tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\vision\test_benchmark_vision_pipeline.py tests\vision\test_evaluate_masks.py tests\vision\test_stress_comic_pages.py -q
```

Expected result: all benchmark-focused tests pass.

- [ ] **Step 5: Commit only the benchmark change**

Run `git diff --check`, inspect the JSON schema in one generated ignored report, verify unrelated workspace changes are untouched, then commit with `feat: add paired vision benchmark`.

---

### Task 3: Add the lazy PreparedPage application adapter

**Files:**
- Create: `vision/app_adapter.py`
- Create: `tests/vision/test_app_adapter.py`
- Modify: `add_text.py:1057-1085` to add a pure appearance helper
- Create or modify: `tests/vision/test_app_integration.py`

**Interfaces:**
- Add `VisionPageAdapter(pipeline: VisionPipeline | None = None, config: VisionConfig | None = None)`.
- Add `VisionPageAdapter.prepare_page(image: np.ndarray, blocks: Iterable[Mapping[str, object]]) -> list[PreparedBlock]`.
- Add `VisionPageAdapter.erase_page(image: np.ndarray, prepared: list[PreparedBlock]) -> tuple[np.ndarray, list[EraseResult]]`.
- Add `VisionPageAdapter.process_page(image: np.ndarray, blocks: Iterable[Mapping[str, object]]) -> VisionPageExecution`; it calls the two methods above exactly once.
- Add `VisionPageExecution.prepared: list[PreparedBlock]`, `erased_image: np.ndarray`, and `erase_results: list[EraseResult]`.
- Add `build_optional_vision_adapter() -> VisionPageAdapter | None`; it returns `None` unless `MANGA_VISION_PIPELINE` is an explicit true value and logs a deterministic warning before returning `None` when runtime construction fails.
- Add pure `appearance_for_prepared(prepared: PreparedBlock) -> dict[str, object]` in `add_text.py`; it must not mutate the image or run inpainting.

- [ ] **Step 1: Write failing adapter and pure-appearance tests**

Add tests that prove the intended seam:

```python
def test_optional_adapter_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MANGA_VISION_PIPELINE", raising=False)
    assert build_optional_vision_adapter() is None


def test_process_page_prepares_and_erases_once():
    pipeline = CountingPipeline()
    adapter = VisionPageAdapter(pipeline=pipeline)
    execution = adapter.process_page(image, [{"text": "TEXT", "bbox": [4, 4, 20, 20]}])
    assert pipeline.prepare_calls == 1
    assert pipeline.erase_calls == 1
    assert len(execution.prepared) == 1


def test_appearance_for_prepared_does_not_change_pixels():
    before = image.copy()
    appearance = appearance_for_prepared(prepared_block)
    assert np.array_equal(image, before)
    assert "text_color" in appearance
```

Also test a fake pipeline exception with `caplog`: the factory must return `None` and log a warning, not return an object that can silently erase with an uninitialized model.

- [ ] **Step 2: Run adapter tests and confirm the expected red failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\vision\test_app_adapter.py tests\vision\test_app_integration.py -q
```

Expected result: import failure because the adapter and pure appearance helper do not exist.

- [ ] **Step 3: Implement the adapter without top-level model initialization**

Implement these rules:

1. Import the existing pipeline/types only; let `VisionPipeline` construct its lazy LaMa backend when the adapter is explicitly enabled.
2. `process_page` calls `prepare_page` once and `erase_page` once, returns the erased page and results, and never stores arrays in a session.
3. `build_optional_vision_adapter` parses only `1/true/yes/on`, loads `configs/vision.json`, and catches unavailable runtime/model errors. With `allow_cpu_fallback=true`, construct the CPU-safe pipeline and retain a warning; with fallback disabled, return `None` and a deterministic warning.
4. In `add_text.py`, refactor only the existing prepared appearance calculation into `appearance_for_prepared`; keep `erase_text_region(image, bbox, source_lang='ja', prepared=None)` and its legacy branch behavior unchanged. `VisionPageAdapter.prepare_page` and `.erase_page` delegate directly to the injected/current `VisionPipeline`.

- [ ] **Step 4: Run adapter and legacy regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\vision\test_app_adapter.py tests\vision\test_app_integration.py translator\test_translator.py tests\vision\test_pipeline.py -q
```

Expected result: adapter tests and all existing legacy/render regression tests pass.

- [ ] **Step 5: Commit only the adapter change**

Run `git diff --check`, confirm session/API files are unchanged, then commit with `feat: add opt-in prepared page adapter`.

---

### Task 4: Integrate the adapter behind a default-off flag

**Files:**
- Modify: `app.py::_do_full_pipeline` and `app.py::translate_and_render`
- Modify: `tests/test_app.py` if present, otherwise create `tests/test_app_vision_adapter.py`

**Interfaces:**
- Preserve all existing positional arguments to `translate_and_render`; add only a keyword-only `vision_adapter=None` parameter.
- Keep `_do_full_pipeline` returning the same processed result shape.
- Keep the nested renderer input tuple `(idx, name, image, blocks)` and output `{"name": name, "image": image}`.

- [ ] **Step 1: Write failing integration tests**

Test these observable contracts. The test module imports `types`, `Mock`, `numpy as np`, `pytest`, `app as app_module`, and `translate_and_render`, and defines this no-op counter so no external service or model is used:

```python
class CountingAdapter:
    def __init__(self):
        self.prepare_page_calls = 0
        self.erase_page_calls = 0

    def process_page(self, image, blocks):
        self.prepare_page_calls += 1
        self.erase_page_calls += 1
        return types.SimpleNamespace(prepared=[], erased_image=image.copy(), erase_results=[])

@pytest.fixture
def sample_results():
    image = np.full((40, 60, 3), 245, dtype=np.uint8)
    return [("page.png", image, [{"text": "TEXT", "bbox": [8, 8, 36, 28]}])]


@pytest.fixture
def fake_translator():
    class FakeGemini:
        def translate_batch(self, texts, source_lang, target_lang):
            return list(texts)

    return types.SimpleNamespace(_gemini_translator=FakeGemini())


def test_pipeline_flag_off_keeps_legacy_erase(monkeypatch, sample_results, fake_translator):
    monkeypatch.delenv("MANGA_VISION_PIPELINE", raising=False)
    legacy = Mock(return_value=(sample_results[0][1], (0, 0, 0), {"should_skip": False}))
    monkeypatch.setattr(app_module, "erase_text_region", legacy)
    monkeypatch.setattr(app_module, "render_all_blocks", lambda image, blocks, font: image)
    translate_and_render(sample_results, fake_translator, "Arial", "gemini", "ja", "vi", "default")
    assert legacy.call_count == 1


def test_pipeline_flag_on_prepares_each_page_once_and_preserves_result_keys(
    monkeypatch, sample_results, fake_translator
):
    adapter = CountingAdapter()
    monkeypatch.setattr(app_module, "render_all_blocks", lambda image, blocks, font: image)
    result = translate_and_render(
        sample_results, fake_translator, "Arial", "gemini", "ja", "vi", "default",
        vision_adapter=adapter,
    )
    assert adapter.prepare_page_calls == len(sample_results)
    assert adapter.erase_page_calls == len(sample_results)
    assert set(result[0]) == {"name", "image"}
```

Add a manual-correction test that calls `_do_full_pipeline` twice with changed bboxes and asserts the adapter receives a fresh page preparation on the second call. Verify Socket.IO progress event keys are still `phase/current/total/message/percent`.

- [ ] **Step 2: Run integration tests and confirm the expected red failure**

Run the exact focused command for the repository’s app test location. Expected result: the new keyword/adapter behavior is absent, while the legacy baseline tests continue to show their existing behavior.

- [ ] **Step 3: Implement the minimal default-off integration**

1. In `_do_full_pipeline`, call `build_optional_vision_adapter` once for the request and pass it as a keyword to `translate_and_render`.
2. In `translate_and_render`, leave the legacy loop untouched when `vision_adapter is None`.
3. When enabled, build the candidate block list using the exact current empty-text, invalid-bbox, and `should_skip_ocr_artifact` checks; process the page once through the adapter; map prepared blocks back to candidates by list order; derive appearance via `appearance_for_prepared`; call unchanged `render_all_blocks`.
4. Do not pass the adapter through session JSON. Since manual correction calls `_do_full_pipeline` again, masks are recreated from corrected image/bboxes automatically.
5. Log adapter warnings through existing warning/log channels without changing HTML, JSON, or Socket.IO payload keys.

- [ ] **Step 4: Run integration, session, and legacy regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_app_vision_adapter.py tests\vision translator\test_translator.py -q
```

If a listed path does not exist, run the repository’s exact equivalent and record the command rather than silently skipping it. Expected result: no legacy regression; opt-in adapter contract passes.

- [ ] **Step 5: Commit only the application integration**

Before commit, run `git diff --check` and GitNexus impact review for `translate_and_render` and the nested `render_single_image` path. Record the previously reported HIGH risk and confirm result/session contracts in the test output. Commit with `feat: gate prepared page flow behind opt-in flag`.

---

### Task 5: Harden low-risk routing and fallback behavior

**Files:**
- Modify: `vision/pipeline.py::_match_bubble` and its call in `VisionPipeline.prepare_page`
- Modify: `tests/vision/test_pipeline.py`
- Modify: `vision/pipeline.py::erase_page` only if the test exposes a config-radius regression

**Interfaces:**
- Keep `_match_bubble(bbox, bubbles, min_confidence=0.0) -> BubbleInstance | None` behavior compatible for callers; add the configured confidence/IoU gate without changing the returned type.
- Keep `EraseResult` fields and `erase_page(image, blocks) -> (image, results)` unchanged.

- [ ] **Step 1: Write failing safety tests**

```python
def test_match_bubble_rejects_low_confidence_overlap():
    mask = np.ones((20, 20), dtype=np.uint8)
    bubble = BubbleInstance("low", (0, 0, 20, 20), mask, 0.20, "speech_bubble")
    assert _match_bubble((5, 5, 15, 15), [bubble], min_confidence=0.45) is None


def test_erase_page_uses_configured_telea_radius_for_compatibility_fallback(monkeypatch):
    import vision.pipeline as pipeline_module

    calls = []

    def recording_inpaint(image, mask, radius, method):
        calls.append(radius)
        return image.copy()

    monkeypatch.setattr(pipeline_module.cv2, "inpaint", recording_inpaint)
    base = VisionConfig.load("configs/vision.json")
    config = replace(base, inpaint=replace(base.inpaint, telea_radius=7))
    pipeline = VisionPipeline(masker=HybridTextMasker(), lama_inpainter=None, config=config)
    pipeline.erase_page(
        np.zeros((20, 30, 3), dtype=np.uint8),
        [replace(_lama_block("complex", (2, 3, 8, 9)), erase_method="lama_full_page")],
    )
    assert calls == [7]
```

- [ ] **Step 2: Run the safety tests and confirm the expected red failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\vision\test_pipeline.py -k "low_confidence or configured_telea_radius" -q
```

Expected result: the low-confidence bubble currently matches and the fallback path currently hard-codes radius 3, so at least one assertion fails.

- [ ] **Step 3: Implement only the low-risk guards**

Pass `self.config.bubble.match_confidence` to the matcher and require both bubble confidence and positive IoU before returning a bubble. Use `self.config.inpaint.telea_radius` in page-level compatibility fallback. Do not alter the CRITICAL masker/filter functions.

- [ ] **Step 4: Run all pipeline and inpainting tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\vision\test_pipeline.py tests\vision\test_lama_inpainter.py tests\vision\test_hybrid_masker.py -q
```

Expected result: all existing and new safety tests pass.

- [ ] **Step 5: Commit only the low-risk hardening**

Run GitNexus impact for `_match_bubble` and `erase_page`, run `git diff --check`, and commit with `fix: enforce vision routing safety gates`.

---

### Task 6: Run the paired before/after proof and decide promotion

**Files:**
- Create: `docs/vision/BENCHMARKS.md`
- Generated ignored outputs: `reports/vision/optimization-before.json`, `reports/vision/optimization-after.json`
- No production source changes in this task.

**Interfaces:**
- The benchmark command is reproducible from the repository root and writes JSON reports outside Git.
- The document reports both achievements and non-achievements; it must not call an unmeasured backend "optimal".

- [ ] **Step 1: Run the complete regression suite before benchmarking**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\vision translator\test_translator.py -q
```

Record exit code, passed/failed count, and runtime.

- [ ] **Step 2: Run identical legacy and prepared synthetic evaluations**

Run:

```powershell
.\.venv\Scripts\python.exe -B tools\benchmark_vision_pipeline.py --manifest datasets\local\vision-eval-v1\manifest.jsonl --config configs\vision.json --mode legacy --backend heuristic --output reports\vision\optimization-before.json
.\.venv\Scripts\python.exe -B tools\benchmark_vision_pipeline.py --manifest datasets\local\vision-eval-v1\manifest.jsonl --config configs\vision.json --mode prepared --backend heuristic --output reports\vision\optimization-after.json
```

Use all 1,000 rows, the fixed manifest hash `2825b96a40eea6db9b811df59969a6088b2e14b016d8b1052b1b836530e08d8e`, at least five warmups when a warmup mode is supported, and report paired per-row deltas. Do not claim GPU/LaMa results on the current host because torch is missing and CUDA devices=0.

- [ ] **Step 3: Run identical operational stress evaluations**

Run the prepared and legacy modes on the same 500-page `datasets/local/comix-v0/stress-manifest.jsonl` sample selection. Report decode/prepare/erase/render stage timings, p50/p95, errors, fallback counts, and `annotation_semantics=detector-proposal`. Include the 10-page representative sample used by the baseline only as a smoke check, and run a larger fixed set before any promotion.

- [ ] **Step 4: Apply explicit gates**

Mark each gate `pass`, `fail`, or `insufficient_evidence`:

- zero crashes/corrupt outputs;
- outside predicted mask delta equals zero on every quality sample;
- bubble-border damage below 1% where bubble boundary gold exists;
- simple uniform bubbles invoke LaMa zero times;
- hybrid relative hard-subset Dice gain is at least 15% with defined denominator, otherwise remain experimental;
- paired p50 must not regress more than 10%, p95 not more than 15% on the same host;
- peak memory must not regress more than 10% when the measurement is available;
- masked residual/background/edge metrics meet the recorded caps, or remain insufficient when no gold set exists.

Do not change production config to enable a backend when any required gate is `fail` or `insufficient_evidence`.

- [ ] **Step 5: Write the evidence document**

`docs/vision/BENCHMARKS.md` must include commands, commit SHA, dataset/config hashes, environment, stage timing tables, quality tables by category, safety/fallback counts, limitations, and an explicit promotion/rollback decision. State that COMIX is operational-only and that the current no-torch/no-CUDA host cannot prove LaMa quality.

- [ ] **Step 6: Run final verification and impact analysis**

Run `git diff --check`, `git status --short`, the complete focused/regression test command, and `mcp__gitnexus__detect_changes({scope: "all"})`. Verify that only intended source/docs changes are present; never stage the pre-existing workspace changes. If no backend passes the evidence gates, leave the feature flag default OFF and report that fact rather than forcing a promotion.

---

## Verification checklist

- [ ] Every new production behavior has a failing test observed before implementation.
- [ ] GitNexus impact was run before each existing-symbol edit; HIGH/CRITICAL risks were explicitly reviewed.
- [ ] Default-off legacy behavior and manual-correction/session contracts remain tested.
- [ ] Benchmark reports use paired identical samples and separate stage timings.
- [ ] Synthetic quality evidence, COMIX operational evidence, and missing human-gold evidence are clearly separated.
- [ ] A fresh full regression and fresh benchmark were run before any completion claim.
- [ ] The goal is marked complete only if the speed and erasure objectives are actually supported by evidence; otherwise the goal remains active with the concrete next gate.
