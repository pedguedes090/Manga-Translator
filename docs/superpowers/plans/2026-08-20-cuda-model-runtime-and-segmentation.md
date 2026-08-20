# CUDA model runtime and segmentation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add verified model delivery, explicit CUDA inference, neural text masks, and page-level bubble segmentation without making ML dependencies part of the base installation.

**Architecture:** ONNX Runtime handles text and bubble segmentation through a small session adapter that verifies active providers. Reproducible PyTorch training and export live under `training/`; production weights are published as versioned release assets and downloaded only after size and SHA-256 validation.

**Tech Stack:** Python 3.10/3.11, PyTorch CUDA, torchvision ResNet34, ONNX, ONNX Runtime GPU, NumPy, OpenCV, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-full-vision-pipeline-design.md`

## Global constraints

- CUDA is the primary ML runtime; CPU fallback must be explicit in logs and metadata.
- Base `requirements.txt` remains usable without PyTorch or ONNX Runtime.
- Model weights, training data, and generated reports are not committed to Git.
- Every downloaded model is checked against exact size and SHA-256 before loading.
- Comic Text Detector remains an external GPL-3.0 benchmark; do not copy its code.
- A production artifact is registered only when its dataset license and evaluation gate permit distribution.
- Complete each task in one commit and push directly to `origin/main` after focused and regression tests pass.
- Never force-push or stage unrelated working-tree changes.

## File map

| Path | Responsibility |
| --- | --- |
| `requirements-vision-cuda.txt` | CUDA inference dependencies. |
| `requirements-training.txt` | Training and ONNX export dependencies. |
| `models/manifest.json` | Versioned artifact metadata. |
| `models/NOTICE.md` | Model source and license notices. |
| `vision/model_registry.py` | Atomic verified download and cache lookup. |
| `vision/runtime.py` | Lazy ONNX Runtime import, provider validation, and session creation. |
| `vision/maskers/neural.py` | OCR-crop preprocessing and probability-map inference. |
| `vision/bubbles/base.py` | Bubble segmenter protocol and OCR matching. |
| `vision/bubbles/onnx_segmenter.py` | Page-level semantic segmentation and instances. |
| `training/dataset.py` | JSONL segmentation dataset loader. |
| `training/models.py` | ResNet34 U-Net shared by text and bubble training. |
| `training/train_text_mask.py` | Deterministic binary text-mask training. |
| `training/train_bubble_seg.py` | Deterministic four-class bubble training. |
| `tools/export_onnx.py` | Export and parity verification. |
| `tools/register_model.py` | Compute artifact metadata and update the manifest. |

---

### Task 1: Add verified model registry and dependency profiles

**Files:**
- Create: `requirements-vision-cuda.txt`
- Create: `requirements-training.txt`
- Create: `models/manifest.json`
- Create: `models/NOTICE.md`
- Create: `vision/model_registry.py`
- Create: `tools/register_model.py`
- Create: `tests/vision/test_model_registry.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `ModelSpec.from_dict(data) -> ModelSpec`.
- Produces: `ModelRegistry.ensure(name: str) -> Path`.
- Produces: `register_model(manifest_path, artifact_path, name, version, url, license_id, source, input_spec) -> ModelSpec`.

- [ ] **Step 1: Write atomic download and checksum tests**

```python
import hashlib
import json

import pytest

from vision.model_registry import ModelRegistry


def test_registry_downloads_and_verifies_local_artifact(tmp_path):
    source = tmp_path / "source.onnx"
    source.write_bytes(b"verified-model")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "models": [{
        "name": "text-mask", "version": "1.0.0", "url": source.as_uri(),
        "sha256": digest, "size_bytes": source.stat().st_size,
        "license": "MIT", "source": "test fixture", "input": {"layout": "NCHW"},
    }]}), encoding="utf-8")
    path = ModelRegistry(manifest, tmp_path / "cache").ensure("text-mask")
    assert path.read_bytes() == b"verified-model"


def test_registry_rejects_checksum_mismatch(tmp_path):
    source = tmp_path / "source.onnx"
    source.write_bytes(b"corrupt")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "models": [{
        "name": "text-mask", "version": "1.0.0", "url": source.as_uri(),
        "sha256": "0" * 64, "size_bytes": source.stat().st_size,
        "license": "MIT", "source": "test fixture", "input": {"layout": "NCHW"},
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ModelRegistry(manifest, tmp_path / "cache").ensure("text-mask")
```

- [ ] **Step 2: Run the test and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_model_registry.py -q`

Expected: FAIL because `vision.model_registry` does not exist.

- [ ] **Step 3: Implement registry, registration tool, and dependency separation**

Initialize the manifest as `{"schema_version": 1, "models": []}`. Download through `urllib.request`, write `<artifact>.part`, verify exact byte length and SHA-256, then replace atomically with `Path.replace()`. The registration tool computes size/hash from bytes; it never accepts caller-provided digest values. Add `models/cache/`, `training_runs/`, and `*.onnx.part` to `.gitignore`.

Pin package versions only after resolving a compatible set on Python 3.11 and the target CUDA runtime. Record the resolved versions in these files in the same commit; do not use unbounded dependency specifiers.

- [ ] **Step 4: Run focused and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_model_registry.py translator\test_translator.py -q`

Expected: PASS and no artifact remains with a `.part` suffix.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add .gitignore requirements-vision-cuda.txt requirements-training.txt models\manifest.json models\NOTICE.md vision\model_registry.py tools\register_model.py tests\vision\test_model_registry.py
git commit -m "feat: add verified vision model registry"
git push origin main
```

### Task 2: Create ONNX sessions with explicit provider verification

**Files:**
- Create: `vision/runtime.py`
- Create: `tests/vision/test_runtime.py`

**Interfaces:**
- Produces: `create_onnx_session(model_path: Path, device: str, allow_cpu_fallback: bool) -> OnnxSession`.
- Produces: `OnnxSession.run(inputs: dict[str, np.ndarray]) -> list[np.ndarray]` and `OnnxSession.providers: tuple[str, ...]`.

- [ ] **Step 1: Write provider selection tests with a fake runtime**

```python
import pytest

from vision.runtime import create_onnx_session


class FakeOrt:
    def __init__(self, available):
        self.available = available
        self.requested = None

    def get_available_providers(self):
        return self.available

    def InferenceSession(self, path, providers):
        self.requested = providers
        return type("Session", (), {
            "get_providers": lambda self: providers,
            "run": lambda self, names, inputs: [],
        })()


def test_cuda_session_requests_cuda_before_cpu(tmp_path):
    ort = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
    session = create_onnx_session(tmp_path / "model.onnx", "cuda", True, ort_module=ort)
    assert session.providers[0] == "CUDAExecutionProvider"


def test_cuda_session_fails_closed_when_fallback_disabled(tmp_path):
    ort = FakeOrt(["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider is unavailable"):
        create_onnx_session(tmp_path / "model.onnx", "cuda", False, ort_module=ort)
```

- [ ] **Step 2: Run the test and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_runtime.py -q`

Expected: FAIL because `vision.runtime` does not exist.

- [ ] **Step 3: Implement lazy imports and provider metadata**

Import `onnxruntime` only inside the factory. For CUDA, call `onnxruntime.preload_dlls()` when available, request CUDA then CPU, and verify the returned session's active provider order. Return an adapter rather than exposing the library session directly. Include device and provider names in runtime debug metadata.

- [ ] **Step 4: Run focused and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_runtime.py translator\test_translator.py -q`

Expected: PASS without installing ONNX Runtime in the base environment because tests inject `FakeOrt`.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add vision\runtime.py tests\vision\test_runtime.py
git commit -m "feat: validate ONNX CUDA providers"
git push origin main
```

### Task 3: Add the neural text-mask inference backend

**Files:**
- Create: `vision/maskers/neural.py`
- Create: `tests/vision/test_neural_masker.py`
- Modify: `vision/pipeline.py`

**Interfaces:**
- Produces: `NeuralTextMasker(session: OnnxSession, config: TextMaskConfig | None = None)` using approved defaults when config is omitted.
- Produces: `NeuralTextMasker.generate(image: np.ndarray, bbox: BBox, text: str, region: RegionAnalysis, bubble: BubbleInstance | None) -> MaskResult`.
- Consumes: `OnnxSession`, `VisionConfig.text_mask`, and postprocessors from the foundation plan.

- [ ] **Step 1: Write preprocessing, mapping, and invalid-output tests**

```python
import numpy as np
import pytest

from vision.maskers.neural import NeuralTextMasker
from vision.region_analysis import analyze_region


class FakeTextSession:
    providers = ("CUDAExecutionProvider", "CPUExecutionProvider")

    def run(self, inputs):
        tensor = next(iter(inputs.values()))
        height, width = tensor.shape[2:]
        probability = np.zeros((1, 1, height, width), np.float32)
        probability[:, :, height // 3:2 * height // 3, width // 3:2 * width // 3] = 0.9
        return [probability]


def test_neural_mask_maps_probability_back_to_roi():
    image = np.full((100, 200, 3), 245, np.uint8)
    bbox = (50, 25, 150, 75)
    result = NeuralTextMasker(FakeTextSession()).generate(
        image, bbox, "TEXT", analyze_region(image, bbox), None,
    )
    assert result.mask.shape == (50, 100)
    assert result.probability.shape == (50, 100)
    assert result.backend == "neural"


def test_neural_mask_rejects_non_finite_output():
    session = FakeTextSession()
    session.run = lambda inputs: [np.full((1, 1, 512, 512), np.nan, np.float32)]
    with pytest.raises(ValueError, match="non-finite"):
        NeuralTextMasker(session).generate(
            np.zeros((40, 40, 3), np.uint8), (5, 5, 35, 35), "X",
            analyze_region(np.zeros((40, 40, 3), np.uint8), (5, 5, 35, 35)), None,
        )
```

- [ ] **Step 2: Run the test and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_neural_masker.py -q`

Expected: FAIL because `vision.maskers.neural` does not exist.

- [ ] **Step 3: Implement crop inference and fallback signaling**

Expand the OCR bbox by configured 12 percent, clamp to the image, letterbox to 512 square pixels, normalize BGR to RGB float NCHW, run the session, validate one finite single-channel probability output, undo letterbox, resize to ROI coordinates, and call the shared postprocessors. Raise `ValueError` for invalid outputs so `VisionPipeline` can run `HybridTextMasker` and record the reason.

- [ ] **Step 4: Run focused and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_neural_masker.py tests\vision\test_pipeline.py translator\test_translator.py -q`

Expected: PASS and the fake session reports CUDA as the first provider.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add vision\maskers\neural.py vision\pipeline.py tests\vision\test_neural_masker.py
git commit -m "feat: add neural text mask inference"
git push origin main
```

### Task 4: Build reproducible segmentation datasets and ResNet34 U-Net

**Files:**
- Create: `training/__init__.py`
- Create: `training/dataset.py`
- Create: `training/models.py`
- Create: `tests/vision/test_training_components.py`

**Interfaces:**
- Produces: `SegmentationDataset(manifest_path, classes, image_size, augment, seed)`.
- Produces: `ResNet34UNet(out_channels: int) -> torch.nn.Module` with NCHW logits at input spatial size.

- [ ] **Step 1: Write a tiny dataset/model smoke test**

```python
import json

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from training.dataset import SegmentationDataset
from training.models import ResNet34UNet


def test_training_dataset_and_model_shapes(tmp_path):
    image = np.full((32, 48, 3), 255, np.uint8)
    mask = np.zeros((32, 48), np.uint8)
    mask[8:24, 16:32] = 255
    cv2.imwrite(str(tmp_path / "image.png"), image)
    cv2.imwrite(str(tmp_path / "mask.png"), mask)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({
        "image": "image.png", "text_mask": "mask.png", "split": "train",
        "category": "synthetic", "language": "en", "source": "generated", "license": "project",
    }) + "\n", encoding="utf-8")
    dataset = SegmentationDataset(manifest, classes=1, image_size=64, augment=False, seed=7)
    batch_image, batch_mask = dataset[0]
    output = ResNet34UNet(out_channels=1)(batch_image.unsqueeze(0))
    assert output.shape == (1, 1, 64, 64)
    assert batch_mask.shape == (1, 64, 64)
```

- [ ] **Step 2: Run in the CUDA/training environment and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_training_components.py -q`

Expected: SKIP in the base environment or FAIL in the training environment because the modules do not exist.

- [ ] **Step 3: Implement deterministic loading and the shared model**

Resolve manifest-relative paths, reject missing license/source fields, use nearest-neighbor interpolation for masks, bilinear interpolation for images, and apply seeded geometric transforms identically to image and mask. Build a torchvision ResNet34 encoder with no pretrained download and a four-stage decoder with skip connections; final logits must match input size.

- [ ] **Step 4: Run the smoke test in the training environment**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_training_components.py -q`

Expected: PASS when training dependencies are installed; base CI records one intentional skip.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add training\__init__.py training\dataset.py training\models.py tests\vision\test_training_components.py
git commit -m "feat: add segmentation training components"
git push origin main
```

### Task 5: Train, export, and gate the text segmentation model

**Files:**
- Create: `training/train_text_mask.py`
- Create: `tools/export_onnx.py`
- Create: `tests/vision/test_onnx_export.py`
- Modify: `models/NOTICE.md`
- Modify: `models/manifest.json`
- Modify: `configs/vision.json`

**Interfaces:**
- Produces: `train_text_mask(config) -> Path` for the best checkpoint.
- Produces: `export_and_compare(checkpoint, output, input_size, out_channels) -> dict[str, float]`.
- Produces: release asset `text-mask-resnet34-v1.0.0.onnx` when quality and licensing gates pass.

- [ ] **Step 1: Write an export parity test**

```python
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from tools.export_onnx import export_and_compare
from training.models import ResNet34UNet


def test_exported_text_model_matches_torch(tmp_path):
    model = ResNet34UNet(out_channels=1).eval()
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "out_channels": 1}, checkpoint)
    report = export_and_compare(checkpoint, tmp_path / "model.onnx", 64, 1)
    assert report["max_abs_error"] <= 1e-4
```

- [ ] **Step 2: Run the test and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_onnx_export.py -q`

Expected: FAIL in the training environment because the exporter does not exist.

- [ ] **Step 3: Implement deterministic training and export**

Train with BCE-with-logits plus soft Dice loss, AdamW, fixed seed `20260820`, best validation Dice checkpointing, and JSON metrics history. Export with dynamic height/width axes disabled for the 512 text model, then compare three seeded inputs between PyTorch and ONNX Runtime. Load checkpoints with `weights_only=True` where supported.

- [ ] **Step 4: Generate training data, train, benchmark, and register only on success**

Run:

```powershell
.\.venv\Scripts\python.exe tools\generate_synthetic_dataset.py --output datasets\generated\text-mask-v1 --samples 5000 --seed 20260820
.\.venv\Scripts\python.exe training\train_text_mask.py --manifest datasets\generated\text-mask-v1\manifest.jsonl --output training_runs\text-mask-v1 --seed 20260820 --epochs 40 --batch-size 8 --device cuda
.\.venv\Scripts\python.exe tools\export_onnx.py --checkpoint training_runs\text-mask-v1\best.pt --output training_runs\text-mask-v1\text-mask-resnet34-v1.0.0.onnx --input-size 512 --out-channels 1
.\.venv\Scripts\python.exe tools\evaluate_masks.py --manifest debug_outputs\vision_baseline\manifest.jsonl --backend neural --config configs\vision.json --output reports\vision\neural-text-v1.json
```

Expected: export parity at or below `1e-4`; neural Dice exceeds hybrid without reducing artwork-preservation precision. If the quality gate fails, keep `neural_gate_passed=false`, retain the report, and do not publish/register the artifact.

- [ ] **Step 5: Publish and register the passing artifact**

After the gate passes and the dataset manifest confirms distributable licenses:

```powershell
gh release create vision-models-v1.0.0 training_runs\text-mask-v1\text-mask-resnet34-v1.0.0.onnx --repo pedguedes090/Manga-Translator --title "Vision models v1.0.0" --notes "Text-mask ResNet34 U-Net and provenance are recorded in models/NOTICE.md."
.\.venv\Scripts\python.exe tools\register_model.py --manifest models\manifest.json --artifact training_runs\text-mask-v1\text-mask-resnet34-v1.0.0.onnx --name text-mask-resnet34 --version 1.0.0 --url https://github.com/pedguedes090/Manga-Translator/releases/download/vision-models-v1.0.0/text-mask-resnet34-v1.0.0.onnx --license MIT --source manga-text-segmentation-and-project-synthetic --input-size 512 --layout NCHW
```

- [ ] **Step 6: Run focused and regression tests, then commit and push**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_onnx_export.py tests\vision\test_neural_masker.py translator\test_translator.py -q`

Expected: PASS in the CUDA/training environment.

```powershell
git fetch origin
git add training\train_text_mask.py tools\export_onnx.py tests\vision\test_onnx_export.py models\NOTICE.md models\manifest.json configs\vision.json
git commit -m "feat: train and register neural text masker"
git push origin main
```

### Task 6: Train and export bubble semantic segmentation

**Files:**
- Create: `training/train_bubble_seg.py`
- Create: `tests/vision/test_bubble_training.py`
- Modify: `tools/generate_synthetic_dataset.py`
- Modify: `models/NOTICE.md`
- Modify: `models/manifest.json`

**Interfaces:**
- Produces: four-class logits for background, speech bubble, thought bubble, and caption box.
- Produces: release asset `bubble-seg-resnet34-v1.0.0.onnx` after its validation gate passes.

- [ ] **Step 1: Write class-preserving dataset and model tests**

```python
import pytest

torch = pytest.importorskip("torch")
from training.models import ResNet34UNet


def test_bubble_model_emits_four_class_logits():
    output = ResNet34UNet(out_channels=4)(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 4, 64, 64)
```

- [ ] **Step 2: Run the focused test**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_bubble_training.py -q`

Expected: FAIL until the test file and bubble training entry point exist.

- [ ] **Step 3: Implement multiclass training and synthetic annotations**

Extend the generator to draw all three approved bubble categories with exact class-index masks. Train with cross-entropy plus multiclass Dice, fixed seed `20260820`, and best mean foreground IoU checkpointing. Export at 1024 square input and record per-class IoU.

- [ ] **Step 4: Train, export, and register after the gate**

Run:

```powershell
.\.venv\Scripts\python.exe tools\generate_synthetic_dataset.py --output datasets\generated\bubble-v1 --samples 5000 --seed 20260820 --bubble-classes
.\.venv\Scripts\python.exe training\train_bubble_seg.py --manifest datasets\generated\bubble-v1\manifest.jsonl --output training_runs\bubble-v1 --seed 20260820 --epochs 40 --batch-size 4 --device cuda
.\.venv\Scripts\python.exe tools\export_onnx.py --checkpoint training_runs\bubble-v1\best.pt --output training_runs\bubble-v1\bubble-seg-resnet34-v1.0.0.onnx --input-size 1024 --out-channels 4
gh release upload vision-models-v1.0.0 training_runs\bubble-v1\bubble-seg-resnet34-v1.0.0.onnx --repo pedguedes090/Manga-Translator
.\.venv\Scripts\python.exe tools\register_model.py --manifest models\manifest.json --artifact training_runs\bubble-v1\bubble-seg-resnet34-v1.0.0.onnx --name bubble-seg-resnet34 --version 1.0.0 --url https://github.com/pedguedes090/Manga-Translator/releases/download/vision-models-v1.0.0/bubble-seg-resnet34-v1.0.0.onnx --license MIT --source project-synthetic-and-licensed-bubble-masks --input-size 1024 --layout NCHW
```

Expected: export parity at or below `1e-4`; validation report contains per-class IoU and source-license summary.

- [ ] **Step 5: Run tests, commit, and push**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_bubble_training.py tests\vision\test_model_registry.py translator\test_translator.py -q`

Expected: PASS.

```powershell
git fetch origin
git add training\train_bubble_seg.py tests\vision\test_bubble_training.py tools\generate_synthetic_dataset.py models\NOTICE.md models\manifest.json
git commit -m "feat: train and register bubble segmenter"
git push origin main
```

### Task 7: Infer bubble instances once per page and match OCR blocks

**Files:**
- Create: `vision/bubbles/__init__.py`
- Create: `vision/bubbles/base.py`
- Create: `vision/bubbles/onnx_segmenter.py`
- Create: `tests/vision/test_bubble_segmenter.py`
- Modify: `vision/pipeline.py`

**Interfaces:**
- Produces: `OnnxBubbleSegmenter.segment(image) -> list[BubbleInstance]`.
- Produces: `match_bubble(bbox: BBox, instances: list[BubbleInstance], min_confidence: float) -> BubbleInstance | None`.
- Changes: `VisionPipeline.prepare_page()` calls `segment()` exactly once per page.

- [ ] **Step 1: Write instance and call-count tests**

```python
from unittest.mock import Mock

import numpy as np

from vision.bubbles.base import match_bubble
from vision.bubbles.onnx_segmenter import OnnxBubbleSegmenter
from vision.pipeline import VisionPipeline


class FakeBubbleSession:
    providers = ("CUDAExecutionProvider", "CPUExecutionProvider")

    def run(self, inputs):
        logits = np.zeros((1, 4, 64, 64), np.float32)
        logits[:, 1, 10:50, 8:56] = 10.0
        return [logits]


def test_bubble_segmentation_returns_matchable_instance():
    instances = OnnxBubbleSegmenter(FakeBubbleSession(), input_size=64).segment(
        np.full((128, 128, 3), 255, np.uint8)
    )
    matched = match_bubble((30, 30, 90, 80), instances, min_confidence=0.45)
    assert matched is not None
    assert matched.category == "speech_bubble"


def test_pipeline_segments_bubbles_once_for_multiple_blocks():
    segmenter = Mock(wraps=OnnxBubbleSegmenter(FakeBubbleSession(), input_size=64))
    pipeline = VisionPipeline(bubble_segmenter=segmenter)
    pipeline.prepare_page(np.full((128, 128, 3), 255, np.uint8), [
        {"text": "A", "bbox": [20, 20, 60, 50]},
        {"text": "B", "bbox": [60, 50, 100, 90]},
    ])
    assert segmenter.segment.call_count == 1
```

- [ ] **Step 2: Run the tests and observe import failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision\test_bubble_segmenter.py -q`

Expected: FAIL because the bubble package does not exist.

- [ ] **Step 3: Implement page inference, instances, and matching**

Letterbox the page to model input, map the argmax class map back to native size with nearest-neighbor interpolation, find connected components per foreground class, and compute instance confidence from softmax probability inside each component. Implement the approved `0.50/0.30/0.20` center, intersection, and proximity score. Apply a hard gate only when match confidence meets config and `bubble_gate_passed` is true; before that, record matches for evaluation without constraining masks.

- [ ] **Step 4: Run all model-runtime and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\vision translator\test_translator.py -q`

Expected: PASS; page segmentation call count is one.

- [ ] **Step 5: Commit and push**

```powershell
git fetch origin
git add vision\bubbles\__init__.py vision\bubbles\base.py vision\bubbles\onnx_segmenter.py tests\vision\test_bubble_segmenter.py vision\pipeline.py
git commit -m "feat: gate text masks with bubble instances"
git push origin main
```
