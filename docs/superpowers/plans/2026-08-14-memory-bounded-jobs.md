# File-Backed, Memory-Bounded Jobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace whole-job image arrays and base64 result payloads with UUID-scoped disk artifacts, OCR chunks of at most two images by default, and sequential one-image rendering.

**Architecture:** A focused `job_storage.py` owns manifests, validated artifact paths, atomic metadata writes, and TTL cleanup. A `job_pipeline.py` decodes images only in bounded OCR chunks and renders one source image at a time to disk. Flask routes exchange job/page identifiers and URLs; correction metadata remains JSON, result images and ZIP archives are served from validated paths.

**Tech Stack:** Python 3.11, Flask, OpenCV/NumPy, Pillow, pytest, filesystem-backed temporary jobs.

## Global Constraints

- Execute the completed `2026-08-14-i18n-and-prompts.md` plan first; this plan consumes `ui_language`, `translate()`, stable selection codes, semantic progress keys, and `prompt_locale`.
- Store job images under the existing `temp_sessions` root in UUID directories.
- Store no OpenCV arrays, PIL images, base64 strings, or copied original images in cross-request memory.
- `OCR_BATCH_SIZE` defaults to `2`, accepts positive integers only, and is capped at `8`.
- Provider-side OCR concurrency may not exceed the current OCR chunk size.
- Rendering concurrency is exactly `1`.
- Correction and result HTML contains image URLs, never image data URLs.
- Multi-image ZIP generation writes to disk, never a complete `BytesIO` archive.
- The existing six-hour session TTL remains the default.
- Preserve unrelated existing working-tree deletions.

---

### Task 1: UUID job store and atomic manifest lifecycle

**Files:**
- Create: `job_storage.py`
- Create: `tests/test_job_storage.py`
- Modify: `app.py:78-267`

**Interfaces:**
- Produces: `JobStore(root: str, ttl_seconds: int)`.
- Produces: `create(locale: str, settings: dict) -> str`.
- Produces: `load(job_id: str) -> dict | None` and `save(job_id: str, manifest: dict) -> None`.
- Produces: `source_path(job_id: str, page_index: int) -> pathlib.Path`.
- Produces: `result_path(job_id: str, page_index: int) -> pathlib.Path`.
- Produces: `zip_path(job_id: str) -> pathlib.Path`.
- Produces: `cleanup_expired(now: float | None = None) -> list[str]`.

- [ ] **Step 1: Write failing job-store tests**

Create `tests/test_job_storage.py`:

```python
import json
import os
import time
import uuid

import pytest

from job_storage import JobStore


def test_create_and_atomic_manifest_roundtrip(tmp_path):
    store = JobStore(tmp_path, ttl_seconds=60)
    job_id = store.create("en", {"source_lang": "ja", "target_lang": "vi"})
    manifest = store.load(job_id)
    assert str(uuid.UUID(job_id)) == job_id
    assert manifest["version"] == 2
    assert manifest["locale"] == "en"
    assert manifest["settings"]["target_lang"] == "vi"
    assert manifest["pages"] == []

    manifest["pages"].append({"index": 0, "name": "page", "blocks": []})
    store.save(job_id, manifest)
    assert store.load(job_id)["pages"][0]["name"] == "page"
    assert not (tmp_path / job_id / "manifest.json.tmp").exists()


def test_artifact_paths_reject_invalid_job_and_page(tmp_path):
    store = JobStore(tmp_path, ttl_seconds=60)
    with pytest.raises(ValueError):
        store.source_path("../../escape", 0)
    job_id = store.create("en", {})
    with pytest.raises(ValueError):
        store.source_path(job_id, -1)
    assert store.source_path(job_id, 2).name == "source_0002.jpg"
    assert store.result_path(job_id, 2).name == "result_0002.jpg"


def test_cleanup_removes_only_expired_uuid_jobs(tmp_path):
    store = JobStore(tmp_path, ttl_seconds=10)
    expired = store.create("en", {})
    active = store.create("vi", {})
    ignored = tmp_path / "notes"
    ignored.mkdir()
    old = time.time() - 20
    os.utime(tmp_path / expired, (old, old))

    removed = store.cleanup_expired(now=time.time())

    assert removed == [expired]
    assert not (tmp_path / expired).exists()
    assert (tmp_path / active).exists()
    assert ignored.exists()
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_job_storage.py -q
```

Expected: import failure for `job_storage`.

- [ ] **Step 3: Implement the manifest schema and path validation**

Create `job_storage.py` around `pathlib.Path`:

```python
import json
import os
import shutil
import time
import uuid
from pathlib import Path


class JobStore:
    def __init__(self, root, ttl_seconds):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = int(ttl_seconds)

    def _id(self, job_id):
        try:
            return str(uuid.UUID(str(job_id)))
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError("invalid job id") from error

    def directory(self, job_id):
        return self.root / self._id(job_id)

    def _page_index(self, value):
        index = int(value)
        if index < 0:
            raise ValueError("invalid page index")
        return index

    def source_path(self, job_id, page_index):
        return self.directory(job_id) / f"source_{self._page_index(page_index):04d}.jpg"

    def result_path(self, job_id, page_index):
        return self.directory(job_id) / f"result_{self._page_index(page_index):04d}.jpg"

    def zip_path(self, job_id):
        return self.directory(job_id) / "translated.zip"

    def create(self, locale, settings):
        job_id = str(uuid.uuid4())
        self.directory(job_id).mkdir(parents=False)
        now = time.time()
        self.save(job_id, {"version": 2, "job_id": job_id, "locale": locale, "created_at": now, "updated_at": now, "settings": dict(settings), "pages": [], "status": "created", "warning_key": None})
        return job_id

    def load(self, job_id):
        path = self.directory(job_id) / "manifest.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def save(self, job_id, manifest):
        directory = self.directory(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        data = dict(manifest)
        data["updated_at"] = time.time()
        temporary = directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, directory / "manifest.json")

    def cleanup_expired(self, now=None):
        cutoff = (time.time() if now is None else now) - self.ttl_seconds
        removed = []
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            try:
                job_id = self._id(child.name)
            except ValueError:
                continue
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child)
                removed.append(job_id)
        return sorted(removed)
```

- [ ] **Step 4: Replace the array cache with one configured store**

In `app.py`, keep `TEMP_DIR` and `SESSION_TTL_SECONDS`, instantiate:

```python
job_store = JobStore(TEMP_DIR, SESSION_TTL_SECONDS)
```

Change `cleanup_old_sessions()` to call `job_store.cleanup_expired()`. Remove `ocr_sessions`, `MAX_MEMORY_SESSIONS`, `_save_session()`, `load_session()`, `_session_json_path()`, and `_session_image_path()`. Do not add another global dictionary.

- [ ] **Step 5: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_job_storage.py -q
git add -- job_storage.py tests/test_job_storage.py app.py
git commit -m "refactor: persist lightweight translation job manifests"
```

Expected: all job-store tests pass.

---

### Task 2: Stream uploads to disk and OCR in bounded chunks

**Files:**
- Create: `job_pipeline.py`
- Create: `tests/test_bounded_ocr.py`
- Modify: `app.py:639-806`
- Verify: `ocr/chrome_lens_ocr.py:355-414`

**Interfaces:**
- Produces: `configured_ocr_batch_size(environ=os.environ) -> int`.
- Produces: `iter_chunks(items: list, size: int)`.
- Produces: `ocr_job(store, job_id, source_lang, batch_size, ocr_factory, filter_blocks, emit_progress) -> dict`.
- Consumes: manifest pages containing `index`, `name`, `width`, `height`, and `blocks`.

- [ ] **Step 1: Write failing batch-bound tests**

Create `tests/test_bounded_ocr.py`:

```python
import cv2
import numpy as np

from job_pipeline import configured_ocr_batch_size, ocr_job
from job_storage import JobStore


def test_ocr_batch_size_default_validation_and_cap():
    assert configured_ocr_batch_size({}) == 2
    assert configured_ocr_batch_size({"OCR_BATCH_SIZE": "1"}) == 1
    assert configured_ocr_batch_size({"OCR_BATCH_SIZE": "99"}) == 8
    assert configured_ocr_batch_size({"OCR_BATCH_SIZE": "bad"}) == 2
    assert configured_ocr_batch_size({"OCR_BATCH_SIZE": "0"}) == 2


def test_ocr_never_decodes_more_than_configured_chunk(tmp_path, monkeypatch):
    store = JobStore(tmp_path, ttl_seconds=60)
    job_id = store.create("en", {"source_lang": "ja"})
    manifest = store.load(job_id)
    for index in range(5):
        cv2.imwrite(str(store.source_path(job_id, index)), np.full((8, 8, 3), 255, np.uint8))
        manifest["pages"].append({"index": index, "name": f"p{index}", "width": 8, "height": 8, "blocks": []})
    store.save(job_id, manifest)

    observed = []
    class FakeOCR:
        def process_batch(self, images):
            observed.append(len(images))
            return [[{"text": str(i), "bbox": [0, 0, 4, 4]}] for i in range(len(images))]
        def __call__(self, image):
            observed.append(1)
            return [{"text": "x", "bbox": [0, 0, 4, 4]}]

    result = ocr_job(
        store, job_id, "ja", 2, lambda language: FakeOCR(),
        lambda blocks, image, language: (blocks, 0), lambda *args, **kwargs: None,
    )
    assert observed == [2, 2, 1]
    assert len(result["pages"]) == 5
    assert all(page["blocks"] for page in result["pages"])
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_bounded_ocr.py -q
```

Expected: import failure for `job_pipeline`.

- [ ] **Step 3: Implement batch configuration and chunked OCR**

Create `job_pipeline.py` with:

```python
import os
import cv2


def configured_ocr_batch_size(environ=os.environ):
    try:
        value = int(environ.get("OCR_BATCH_SIZE", "2"))
    except (TypeError, ValueError):
        return 2
    return min(value, 8) if value > 0 else 2


def iter_chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def ocr_job(store, job_id, source_lang, batch_size, ocr_factory, filter_blocks, emit_progress):
    manifest = store.load(job_id)
    engine = ocr_factory(source_lang)
    total = len(manifest["pages"])
    for chunk in iter_chunks(manifest["pages"], batch_size):
        decoded = []
        try:
            for page in chunk:
                image = cv2.imread(str(store.source_path(job_id, page["index"])), cv2.IMREAD_COLOR)
                if image is None:
                    decoded.append(None)
                else:
                    decoded.append(image)
            valid = [(page, image) for page, image in zip(chunk, decoded) if image is not None]
            if not valid:
                continue
            arrays = [image for _, image in valid]
            results = engine.process_batch(arrays) if len(arrays) > 1 else [engine(arrays[0])]
            for (page, image), blocks in zip(valid, results):
                blocks, skipped = filter_blocks(blocks, image, source_lang)
                page["blocks"] = blocks
                page["skipped_artifacts"] = skipped
                completed = page["index"] + 1
                emit_progress("ocr", completed, total, "progress.ocr_image", name=page["name"])
            store.save(job_id, manifest)
        finally:
            decoded.clear()
    manifest["status"] = "ocr_complete"
    store.save(job_id, manifest)
    return manifest
```

Use the existing bilingual catalog key `progress.ocr_image` (`"OCR: {name}"` in both locales).

- [ ] **Step 4: Save and validate uploads one at a time**

Refactor `upload_file()` so it creates the job before decoding files. For each accepted `FileStorage`:

1. save the stream to `job_store.directory(job_id) / f"upload_{index:04d}.bin"`;
2. decode only that staging file with `cv2.imread`;
3. reject unreadable images without adding a page;
4. write `source_{index:04d}.jpg` at JPEG quality 92;
5. append `{index, name, width, height, blocks: []}` to the manifest and save it;
6. delete the staging file in `finally`, then release `encoded` and `image`.

Use this loop after the job manifest has been created:

```python
for index, file in enumerate(files):
    if not file or not file.filename:
        continue
    staging = job_store.directory(job_id) / f"upload_{index:04d}.bin"
    encoded = None
    image = None
    try:
        file.save(staging)
        encoded = np.fromfile(staging, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            continue
        source = job_store.source_path(job_id, index)
        if not cv2.imwrite(str(source), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            continue
        manifest["pages"].append({
            "index": index,
            "name": clean_image_name(file.filename),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "blocks": [],
        })
        job_store.save(job_id, manifest)
    finally:
        if staging.exists():
            staging.unlink()
        encoded = None
        image = None
```

Call `ocr_job(..., configured_ocr_batch_size(), lambda language: ChromeLensOCR(ocr_language=language), filter_ocr_blocks, emit_progress)`. Do not create `all_images`, `raw_images`, `batch_results`, or `all_ocr_results` lists containing arrays.

- [ ] **Step 5: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_bounded_ocr.py tests/test_job_storage.py -q
git add -- job_pipeline.py tests/test_bounded_ocr.py app.py localization.py
git commit -m "perf: bound upload and OCR image memory"
```

Expected: observed OCR chunk sizes are exactly `2, 2, 1`.

---

### Task 3: URL-backed correction pages and single-image region OCR

**Files:**
- Create: `tests/test_job_image_routes.py`
- Modify: `job_storage.py`
- Modify: `job_pipeline.py`
- Modify: `app.py:414-447,766-805,866-1000`
- Modify: `templates/correction.html`
- Modify: `static/js/correction.js:1-110`

**Interfaces:**
- Produces: `GET /jobs/<uuid:job_id>/images/source/<int:page_index>`.
- Produces: correction page items `{index, name, url, width, height, blocks}`.
- Produces: `prepare_correction_items(job_id: str, manifest: dict) -> list[dict]` that refines one page at a time.
- Changes: `/ocr-region` loads only the requested source file.
- Changes: `/continue-translate` writes corrected blocks to the manifest.

- [ ] **Step 1: Write failing route and HTML tests**

Create `tests/test_job_image_routes.py`:

```python
import cv2
import numpy as np


def make_job(app_module, tmp_path, monkeypatch):
    from job_storage import JobStore
    store = JobStore(tmp_path, ttl_seconds=60)
    monkeypatch.setattr(app_module, "job_store", store)
    job_id = store.create("en", {"source_lang": "ja", "target_lang": "vi", "selected_translator": "google", "selected_font": "animeace_", "style_code": "default", "custom_prompt": ""})
    cv2.imwrite(str(store.source_path(job_id, 0)), np.full((12, 16, 3), 255, np.uint8))
    manifest = store.load(job_id)
    manifest["pages"] = [{"index": 0, "name": "page", "width": 16, "height": 12, "blocks": [{"text": "x", "bbox": [1, 1, 5, 5]}]}]
    store.save(job_id, manifest)
    return store, job_id


def test_source_route_serves_only_manifest_page(tmp_path, monkeypatch):
    import app as app_module
    _, job_id = make_job(app_module, tmp_path, monkeypatch)
    client = app_module.app.test_client()
    assert client.get(f"/jobs/{job_id}/images/source/0").status_code == 200
    assert client.get(f"/jobs/{job_id}/images/source/1").status_code == 404
    assert client.get("/jobs/not-a-uuid/images/source/0").status_code == 404


def test_correction_html_uses_urls_not_base64(tmp_path, monkeypatch):
    import app as app_module
    _, job_id = make_job(app_module, tmp_path, monkeypatch)
    html = app_module.app.test_client().get(f"/correction/{job_id}").get_data(as_text=True)
    assert f"/jobs/{job_id}/images/source/0" in html
    assert "data:image" not in html
    assert '"url"' in html
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_job_image_routes.py -q
```

Expected: image route is missing and correction still builds base64 previews.

- [ ] **Step 3: Add the validated source/result file route**

Add one route with an allow-listed artifact kind:

```python
@app.get("/jobs/<uuid:job_id>/images/<kind>/<int:page_index>")
def job_image(job_id, kind, page_index):
    job_id = str(job_id)
    manifest = job_store.load(job_id)
    if manifest is None or kind not in {"source", "result"}:
        return "", 404
    if not any(page["index"] == page_index for page in manifest["pages"]):
        return "", 404
    path = job_store.source_path(job_id, page_index) if kind == "source" else job_store.result_path(job_id, page_index)
    if not path.is_file():
        return "", 404
    return send_file(path, mimetype="image/jpeg", conditional=True)
```

- [ ] **Step 4: Replace correction previews with metadata and URLs**

Replace `build_preview_images()` with a helper that refines bboxes while only one decoded page is live, then returns metadata and URLs:

```python
def build_correction_items(job_id, manifest):
    items = []
    source_lang = manifest["settings"].get("source_lang", "ja")
    changed = False
    for page in manifest["pages"]:
        if not page.get("correction_ready"):
            image = cv2.imread(str(job_store.source_path(job_id, page["index"])), cv2.IMREAD_COLOR)
            if image is not None:
                page["blocks"] = [{
                    **block,
                    "bbox": normalize_bbox_for_json(
                        refine_tall_narrow_ocr_bbox(
                            image, block.get("bbox"), source_lang=source_lang,
                            text=block.get("text", ""),
                        ),
                        image_shape=image.shape,
                        expand_ratio=0 if block.get("_bbox_expanded") else BBOX_EXPAND_RATIO,
                    ),
                    "_bbox_expanded": True,
                } for block in page.get("blocks", [])]
                page["correction_ready"] = True
                changed = True
            image = None
        items.append({
            "index": page["index"], "name": page["name"],
            "url": url_for("job_image", job_id=job_id, kind="source", page_index=page["index"]),
            "width": page["width"], "height": page["height"],
            "blocks": page.get("blocks", []),
        })
    if changed:
        job_store.save(job_id, manifest)
    return items
```

Change `correction.js` image loading from `'data:image/jpeg;base64,' + images[idx].data` to `images[idx].url`. The browser image cache remains page-owned; the server sends no encoded copy.

- [ ] **Step 5: Refactor correction and region routes to metadata-only jobs**

Read `job_id = request.form.get("job_id") or request.form.get("session_id", "")` during the compatibility window, then load the manifest with `job_store.load(job_id)`. In `/continue-translate`, normalize and sort posted blocks using `width`/`height` from each page, update `page["blocks"]`, and save the manifest; never rebuild `(name, image, blocks)` tuples. In `/ocr-region`, read only `job_store.source_path(job_id, image_idx)`, crop it, run OCR, and `del original_image, cropped` before returning. New templates and JavaScript post `job_id`; `session_id` is accepted only for old saved pages.

The correction update loop is:

```python
pages_by_index = {page["index"]: page for page in manifest["pages"]}
for image_data in modified_blocks:
    page = pages_by_index.get(int(image_data.get("image_idx", -1)))
    if page is None:
        continue
    blocks = []
    image_shape = (page["height"], page["width"], 3)
    for submitted in image_data.get("blocks", []):
        bbox = normalize_bbox_for_json(submitted.get("bbox"), image_shape=image_shape, expand_ratio=0)
        if bbox:
            blocks.append({
                "text": str(submitted.get("text", "")).strip(),
                "bbox": bbox,
                "_bbox_expanded": True,
            })
    page["blocks"] = sort_ocr_blocks_reading_order(blocks)
job_store.save(job_id, manifest)
return _do_full_pipeline(job_id, manifest.get("locale", g.ui_language))
```

The region image lifetime is explicit:

```python
original_image = cv2.imread(str(job_store.source_path(job_id, image_idx)), cv2.IMREAD_COLOR)
if original_image is None:
    return {"text": ""}, 404
cropped = original_image[cy1:cy2, cx1:cx2]
blocks = sort_ocr_blocks_reading_order(ChromeLensOCR(ocr_language=source_lang)(cropped))
text = " ".join(block.get("text", "").strip() for block in blocks if block.get("text", "").strip())
del cropped, original_image
return {"text": text}
```

- [ ] **Step 6: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_job_image_routes.py translator/test_translator.py -q
node --check static/js/correction.js
git add -- tests/test_job_image_routes.py job_storage.py app.py templates/correction.html static/js/correction.js
git commit -m "perf: serve correction images from job files"
```

Expected: correction HTML contains URLs and no `data:image` value.

---

### Task 4: Sequential render-to-disk and URL-only result pages

**Files:**
- Create: `tests/test_bounded_render.py`
- Modify: `job_pipeline.py`
- Modify: `app.py:448-632,807-959`
- Modify: `templates/translate.html`

**Interfaces:**
- Produces: `assign_text_indexes(manifest: dict) -> list[str]`.
- Produces: `render_job(store, job_id, translated_texts, font_path, source_lang, render_page, emit_progress) -> dict`.
- Produces: result items `{index, name, original_url, translated_url}`.
- Removes: `encode_image_jpeg`, `build_result_images`, `snapshot_original_images`, and array-returning `translate_and_render` behavior.

- [ ] **Step 1: Write failing sequential-lifetime and result HTML tests**

Create `tests/test_bounded_render.py`:

```python
import gc
import weakref

import cv2
import numpy as np

from job_pipeline import assign_text_indexes, render_job
from job_storage import JobStore


def test_render_releases_previous_page_before_opening_next(tmp_path, monkeypatch):
    store = JobStore(tmp_path, ttl_seconds=60)
    job_id = store.create("en", {})
    manifest = store.load(job_id)
    for index in range(3):
        cv2.imwrite(str(store.source_path(job_id, index)), np.full((8, 8, 3), 255, np.uint8))
        manifest["pages"].append({"index": index, "name": f"p{index}", "width": 8, "height": 8, "blocks": []})
    store.save(job_id, manifest)

    prior = []
    real_imread = cv2.imread
    def tracked_imread(path, flags):
        gc.collect()
        assert all(reference() is None for reference in prior)
        image = real_imread(path, flags)
        prior.append(weakref.ref(image))
        return image
    monkeypatch.setattr("job_pipeline.cv2.imread", tracked_imread)

    render_job(store, job_id, [], "fonts/animeace_i.ttf", "ja", lambda image, page, texts, font, lang: image, lambda *args, **kwargs: None)
    gc.collect()
    assert all(reference() is None for reference in prior)


def test_assign_text_indexes_uses_metadata_only():
    manifest = {"pages": [{"blocks": [{"text": "a"}, {"text": ""}]}, {"blocks": [{"text": "b"}]}]}
    assert assign_text_indexes(manifest) == ["a", "b"]
    assert manifest["pages"][0]["blocks"][0]["_text_idx"] == 0
    assert manifest["pages"][1]["blocks"][0]["_text_idx"] == 1
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_bounded_render.py -q
```

Expected: imports fail because the metadata render functions do not exist.

- [ ] **Step 3: Implement metadata indexing and sequential rendering**

Add to `job_pipeline.py`:

```python
def assign_text_indexes(manifest):
    texts = []
    for page in manifest["pages"]:
        for block in page.get("blocks", []):
            text = str(block.get("text", "")).strip()
            if text:
                block["_text_idx"] = len(texts)
                texts.append(text)
            else:
                block.pop("_text_idx", None)
    return texts


def render_job(store, job_id, translated_texts, font_path, source_lang, render_page, emit_progress):
    manifest = store.load(job_id)
    total = len(manifest["pages"])
    for position, page in enumerate(manifest["pages"], 1):
        image = None
        rendered = None
        try:
            image = cv2.imread(str(store.source_path(job_id, page["index"])), cv2.IMREAD_COLOR)
            if image is None:
                page["result_error"] = "unreadable_source"
                continue
            rendered = render_page(image, page, translated_texts, font_path, source_lang)
            if not cv2.imwrite(str(store.result_path(job_id, page["index"])), rendered, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                page["result_error"] = "write_failed"
                continue
            page.pop("result_error", None)
            page["has_result"] = True
            emit_progress("rendering", position, total, "progress.render_image", name=page["name"])
            store.save(job_id, manifest)
        finally:
            rendered = None
            image = None
    manifest["status"] = "complete"
    store.save(job_id, manifest)
    return manifest
```

- [ ] **Step 4: Move the existing one-page render logic behind `render_page`**

In `app.py`, keep the current block filtering, `erase_text_region`, and `render_all_blocks` logic in:

```python
def render_page(image, page, translated_texts, font_path, source_lang):
    render_blocks = []
    for block in page.get("blocks", []):
        text = str(block.get("text", "")).strip()
        bbox = block.get("bbox")
        if not text or not bbox or len(bbox) < 4:
            continue
        text_index = block.get("_text_idx", -1)
        translated = translated_texts[text_index] if 0 <= text_index < len(translated_texts) else text
        if not str(translated or "").strip():
            continue
        if should_skip_ocr_artifact(text, bbox, image_shape=image.shape, source_lang=source_lang):
            continue
        image, text_color, appearance = erase_text_region(image, bbox, source_lang=source_lang)
        appearance["should_skip"] = False
        render_blocks.append({
            "text": translated, "bbox": bbox,
            "text_color": text_color, "appearance": appearance,
        })
    return render_all_blocks(image, render_blocks, font_path) if render_blocks else image
```

Do not append the returned array to a list.

- [ ] **Step 5: Refactor `_do_full_pipeline` around manifest metadata**

Change its leading arguments to `job_id, prompt_locale`. Load the manifest, call `assign_text_indexes()`, translate the text list once, save warning/status metadata, then call `render_job(...)`. Build result items only from pages with `has_result`:

```python
items = [{
    "index": page["index"],
    "name": page["name"],
    "original_url": url_for("job_image", job_id=job_id, kind="source", page_index=page["index"]),
    "translated_url": url_for("job_image", job_id=job_id, kind="result", page_index=page["index"]),
} for page in manifest["pages"] if page.get("has_result")]
```

Delete `encode_image_jpeg`, `build_result_images`, `snapshot_original_images`, and any creation of `processed_results` or `original_images_by_name`.

- [ ] **Step 6: Convert the result template to URLs**

Use:

```html
<img class="gallery-image" src="{{ img.translated_url }}"
     data-translated="{{ img.translated_url }}"
     data-original="{{ img.original_url }}" alt="{{ img.name }}">
<a class="download-btn" href="{{ img.translated_url }}" download="{{ img.name }}_translated.jpg">{{ t('common.download') }}</a>
```

Remove base64 decoding, `data-image`, and JSZip. The compare tabs switch between the two URLs.

- [ ] **Step 7: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_bounded_render.py tests/test_job_image_routes.py translator/test_translator.py -q
git add -- tests/test_bounded_render.py job_pipeline.py app.py templates/translate.html
git commit -m "perf: render and serve one job image at a time"
```

Expected: the previous page array is collectible before the next file is decoded.

---

### Task 5: Disk-backed ZIP downloads and expired-job behavior

**Files:**
- Create: `tests/test_job_downloads.py`
- Modify: `job_storage.py`
- Modify: `app.py:57-75,1002-1027`
- Modify: `templates/translate.html`

**Interfaces:**
- Produces: `build_result_zip(store, job_id) -> pathlib.Path`.
- Changes: `POST /download-zip` consumes only `job_id`.
- Produces: localized expired-job redirect/error behavior.

- [ ] **Step 1: Write failing disk-ZIP tests**

Create `tests/test_job_downloads.py`:

```python
import zipfile

import cv2
import numpy as np

from job_pipeline import build_result_zip
from job_storage import JobStore


def test_zip_is_written_inside_job_directory(tmp_path):
    store = JobStore(tmp_path, ttl_seconds=60)
    job_id = store.create("en", {})
    manifest = store.load(job_id)
    for index, name in enumerate(("page", "page")):
        cv2.imwrite(str(store.result_path(job_id, index)), np.full((4, 4, 3), 255, np.uint8))
        manifest["pages"].append({"index": index, "name": name, "has_result": True, "blocks": []})
    store.save(job_id, manifest)

    archive = build_result_zip(store, job_id)

    assert archive == store.zip_path(job_id)
    with zipfile.ZipFile(archive) as opened:
        assert opened.namelist() == ["page_translated.jpg", "page_2_translated.jpg"]
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_job_downloads.py -q
```

Expected: `build_result_zip` is missing.

- [ ] **Step 3: Build ZIP archives directly on disk**

Add to `job_pipeline.py`:

```python
import zipfile


def build_result_zip(store, job_id):
    manifest = store.load(job_id)
    if manifest is None:
        raise FileNotFoundError(job_id)
    archive = store.zip_path(job_id)
    used = {}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for page in manifest["pages"]:
            if not page.get("has_result"):
                continue
            source = store.result_path(job_id, page["index"])
            if not source.is_file():
                continue
            base = f"{page['name']}_translated"
            used[base] = used.get(base, 0) + 1
            suffix = "" if used[base] == 1 else f"_{used[base]}"
            output.write(source, arcname=f"{base}{suffix}.jpg")
    return archive
```

- [ ] **Step 4: Replace client-payload ZIP handling**

The result form posts only:

```html
<input type="hidden" name="job_id" value="{{ job_id }}">
```

The route validates/loads the job and never accepts image bytes:

```python
@app.post("/download-zip")
def download_zip():
    job_id = request.form.get("job_id", "")
    try:
        manifest = job_store.load(job_id)
    except ValueError:
        manifest = None
    if manifest is None:
        return render_template("index.html", error=translate(g.ui_language, "validation.expired_job")), 404
    archive = build_result_zip(job_store, job_id)
    return send_file(
        archive, mimetype="application/zip", as_attachment=True,
        download_name="manga_translated.zip", conditional=True,
    )
```

Remove imports `io` and `base64` when no longer used. Missing or expired jobs render the locale-aware recovery message rather than attempting to decode client data.

- [ ] **Step 5: Verify cleanup and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_job_downloads.py tests/test_job_storage.py tests/test_job_image_routes.py -q
git add -- tests/test_job_downloads.py job_pipeline.py job_storage.py app.py templates/translate.html
git commit -m "perf: create result archives on disk"
```

Expected: ZIP contains unique names and the request body carries no image bytes.

---

### Task 6: Structural RAM assertions and full regression verification

**Files:**
- Create: `tests/test_memory_contract.py`
- Create: `tests/test_pipeline_integration.py`
- Modify only if a failure proves necessary: files changed in Tasks 1-5

**Interfaces:**
- Verifies: bounded OCR, one-image rendering, URL-only templates, no array cache, disk ZIP, TTL cleanup.

- [ ] **Step 1: Add source-level contract guards for removed memory patterns**

Create `tests/test_memory_contract.py`:

```python
from pathlib import Path


def test_app_has_no_cross_request_image_or_base64_cache():
    source = Path("app.py").read_text(encoding="utf-8")
    forbidden = (
        "ocr_sessions = {}", "snapshot_original_images", "build_result_images",
        "build_preview_images", "base64.b64encode", "base64.b64decode", "io.BytesIO()",
    )
    assert not any(pattern in source for pattern in forbidden)


def test_correction_and_result_templates_never_embed_image_data():
    for path in ("templates/correction.html", "templates/translate.html"):
        source = Path(path).read_text(encoding="utf-8")
        assert "data:image" not in source
        assert "base64" not in source.lower()


def test_default_limits_are_conservative(monkeypatch):
    from job_pipeline import configured_ocr_batch_size
    monkeypatch.delenv("OCR_BATCH_SIZE", raising=False)
    assert configured_ocr_batch_size() == 2
```

- [ ] **Step 2: Run the complete automated suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
node --check static/js/i18n.js
node --check static/js/app.js
node --check static/js/correction.js
git diff --check
```

Expected: all tests pass, JavaScript checks are silent, and no whitespace errors are reported.

- [ ] **Step 3: Add and run a five-image pipeline integration test**

Create `tests/test_pipeline_integration.py`:

```python
import zipfile

import cv2
import numpy as np

from job_pipeline import assign_text_indexes, build_result_zip, ocr_job, render_job
from job_storage import JobStore


def test_five_image_job_stays_chunked_and_writes_every_artifact(tmp_path):
    store = JobStore(tmp_path, ttl_seconds=60)
    job_id = store.create("en", {"source_lang": "ja", "target_lang": "vi"})
    manifest = store.load(job_id)
    for index in range(5):
        cv2.imwrite(str(store.source_path(job_id, index)), np.full((300, 200, 3), 255, np.uint8))
        manifest["pages"].append({"index": index, "name": f"page_{index + 1}", "width": 200, "height": 300, "blocks": []})
    store.save(job_id, manifest)

    batches = []
    class FakeOCR:
        def process_batch(self, images):
            batches.append(len(images))
            return [[{"text": "hello", "bbox": [10, 10, 80, 40]}] for _ in images]
        def __call__(self, image):
            batches.append(1)
            return [{"text": "hello", "bbox": [10, 10, 80, 40]}]

    manifest = ocr_job(store, job_id, "ja", 2, lambda language: FakeOCR(), lambda blocks, image, language: (blocks, 0), lambda *args, **kwargs: None)
    texts = assign_text_indexes(manifest)
    store.save(job_id, manifest)
    render_job(store, job_id, ["xin chào"] * len(texts), "fonts/animeace_i.ttf", "ja", lambda image, page, translations, font, language: image, lambda *args, **kwargs: None)
    archive = build_result_zip(store, job_id)

    assert batches == [2, 2, 1]
    assert all(store.source_path(job_id, index).is_file() for index in range(5))
    assert all(store.result_path(job_id, index).is_file() for index in range(5))
    with zipfile.ZipFile(archive) as opened:
        assert len(opened.namelist()) == 5
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline_integration.py -q
```

Expected: one passing integration test, recorded OCR batches `2, 2, 1`, and five ZIP entries without network access.

- [ ] **Step 4: Observe process RAM on a representative real job**

Start with the isolated environment:

```powershell
$env:OCR_BATCH_SIZE = "2"
.\run_app.ps1
```

Submit at least five representative manga pages. In Task Manager or Resource Monitor, observe that OCR peak is bounded by two decoded source images plus provider overhead and rendering does not increase progressively page by page. Refresh result/correction pages and confirm the Flask process does not rebuild all images in RAM.

- [ ] **Step 5: Verify both result modes and cleanup**

In both `vi` and `en`, verify direct translation and manual correction, original/translated compare, individual download, ZIP download, revisit correction, and a deliberately expired job. Confirm expired paths return 404 or the localized recovery path and never expose another job.

- [ ] **Step 6: Run the frontend detector and final status check**

```powershell
node C:\Users\dun\.agents\skills\impeccable\scripts\detect.mjs --json templates/correction.html templates/translate.html static/js/correction.js static/css/correction.css static/css/style.css
git status --short
```

Expected: no unresolved high-confidence issue in changed targets; status contains only intended implementation changes plus untouched pre-existing deletions.

- [ ] **Step 7: Commit verification fixes if present**

```powershell
git add -- job_storage.py job_pipeline.py app.py templates/correction.html templates/translate.html static/js/correction.js static/css/correction.css static/css/style.css tests
git commit -m "test: enforce bounded image memory contracts"
```

Skip this commit if Steps 2-6 required no file changes.
