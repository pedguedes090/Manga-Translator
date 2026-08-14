# Reading Order and Isolated Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize OCR text into top-to-bottom rows with left-to-right order, remove obsolete test/release artifacts, and verify installation in an isolated `.venv`.

**Architecture:** A pure `sort_ocr_blocks_reading_order()` helper in `add_text.py` will own spatial ordering. OCR merge output, manual correction input, and region OCR will all pass through this helper before text indexes or joined text are produced. Existing PowerShell environment scripts remain the single setup/run path; dependency versions change only if a fresh `.venv` proves a real conflict.

**Tech Stack:** Python 3.11, pytest, Flask test client, PowerShell virtual environments, pip.

## Global Constraints

- Reading direction is fixed for every language: rows top to bottom; blocks within a row left to right.
- Preserve bbox coordinates and do not mutate input block dictionaries.
- Keep invalid or missing bbox blocks stable at the end of the helper result.
- Do not change plaintext Gemini session storage or network-facing security configuration.
- Do not install or modify machine-wide Python packages.
- Preserve unrelated existing working-tree deletions.

---

### Task 1: Pure row-aware reading order

**Files:**
- Modify: `translator/test_translator.py:11-20,528-560`
- Modify: `add_text.py:68-178`

**Interfaces:**
- Consumes: OCR block dictionaries with optional `bbox=[x1,y1,x2,y2]`.
- Produces: `sort_ocr_blocks_reading_order(blocks: list[dict]) -> list[dict]`.
- Produces: `merge_nearby_ocr_blocks()` output and merged text parts normalized through the shared helper.

- [ ] **Step 1: Import the wished-for helper and write failing unit tests**

Add `sort_ocr_blocks_reading_order` to the import list in `translator/test_translator.py` and add:

```python
def test_reading_order_groups_y_jitter_into_left_to_right_rows_without_mutation():
    blocks = [
        {"text": "right", "bbox": [120, 10, 170, 50]},
        {"text": "next-row", "bbox": [15, 90, 70, 130]},
        {"text": "left", "bbox": [10, 20, 60, 60]},
    ]
    original_bboxes = [list(block["bbox"]) for block in blocks]

    ordered = sort_ocr_blocks_reading_order(blocks)

    assert [block["text"] for block in ordered] == ["left", "right", "next-row"]
    assert blocks[0]["text"] == "right"
    assert [block["bbox"] for block in blocks] == original_bboxes


def test_reading_order_keeps_invalid_bboxes_stable_at_end():
    blocks = [
        {"text": "missing"},
        {"text": "bottom", "bbox": [0, 80, 40, 120]},
        {"text": "invalid", "bbox": [10, 10, 10, 20]},
        {"text": "top", "bbox": [0, 5, 40, 45]},
    ]

    ordered = sort_ocr_blocks_reading_order(blocks)

    assert [block["text"] for block in ordered] == ["top", "bottom", "missing", "invalid"]


def test_merge_same_line_fragments_uses_left_to_right_text_order_with_y_jitter():
    blocks = [
        {"text": "RIGHT", "bbox": [65, 10, 100, 45]},
        {"text": "LEFT", "bbox": [20, 16, 58, 51]},
    ]

    merged = merge_nearby_ocr_blocks(blocks)

    assert len(merged) == 1
    assert merged[0]["text"] == "LEFT\nRIGHT"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest translator/test_translator.py -q -k "reading_order or merge_same_line_fragments"
```

Expected: collection/import failure because `sort_ocr_blocks_reading_order` does not exist yet.

- [ ] **Step 3: Implement the minimal pure sorter**

In `add_text.py`, immediately after `_range_overlap_ratio`, implement:

```python
def sort_ocr_blocks_reading_order(blocks):
    positioned = []
    unpositioned = []
    for index, block in enumerate(blocks or []):
        bbox = _coerce_bbox_for_merge(
            block.get("bbox") if isinstance(block, dict) else None
        )
        if bbox is None:
            unpositioned.append((index, block))
            continue
        positioned.append({
            "index": index,
            "block": block,
            "bbox": bbox,
            "center_y": (bbox[1] + bbox[3]) / 2.0,
            "height": bbox[3] - bbox[1],
        })

    positioned.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["index"]))
    rows = []
    for item in positioned:
        best_row = None
        best_score = None
        for row in rows:
            row_half_height = row["average_height"] / 2.0
            row_y1 = row["center_y"] - row_half_height
            row_y2 = row["center_y"] + row_half_height
            overlap = _range_overlap_ratio(item["bbox"][1], item["bbox"][3], row_y1, row_y2)
            center_delta = abs(item["center_y"] - row["center_y"])
            tolerance = max(8.0, min(item["height"], row["average_height"]) * 0.50)
            if overlap < 0.30 and center_delta > tolerance:
                continue
            score = (overlap, -center_delta)
            if best_score is None or score > best_score:
                best_row = row
                best_score = score

        if best_row is None:
            rows.append({
                "items": [item],
                "top": item["bbox"][1],
                "center_y": item["center_y"],
                "average_height": float(item["height"]),
            })
            continue

        best_row["items"].append(item)
        count = len(best_row["items"])
        best_row["top"] = min(best_row["top"], item["bbox"][1])
        best_row["center_y"] = sum(entry["center_y"] for entry in best_row["items"]) / count
        best_row["average_height"] = sum(entry["height"] for entry in best_row["items"]) / count

    rows.sort(key=lambda row: (row["top"], row["center_y"]))
    ordered = []
    for row in rows:
        row["items"].sort(
            key=lambda item: (item["bbox"][0], item["bbox"][1], item["index"])
        )
        ordered.extend(item["block"] for item in row["items"])
    ordered.extend(block for _, block in unpositioned)
    return ordered
```

This returns references from the input list but never changes the list, dictionaries, or bbox values.

- [ ] **Step 4: Normalize merged text parts and merged block output**

In `merge_nearby_ocr_blocks`, replace the direct `(y, x)` sort at line 165 with a lookup from helper-ordered block identities:

```python
        ordered_blocks = sort_ocr_blocks_reading_order(
            [item["block"] for item in group["items"]]
        )
        items_by_id = {id(item["block"]): item for item in group["items"]}
        items = [items_by_id[id(block)] for block in ordered_blocks]
```

Return normalized merged regions:

```python
    return sort_ocr_blocks_reading_order(merged)
```

- [ ] **Step 5: Run focused and maintained tests**

Run:

```powershell
python -m pytest translator/test_translator.py -q -k "reading_order or merge_nearby or merge_same_line_fragments"
python -m pytest translator/test_translator.py -q
```

Expected: new reading-order tests pass; the pre-existing `risky_background` assertion remains the only failure in the second command.

- [ ] **Step 6: Commit Task 1**

```powershell
git add -- add_text.py translator/test_translator.py
git commit -m "fix: normalize OCR block reading order"
```

---

### Task 2: Integrate ordering into manual correction and region OCR

**Files:**
- Modify: `translator/test_translator.py:355-395`
- Modify: `app.py:28-36,911-930,992-996`

**Interfaces:**
- Consumes: `sort_ocr_blocks_reading_order()` from Task 1.
- Produces: normalized `_text_idx` values in `/continue-translate`.
- Produces: normalized joined text from `/ocr-region`.

- [ ] **Step 1: Expand the manual-correction route test and write a region OCR test**

Change the posted manual blocks in `test_continue_translate_keeps_correction_bboxes_exact` to deliberately unordered blocks:

```python
"blocks": [
    {"text": "right", "bbox": [35, 5, 55, 25]},
    {"text": "bottom", "bbox": [5, 40, 25, 60]},
    {"text": "left", "bbox": [5, 10, 25, 30]},
]
```

Then assert:

```python
captured_blocks = captured["all_ocr_results"][0][2]
assert [block["text"] for block in captured_blocks] == ["left", "right", "bottom"]
assert [block["_text_idx"] for block in captured_blocks] == [0, 1, 2]
assert captured_blocks[0]["bbox"] == [5, 10, 25, 30]
```

Add:

```python
def test_ocr_region_joins_blocks_in_reading_order(monkeypatch):
    import app as app_module

    session_id = str(uuid.uuid4())
    app_module.ocr_sessions.clear()
    app_module.ocr_sessions[session_id] = {
        "all_ocr_results": [("page", np.full((80, 80, 3), 255, dtype=np.uint8), [])],
        "source_lang": "en",
    }

    class FakeOCR:
        def __init__(self, ocr_language):
            assert ocr_language == "en"

        def __call__(self, image):
            return [
                {"text": "right", "bbox": [30, 2, 50, 22]},
                {"text": "bottom", "bbox": [2, 35, 25, 55]},
                {"text": "left", "bbox": [2, 6, 22, 26]},
            ]

    monkeypatch.setattr(app_module, "ChromeLensOCR", FakeOCR)
    response = app_module.app.test_client().post(
        "/ocr-region",
        data={
            "session_id": session_id,
            "image_idx": "0",
            "x1": "0",
            "y1": "0",
            "x2": "70",
            "y2": "70",
        },
    )

    assert response.status_code == 200
    assert response.get_json()["text"] == "left right bottom"
```

- [ ] **Step 2: Run both route tests and verify RED**

Run:

```powershell
python -m pytest translator/test_translator.py -q -k "continue_translate_keeps_correction_bboxes_exact or ocr_region_joins_blocks_in_reading_order"
```

Expected: failures show the received order is `right, bottom, left`.

- [ ] **Step 3: Import and apply the helper in `app.py`**

Add `sort_ocr_blocks_reading_order` to the `from add_text import (...)` block.

In `/continue-translate`, first normalize all submitted blocks without assigning indexes. Then sort and assign indexes:

```python
        blocks = []
        for b in img_data["blocks"]:
            text = b.get("text", "").strip()
            bbox = normalize_bbox_for_json(
                b.get("bbox"), image_shape=original_image.shape, expand_ratio=0
            )
            if bbox and len(bbox) == 4:
                blocks.append({"text": text, "bbox": bbox, "_bbox_expanded": True})

        blocks = sort_ocr_blocks_reading_order(blocks)
        for block in blocks:
            if block["text"]:
                block["_text_idx"] = text_index
                text_index += 1
                all_texts.append(block["text"])
```

In `/ocr-region`, normalize before joining:

```python
    blocks = sort_ocr_blocks_reading_order(ocr_engine(cropped))
```

- [ ] **Step 4: Verify route and maintained tests**

Run:

```powershell
python -m pytest translator/test_translator.py -q -k "continue_translate_keeps_correction_bboxes_exact or ocr_region_joins_blocks_in_reading_order"
python -m pytest translator/test_translator.py -q
```

Expected: route tests pass; only the known `risky_background` assertion remains.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- app.py translator/test_translator.py
git commit -m "fix: preserve reading order after OCR correction"
```

---

### Task 3: Remove obsolete checks and align the active safety assertion

**Files:**
- Delete: `test_translator_batch.py`
- Modify: `translator/test_translator.py:746-758`
- Delete: `.github/workflows/release.yml` (already deleted in the approved working tree)

**Interfaces:**
- Consumes: maintained current translator behavior.
- Produces: a pytest suite without global `sys.modules` contamination or removed NLLB expectations.

- [ ] **Step 1: Confirm the isolated maintained failure**

Run:

```powershell
python -m pytest translator/test_translator.py::test_assess_erasability_skips_text_on_complex_artwork -q
```

Expected: FAIL because the safe rejection reason is `risky_background`, while the test only accepts `complex_artwork` and `excessive_mask`.

- [ ] **Step 2: Align the assertion with the safe behavior**

Change the accepted reasons to:

```python
assert result["reason"] in {"complex_artwork", "excessive_mask", "risky_background"}
```

Do not change `assess_erasability`; it already returns `safe=False` and a score below `0.55`.

- [ ] **Step 3: Delete stale files within approved scope**

Delete `test_translator_batch.py` with `apply_patch`. Keep `.github/workflows/release.yml` deleted. Do not stage or alter any other `.commandcode` or `.jules` deletion.

- [ ] **Step 4: Run the repository test suite with the current interpreter**

Run:

```powershell
python -m pytest -q
```

Expected: all collected maintained tests pass with no `google.protobuf` import contamination.

- [ ] **Step 5: Commit Task 3 using explicit paths**

```powershell
git add -- translator/test_translator.py test_translator_batch.py .github/workflows/release.yml
git commit -m "test: remove obsolete NLLB and release checks"
```

---

### Task 4: Build and verify the isolated project environment

**Files:**
- Verify: `setup_venv.ps1`
- Verify: `run_app.ps1`
- Modify only on demonstrated conflict: `requirements.txt`

**Interfaces:**
- Consumes: Python 3.11 installed through the Windows `py` launcher.
- Produces: ignored local `.venv` used for all project commands.

- [ ] **Step 1: Create `.venv` and install only inside it**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1 -PythonVersion 3.11
```

Expected: `.venv\Scripts\python.exe` exists and requirements install successfully. No global `pip install` command is used.

- [ ] **Step 2: Verify interpreter isolation and package consistency**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable); print(sys.prefix); print(sys.base_prefix)"
.\.venv\Scripts\python.exe -m pip check
```

Expected: executable and prefix point inside `F:\duancanhan\newbiew\.venv`, `base_prefix` points to Python 3.11, and `pip check` prints `No broken requirements found.`

- [ ] **Step 3: Resolve only a demonstrated in-venv conflict**

If and only if Step 1 or Step 2 fails, capture the exact resolver message, identify the two incompatible constraints with:

```powershell
.\.venv\Scripts\python.exe -m pip show protobuf chrome-lens-py google-genai
.\.venv\Scripts\python.exe -m pip index versions protobuf
```

If the resolver names a different package, run `pip show` and `pip index versions` with that exact reported package name. Then constrain the direct dependency in `requirements.txt` to the narrowest compatible range. Before recreating the environment, verify and remove only the project-local path:

```powershell
$ProjectVenv = (Resolve-Path -LiteralPath .\.venv).Path
if ($ProjectVenv -ne "F:\duancanhan\newbiew\.venv") {
    throw "Refusing to remove unexpected path: $ProjectVenv"
}
Remove-Item -LiteralPath $ProjectVenv -Recurse -Force
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1 -PythonVersion 3.11
```

Repeat Step 2. Do not alter the global environment.

- [ ] **Step 4: Run final verification in `.venv`**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -c "import app; client=app.app.test_client(); response=client.get('/'); assert response.status_code == 200; print('GET / OK')"
node --check static/js/app.js
node --check static/js/correction.js
git diff --check
git status --short
```

Expected: pytest passes, Flask smoke check prints `GET / OK`, JavaScript syntax checks are silent with exit code 0, and the final status contains only intended changes plus untouched pre-existing deletions.

- [ ] **Step 5: Commit a dependency constraint only if Step 3 changed it**

```powershell
git add -- requirements.txt
git commit -m "build: constrain isolated Python dependencies"
```

Skip this commit when the fresh `.venv` has no dependency conflict.
