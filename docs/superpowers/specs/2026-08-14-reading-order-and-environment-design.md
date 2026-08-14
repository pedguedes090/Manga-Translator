# Reading Order and Isolated Environment Design

## Scope

This change addresses the approved project cleanup:

- Make text block reading order deterministic: rows from top to bottom and blocks within each row from left to right.
- Apply the same ordering to automatic OCR, manually edited blocks, and region re-OCR results.
- Remove the obsolete root-level NLLB batch test.
- Keep the already removed GitHub release workflow removed.
- Use the project's `.venv` setup so dependency installation does not alter or depend on the machine-wide Python environment.

The existing plaintext Gemini session storage and network-facing security defaults are intentionally outside this change.

## Reading-order algorithm

Add one pure helper in `add_text.py` that accepts a list of OCR block dictionaries and returns a newly ordered list without mutating the input.

Blocks with valid bounding boxes are clustered into visual rows. Row membership uses vertical overlap together with a height-relative center tolerance, so small OCR `y` jitter does not put a right-hand block before a left-hand block. Rows are sorted by their top position. Blocks inside a row are sorted by their left position, with original input index as the stable tie-breaker.

Blocks without a valid bounding box remain in their original relative order and are placed after positioned blocks. The reading direction is fixed for all source languages as approved; no right-to-left or vertical-language mode is introduced.

## Integration points

The shared helper is applied at the boundaries where block order becomes user-visible:

1. The output of automatic OCR filtering and merging, before translation indexes are assigned.
2. The block list rebuilt from the manual-correction form, before translation indexes are assigned.
3. OCR blocks returned for a manually selected region, before their text is joined.

The merge implementation continues using its existing geometric rules. Only the final reading order is normalized, so bounding boxes, merge decisions, erasing, and rendering behavior stay unchanged.

## Test cleanup and coverage

Delete `test_translator_batch.py`. It targets the removed NLLB implementation and globally replaces modules in `sys.modules`, contaminating unrelated tests.

Keep the active suite in `translator/test_translator.py`. Add focused tests covering:

- Same-row blocks with different `y` values still sort left to right.
- Separate rows sort top to bottom.
- Invalid or missing bounding boxes remain stable at the end.
- Manual-correction submission assigns translation order from the normalized block order.

The existing complex-background test should accept `risky_background` as a valid conservative rejection reason because the tested safety result remains `safe=False`.

## Isolated dependency environment

Reuse `setup_venv.ps1` and `run_app.ps1`; they already create and exclusively run `.venv`. Create the local environment with Python 3.10 or 3.11, install `requirements.txt` inside it, then run `pip check` and the test suite using `.venv/Scripts/python.exe`.

Do not change global Python packages. Do not add speculative version pins. If dependency resolution or `pip check` fails inside the fresh environment, adjust only the conflicting entries in `requirements.txt`, recreate the environment, and repeat verification.

## Acceptance criteria

- Automatic, manual, and region OCR text follows top-to-bottom row order and left-to-right order within a row.
- Reordering does not mutate input block dictionaries or bbox values.
- The obsolete NLLB test file and release workflow are absent.
- A fresh `.venv` installs successfully and reports no broken requirements.
- The maintained test suite passes when invoked through `.venv`.
