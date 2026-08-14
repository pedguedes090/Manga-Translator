# Internationalization and Memory-Bounded Pipeline Design

**Date:** 2026-08-14

**Status:** Approved design; implementation pending

**Scope:** Flask UI, OCR/correction/result workflow, Gemini and local-LLM prompts

## 1. Goals

The application must serve Vietnamese and international users without coupling the interface language to the requested translation language. It must also process multi-image jobs with bounded memory instead of retaining every decoded image and every encoded preview in RAM.

The implementation will:

- provide a visible `VI | EN` language switch on upload, OCR correction, and result pages;
- use Vietnamese prompts when the UI is Vietnamese and English prompts when the UI is English;
- keep the target translation language as a separate user choice;
- localize UI labels, validation, warnings, progress, and recoverable errors;
- replace localized display strings used as form values with stable semantic codes;
- store job images and generated results on disk and keep only bounded working sets in memory;
- preserve the existing OCR correction and translation behavior unless this document explicitly changes it.

## 2. Non-goals

- Adding a third interface language.
- Translating user-entered custom instructions. Custom instructions are passed through verbatim.
- Changing Google Translate behavior; it does not use an application-authored LLM prompt.
- Introducing Redis, Celery, a database, or Flask-Babel solely for this two-language scope.
- Redesigning the application's visual identity.

## 3. Locale Selection and Persistence

The supported UI locale codes are `vi` and `en`.

Locale resolution follows this precedence:

1. an explicit, validated language selection;
2. the `ui_language` cookie;
3. the browser `Accept-Language` header (`vi` selects Vietnamese; every other value selects English);
4. English as the final fallback.

The language switch submits to a small server endpoint that validates the locale, writes a long-lived `SameSite=Lax` cookie, and redirects only to a validated local path. Templates are rendered in the selected locale on the same response cycle. The cookie is authoritative so server-rendered validation and error pages cannot disagree with client-side state.

For manual OCR sessions and result jobs, the resolved locale is saved in job metadata and carried through correction and translation requests. A deliberate switch by the user updates both the cookie and the locale used by subsequent requests.

Every page sets `<html lang>` to the active locale and exposes the same accessible language switch. The current choice is visually and programmatically indicated.

## 4. Translation Catalog and UI Contract

A small application-owned localization module contains:

- supported locale definitions;
- locale normalization and request resolution;
- keyed Vietnamese and English catalogs;
- a `t(key, **params)` helper with safe English fallback;
- the subset of catalog entries needed by JavaScript, serialized with Jinja's JSON encoder.

Templates use translation keys rather than literal user-facing strings. This includes page titles, headings, labels, buttons, help text, empty states, confirmations, validation messages, warnings, progress labels, and result/download controls.

JavaScript receives a namespaced catalog object and translates dynamic messages through a small lookup/interpolation helper. Missing keys fall back to English and remain visible during development rather than crashing the workflow.

Server progress events emit semantic data:

```json
{
  "message_key": "progress.ocr_processing",
  "message_params": {"current": 2, "total": 8},
  "percent": 25
}
```

The browser localizes this event. The server must not broadcast an already localized sentence because concurrent users may use different locales.

## 5. Stable Form Values

All selectable values use stable codes independent of their visible labels. Examples:

- source language: `auto`, `vi`, `en`, `ja`, ...;
- target language: `vi`, `en`, `ja`, ...;
- translation engine: `google`, `gemini`, `local_llm`;
- style: `natural`, `formal`, `casual`, `concise`, `custom`.

Visible labels come from the active catalog. The browser stores stable codes in local storage, never the rendered text. The backend maps codes to provider-specific language names and localized prompt fragments.

During the migration, the backend may accept known legacy Vietnamese display values so existing saved browser preferences do not break. Responses and newly persisted selections always use stable codes.

## 6. Prompt Language Architecture

Gemini and local-LLM prompt construction moves behind a shared prompt-building contract. Each request supplies:

- `prompt_locale`: `vi` or `en`, derived only from the active UI locale;
- source language code;
- target language code;
- style code;
- optional custom instruction;
- text or grouped text payload.

The invariant is:

> Prompt language follows the UI locale; requested output language follows the target-language selection.

Examples:

- English UI + Japanese target: an English instruction asking the model to translate into Japanese.
- Vietnamese UI + English target: a Vietnamese instruction asking the model to translate into English.

The shared builder supplies parallel Vietnamese and English templates for single, batch, and grouped translations. Style guidance is localized in the prompt locale. Custom instructions are inserted verbatim and clearly delimited; they are not machine-translated.

Provider adapters remain responsible for provider calls and response parsing, but they do not embed mixed Vietnamese/English prompt literals. Google Translate ignores `prompt_locale`.

## 7. File-backed Job Lifecycle

Each upload creates a UUID job directory under the existing temporary-session root. The directory contains:

- normalized source image files;
- OCR/correction metadata as JSON;
- translated/rendered image files when produced;
- a small job manifest containing locale, selections, status, and timestamps.

In-memory job state contains paths and lightweight metadata only. It must not cache full OpenCV arrays, PIL images, base64 strings, or duplicate original-image snapshots across requests.

Validated UUID-based routes serve source and result images with `send_file`. Route handlers resolve paths from the manifest, reject unknown image identifiers and path traversal, and never accept arbitrary filesystem paths from the client.

Correction and result templates receive image URLs plus block metadata. They do not receive image data URLs. Individual downloads use the validated file route. Multi-image downloads are built into a temporary ZIP file on disk and served without constructing the complete archive in `BytesIO`.

Job directories use the existing TTL cleanup policy. Cleanup ignores active/recent jobs, removes expired directories, and treats missing files as a recoverable expired-session error in the active locale.

## 8. Bounded-memory Processing

### Upload

Uploaded streams are validated and saved directly to the job directory. The application does not decode all uploads into a single `all_images` list. Image dimensions and format are checked one file at a time, and decoded objects are released after validation.

### OCR

OCR reads source files in bounded chunks controlled by `OCR_BATCH_SIZE`, defaulting to `2`. At most one chunk of decoded images and corresponding PIL conversions may be live in the application pipeline. Results are appended to job metadata, then the chunk's arrays/PIL objects are dereferenced before the next chunk.

Provider-side request concurrency may not exceed the current chunk size. A failed item records its error without retaining other decoded images.

### Correction

The correction page loads images by URL. Applying corrections updates JSON metadata only. Navigating back to correction uses the immutable source files; no full-image `.copy()` snapshot is required.

### Rendering and result delivery

Rendering is sequential by default:

1. decode one source image;
2. render translated blocks;
3. write the result JPEG to the job directory;
4. release the source and result arrays;
5. continue to the next image.

The result page contains URLs, not base64 duplicates in `src`, `data-original`, or `data-translated`. Browser memory therefore grows according to browser image loading behavior rather than duplicated HTML strings. The application server never builds a list containing every rendered array.

### Configuration

- `OCR_BATCH_SIZE=2` by default; valid positive integers only, with a conservative upper bound.
- Rendering concurrency defaults to `1` and is not increased automatically.
- Existing session TTL remains configurable and applies to all job artifacts.

## 9. Error Handling and Security

- Invalid locale values fall back safely and are never interpolated into paths.
- Redirect targets from the language switch must be local application paths.
- Job and image identifiers are allow-listed/UUID validated before path resolution.
- User-visible exceptions are mapped to translation keys; logs keep technical details and stable error identifiers.
- Prompt construction treats source text and custom instructions as data bounded by explicit delimiters.
- Partial jobs remain recoverable where possible and are eligible for TTL cleanup.

## 10. Compatibility and Migration

- Existing routes remain usable; new locale/job fields have defaults.
- Existing session metadata without a locale resolves through the request locale.
- Known legacy select labels are normalized at the request boundary.
- Provider interfaces receive `prompt_locale` through optional/defaulted parameters during migration so tests and integrations can be updated incrementally.
- No user-owned API keys or unrelated project configuration are changed.

## 11. Verification Strategy

### Automated tests

- Locale precedence: explicit selection, cookie, `Accept-Language`, fallback.
- Both catalogs contain every required key and interpolation parameter.
- Stable select codes round-trip independently of visible labels.
- Prompt snapshots for Gemini and local LLM cover single, batch, and grouped flows in both locales.
- Prompt tests prove UI locale and target language are independent.
- Custom instructions remain byte-for-byte unchanged inside both prompt variants.
- Progress/error payloads use keys and parameters instead of localized server strings.
- Job routes reject invalid UUIDs, unknown files, and traversal attempts.
- TTL cleanup removes expired jobs but preserves active/recent jobs.
- OCR instrumentation proves the maximum simultaneously decoded source count is no greater than `OCR_BATCH_SIZE`.
- Render instrumentation proves no more than one source/result image pair is live in the default flow.
- Template tests prove image bytes/base64 are absent from correction and result HTML.

### Manual verification

- Exercise upload, correction, result, single download, and ZIP download in both locales.
- Verify first-visit browser detection and persistence after refresh/navigation.
- Test English UI with a non-English target and Vietnamese UI with English target.
- Inspect desktop and narrow/mobile layouts, including longer English labels and error messages.
- Run the frontend quality detector against every changed HTML/CSS/JS target.
- Compare a representative multi-image run before and after the change; confirm decoded-image lifetime is bounded structurally and observe lower process peak RAM.

## 12. Acceptance Criteria

The work is accepted when:

1. all three user-facing screens can switch completely between Vietnamese and English, and the selection persists;
2. every Gemini/local-LLM translation path uses prompts matching the UI locale while honoring the independently selected target language;
3. form submission and saved preferences use stable codes rather than localized labels;
4. no correction/result template embeds original or translated images as base64;
5. the server retains no list of all decoded uploads, no full copied original-image snapshot, and no list of all rendered image arrays;
6. OCR's decoded working set is bounded by `OCR_BATCH_SIZE` (default `2`) and rendering's is one image at a time;
7. expired disk artifacts are cleaned safely, and invalid job access is rejected;
8. automated tests and the existing relevant test suite pass;
9. manual VI/EN and responsive UI checks pass without clipped or untranslated critical controls.
