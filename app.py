"""
Manga Translator - Flask Web Application
OCR full image with Chrome Lens (text blocks + bounding boxes),
then erase original text and render translated text in place.
"""
from flask import Flask, render_template, request, redirect, send_file
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
import io
import zipfile
import json
import warnings
import os
import re
import sys
import time
import uuid

warnings.filterwarnings("ignore", category=DeprecationWarning)

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from translator.translator import MangaTranslator
from add_text import (
    add_text_bbox,
    appearance_for_prepared,
    assess_erasability,
    erase_mask_region,
    erase_text_region,
    merge_nearby_ocr_blocks,
    refine_tall_narrow_ocr_bbox,
    render_all_blocks,
    resolve_font_path_for_style,
    should_skip_ocr_artifact,
    sort_ocr_blocks_reading_order,
)
from ocr.chrome_lens_ocr import ChromeLensOCR
from vision.app_adapter import build_optional_vision_adapter
from PIL import Image
import numpy as np
import base64
import cv2
import threading
import math


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "secret_key")
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024


def get_async_mode():
    return os.environ.get("SOCKETIO_ASYNC_MODE", "threading")


# ── Background session cleanup thread ──
def _start_cleanup_thread(interval_seconds=600):
    """Run cleanup_old_sessions periodically in a daemon thread."""
    import time as _time

    def _cleanup_loop():
        while True:
            _time.sleep(interval_seconds)
            try:
                cleanup_old_sessions()
            except Exception as e:
                print(f"Session cleanup error: {e}")

    t = threading.Thread(target=_cleanup_loop, daemon=True, name="session-cleanup")
    t.start()
    return t


socketio = SocketIO(app, cors_allowed_origins="*", async_mode=get_async_mode())
_start_cleanup_thread(interval_seconds=600)  # every 10 minutes

# In-memory session storage for manual correction
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_sessions")
os.makedirs(TEMP_DIR, exist_ok=True)
ocr_sessions = {}
MAX_MEMORY_SESSIONS = 20
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", 6 * 3600))
BBOX_EXPAND_RATIO = 0.03
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "avif"}
SUPPORTED_IMAGE_FORMATS_LABEL = "JPG, JPEG, PNG, WebP, BMP, TIFF hoặc AVIF"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def parse_gemini_api_keys(raw_value):
    """Parse newline/comma/semicolon separated Gemini keys, preserving order."""
    seen = set()
    keys = []
    for key in re.split(r"[\s,;]+", raw_value or ""):
        key = key.strip()
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def is_allowed_image_file(filename):
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_IMAGE_EXTENSIONS


def clean_image_name(filename):
    raw_stem = os.path.splitext(os.path.basename(filename or ""))[0]
    if not re.search(r"[A-Za-z0-9]", raw_stem):
        return "image"
    safe = secure_filename(filename or "")
    name = os.path.splitext(safe)[0].strip("._- ")
    return name or "image"


def _safe_session_path(session_id):
    """Validate session_id as UUID and return its directory path (prevents path traversal)."""
    if not session_id or not _UUID_RE.match(session_id.strip().lower()):
        return None
    sid = session_id.strip().lower()
    return os.path.join(TEMP_DIR, sid)


def _session_json_path(session_id):
    """Return path to the session JSON file."""
    base = _safe_session_path(session_id)
    if not base:
        return None
    return os.path.join(base, "session.json")


def _session_image_path(session_id, idx):
    """Return path to a session image JPEG file."""
    base = _safe_session_path(session_id)
    if not base:
        return None
    return os.path.join(base, f"page_{idx}.jpg")


def cleanup_old_sessions():
    """Remove expired session directories from disk and trim the in-memory cache."""
    now = time.time()
    try:
        for fname in os.listdir(TEMP_DIR):
            fpath = os.path.join(TEMP_DIR, fname)
            if not os.path.isdir(fpath):
                # Clean up legacy .pkl files
                if fname.endswith(".pkl"):
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass
                continue
            try:
                if now - os.path.getmtime(fpath) > SESSION_TTL_SECONDS:
                    import shutil
                    shutil.rmtree(fpath, ignore_errors=True)
                    ocr_sessions.pop(fname, None)
            except OSError:
                pass
    except OSError:
        pass
    # Bound in-memory cache size (drop oldest inserted first)
    while len(ocr_sessions) > MAX_MEMORY_SESSIONS:
        ocr_sessions.pop(next(iter(ocr_sessions)), None)


def _save_session(session_id, session_data):
    """Save session to disk as JSON + JPEG images (lightweight, no pickle)."""
    base = _safe_session_path(session_id)
    if not base:
        return
    os.makedirs(base, exist_ok=True)

    # Write images as JPEG files
    all_ocr_results = session_data.get('all_ocr_results', [])
    for i, (name, image, blocks) in enumerate(all_ocr_results):
        img_path = os.path.join(base, f"page_{i}.jpg")
        # Only write if not already cached (avoid redundant writes)
        if not os.path.exists(img_path):
            cv2.imwrite(img_path, image, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Write metadata as JSON (no numpy arrays, no pickle)
    json_data = {}
    for key, value in session_data.items():
        if key == 'all_ocr_results':
            # Store block metadata too. The correction flow needs flags such as
            # _bbox_expanded to avoid expanding already-normalized boxes again.
            json_data[key] = [
                {
                    'name': str(name),
                    'blocks': [
                        {
                            block_key: block_value
                            for block_key, block_value in dict(b).items()
                            if _is_json_serializable(block_value)
                        }
                        for b in blocks
                    ],
                }
                for name, _, blocks in all_ocr_results
            ]
        else:
            # Only store JSON-serializable values
            try:
                json.dumps(value)
                json_data[key] = value
            except (TypeError, ValueError):
                json_data[key] = str(value)

    json_path = os.path.join(base, "session.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # Cache in memory (keep the original data with images for active use)
    ocr_sessions[session_id.strip().lower()] = session_data


def load_session(session_id):
    """Load session data from memory cache, falling back to disk (JSON + JPEG)."""
    base = _safe_session_path(session_id)
    if not base:
        return None
    sid = session_id.strip().lower()
    if sid in ocr_sessions:
        return ocr_sessions[sid]
    if not os.path.isdir(base):
        return None

    json_path = os.path.join(base, "session.json")
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Failed to load session JSON {sid}: {e}")
        return None

    # Reconstruct all_ocr_results with images loaded from JPEG files
    all_ocr_results = []
    for i, img_meta in enumerate(json_data.get('all_ocr_results', [])):
        img_path = os.path.join(base, f"page_{i}.jpg")
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
            if image is None:
                continue
        else:
            continue
        blocks = img_meta.get('blocks', [])
        all_ocr_results.append((img_meta['name'], image, blocks))

    # Rebuild session data
    data = dict(json_data)
    data['all_ocr_results'] = all_ocr_results
    data.pop('all_texts', None)  # Will be rebuilt by caller

    ocr_sessions[sid] = data
    return data


def _is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def get_font_path(font_name: str) -> str:
    """Map a font name to its font file (base fonts: <name>i.ttf)."""
    return resolve_font_path_for_style(font_name) or f"fonts/{font_name}.ttf"


def _fonts_dir_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def list_available_fonts():
    """Scan fonts/ for .ttf files -> [{"name", "label"}] (spec section 4.4).

    animeace_i.ttf -> animeace_/Animeace, ariali.ttf -> arial/Arial,
    mangati.ttf -> mangat/Mangat, Yuki-*.ttf -> name without extension.
    The 3 base fonts come first, then Yuki fonts alphabetically.
    """
    entries = []
    fonts_dir = _fonts_dir_path()
    if os.path.isdir(fonts_dir):
        for filename in sorted(os.listdir(fonts_dir)):
            if not filename.lower().endswith(".ttf"):
                continue
            name = filename[:-4]
            if name == "animeace_i":
                entry = ("animeace_", "Animeace")
            elif name == "ariali":
                entry = ("arial", "Arial")
            elif name == "mangati":
                entry = ("mangat", "Mangat")
            else:
                entry = (name, name)
            entries.append(entry)
    base_names = {"animeace_", "arial", "mangat"}
    base = [e for e in entries if e[0] in base_names]
    others = [e for e in entries if e[0] not in base_names]
    others.sort(key=lambda e: e[0].lower())
    return [{"name": name, "label": label} for name, label in base + others]


def normalize_block_style(raw, default_font):
    """Validate/normalize a per-block style dict (V3 spec F5 / section 4.7).

    Returns a dict {font, font_size, text_color, bold, italic, align} with all
    keys, or None when raw is not a dict. Invalid values are replaced with
    defaults: unknown font -> default_font, font_size clamped 0-120, invalid
    hex color -> None (auto), align -> "center", bold/italic coerced to bool.
    """
    if not isinstance(raw, dict):
        return None
    font_names = {entry["name"] for entry in list_available_fonts()}
    font = raw.get("font")
    if not isinstance(font, str) or font not in font_names:
        font = default_font
    try:
        font_size = int(raw.get("font_size", 0))
    except (TypeError, ValueError):
        font_size = 0
    font_size = max(0, min(120, font_size))
    text_color = raw.get("text_color")
    if not (isinstance(text_color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", text_color)):
        text_color = None
    else:
        text_color = text_color.upper()
    bold = bool(raw.get("bold", False))
    italic = bool(raw.get("italic", False))
    align = raw.get("align", "center")
    if align not in ("left", "center", "right"):
        align = "center"
    return {
        "font": font,
        "font_size": font_size,
        "text_color": text_color,
        "bold": bold,
        "italic": italic,
        "align": align,
    }


def emit_progress(phase, current, total, message):
    try:
        socketio.emit('progress', {
            'phase': phase,
            'current': current,
            'total': total,
            'message': message,
            'percent': int((current / max(total, 1)) * 100)
        })
    except Exception:
        pass


def _short_log_text(text, max_len=36):
    cleaned = re.sub(r'\s+', ' ', str(text or '')).strip()
    return cleaned if len(cleaned) <= max_len else cleaned[:max_len - 3] + "..."


def normalize_bbox_for_json(bbox, image_shape=None, expand_ratio=0):
    if not bbox or len(bbox) < 4:
        return None

    coords = []
    for value in bbox[:4]:
        try:
            if isinstance(value, np.generic):
                value = value.item()
            coords.append(int(round(float(value))))
        except (TypeError, ValueError):
            return None

    x1, y1, x2, y2 = coords
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if expand_ratio and x2 > x1 and y2 > y1:
        pad_x = max(1, int(math.ceil((x2 - x1) * expand_ratio)))
        pad_y = max(1, int(math.ceil((y2 - y1) * expand_ratio)))
        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y

    if image_shape is not None:
        h, w = image_shape[:2]
        x1 = max(0, min(int(w), x1))
        y1 = max(0, min(int(h), y1))
        x2 = max(0, min(int(w), x2))
        y2 = max(0, min(int(h), y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def filter_ocr_blocks(blocks, image, source_lang):
    image_shape = image.shape
    candidate_blocks = []
    skipped = 0
    for raw_index, block in enumerate(blocks):
        raw_bbox = block.get("bbox")
        if should_skip_ocr_artifact(
            block.get("text", ""),
            raw_bbox,
            image_shape=image_shape,
            source_lang=source_lang,
        ):
            skipped += 1
            continue

        refined_bbox = refine_tall_narrow_ocr_bbox(
            image,
            raw_bbox,
            source_lang=source_lang,
            text=block.get("text", ""),
        )
        expanded_bbox = normalize_bbox_for_json(
            refined_bbox,
            image_shape=image_shape,
            expand_ratio=BBOX_EXPAND_RATIO,
        )
        if not expanded_bbox:
            skipped += 1
            continue

        block = dict(block)
        block["bbox"] = expanded_bbox
        block["_bbox_expanded"] = True
        block["_ocr_index"] = raw_index
        candidate_blocks.append(block)

    merged_blocks = merge_nearby_ocr_blocks(candidate_blocks)
    if len(merged_blocks) < len(candidate_blocks):
        print(f"  [MERGE OCR] {len(candidate_blocks)} block(s) -> {len(merged_blocks)} region(s)")

    filtered_blocks = []
    for block in merged_blocks:
        expanded_bbox = block.get("bbox")
        erasability = assess_erasability(
            image,
            expanded_bbox,
            text=block.get("text", ""),
            source_lang=source_lang,
        )
        if not erasability.get("safe"):
            skipped += 1
            print(
                f"  [SKIP ERASE] '{_short_log_text(block.get('text', ''))}' "
                f"reason={erasability.get('reason')} "
                f"score={float(erasability.get('score', 0)):.2f}"
            )
            continue

        block["_erasability"] = {
            "reason": erasability.get("reason"),
            "score": erasability.get("score"),
        }
        print(
            f"  [SAFE ERASE] '{_short_log_text(block.get('text', ''))}' "
            f"reason={erasability.get('reason')} "
            f"score={float(erasability.get('score', 0)):.2f}"
        )
        filtered_blocks.append(block)
    return filtered_blocks, skipped


def build_preview_images(all_ocr_results, source_lang="ja"):
    """Build preview images for the correction page.
    Uses JPEG compression to reduce transfer size while keeping full resolution
    for accurate coordinate mapping.
    """
    preview_images = []
    for name, image, blocks in all_ocr_results:
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
        preview_images.append({
            "name": str(name),
            "data": encoded,
            "blocks": [
                {
                    "text": str(b.get("text", "") or ""),
                    "bbox": normalize_bbox_for_json(
                        refine_tall_narrow_ocr_bbox(
                            image,
                            b.get("bbox"),
                            source_lang=source_lang,
                            text=b.get("text", ""),
                        ),
                        image_shape=image.shape,
                        expand_ratio=0 if b.get("_bbox_expanded") else BBOX_EXPAND_RATIO,
                    ),
                }
                for b in blocks
            ],
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        })
    return preview_images


def encode_image_jpeg(image, quality=95):
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode image as JPEG")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


# ── Post-render editing (Manual Mode V2) ────────────────────────────────────
# render_plan lives in session.json (only for manual-correction sessions that
# finished a first render). Every re-render starts from the ORIGINAL page_<i>.jpg
# (all_ocr_results is never mutated), so re-rendering is always idempotent.
# Contract (docs/manual-mode-v2-spec.md §4):
#   render_plan[i] = {"name", "erase_regions": [[x1,y1,x2,y2], ...],
#                    "blocks": [{"text", "translated", "bbox"}]}
#   page_<i>_rendered.jpg  = persisted last-rendered image (JPEG q92)
RENDERED_JPEG_QUALITY = 92


def _save_rendered_jpeg(session_id, idx, image):
    """Persist a rendered image as page_<i>_rendered.jpg (JPEG q92)."""
    base = _safe_session_path(session_id)
    if not base:
        return
    os.makedirs(base, exist_ok=True)
    rendered_path = os.path.join(base, "page_" + str(idx) + "_rendered.jpg")
    cv2.imwrite(rendered_path, image, [cv2.IMWRITE_JPEG_QUALITY, RENDERED_JPEG_QUALITY])


def _load_rendered_image(session_id, idx, fallback_image=None):
    """Load page_<i>_rendered.jpg; return None if missing/corrupt."""
    base = _safe_session_path(session_id)
    if not base:
        return None
    rendered_path = os.path.join(base, "page_" + str(idx) + "_rendered.jpg")
    if not os.path.exists(rendered_path):
        return None
    image = cv2.imread(rendered_path)
    if image is None:
        return None
    if fallback_image is not None and image.shape[:2] != fallback_image.shape[:2]:
        # Guard against truncated/stale files: never feed a mismatched canvas.
        return None
    return image


def _normalize_render_plan_entry(entry, image_shape=None):
    """Validate/repair one render_plan entry. Returns a cleaned entry or None.

    Keeps old sessions working: malformed entries are dropped, empty-but-
    recoverable entries get erase_regions rebuilt from all_ocr_results bboxes.
    """
    if not isinstance(entry, dict):
        return None
    blocks_raw = entry.get("blocks")
    if not isinstance(blocks_raw, list):
        return None
    blocks = []
    for block in blocks_raw:
        if not isinstance(block, dict):
            continue
        bbox = normalize_bbox_for_json(
            block.get("bbox"),
            image_shape=image_shape,
            expand_ratio=0,
        )
        if not bbox:
            continue
        block_out = {
            "text": str(block.get("text", "") or ""),
            "translated": str(block.get("translated", "") or ""),
            "bbox": bbox,
        }
        # V3: keep a valid per-block style (spec 4.7); drop invalid styles
        # so legacy sessions never crash.
        style = normalize_block_style(block.get("style"), "animeace_")
        if style is not None:
            block_out["style"] = style
        blocks.append(block_out)
    erase_regions = []
    for region in entry.get("erase_regions", []):
        normalized = normalize_bbox_for_json(
            region,
            image_shape=image_shape,
            expand_ratio=0,
        )
        if normalized:
            erase_regions.append(normalized)
    normalized_entry = {
        "name": str(entry.get("name", "") or ""),
        "erase_regions": erase_regions,
        "blocks": blocks,
    }
    # V3: erase_mask (base64 PNG, white = erase) accumulates across re-renders.
    erase_mask = entry.get("erase_mask")
    if isinstance(erase_mask, str) and erase_mask:
        normalized_entry["erase_mask"] = erase_mask
    return normalized_entry


def _iter_plan_blocks(entry):
    """Yield (original_text, translated_text, bbox, style) from a plan entry."""
    for block in entry.get("blocks", []):
        yield (
            block.get("text", "") or "",
            block.get("translated", "") or "",
            list(block["bbox"]),
            block.get("style") or None,
        )


def _render_plan_from_block_lists(names, block_lists):
    """Build a render_plan from per-image block lists ({text, translated, bbox}).

    erase_regions = bbox of EVERY rendered block so a later re-render always
    erases the original text under a moved/resized bbox (spec §F5, risk R1).
    """
    plan = []
    for name, blocks in zip(names, block_lists):
        entry = {
            "name": name,
            "erase_regions": [list(b["bbox"]) for b in blocks],
            "blocks": blocks,
        }
        plan.append(entry)
    return plan


def _entry_erase_regions(entry, fallback_ocr_bboxes):
    """Erase regions for a plan entry; fall back to OCR bboxes (normalized)."""
    if entry is not None and entry.get("erase_regions"):
        return [list(r) for r in entry["erase_regions"]]
    return list(fallback_ocr_bboxes)


def _purge_stale_rendered_jpegs(session_id, valid_count):
    """Remove page_<i>_rendered.jpg beyond the current plan length."""
    base = _safe_session_path(session_id)
    if not base:
        return
    index = valid_count
    while True:
        path = os.path.join(base, "page_" + str(index) + "_rendered.jpg")
        if not os.path.exists(path):
            break
        try:
            os.remove(path)
        except OSError:
            break
        index += 1


def _persist_render_plan(session_id, session_data, render_plan, rendered_images):
    """Save render_plan + page_*_rendered.jpg into session.json and disk."""
    cleaned_plan = []
    for entry in render_plan:
        normalized = _normalize_render_plan_entry(entry)
        if normalized is not None:
            cleaned_plan.append(normalized)
    session_data["render_plan"] = cleaned_plan
    _save_session(session_id, session_data)
    for idx, image in rendered_images.items():
        _save_rendered_jpeg(session_id, idx, image)
    _purge_stale_rendered_jpegs(session_id, len(cleaned_plan))
    return cleaned_plan


def render_image_with_blocks(name, image, blocks, font_path, source_lang,
                             vision_adapter=None, extra_erase_regions=None,
                             extra_erase_mask=None):
    """Render one image from its ORIGINAL pixels (no OCR, no translation).

    Args:
        name: image name (unused, kept for call-site symmetry).
        image: original BGR numpy image (never mutated).
        blocks: [{"text": original, "translated": str, "bbox": [x1,y1,x2,y2],
                  "style": optional per-block V3 style dict}]
        font_path: font file for translated text.
        source_lang: language code used by erase heuristics.
        vision_adapter: optional adapter that erases + reports appearance
            (determinism across re-renders is NOT guaranteed — spec risk R5).
        extra_erase_regions: additional regions to erase AFTER the blocks
            (original text left behind when a bbox was moved/resized, or a
            deleted block) — spec §F5, risk R2 ordering.
        extra_erase_mask: optional uint8 mask (white = erase) applied after
            the extra regions, before text render (V3 spec F7, P1).

    Returns:
        (image, render_plan_entry) — image is a new array; entry holds the
        normalized blocks that were actually rendered.
    """
    base = image.copy()

    candidates = []
    for block in blocks:
        translated = (block.get("translated") or "").strip()
        if not translated:
            continue
        bbox = normalize_bbox_for_json(block.get("bbox"), image_shape=base.shape, expand_ratio=0)
        if not bbox:
            continue
        candidates.append({
            "text": str(block.get("text", "") or ""),
            "translated": translated,
            "bbox": bbox,
            "style": block.get("style") or None,
        })

    render_blocks = []
    legacy_image = base
    adapter_completed = False
    if candidates and vision_adapter is not None:
        try:
            execution = vision_adapter.process_page(
                base,
                [{"text": item["text"], "bbox": item["bbox"]} for item in candidates],
            )
            if len(execution.prepared) != len(candidates):
                raise RuntimeError(
                    "vision adapter returned a mismatched prepared block count"
                )
            base = execution.erased_image
            erase_results = list(execution.erase_results)
            adapter_completed = True
            for item_index, (item, prepared) in enumerate(zip(candidates, execution.prepared)):
                appearance = appearance_for_prepared(prepared)
                if item_index < len(erase_results):
                    erase_warning = erase_results[item_index].warning
                    if erase_warning:
                        appearance["erase_warning"] = erase_warning
                if appearance.get("should_skip"):
                    continue
                render_blocks.append({
                    "text": item["translated"],
                    "bbox": item["bbox"],
                    "text_color": appearance["text_color"],
                    "appearance": appearance,
                    "style": item.get("style") or None,
                })
        except Exception as exc:
            print("Vision pipeline failed; used legacy erasure: " + str(exc))
            base = legacy_image
            adapter_completed = False
            render_blocks = []

    if candidates and not adapter_completed:
        for item in candidates:
            base, text_color, appearance = erase_text_region(
                base, item["bbox"], source_lang=source_lang
            )
            appearance["should_skip"] = False
            render_blocks.append({
                "text": item["translated"],
                "bbox": item["bbox"],
                "text_color": text_color,
                "appearance": appearance,
                "style": item.get("style") or None,
            })

    # Erase extra regions (original text under moved/resized/deleted bboxes).
    # Runs AFTER block erasure so appearance sampling is not distorted (R2).
    if extra_erase_regions:
        for region in extra_erase_regions:
            normalized = normalize_bbox_for_json(region, image_shape=base.shape, expand_ratio=0)
            if not normalized:
                continue
            overlaps_current = any(
                not (normalized[2] <= rb["bbox"][0] or rb["bbox"][2] <= normalized[0]
                     or normalized[3] <= rb["bbox"][1] or rb["bbox"][3] <= normalized[1])
                for rb in render_blocks
            )
            if overlaps_current:
                continue
            base, _, _ = erase_text_region(base, normalized, source_lang=source_lang)

    # V3: erase arbitrary mask regions (brush/rect strokes, spec F4.6/F7) —
    # also idempotent because the mask is accumulated server-side.
    if extra_erase_mask is not None:
        try:
            base = erase_mask_region(base, extra_erase_mask, source_lang=source_lang)
        except Exception as exc:
            print(f"  [erase_mask] failed: {exc}")

    if render_blocks:
        base = render_all_blocks(base, render_blocks, font_path)

    plan_blocks = []
    for item in candidates:
        plan_block = {
            "text": item["text"],
            "translated": item["translated"],
            "bbox": list(item["bbox"]),
        }
        if item.get("style"):
            plan_block["style"] = item["style"]
        plan_blocks.append(plan_block)
    return base, {"name": name, "erase_regions": [], "blocks": plan_blocks}


def build_result_images(processed_results, original_images_by_name=None):
    original_images_by_name = original_images_by_name or {}
    processed_images = []
    for result in processed_results:
        image = result['image']
        name = result['name']
        item = {
            "name": name,
            "data": encode_image_jpeg(image, quality=95),
        }
        original_image = original_images_by_name.get(name)
        if original_image is not None:
            item["original_data"] = encode_image_jpeg(original_image, quality=90)
        processed_images.append(item)
    return processed_images


def snapshot_original_images(all_ocr_results):
    return {name: image.copy() for name, image, _ in all_ocr_results}


def translate_texts_all(all_texts, translator_obj, translator_type,
                        source_lang='ja', target_lang='vi'):
    """Phase 2: batch-translate all texts (shared by the full pipeline and
    the V3 style-editor prepare flow).

    Mirrors the original translate_and_render Phase 2 exactly: on failure the
    original texts are kept and translator_obj.last_warning is set so the UI
    can warn the user.

    Returns:
        (translated_texts, warning) — warning is None when translation
        succeeded without a fallback.
    """
    emit_progress('translation', 0, 1, f'Đang dịch {len(all_texts)} đoạn text...')
    print(f"\n[Phase 2] Translating {len(all_texts)} text segments...")

    if translator_type == "gemini":
        gemini_translator = getattr(translator_obj, '_gemini_translator', None)
        if gemini_translator is None:
            print("Gemini translator is not initialized; keeping original texts")
            translator_obj.last_warning = (
                "Gemini chưa được khởi tạo nên app giữ nguyên text gốc. "
                "Hãy kiểm tra API key rồi thử lại."
            )
            translated_texts = all_texts
        else:
            try:
                translated_texts = gemini_translator.translate_batch(all_texts, source_lang, target_lang)
            except Exception as e:
                print(f"Gemini batch failed: {e}, falling back to single")
                try:
                    translated_texts = [
                        gemini_translator.translate_single(t, source_lang, target_lang)
                        for t in all_texts
                    ]
                except Exception as e2:
                    print(f"Gemini single translation also failed: {e2}")
                    translator_obj.last_warning = (
                        "Gemini không dịch được nên app giữ nguyên text gốc. "
                        "Hãy kiểm tra API key/quota rồi thử lại."
                    )
                    translated_texts = all_texts

    elif translator_type == "copilot":
        try:
            from translator.local_llm_translator import LocalLLMTranslator
            if not hasattr(translator_obj, '_local_llm_tr') or translator_obj._local_llm_tr is None:
                translator_obj._local_llm_tr = LocalLLMTranslator(
                    server_url=getattr(translator_obj, '_copilot_server', 'http://localhost:8080'),
                    model=getattr(translator_obj, '_copilot_model', 'gpt-4o'),
                    custom_prompt=getattr(translator_obj, '_copilot_custom_prompt', None)
                )
            translated_texts = translator_obj._local_llm_tr.translate_batch(all_texts, source_lang, target_lang)
        except Exception as e:
            print(f"Local LLM batch failed: {e}, falling back to single translations")
            emit_progress('translation', 0, 1, f'Batch failed, falling back to single translations...')
            try:
                translated_texts = [translator_obj._local_llm_tr.translate_single(t, source_lang, target_lang) for t in all_texts]
            except Exception as e2:
                print(f"Local LLM single translation also failed: {e2}")
                translator_obj.last_warning = (
                    "Local LLM không dịch được nên app giữ nguyên text gốc. "
                    "Hãy kiểm tra server URL/model rồi thử lại."
                )
                translated_texts = all_texts

    elif translator_type == "google":
        try:
            translated_texts = translator_obj.translate_batch_google(all_texts)
        except Exception as e:
            print(f"Google batch translation failed: {e}")
            translator_obj.last_warning = "Google Translate lỗi nên app giữ nguyên text gốc."
            translated_texts = all_texts

    else:
        print(f"WARNING: Unrecognized translator type '{translator_type}', no translation performed")
        emit_progress('translation', 0, 1, f'Cảnh báo: Translator không xác định, text không được dịch')
        translator_obj.last_warning = "Translator không xác định nên app giữ nguyên text gốc."
        translated_texts = all_texts

    print("OK Translation completed")
    emit_progress('translation', 1, 1, 'Dịch hoàn tất')
    return translated_texts, getattr(translator_obj, "last_warning", None)

def translate_and_render(all_ocr_results, translator_obj, selected_font, translator_type,
                         source_lang, target_lang, style, *, vision_adapter=None,
                         collect_render_plan=False):
    """Phase 2+3: Translate all texts then render (used when OCR is already done)

    Returns a list of {'name', 'image'} results (backward compatible). When
    collect_render_plan=True, returns (processed_results, render_plan) instead
    so the manual-mode session can persist post-render editing state.
    """
    total_images = len(all_ocr_results)
    all_texts = []
    for _, _, blocks in all_ocr_results:
        for block in blocks:
            text = block.get('text', '').strip()
            if text:
                block['_text_idx'] = len(all_texts)
                all_texts.append(text)
    
    if not all_texts:
        print("No text to translate.")
        emit_progress('done', total_images, total_images, 'Không có text để dịch')
        results = [{'name': name, 'image': image} for name, image, _ in all_ocr_results]
        if collect_render_plan:
            empty_plan = [
                {"name": name, "erase_regions": [], "blocks": []}
                for name, _, _ in all_ocr_results
            ]
            return results, empty_plan
        return results
    
    # Phase 2: Batch translate
    translated_texts, _phase2_warning = translate_texts_all(
        all_texts, translator_obj, translator_type,
        source_lang=source_lang, target_lang=target_lang,
    )

    # Phase 3: Render
    emit_progress('rendering', 0, total_images, 'Đang render text vào ảnh...')
    print(f"\n[Phase 3] Rendering translated text...")

    font_path = get_font_path(selected_font)
    processed_results = []
    render_plan = []

    # Prepare rendering data for each image
    skipped_count = 0

    # Sequential rendering — ThreadPoolExecutor provides negligible speedup here
    # because PIL ImageDraw operations hold the GIL, and the progress_lock +
    # emit_progress calls serialize most of the parallel work anyway.
    for idx, (name, image, blocks) in enumerate(all_ocr_results):
        emit_progress('rendering', idx + 1, total_images, f'Render: {name}')

        # Build per-image render candidates (same filtering as before)
        candidates = []
        for block in blocks:
            text = block.get('text', '').strip()
            if not text:
                continue
            bbox = block.get('bbox')
            if not bbox or len(bbox) < 4:
                continue

            text_idx = block.get('_text_idx', -1)
            if 0 <= text_idx < len(translated_texts):
                translated = translated_texts[text_idx]
            else:
                translated = text
            if not translated or not translated.strip():
                continue
            if should_skip_ocr_artifact(text, bbox, image_shape=image.shape,
                                        source_lang=source_lang):
                skipped_count += 1
                print(f"  [SKIP OCR ARTIFACT] '{text}'")
                continue
            candidates.append({
                'text': text,
                'translated': translated,
                'bbox': list(bbox),
            })

        rendered_image, plan_entry = render_image_with_blocks(
            name, image, candidates, font_path, source_lang,
            vision_adapter=vision_adapter,
        )
        processed_results.append({'name': name, 'image': rendered_image})

        # erase_regions = bbox of EVERY rendered block so a later re-render
        # always erases the original text under a moved/resized bbox (spec F5/R1).
        plan_entry['erase_regions'] = [list(item['bbox']) for item in candidates]
        render_plan.append(plan_entry)

    print("OK Rendering completed")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} OCR artifact block(s)")
    emit_progress('done', total_images, total_images, f'Hoàn tất! {total_images} ảnh')

    if collect_render_plan:
        return processed_results, render_plan
    return processed_results


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def upload_file():
    manual_correction = request.form.get("manual_correction", "").strip() == "on"
    
    translator_map = {
        "Local LLM": "copilot",
        "Copilot": "copilot"
    }
    selected_translator_raw = request.form["selected_translator"]
    selected_translator = translator_map.get(selected_translator_raw, selected_translator_raw.lower())
    
    copilot_server = request.form.get("copilot_server", "http://localhost:8080")
    copilot_model = request.form.get("copilot_model_input", "gpt-4o")
    gemini_model = request.form.get("gemini_model_input", DEFAULT_GEMINI_MODEL).strip()
    gemini_api_keys = parse_gemini_api_keys(request.form.get("gemini_api_key", ""))
    if selected_translator == "gemini" and not gemini_api_keys:
        return render_template("index.html", error="Vui lòng nhập ít nhất 1 Gemini API Key.")
    if selected_translator == "gemini" and not gemini_model:
        return render_template("index.html", error="Vui lòng nhập tên model Gemini.")
    
    selected_font_raw = request.form["selected_font"]
    selected_font = selected_font_raw.lower()
    if selected_font == "auto (match original)":
        selected_font = "animeace_"
    elif selected_font == "animeace":
        selected_font = "animeace_"
    elif selected_font_raw.startswith("Yuki-"):
        selected_font = selected_font_raw
    
    source_lang_map = {
        "japanese (manga)": "ja",
        "chinese (manhua)": "zh",
        "korean (manhwa)": "ko",
        "english (comic)": "en"
    }
    selected_source = request.form.get("selected_source_lang", "Japanese (Manga)").lower()
    source_lang = source_lang_map.get(selected_source, "ja")
    
    target_lang_map = {
        "english": "en", "vietnamese": "vi", "chinese": "zh", "korean": "ko",
        "thai": "th", "indonesian": "id", "french": "fr", "german": "de",
        "spanish": "es", "russian": "ru"
    }
    selected_language = request.form.get("selected_language", "Vietnamese").lower()
    target_lang = target_lang_map.get(selected_language, "vi")
    
    style_map = {
        "default": "", "casual (thân mật)": "casual", "formal (trang trọng)": "formal",
        "keep honorifics (-san, senpai...)": "keep_honorifics",
        "web novel style": "web_novel", "action (ngắn gọn)": "action",
        "literal (sát nghĩa)": "literal", "custom...": ""
    }
    selected_style = request.form.get("selected_style", "Default").lower()
    style = style_map.get(selected_style, "")
    custom_prompt = request.form.get("custom_prompt", "").strip()
    if custom_prompt:
        style = custom_prompt
    
    files = request.files.getlist("files")
    if not files or files[0].filename == '':
        return render_template("index.html", error="Vui lòng chọn ít nhất 1 ảnh để dịch.")
    
    ocr_engine = ChromeLensOCR(ocr_language=source_lang)
    
    all_images = []
    unsupported_files = [
        file.filename for file in files
        if file and file.filename and not is_allowed_image_file(file.filename)
    ]
    if unsupported_files:
        return render_template(
            "index.html",
            error=f"Chỉ hỗ trợ ảnh {SUPPORTED_IMAGE_FORMATS_LABEL}.",
        )

    for file in files:
        if file and file.filename:
            try:
                file_stream = file.stream
                file_bytes = np.frombuffer(file_stream.read(), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                name = clean_image_name(file.filename)
                all_images.append({'image': image, 'name': name})
            except Exception as e:
                print(f"Error reading {file.filename}: {e}")
    
    if not all_images:
        return render_template(
            "index.html",
            error=f"Không đọc được ảnh. Hãy thử file {SUPPORTED_IMAGE_FORMATS_LABEL} khác.",
        )
    
    # Phase 1: OCR all images (batch concurrent)
    print("\n[Phase 1] OCR full images with Chrome Lens...")
    emit_progress('ocr', 0, len(all_images), 'Bắt đầu OCR toàn ảnh...')

    # Use batch processing when multiple images
    raw_images = [d['image'] for d in all_images]
    names = [d['name'] for d in all_images]

    if len(raw_images) > 1:
        batch_results = ocr_engine.process_batch(raw_images)
    else:
        batch_results = [ocr_engine(raw_images[0])]

    all_ocr_results = []
    all_texts = []
    text_index = 0

    for idx, blocks in enumerate(batch_results):
        name = names[idx]
        image = raw_images[idx]
        blocks, skipped_artifacts = filter_ocr_blocks(blocks, image, source_lang)
        skip_msg = f", skipped {skipped_artifacts} artifact(s)" if skipped_artifacts else ""
        print(f"  [{idx+1}/{len(all_images)}] OCR {name}: {len(blocks)} text blocks found{skip_msg}")

        all_ocr_results.append((name, image, blocks))
        all_texts.extend([b['text'] for b in blocks if b.get('text', '').strip()])

        for block in blocks:
            if block.get('text', '').strip():
                block['_text_idx'] = text_index
                text_index += 1
    
    ocr_blocks_count = sum(len(blocks) for _, _, blocks in all_ocr_results)
    print(f"OK OCR completed: {ocr_blocks_count} text blocks across {len(all_images)} images")
    emit_progress('ocr', len(all_images), len(all_images), f'OCR hoàn tất: {ocr_blocks_count} text blocks')
    
    # If manual correction is enabled, store session and redirect to correction page
    if manual_correction:
        cleanup_old_sessions()
        session_id = str(uuid.uuid4())
        session_data = {
            'all_ocr_results': all_ocr_results,
            'all_texts': all_texts,
            'selected_translator': selected_translator,
            'selected_font': selected_font,
            'source_lang': source_lang,
            'target_lang': target_lang,
            'style': style,
            'gemini_api_keys': gemini_api_keys,
            'gemini_api_key': "\n".join(gemini_api_keys),
            'gemini_model': gemini_model,
            'copilot_server': copilot_server,
            'copilot_model': copilot_model,
            'translator_type': selected_translator,
        }
        
        # Save to disk as JSON + JPEG (lightweight, no pickle)
        _save_session(session_id, session_data)
        
        # Generate preview images for correction page
        preview_images = build_preview_images(all_ocr_results, source_lang=source_lang)
        
        return render_template("correction.html", 
                             session_id=session_id,
                             images=preview_images,
                             total_blocks=ocr_blocks_count)
    
    # Normal flow: continue to translate -> render
    return _do_full_pipeline(all_images, all_ocr_results, all_texts,
                           selected_translator, selected_font,
                           source_lang, target_lang, style,
                           gemini_api_keys, gemini_model, copilot_server, copilot_model)


def _do_full_pipeline(all_images, all_ocr_results, all_texts,
                      selected_translator, selected_font,
                      source_lang, target_lang, style,
                      gemini_api_keys, gemini_model, copilot_server, copilot_model,
                      correction_session_id=None):
    
    translator_obj = MangaTranslator(source=source_lang, target=target_lang)
    
    if selected_translator == "gemini":
        translator_obj._gemini_custom_prompt = style if style else None
        translator_obj._gemini_api_keys = gemini_api_keys
        translator_obj._gemini_model = gemini_model
        from translator.gemini_translator import GeminiTranslator
        translator_obj._gemini_translator = GeminiTranslator(
            api_keys=gemini_api_keys,
            custom_prompt=style if style else None,
            model_name=gemini_model,
        )
        print(f"Gemini translator initialized with {len(gemini_api_keys)} key(s), model: {gemini_model}")
    
    elif selected_translator == "copilot":
        translator_obj._copilot_server = copilot_server
        translator_obj._copilot_model = copilot_model
        translator_obj._copilot_custom_prompt = style if style else None
        print(f"Local LLM: {copilot_server} / model: {copilot_model}")
    
    elif selected_translator == "google":
        print(f"Using Google Translate")
    
    original_images_by_name = snapshot_original_images(all_ocr_results)

    vision_adapter = build_optional_vision_adapter()
    processed_results, render_plan = translate_and_render(
        all_ocr_results, translator_obj, selected_font,
        translator_type=selected_translator,
        source_lang=source_lang, target_lang=target_lang, style=style,
        vision_adapter=vision_adapter,
        collect_render_plan=True,
    )
    warning = None
    warning = getattr(translator_obj, "last_warning", None)
    gemini_translator = getattr(translator_obj, "_gemini_translator", None)
    if gemini_translator is not None:
        warning = getattr(gemini_translator, "last_warning", None) or warning
    
    # Manual Mode V2: persist render_plan + rendered JPEGs so the post-render
    # editor can re-render single images without OCR/translation (spec §4.4).
    if correction_session_id:
        try:
            session_data = load_session(correction_session_id)
            if session_data is not None:
                rendered_images = {
                    idx: result['image']
                    for idx, result in enumerate(processed_results)
                }
                _persist_render_plan(
                    correction_session_id, session_data, render_plan, rendered_images
                )
        except Exception as e:
            print(f"Failed to persist render plan for {correction_session_id}: {e}")
    
    try:
        processed_images = build_result_images(processed_results, original_images_by_name)
    except Exception as e:
        print(f"Error encoding result images: {e}")
        processed_images = []
    
    if not processed_images:
        return redirect("/")
    
    return render_template(
        "translate.html",
        images=processed_images,
        warning=warning,
        correction_session_id=correction_session_id,
    )


@app.route("/correction/<session_id>")
def correction_page(session_id):
    """Reload correction page from session data"""
    session_data = load_session(session_id)
    if session_data is None:
        return redirect("/")
    
    all_ocr_results = session_data['all_ocr_results']
    preview_images = build_preview_images(
        all_ocr_results,
        source_lang=session_data.get('source_lang', 'ja'),
    )
    
    ocr_blocks_count = sum(len(blocks) for _, _, blocks in all_ocr_results)
    return render_template("correction.html",
                         session_id=session_id,
                         images=preview_images,
                         total_blocks=ocr_blocks_count)


def rebuild_ocr_from_modified_blocks(session_data, modified_blocks):
    """Rebuild all_ocr_results from frontend-modified blocks (V2/V3 shared).

    Returns (new_ocr_results, all_texts): new_ocr_results is a list of
    (name, image, blocks) with blocks sorted in reading order and every text
    block tagged with a global _text_idx. Invalid image indices are skipped.
    """
    all_ocr_results = session_data['all_ocr_results']
    all_texts = []
    text_index = 0

    new_ocr_results = []
    for img_data in modified_blocks:
        try:
            img_idx = int(img_data.get('image_idx'))
        except (TypeError, ValueError, AttributeError):
            continue
        if img_idx < 0 or img_idx >= len(all_ocr_results):
            continue
        name, original_image, _ = all_ocr_results[img_idx]

        blocks = []
        for b in img_data.get('blocks', []):
            if not isinstance(b, dict):
                continue
            text = b.get('text', '').strip()
            bbox = normalize_bbox_for_json(b.get('bbox', None),
                                           image_shape=original_image.shape,
                                           expand_ratio=0)
            if bbox and len(bbox) == 4:
                blocks.append({'text': text, 'bbox': bbox, '_bbox_expanded': True})

        blocks = sort_ocr_blocks_reading_order(blocks)
        for block in blocks:
            if block['text']:
                block['_text_idx'] = text_index
                text_index += 1
                all_texts.append(block['text'])

        new_ocr_results.append((name, original_image, blocks))
    return new_ocr_results, all_texts

@app.route("/continue-translate", methods=["POST"])
def continue_translate():
    """Continue pipeline after manual correction"""
    session_id = request.form.get("session_id", "")
    modified_blocks_json = request.form.get("modified_blocks", "[]")
    
    session_data = load_session(session_id)
    if session_data is None:
        return redirect("/")
    
    # Parse modified blocks from frontend
    # modified_blocks = [{"image_idx": 0, "blocks": [{"text": "...", "bbox": [...]}, ...]}, ...]
    try:
        modified_blocks = json.loads(modified_blocks_json)
    except json.JSONDecodeError:
        return redirect("/")
    if isinstance(modified_blocks, dict):
        modified_blocks = [modified_blocks]
    if not isinstance(modified_blocks, list):
        return redirect("/")
    
    # Rebuild all_ocr_results from modified blocks (shared V2/V3 helper)
    new_ocr_results, all_texts = rebuild_ocr_from_modified_blocks(
        session_data, modified_blocks
    )
    
    # Build all_images list
    all_images = [{'image': img, 'name': name} for name, img, _ in new_ocr_results]
    
    if not all_texts:
        emit_progress('done', 0, 0, 'Không có text để dịch')
        processed_results = [{'name': name, 'image': image} for name, image, _ in new_ocr_results]
        original_images_by_name = {name: image for name, image, _ in new_ocr_results}
        processed_images = build_result_images(processed_results, original_images_by_name)
        return render_template(
            "translate.html",
            images=processed_images,
            correction_session_id=session_id,
        )
    
    return _do_full_pipeline(
        all_images, new_ocr_results, all_texts,
        session_data['selected_translator'], session_data['selected_font'],
        session_data['source_lang'], session_data['target_lang'], session_data['style'],
        session_data.get('gemini_api_keys') or parse_gemini_api_keys(session_data.get('gemini_api_key', '')),
        session_data.get('gemini_model', DEFAULT_GEMINI_MODEL),
        session_data['copilot_server'], session_data['copilot_model'],
        correction_session_id=session_id,
    )


@app.route("/ocr-region", methods=["POST"])
def ocr_region():
    session_id = request.form.get("session_id", "")
    try:
        image_idx = int(request.form.get("image_idx", "0"))
        x1 = int(request.form.get("x1", "0"))
        y1 = int(request.form.get("y1", "0"))
        x2 = int(request.form.get("x2", "0"))
        y2 = int(request.form.get("y2", "0"))
    except (TypeError, ValueError):
        return {"text": ""}, 400

    session_data = load_session(session_id)
    if session_data is None:
        return {"text": ""}, 404

    all_ocr_results = session_data['all_ocr_results']
    if image_idx < 0 or image_idx >= len(all_ocr_results):
        return {"text": ""}, 400

    _, original_image, _ = all_ocr_results[image_idx]
    h, w = original_image.shape[:2]

    # Pad the region slightly
    pad = 4
    cx1 = max(0, int(x1) - pad)
    cy1 = max(0, int(y1) - pad)
    cx2 = min(w, int(x2) + pad)
    cy2 = min(h, int(y2) + pad)

    if cx2 <= cx1 or cy2 <= cy1:
        return {"text": ""}

    cropped = original_image[cy1:cy2, cx1:cx2]

    ocr_engine = ChromeLensOCR(ocr_language=session_data.get('source_lang', 'ja'))
    blocks = sort_ocr_blocks_reading_order(ocr_engine(cropped))

    text = " ".join(b.get("text", "").strip() for b in blocks if b.get("text", "").strip())
    return {"text": text}


# ── Manual Mode V3: style editor (spec docs/manual-mode-v3-spec.md §4) ─────


def _build_translator_from_session(session_data):
    """Construct a configured MangaTranslator from stored session options."""
    translator_obj = MangaTranslator(
        source=session_data.get('source_lang', 'ja'),
        target=session_data.get('target_lang', 'vi'),
    )
    style = session_data.get('style', '')
    translator_type = session_data.get('translator_type', 'google')
    if translator_type == "gemini":
        gemini_keys = session_data.get('gemini_api_keys') or parse_gemini_api_keys(
            session_data.get('gemini_api_key', '')
        )
        translator_obj._gemini_custom_prompt = style if style else None
        translator_obj._gemini_api_keys = gemini_keys
        translator_obj._gemini_model = session_data.get('gemini_model', DEFAULT_GEMINI_MODEL)
        from translator.gemini_translator import GeminiTranslator
        translator_obj._gemini_translator = GeminiTranslator(
            api_keys=gemini_keys,
            custom_prompt=style if style else None,
            model_name=session_data.get('gemini_model', DEFAULT_GEMINI_MODEL),
        )
    elif translator_type == "copilot":
        translator_obj._copilot_server = session_data.get('copilot_server', 'http://localhost:8080')
        translator_obj._copilot_model = session_data.get('copilot_model', 'gpt-4o')
        translator_obj._copilot_custom_prompt = style if style else None
    return translator_obj


def _save_erased_jpeg(session_id, idx, image):
    """Persist page_<i>_erased.jpg (JPEG q92) — erased background canvas of the
    V3 style editor (spec F2).
    """
    base = _safe_session_path(session_id)
    if not base:
        return
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, "page_" + str(idx) + "_erased.jpg")
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 92])


def _load_erased_image(session_id, idx, fallback_image=None):
    """Load page_<i>_erased.jpg; None when missing/corrupt/mismatched."""
    base = _safe_session_path(session_id)
    if not base:
        return None
    path = os.path.join(base, "page_" + str(idx) + "_erased.jpg")
    if not os.path.exists(path):
        return None
    image = cv2.imread(path)
    if image is None:
        return None
    if fallback_image is not None and image.shape[:2] != fallback_image.shape[:2]:
        return None
    return image


def _decode_erase_mask(mask_b64, image_shape):
    """Decode a base64 PNG erase mask (white = erase) and resize to image size.

    Returns a uint8 BGR mask at image resolution, or None when invalid
    (spec F7: client downscales to <= 2048 long side; server upsamples).
    """
    if not mask_b64 or not isinstance(mask_b64, str):
        return None
    try:
        raw = base64.b64decode(mask_b64)
    except (ValueError, TypeError):
        return None
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        return None
    if decoded.ndim == 3 and decoded.shape[2] >= 3:
        mask = decoded[:, :, :3]
    else:
        mask = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
    h, w = image_shape[:2]
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def _sync_draft_after_render(session_id, session_data, image_idx, client_blocks):
    """Merge the last rendered client state back into v3_draft (V3 A5.10).

    The client block list is the editor source of truth: styles and edits
    applied in the editor must survive a reload of /styleditor. Blocks that
    are missing from the client list were deleted in the editor and are
    removed from the draft; blocks with an empty translation are kept as
    "-" chips (spec A1.6). Matching is by original text (stable within an
    editor session), which stays correct when blocks are moved or deleted.
    """
    draft = session_data.get("v3_draft")
    if not isinstance(draft, dict):
        return
    draft_images = draft.get("images")
    if not isinstance(draft_images, list) or image_idx >= len(draft_images):
        return
    draft_entry = draft_images[image_idx]
    if not isinstance(draft_entry, dict):
        return
    draft_blocks = draft_entry.get("blocks")
    if not isinstance(draft_blocks, list):
        return

    remaining = [cb for cb in client_blocks if isinstance(cb, dict)]
    used = [False] * len(remaining)
    kept = []
    for db in draft_blocks:
        if not isinstance(db, dict):
            continue
        match = None
        for i, cb in enumerate(remaining):
            if not used[i] and str(cb.get("text") or "") == str(db.get("text") or ""):
                match = cb
                used[i] = True
                break
        if match is None:
            continue  # deleted in the editor -> drop from the draft
        merged = dict(db)
        merged["translated"] = str(match.get("translated") or "")
        bbox = match.get("bbox")
        if bbox:
            merged["bbox"] = list(bbox)
        style = normalize_block_style(
            match.get("style"), session_data.get("selected_font", "animeace_")
        )
        if style is not None:
            merged["style"] = style
        kept.append(merged)
    draft_entry["blocks"] = kept
    _save_session(session_id, session_data)


@app.route("/api/fonts")
def api_fonts():
    """List available fonts from fonts/ (spec section 4.4)."""
    return {"fonts": list_available_fonts()}


@app.route("/font-file/<name>")
def font_file(name):
    """Serve a font TTF for FontFace (spec section 4.5). Whitelist only —
    name must exactly match a fonts/ entry, so path traversal is impossible.
    """
    font_names = {entry['name'] for entry in list_available_fonts()}
    if name not in font_names:
        return {"error": "font_not_found"}, 404
    path = get_font_path(name)
    if not path or not os.path.isfile(path):
        return {"error": "font_not_found"}, 404
    return send_file(path, mimetype="font/ttf", as_attachment=False)


@app.route("/styleditor-prepare", methods=["POST"])
def styleditor_prepare():
    """V3: translate (Phase 2 ONLY) + erase original text, then open the
    style editor on the erased backgrounds. Never renders and never re-runs
    OCR (spec F1/A1.4). Payload identical to /continue-translate.
    """
    session_id = request.form.get("session_id", "")
    modified_blocks_json = request.form.get("modified_blocks", "[]")

    session_data = load_session(session_id)
    if session_data is None:
        return redirect("/")

    try:
        modified_blocks = json.loads(modified_blocks_json)
    except json.JSONDecodeError:
        return redirect("/")
    if isinstance(modified_blocks, dict):
        modified_blocks = [modified_blocks]
    if not isinstance(modified_blocks, list):
        return redirect("/")

    new_ocr_results, all_texts = rebuild_ocr_from_modified_blocks(
        session_data, modified_blocks
    )

    if not all_texts:
        # No text at all: behave like V2 /continue-translate (straight to the
        # results page, spec A1.8).
        emit_progress('done', 0, 0, 'Không có text để dịch')
        processed_results = [{'name': name, 'image': image} for name, image, _ in new_ocr_results]
        original_images_by_name = {name: image for name, image, _ in new_ocr_results}
        processed_images = build_result_images(processed_results, original_images_by_name)
        return render_template(
            "translate.html",
            images=processed_images,
            correction_session_id=session_id,
        )

    try:
        translator_obj = _build_translator_from_session(session_data)
        translated_texts, warning = translate_texts_all(
            all_texts, translator_obj,
            session_data.get('translator_type', 'google'),
            source_lang=session_data.get('source_lang', 'ja'),
            target_lang=session_data.get('target_lang', 'vi'),
        )
    except Exception as exc:
        # Translator failure must still open the editor with original texts
        # (spec F1.5/A10.3).
        print(f"[styleditor-prepare] translator init failed: {exc}")
        translated_texts = all_texts
        warning = f"Lỗi dịch: {exc}"

    default_font = session_data.get('selected_font', 'animeace_')
    default_style = {
        "font": default_font,
        "font_size": 0,
        "text_color": None,
        "bold": False,
        "italic": False,
        "align": "center",
    }
    source_lang = session_data.get('source_lang', 'ja')

    draft_images = []
    for idx, (name, image, blocks) in enumerate(new_ocr_results):
        draft_blocks = []
        erased_base = image.copy()
        for block in blocks:
            text = block.get('text', '')
            text_idx = block.get('_text_idx', -1)
            translated = translated_texts[text_idx] if 0 <= text_idx < len(translated_texts) else text
            bbox = block.get('bbox')
            if not bbox or len(bbox) < 4:
                continue
            draft_blocks.append({
                "text": str(text),
                "translated": str(translated or ""),
                "bbox": list(bbox),
                "style": dict(default_style),
            })
            erased_base, _, _ = erase_text_region(erased_base, bbox, source_lang=source_lang)
        draft_images.append({"name": str(name), "blocks": draft_blocks})
        _save_erased_jpeg(session_id, idx, erased_base)

    session_data["v3_draft"] = {"images": draft_images}
    session_data["v3_last_warning"] = warning or None
    # A new draft generation starts clean: drop any render state from a
    # previous prepare so /styleditor can never serve stale plan blocks or
    # stale rendered jpegs (MERGE RULE gate = page_<i>_rendered.jpg exists).
    session_data.pop("render_plan", None)
    _purge_stale_rendered_jpegs(session_id, 0)
    _save_session(session_id, session_data)

    print(
        f"[styleditor-prepare] {len(draft_images)} image(s), {len(all_texts)} text(s) translated;",
        f" warning={bool(warning)}",
    )
    return redirect("/styleditor/" + session_id + "?img=0")


@app.route("/styleditor/<session_id>")
def styleditor_page(session_id):
    """V3 style editor page: erased background + translated blocks (spec 4.2).

    Missing session -> home; session without v3_draft -> correction page
    (risk R6 fallback).
    """
    session_data = load_session(session_id)
    if session_data is None:
        return redirect("/")

    draft = session_data.get("v3_draft")
    if not isinstance(draft, dict) or not isinstance(draft.get("images"), list) or not draft["images"]:
        return redirect("/correction/" + session_id)

    draft_images = draft["images"]
    raw_idx = request.args.get("img", "0")
    try:
        image_idx = int(raw_idx)
    except (TypeError, ValueError):
        image_idx = 0
    if image_idx < 0 or image_idx >= len(draft_images):
        image_idx = 0

    all_ocr_results = session_data['all_ocr_results']
    if image_idx >= len(all_ocr_results):
        return redirect("/correction/" + session_id)

    name, original_image, _ = all_ocr_results[image_idx]
    erased_image = _load_erased_image(session_id, image_idx, original_image)
    if erased_image is None:
        erased_image = original_image

    _, buffer = cv2.imencode(".jpg", erased_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")

    default_font = session_data.get('selected_font', 'animeace_')
    default_style = {
        "font": default_font,
        "font_size": 0,
        "text_color": None,
        "bold": False,
        "italic": False,
        "align": "center",
    }

    # [MERGE RULE — spec 4.2] render_plan[i] is the source of truth for
    # images that were already rendered: blocks (text/translated/bbox/
    # style) come from the plan so reloading the editor keeps styles and
    # edits (A5.10); erase state is returned for the client to restore
    # the erase preview (A4.10). v3_draft is used for images never
    # rendered, and draft-only blocks (empty translation -> "—" chip,
    # A1.6) are appended after the plan blocks.
    # Gate: only merge from the plan when this image was ACTUALLY rendered.
    # _load_render_plan fabricates a legacy plan for sessions without a raw
    # render_plan — a fresh prepare session must keep using the draft, so the
    # merge requires page_<i>_rendered.jpg to exist (written by every render).
    base_dir = _safe_session_path(session_id)
    rendered_jpeg_exists = bool(base_dir) and os.path.exists(
        os.path.join(base_dir, "page_%d_rendered.jpg" % image_idx)
    )
    # A REAL plan must also exist in session_data: _load_render_plan would
    # otherwise fabricate a legacy plan (empty translations) for a session
    # that has a stale jpeg but no render state.
    raw_plan = session_data.get("render_plan")
    has_real_plan = isinstance(raw_plan, list) and bool(raw_plan)
    plan, _ocr_bboxes = _load_render_plan(session_data)
    plan_entry = (
        plan[image_idx]
        if rendered_jpeg_exists and has_real_plan and image_idx < len(plan)
        else None
    )
    draft_blocks = draft_images[image_idx].get("blocks") or []
    if not isinstance(draft_blocks, list):
        draft_blocks = []

    erase_regions = []
    erase_mask = None
    blocks_out = []
    if plan_entry is not None:
        plan_blocks = plan_entry.get("blocks") or []
        used_texts = set()
        for pb in plan_blocks:
            if not isinstance(pb, dict):
                continue
            bbox = normalize_bbox_for_json(
                pb.get("bbox"), image_shape=erased_image.shape, expand_ratio=0
            )
            if not bbox:
                continue
            used_texts.add(str(pb.get("text") or ""))
            blocks_out.append({
                "text": str(pb.get("text") or ""),
                "translated": str(pb.get("translated") or ""),
                "bbox": bbox,
                "style": normalize_block_style(pb.get("style"), default_font) or dict(default_style),
            })
        for db in draft_blocks:
            if not isinstance(db, dict):
                continue
            if str(db.get("text") or "") in used_texts:
                continue
            bbox = normalize_bbox_for_json(
                db.get("bbox"), image_shape=erased_image.shape, expand_ratio=0
            )
            if not bbox:
                continue
            blocks_out.append({
                "text": str(db.get("text") or ""),
                "translated": str(db.get("translated") or ""),
                "bbox": bbox,
                "style": normalize_block_style(db.get("style"), default_font) or dict(default_style),
            })
        erase_regions = [list(r) for r in (plan_entry.get("erase_regions") or [])]
        raw_mask = plan_entry.get("erase_mask")
        if isinstance(raw_mask, str) and raw_mask:
            erase_mask = raw_mask
    else:
        for b in draft_blocks:
            if not isinstance(b, dict):
                continue
            bbox = normalize_bbox_for_json(
                b.get("bbox"), image_shape=erased_image.shape, expand_ratio=0
            )
            if not bbox:
                continue
            blocks_out.append({
                "text": b.get("text", "") or "",
                "translated": b.get("translated", "") or "",
                "bbox": bbox,
                "style": normalize_block_style(b.get("style"), default_font) or dict(default_style),
            })

    image_data = {
        "name": str(draft_images[image_idx].get("name") or name),
        "data": encoded,
        "blocks": blocks_out,
        "erase_regions": erase_regions,
        "width": int(erased_image.shape[1]),
        "height": int(erased_image.shape[0]),
    }
    if erase_mask:
        image_data["erase_mask"] = erase_mask
    total_blocks = sum(len(img.get("blocks", [])) for img in draft_images)
    all_images = [
        {"name": str(img.get("name") or ("page%d" % i)), "idx": i}
        for i, img in enumerate(draft_images)
    ]
    return render_template(
        "correction.html",
        session_id=session_id,
        images=[image_data],
        total_blocks=total_blocks,
        mode="styleditor",
        postrender_image_idx=image_idx,
        all_images=all_images,
        warning=session_data.get("v3_last_warning"),
    )

# ── Post-render editing routes (Manual Mode V2, spec §4.1–4.3) ──────────────

def _load_render_plan(session_data):
    """Return (render_plan, ocr_bboxes) with lazy repair of legacy sessions.

    render_plan: cleaned list of entries (may be shorter than all_ocr_results).
    ocr_bboxes: per-image list of normalized original bboxes (fallback erase
    regions and a rebuild source when an entry is missing/corrupt).
    """
    all_ocr_results = session_data.get('all_ocr_results', [])
    ocr_bboxes = []
    for _, image, blocks in all_ocr_results:
        image_bboxes = []
        for block in blocks:
            bbox = normalize_bbox_for_json(
                block.get('bbox'), image_shape=image.shape, expand_ratio=0
            )
            if bbox:
                image_bboxes.append(bbox)
        ocr_bboxes.append(image_bboxes)

    raw_plan = session_data.get('render_plan')
    if not isinstance(raw_plan, list):
        # Legacy session (rendered before V2): rebuild a plan from current
        # OCR blocks so post-render editing still works.
        plan = []
        for i, (name, _, blocks) in enumerate(all_ocr_results):
            if i >= len(ocr_bboxes):
                continue
            plan.append({
                "name": str(name),
                "erase_regions": list(ocr_bboxes[i]),
                "blocks": [
                    {"text": str(b.get('text', '') or ''), "translated": "",
                     "bbox": normalize_bbox_for_json(
                         b.get('bbox'), image_shape=None, expand_ratio=0)}
                    for b in blocks
                    if normalize_bbox_for_json(
                        b.get('bbox'), image_shape=None, expand_ratio=0)
                ],
            })
        return plan, ocr_bboxes

    cleaned = []
    for i, entry in enumerate(raw_plan):
        image_shape = None
        if i < len(all_ocr_results):
            _, image, _ = all_ocr_results[i]
            image_shape = image.shape
        normalized = _normalize_render_plan_entry(entry, image_shape=image_shape)
        if normalized is None:
            # Repair from OCR blocks when possible (session tolerance).
            if i < len(all_ocr_results):
                name, _, blocks = all_ocr_results[i]
                if i < len(ocr_bboxes):
                    normalized = {
                        "name": str(name),
                        "erase_regions": list(ocr_bboxes[i]),
                        "blocks": [
                            {"text": str(b.get('text', '') or ''), "translated": "",
                             "bbox": normalize_bbox_for_json(
                                 b.get('bbox'), image_shape=image_shape, expand_ratio=0)}
                            for b in blocks
                            if normalize_bbox_for_json(
                                b.get('bbox'), image_shape=image_shape, expand_ratio=0)
                        ],
                    }
        if normalized is not None:
            cleaned.append(normalized)
    return cleaned, ocr_bboxes


@app.route("/postrender/<session_id>")
def postrender_page(session_id):
    """Post-render editor for one image (spec §4.3).

    Falls back to the OCR correction page when the session has no render_plan
    (legacy session → risk R3 fallback) and redirects home on missing session.
    """
    session_data = load_session(session_id)
    if session_data is None:
        return redirect("/")

    if not isinstance(session_data.get("render_plan"), list):
        # Legacy session (never rendered through V2) — fall back to the OCR
        # correction page (spec risk R3).
        return redirect("/correction/" + session_id)

    plan, _ocr_bboxes = _load_render_plan(session_data)
    if not plan:
        return redirect("/correction/" + session_id)

    raw_idx = request.args.get("img", "0")
    try:
        image_idx = int(raw_idx)
    except (TypeError, ValueError):
        image_idx = 0
    if image_idx < 0 or image_idx >= len(plan):
        image_idx = 0

    all_ocr_results = session_data['all_ocr_results']
    if image_idx >= len(all_ocr_results):
        return redirect("/correction/" + session_id)

    name, original_image, _ = all_ocr_results[image_idx]
    rendered_image = _load_rendered_image(session_id, image_idx, original_image)
    if rendered_image is None:
        # Fall back to the original canvas (blocks still editable).
        rendered_image = original_image

    _, buffer = cv2.imencode(".jpg", rendered_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")

    entry = plan[image_idx]
    image_data = {
        "name": str(entry.get("name") or name),
        "data": encoded,
        "blocks": [
            {
                "text": b.get("text", "") or "",
                "translated": b.get("translated", "") or "",
                "bbox": list(b["bbox"]),
                **({"style": b["style"]} if b.get("style") else {}),
            }
            for b in entry.get("blocks", [])
        ],
        "width": int(rendered_image.shape[1]),
        "height": int(rendered_image.shape[0]),
    }
    total_blocks = sum(len(p.get("blocks", [])) for p in plan)
    return render_template(
        "correction.html",
        session_id=session_id,
        images=[image_data],
        total_blocks=total_blocks,
        mode="postrender",
        postrender_image_idx=image_idx,
    )


@app.route("/re-render-image", methods=["POST"])
def rerender_image():
    """Re-render ONE image from its original pixels (spec §4.1).

    Never calls OCR or the translator — only erase + render. Idempotent:
    always starts from the ORIGINAL page_<i>.jpg and overwrites the rendered
    JPEG, so double submits are safe (frontend still locks the button).
    """
    session_id = request.form.get("session_id", "")
    raw_idx = request.form.get("image_idx", "")
    try:
        image_idx = int(raw_idx)
    except (TypeError, ValueError):
        return {"error": "invalid_image_idx"}, 400

    try:
        blocks_json = json.loads(request.form.get("blocks_json", "[]"))
    except (json.JSONDecodeError, TypeError):
        return {"error": "invalid_blocks_json"}, 400
    if isinstance(blocks_json, dict):
        blocks_json = [blocks_json]
    if not isinstance(blocks_json, list):
        return {"error": "invalid_blocks_json"}, 400

    try:
        deleted_regions_json = json.loads(
            request.form.get("deleted_regions_json", "[]")
        )
    except (json.JSONDecodeError, TypeError):
        return {"error": "invalid_deleted_regions_json"}, 400
    if isinstance(deleted_regions_json, dict):
        deleted_regions_json = [deleted_regions_json]
    if not isinstance(deleted_regions_json, list):
        return {"error": "invalid_deleted_regions_json"}, 400

    # V3: erase_regions_json is the canonical erase payload (rect/brush/delete);
    # deleted_regions_json stays as the V2 alias (spec section 4.3).
    try:
        erase_regions_json = json.loads(request.form.get("erase_regions_json", "[]"))
    except (json.JSONDecodeError, TypeError):
        return {"error": "invalid_erase_regions_json"}, 400
    if isinstance(erase_regions_json, dict):
        erase_regions_json = [erase_regions_json]
    if not isinstance(erase_regions_json, list):
        return {"error": "invalid_erase_regions_json"}, 400

    session_data = load_session(session_id)
    if session_data is None:
        return {"error": "session_not_found"}, 404

    all_ocr_results = session_data['all_ocr_results']
    if image_idx < 0 or image_idx >= len(all_ocr_results):
        return {"error": "invalid_image_idx"}, 400

    plan, ocr_bboxes = _load_render_plan(session_data)
    old_entry = plan[image_idx] if image_idx < len(plan) else None

    name, original_image, _ = all_ocr_results[image_idx]
    h, w = original_image.shape[:2]

    default_font = session_data.get('selected_font', 'animeace_')
    old_plan_blocks = (old_entry.get("blocks") or []) if old_entry is not None else []
    blocks = []
    for raw_index, raw_block in enumerate(blocks_json):
        if not isinstance(raw_block, dict):
            continue
        bbox = normalize_bbox_for_json(
            raw_block.get("bbox"), image_shape=(h, w), expand_ratio=0
        )
        if not bbox:
            return {"error": "invalid_bbox"}, 422
        style = normalize_block_style(raw_block.get("style"), default_font)
        if style is None and raw_index < len(old_plan_blocks):
            # A7.8: blocks without style reuse the plan's previous style so
            # V2 clients never lose V3 styling.
            style = normalize_block_style(
                old_plan_blocks[raw_index].get("style"), default_font
            )
        block_entry = {
            "text": str(raw_block.get("text", "") or ""),
            "translated": str(raw_block.get("translated", "") or "").strip(),
            "bbox": bbox,
        }
        if style is not None:
            block_entry["style"] = style
        blocks.append(block_entry)

    deleted_regions = [
        region for region in (
            normalize_bbox_for_json(r, image_shape=(h, w), expand_ratio=0)
            for r in deleted_regions_json
            if isinstance(r, list)
        )
        if region is not None
    ]
    erase_regions = [
        region for region in (
            normalize_bbox_for_json(r, image_shape=(h, w), expand_ratio=0)
            for r in erase_regions_json
            if isinstance(r, list)
        )
        if region is not None
    ]
    # Canonical field wins over the V2 alias when both are provided.
    if request.form.get("erase_regions_json") not in (None, ""):
        deleted_regions = erase_regions

    # V3: optional erase mask (PNG b64, white = erase; spec F7 P1).
    erase_mask = None
    erase_mask_raw = request.form.get("erase_mask", "")
    if erase_mask_raw:
        erase_mask = _decode_erase_mask(erase_mask_raw, (h, w))
        if erase_mask is None:
            return {"error": "invalid_erase_mask"}, 400
    # Monotonic: a request without a mask still applies the accumulated mask
    # from previous renders; a new mask is OR-ed into the stored one BEFORE
    # rendering, so the accumulated set always drives this render.
    if erase_mask is None and old_entry is not None:
        erase_mask = _decode_erase_mask(str(old_entry.get("erase_mask") or ""), (h, w))
    elif erase_mask is not None and old_entry is not None:
        old_mask = _decode_erase_mask(str(old_entry.get("erase_mask") or ""), (h, w))
        if old_mask is not None:
            erase_mask = cv2.bitwise_or(old_mask, erase_mask)

    extra_erase_regions = _entry_erase_regions(
        old_entry,
        ocr_bboxes[image_idx] if image_idx < len(ocr_bboxes) else [],
    ) + deleted_regions

    font_path = get_font_path(default_font)
    source_lang = session_data.get('source_lang', 'ja')
    vision_adapter = build_optional_vision_adapter()

    try:
        rendered_image, new_entry = render_image_with_blocks(
            name, original_image, blocks, font_path, source_lang,
            vision_adapter=vision_adapter,
            extra_erase_regions=extra_erase_regions,
            extra_erase_mask=erase_mask,
        )
    except Exception as exc:
        print(f"Re-render failed for {session_id}/{image_idx}: {exc}")
        return {"error": "render_failed"}, 500

    # erase_regions must NEVER shrink: they cover the ORIGINAL text positions
    # (recorded at the first render) plus deleted blocks. Each re-render starts
    # from the original image, so persisting the accumulated set keeps re-renders
    # idempotent and prevents original text from leaking (spec F5, risk R1).
    if old_entry is not None and old_entry.get("erase_regions"):
        merged_erase_regions = [list(r) for r in old_entry["erase_regions"]]
    else:
        merged_erase_regions = list(
            ocr_bboxes[image_idx] if image_idx < len(ocr_bboxes) else []
        )
    # Accumulate this request's deleted regions so FUTURE re-renders also erase
    # them (server-side guarantee; independent of what the client re-sends).
    for region in deleted_regions:
        if region not in merged_erase_regions:
            merged_erase_regions.append(region)
    new_entry["erase_regions"] = merged_erase_regions

    # Persist the (accumulated) mask so future re-renders stay monotonic
    # (spec F4.3/A4.5): once erased, always erased.
    if erase_mask is not None:
        new_entry["erase_mask"] = base64.b64encode(
            cv2.imencode(".png", erase_mask)[1].tobytes()
        ).decode("utf-8")

    new_plan = []
    for i in range(max(len(plan), len(all_ocr_results))):
        if i == image_idx:
            new_plan.append(new_entry)
        elif i < len(plan):
            new_plan.append(plan[i])
        else:
            # Preserve never-edited images beyond a short plan.
            entry_name, _, entry_blocks = all_ocr_results[i]
            ocr_regions = ocr_bboxes[i] if i < len(ocr_bboxes) else []
            new_plan.append({
                "name": str(entry_name),
                "erase_regions": list(ocr_regions),
                "blocks": [
                    {"text": str(b.get('text', '') or ''),
                     "translated": str(b.get('translated', '') or ''),
                     "bbox": normalize_bbox_for_json(
                         b.get('bbox'), image_shape=None, expand_ratio=0)}
                    for b in entry_blocks
                    if normalize_bbox_for_json(
                        b.get('bbox'), image_shape=None, expand_ratio=0)
                ],
            })

    _persist_render_plan(session_id, session_data, new_plan, {image_idx: rendered_image})

    # V3 A5.10: keep v3_draft in sync with the client state so reloading the
    # style editor preserves styles/edits (and deleted blocks stay deleted).
    try:
        _sync_draft_after_render(session_id, session_data, image_idx, blocks)
    except Exception as exc:
        print(f"[v3_draft sync] failed: {exc}")

    response_blocks = []
    for b in new_entry.get("blocks", []):
        item = {
            "text": b.get("text", "") or "",
            "translated": b.get("translated", "") or "",
            "bbox": list(b["bbox"]),
        }
        if b.get("style"):
            item["style"] = b["style"]
        response_blocks.append(item)
    return {
        "name": name,
        "data": encode_image_jpeg(rendered_image, quality=92),
        "blocks": response_blocks,
    }


@app.route("/translate-result/<session_id>")
def translate_result_page(session_id):
    """Show the latest persisted results page for a correction session.

    Used by the post-render editor's "Về kết quả" / cancel buttons (spec §2.3
    footer). Reads page_*_rendered.jpg when available, else the original.
    """
    session_data = load_session(session_id)
    if session_data is None:
        return redirect("/")

    all_ocr_results = session_data['all_ocr_results']
    processed_results = []
    original_images_by_name = {}
    for idx, (name, original_image, _) in enumerate(all_ocr_results):
        rendered_image = _load_rendered_image(session_id, idx, original_image)
        if rendered_image is None:
            rendered_image = original_image
        processed_results.append({"name": name, "image": rendered_image})
        original_images_by_name[name] = original_image
    processed_images = build_result_images(processed_results, original_images_by_name)
    return render_template(
        "translate.html",
        images=processed_images,
        warning=None,
        correction_session_id=session_id,
    )


@app.route("/re-render-all", methods=["POST"])
def rerender_all():
    """Re-render dirty images and return the results page (spec §4.2).

    Iterates only dirty_indices_json; clean images keep their last rendered
    JPEG. Reuses build_result_images so translate.html stays unchanged.
    """
    session_id = request.form.get("session_id", "")
    try:
        dirty_indices = json.loads(request.form.get("dirty_indices_json", "[]"))
    except (json.JSONDecodeError, TypeError):
        return redirect("/")
    if isinstance(dirty_indices, int):
        dirty_indices = [dirty_indices]
    if not isinstance(dirty_indices, list):
        return redirect("/")

    session_data = load_session(session_id)
    if session_data is None:
        return redirect("/")

    all_ocr_results = session_data['all_ocr_results']
    plan, ocr_bboxes = _load_render_plan(session_data)
    font_path = get_font_path(session_data.get('selected_font', 'animeace_'))
    source_lang = session_data.get('source_lang', 'ja')
    vision_adapter = build_optional_vision_adapter()

    rendered_images = {}
    new_plan = list(plan)
    for raw_idx in dirty_indices:
        try:
            image_idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        if image_idx < 0 or image_idx >= len(all_ocr_results):
            continue
        name, original_image, _ = all_ocr_results[image_idx]
        old_entry = new_plan[image_idx] if image_idx < len(new_plan) else None
        blocks = []
        if old_entry is not None:
            for text, translated, bbox, style in _iter_plan_blocks(old_entry):
                item = {"text": text, "translated": translated, "bbox": bbox}
                if style:
                    item["style"] = style
                blocks.append(item)
        extra_erase_regions = _entry_erase_regions(
            old_entry,
            ocr_bboxes[image_idx] if image_idx < len(ocr_bboxes) else [],
        )
        try:
            rendered_image, new_entry = render_image_with_blocks(
                name, original_image, blocks, font_path, source_lang,
                vision_adapter=vision_adapter,
                extra_erase_regions=extra_erase_regions,
            )
        except Exception as exc:
            print(f"Re-render-all failed for {session_id}/{image_idx}: {exc}")
            continue
        new_entry["erase_regions"] = (
            [list(r) for r in old_entry["erase_regions"]]
            if old_entry is not None and old_entry.get("erase_regions")
            else (ocr_bboxes[image_idx] if image_idx < len(ocr_bboxes) else [])
        )
        while len(new_plan) <= image_idx:
            entry_name, _, entry_blocks = all_ocr_results[len(new_plan)]
            entry_regions = ocr_bboxes[len(new_plan)] if len(new_plan) < len(ocr_bboxes) else []
            new_plan.append({
                "name": str(entry_name),
                "erase_regions": list(entry_regions),
                "blocks": [
                    {"text": str(b.get('text', '') or ''),
                     "translated": str(b.get('translated', '') or ''),
                     "bbox": normalize_bbox_for_json(
                         b.get('bbox'), image_shape=None, expand_ratio=0)}
                    for b in entry_blocks
                    if normalize_bbox_for_json(
                        b.get('bbox'), image_shape=None, expand_ratio=0)
                ],
            })
        new_plan[image_idx] = new_entry
        rendered_images[image_idx] = rendered_image

    _persist_render_plan(session_id, session_data, new_plan, rendered_images)

    processed_results = []
    original_images_by_name = {}
    for idx, (name, original_image, _) in enumerate(all_ocr_results):
        rendered_image = _load_rendered_image(session_id, idx, original_image)
        if rendered_image is None:
            rendered_image = original_image
        processed_results.append({"name": name, "image": rendered_image})
        original_images_by_name[name] = original_image
    processed_images = build_result_images(processed_results, original_images_by_name)

    return render_template(
        "translate.html",
        images=processed_images,
        warning=None,
        correction_session_id=session_id,
    )


@app.route("/download-zip", methods=["POST"])
def download_zip():
    try:
        images_data = request.form.get("images_data", "[]")
        images = json.loads(images_data)
        if not images:
            return redirect("/")
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, img in enumerate(images):
                name = img.get('name', f'image_{i+1}')
                data = img.get('data', '')
                image_bytes = base64.b64decode(data)
                # Images are JPEG-encoded by the pipeline; use a matching extension
                filename = f"{name}_translated.jpg"
                zip_file.writestr(filename, image_bytes)
        
        zip_buffer.seek(0)
        return send_file(zip_buffer, mimetype='application/zip',
                        as_attachment=True, download_name='manga_translated.zip')
    except Exception as e:
        print(f"Error creating ZIP: {e}")
        return redirect("/")


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
