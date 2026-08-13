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
    assess_erasability,
    erase_text_region,
    merge_nearby_ocr_blocks,
    refine_tall_narrow_ocr_bbox,
    render_all_blocks,
    should_skip_ocr_artifact,
)
from ocr.chrome_lens_ocr import ChromeLensOCR
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
    if font_name in ["animeace_", "arial", "mangat"]:
        return f"fonts/{font_name}i.ttf"
    elif font_name.startswith("Yuki-") or font_name.startswith("yuki-"):
        return f"fonts/{font_name}.ttf"
    else:
        return f"fonts/{font_name}.ttf"


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


def translate_and_render(all_ocr_results, translator_obj, selected_font, translator_type,
                         source_lang, target_lang, style):
    """Phase 2+3: Translate all texts then render (used when OCR is already done)"""
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
        return [{'name': name, 'image': image} for name, image, _ in all_ocr_results]
    
    # Phase 2: Batch translate
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

    # Phase 3: Render
    emit_progress('rendering', 0, total_images, 'Đang render text vào ảnh...')
    print(f"\n[Phase 3] Rendering translated text...")

    font_path = get_font_path(selected_font)
    processed_results = []

    # Prepare rendering data for each image
    skipped_count = 0

    def render_single_image(idx_name_image_blocks):
        nonlocal skipped_count
        idx, name, image, blocks = idx_name_image_blocks
        render_blocks = []
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

            # Analyze background and erase original text
            image, text_color, appearance = erase_text_region(
                image, bbox, source_lang=source_lang
            )

            appearance['should_skip'] = False
            render_blocks.append({
                'text': translated,
                'bbox': bbox,
                'text_color': text_color,
                'appearance': appearance,
            })

        if render_blocks:
            image = render_all_blocks(image, render_blocks, font_path)

        return {'name': name, 'image': image}

    # Sequential rendering — ThreadPoolExecutor provides negligible speedup here
    # because PIL ImageDraw operations hold the GIL, and the progress_lock +
    # emit_progress calls serialize most of the parallel work anyway.
    for idx, (name, image, blocks) in enumerate(all_ocr_results):
        emit_progress('rendering', idx + 1, total_images, f'Render: {name}')
        result = render_single_image((idx, name, image, blocks))
        processed_results.append(result)
    print("OK Rendering completed")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} OCR artifact block(s)")
    emit_progress('done', total_images, total_images, f'Hoàn tất! {total_images} ảnh')

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

    processed_results = translate_and_render(
        all_ocr_results, translator_obj, selected_font,
        translator_type=selected_translator,
        source_lang=source_lang, target_lang=target_lang, style=style
    )
    warning = None
    warning = getattr(translator_obj, "last_warning", None)
    gemini_translator = getattr(translator_obj, "_gemini_translator", None)
    if gemini_translator is not None:
        warning = getattr(gemini_translator, "last_warning", None) or warning
    
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
    
    # Rebuild all_ocr_results from modified blocks
    all_ocr_results = session_data['all_ocr_results']
    all_texts = []
    text_index = 0
    
    new_ocr_results = []
    for img_data in modified_blocks:
        img_idx = img_data['image_idx']
        name, original_image, _ = all_ocr_results[img_idx]
        
        blocks = []
        for b in img_data['blocks']:
            text = b.get('text', '').strip()
            bbox = normalize_bbox_for_json(b.get('bbox', None),
                                           image_shape=original_image.shape,
                                           expand_ratio=0)
            if bbox and len(bbox) == 4:
                block = {'text': text, 'bbox': bbox, '_bbox_expanded': True}
                if text:
                    block['_text_idx'] = text_index
                    text_index += 1
                    all_texts.append(text)
                blocks.append(block)
        
        new_ocr_results.append((name, original_image, blocks))
    
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
    blocks = ocr_engine(cropped)

    text = " ".join(b.get("text", "").strip() for b in blocks if b.get("text", "").strip())
    return {"text": text}


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
