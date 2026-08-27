"""
Text rendering for manga/comic translation.
Supports automatic background-aware text coloring and outline rendering.
"""
from dataclasses import asdict

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import math
import os
import re

from vision.region_analysis import _analyze_region_with_diagnostics
from vision.maskers.heuristic import (
    build_text_stroke_mask as _vision_build_text_stroke_mask,
    edge_touching_component_coverage as _vision_edge_touching_component_coverage,
    filter_components_outside_inner as _vision_filter_components_outside_inner,
    filter_text_mask_components as _vision_filter_text_mask_components,
    remove_screentone_dots as _vision_remove_screentone_dots,
)
from vision.pipeline import erase_prepared_block

_font_cache = {}
_font_coverage_cache = {}

MIN_FONT_SIZE = 12
MAX_FONT_SIZE = 60
PADDING_RATIO = 0.12
LINE_SPACING = 1.30  # room for Vietnamese diacritics without shrinking text too much

# Outline configuration
OUTLINE_RATIO_DEFAULT = 0.08  # outline width = 8% of font size
OUTLINE_RATIO_MIN = 0.06
OUTLINE_RATIO_MAX = 0.14
MIN_OUTLINE_WIDTH = 1

# ── SFX detection patterns ──
# Japanese katakana range (including half-width)
_KATAKANA_RE = re.compile(r'[ァ-ヿ･-ﾟ]')
# Hiragana range
_HIRAGANA_RE = re.compile(r'[ぁ-ゟ]')
_HANGUL_TEXT_RE = re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣]')
# SFX indicator: short, mostly katakana, with possible repetition/symbols
_SFX_PATTERNS = [
    # Pure katakana SFX: ドン, ゴゴゴ, バキッ, ザワザワ, ドドドド
    re.compile(r'^[ァ-ヿ･-ﾟー　！？!?…]{1,8}$'),
    # Korean SFX-like: 콰직, 쾅, 두근두근 (short, often with !?)
    re.compile(r'^[ㄱ-ㅎ가-힣!?！？…]{1,6}$'),
    # Chinese onomatopoeia: 啪, 轰隆隆, 哗啦
    re.compile(r'^[一-鿿]{1,4}[!！?？…～]*$'),
]
# Characters that appear in SFX but rarely in normal dialogue
_SFX_SYMBOLS = set('！!？?…～♪☆★◇◆□■△▲▽▼※〆〼')
# Repetition pattern: same char repeated 3+ times (e.g. ドドド, 啊啊啊, gooo)
_REPETITION_RE = re.compile(r'(.)\1{2,}')
# English SFX pattern: ALL CAPS with optional bang symbols
_ENGLISH_SFX_RE = re.compile(r'^[A-Z]{2,6}[!！?？…]*$')
_URL_OR_WATERMARK_RE = re.compile(
    r'(?:https?://|www\.|[\w.-]+\.(?:com|net|org|io|app|vn|jp|kr|cn)\b)',
    re.IGNORECASE,
)
_ASCII_SHORT_RE = re.compile(r'^[A-Za-z0-9]{1,2}[.!?！？，,。:：;；-]*$')

FONT_FALLBACKS = [
    "fonts/ariali.ttf",
    "fonts/arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
]


def _coerce_bbox_for_merge(bbox):
    if not bbox or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _range_overlap_ratio(a1, a2, b1, b2):
    overlap = max(0, min(a2, b2) - max(a1, b1))
    return overlap / float(max(1, min(a2 - a1, b2 - b1)))


def sort_ocr_blocks_reading_order(blocks):
    """Return OCR blocks ordered by visual rows, then left to right."""
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

    positioned.sort(
        key=lambda item: (item["bbox"][1], item["bbox"][0], item["index"])
    )
    rows = []
    for item in positioned:
        best_row = None
        best_score = None
        for row in rows:
            row_half_height = row["average_height"] / 2.0
            row_y1 = row["center_y"] - row_half_height
            row_y2 = row["center_y"] + row_half_height
            overlap = _range_overlap_ratio(
                item["bbox"][1], item["bbox"][3], row_y1, row_y2
            )
            center_delta = abs(item["center_y"] - row["center_y"])
            tolerance = max(
                8.0,
                min(item["height"], row["average_height"]) * 0.50,
            )
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
        best_row["center_y"] = (
            sum(entry["center_y"] for entry in best_row["items"]) / count
        )
        best_row["average_height"] = (
            sum(entry["height"] for entry in best_row["items"]) / count
        )

    rows.sort(key=lambda row: (row["top"], row["center_y"]))
    ordered = []
    for row in rows:
        row["items"].sort(
            key=lambda item: (item["bbox"][0], item["bbox"][1], item["index"])
        )
        ordered.extend(item["block"] for item in row["items"])
    ordered.extend(block for _, block in unpositioned)
    return ordered


def _bbox_gap(a1, a2, b1, b2):
    if a2 < b1:
        return b1 - a2
    if b2 < a1:
        return a1 - b2
    return 0


def _ocr_bboxes_should_merge(a, b):
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    avg_h = (ah + bh) / 2.0

    x_overlap = _range_overlap_ratio(a[0], a[2], b[0], b[2])
    y_overlap = _range_overlap_ratio(a[1], a[3], b[1], b[3])
    vertical_gap = _bbox_gap(a[1], a[3], b[1], b[3])
    horizontal_gap = _bbox_gap(a[0], a[2], b[0], b[2])
    width_ratio = min(aw, bw) / float(max(aw, bw, 1))
    center_delta_x = abs(((a[0] + a[2]) / 2.0) - ((b[0] + b[2]) / 2.0))
    horizontally_aligned = (
        width_ratio >= 0.32
        and center_delta_x <= max(aw, bw) * 0.38
    )

    stacked_lines = (
        vertical_gap <= max(10, min(80, avg_h * 0.55))
        and x_overlap >= 0.45
        and horizontally_aligned
    )
    same_line_fragments = (
        horizontal_gap <= max(8, min(35, avg_h * 0.35))
        and y_overlap >= 0.60
    )
    return stacked_lines or same_line_fragments


def merge_nearby_ocr_blocks(blocks):
    """Merge OCR blocks that are close enough to be one readable text region."""
    normalized = []
    for index, block in enumerate(blocks or []):
        bbox = _coerce_bbox_for_merge(block.get('bbox') if isinstance(block, dict) else None)
        if bbox is None:
            continue
        source_index = block.get('_ocr_index', index)
        normalized.append({
            'index': index,
            'source_index': source_index,
            'block': dict(block),
            'bbox': bbox,
        })

    normalized.sort(key=lambda item: (item['bbox'][1], item['bbox'][0]))
    groups = []
    for item in normalized:
        target = None
        for group in groups:
            if _ocr_bboxes_should_merge(group['bbox'], item['bbox']):
                target = group
                break
        if target is None:
            groups.append({
                'bbox': list(item['bbox']),
                'items': [item],
            })
            continue

        target['bbox'] = [
            min(target['bbox'][0], item['bbox'][0]),
            min(target['bbox'][1], item['bbox'][1]),
            max(target['bbox'][2], item['bbox'][2]),
            max(target['bbox'][3], item['bbox'][3]),
        ]
        target['items'].append(item)

    merged = []
    for group in groups:
        ordered_blocks = sort_ocr_blocks_reading_order(
            [item['block'] for item in group['items']]
        )
        items_by_id = {id(item['block']): item for item in group['items']}
        items = [items_by_id[id(block)] for block in ordered_blocks]
        result = dict(items[0]['block'])
        text_parts = [
            str(item['block'].get('text', '')).strip()
            for item in items
            if str(item['block'].get('text', '')).strip()
        ]
        result['text'] = '\n'.join(text_parts)
        result['bbox'] = [int(v) for v in group['bbox']]
        if len(items) > 1:
            result['_merged_from'] = [item['source_index'] for item in items]
        merged.append(result)

    return sort_ocr_blocks_reading_order(merged)


def _has_hangul_text(text):
    return bool(_HANGUL_TEXT_RE.search(str(text or '')))


def refine_tall_narrow_ocr_bbox(image, bbox, source_lang='ja', text=None):
    """Expand Korean OCR boxes that captured only a vertical slice of text."""
    source_is_korean = str(source_lang or '').lower().startswith('ko')
    if (not source_is_korean and not _has_hangul_text(text)) or image is None:
        return bbox

    parsed = _coerce_bbox_for_merge(bbox)
    if parsed is None:
        return bbox

    x1, y1, x2, y2 = parsed
    raw_w = x2 - x1
    raw_h = y2 - y1
    img_h, img_w = image.shape[:2]
    if raw_h < 50 or raw_h / max(raw_w, 1) < 2.2 or raw_w > img_w * 0.14:
        return parsed

    pad_x = int(max(30, min(raw_h * 0.75, img_w * 0.25)))
    pad_y = int(max(4, raw_h * 0.10))
    sx1 = max(0, x1 - pad_x)
    sy1 = max(0, y1 - pad_y)
    sx2 = min(img_w, x2 + pad_x)
    sy2 = min(img_h, y2 + pad_y)
    roi = image[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return parsed

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = (gray < 95).astype(np.uint8) * 255
    if np.count_nonzero(mask) == 0:
        return parsed

    ox1, oy1 = x1 - sx1, y1 - sy1
    ox2, oy2 = x2 - sx1, y2 - sy1
    original = np.zeros(mask.shape, dtype=bool)
    original[oy1:oy2, ox1:ox2] = True

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    kept = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, num_labels):
        x, y, comp_w, comp_h, area = stats[label]
        if area < 10 or comp_h < 8:
            continue
        touches_search_edge = (
            x <= 1 or y <= 1
            or x + comp_w >= mask.shape[1] - 1
            or y + comp_h >= mask.shape[0] - 1
        )
        if touches_search_edge:
            continue
        if comp_w >= mask.shape[1] * 0.80 or comp_h >= mask.shape[0] * 0.80:
            continue

        vertical_overlap = (
            max(0, min(y + comp_h, oy2) - max(y, oy1))
            / float(max(1, min(comp_h, oy2 - oy1)))
        )
        comp = labels == label
        overlap = np.count_nonzero(comp & original) / float(max(area, 1))
        if vertical_overlap > 0.35 or overlap > 0.05:
            kept[comp] = 255

    ys, xs = np.where(kept > 0)
    if len(xs) == 0:
        return parsed

    refined = [
        max(0, int(xs.min() + sx1 - 3)),
        max(0, int(ys.min() + sy1 - 3)),
        min(img_w, int(xs.max() + sx1 + 4)),
        min(img_h, int(ys.max() + sy1 + 4)),
    ]
    if refined[2] - refined[0] <= raw_w * 1.25:
        return parsed
    return refined


def get_cached_font(font_path, size):
    cache_key = (font_path, size)
    if cache_key not in _font_cache:
        try:
            _font_cache[cache_key] = ImageFont.truetype(font_path, size=size)
        except:
            _font_cache[cache_key] = ImageFont.load_default()
    return _font_cache[cache_key]


def _font_codepoints(font_path):
    if font_path in _font_coverage_cache:
        return _font_coverage_cache[font_path]

    codepoints = None
    try:
        from fontTools.ttLib import TTCollection, TTFont

        fonts = TTCollection(font_path).fonts if font_path.lower().endswith((".ttc", ".otc")) else [TTFont(font_path, lazy=True)]
        codepoints = set()
        for font in fonts:
            for table in font["cmap"].tables:
                codepoints.update(table.cmap.keys())
            try:
                font.close()
            except Exception:
                pass
    except Exception:
        codepoints = None

    _font_coverage_cache[font_path] = codepoints
    return codepoints


def _font_supports_text(font_path, text):
    codepoints = _font_codepoints(font_path)
    if codepoints is None:
        return True

    for ch in text:
        if ch.isspace() or ord(ch) in (0x200b, 0x200c, 0x200d, 0xfeff):
            continue
        if ord(ch) not in codepoints:
            return False
    return True


def resolve_font_path_for_text(font_path, text):
    candidates = [font_path]
    candidates.extend(p for p in FONT_FALLBACKS if p not in candidates)

    existing = [p for p in candidates if p and os.path.exists(p)]
    for candidate in existing:
        if _font_supports_text(candidate, text):
            return candidate

    return existing[0] if existing else font_path


def should_skip_ocr_artifact(text, bbox=None, image_shape=None, source_lang='ja'):
    """Return True for OCR blocks that are likely decoration, credits, or watermarks."""
    if not text or not text.strip():
        return True

    normalized = re.sub(r'\s+', ' ', text.strip())
    compact = re.sub(r'\s+', '', normalized)

    if _URL_OR_WATERMARK_RE.search(normalized):
        return True

    # On CJK/Korean pages, isolated Latin letters/numbers are usually signs,
    # decorative marks, or OCR noise rather than dialogue bubbles.
    if source_lang in {'ja', 'zh', 'ko'} and _ASCII_SHORT_RE.match(compact):
        return True

    if bbox and image_shape is not None and len(bbox) >= 4:
        try:
            img_h, img_w = image_shape[:2]
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        except (TypeError, ValueError):
            return False

        block_w = max(0, x2 - x1)
        block_h = max(0, y2 - y1)
        near_edge = x1 <= 4 or y1 <= 4 or x2 >= img_w - 4 or y2 >= img_h - 4
        small_edge_credit = (
            near_edge
            and len(compact) <= 32
            and (block_w <= img_w * 0.18 or block_h <= img_h * 0.035)
            and sum(ch.isascii() for ch in compact) / max(len(compact), 1) > 0.7
        )
        if small_edge_credit:
            return True

    return False


def _detect_sfx(text, source_lang='ja'):
    """
    Detect if text is likely a sound effect (SFX) rather than dialogue/narration.

    Returns dict: {is_sfx: bool, sfx_type: str, confidence: float, should_skip: bool}
    """
    if not text or not text.strip():
        return {'is_sfx': False, 'sfx_type': 'none', 'confidence': 0.0, 'should_skip': False}

    t = text.strip()
    t_len = len(t)
    score = 0.0

    # ── Length heuristic: SFX tend to be very short ──
    if t_len <= 3:
        score += 0.20
    elif t_len <= 5:
        score += 0.10
    elif t_len > 15:
        score -= 0.30  # long text is almost never SFX

    # ── Language-specific checks ──
    kata_ratio = 0.0
    hira_count = 0
    all_hira = False
    has_repetition = bool(_REPETITION_RE.search(t))  # compute early, used everywhere

    if source_lang == 'ja':
        # Count katakana ratio
        kata_count = len(_KATAKANA_RE.findall(t))
        hira_count = len(_HIRAGANA_RE.findall(t))

        if t_len > 0:
            kata_ratio = kata_count / t_len
            hira_ratio = hira_count / t_len

            if kata_ratio >= 0.8:
                score += 0.35
            elif kata_ratio >= 0.5:
                score += 0.15

            # Pure hiragana = NOT SFX (is dialogue), unless with repetition
            all_hira = hira_ratio >= 1.0 and t_len > 1

        # Match SFX patterns (katakana-heavy)
        for pattern in _SFX_PATTERNS:
            if pattern.match(t):
                score += 0.25
                break

    elif source_lang == 'ko':
        # Korean: check SFX patterns
        for pattern in _SFX_PATTERNS:
            if pattern.match(t):
                score += 0.25
                break

        # Korean greeting/standard phrases penalty
        _korean_common = {'안녕하세요', '감사합니다', '사랑해요', '미안합니다', '잘먹겠습니다'}
        if t in _korean_common:
            score -= 0.35

    elif source_lang == 'zh':
        for pattern in _SFX_PATTERNS:
            if pattern.match(t):
                score += 0.25
                break

    # ── English SFX (BOOM, WHAM, BANG, etc.) ──
    if _ENGLISH_SFX_RE.match(t):
        score += 0.40  # very strong signal: all caps short word
    elif t.isupper() and t_len <= 6 and t_len >= 2:
        score += 0.25  # all caps but maybe has spaces/punctuation
    # English word with repetitive letter (gooo, zzzz)
    if source_lang in ('en', 'ja') and has_repetition and t_len <= 8:
        score += 0.10  # extra for repeated chars in short text

    # ── Symbol density ──
    symbol_count = sum(1 for ch in t if ch in _SFX_SYMBOLS)
    if symbol_count >= 1:
        score += 0.10
    if symbol_count >= 2:
        score += 0.10

    # ── Repetition (ドドドド, ガガガ, あああ, gooo) ──
    if has_repetition:
        score += 0.30  # strong SFX signal regardless of script

    # ── All-caps HANGUL (Korean SFX) ──
    if source_lang == 'ko' and t_len <= 5:
        if all('가' <= ch <= '힣' or ch in '!?！？' for ch in t):
            score += 0.20

    # ── Hiragana-only penalty (pure hiragana is dialogue, not SFX) ──
    if all_hira and not has_repetition:
        score -= 0.25

    # ── Decision ──
    is_sfx = score >= 0.45

    # Determine SFX sub-type
    if is_sfx:
        if kata_ratio >= 0.7 if source_lang == 'ja' else False:
            sfx_type = 'katakana_sfx'
        elif symbol_count >= 1:
            sfx_type = 'symbolic_sfx'
        elif _REPETITION_RE.search(t):
            sfx_type = 'repeated_sfx'
        elif t_len <= 3:
            sfx_type = 'short_sfx'
        else:
            sfx_type = 'likely_sfx'
    else:
        sfx_type = 'dialogue'

    return {
        'is_sfx': is_sfx,
        'sfx_type': sfx_type,
        'confidence': score,
        'should_skip': False,  # Will be determined later based on context
    }


def _decide_skip_render(text, sfx_info, analysis, source_lang='ja'):
    """
    Decide whether to SKIP rendering entirely for this text block.
    SFX-specific filtering is disabled; background safety is handled earlier
    by assess_erasability().

    Returns True if this block should be skipped.
    """
    return False


def _analyze_region(image, bbox):
    """Return legacy dict analysis backed by the shared typed implementation."""
    analysis, diagnostics = _analyze_region_with_diagnostics(image, tuple(bbox))
    if analysis is None or diagnostics is None:
        return None

    legacy = asdict(analysis)
    legacy["mean_bgr"] = diagnostics.mean_bgr
    legacy["border_interior_contrast"] = diagnostics.border_interior_contrast
    legacy["is_bubble"] = diagnostics.is_bubble
    return legacy



def _decide_text_appearance(analysis):
    """
    Decide fill color, text color, outline for optimal readability.
    Uses bubble_context to make fundamentally different decisions
    for in-bubble vs on-artwork text.

    Speech bubble (in_bubble):
      - Fill with the detected interior color (actual bubble fill)
      - Text: black on light bubbles, white on dark bubbles
      - Outline: NEVER for light bubbles, OPTIONAL thin for dark bubbles

    On artwork (on_artwork_*):
      - Fill with detected average background (blends in)
      - Text: maximum contrast color
      - Outline: ALWAYS needed for readability against artwork
      - Dark artwork → white text + dark outline
      - Light artwork → black text + light outline
      - Mixed artwork → white text + strong outline
    """
    if analysis is None:
        return {'fill_color': (0, 0, 0), 'text_color': (255, 255, 255),
                'need_outline': False, 'outline_color': (0, 0, 0),
                'outline_width': 0, 'bubble_context': 'on_artwork_dark'}

    bubble_context = analysis.get('bubble_context', 'on_artwork_mixed')
    tone = analysis['dominant_tone']
    intensity = analysis['mean_intensity']
    intensity_std = analysis['intensity_std']
    edge = analysis['edge_score']
    uniformity = analysis.get('uniformity', 'complex')
    mean_bgr = analysis['mean_bgr']

    # Fill: always use detected interior color (bubble fill or artwork bg)
    fill_bgr = tuple(int(round(c)) for c in mean_bgr)

    # --- Defaults ---
    text_color = (255, 255, 255)  # BGR white
    need_outline = False
    outline_color = (0, 0, 0)     # BGR black
    outline_ratio = OUTLINE_RATIO_DEFAULT

    if bubble_context == 'in_bubble':
        # ──────────────────────────────────────────────
        # IN BUBBLE: clean fill, natural text color
        # ──────────────────────────────────────────────
        if tone == 'dark':
            # Dark bubble (rare: black-bordered with dark fill)
            text_color = (255, 255, 255)  # white text
            need_outline = False
        elif tone == 'light':
            # Standard manga bubble: white/light fill
            text_color = (0, 0, 0)  # black text
            need_outline = False
        else:
            # Mid-tone bubble (colored bubble)
            text_color = (0, 0, 0) if intensity > 128 else (255, 255, 255)
            need_outline = False  # interior is uniform, no outline needed

    elif bubble_context == 'on_artwork_dark':
        # ──────────────────────────────────────────────
        # ARTWORK DARK: text over dark/black art panels
        # ──────────────────────────────────────────────
        text_color = (255, 255, 255)  # white text for dark bg
        need_outline = True
        outline_ratio = OUTLINE_RATIO_MIN  # thin outline enough on dark
        # Dark outline blends with dark bg, doesn't distract
        outline_color = (30, 30, 30)  # near-black

    elif bubble_context == 'on_artwork_light':
        # ──────────────────────────────────────────────
        # ARTWORK LIGHT: text over bright/sky/white art
        # ──────────────────────────────────────────────
        text_color = (0, 0, 0)  # black text for light bg
        # Only outline if background is actually complex (not uniform white)
        if uniformity == 'uniform':
            need_outline = False
        elif uniformity == 'textured' and intensity_std > 20:
            need_outline = True
            outline_ratio = OUTLINE_RATIO_MIN
            outline_color = (210, 210, 210)
        else:
            need_outline = False

    else:  # on_artwork_mixed
        # ──────────────────────────────────────────────
        # ARTWORK MIXED: text over detailed/drawn panels
        # This is the hardest case — needs strong outline
        # ──────────────────────────────────────────────
        # Choose text color for maximum contrast
        if intensity < 110:
            text_color = (255, 255, 255)  # white
            outline_color = (0, 0, 0)     # black outline
        else:
            text_color = (0, 0, 0)        # black
            outline_color = (255, 255, 255)  # white outline

        need_outline = True

        # Outline thickness scales with background complexity
        if intensity_std > 60 or edge > 60:
            outline_ratio = OUTLINE_RATIO_MAX  # very complex → thick outline
        elif intensity_std > 35:
            outline_ratio = OUTLINE_RATIO_DEFAULT  # moderate
        else:
            outline_ratio = OUTLINE_RATIO_MIN  # lighter

    return {
        'fill_color': fill_bgr,
        'text_color': text_color,
        'need_outline': need_outline,
        'outline_color': outline_color,
        'outline_ratio': outline_ratio,
        'bubble_context': bubble_context,
        'uniformity': uniformity,
        'intensity_std': float(intensity_std),
        'edge_score': float(edge),
    }


def _luminance_bgr(color):
    return float(0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2])


def _filter_text_mask_components(mask):
    return _vision_filter_text_mask_components(mask)


def _build_text_stroke_mask(roi, fill_color, appearance):
    return _vision_build_text_stroke_mask(roi, fill_color, appearance)


def _filter_components_outside_inner(mask, inner_rect, min_overlap=0.30):
    return _vision_filter_components_outside_inner(mask, inner_rect, min_overlap)


def _remove_screentone_dots(mask, tiny_area=30, min_count=40, min_ratio=0.6):
    return _vision_remove_screentone_dots(mask, tiny_area, min_count, min_ratio)


def _edge_touching_component_coverage(mask):
    return _vision_edge_touching_component_coverage(mask)



def _erase_strokes_only(image, x1, y1, x2, y2, fill_color, appearance, inner_rect=None):
    roi = image[y1:y2, x1:x2]
    mask = _build_text_stroke_mask(roi, fill_color, appearance)
    if mask is None or np.count_nonzero(mask) == 0:
        return 'no-text-mask', 0.0

    # Drop screentone/halftone dots so artwork texture is not painted over.
    mask = _remove_screentone_dots(mask)
    if np.count_nonzero(mask) == 0:
        return 'no-text-mask', 0.0

    # When erasing a padded region, keep only components anchored in the
    # original bbox so bubble borders crossing the padding are preserved.
    if inner_rect is not None:
        mask = _filter_components_outside_inner(mask, inner_rect)
        if np.count_nonzero(mask) == 0:
            return 'no-text-mask', 0.0

    coverage = np.count_nonzero(mask) / max(mask.size, 1)
    bg_pixels = roi[mask == 0]
    sampled_fill = fill_color
    bg_texture_std = 0.0
    if len(bg_pixels) >= 20:
        gray_bg = (
            0.114 * bg_pixels[:, 0].astype(np.float32)
            + 0.587 * bg_pixels[:, 1].astype(np.float32)
            + 0.299 * bg_pixels[:, 2].astype(np.float32)
        )
        bg_texture_std = float(gray_bg.std())
        bright_bg = bg_pixels[gray_bg > 165]
        source_pixels = bright_bg if len(bright_bg) >= 20 else bg_pixels
        sampled_fill = tuple(int(v) for v in np.median(source_pixels, axis=0))

    bubble_context = appearance.get('bubble_context', 'on_artwork_mixed')
    sampled_luma = _luminance_bgr(sampled_fill)
    # Flat fill destroys textured backgrounds (screentone/halftone dots);
    # only use it when the surrounding background is genuinely uniform.
    textured_bg = bg_texture_std > 16.0
    appearance['bg_texture_std'] = bg_texture_std
    should_flat_fill = (
        bubble_context in {'in_bubble', 'on_artwork_light'}
        and sampled_luma > 150
        and not textured_bg
    )

    if should_flat_fill:
        alpha = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
        alpha = alpha[:, :, None]
        fill = np.array(sampled_fill, dtype=np.float32).reshape(1, 1, 3)
        roi[:, :, :] = np.clip(
            roi.astype(np.float32) * (1.0 - alpha) + fill * alpha,
            0, 255
        ).astype(np.uint8)
        if (
            inner_rect is not None
            and appearance.get('uniformity') == 'uniform'
            and coverage > 0.45
            and bg_texture_std < 8.0
        ):
            ix1, iy1, ix2, iy2 = inner_rect
            sub_roi = roi[iy1:iy2, ix1:ix2]
            if sub_roi.size:
                fill_gray = int(round(_luminance_bgr(sampled_fill)))
                sub_gray = cv2.cvtColor(sub_roi, cv2.COLOR_BGR2GRAY).astype(np.int16)
                delta = np.abs(sub_gray - fill_gray)
                residual = (delta > 2) & (delta < 24)
                sub_roi[residual] = sampled_fill
        appearance['sampled_fill_color'] = sampled_fill
        return 'stroke-fill-sampled', coverage

    use_local_inpaint = coverage < 0.70

    if use_local_inpaint:
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask
        restored = cv2.inpaint(image, full_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        image[full_mask > 0] = restored[full_mask > 0]
        appearance['sampled_fill_color'] = sampled_fill
        return 'stroke-inpaint', coverage

    alpha = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
    alpha = alpha[:, :, None]
    fill = np.array(sampled_fill, dtype=np.float32).reshape(1, 1, 3)
    roi[:, :, :] = np.clip(roi.astype(np.float32) * (1.0 - alpha) + fill * alpha, 0, 255).astype(np.uint8)
    appearance['sampled_fill_color'] = sampled_fill
    return 'stroke-fill', coverage


def assess_erasability(image, bbox, text=None, source_lang='ja', prepared=None):
    """Return whether an OCR block is safe to erase without mutating image."""
    if prepared is not None:
        decision = prepared.decision
        region = prepared.region
        mask_result = prepared.mask_result
        return {
            'safe': decision.safe,
            'reason': decision.reason,
            'score': decision.score,
            'analysis': {
                'bubble_context': region.bubble_context,
                'uniformity': region.uniformity,
                'intensity_std': region.intensity_std,
                'edge_score': region.edge_score,
                'mask_coverage': mask_result.coverage,
                'raw_mask_coverage': mask_result.coverage,
                'edge_component_coverage': mask_result.edge_touch_ratio,
                'bg_texture_std': region.texture_std,
                'fill_luma': region.mean_intensity,
                'is_sfx': False,
            },
        }

    if image is None or bbox is None or len(bbox) < 4:
        return {
            'safe': False,
            'reason': 'invalid_bbox',
            'score': 0.0,
            'analysis': {},
        }

    h_img, w_img = image.shape[:2]
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    except (TypeError, ValueError):
        return {
            'safe': False,
            'reason': 'invalid_bbox',
            'score': 0.0,
            'analysis': {},
        }

    x1 = max(0, min(w_img, x1))
    y1 = max(0, min(h_img, y1))
    x2 = max(0, min(w_img, x2))
    y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return {
            'safe': False,
            'reason': 'invalid_bbox',
            'score': 0.0,
            'analysis': {},
        }

    analysis = _analyze_region(image, [x1, y1, x2, y2])
    appearance = _decide_text_appearance(analysis)
    fill_color = appearance.get('fill_color', (0, 0, 0))
    bubble_context = appearance.get('bubble_context', 'on_artwork_mixed')
    uniformity = appearance.get('uniformity', 'complex')
    intensity_std = float(appearance.get('intensity_std', 0) or 0)
    edge_score = float(appearance.get('edge_score', 0) or 0)

    pad_x = min(8, max(2, int((x2 - x1) * 0.02)))
    pad_y = min(8, max(2, int((y2 - y1) * 0.06)))
    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(w_img, x2 + pad_x)
    y2_pad = min(h_img, y2 + pad_y)
    inner_rect = (x1 - x1_pad, y1 - y1_pad, x2 - x1_pad, y2 - y1_pad)

    roi = image[y1_pad:y2_pad, x1_pad:x2_pad]
    raw_mask = _build_text_stroke_mask(roi, fill_color, appearance)
    raw_coverage = (
        np.count_nonzero(raw_mask) / float(max(raw_mask.size, 1))
        if raw_mask is not None else 0.0
    )
    edge_component_coverage = _edge_touching_component_coverage(raw_mask)

    mask = raw_mask
    if mask is not None and np.count_nonzero(mask) > 0:
        mask = _remove_screentone_dots(mask)
    if mask is not None and np.count_nonzero(mask) > 0:
        mask = _filter_components_outside_inner(mask, inner_rect)

    mask_coverage = (
        np.count_nonzero(mask) / float(max(mask.size, 1))
        if mask is not None else 0.0
    )
    if mask is not None and roi.size:
        bg_pixels = roi[mask == 0]
    else:
        bg_pixels = roi.reshape(-1, 3) if roi.size else np.empty((0, 3), dtype=np.uint8)
    if len(bg_pixels) >= 20:
        bg_gray = (
            0.114 * bg_pixels[:, 0].astype(np.float32)
            + 0.587 * bg_pixels[:, 1].astype(np.float32)
            + 0.299 * bg_pixels[:, 2].astype(np.float32)
        )
        bg_texture_std = float(bg_gray.std())
    else:
        bg_texture_std = float(intensity_std)

    sfx_info = _detect_sfx(text or '', source_lang)
    risky_texture = (
        bg_texture_std > 65
        or (bg_texture_std > 42 and edge_component_coverage > 0.10)
        or (bg_texture_std > 55 and uniformity != 'uniform')
    )
    fill_luma = _luminance_bgr(fill_color)
    bubble_text_like_mask = (
        bubble_context == 'in_bubble'
        and fill_luma > 135
        and 0.01 <= mask_coverage <= 0.60
        and raw_coverage <= 0.70
    )
    safe_in_bubble_text = (
        bubble_text_like_mask
        and uniformity == 'uniform'
        and fill_luma > 235
        and intensity_std < 8
    )
    flat_light_uniform_text = (
        bubble_context in {'on_artwork_light', 'in_bubble'}
        and uniformity == 'uniform'
        and fill_luma > 150
        and intensity_std < 8
        and 0.01 <= mask_coverage <= 0.70
        and raw_coverage <= 0.70
        and (bubble_context != 'in_bubble' or fill_luma > 235 or bg_texture_std < 42)
    )
    dark_complex_bubble_like_artwork = (
        bubble_context == 'in_bubble'
        and fill_luma < 135
        and uniformity == 'complex'
        and edge_component_coverage > 0.10
    )
    score = 0.0
    if bubble_context == 'in_bubble':
        score += 0.45
    if uniformity == 'uniform':
        score += 0.30
    elif uniformity == 'textured' and intensity_std < 28:
        score += 0.15
    if intensity_std < 12:
        score += 0.15
    if bg_texture_std < 28:
        score += 0.15
    elif bg_texture_std < 42:
        score += 0.08
    if bg_texture_std < 30 and bubble_context in {'on_artwork_light', 'on_artwork_dark'}:
        score += 0.30
    if edge_score < 25:
        score += 0.15
    elif (
        edge_score > 55
        and bubble_context != 'in_bubble'
        and uniformity != 'uniform'
        and bg_texture_std > 35
    ):
        score -= 0.35
    if 0.01 <= mask_coverage <= 0.55 or flat_light_uniform_text:
        score += 0.15
    if (mask_coverage > 0.62 or raw_coverage > 0.70) and not flat_light_uniform_text:
        score -= 0.35
    if edge_component_coverage > 0.28 and bubble_context != 'in_bubble':
        score -= 0.20
    if (mask_coverage > 0.62 or raw_coverage > 0.70) and not flat_light_uniform_text:
        reason = 'excessive_mask'
        safe = False
        score = min(score, 0.25)
    elif dark_complex_bubble_like_artwork:
        reason = 'complex_artwork'
        safe = False
        score = min(score, 0.30)
    elif (
        risky_texture
        and not safe_in_bubble_text
        and not flat_light_uniform_text
        and (bubble_context != 'in_bubble' or edge_component_coverage > 0.10)
    ):
        reason = 'complex_artwork'
        safe = False
        score = min(score, 0.30)
    elif (
        bubble_context != 'in_bubble'
        and uniformity == 'complex'
        and (bg_texture_std > 35 or intensity_std > 40)
    ):
        reason = 'complex_artwork'
        safe = False
        score = min(score, 0.35)
    elif mask_coverage <= 0:
        reason = 'no_text_mask'
        safe = False
        score = min(score, 0.0)
    else:
        safe = score >= 0.55
        if bubble_context == 'in_bubble':
            reason = 'in_bubble'
        elif uniformity == 'uniform':
            reason = 'uniform_background'
        elif safe:
            reason = 'text_like_mask'
        else:
            reason = 'risky_background'

    score = max(0.0, min(1.0, float(score)))
    return {
        'safe': bool(safe),
        'reason': reason,
        'score': score,
        'analysis': {
            'bubble_context': bubble_context,
            'uniformity': uniformity,
            'intensity_std': intensity_std,
            'edge_score': edge_score,
            'mask_coverage': float(mask_coverage),
            'raw_mask_coverage': float(raw_coverage),
            'edge_component_coverage': float(edge_component_coverage),
            'bg_texture_std': float(bg_texture_std),
            'fill_luma': float(fill_luma),
            'is_sfx': bool(sfx_info.get('is_sfx', False)),
        },
    }


def appearance_for_prepared(prepared):
    """Build renderer appearance metadata without changing image pixels."""
    appearance = _decide_text_appearance(asdict(prepared.region))
    method_names = {
        'preserve': 'no-text-mask',
        'flat': 'stroke-fill-sampled',
        'telea': 'stroke-inpaint',
        'lama_full_page': 'stroke-inpaint',
    }
    appearance['erase_method'] = method_names[prepared.erase_method]
    appearance['erase_mask_coverage'] = prepared.mask_result.coverage
    appearance['should_skip'] = bool(
        prepared.decision.requires_review or not prepared.decision.safe
    )
    text_bgr = appearance['text_color']
    appearance['text_color'] = (text_bgr[2], text_bgr[1], text_bgr[0])
    return appearance


def erase_text_region(image, bbox, source_lang='ja', prepared=None):
    """
    Analyze surrounding background and fill the bbox area with appropriate color.
    Uses inpainting for complex backgrounds, flat fill for uniform ones.

    Args:
        image: numpy array (BGR)
        bbox: [x1, y1, x2, y2]
        source_lang: Source language code for SFX detection

    Returns:
        tuple: (image, text_color_rgb, appearance_info)
    """
    if prepared is not None:
        appearance = appearance_for_prepared(prepared)
        erase_result = erase_prepared_block(image, prepared)
        if erase_result.warning:
            appearance['erase_warning'] = erase_result.warning
        return image, appearance['text_color'], appearance

    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    h_img, w_img = image.shape[:2]
    x2 = min(x2, w_img)
    y2 = min(y2, h_img)

    if x2 <= x1 or y2 <= y1:
        return image, (0, 0, 0), {'fill_color': (0, 0, 0), 'text_color': (0, 0, 0),
                                   'need_outline': False, 'outline_color': (0, 0, 0),
                                   'outline_ratio': 0, 'should_skip': False,
                                   'bubble_context': 'unknown'}

    # Analyze the background around the bbox
    analysis = _analyze_region(image, bbox)
    appearance = _decide_text_appearance(analysis)

    fill_color = appearance['fill_color']
    uniformity = analysis.get('uniformity', 'complex') if analysis else 'complex'
    intensity_std = analysis.get('intensity_std', 0) if analysis else 0
    bubble_context = appearance.get('bubble_context', 'on_artwork_mixed')
    mean_intensity = analysis.get('mean_intensity', 0) if analysis else 0

    # Pad the erase region slightly to catch ascenders/descenders and JPEG
    # halos that overflow the OCR bbox. Components living mostly in the
    # padding strip (e.g. bubble borders) are filtered out downstream, so
    # borders remain intact.
    pad_x = min(8, max(2, int((x2 - x1) * 0.02)))
    pad_y = min(8, max(2, int((y2 - y1) * 0.06)))
    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(w_img, x2 + pad_x)
    y2_pad = min(h_img, y2 + pad_y)

    inner_rect = (x1 - x1_pad, y1 - y1_pad, x2 - x1_pad, y2 - y1_pad)
    erase_method, mask_coverage = _erase_strokes_only(
        image, x1_pad, y1_pad, x2_pad, y2_pad, fill_color, appearance,
        inner_rect=inner_rect
    )
    appearance['erase_method'] = erase_method
    appearance['erase_mask_coverage'] = mask_coverage

    # text_color is RGB (PIL format) — convert from BGR
    text_bgr = appearance['text_color']
    text_color_rgb = (text_bgr[2], text_bgr[1], text_bgr[0])

    # Debug log
    ctx = appearance.get('bubble_context', '?')
    outline_tag = '+outline' if appearance.get('need_outline') else ''
    skip_tag = ' [SKIP]' if appearance.get('should_skip') else ''
    sampled = appearance.get('sampled_fill_color', fill_color)
    print(f"    [bg: {ctx} {outline_tag}{skip_tag}] erase={erase_method} mask={mask_coverage:.2f} fill=BGR({sampled[0]},{sampled[1]},{sampled[2]}) text=RGB{text_color_rgb}")

    return image, text_color_rgb, appearance


def _compute_font_and_wrap(text, bbox, font_path):
    """Compute optimal font size and text wrapping for a given bbox.
    Uses pixel-accurate wrapping (measures actual rendered width per word/char).
    Returns (font, wrapped_text, line_height) or None if nothing fits."""
    if not text or not text.strip():
        return None
    text = re.sub(r'\s*\n\s*', ' ', text.strip())
    text = re.sub(r'[ \t]{2,}', ' ', text)

    x1, y1, x2, y2 = [int(v) for v in bbox]
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    render_font_path = resolve_font_path_for_text(font_path, text)

    usable_w = int(w * (1 - 2 * PADDING_RATIO))
    usable_h = int(h * (1 - 2 * PADDING_RATIO))
    if usable_w <= 0 or usable_h <= 0:
        usable_w, usable_h = w, h

    # Detect if text is CJK (no spaces between words) or Latin (space-separated)
    has_cjk = any('\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff' or
                  '\uac00' <= ch <= '\ud7af' for ch in text)
    has_spaces = ' ' in text
    use_char_wrap = has_cjk and not has_spaces  # CJK: wrap per character

    bubble_area = usable_w * usable_h
    char_count = max(len(text), 1)
    estimated_size = int(math.sqrt(bubble_area / (char_count * 0.8)))
    font_size_guess = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, estimated_size))

    best_font_size = 0  # 0 means no fit found yet
    best_wrapped_lines = None

    # Binary search for optimal font size
    lo, hi = MIN_FONT_SIZE, font_size_guess
    while lo <= hi:
        mid = (lo + hi) // 2
        font = get_cached_font(render_font_path, mid)
        line_height = int(mid * LINE_SPACING)

        # --- Pixel-based wrapping ---
        lines = []
        if use_char_wrap:
            # CJK: wrap character by character
            current_line = ""
            for ch in text:
                if ch == '\n':
                    if current_line:
                        lines.append(current_line)
                    current_line = ""
                    continue
                test_line = current_line + ch
                try:
                    test_w = font.getlength(test_line)
                except Exception:
                    test_w = len(test_line) * mid * 0.6
                if test_w > usable_w and current_line:
                    lines.append(current_line)
                    current_line = ch
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)
        else:
            # Latin / space-separated: wrap by words
            words = text.split(' ')
            current_line = ""
            for word in words:
                if not word:
                    continue
                sep = " " if current_line else ""
                test_line = current_line + sep + word
                try:
                    test_w = font.getlength(test_line)
                except Exception:
                    test_w = len(test_line) * mid * 0.6
                if test_w > usable_w:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                    # If single word is wider than usable_w, force character-level break
                    try:
                        if font.getlength(word) > usable_w:
                            # Break word into chunks
                            char_lines = []
                            chunk = ""
                            for ch in word:
                                test_chunk = chunk + ch
                                if font.getlength(test_chunk) > usable_w and chunk:
                                    char_lines.append(chunk)
                                    chunk = ch
                                else:
                                    chunk = test_chunk
                            if chunk:
                                char_lines.append(chunk)
                            # Replace current_line with first chunk, append rest
                            lines.extend(char_lines[:-1])
                            if len(char_lines) == 1 and not lines:
                                lines.append(char_lines[0])
                                current_line = ""
                            else:
                                current_line = char_lines[-1] if char_lines else ""
                    except Exception:
                        pass
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)

        if not lines:
            lines = [text]

        # Check if all lines fit horizontally
        all_fit = True
        for line in lines:
            try:
                if font.getlength(line) > usable_w:
                    all_fit = False
                    break
            except Exception:
                if len(line) * mid * 0.6 > usable_w:
                    all_fit = False
                    break

        total_height = len(lines) * line_height

        if total_height <= usable_h and all_fit:
            best_font_size = mid
            best_wrapped_lines = lines
            lo = mid + 1  # try larger
        else:
            hi = mid - 1  # try smaller

    # Fallback: if no font size fit, use MIN_FONT_SIZE but still wrap properly
    if best_font_size == 0:
        best_font_size = MIN_FONT_SIZE
        font = get_cached_font(render_font_path, best_font_size)

        # Wrap at minimum size
        if use_char_wrap:
            best_wrapped_lines = []
            current_line = ""
            for ch in text:
                if ch == '\n':
                    if current_line:
                        best_wrapped_lines.append(current_line)
                    current_line = ""
                    continue
                test_line = current_line + ch
                try:
                    test_w = font.getlength(test_line)
                except Exception:
                    test_w = len(test_line) * best_font_size * 0.6
                if test_w > usable_w and current_line:
                    best_wrapped_lines.append(current_line)
                    current_line = ch
                else:
                    current_line = test_line
            if current_line:
                best_wrapped_lines.append(current_line)
        else:
            words = text.split(' ')
            best_wrapped_lines = []
            current_line = ""
            for word in words:
                if not word:
                    continue
                sep = " " if current_line else ""
                test_line = current_line + sep + word
                try:
                    test_w = font.getlength(test_line)
                except Exception:
                    test_w = len(test_line) * best_font_size * 0.6
                if test_w > usable_w:
                    if current_line:
                        best_wrapped_lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                best_wrapped_lines.append(current_line)

        if not best_wrapped_lines:
            best_wrapped_lines = [text]

    font = get_cached_font(render_font_path, best_font_size)
    line_height = int(best_font_size * LINE_SPACING)
    return font, best_wrapped_lines, line_height


def _draw_text_on_pil(pil_image, text, bbox, font, lines, line_height,
                       text_color=(0, 0, 0), appearance=None):
    """Draw wrapped text onto a PIL image at the given bbox.
    Renders outline + main text as whole lines (no per-char spacing
    that could cause ghost doubling from overlapping outlines).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = x2 - x1
    bh = y2 - y1
    total_text_height = len(lines) * line_height
    text_y = y1 + (bh - total_text_height) // 2
    draw = ImageDraw.Draw(pil_image)

    need_outline = appearance and appearance.get('need_outline', False)
    outline_color = appearance.get('outline_color', (0, 0, 0)) if appearance else (0, 0, 0)
    outline_ratio = appearance.get('outline_ratio', OUTLINE_RATIO_DEFAULT) if appearance else OUTLINE_RATIO_DEFAULT

    font_size = font.size
    outline_width = max(MIN_OUTLINE_WIDTH, int(round(font_size * outline_ratio)))

    for line in lines:
        try:
            line_px_w = font.getlength(line)
        except Exception:
            line_px_w = len(line) * font_size * 0.6

        text_x = x1 + (bw - line_px_w) // 2

        if need_outline and outline_width >= 1:
            for dx in (-outline_width, 0, outline_width):
                for dy in (-outline_width, 0, outline_width):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((text_x + dx, text_y + dy), line,
                              font=font, fill=outline_color)

        draw.text((text_x, text_y), line, font=font, fill=text_color)
        text_y += line_height


def add_text_bbox(image, text, bbox, font_path, text_color=(0, 0, 0),
                  appearance=None):
    """
    Add text inside a bounding box with dynamic font sizing.

    Args:
        image: numpy array (BGR)
        text: Translated text to render
        bbox: [x1, y1, x2, y2]
        font_path: Path to font file
        text_color: RGB tuple for text color
        appearance: Optional dict with need_outline, outline_color, outline_ratio

    Returns:
        numpy array with text rendered
    """
    if not text or not text.strip():
        return image

    result = _compute_font_and_wrap(text, bbox, font_path)
    if result is None:
        return image

    font, lines, line_height = result
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    _draw_text_on_pil(pil_image, text, bbox, font, lines, line_height, text_color,
                      appearance=appearance)
    image[:, :, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image


def render_all_blocks(image, blocks, font_path):
    """
    Render all text blocks on a single image with ONE conversion round-trip.
    Much faster than calling add_text_bbox per-block.
    Supports per-block outline for readability on complex backgrounds.

    Args:
        image: numpy array (BGR)
        blocks: list of dicts with keys:
            'text' (translated), 'bbox', 'text_color' (RGB),
            and optionally 'appearance' dict with outline settings
        font_path: Path to font file

    Returns:
        numpy array with all text rendered
    """
    if not blocks:
        return image

    # Compute fonts and wrapping for all blocks first
    render_blocks = []
    for block in blocks:
        text = block.get('text', '').strip()
        bbox = block.get('bbox')
        if not text or not bbox or len(bbox) < 4:
            continue
        text_color = block.get('text_color', (0, 0, 0))
        appearance = block.get('appearance', None)
        result = _compute_font_and_wrap(text, bbox, font_path)
        if result is not None:
            font, lines, line_height = result
            render_blocks.append((bbox, font, lines, line_height, text_color, appearance))

    if not render_blocks:
        return image

    # Single BGR→RGB conversion
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw all blocks
    for bbox, font, lines, line_height, text_color, appearance in render_blocks:
        _draw_text_on_pil(pil_image, '', bbox, font, lines, line_height,
                          text_color, appearance=appearance)

    # Single RGB→BGR conversion back
    image[:, :, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image
