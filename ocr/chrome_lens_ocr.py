"""
Chrome Lens OCR module using chrome-lens-py library.
Provides OCR with text block segmentation (text + bounding boxes).
"""
import asyncio
import os
import random
import re
import tempfile
import time
from PIL import Image
import cv2
import numpy as np

from chrome_lens_py import LensAPI

MAX_CONCURRENT_OCR = 10
MAX_RETRIES = 3
TOTAL_TIMEOUT_SECONDS = 45  # cap total retry wall-clock time
_HANGUL_TEXT_RE = re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣]')


def _has_hangul_text(text):
    return bool(_HANGUL_TEXT_RE.search(str(text or "")))


def _bbox_dims(block):
    bbox = block.get("bbox") if isinstance(block, dict) else None
    if not bbox or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w <= 0 or h <= 0:
        return None
    return x1, y1, x2, y2, w, h


def _looks_like_tall_narrow_ocr_error(block, image_size, language):
    text = str(block.get("text", "")).strip()
    language_is_korean = str(language or "").lower().startswith("ko")
    if not language_is_korean and not _has_hangul_text(text):
        return False
    dims = _bbox_dims(block)
    if dims is None or not image_size:
        return False
    _, _, _, _, w, h = dims
    img_w, _ = image_size
    return (
        len(text) >= 2
        and h >= 50
        and h / max(w, 1.0) >= 2.2
        and w <= img_w * 0.12
    )


def _bbox_center(block):
    dims = _bbox_dims(block)
    if dims is None:
        return None
    x1, y1, x2, y2, _, _ = dims
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _merge_scaled_retry_blocks(original_blocks, scaled_blocks, image_size, language):
    refined = []
    used_scaled = set()

    for block in original_blocks:
        replacement = None
        if _looks_like_tall_narrow_ocr_error(block, image_size, language):
            raw_dims = _bbox_dims(block)
            raw_center = _bbox_center(block)
            if raw_dims and raw_center:
                _, _, _, _, raw_w, raw_h = raw_dims
                best = None
                best_score = None
                for idx, candidate in enumerate(scaled_blocks):
                    if idx in used_scaled:
                        continue
                    cand_dims = _bbox_dims(candidate)
                    cand_center = _bbox_center(candidate)
                    if cand_dims is None or cand_center is None:
                        continue
                    _, _, _, _, cand_w, cand_h = cand_dims
                    if _looks_like_tall_narrow_ocr_error(candidate, image_size, language):
                        continue
                    dx = abs(raw_center[0] - cand_center[0])
                    dy = abs(raw_center[1] - cand_center[1])
                    if dx > max(80, raw_h * 0.55) or dy > max(60, raw_h * 0.35):
                        continue
                    if cand_w < raw_w * 1.5 or cand_h > raw_h * 0.85:
                        continue
                    score = dx + dy + abs(cand_h - raw_h * 0.25)
                    if best_score is None or score < best_score:
                        best = (idx, candidate)
                        best_score = score
                if best is not None:
                    used_scaled.add(best[0])
                    replacement = dict(best[1])
                    replacement["_scaled_retry"] = True

        refined.append(replacement if replacement is not None else block)

    return refined


class ChromeLensOCR:
    """
    OCR engine using Google Chrome Lens API via chrome-lens-py.
    Uses 'blocks' output format to get text + bounding box for each text region.
    """

    def __init__(self, ocr_language: str = "ja", max_concurrent: int = MAX_CONCURRENT_OCR):
        self.api = LensAPI()
        self.ocr_language = ocr_language
        self.max_concurrent = max_concurrent
        # Initialize semaphore eagerly so it is shared across all callers
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._loop = None

    def __call__(self, image) -> list:
        """
        Process an image and return text blocks with bounding boxes.

        Returns:
            list[dict]: Each dict has {'text': str, 'bbox': [x1, y1, x2, y2]}
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 4:
                image = Image.fromarray(image[:, :, [2, 1, 0, 3]])
            elif image.ndim == 3 and image.shape[2] >= 3:
                image = Image.fromarray(image[:, :, [2, 1, 0]])
            else:
                image = Image.fromarray(image)

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(self._process_blocks(image), loop)
            return future.result(timeout=TOTAL_TIMEOUT_SECONDS + 15)
        except RuntimeError:
            if not hasattr(self, '_loop') or self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            try:
                return self._loop.run_until_complete(self._process_blocks(image))
            finally:
                # Close the loop to free OS resources (file descriptors, thread pool)
                if self._loop is not None and not self._loop.is_closed():
                    self._loop.close()
                    self._loop = None

    async def _process_blocks(self, image, max_retries: int = MAX_RETRIES) -> list:
        """
        Process image with Chrome Lens API in blocks mode.
        Returns text blocks with bbox coordinates.

        Retries ALL transient failures (DNS, connection, SSL, timeout, server errors)
        with exponential backoff, capped by a total deadline.
        """
        deadline = time.monotonic() + TOTAL_TIMEOUT_SECONDS

        async with self._semaphore:
            for attempt in range(max_retries):
                # Only add jitter on retries (not the first attempt)
                if attempt > 0:
                    base_wait = 2 ** attempt  # 2s, 4s, 8s
                    jitter = random.uniform(0, base_wait)
                    wait_time = min(base_wait + jitter, deadline - time.monotonic())
                    if wait_time <= 0:
                        print(f"Chrome Lens OCR: total timeout reached before attempt {attempt + 1}")
                        return []
                    print(f"OCR retry {attempt + 1}/{max_retries}, waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)

                try:
                    result = await self.api.process_image(
                        image_path=image,
                        ocr_language=self.ocr_language,
                        output_format='blocks'
                    )

                    # Parse text_blocks into standardized format
                    raw_blocks = result.get("text_blocks", [])
                    blocks = self._normalize_blocks(raw_blocks, self._get_image_size(image))
                    blocks = await self._refine_with_scaled_retry(image, blocks)
                    return blocks

                except Exception as e:
                    error_str = str(e)

                    # Check if we still have time for another retry
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        print(f"Chrome Lens OCR: total timeout reached after error: {e}")
                        return []

                    if attempt < max_retries - 1:
                        # Retry all errors — network blips, DNS, SSL, timeouts, server errors
                        print(f"OCR error (attempt {attempt + 1}/{max_retries}): {e}")
                    else:
                        # Exhausted all retries
                        print(f"Chrome Lens OCR failed after {max_retries} attempts: {e}")
                        return []

        return []

    def _save_scaled_retry_image(self, image, scale=1.5):
        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
            source = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            source = image
        else:
            source = cv2.imread(image, cv2.IMREAD_COLOR)
            if source is None:
                rgb = np.array(Image.open(image).convert("RGB"))
                source = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        height, width = source.shape[:2]
        scaled = cv2.resize(
            source,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
        cv2.imwrite(temp_path, scaled)
        return temp_path, (scaled.shape[1], scaled.shape[0])

    async def _refine_with_scaled_retry(self, image, blocks, scale=1.5):
        image_size = self._get_image_size(image)
        if not any(
            _looks_like_tall_narrow_ocr_error(block, image_size, self.ocr_language)
            for block in blocks
        ):
            return blocks

        try:
            temp_path = None
            try:
                temp_path, scaled_size = self._save_scaled_retry_image(image, scale=scale)
                result = await self.api.process_image(
                    image_path=temp_path,
                    ocr_language=self.ocr_language,
                    output_format='blocks',
                )
                scaled_blocks = self._normalize_blocks(
                    result.get("text_blocks", []),
                    scaled_size,
                )
            finally:
                if temp_path:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
        except Exception as e:
            print(f"Scaled OCR retry failed: {e}")
            return blocks

        mapped_blocks = []
        for block in scaled_blocks:
            mapped = dict(block)
            mapped["bbox"] = [int(round(v / scale)) for v in block.get("bbox", [])[:4]]
            mapped_blocks.append(mapped)

        return _merge_scaled_retry_blocks(
            blocks,
            mapped_blocks,
            image_size,
            self.ocr_language,
        )

    def _get_image_size(self, image):
        if isinstance(image, Image.Image):
            return image.size
        try:
            with Image.open(image) as img:
                return img.size
        except Exception:
            return None

    def _bbox_from_geometry(self, geometry: dict, image_size):
        if not geometry or not image_size:
            return None

        img_w, img_h = image_size
        try:
            cx = float(geometry["center_x"])
            cy = float(geometry["center_y"])
            width = float(geometry["width"])
            height = float(geometry["height"])
        except (KeyError, TypeError, ValueError):
            return None

        coordinate_type = str(geometry.get("coordinate_type", "NORMALIZED")).upper()
        if coordinate_type == "NORMALIZED" or (0 <= cx <= 1 and 0 <= cy <= 1 and width <= 1 and height <= 1):
            x1 = (cx - width / 2) * img_w
            y1 = (cy - height / 2) * img_h
            x2 = (cx + width / 2) * img_w
            y2 = (cy + height / 2) * img_h
        else:
            x1 = cx - width / 2
            y1 = cy - height / 2
            x2 = cx + width / 2
            y2 = cy + height / 2

        x1 = max(0, min(img_w, int(round(x1))))
        y1 = max(0, min(img_h, int(round(y1))))
        x2 = max(0, min(img_w, int(round(x2))))
        y2 = max(0, min(img_h, int(round(y2))))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _normalize_blocks(self, raw_blocks: list, image_size=None) -> list:
        """
        Normalize text blocks into a consistent format.
        Handles various possible formats from the API:
        - [x1, y1, x2, y2, text] (list format)
        - {'text': ..., 'bbox': [x1, y1, x2, y2]} (dict format)
        - {'text': ..., 'x': ..., 'y': ..., 'width': ..., 'height': ...} (alternate dict)
        - {'text': ..., 'geometry': {'center_x': ..., ...}} (chrome-lens-py blocks)
        """
        blocks = []
        for block in raw_blocks:
            if isinstance(block, dict):
                text = block.get('text', '') or block.get('ocr_text', '')
                if 'bbox' in block:
                    bbox = block['bbox']
                    if len(bbox) >= 4:
                        blocks.append({'text': text, 'bbox': list(bbox[:4])})
                elif 'x' in block and 'y' in block:
                    x, y = block['x'], block['y']
                    w = block.get('width', 0)
                    h = block.get('height', 0)
                    blocks.append({'text': text, 'bbox': [x, y, x + w, y + h]})
                elif 'geometry' in block:
                    bbox = self._bbox_from_geometry(block.get('geometry'), image_size)
                    if bbox:
                        blocks.append({'text': text, 'bbox': bbox})
            elif isinstance(block, (list, tuple)) and len(block) >= 5:
                x1, y1, x2, y2 = block[:4]
                text = block[4]
                blocks.append({'text': text, 'bbox': [x1, y1, x2, y2]})

        return blocks

    def process_batch(self, images: list) -> list:
        """
        Process multiple images concurrently.

        Returns:
            list[list[dict]]: List of text blocks for each image
        """
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[2] == 4:
                    pil_images.append(Image.fromarray(img[:, :, [2, 1, 0, 3]]))
                elif img.ndim == 3 and img.shape[2] >= 3:
                    pil_images.append(Image.fromarray(img[:, :, [2, 1, 0]]))
                else:
                    pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                self._process_batch_blocks(pil_images), loop
            )
            return future.result(timeout=TOTAL_TIMEOUT_SECONDS * 3)
        except RuntimeError:
            if not hasattr(self, '_loop') or self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            try:
                return self._loop.run_until_complete(self._process_batch_blocks(pil_images))
            finally:
                if self._loop is not None and not self._loop.is_closed():
                    self._loop.close()
                    self._loop = None

    async def _process_batch_blocks(self, images: list) -> list:
        tasks = [self._process_blocks(img) for img in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"  [Image {i+1}] ERROR: {r}")
                processed.append([])
            else:
                blocks = r
                if blocks:
                    total_text = sum(len(b.get('text', '')) for b in blocks)
                    print(f"  [Image {i+1}] OK: {len(blocks)} blocks, {total_text} chars")
                else:
                    print(f"  [Image {i+1}] EMPTY (no text detected)")
                processed.append(blocks)

        print(f"OCR completed: {sum(1 for b in processed if b)}/{len(images)} images with text")
        return processed
