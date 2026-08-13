"""Manga Translator CLI - OCR full image then translate and render."""
from translator.translator import MangaTranslator
from add_text import (
    add_text_bbox,
    erase_text_region,
    should_skip_ocr_artifact,
    _detect_sfx,
    _decide_skip_render,
)
from ocr.chrome_lens_ocr import ChromeLensOCR
import numpy as np
import cv2
import argparse
import os
import sys


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manga Translator")
    parser.add_argument("--image-path", "-i", type=str, required=True, help="Path to image")
    parser.add_argument("--font-path", "-f", type=str, default="fonts/animeace_i.ttf", help="Path to font")
    parser.add_argument("--translator", "-t", type=str, choices=["google", "gemini"], default="google",
                        help="Translator to use (google/gemini)")
    parser.add_argument("--source-lang", type=str, default="ja", help="Source language code")
    parser.add_argument("--target-lang", type=str, default="en", help="Target language code")
    parser.add_argument("--save-path", "-s", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Cannot read image from {args.image_path}")
        exit(1)

    ocr = ChromeLensOCR(ocr_language=args.source_lang)
    blocks = ocr(image)
    print(f"Detected {len(blocks)} text blocks")

    translator = MangaTranslator(source=args.source_lang, target=args.target_lang)

    for block in blocks:
        text = block.get('text', '').strip()
        if not text:
            continue

        bbox = block.get('bbox')
        if not bbox or len(bbox) < 4:
            continue

        if should_skip_ocr_artifact(text, bbox, image_shape=image.shape,
                                    source_lang=args.source_lang):
            print(f"  [SKIP OCR ARTIFACT] {text}")
            continue

        translated = translator.translate(text, method=args.translator)
        print(f"  {text} → {translated}")

        image, text_color, appearance = erase_text_region(image, bbox, source_lang=args.source_lang)

        # SFX skip check
        sfx_info = _detect_sfx(translated, args.source_lang)
        analysis_for_skip = {k: appearance.get(k) for k in
            ['bubble_context', 'uniformity', 'intensity_std', 'edge_score']}
        if _decide_skip_render(translated, sfx_info, analysis=analysis_for_skip,
                               source_lang=args.source_lang):
            print(f"  [SKIP SFX] '{translated}' on complex bg")
            continue

        add_text_bbox(image, translated, bbox, args.font_path, text_color, appearance=appearance)

    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, "output_image.jpg")
    cv2.imwrite(save_path, image)
    print(f"Saved to {save_path}")
