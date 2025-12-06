from detect_bubbles import detect_bubbles
from process_bubble import process_bubble
from translator.translator import MangaTranslator
from add_text import add_text
from manga_ocr import MangaOcr
from ocr.chrome_lens_ocr import ChromeLensOCR
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manga Translator")

    parser.add_argument("--model-path", "-m", type=str, required=True, help="Path to model")
    parser.add_argument("--image-path", "-i", type=str, required=True, help="Path to image")
    parser.add_argument("--font-path", "-f", type=str, default="fonts/animeace_i.ttf", help="Path to font")
    parser.add_argument("--translator", "-t", type=str, choices=["google", "hf", "baidu", "bing", "nllb"], default="google",
                        help="Translator to use (google/hf/baidu/bing/nllb)")
    parser.add_argument("--ocr-engine", "-o", type=str, choices=["manga-ocr", "chrome-lens"], default="manga-ocr",
                        help="OCR engine to use (manga-ocr/chrome-lens)")
    parser.add_argument("--source-lang", type=str, default="ja",
                        help="Source language code (default: ja)")
    parser.add_argument("--target-lang", type=str, default="en",
                        help="Target language code: en, vi, zh, ko, th, id, fr, de, es, ru (default: en)")
    parser.add_argument("--save-path", "-s", type=str, required=True, help="Path where the output image to be saved")

    args = parser.parse_args()

    results = detect_bubbles(args.model_path, args.image_path)

    manga_translator = MangaTranslator(source=args.source_lang, target=args.target_lang)
    
    # Initialize OCR engine based on selection
    if args.ocr_engine == "chrome-lens":
        print("Using Chrome Lens OCR...")
        mocr = ChromeLensOCR()
    else:
        print("Using Manga OCR...")
        mocr = MangaOcr()

    image = cv2.imread(args.image_path)

    for result in tqdm(results):
        x1, y1, x2, y2, score, class_id = result

        detected_image = image[int(y1):int(y2), int(x1):int(x2)]

        im = Image.fromarray(np.uint8(detected_image * 255))
        text = mocr(im)

        detected_image, cont = process_bubble(detected_image)
        text_translated = manga_translator.translate(text, method=args.translator)

        add_text(detected_image, text_translated, args.font_path, cont)

    save_path = args.save_path + "/output_image.jpg"
    cv2.imwrite(save_path, image)
