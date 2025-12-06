from add_text import add_text
from detect_bubbles import detect_bubbles
from process_bubble import process_bubble
from translator import MangaTranslator
from ultralytics import YOLO
from manga_ocr import MangaOcr
from PIL import Image
import gradio as gr
import numpy as np
import cv2


MODEL = "model.pt"
EXAMPLE_LIST = [["examples/0.png"],
                 ["examples/ex0.png"]]
TITLE = "Manga Translator"
DESCRIPTION = "Translate text in manga bubbles!"


def predict(img, translation_method, font):
    if translation_method == None:
        translation_method = "google"
    if font == None:
        font = "fonts/animeace_i.ttf"

    results = detect_bubbles(MODEL, img)

    manga_translator = MangaTranslator()
    mocr = MangaOcr()

    image = np.array(img)

    for result in results:
        x1, y1, x2, y2, score, class_id = result

        detected_image = image[int(y1):int(y2), int(x1):int(x2)]

        im = Image.fromarray(np.uint8((detected_image)*255))
        text = mocr(im)

        detected_image, cont = process_bubble(detected_image)

        text_translated = manga_translator.translate(text,
                                                     method=translation_method)

        image[int(y1):int(y2), int(x1):int(x2)] = add_text(detected_image, text_translated, font, cont)

    return Image.fromarray(image)

demo = gr.Interface(fn=predict,
                    inputs=["image",
                            gr.Dropdown([("Google", "google"),
                                         ("Helsinki-NLP's opus-mt-ja-en model",
                                          "hf"),
                                         ("Sogou", "sogou"),
                                         ("Bing", "bing")],
                                        label="Translation Method",
                                        value="google"),
                            gr.Dropdown([("animeace_i", "fonts/animeace_i.ttf"),
                                         ("mangati", "fonts/mangati.ttf"),
                                         ("ariali", "fonts/ariali.ttf")],
                                        label="Text Font",
                                        value="fonts/animeace_i.ttf")
                            ],
                    outputs=[gr.Image()],
                    examples=EXAMPLE_LIST,
                    title=TITLE,
                    description=DESCRIPTION)


demo.launch(debug=False,
            share=False)
