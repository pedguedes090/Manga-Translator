from deep_translator import GoogleTranslator
from transformers import pipeline
import translators as ts
import random
import time


class MangaTranslator:
    def __init__(self):
        self.target = "en"
        self.source = "ja"
        self.translators = {
            "google": self._translate_with_google,
            "hf": self._translate_with_hf,
            "sogou": self._translate_with_sogou,
            "bing": self._translate_with_bing
        }

    def translate(self, text, method="google"):
        """
        Translates the given text to the target language using the specified method.

        Args:
            text (str): The text to be translated.
            method (str):"google" for Google Translator, 
                         "hf" for Helsinki-NLP's opus-mt-ja-en model (HF pipeline)
                         "sogou" for Sogou Translate
                         "bing" for Microsoft Bing Translator

        Returns:
            str: The translated text.
        """
        translator_func = self.translators.get(method)

        if translator_func:
            return translator_func(self._preprocess_text(text))
        else:
            raise ValueError("Invalid translation method.")
            
    def _translate_with_google(self, text):
        self._delay()
        translator = GoogleTranslator(source=self.source, target=self.target)
        translated_text = translator.translate(text)
        return translated_text if translated_text is not None else text

    def _translate_with_hf(self, text):
        pipe = pipeline("translation", model=f"Helsinki-NLP/opus-mt-ja-en")
        translated_text = pipe(text)[0]["translation_text"]
        return translated_text if translated_text is not None else text

    def _translate_with_sogou(self, text):
        self._delay()
        translated_text = ts.translate_text(text, translator="sogou",
                                            from_language=self.source,
                                            to_language=self.target)
        return translated_text if translated_text is not None else text

    def _translate_with_bing(self, text):
        self._delay()
        translated_text = ts.translate_text(text, translator="bing",
                                            from_language=self.source, 
                                            to_language=self.target)
        return translated_text if translated_text is not None else text

    def _preprocess_text(self, text):
        preprocessed_text = text.replace("．", ".")
        return preprocessed_text

    def _delay(self):
        time.sleep(random.randint(3, 5))
