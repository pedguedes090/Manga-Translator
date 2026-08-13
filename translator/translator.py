from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import threading
import re


GOOGLE_LANGUAGE_CODES = {
    "zh": "zh-CN",
}


def _google_language_code(language_code):
    return GOOGLE_LANGUAGE_CODES.get(language_code, language_code)


class MangaTranslator:
    def __init__(self, source="ja", target="en", gemini_api_key=None, gemini_api_keys=None,
                 gemini_model=None):
        self.target = target
        self.source = source
        self.gemini_api_key = gemini_api_key
        self.gemini_api_keys = gemini_api_keys
        self.gemini_model = gemini_model
        self.translators = {
            "google": self._translate_with_google,
            "gemini": self._translate_with_gemini
        }
        self._gemini_translator = None
        self._google_translator = None
        self._google_translator_lock = threading.Lock()
        self._batch_executor = None
        self._max_workers = min(8, (__import__('os').cpu_count() or 4))

    def set_languages(self, source=None, target=None):
        if source:
            self.source = source
        if target:
            self.target = target
        # Invalidate cached Google translator on language change
        self._google_translator = None

    def translate(self, text, method="google"):
        translator_func = self.translators.get(method)
        if translator_func:
            return translator_func(self._preprocess_text(text))
        else:
            raise ValueError("Invalid translation method.")

    def _get_google_translator(self):
        """Get or create a cached GoogleTranslator instance."""
        if self._google_translator is None:
            with self._google_translator_lock:
                if self._google_translator is None:
                    self._google_translator = GoogleTranslator(
                        source=_google_language_code(self.source),
                        target=_google_language_code(self.target),
                    )
        return self._google_translator

    def _translate_with_google(self, text):
        try:
            translator = self._get_google_translator()
            translated_text = translator.translate(text)
            return translated_text if translated_text is not None else text
        except Exception as e:
            print(f"Google translation failed: {e}")
            return text

    def translate_batch_google(self, texts):
        """Translate multiple texts using Google Translate in parallel via ThreadPoolExecutor.
        Uses thread-local storage so each worker thread reuses its own GoogleTranslator instance.
        """
        if not texts:
            return []

        valid = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not valid:
            return list(texts)

        if self._batch_executor is None:
            self._batch_executor = ThreadPoolExecutor(max_workers=self._max_workers)

        source = _google_language_code(self.source)
        target = _google_language_code(self.target)

        # Thread-local storage — one GoogleTranslator per worker thread, reused across tasks
        _thread_local = threading.local()

        def _get_translator():
            if not hasattr(_thread_local, 'translator'):
                _thread_local.translator = GoogleTranslator(source=source, target=target)
            return _thread_local.translator

        def _translate_one(idx, text):
            translator = _get_translator()
            preprocessed = self._preprocess_text(text)
            translated = translator.translate(preprocessed)
            return idx, translated if translated is not None else text

        results = dict(valid)
        futures = [self._batch_executor.submit(_translate_one, i, t) for i, t in valid]

        try:
            for future in as_completed(futures, timeout=60):
                try:
                    idx, translated = future.result()
                    results[idx] = translated
                except Exception as e:
                    print(f"Google batch translate failed for item: {e}")
        except TimeoutError:
            print("Google batch translate timed out; keeping original text for pending items.")

        return [results.get(i, t) for i, t in enumerate(texts)]

    def _translate_with_gemini(self, text):
        try:
            if self._gemini_translator is None:
                from .gemini_translator import GeminiTranslator
                api_keys = (
                    getattr(self, '_gemini_api_keys', None)
                    or self.gemini_api_keys
                    or getattr(self, '_gemini_api_key', None)
                    or self.gemini_api_key
                )
                if not api_keys:
                    raise ValueError("Gemini API key required. Please enter it in the web form.")
                custom_prompt = getattr(self, '_gemini_custom_prompt', None)
                model_name = getattr(self, '_gemini_model', None) or self.gemini_model
                self._gemini_translator = GeminiTranslator(
                    api_keys=api_keys,
                    custom_prompt=custom_prompt,
                    model_name=model_name,
                )
            
            return self._gemini_translator.translate_single(
                text, 
                source=self.source, 
                target=self.target
            )
        except Exception as e:
            print(f"Gemini translation error: {e}")
            return text

    def _preprocess_text(self, text):
        preprocessed_text = text.replace("．", ".")
        cjk = r"\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af"
        preprocessed_text = re.sub(
            rf"([{cjk}])\s*\n\s*(?=[{cjk}])",
            r"\1",
            preprocessed_text,
        )
        preprocessed_text = re.sub(r"\s*\n\s*", " ", preprocessed_text)
        preprocessed_text = re.sub(r"[ \t]{2,}", " ", preprocessed_text)
        return preprocessed_text.strip()
