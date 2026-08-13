# Translator modules
from .translator import MangaTranslator

# Lazy import to avoid hard dependency on google-genai SDK at startup
__all__ = ["MangaTranslator", "GeminiTranslator"]


def get_gemini_translator():
    from .gemini_translator import GeminiTranslator
    return GeminiTranslator
