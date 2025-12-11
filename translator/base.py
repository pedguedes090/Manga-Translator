"""
Base Translator - Shared constants and utilities for all translators
"""
from typing import Dict


class BaseTranslator:
    """
    Base class for all translators.
    Contains shared constants and utility methods.
    """
    
    # Language code to full name mapping
    LANG_NAMES: Dict[str, str] = {
        "ja": "Japanese",
        "zh": "Chinese",
        "ko": "Korean",
        "en": "English",
        "vi": "Vietnamese",
        "th": "Thai",
        "id": "Indonesian",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian"
    }
    
    # Preset style templates for translation
    STYLE_PRESETS: Dict[str, str] = {
        "default": "",
        "formal": "Use formal, polite language. Use respectful pronouns and expressions.",
        "casual": "Use casual, natural everyday language. Like friends talking to each other.",
        "keep_honorifics": "Keep Japanese honorifics like -san, -kun, -chan, -sama, senpai, sensei untranslated.",
        "localize": "Fully localize cultural references. Adapt idioms and expressions to feel native.",
        "literal": "Translate meaning accurately but ensure it still sounds natural when spoken.",
        "web_novel": "Use dramatic web novel style with impactful expressions and emotional weight.",
        "action": "Use short, punchy sentences. Quick pace. Impactful dialogue.",
    }
    
    def __init__(self, custom_prompt: str = None, style: str = "default"):
        """
        Initialize base translator.
        
        Args:
            custom_prompt: Custom instructions for translation style.
            style: Preset style name from STYLE_PRESETS.
        """
        self.custom_prompt = custom_prompt or self.STYLE_PRESETS.get(style, "")
    
    def set_custom_prompt(self, prompt: str):
        """Update custom prompt for translation style."""
        self.custom_prompt = prompt
    
    def _build_style_instructions(self) -> str:
        """Build style instructions for the prompt."""
        if self.custom_prompt:
            return f"\nStyle instructions: {self.custom_prompt}"
        return ""
    
    def get_lang_name(self, code: str, default: str = "Japanese") -> str:
        """Get full language name from code."""
        return self.LANG_NAMES.get(code, default)
