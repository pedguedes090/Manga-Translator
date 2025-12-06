"""
Gemini Translator with Batch Processing
Uses Gemini 2.5 Flash-Lite for cost-effective translation
Supports multiple source languages and custom prompts
"""
import google.generativeai as genai
import json
import os
import time
from typing import List, Dict, Optional

# Constants for retry logic
MAX_RETRIES = 3
RETRY_DELAY_BASE = 0.5  # Faster recovery: 0.5s → 1s → 2s


class GeminiTranslator:
    """
    Translator using Google Gemini 2.5 Flash-Lite.
    Supports batch translation to minimize API calls.
    """
    
    LANG_NAMES = {
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
    
    # Preset style templates
    STYLE_PRESETS = {
        "default": "",
        "formal": "Use formal, polite language. Use respectful pronouns and expressions.",
        "casual": "Use casual, natural everyday language. Like friends talking to each other.",
        "keep_honorifics": "Keep Japanese honorifics like -san, -kun, -chan, -sama, senpai, sensei untranslated.",
        "localize": "Fully localize cultural references. Adapt idioms and expressions to feel native.",
        "literal": "Translate meaning accurately but ensure it still sounds natural when spoken.",
        "web_novel": "Use dramatic web novel style with impactful expressions and emotional weight.",
        "action": "Use short, punchy sentences. Quick pace. Impactful dialogue.",
    }
    
    def __init__(self, api_key: str = None, custom_prompt: str = None, style: str = "default"):
        """
        Initialize Gemini translator.
        
        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
            custom_prompt: Custom instructions for translation style.
            style: Preset style name from STYLE_PRESETS.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY or pass api_key.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")
        
        # Set custom prompt (user prompt takes priority over preset)
        self.custom_prompt = custom_prompt or self.STYLE_PRESETS.get(style, "")
    
    def set_custom_prompt(self, prompt: str):
        """Update custom prompt for translation style."""
        self.custom_prompt = prompt
    
    def _build_style_instructions(self) -> str:
        """Build style instructions for the prompt."""
        if self.custom_prompt:
            return f"\n\nStyle instructions: {self.custom_prompt}"
        return ""
        
    def translate_single(
        self, 
        text: str, 
        source: str = "ja", 
        target: str = "en",
        custom_prompt: str = None
    ) -> str:
        """
        Translate a single text string.
        
        Args:
            text: Text to translate
            source: Source language code (ja, zh, ko, etc.)
            target: Target language code
            custom_prompt: Override custom prompt for this call
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text
            
        source_name = self.LANG_NAMES.get(source, "Japanese")
        target_name = self.LANG_NAMES.get(target, "English")
        style = custom_prompt or self.custom_prompt
        style_text = f"\nStyle: {style}" if style else ""
        
        prompt = f"""You are an expert manga/comic translator specializing in {source_name} to {target_name} translation.

Translation Guidelines:
- Translate for SPOKEN dialogue, not written text. It should sound natural when read aloud.
- Preserve the character's tone, emotion, and personality through word choice.
- Use natural sentence structures in {target_name}. Avoid awkward literal translations.
- For Vietnamese: Use appropriate pronouns (tao/mày for close friends, tôi/anh/em for normal, etc.) based on context.
- Keep exclamations and emotional expressions feeling authentic.
- Maintain the impact and rhythm of short/punchy lines.{style_text}

IMPORTANT: Return ONLY the translated text. No explanations, no quotes, no formatting.

Original text: {text}"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini translation error: {e}")
            return text
    
    def translate_batch(
        self, 
        texts: List[str], 
        source: str = "ja", 
        target: str = "en",
        custom_prompt: str = None
    ) -> List[str]:
        """
        Translate multiple texts in a single API call with retry logic.
        
        Args:
            texts: List of texts to translate
            source: Source language code
            target: Target language code
            custom_prompt: Override custom prompt for this call
            
        Returns:
            List of translated texts (same order)
        """
        if not texts:
            return []
            
        # Filter empty texts but keep track of indices
        indexed_texts = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        
        if not indexed_texts:
            return texts
        
        texts_to_translate = [t for _, t in indexed_texts]
        translations = self._translate_batch_internal(texts_to_translate, source, target, custom_prompt)
        
        # Rebuild full list with original empty strings preserved
        result = list(texts)
        for (orig_idx, _), trans in zip(indexed_texts, translations):
            result[orig_idx] = trans
            
        return result
    
    def _translate_batch_internal(
        self,
        texts_to_translate: List[str],
        source: str,
        target: str,
        custom_prompt: str = None
    ) -> List[str]:
        """Internal method to translate a single chunk with retry logic."""
        source_name = self.LANG_NAMES.get(source, "Japanese")
        target_name = self.LANG_NAMES.get(target, "English")
        
        style = custom_prompt or self.custom_prompt
        style_text = f"\nStyle instructions: {style}" if style else ""
        
        prompt = f"""You are an expert manga/comic translator with years of experience in {source_name} to {target_name} translation.

Translation Guidelines:
- These are speech bubble texts from the SAME comic page - maintain consistency in character voices.
- Translate for SPOKEN dialogue. It must sound natural when read aloud, not stiff or robotic.
- Preserve each character's tone, emotion, and personality through appropriate word choice.
- Use natural {target_name} sentence structures. AVOID awkward literal word-for-word translations.
- For Vietnamese specifically:
  + Use appropriate pronouns based on relationship (tao/mày, tôi/cậu, anh/em, etc.)
  + Translate exclamations naturally (くそ → Chết tiệt, やばい → Chết rồi, etc.)
  + Keep dialogue feeling authentic to how Vietnamese people actually speak
- Maintain the impact of short/punchy lines. Don't over-explain.
- Keep emotional expressions and interjections feeling authentic.{style_text}

Input texts (JSON array - each is a separate speech bubble):
{json.dumps(texts_to_translate, ensure_ascii=False)}

IMPORTANT: Return ONLY a valid JSON array with translated texts in the EXACT same order.
Format: ["translation 1", "translation 2", ...]"""
        
        # Retry with exponential backoff
        for attempt in range(MAX_RETRIES):
            try:
                response = self.model.generate_content(prompt)
                result_text = response.text.strip()
                
                # Clean up response if needed
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.startswith("```"):
                    result_text = result_text[3:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                result_text = result_text.strip()
                
                translations = json.loads(result_text)
                
                # Validate response length
                if len(translations) != len(texts_to_translate):
                    raise ValueError(f"Expected {len(texts_to_translate)} translations, got {len(translations)}")
                
                return translations
                
            except Exception as e:
                error_str = str(e)
                print(f"Gemini batch attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                
                # Check if it's a quota error - don't retry or fallback
                if "429" in error_str or "quota" in error_str.lower():
                    print("⚠️ Quota exceeded! Returning original texts to avoid more API calls.")
                    print("   Wait 1 minute or upgrade your Gemini API plan.")
                    return texts_to_translate  # Return original texts
                
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    # Only fallback to single translations if NOT quota error
                    print("All retries failed, falling back to single translations")
                    return [self.translate_single(t, source, target) for t in texts_to_translate]
        
        return texts_to_translate  # Fallback: return original
    
    def translate_pages_batch(
        self, 
        pages_texts: Dict[str, List[str]], 
        source: str = "ja", 
        target: str = "en",
        custom_prompt: str = None,
        context: Dict[str, List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Translate texts from multiple pages in a single API call.
        Ideal for batch processing 10 manga pages at once.
        
        Args:
            pages_texts: Dict mapping page names to list of texts
            source: Source language code
            target: Target language code
            custom_prompt: Override custom prompt for this call
            context: Optional dict of ALL page texts for context (helps maintain consistency)
            
        Returns:
            Dict with same structure but translated texts
        """
        if not pages_texts:
            return {}
        
        source_name = self.LANG_NAMES.get(source, "Japanese")
        target_name = self.LANG_NAMES.get(target, "English")
        
        style = custom_prompt or self.custom_prompt
        style_text = f"\nStyle instructions: {style}" if style else ""
        
        # Build context section if context is provided
        context_section = ""
        if context and context != pages_texts:
            other_pages = {k: v for k, v in context.items() if k not in pages_texts}
            if other_pages:
                context_preview = []
                for page, texts in list(other_pages.items())[:5]:
                    context_preview.append(f"{page}: {' | '.join(texts[:3])}...")
                context_section = f"""
STORY CONTEXT (from other pages - use for character/tone consistency):
{chr(10).join(context_preview)}
---
"""
        
        prompt = f"""You are an expert manga/comic translator with deep understanding of {source_name} to {target_name} translation.
{context_section}
Context: These are SEQUENTIAL comic pages telling a continuous story. Maintain narrative flow and character voice consistency across all pages.

Translation Guidelines:
- Translate for SPOKEN dialogue - it must sound natural when read aloud.
- Each character should have a consistent voice/speaking style across pages.
- Preserve tone, emotion, and personality through careful word choice.
- Use natural {target_name} sentence structures. NEVER translate word-for-word literally.
- For Vietnamese:
  + Choose appropriate pronouns based on character relationships and social context
  + Translate interjections and exclamations to feel authentic (not literal)
  + Use natural Vietnamese speech patterns, not textbook Vietnamese
- Keep short lines impactful. Don't pad or over-explain.
- Sound effects and onomatopoeia: translate the meaning/feeling, not literally.{style_text}

Input (JSON - sequential pages with their speech bubbles):
{json.dumps(pages_texts, ensure_ascii=False, indent=2)}

IMPORTANT: Return ONLY a valid JSON object with the exact same structure but with translated texts.
Keep page names and bubble order exactly the same. No explanations or markdown."""

        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean up response
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            return json.loads(result_text)
            
        except Exception as e:
            print(f"Gemini pages batch translation error: {e}")
            # Fallback: translate each page separately
            result = {}
            for page_name, texts in pages_texts.items():
                result[page_name] = self.translate_batch(texts, source, target)
            return result


# Convenience function for batch size of 10 pages
def translate_manga_batch(
    pages_texts: Dict[str, List[str]],
    api_key: str,
    source_lang: str = "ja",
    target_lang: str = "en",
    custom_prompt: str = None,
    batch_size: int = 10
) -> Dict[str, List[str]]:
    """
    Translate manga pages in batches of 10.
    
    Args:
        pages_texts: All pages' texts
        api_key: Gemini API key
        source_lang: Source language code (ja, zh, ko, etc.)
        target_lang: Target language code
        custom_prompt: Custom style instructions
        batch_size: Number of pages per API call (default: 10)
        
    Returns:
        All translated texts
    """
    translator = GeminiTranslator(api_key, custom_prompt=custom_prompt)
    
    page_names = list(pages_texts.keys())
    all_results = {}
    
    # Process in batches
    for i in range(0, len(page_names), batch_size):
        batch_pages = page_names[i:i + batch_size]
        batch_texts = {name: pages_texts[name] for name in batch_pages}
        
        print(f"Translating pages {i+1} to {min(i+batch_size, len(page_names))}...")
        batch_results = translator.translate_pages_batch(
            batch_texts, 
            source=source_lang, 
            target=target_lang
        )
        all_results.update(batch_results)
    
    return all_results
