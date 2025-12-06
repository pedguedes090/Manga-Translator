"""
Copilot API Translator
Uses copilot-api proxy server (OpenAI-compatible endpoint)
https://github.com/ericc-ch/copilot-api
"""
import requests
import json
from typing import List


class CopilotTranslator:
    """
    Translator using Copilot API proxy server.
    Communicates via OpenAI-compatible /v1/chat/completions endpoint.
    Supports custom prompts and style presets like GeminiTranslator.
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
    
    # Preset style templates (same as GeminiTranslator for consistency)
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
    
    # Available models (from Copilot API)
    MODELS = [
        # GPT-5 Series
        "gpt-5",
        "gpt-5-mini",
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.1-codex-mini",
        "gpt-5.1-codex-max",
        "gpt-5-codex",
        # GPT-4.1 Series
        "gpt-4.1",
        "gpt-41-copilot",
        # GPT-4o Series
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-2024-11-20",
        # GPT-4 Series
        "gpt-4",
        "gpt-4-0125-preview",
        # GPT-3.5
        "gpt-3.5-turbo",
        # Claude Series
        "claude-sonnet-4.5",
        "claude-sonnet-4",
        "claude-opus-4.5",
        "claude-haiku-4.5",
        # Gemini
        "gemini-3-pro-preview",
        "gemini-2.5-pro",
        # Other
        "grok-code-fast-1",
    ]
    
    def __init__(self, server_url: str = "http://localhost:8080", model: str = "gpt-4o", custom_prompt: str = None, style: str = "default"):
        """
        Initialize Copilot translator.
        
        Args:
            server_url: Copilot API proxy server URL (e.g., http://localhost:8080)
            model: Model to use (e.g., gpt-4o, claude-3.5-sonnet)
            custom_prompt: Custom instructions for translation style.
            style: Preset style name from STYLE_PRESETS.
        """
        self.base_url = server_url.rstrip("/")
        self.model = model
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        # Set custom prompt (user prompt takes priority over preset)
        self.custom_prompt = custom_prompt or self.STYLE_PRESETS.get(style, "")
    
    def set_custom_prompt(self, prompt: str):
        """Update custom prompt for translation style."""
        self.custom_prompt = prompt
    
    def _build_style_instructions(self) -> str:
        """Build style instructions for the prompt."""
        if self.custom_prompt:
            return f"\nStyle instructions: {self.custom_prompt}"
        return ""
    
    def translate_single(self, text: str, source: str = "ja", target: str = "en") -> str:
        """Translate a single text string."""
        if not text or not text.strip():
            return text
        
        source_name = self.LANG_NAMES.get(source, "Japanese")
        target_name = self.LANG_NAMES.get(target, "English")
        
        style_text = self._build_style_instructions()
        
        prompt = f"""You are an expert manga/comic translator. Translate the following {source_name} text to {target_name}.

Rules:
- Translate for SPOKEN dialogue, natural when read aloud
- Preserve tone, emotion, and personality
- For Vietnamese: use appropriate pronouns based on context
- Return ONLY the translated text, nothing else{style_text}

Text: {text}"""

        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Copilot translation error: {e}")
            return text
    
    def translate_batch(self, texts: List[str], source: str = "ja", target: str = "en") -> List[str]:
        """
        Translate multiple texts in a single API call.
        
        Args:
            texts: List of texts to translate
            source: Source language code
            target: Target language code
            
        Returns:
            List of translated texts (same order)
        """
        if not texts:
            return []
        
        # Filter empty texts
        indexed_texts = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not indexed_texts:
            return texts
        
        texts_to_translate = [t for _, t in indexed_texts]
        
        source_name = self.LANG_NAMES.get(source, "Japanese")
        target_name = self.LANG_NAMES.get(target, "English")
        
        style_text = self._build_style_instructions()
        
        prompt = f"""You are an expert manga/comic translator. Translate the following {source_name} texts to {target_name}.

Rules:
- These are speech bubble texts from the SAME comic page - maintain consistency
- Translate for SPOKEN dialogue, natural when read aloud
- Preserve tone, emotion, and personality
- For Vietnamese: use appropriate pronouns based on context
- Keep short lines impactful{style_text}

Input (JSON array of texts):
{json.dumps(texts_to_translate, ensure_ascii=False)}

Return ONLY a JSON array with translated texts in the EXACT same order.
Example: ["translation 1", "translation 2"]"""

        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            result_text = result["choices"][0]["message"]["content"].strip()
            
            # Clean up response
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            translations = json.loads(result_text)
            
            # Validate length
            if len(translations) != len(texts_to_translate):
                print(f"Warning: Expected {len(texts_to_translate)} translations, got {len(translations)}")
                # Pad or truncate
                while len(translations) < len(texts_to_translate):
                    translations.append(texts_to_translate[len(translations)])
                translations = translations[:len(texts_to_translate)]
            
            # Rebuild full list
            result_list = list(texts)
            for (orig_idx, _), trans in zip(indexed_texts, translations):
                result_list[orig_idx] = trans
            
            return result_list
            
        except Exception as e:
            print(f"Copilot batch translation error: {e}")
            # Fallback to single translations
            return [self.translate_single(t, source, target) for t in texts]
    
    def translate_pages_batch(
        self, 
        pages_texts: dict, 
        source: str = "ja", 
        target: str = "en",
        context: dict = None
    ) -> dict:
        """
        Translate texts from multiple pages in a single API call.
        Ideal for batch processing 10+ manga pages at once.
        
        Args:
            pages_texts: Dict mapping page names to list of texts
                         e.g., {"page1": ["text1", "text2"], "page2": ["text3"]}
            source: Source language code
            target: Target language code
            context: Optional dict of ALL page texts for context (helps maintain consistency)
            
        Returns:
            Dict with same structure but translated texts
        """
        if not pages_texts:
            return {}
        
        source_name = self.LANG_NAMES.get(source, "Japanese")
        target_name = self.LANG_NAMES.get(target, "English")
        
        # Build context section if context is provided
        context_section = ""
        if context and context != pages_texts:
            # Show summary of other pages for context
            other_pages = {k: v for k, v in context.items() if k not in pages_texts}
            if other_pages:
                context_preview = []
                for page, texts in list(other_pages.items())[:5]:  # First 5 pages for context
                    context_preview.append(f"{page}: {' | '.join(texts[:3])}...")
                context_section = f"""
STORY CONTEXT (from other pages in this batch - use for character/tone consistency):
{chr(10).join(context_preview)}
---
"""
        
        style_text = self._build_style_instructions()
        
        prompt = f"""You are an expert manga/comic translator. Translate the following {source_name} texts to {target_name}.
{context_section}
Context: These are SEQUENTIAL comic pages telling a continuous story. Maintain narrative flow and character voice consistency across all pages.

Rules:
- Translate for SPOKEN dialogue - it must sound natural when read aloud
- Each character should have a consistent voice/speaking style across pages
- Preserve tone, emotion, and personality through careful word choice
- For Vietnamese: Choose appropriate pronouns based on character relationships
- Keep short lines impactful. Don't pad or over-explain.{style_text}

Input (JSON - sequential pages with their speech bubbles):
{json.dumps(pages_texts, ensure_ascii=False, indent=2)}

IMPORTANT: Return ONLY a valid JSON object with the exact same structure but with translated texts.
Keep page names and bubble order exactly the same. No explanations or markdown."""

        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                },
                timeout=120  # Longer timeout for multi-page batch
            )
            response.raise_for_status()
            result = response.json()
            result_text = result["choices"][0]["message"]["content"].strip()
            
            # Clean up response
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            translated = json.loads(result_text)
            print(f"✓ Translated {len(pages_texts)} pages in single batch")
            return translated
            
        except Exception as e:
            print(f"Copilot pages batch translation error: {e}")
            # Fallback: translate each page separately
            result = {}
            for page_name, texts in pages_texts.items():
                result[page_name] = self.translate_batch(texts, source, target)
            return result
    
    def test_connection(self) -> bool:
        """Test if the server is reachable."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from server."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["id"] for m in data.get("data", [])]
        except:
            pass
        return self.MODELS  # Return default list


def translate_manga_pages_batch(
    pages_texts: dict,
    server_url: str = "http://localhost:8080",
    model: str = "gpt-4o",
    source_lang: str = "ja",
    target_lang: str = "en",
    batch_size: int = 10
) -> dict:
    """
    Translate manga pages in batches.
    
    Args:
        pages_texts: All pages' texts {page_name: [texts]}
        server_url: Copilot API server URL
        model: Model to use
        source_lang: Source language code
        target_lang: Target language code
        batch_size: Number of pages per API call (default: 10)
        
    Returns:
        All translated texts
    """
    translator = CopilotTranslator(server_url=server_url, model=model)
    
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
