"""
Local LLM Translator
Uses OpenAI-compatible API endpoints (Ollama, LM Studio, LocalAI, vLLM, Copilot-API, etc.)
"""
import requests
import json
from typing import List, TYPE_CHECKING

from .base import BaseTranslator

if TYPE_CHECKING:
    from .context_memory import ContextMemory


class LocalLLMTranslator(BaseTranslator):
    """
    Translator using OpenAI-compatible local LLM servers.
    Works with Ollama, LM Studio, LocalAI, vLLM, Copilot-API, and similar servers.
    Communicates via /v1/chat/completions endpoint.
    """

    
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
        super().__init__(custom_prompt=custom_prompt, style=style)
        
        self.base_url = server_url.rstrip("/")
        self.model = model
        self.endpoint = f"{self.base_url}/v1/chat/completions"
    
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
        
        prompt = f"""Dịch manga/comic từ {source_name} sang {target_name}.

QUY TẮC:
1. HỘI THOẠI NÓI - phải nghe tự nhiên như người thật nói
2. KHÔNG dịch word-by-word, diễn đạt lại theo cách người Việt nói
3. Giữ cảm xúc, tính cách nhân vật

TIẾNG VIỆT:
- TÊN: giữ nguyên (Tanaka, Kim, Lý...), không dịch nghĩa. Kính ngữ: sunbae→tiền bối, sensei→thầy
- Đại từ: tao/mày, tôi/cậu, anh/em... phù hợp quan hệ
- Thán từ tự nhiên: くそ→Đ*t/Chết tiệt, やばい→Toang, すごい→Đỉnh
- Khẩu ngữ: oke, ngon, tởm, đỉnh, chill...
- TRÁNH dịch kiểu sách giáo khoa{style_text}

Input:
{json.dumps(texts_to_translate, ensure_ascii=False)}

Trả về JSON array với bản dịch theo ĐÚNG THỨ TỰ.
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
        context_memory: 'ContextMemory' = None
    ) -> dict:
        """
        Translate texts from multiple pages in a single API call.
        Ideal for batch processing 10+ manga pages at once.
        
        Args:
            pages_texts: Dict mapping page names to list of texts
                         e.g., {"page1": ["text1", "text2"], "page2": ["text3"]}
            source: Source language code
            target: Target language code
            context_memory: Optional ContextMemory object for consistent translation
            
        Returns:
            Dict with same structure but translated texts
        """
        if not pages_texts:
            return {}
        
        source_name = self.LANG_NAMES.get(source, "Japanese")
        target_name = self.LANG_NAMES.get(target, "English")
        
        # Build context section from ContextMemory if provided
        context_section = ""
        if context_memory:
            context_section = context_memory.generate_context_prompt()
        
        style_text = self._build_style_instructions()
        
        prompt = f"""Dịch manga/comic từ {source_name} sang {target_name}.
{context_section}
Đây là các trang LIÊN TIẾP trong cùng 1 story.

QUY TẮC DỊCH:
1. HỘI THOẠI NÓI - phải nghe tự nhiên như người thật nói
2. KHÔNG dịch word-by-word, diễn đạt lại theo cách người Việt nói
3. Mỗi nhân vật giữ giọng điệu nhất quán

HƯỚNG DẪN TIẾNG VIỆT:
- TÊN: giữ nguyên (Tanaka, Kim, Lý...). Kính ngữ: sunbae→tiền bối, sensei→thầy, oppa→anh
- Đại từ: tao/mày (thân), tôi/anh (trang trọng), anh/em (yêu), con/bố (gia đình)
- Thán từ: くそ→Đ*t/Chết tiệt, やばい→Toang, すごい→Đỉnh, なに→Cái gì
- Khẩu ngữ: oke, ngon, tởm, đỉnh, chill, toang...
- Câu ngắn giữ ngắn, impact mạnh
- TRÁNH: dịch kiểu sách giáo khoa, từ Hán Việt nhiều, thêm thắt lê thê{style_text}

Input (JSON - các trang liên tiếp):
{json.dumps(pages_texts, ensure_ascii=False, indent=2)}

Trả về ĐÚNG JSON object với cấu trúc GIỐNG HỆT, đã dịch. Không giải thích."""

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
