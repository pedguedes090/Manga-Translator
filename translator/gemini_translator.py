"""
Gemini Translator with Batch Processing
Uses Gemini 2.5 Flash-Lite for cost-effective translation
Supports multiple source languages and custom prompts
"""
import google.generativeai as genai
import json
import os
import time
from typing import List, Dict, Optional, TYPE_CHECKING

from .base import BaseTranslator

if TYPE_CHECKING:
    from .context_memory import ContextMemory

# Constants for retry logic
MAX_RETRIES = 3
RETRY_DELAY_BASE = 0.5  # Faster recovery: 0.5s → 1s → 2s


class GeminiTranslator(BaseTranslator):
    """
    Translator using Google Gemini 2.5 Flash-Lite.
    Supports batch translation to minimize API calls.
    """
    
    def __init__(self, api_key: str = None, custom_prompt: str = None, style: str = "default"):
        """
        Initialize Gemini translator.
        
        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
            custom_prompt: Custom instructions for translation style.
            style: Preset style name from STYLE_PRESETS.
        """
        super().__init__(custom_prompt=custom_prompt, style=style)
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY or pass api_key.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")
        
        
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
        
        prompt = f"""Bạn là chuyên gia dịch manga/comic từ {source_name} sang {target_name}.

QUY TẮC DỊCH:
1. ĐÂY LÀ HỘI THOẠI NÓI - phải nghe tự nhiên như người thật nói chuyện
2. TUYỆT ĐỐI KHÔNG dịch word-by-word, phải diễn đạt lại theo cách người Việt nói
3. Giữ nguyên cảm xúc, tính cách nhân vật qua cách dùng từ

HƯỚNG DẪN CHO TIẾNG VIỆT:
- TÊN NHÂN VẬT: GIỮ NGUYÊN tên gốc, KHÔNG dịch nghĩa
  + Nhật: Tanaka, Yamato, Sakura (-san, -kun, -chan, senpai, sensei)
  + Hàn: Kim, Park, Lee, Hyun (sunbae, oppa, hyung, noona)
  + Trung: Lý, Trương, Vương (sư huynh, sư đệ, đại nhân)
  + Có thể Việt hóa nhẹ: Tanaka-san → anh Tanaka, sunbae → tiền bối
- Đại từ nhân xưng: chọn phù hợp với quan hệ (tao/mày, tôi/cậu, anh/em, ông/bà, con/mẹ...)
- Thán từ: dịch tự nhiên (くそ→Đ*t/Chết tiệt, やばい→Toang rồi, すごい→Đỉnh thật, なに→Cái gì)
- Câu ngắn giữ ngắn, đừng thêm thắt dài dòng
- Dùng từ lóng, khẩu ngữ phù hợp ngữ cảnh (oke, ngon, chill, tởm...)
- Câu cảm thán: ôi, trời ơi, ủa, hả, ê, này...
- TRÁNH: dịch kiểu sách giáo khoa, dùng từ Hán Việt quá nhiều, câu dài lê thê{style_text}

Input texts (JSON array - mỗi item là 1 bubble):
{json.dumps(texts_to_translate, ensure_ascii=False)}

IMPORTANT: Trả về ĐÚNG JSON array với bản dịch theo THỨ TỰ GIỐNG HỆT.
Format: ["bản dịch 1", "bản dịch 2", ...]"""
        
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
        context_memory: 'ContextMemory' = None
    ) -> Dict[str, List[str]]:
        """
        Translate texts from multiple pages in a single API call.
        Ideal for batch processing 10 manga pages at once.
        
        Args:
            pages_texts: Dict mapping page names to list of texts
            source: Source language code
            target: Target language code
            custom_prompt: Override custom prompt for this call
            context_memory: Optional ContextMemory object for consistent translation
            
        Returns:
            Dict with same structure but translated texts
        """
        if not pages_texts:
            return {}
        
        source_name = self.LANG_NAMES.get(source, "Japanese")
        target_name = self.LANG_NAMES.get(target, "English")
        
        style = custom_prompt or self.custom_prompt
        style_text = f"\nStyle instructions: {style}" if style else ""
        
        # Build context section from ContextMemory if provided
        context_section = ""
        if context_memory:
            context_section = context_memory.generate_context_prompt()
        
        prompt = f"""Bạn là chuyên gia dịch manga/comic từ {source_name} sang {target_name}.
{context_section}
Đây là các trang LIÊN TIẾP trong cùng 1 story. Giữ mạch truyện và giọng nhân vật nhất quán.

QUY TẮC DỊCH:
1. ĐÂY LÀ HỘI THOẠI NÓI - phải nghe tự nhiên như người thật nói chuyện
2. TUYỆT ĐỐI KHÔNG dịch word-by-word, phải diễn đạt lại theo cách người Việt nói
3. Mỗi nhân vật có giọng điệu riêng, giữ nhất quán xuyên suốt

HƯỚNG DẪN CHO TIẾNG VIỆT:
- TÊN NHÂN VẬT: GIỮ NGUYÊN tên gốc, KHÔNG dịch nghĩa
  + Nhật: Tanaka, Yamato, Sakura (-san, -kun, -chan, senpai, sensei)
  + Hàn: Kim, Park, Lee, Hyun (sunbae, oppa, hyung, noona)
  + Trung: Lý, Trương, Vương (sư huynh, sư đệ, đại nhân)
  + Việt hóa nhẹ: sunbae → tiền bối, sensei → thầy
- Đại từ nhân xưng: chọn phù hợp với quan hệ và giữ nhất quán
  + Bạn bè thân: tao/mày, tớ/cậu
  + Người yêu: anh/em, mình/bạn  
  + Người lạ/trang trọng: tôi/anh/chị
  + Gia đình: con/bố/mẹ/ông/bà
- Thán từ dịch tự nhiên:
  + くそ/チクショウ → Đ*t/Chết tiệt/Khốn kiếp
  + やばい → Toang rồi/Xong đời
  + すごい → Đỉnh thật/Bá đạo
  + なに/何 → Cái gì/Hả
  + 大丈夫 → Ổn mà/Không sao
- Câu ngắn giữ ngắn, impact mạnh
- Dùng khẩu ngữ tự nhiên: oke, ngon, tởm, đỉnh, toang, chill...
- TRÁNH: 
  + Dịch kiểu sách giáo khoa cứng nhắc
  + Dùng quá nhiều từ Hán Việt  
  + Thêm thắt dài dòng không cần thiết
  + Giữ nguyên cấu trúc câu gốc{style_text}

Input (JSON - các trang liên tiếp):
{json.dumps(pages_texts, ensure_ascii=False, indent=2)}

IMPORTANT: Trả về ĐÚNG JSON object với cấu trúc GIỐNG HỆT nhưng đã dịch.
Giữ nguyên tên page và thứ tự bubble. Không giải thích, không markdown."""

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
