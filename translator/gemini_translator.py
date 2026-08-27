"""
Gemini Translator with Batch Processing
Uses Gemini 3.1 Flash-Lite via the new google-genai SDK
Supports multiple source languages and custom prompts
"""
import json
import os
import re
import time
from typing import List, Dict

from .base import BaseTranslator

# Constants for retry logic
MAX_RETRIES = 3
RETRY_DELAY_BASE = 0.5  # Faster recovery: 0.5s → 1s → 2s
MAX_BATCH_SIZE = 30  # Max texts per single Gemini API call
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"


def _normalize_api_keys(api_key: str = None, api_keys=None) -> List[str]:
    raw_keys = []
    if api_keys:
        if isinstance(api_keys, str):
            raw_keys.extend(re.split(r"[\s,;]+", api_keys))
        else:
            raw_keys.extend(api_keys)
    if api_key:
        raw_keys.extend(re.split(r"[\s,;]+", api_key))

    seen = set()
    keys = []
    for key in raw_keys:
        key = str(key or "").strip()
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _looks_like_key_failure(error: Exception) -> bool:
    text = str(error).lower()
    key_failure_markers = (
        "401",
        "403",
        "429",
        "api key",
        "apikey",
        "auth",
        "exhausted",
        "forbidden",
        "invalid",
        "permission",
        "quota",
        "resource_exhausted",
        "unauthorized",
    )
    return any(marker in text for marker in key_failure_markers)


class GeminiTranslator(BaseTranslator):
    """
    Translator using Google Gemini 3.1 Flash-Lite.
    Supports batch translation to minimize API calls.
    """
    
    def __init__(
        self,
        api_key: str = None,
        api_keys=None,
        custom_prompt: str = None,
        style: str = "default",
        model_name: str = DEFAULT_GEMINI_MODEL,
        client_factory=None,
    ):
        super().__init__(custom_prompt=custom_prompt, style=style)

        env_keys = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY")
        self.api_keys = _normalize_api_keys(api_key=api_key, api_keys=api_keys or env_keys)
        if not self.api_keys:
            raise ValueError(
                "Gemini credentials required: set GEMINI_API_KEY(S) "
                "or pass api_key/api_keys."
            )

        self.api_key = self.api_keys[0]
        self._client_factory = client_factory
        self._clients = {}
        self._current_key_index = 0
        self.exhausted_api_keys = set()
        self.key_errors = {}
        self.last_warning = None
        self._all_keys_failed = False
        self.model_name = str(model_name or DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL

    def _default_client_factory(self, api_key):
        try:
            from google import genai
        except Exception as error:
            raise RuntimeError(
                "google-genai SDK is required. Install/update it with: "
                "python -m pip install --upgrade google-genai typing-extensions"
            ) from error

        return genai.Client(api_key=api_key)

    def _get_client(self, api_key):
        if api_key not in self._clients:
            factory = self._client_factory or self._default_client_factory
            self._clients[api_key] = factory(api_key)
        return self._clients[api_key]

    def _available_key_indices(self):
        total = len(self.api_keys)
        for offset in range(total):
            idx = (self._current_key_index + offset) % total
            if self.api_keys[idx] not in self.exhausted_api_keys:
                yield idx

    def _mark_key_exhausted(self, api_key, error):
        self.exhausted_api_keys.add(api_key)
        self.key_errors[api_key] = str(error)
        print(f"Gemini key failed, rotating to next key: {error}")

    def _generate_content_with_rotation(self, prompt):
        last_error = None

        for idx in list(self._available_key_indices()):
            api_key = self.api_keys[idx]
            client = self._get_client(api_key)

            for attempt in range(MAX_RETRIES):
                try:
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                    )
                    self._current_key_index = idx
                    self.api_key = api_key
                    self.last_warning = None
                    self._all_keys_failed = False
                    return response
                except Exception as e:
                    last_error = e
                    if _looks_like_key_failure(e):
                        self._mark_key_exhausted(api_key, e)
                        break

                    print(f"Gemini request attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_DELAY_BASE * (2 ** attempt)
                        print(f"Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        break

        self.last_warning = {"key": "backend.warn.geminiAllKeysFailed", "params": {}}
        self._all_keys_failed = True
        raise RuntimeError("All Gemini credentials failed or all requests failed") from last_error

    def translate_single(
        self, 
        text: str, 
        source: str = "ja", 
        target: str = "en",
        custom_prompt: str = None
    ) -> str:
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
            response = self._generate_content_with_rotation(prompt)
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
        if not texts:
            return []

        indexed_texts = [(i, t) for i, t in enumerate(texts) if t and t.strip()]

        if not indexed_texts:
            return texts

        texts_to_translate = [t for _, t in indexed_texts]

        # Chunk large batches to avoid timeouts and rate limits
        if len(texts_to_translate) > MAX_BATCH_SIZE:
            print(f"Chunking {len(texts_to_translate)} texts into batches of {MAX_BATCH_SIZE}")
            all_translations = []
            for chunk_start in range(0, len(texts_to_translate), MAX_BATCH_SIZE):
                chunk = texts_to_translate[chunk_start:chunk_start + MAX_BATCH_SIZE]
                chunk_translations = self._translate_batch_internal(chunk, source, target, custom_prompt)
                all_translations.extend(chunk_translations)
        else:
            all_translations = self._translate_batch_internal(texts_to_translate, source, target, custom_prompt)

        result = list(texts)
        for (orig_idx, _), trans in zip(indexed_texts, all_translations):
            result[orig_idx] = trans

        return result
    
    def _translate_batch_internal(
        self,
        texts_to_translate: List[str],
        source: str,
        target: str,
        custom_prompt: str = None
    ) -> List[str]:
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
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self._generate_content_with_rotation(prompt)
                result_text = response.text.strip()
                
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.startswith("```"):
                    result_text = result_text[3:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                result_text = result_text.strip()
                
                translations = json.loads(result_text)
                
                if len(translations) != len(texts_to_translate):
                    raise ValueError(f"Expected {len(texts_to_translate)} translations, got {len(translations)}")
                
                return translations
                
            except Exception as e:
                print(f"Gemini batch attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                
                if getattr(self, "_all_keys_failed", False):
                    print("All Gemini keys failed. Returning original texts.")
                    return texts_to_translate
                
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print("All retries failed, falling back to single translations")
                    return [self.translate_single(t, source, target) for t in texts_to_translate]
        
        return texts_to_translate
    
    def translate_pages_batch(
        self, 
        pages_texts: Dict[str, List[str]], 
        source: str = "ja", 
        target: str = "en",
        custom_prompt: str = None
    ) -> Dict[str, List[str]]:
        if not pages_texts:
            return {}
        
        source_name = self.LANG_NAMES.get(source, "Japanese")
        target_name = self.LANG_NAMES.get(target, "English")
        
        style = custom_prompt or self.custom_prompt
        style_text = f"\nStyle instructions: {style}" if style else ""
        
        prompt = f"""Bạn là chuyên gia dịch manga/comic từ {source_name} sang {target_name}.

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
            response = self._generate_content_with_rotation(prompt)
            result_text = response.text.strip()
            
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
            result = {}
            for page_name, texts in pages_texts.items():
                result[page_name] = self.translate_batch(texts, source, target)
            return result
