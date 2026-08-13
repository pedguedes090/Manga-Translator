"""
Local LLM Translator
Uses OpenAI-compatible API endpoints (Ollama, LM Studio, LocalAI, vLLM, Copilot-API, etc.)
"""
import requests
import json
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import BaseTranslator

MAX_BATCH_SIZE = 30  # Max texts per single LLM API call
MAX_PARALLEL_CHUNKS = 4  # Max concurrent chunk requests


class LocalLLMTranslator(BaseTranslator):
    """
    Translator using OpenAI-compatible local LLM servers.
    Works with Ollama, LM Studio, LocalAI, vLLM, Copilot-API, and similar servers.
    Communicates via /v1/chat/completions endpoint.
    """
    
    MODELS = [
        "gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.1-codex", "gpt-5.1-codex-mini",
        "gpt-5.1-codex-max", "gpt-5-codex", "gpt-4.1", "gpt-41-copilot",
        "gpt-4o", "gpt-4o-mini", "gpt-4o-2024-11-20", "gpt-4", "gpt-4-0125-preview",
        "gpt-3.5-turbo", "claude-sonnet-4.5", "claude-sonnet-4", "claude-opus-4.5",
        "claude-haiku-4.5", "gemini-3-pro-preview", "gemini-2.5-pro", "grok-code-fast-1",
    ]
    
    def __init__(self, server_url: str = "http://localhost:8080", model: str = "gpt-4o",
                 custom_prompt: str = None, style: str = "default"):
        super().__init__(custom_prompt=custom_prompt, style=style)
        self.base_url = server_url.rstrip("/")
        self.model = model
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        self._session = None

    def _get_session(self):
        """Get or create a persistent requests Session for connection pooling."""
        if self._session is None:
            self._session = requests.Session()
            # Configure connection pooling
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=5,
                pool_maxsize=10,
                max_retries=0,
            )
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
        return self._session

    def _extract_content(self, response: requests.Response) -> str:
        response.encoding = "utf-8"
        content_type = (response.headers.get("content-type") or "").lower()

        if "text/event-stream" not in content_type and not response.text.lstrip().startswith("data:"):
            data = response.json()
            choice = data["choices"][0]
            message = choice.get("message") or {}
            if "content" in message:
                return (message.get("content") or "").strip()
            return (choice.get("text") or "").strip()

        chunks = []
        raw_text = response.content.decode("utf-8", errors="replace")
        for line in raw_text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in data.get("choices", []):
                delta = choice.get("delta") or {}
                message = choice.get("message") or {}
                text = delta.get("content") or message.get("content") or choice.get("text")
                if text:
                    chunks.append(text)
        return "".join(chunks).strip()

    def _post_chat(self, prompt: str, timeout: int) -> str:
        session = self._get_session()
        response = session.post(
            self.endpoint,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "stream": False,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return self._extract_content(response)

    def _parse_json_array(self, result_text: str) -> list:
        result_text = result_text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()

        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            start = result_text.find("[")
            end = result_text.rfind("]")
            if start >= 0 and end > start:
                return json.loads(result_text[start:end + 1])
            raise
    
    def translate_single(self, text: str, source: str = "ja", target: str = "en") -> str:
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
            return self._post_chat(prompt, timeout=30)
        except Exception as e:
            print(f"Local LLM translation error: {e}")
            return text
    
    def translate_batch(self, texts: List[str], source: str = "ja", target: str = "en") -> List[str]:
        if not texts:
            return []

        indexed_texts = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not indexed_texts:
            return texts

        texts_to_translate = [t for _, t in indexed_texts]

        # Chunk large batches to avoid prompt-too-long and timeout issues
        if len(texts_to_translate) <= MAX_BATCH_SIZE:
            # Single batch — one request
            translations = self._translate_batch_internal(texts_to_translate, source, target)
        else:
            # Multiple chunks — parallel with max 4 concurrent
            chunks = []
            for chunk_start in range(0, len(texts_to_translate), MAX_BATCH_SIZE):
                chunk = texts_to_translate[chunk_start:chunk_start + MAX_BATCH_SIZE]
                chunks.append((chunk_start, chunk))

            print(f"Chunking {len(texts_to_translate)} texts into {len(chunks)} batches of {MAX_BATCH_SIZE}, max {MAX_PARALLEL_CHUNKS} parallel")

            # Parallel chunk processing
            chunk_results = {}
            with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_CHUNKS, len(chunks))) as executor:
                futures = {}
                for chunk_start, chunk in chunks:
                    future = executor.submit(self._translate_batch_internal, chunk, source, target)
                    futures[future] = chunk_start

                for future in as_completed(futures):
                    chunk_start = futures[future]
                    try:
                        chunk_results[chunk_start] = future.result()
                    except Exception as e:
                        print(f"Chunk starting at {chunk_start} failed: {e}, falling back to single")
                        # Fallback: translate each text in this chunk individually
                        failed_chunk = texts_to_translate[chunk_start:chunk_start + MAX_BATCH_SIZE]
                        chunk_results[chunk_start] = [self.translate_single(t, source, target) for t in failed_chunk]

            # Merge chunks in order
            translations = []
            for chunk_start, _ in chunks:
                translations.extend(chunk_results.get(chunk_start, texts_to_translate[chunk_start:chunk_start + MAX_BATCH_SIZE]))

        result = list(texts)
        for (orig_idx, _), trans in zip(indexed_texts, translations):
            result[orig_idx] = trans

        return result

    def _translate_batch_internal(self, texts_to_translate: List[str], source: str, target: str) -> List[str]:
        """Translate a single chunk of texts (called by translate_batch, possibly in parallel)."""
        if not texts_to_translate:
            return []

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
            result_text = self._post_chat(prompt, timeout=60)
            translations = self._parse_json_array(result_text)

            if len(translations) != len(texts_to_translate):
                print(f"Warning: Expected {len(texts_to_translate)} translations, got {len(translations)}. Falling back to single translations.")
                # Don't silently pad/truncate — fall back to individual translations for correctness
                return [self.translate_single(t, source, target) for t in texts_to_translate]

            return translations

        except Exception as e:
            print(f"Local LLM batch translation error: {e}")
            return [self.translate_single(t, source, target) for t in texts_to_translate]

    def test_connection(self) -> bool:
        try:
            session = self._get_session()
            response = session.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_available_models(self) -> List[str]:
        try:
            session = self._get_session()
            response = session.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                return [m["id"] for m in response.json().get("data", [])]
        except:
            pass
        return self.MODELS
