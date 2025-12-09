"""
Chrome Lens OCR module using chrome-lens-py library.
Provides OCR functionality using Google Lens API.
"""
import asyncio
import random
from PIL import Image
import numpy as np

from chrome_lens_py import LensAPI

# Limit concurrent requests to avoid overwhelming the API
# Balance: 15 is fast enough while reducing 502 errors
MAX_CONCURRENT_OCR = 10


class ChromeLensOCR:
    """
    OCR engine using Google Chrome Lens API via chrome-lens-py.
    
    This provides an alternative to manga-ocr with the following benefits:
    - Free Google Lens OCR API
    - Multi-language support with auto-detection
    - Text block segmentation for comics/manga
    - Batch processing for faster multi-image OCR
    """
    
    def __init__(self, ocr_language: str = "ja", max_concurrent: int = MAX_CONCURRENT_OCR):
        """
        Initialize Chrome Lens OCR.
        
        Args:
            ocr_language: BCP 47 language code for OCR (default: "ja" for Japanese)
            max_concurrent: Maximum concurrent OCR requests (default: 10)
        """
        self.api = LensAPI()
        self.ocr_language = ocr_language
        self.max_concurrent = max_concurrent
        self._semaphore = None  # Created lazily when needed
    
    def __call__(self, image) -> str:
        """
        Process an image and extract text.
        
        Args:
            image: Can be a PIL Image, numpy array, file path, or URL
            
        Returns:
            str: Extracted text from the image
        """
        # Handle different image input types
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(image)
        
        # Use cached event loop to avoid overhead
        try:
            loop = asyncio.get_running_loop()
            # If there's a running loop, use run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(self._process(image), loop)
            return future.result(timeout=30)
        except RuntimeError:
            # No running loop, create one (but try to reuse)
            if not hasattr(self, '_loop') or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop.run_until_complete(self._process(image))
    
    async def _process(self, image, max_retries: int = 5) -> str:
        """
        Async method to process image with Chrome Lens API.
        Includes retry logic with exponential backoff + jitter for server errors.
        Uses semaphore to limit concurrent requests.
        
        Args:
            image: PIL Image, file path, or URL
            max_retries: Maximum number of retry attempts for server errors
            
        Returns:
            str: Extracted text
        """
        # Create semaphore lazily (must be done in async context)
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        last_error = None
        
        # Use semaphore to limit concurrent requests
        async with self._semaphore:
            for attempt in range(max_retries):
                try:
                    # Add small random delay before each request to spread load
                    if attempt == 0:
                        await asyncio.sleep(random.uniform(0.1, 0.5))
                    
                    result = await self.api.process_image(
                        image_path=image,
                        ocr_language=self.ocr_language
                    )
                    return result.get("ocr_text", "")
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    
                    # Check if it's a retryable server error (502, 503, 504, 429)
                    is_server_error = any(code in error_str for code in ['502', '503', '504', '429'])
                    
                    if is_server_error and attempt < max_retries - 1:
                        # Exponential backoff with jitter: 2-4s, 4-8s, 8-16s, 16-32s
                        base_wait = 2 ** (attempt + 1)
                        jitter = random.uniform(0, base_wait)
                        wait_time = base_wait + jitter
                        print(f"Server error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                    elif is_server_error:
                        print(f"Chrome Lens OCR failed after {max_retries} attempts: {e}")
                        return ""
                    else:
                        # Non-retryable error, fail immediately
                        print(f"Chrome Lens OCR error: {e}")
                        return ""
        
        print(f"Chrome Lens OCR error: {last_error}")
        return ""
    
    def process_batch(self, images: list) -> list:
        """
        Process multiple images concurrently for faster OCR.
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            list: List of extracted texts in same order
        """
        # Convert numpy arrays to PIL Images
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)
        
        # Run batch processing
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                self._process_batch(pil_images), loop
            )
            return future.result(timeout=120)
        except RuntimeError:
            if not hasattr(self, '_loop') or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop.run_until_complete(self._process_batch(pil_images))
    
    async def _process_batch(self, images: list) -> list:
        """
        Async batch processing with concurrency limiting.
        The semaphore in _process ensures only MAX_CONCURRENT_OCR requests run at once.
        
        Args:
            images: List of PIL Images
            
        Returns:
            list: List of extracted texts
        """
        print(f"Processing {len(images)} images with max {self.max_concurrent} concurrent requests...")
        
        # Create tasks - semaphore in _process limits actual concurrent execution
        tasks = [self._process(img) for img in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions with detailed logging
        processed = []
        success_count = 0
        empty_count = 0
        error_count = 0
        
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"  [Bubble {i+1}] ERROR: {r}")
                processed.append("")
                error_count += 1
            elif r:  # Non-empty result
                # Truncate long text for logging
                preview = r[:50].replace('\n', ' ') + ('...' if len(r) > 50 else '')
                print(f"  [Bubble {i+1}] OK: {preview}")
                processed.append(r)
                success_count += 1
            else:  # Empty result
                print(f"  [Bubble {i+1}] EMPTY (no text detected)")
                processed.append("")
                empty_count += 1
        
        print(f"\nOCR Summary:")
        print(f"  ✓ Success: {success_count}/{len(images)}")
        print(f"  ○ Empty (no text): {empty_count}/{len(images)}")
        print(f"  ✗ Errors: {error_count}/{len(images)}")
        print(f"OCR completed: {success_count}/{len(images)} successful")
        return processed
    
    async def process_with_blocks(self, image) -> dict:
        """
        Process image and return text segmented into blocks.
        Useful for manga/comics with multiple speech bubbles.
        
        Args:
            image: PIL Image, file path, or URL
            
        Returns:
            dict: Contains 'text_blocks' with segmented text and geometry
        """
        try:
            result = await self.api.process_image(
                image_path=image,
                ocr_language=self.ocr_language,
                output_format='blocks'
            )
            return result
        except Exception as e:
            print(f"Chrome Lens OCR error: {e}")
            return {"text_blocks": []}
    
    def get_text_blocks(self, image) -> list:
        """
        Synchronous wrapper to get text blocks from image.
        
        Args:
            image: PIL Image, numpy array, file path, or URL
            
        Returns:
            list: List of text blocks with text and geometry
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        result = asyncio.run(self.process_with_blocks(image))
        return result.get("text_blocks", [])
