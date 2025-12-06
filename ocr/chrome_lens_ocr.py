"""
Chrome Lens OCR module using chrome-lens-py library.
Provides OCR functionality using Google Lens API.
"""
import asyncio
from PIL import Image
import numpy as np

from chrome_lens_py import LensAPI


class ChromeLensOCR:
    """
    OCR engine using Google Chrome Lens API via chrome-lens-py.
    
    This provides an alternative to manga-ocr with the following benefits:
    - Free Google Lens OCR API
    - Multi-language support with auto-detection
    - Text block segmentation for comics/manga
    - Batch processing for faster multi-image OCR
    """
    
    def __init__(self, ocr_language: str = "ja"):
        """
        Initialize Chrome Lens OCR.
        
        Args:
            ocr_language: BCP 47 language code for OCR (default: "ja" for Japanese)
        """
        self.api = LensAPI()
        self.ocr_language = ocr_language
    
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
    
    async def _process(self, image) -> str:
        """
        Async method to process image with Chrome Lens API.
        
        Args:
            image: PIL Image, file path, or URL
            
        Returns:
            str: Extracted text
        """
        try:
            result = await self.api.process_image(
                image_path=image,
                ocr_language=self.ocr_language
            )
            return result.get("ocr_text", "")
        except Exception as e:
            print(f"Chrome Lens OCR error: {e}")
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
        Async batch processing using asyncio.gather for concurrent OCR.
        
        Args:
            images: List of PIL Images
            
        Returns:
            list: List of extracted texts
        """
        # Process all images concurrently
        tasks = [self._process(img) for img in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed = []
        for r in results:
            if isinstance(r, Exception):
                print(f"Batch OCR error: {r}")
                processed.append("")
            else:
                processed.append(r)
        
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

