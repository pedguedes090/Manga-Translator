"""
Font Analyzer - Analyze manga font style and match with available fonts
Uses Gemini Vision to directly select the best matching font from available options
"""
import google.generativeai as genai
import json
import os
from PIL import Image
import numpy as np
from typing import Optional, Dict, Any, List


class FontAnalyzer:
    """
    Analyzes font style from manga speech bubbles using Gemini Vision
    and directly selects the best matching font from available fonts.
    """
    
    # Available fonts with descriptions for Gemini to understand
    FONT_OPTIONS = {
        "animeace_": "Classic manga font, clean and readable, standard comic style",
        "mangat": "Standard manga font, similar to animeace, good readability",
        "arial": "Clean sans-serif, formal and professional",
        "Yuki-Arenzi": "Simple casual handwritten style",
        "Yuki-Burobu": "Bold brush strokes, dynamic action style, Japanese brush feel",
        "Yuki-CCMarianChurchlandJournal": "Journal/diary handwritten, personal feel",
        "Yuki-CDX Starstreak": "Dynamic sci-fi style, bold and futuristic",
        "Yuki-CHICKEN Pie": "Playful, chunky, cute comedy style",
        "Yuki-CrashLanding BB": "Heavy impact font, bold action/shouting style",
        "Yuki-Downhill Dive": "Dynamic sports/action font, energetic",
        "Yuki-Gingerline DEMO Regular": "Elegant flowing handwritten, romantic style",
        "Yuki-Gorrilaz_Story": "Grunge alternative style, rough edges",
        "Yuki-KG Only Angel": "Delicate feminine handwritten, soft romantic",
        "Yuki-LF SwandsHand": "Natural handwritten, casual personal",
        "Yuki-La Belle Aurore": "Elegant cursive, fancy romantic style",
        "Yuki-Little Cupcakes": "Cute kawaii style, bubbly and fun",
        "Yuki-Nagurigaki Crayon": "Crayon/childish handwritten, playful comedy",
        "Yuki-Ripsnort BB": "Heavy bold impact, action/shouting",
        "Yuki-Roasthink": "Modern clean sans-serif, general purpose",
        "Yuki-Screwball": "Comic style, funny and expressive",
        "Yuki-Shark Crash": "Aggressive dynamic, action manga style",
        "Yuki-Skulduggery": "Gothic dark style, horror/mystery",
        "Yuki-Superscratchy": "Scratchy rough handwritten, grungy feel",
        "Yuki-Tea And Oranges Regular": "Soft warm handwritten, gentle drama",
    }
    
    DEFAULT_FONT = "animeace_"
    
    def __init__(self, api_key: str = None):
        """Initialize with Gemini API key."""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY or pass api_key.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")
    
    def _image_to_pil(self, image) -> Image.Image:
        """Convert various image formats to PIL Image."""
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            import cv2
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _build_font_list_prompt(self) -> str:
        """Build the font options list for the prompt."""
        lines = []
        for font_name, description in self.FONT_OPTIONS.items():
            lines.append(f"- {font_name}: {description}")
        return "\n".join(lines)
    
    def analyze_and_match(self, bubble_image) -> str:
        """
        Analyze the font in the image and directly select the best matching font.
        
        Args:
            bubble_image: Speech bubble image (PIL, numpy array)
            
        Returns:
            Font name to use
        """
        try:
            pil_image = self._image_to_pil(bubble_image)
            print(f"[FontAnalyzer] Analyzing image size: {pil_image.size}")
            
            font_list = self._build_font_list_prompt()
            
            prompt = f"""Look at this manga/comic speech bubble image and analyze the text font style.

Then choose the BEST matching font from this list based on visual similarity:

{font_list}

Consider these factors when matching:
1. Font weight (thin, normal, bold, heavy)
2. Style (clean, handwritten, decorative, brush)
3. Mood/genre (action, comedy, romance, horror, drama, casual)
4. Overall visual feel

Return ONLY the font name (exactly as written above), nothing else.
Example response: Yuki-Burobu"""

            print("[FontAnalyzer] Sending request to Gemini Vision...")
            response = self.model.generate_content([prompt, pil_image])
            result = response.text.strip()
            
            print(f"[FontAnalyzer] Gemini raw response: '{result}'")
            
            # Clean up response
            result = result.replace('"', '').replace("'", "").strip()
            
            # Remove common prefixes that Gemini might add
            prefixes_to_remove = ["The best matching font is ", "Best match: ", "Font: ", "I recommend "]
            for prefix in prefixes_to_remove:
                if result.lower().startswith(prefix.lower()):
                    result = result[len(prefix):].strip()
            
            print(f"[FontAnalyzer] Cleaned response: '{result}'")
            
            # Validate the result is in our font list
            if result in self.FONT_OPTIONS:
                print(f"[FontAnalyzer] ✓ Matched: {result}")
                return result
            
            # Try to find partial match (case-insensitive)
            result_lower = result.lower()
            for font_name in self.FONT_OPTIONS.keys():
                if font_name.lower() == result_lower:
                    print(f"[FontAnalyzer] ✓ Matched (case-insensitive): {font_name}")
                    return font_name
                if font_name.lower() in result_lower or result_lower in font_name.lower():
                    print(f"[FontAnalyzer] ✓ Matched (partial): {font_name}")
                    return font_name
            
            print(f"[FontAnalyzer] ✗ Font not in list: '{result}', using default")
            return self.DEFAULT_FONT
            
        except Exception as e:
            print(f"[FontAnalyzer] ✗ Error: {e}")
            return self.DEFAULT_FONT

