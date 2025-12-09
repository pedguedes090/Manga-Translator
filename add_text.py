from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
import cv2
import math

# Font cache to avoid reloading fonts from disk
_font_cache = {}

# Font sizing configuration
MIN_FONT_SIZE = 10
MAX_FONT_SIZE = 60
PADDING_RATIO = 0.1  # 10% padding inside bubble


def get_cached_font(font_path, size):
    """Get font from cache or load it."""
    cache_key = (font_path, size)
    if cache_key not in _font_cache:
        try:
            _font_cache[cache_key] = ImageFont.truetype(font_path, size=size)
        except:
            # Fallback to default font if custom font fails
            _font_cache[cache_key] = ImageFont.load_default()
    return _font_cache[cache_key]


def smart_wrap_text(text, chars_per_line):
    """
    Smart text wrapping that respects word boundaries.
    Avoids breaking Vietnamese words mid-character.
    
    Args:
        text: Text to wrap
        chars_per_line: Maximum characters per line
        
    Returns:
        Wrapped text with newlines
    """
    if not text or chars_per_line <= 0:
        return text
    
    # First try standard word-based wrapping (don't break words)
    wrapped = textwrap.fill(
        text, 
        width=chars_per_line, 
        break_long_words=False,  # Never break words mid-character!
        break_on_hyphens=False   # Don't break on hyphens
    )
    
    # If a single word is longer than the line, we need special handling
    lines = wrapped.split('\n')
    result_lines = []
    
    for line in lines:
        if len(line) <= chars_per_line:
            result_lines.append(line)
        else:
            # Line still too long (single long word) - break at space boundaries
            # For Vietnamese, try to break at spaces only
            words = line.split(' ')
            current_line = ""
            
            for word in words:
                if not current_line:
                    current_line = word
                elif len(current_line) + 1 + len(word) <= chars_per_line:
                    current_line += " " + word
                else:
                    if current_line:
                        result_lines.append(current_line)
                    current_line = word
            
            if current_line:
                result_lines.append(current_line)
    
    return '\n'.join(result_lines)


def calculate_optimal_font_size(text, w, h, font_path):
    """
    Calculate optimal font size to fill the bubble nicely.
    
    Args:
        text: Text to render
        w: Bubble width
        h: Bubble height
        font_path: Path to font file
        
    Returns:
        tuple: (font_size, line_height, wrapped_text, font)
    """
    # Apply padding
    usable_w = int(w * (1 - 2 * PADDING_RATIO))
    usable_h = int(h * (1 - 2 * PADDING_RATIO))
    
    if usable_w <= 0 or usable_h <= 0:
        return MIN_FONT_SIZE, MIN_FONT_SIZE, text, get_cached_font(font_path, MIN_FONT_SIZE)
    
    # Estimate initial font size based on bubble area and text length
    bubble_area = usable_w * usable_h
    char_count = max(len(text), 1)
    
    # Each character needs approximately (font_size * 0.6) * (font_size * 1.2) pixels
    # So font_size^2 * 0.72 ≈ area / char_count
    estimated_size = int(math.sqrt(bubble_area / (char_count * 0.8)))
    
    # Clamp to reasonable range
    font_size = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, estimated_size))
    
    # Binary search for optimal font size
    best_font_size = MIN_FONT_SIZE
    best_wrapped = text
    
    for size in range(font_size, MIN_FONT_SIZE - 1, -2):
        font = get_cached_font(font_path, size)
        line_height = int(size * 1.3)
        
        # Calculate characters per line based on font size
        avg_char_width = size * 0.6  # Approximate average character width
        chars_per_line = max(1, int(usable_w / avg_char_width))
        
        # Wrap text using smart wrapper (no word-breaking!)
        wrapped = smart_wrap_text(text, chars_per_line)
        lines = wrapped.split('\n')
        
        # Calculate total height needed
        total_height = len(lines) * line_height
        
        # Check if text fits
        if total_height <= usable_h:
            # Check if all lines fit width-wise
            fits_width = True
            for line in lines:
                try:
                    line_width = font.getlength(line)
                except:
                    line_width = len(line) * avg_char_width
                if line_width > usable_w:
                    fits_width = False
                    break
            
            if fits_width:
                best_font_size = size
                best_wrapped = wrapped
                break
    
    return best_font_size, int(best_font_size * 1.3), best_wrapped, get_cached_font(font_path, best_font_size)



def add_text(image, text, font_path, bubble_contour, text_color=(0, 0, 0)):
    """
    Add text inside a speech bubble contour with dynamic font sizing.

    Args:
        image (numpy.ndarray): Processed bubble image (cv2 format - BGR).
        text (str): Text to be placed inside the speech bubble.
        font_path (str): Font path.
        bubble_contour (numpy.ndarray): Contour of the detected speech bubble.
        text_color (tuple): RGB color for text. Default is black (0,0,0).
                           Use (255,255,255) for white text on dark bubbles.

    Returns:
        numpy.ndarray: Image with text placed inside the speech bubble.
    """
    if not text or not text.strip():
        return image
    
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    x, y, w, h = cv2.boundingRect(bubble_contour)
    
    # Calculate optimal font size
    font_size, line_height, wrapped_text, font = calculate_optimal_font_size(
        text, w, h, font_path
    )
    
    lines = wrapped_text.split('\n')
    total_text_height = len(lines) * line_height

    # Vertical centering
    text_y = y + (h - total_text_height) // 2

    for line in lines:
        try:
            text_length = font.getlength(line)
        except:
            text_length = len(line) * font_size * 0.6

        # Horizontal centering
        text_x = x + (w - text_length) // 2

        draw.text((text_x, text_y), line, font=font, fill=text_color)
        text_y += line_height

    image[:, :, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image

