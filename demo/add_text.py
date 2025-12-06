from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
import cv2


def add_text(image, text, font_path, bubble_contour):
    """
    Add text inside a speech bubble contour.

    Args:
        image (numpy.ndarray): Processed bubble image (cv2 format - BGR).
        text (str): Text to be placed inside the speech bubble.
        font_path (str): Font path.
        bubble_contour (numpy.ndarray): Contour of the detected speech bubble.

    Returns:
        numpy.ndarray: Image with text placed inside the speech bubble.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    x, y, w, h = cv2.boundingRect(bubble_contour)

    line_height = 16
    font_size = 14
    wrapping_ratio = 0.075

    wrapped_text = textwrap.fill(text, width=int(w * wrapping_ratio), 
                                 break_long_words=True)
    
    font = ImageFont.truetype(font_path, size=font_size)

    lines = wrapped_text.split('\n')
    total_text_height = (len(lines)) * line_height

    while total_text_height > h:
        line_height -= 2
        font_size -= 2
        wrapping_ratio += 0.025

        wrapped_text = textwrap.fill(text, width=int(w * wrapping_ratio), 
                                 break_long_words=True)
                                 
        font = ImageFont.truetype(font_path, size=font_size)

        lines = wrapped_text.split('\n')
        total_text_height = (len(lines)) * line_height                         

    # Vertical centering
    text_y = y + (h - total_text_height) // 2

    for line in lines:
        text_length = draw.textlength(line, font=font)

        # Horizontal centering
        text_x = x + (w - text_length) // 2

        draw.text((text_x, text_y), line, font=font, fill=(0, 0, 0))

        text_y += line_height

    image[:, :, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image
