import cv2
import numpy as np


def is_dark_bubble(image, threshold=100):
    """
    Determine if a bubble image is dark (black bubble with white text).
    
    Args:
        image: Input bubble image (BGR)
        threshold: Intensity threshold (below = dark bubble)
        
    Returns:
        bool: True if dark bubble, False if light bubble
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    return mean_intensity < threshold


def process_dark_bubble(image):
    """
    Processes a dark speech bubble (black with white text).
    Fills the bubble contents with black.
    
    Args:
        image (numpy.ndarray): Input dark bubble image.
        
    Returns:
        tuple: (processed_image, largest_contour)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # For dark bubbles, find the dark region (invert threshold)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        h, w = image.shape[:2]
        largest_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
        image[:] = (0, 0, 0)  # Fill with black
        return image, largest_contour
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)
    
    # Fill with black instead of white
    image[mask == 255] = (0, 0, 0)
    
    return image, largest_contour


def process_bubble(image):
    """
    Processes the speech bubble in the given image, making its contents white.

    Parameters:
    - image (numpy.ndarray): Input image.

    Returns:
    - image (numpy.ndarray):  Image with the speech bubble content set to white.
    - largest_contour (numpy.ndarray): Contour of the detected speech bubble (or None if not found).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Handle case when no contours found
    if not contours:
        # Return original image with a simple rectangular contour
        h, w = image.shape[:2]
        largest_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
        # Fill with white anyway
        image[:] = (255, 255, 255)
        return image, largest_contour
    
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)

    image[mask == 255] = (255, 255, 255)

    return image, largest_contour


def process_bubble_auto(image, force_dark=False):
    """
    Automatically detect bubble type and process accordingly.
    
    Args:
        image: Input bubble image (BGR)
        force_dark: If True, treat as dark bubble regardless of detection
        
    Returns:
        tuple: (processed_image, contour, is_dark)
    """
    if force_dark or is_dark_bubble(image):
        processed, contour = process_dark_bubble(image)
        return processed, contour, True
    else:
        processed, contour = process_bubble(image)
        return processed, contour, False
