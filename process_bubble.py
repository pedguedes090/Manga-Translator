import cv2
import numpy as np


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

