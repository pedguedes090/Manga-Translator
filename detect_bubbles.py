from ultralytics import YOLO
import numpy as np
import cv2

# Global cache for YOLO models to avoid reloading on every call
_yolo_model_cache = {}

# Configuration for long image handling
MAX_ASPECT_RATIO = 3.0  # When height/width > 3, start slicing
MIN_CHUNK_HEIGHT = 800  # Minimum chunk height in pixels
MAX_CHUNK_HEIGHT = 1500  # Target chunk height
GUTTER_MIN_HEIGHT = 10  # Minimum gutter height to consider valid
OVERLAP_SIZE = 200  # Fallback overlap if no gutter found
WHITE_THRESHOLD = 245  # Pixel value to consider "white"
BLACK_THRESHOLD = 15   # Pixel value to consider "black"
IOU_THRESHOLD = 0.5    # For removing duplicate detections

# Black bubble detection constants
BLACK_BUBBLE_THRESHOLD = 50  # Max intensity for black regions
BLACK_BUBBLE_MIN_AREA = 1000  # Minimum area in pixels
BLACK_BUBBLE_MAX_AREA_RATIO = 0.4  # Maximum bubble area relative to image
BLACK_BUBBLE_MIN_ASPECT = 0.2  # Minimum width/height ratio
BLACK_BUBBLE_MAX_ASPECT = 5.0  # Maximum width/height ratio


def detect_black_bubbles(image, min_area=None, max_area_ratio=None):
    """
    Detect black speech bubbles using OpenCV contour detection.
    Used as fallback when YOLO doesn't detect dark bubbles.
    
    Args:
        image: Input image (numpy array, BGR)
        min_area: Minimum bubble area in pixels (default: BLACK_BUBBLE_MIN_AREA)
        max_area_ratio: Maximum bubble area as ratio of image (default: BLACK_BUBBLE_MAX_AREA_RATIO)
        
    Returns:
        list: Detections in format [x1, y1, x2, y2, confidence, class_id]
    """
    if min_area is None:
        min_area = BLACK_BUBBLE_MIN_AREA
    if max_area_ratio is None:
        max_area_ratio = BLACK_BUBBLE_MAX_AREA_RATIO
    
    height, width = image.shape[:2]
    max_area = int(width * height * max_area_ratio)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find dark regions (invert threshold to get black areas)
    _, thresh = cv2.threshold(gray, BLACK_BUBBLE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio (bubbles are usually somewhat round/oval)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < BLACK_BUBBLE_MIN_ASPECT or aspect_ratio > BLACK_BUBBLE_MAX_ASPECT:
            continue
        
        # Filter: bubble should be mostly filled (not just a thin border)
        rect_area = w * h
        fill_ratio = area / rect_area if rect_area > 0 else 0
        if fill_ratio < 0.3:  # At least 30% filled
            continue
        
        # Check if region is actually dark (verify it's a black bubble)
        roi = gray[y:y+h, x:x+w]
        mean_intensity = np.mean(roi)
        if mean_intensity > BLACK_BUBBLE_THRESHOLD + 30:  # Allow some tolerance
            continue
        
        # Calculate confidence based on fill ratio and darkness
        confidence = min(0.8, fill_ratio * (1 - mean_intensity / 255))
        
        x1, y1, x2, y2 = x, y, x + w, y + h
        detections.append([x1, y1, x2, y2, confidence, 0])  # class_id=0 for speech bubble
    
    return detections


def find_safe_cut_points(image, target_height=MAX_CHUNK_HEIGHT):
    """
    Find safe places to cut the image (white/black gutters between panels).
    
    Args:
        image: Input image (numpy array, BGR)
        target_height: Approximate target height for each chunk
        
    Returns:
        list: List of y-coordinates where it's safe to cut
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate mean intensity for each row
    row_means = np.mean(gray, axis=1)
    
    # Find rows that are mostly white or mostly black (gutters)
    is_gutter = (row_means > WHITE_THRESHOLD) | (row_means < BLACK_THRESHOLD)
    
    # Find continuous gutter regions
    gutter_regions = []
    start = None
    
    for i, is_gut in enumerate(is_gutter):
        if is_gut and start is None:
            start = i
        elif not is_gut and start is not None:
            if i - start >= GUTTER_MIN_HEIGHT:  # Only valid gutters
                gutter_regions.append((start, i, (start + i) // 2))  # start, end, center
            start = None
    
    # Handle gutter at the end
    if start is not None and height - start >= GUTTER_MIN_HEIGHT:
        gutter_regions.append((start, height, (start + height) // 2))
    
    if not gutter_regions:
        return []
    
    # Select cut points at approximately target_height intervals
    cut_points = []
    last_cut = 0
    
    for start, end, center in gutter_regions:
        # Check if this gutter is far enough from last cut
        if center - last_cut >= MIN_CHUNK_HEIGHT:
            # Check if we should cut here (approaching target height)
            if center - last_cut >= target_height * 0.7:
                cut_points.append(center)
                last_cut = center
    
    return cut_points


def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def remove_duplicate_detections(detections, iou_threshold=IOU_THRESHOLD):
    """Remove duplicate detections based on IoU, keeping higher confidence ones."""
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (index 4) descending
    sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)
    
    keep = []
    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)
        
        # Remove detections with high IoU
        sorted_dets = [
            det for det in sorted_dets 
            if calculate_iou(best, det) < iou_threshold
        ]
    
    return keep


def detect_bubbles_on_chunks(model, image, cut_points):
    """
    Detect bubbles on image chunks and merge results.
    
    Args:
        model: Loaded YOLO model
        image: Full image (numpy array)
        cut_points: List of y-coordinates to cut at
        
    Returns:
        list: Merged bubble detections with adjusted coordinates
    """
    height = image.shape[0]
    all_detections = []
    
    # Create chunk boundaries
    boundaries = [0] + cut_points + [height]
    
    print(f"Processing image in {len(boundaries) - 1} chunks...")
    
    for i in range(len(boundaries) - 1):
        y_start = boundaries[i]
        y_end = boundaries[i + 1]
        
        chunk = image[y_start:y_end]
        
        # Skip very small chunks
        if chunk.shape[0] < 50:
            continue
        
        # Detect bubbles in chunk
        results = model(chunk, verbose=False)[0]
        chunk_detections = results.boxes.data.tolist()
        
        # Adjust y-coordinates to original image space
        for det in chunk_detections:
            det[1] += y_start  # y1
            det[3] += y_start  # y2
            all_detections.append(det)
        
        print(f"  Chunk {i+1}: y={y_start}-{y_end}, found {len(chunk_detections)} bubbles")
    
    # Remove duplicates from overlapping regions
    merged = remove_duplicate_detections(all_detections)
    print(f"Total: {len(all_detections)} detections → {len(merged)} after merge")
    
    return merged


def detect_bubbles_with_fallback(model, image):
    """
    Detect bubbles using overlap-based slicing when no gutters found.
    
    Args:
        model: Loaded YOLO model
        image: Full image (numpy array)
        
    Returns:
        list: Merged bubble detections
    """
    height = image.shape[0]
    all_detections = []
    
    # Calculate chunks with overlap
    chunk_height = MAX_CHUNK_HEIGHT
    overlap = OVERLAP_SIZE
    
    y = 0
    chunk_num = 0
    
    print(f"No gutters found. Using overlap-based slicing...")
    
    while y < height:
        y_end = min(y + chunk_height, height)
        chunk = image[y:y_end]
        
        if chunk.shape[0] < 50:
            break
        
        # Detect bubbles
        results = model(chunk, verbose=False)[0]
        chunk_detections = results.boxes.data.tolist()
        
        # Adjust coordinates
        for det in chunk_detections:
            det[1] += y
            det[3] += y
            all_detections.append(det)
        
        chunk_num += 1
        print(f"  Chunk {chunk_num}: y={y}-{y_end}, found {len(chunk_detections)} bubbles")
        
        # Move to next chunk with overlap
        y = y_end - overlap
        if y_end >= height:
            break
    
    # Remove duplicates
    merged = remove_duplicate_detections(all_detections)
    print(f"Total: {len(all_detections)} detections → {len(merged)} after merge")
    
    return merged


def detect_bubbles(model_path, image_input, enable_black_bubble=True):
    """
    Detects bubbles in an image using a YOLOv8 model.
    Also detects black speech bubbles using OpenCV fallback (optional).
    Automatically handles long vertical images (webtoons) by slicing.
    
    Args:
        model_path (str): The file path to the YOLO model.
        image_input: File path to image OR numpy array (BGR).
        enable_black_bubble (bool): Whether to detect black bubbles using OpenCV.

    Returns:
        list: A list containing the coordinates, score and class_id of 
              the detected bubbles. Each detection also includes is_dark_bubble flag.
    """
    global _yolo_model_cache
    
    # Cache model to avoid reloading (~2-5s savings per image)
    if model_path not in _yolo_model_cache:
        print(f"Loading YOLO model from {model_path}...")
        _yolo_model_cache[model_path] = YOLO(model_path)
        print("YOLO model loaded and cached!")
    
    model = _yolo_model_cache[model_path]
    
    # Load image if path is provided
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    else:
        image = image_input
    
    if image is None:
        return []
    
    height, width = image.shape[:2]
    aspect_ratio = height / width
    
    # Get YOLO detections
    if aspect_ratio > MAX_ASPECT_RATIO:
        print(f"Long image detected: {width}x{height} (ratio: {aspect_ratio:.1f})")
        
        # Try to find safe cut points (gutters)
        cut_points = find_safe_cut_points(image)
        
        if cut_points:
            print(f"Found {len(cut_points)} safe cut points (gutters)")
            yolo_detections = detect_bubbles_on_chunks(model, image, cut_points)
        else:
            # Fallback to overlap-based slicing
            yolo_detections = detect_bubbles_with_fallback(model, image)
    else:
        # Normal image - process directly
        bubbles = model(image, verbose=False)[0]
        yolo_detections = bubbles.boxes.data.tolist()
    
    # Get black bubble detections using OpenCV (if enabled)
    if enable_black_bubble:
        black_bubble_detections = detect_black_bubbles(image)
    else:
        black_bubble_detections = []
    
    if black_bubble_detections:
        print(f"OpenCV found {len(black_bubble_detections)} potential black bubbles")
        
        # Mark black bubbles with a flag (append 1 to detection)
        for det in black_bubble_detections:
            det.append(1)  # is_dark_bubble = 1
        
        # Mark YOLO detections as normal bubbles
        for det in yolo_detections:
            if len(det) == 6:  # Only if not already marked
                det.append(0)  # is_dark_bubble = 0
        
        # Merge all detections and remove duplicates
        all_detections = yolo_detections + black_bubble_detections
        merged = remove_duplicate_detections(all_detections)
        
        print(f"Total: {len(yolo_detections)} YOLO + {len(black_bubble_detections)} black = {len(merged)} after merge")
        return merged
    else:
        # No black bubbles found, return YOLO only (add is_dark_bubble=0)
        for det in yolo_detections:
            if len(det) == 6:
                det.append(0)
        return yolo_detections


def clear_model_cache():
    """Clear the YOLO model cache to free memory."""
    global _yolo_model_cache
    _yolo_model_cache.clear()

