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


def detect_bubbles(model_path, image_input):
    """
    Detects bubbles in an image using a YOLOv8 model.
    Automatically handles long vertical images (webtoons) by slicing.
    
    Args:
        model_path (str): The file path to the YOLO model.
        image_input: File path to image OR numpy array (BGR).

    Returns:
        list: A list containing the coordinates, score and class_id of 
              the detected bubbles.
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
    
    # Check if image needs slicing (long vertical image)
    if aspect_ratio > MAX_ASPECT_RATIO:
        print(f"Long image detected: {width}x{height} (ratio: {aspect_ratio:.1f})")
        
        # Try to find safe cut points (gutters)
        cut_points = find_safe_cut_points(image)
        
        if cut_points:
            print(f"Found {len(cut_points)} safe cut points (gutters)")
            return detect_bubbles_on_chunks(model, image, cut_points)
        else:
            # Fallback to overlap-based slicing
            return detect_bubbles_with_fallback(model, image)
    else:
        # Normal image - process directly
        bubbles = model(image, verbose=False)[0]
        return bubbles.boxes.data.tolist()


def clear_model_cache():
    """Clear the YOLO model cache to free memory."""
    global _yolo_model_cache
    _yolo_model_cache.clear()

