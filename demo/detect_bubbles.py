import torch.serialization
from ultralytics import YOLO

def detect_bubbles(model_path, image_path):
    """
    Detects bubbles in an image using a YOLOv8 model.
    Args:
        model_path (str): The file path to the YOLO model.
        image_path (str): The file path to the input image.
    Returns:
        list: A list containing the coordinates, score, and class_id of 
              the detected bubbles.
    """
    with torch.serialization.safe_globals([YOLO]):
        model = YOLO(model_path)

    bubbles = model(image_path)[0]

    return bubbles.boxes.data.tolist()
