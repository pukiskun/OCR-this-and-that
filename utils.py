import os
import json
import numpy as np

def load_template(template_path: str) -> dict:
    """
    Loads template configuration from a JSON file.
    Returns a dictionary. If the file does not exist, returns an empty dictionary structure.
    """
    if not os.path.exists(template_path):
        return {"reference_image_path": "", "width": 0, "height": 0, "fields": []}
    
    with open(template_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_template(template_path: str, data: dict) -> None:
    """
    Saves template configuration to a JSON file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(template_path)), exist_ok=True)
    with open(template_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def crop_field(image: np.ndarray, bbox: dict) -> np.ndarray:
    """
    Safely crops a bounding box field from a numpy image.
    bbox should be a dictionary with keys: 'x', 'y', 'w', 'h'.
    """
    h_img, w_img = image.shape[:2]
    
    # Extract & convert to integers
    x = int(round(bbox.get('x', 0)))
    y = int(round(bbox.get('y', 0)))
    w = int(round(bbox.get('w', 0)))
    h = int(round(bbox.get('h', 0)))
    
    # Clip coordinates to image boundaries
    x1 = max(0, min(w_img, x))
    y1 = max(0, min(h_img, y))
    x2 = max(0, min(w_img, x + w))
    y2 = max(0, min(h_img, y + h))
    
    # Return cropped region if valid, otherwise empty array
    if x2 > x1 and y2 > y1:
        return image[y1:y2, x1:x2]
    else:
        # Return a 1x1 empty pixel with same channel depth if possible
        if len(image.shape) == 3:
            return np.zeros((1, 1, image.shape[2]), dtype=image.dtype)
        return np.zeros((1, 1), dtype=image.dtype)
