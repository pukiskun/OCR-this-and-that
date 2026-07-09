import easyocr
import numpy as np
from utils import crop_field

class OCRExtractor:
    def __init__(self, languages=['en'], gpu=True):
        """
        Initializes the EasyOCR reader.
        By default, uses English ('en') and attempts to use GPU.
        """
        self.reader = easyocr.Reader(languages, gpu=gpu, verbose=False)

    def extract_fields(self, aligned_image: np.ndarray, template_data: dict) -> dict:
        """
        Iterates over the fields specified in template_data, crops them from aligned_image,
        runs OCR, and returns a dictionary mapping labels to extracted text.
        """
        results = {}
        fields = template_data.get("fields", [])
        
        for field in fields:
            label = field.get("label", "unnamed_field")
            
            # Crop the bounding box safely
            crop = crop_field(aligned_image, field)
            
            # Check if crop has positive dimensions
            if crop.size == 0:
                results[label] = ""
                continue
                
            # Perform OCR on the cropped segment
            # detail=0 returns only the text strings
            ocr_results = self.reader.readtext(crop, detail=0)
            
            # Join the text results with a space (in case the field contains multiple lines or words)
            text = " ".join(ocr_results).strip()
            results[label] = text
            
        return results
