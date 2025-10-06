"""
NOTE: EXPERIMENTAL/NOT USED FOR THE PAPER
"""

import torch
import json
from transformers import pipeline
import logging
from typing import Optional, Union
from PIL import Image

logger = logging.getLogger(__name__)

class MapBuilder:
    def __init__(self, device: Optional[Union[int, str]] = None):
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        
        logger.info(f"initializing object detector on device {device}...")
        try:
            self.detector = pipeline(
                "object-detection", 
                model="hustvl/yolos-tiny", 
                device=device
            )
            logger.info("object detector loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load object detection model: {e}")
            self.detector = None
    
    def build_map(self, pil_img: Image.Image, threshold: float = 0.7) -> Optional[str]:
        """
        builds a cognitive map from an image using object detection  

        args:
            pil_img: PIL Image to process
            threshold: Confidence threshold for detections (0.0-1.0)            
        returns:
            JSON string of detected objects or None if failed
        """
        if not self.detector:
            logger.warning("Object detector not available")
            return json.dumps([])  
        
        try:
            detections = self.detector(pil_img)
            obj_list = [
                {
                    "label": obj["label"],
                    "confidence": round(obj["score"], 3),  #include confidence for debugging
                    "xy": [round(c, 2) for c in obj["box"].values()]  #keep original "xy" naming
                }
                for obj in detections if obj["score"] > threshold
            ]
            
            logger.debug(f"Detected {len(obj_list)} objects above threshold {threshold}")
            return json.dumps(obj_list, separators=(",", ":"))
            
        except Exception as e:
            logger.error(f"Error during map building: {e}")
            return json.dumps([])  #return empty list on error

#convenience function for backward compatibility
def build_map(pil_img: Image.Image, threshold: float = 0.7) -> Optional[str]:
    """legacy function wrapper for the original API"""
    builder = MapBuilder()
    return builder.build_map(pil_img, threshold)