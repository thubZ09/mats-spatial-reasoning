# src/mats/perturbations.py
import re
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def ensure_pil_image(image):
    """A simple utility to ensure the input is a PIL Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Expected a PIL Image, but got {type(image)}")

def perturb_spatial_words(sentence):
    """Perturbs spatial words in a sentence by swapping opposites."""
    # Use a temporary placeholder to avoid double-swapping (e.g., left -> right -> left)
    substitutions = [
        (r'\bleft\b', '__RIGHT__'), (r'\bright\b', '__LEFT__'),
        (r'\babove\b', '__BELOW__'), (r'\bbelow\b', '__ABOVE__'),
        (r'\btop\b', '__BOTTOM__'), (r'\bbottom\b', '__TOP__'),
        (r'\bin front of\b', '__BEHIND__'), (r'\bbehind\b', '__IN_FRONT_OF__'),
        (r'\bfront\b', '__BACK__'), (r'\bback\b', '__FRONT__')
    ]
    for pattern, placeholder in substitutions:
        sentence = re.sub(pattern, placeholder, sentence, flags=re.IGNORECASE)
    
    # Final replacements
    sentence = sentence.replace('__RIGHT__', 'right').replace('__LEFT__', 'left')
    sentence = sentence.replace('__BELOW__', 'below').replace('__ABOVE__', 'above')
    sentence = sentence.replace('__BOTTOM__', 'bottom').replace('__TOP__', 'top')
    sentence = sentence.replace('__BEHIND__', 'behind').replace('__IN_FRONT_OF__', 'in front of')
    sentence = sentence.replace('__BACK__', 'back').replace('__FRONT__', 'front')
    return sentence

def rotate_image(image, angle=15):
    """Rotate image by specified angle."""
    try:
        img = ensure_pil_image(image)
        return img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    except Exception as e:
        logger.error(f"Rotation failed: {e}")
        return image

def flip_image_horizontally(image):
    """Flip image left-to-right."""
    try:
        img = ensure_pil_image(image)
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    except Exception as e:
        logger.error(f"Flip failed: {e}")
        return image