# src/mats/perturbations.py
import re
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# --- This swap dictionary is now more comprehensive ---
SWAP_DICTIONARY = {
    # Opposites
    r'\bleft\b': 'right', r'\bright\b': 'left',
    r'\babove\b': 'below', r'\bbelow\b': 'above',
    r'\btop\b': 'bottom', r'\bbottom\b': 'top',
    r'\bin front of\b': 'behind', r'\bbehind\b': 'in front of',
    r'\bfront\b': 'back', r'\bback\b': 'front',
    r'\binside\b': 'outside', r'\boutside\b': 'inside', # <-- ADDED
    r'\bnear\b': 'far from', r'\bfar from\b': 'near',     # <-- ADDED
    r'\bclose to\b': 'far from', r'\bfar from\b': 'close to', # <-- ADDED
    r'\bupper\b': 'lower', r'\blower\b': 'upper',
    r'\badjacent to\b': 'far from',  r'\bnext to\b': 'away from', 
    r'\bcloser to\b': 'farther from',
}

def ensure_pil_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Expected a PIL Image, but got {type(image)}")

def perturb_spatial_words(sentence: str) -> tuple[str, bool]:
    """
    Perturbs spatial words and returns the new sentence and a flag
    indicating if a swap was successfully made.
    """
    original_sentence = sentence
    
    # Use a temporary placeholder strategy to handle complex swaps
    for pattern, replacement in SWAP_DICTIONARY.items():
        # Create a temporary unique placeholder for the replacement
        placeholder = f"__{replacement.upper().replace(' ', '_')}__"
        sentence = re.sub(pattern, placeholder, sentence, flags=re.IGNORECASE)

    # Now, replace placeholders with final words
    for _, replacement in SWAP_DICTIONARY.items():
        placeholder = f"__{replacement.upper().replace(' ', '_')}__"
        sentence = sentence.replace(placeholder, replacement)
        
    was_changed = (original_sentence.lower() != sentence.lower())
    return sentence, was_changed

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