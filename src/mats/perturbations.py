# src/mats/perturbations.py
import re
from PIL import Image
import logging
import json

logger = logging.getLogger(__name__)

SWAP_DICTIONARY = {
    r'\bleft\b': 'right', r'\bright\b': 'left',
    r'\babove\b': 'below', r'\bbelow\b': 'above',
    r'\btop\b': 'bottom', r'\bbottom\b': 'top',
    r'\bin front of\b': 'behind', r'\bbehind\b': 'in front of',
    r'\bfront\b': 'back', r'\bback\b': 'front',
    r'\binside\b': 'outside', r'\boutside\b': 'inside',
    r'\bnear\b': 'far from', r'\bfar from\b': 'near',
    r'\bclose to\b': 'far from', r'\bfar from\b': 'close to',
    r'\bnext to\b': 'away from', r'\badjacent to\b': 'away from',
    r'\btouching\b': 'separated from', r'\battached to\b': 'detached from',
    r'\bon\b': 'under', r'\bunder\b': 'over',
    r'\bupper\b': 'lower', r'\blower\b': 'upper',
    r'\bcloser to\b': 'farther from', r'\bfarther from\b': 'closer to',
    r'\bwithin\b': 'outside', r'\bcontained in\b': 'outside of',
    r'\bsurrounding\b': 'surrounded by', r'\bsurrounded by\b': 'surrounding',
    r'\bover\b': 'under', r'\bunderneath\b': 'on top of',
    r'\bupon\b': 'beneath', r'\bbeside\b': 'away from',
    r'\bat the left side of\b': 'at the right side of',
    r'\bat the right side of\b': 'at the left side of',
}

def ensure_pil_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Expected a PIL Image, but got {type(image)}")

def perturb_spatial_words(sentence: str) -> tuple[str, bool]:
    """
    perturbs spatial words in a sentence and returns both the modified sentence
    and a flag indicating if any changes were made
    """
    original = sentence
    
    #replace all matches with unique placeholders
    placeholders = {}
    for pattern, replacement in SWAP_DICTIONARY.items():
        placeholder = f"__{replacement.upper().replace(' ', '_')}__"
        sentence, count = re.subn(pattern, placeholder, sentence, flags=re.IGNORECASE)
        if count > 0:
            placeholders[placeholder] = replacement
    
    #replace placeholders with final values
    for placeholder, replacement in placeholders.items():
        sentence = sentence.replace(placeholder, replacement)
    
    #check if any changes were made
    was_changed = (original.lower() != sentence.lower())
    
    if was_changed:
        logger.debug(f"Perturbed: '{original}' → '{sentence}'")
    else:
        logger.warning(f"No spatial terms found in: '{original}'")
    
    return sentence, was_changed

def rotate_image(image, angle=15):
    """rotate image by specified angle"""
    try:
        img = ensure_pil_image(image)
        return img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    except Exception as e:
        logger.error(f"Rotation failed: {e}")
        return image

def flip_image_horizontally(image):
    """flip image left-to-right"""
    try:
        img = ensure_pil_image(image)
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    except Exception as e:
        logger.error(f"Flip failed: {e}")
        return image

def generate_cognitive_map(caption: str) -> str:
    """
    creates a simple, rule-based cognitive map from a VSR caption.
    this map represents a "perfect understanding" of the caption's statement
    example: "The cat is to the left of the dog." -> places cat at x=4, dog at x=6
    """
    try:
        #a simple parser for "The [obj1] is [relation] the [obj2]"
        parts = caption.replace('.', '').lower().split()
        if len(parts) < 5: return None

        obj1 = parts[1]
        obj2 = parts[-1]
        relation = " ".join(parts[2:-1])

        map_data = {"objects": []}
        #define positions based on the relation
        if "left" in relation:
            map_data["objects"].append({"name": obj1, "position": {"x": 0.25, "y": 0.5}})
            map_data["objects"].append({"name": obj2, "position": {"x": 0.75, "y": 0.5}})
        elif "right" in relation:
            map_data["objects"].append({"name": obj1, "position": {"x": 0.75, "y": 0.5}})
            map_data["objects"].append({"name": obj2, "position": {"x": 0.25, "y": 0.5}})
        elif "above" in relation or "on" in relation:
            map_data["objects"].append({"name": obj1, "position": {"x": 0.5, "y": 0.25}})
            map_data["objects"].append({"name": obj2, "position": {"x": 0.5, "y": 0.75}})
        elif "below" in relation or "under" in relation:
            map_data["objects"].append({"name": obj1, "position": {"x": 0.5, "y": 0.75}})
            map_data["objects"].append({"name": obj2, "position": {"x": 0.5, "y": 0.25}})
        else:
            #for other relations, just place them apart as a default
            map_data["objects"].append({"name": obj1, "position": {"x": 0.25, "y": 0.5}})
            map_data["objects"].append({"name": obj2, "position": {"x": 0.75, "y": 0.5}})

        return json.dumps(map_data, indent=2)
    except Exception:
        #if the caption format is unexpected, we can't create a map.
        return None