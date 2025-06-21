# In scripts/generate_clevr.py

import json
from PIL import Image, ImageDraw

def create_clevr_sample(relation):
    """Creates a synthetic CLEVR-style image and caption."""
    # ... (Your logic to draw shapes on a canvas) ...
    # Example: draw a red circle and a blue square
    # Return a dictionary with {'image': pil_image, 'caption': "...", 'relation_type': ...}

# ... (Loop to generate 500 samples and save them to a .jsonl file in data/clevr-rel/) ...