from datasets import load_dataset
from PIL import Image
import io
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

SPATIAL_KEYWORDS = ['left', 'right', 'above', 'below', 'top', 'bottom', 'in front of', 'behind', 'front', 'back']

def get_relation_type(caption):
    """Finds the first spatial keyword in a caption."""
    caption_lower = caption.lower()
    for keyword in SPATIAL_KEYWORDS:
        if keyword in caption_lower:
            return keyword
    return 'non-spatial'

def process_and_filter_dataset(num_samples=100):
    """
    Loads, processes, and filters the VSR dataset.

    Args:
        num_samples (int): The number of samples to load and process.

    Returns:
        A list of processed data items.
    """
    dataset_name = "juletxara/visual-spatial-reasoning"
    config_name = "random"  # This dataset has a 'random' and 'zeroshot' configuration
    split = "test"

    try:
        # Load the dataset from Hugging Face
        logger.info(f"Loading dataset '{dataset_name}' (config: '{config_name}') with split '{split}'...")
        # Use streaming=True and trust_remote_code=True for robustness
        full_dataset = load_dataset(dataset_name, config_name, split=split, streaming=True, trust_remote_code=True)
        logger.info("Dataset loaded successfully.")

        processed_data = []
        count = 0
        
        print("Filtering for spatial captions and processing images...")
        for item in full_dataset:
            if count >= num_samples:
                break

            # Ensure item is treated as a dictionary
            if isinstance(item, dict):
                caption = item.get('caption')
                image_data = item.get('image')
                image_url = item.get('image_link')
                label = item.get('label')
            else:
                # If item is not a dict, skip it
                logger.warning(f"Skipping non-dictionary item: {type(item)}")
                continue

            if not caption or not image_url:
                continue # Skip items with missing data

            # Filter for captions that contain our spatial keywords
            relation_type = get_relation_type(caption)
            if relation_type == 'non-spatial':
                continue # Skip non-spatial captions

            try:
                # Download the image from the URL
                import requests
                response = requests.get(image_url, stream=True)
                response.raise_for_status() # Raise an exception for bad status codes
                image = Image.open(response.raw).convert("RGB")
                
                processed_data.append({
                    'image': image,
                    'caption': caption,
                    'label': label,
                    'relation_type': relation_type
                })
                count += 1

            except Exception as e:
                logger.warning(f"Skipping item due to image download/processing error: {e}")
                continue

        logger.info(f"Processed and filtered {len(processed_data)} spatial reasoning samples.")
        return processed_data

    except Exception as e:
        logger.error(f"Failed to load or process dataset '{dataset_name}': {e}")
        return [] # Return an empty list on failure