# src/utils/data_loader.py

import os
import json
import requests
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

def create_synthetic_image(caption, size=(300, 300)):
    """Creates a synthetic image with a visual representation of the caption."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except ImportError:
        font = None
    draw.text((10, 10), caption, fill=(0, 0, 0), font=font)
    if "red ball" in caption and "blue box" in caption:
        if "left" in caption:
            draw.ellipse((50, 150, 100, 200), fill='red')
            draw.rectangle((200, 150, 250, 200), fill='blue')
        else:
            draw.ellipse((200, 150, 250, 200), fill='red')
            draw.rectangle((50, 150, 100, 200), fill='blue')
    return img

def load_vsr_dataset(split='test', num_samples=100):
    """
    Loads and processes the VSR dataset from local .jsonl files.
    Falls back to synthetic data if the local file is not found.
    """
    RELATION_MAPPING = {
        'left': ['left', 'left of', 'on the left of', 'at the left side of'],
        'right': ['right', 'right of', 'on the right of', 'at the right side of'],
        'above': ['above', 'over', 'on top of'],
        'below': ['below', 'under', 'underneath', 'beneath'],
        'front': ['in front of', 'front', 'ahead of'],
        'behind': ['behind', 'back', 'in back of', 'at the back of']
    }
    ALL_VALID_RELATIONS = {alt: main for main, alts in RELATION_MAPPING.items() for alt in alts}

    dataset = []
    stats = {'lines_read': 0, 'skipped_relations': 0, 'image_failures': 0}
    file_path = f"{split}.jsonl"

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}. Falling back to synthetic data.")
        return [
            {'image': create_synthetic_image("A red ball is to the left of a blue box"), 'caption': "A red ball is to the left of a blue box", 'relation_type': 'left'},
            {'image': create_synthetic_image("A cat is sitting above the mat"), 'caption': "A cat is sitting above the mat", 'relation_type': 'above'}
        ]

    try:
        logger.info(f"Loading VSR from local file: {file_path}...")
        with open(file_path, 'r') as f:
            for line in f:
                stats['lines_read'] += 1
                if len(dataset) >= num_samples:
                    break
                try:
                    item = json.loads(line)
                    raw_relation = item.get('relation', '').lower().strip()
                    relation = ALL_VALID_RELATIONS.get(raw_relation)
                    if not relation:
                        stats['skipped_relations'] += 1
                        continue
                    
                    image_filename = item['image']
                    image_url = f"http://images.cocodataset.org/train2014/COCO_train2014_{image_filename}"
                    
                    try:
                        response = requests.get(image_url, timeout=15)
                        response.raise_for_status()
                        image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    except Exception as e:
                        stats['image_failures'] += 1
                        logger.debug(f"Image download failed for {image_url}: {e}")
                        continue
                    
                    dataset.append({
                        "image": image,
                        "caption": item['caption'],
                        "relation_type": relation
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed line #{stats['lines_read']}: {e}")

        logger.info(f"Loaded {len(dataset)} valid samples after reading {stats['lines_read']} lines.")
        logger.info(f"Stats: Skipped Relations={stats['skipped_relations']}, Image Failures={stats['image_failures']}")
        return dataset
    except Exception as e:
        logger.error(f"Critical error during file processing: {e}")
        return []

# You can add load_clevr_dataset() here in the future