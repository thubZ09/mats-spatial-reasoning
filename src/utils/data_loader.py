# src/utils/data_loader.py
import os
import json
import requests
from PIL import Image
import io
import logging
import zipfile
from datasets import load_dataset
from tqdm import tqdm
import tempfile

logger = logging.getLogger(__name__)

# Updated Dropbox URL for VSR images with direct download
VSR_IMAGE_BASE = "https://www.dropbox.com/scl/fi/efvlqxp4zhxfp60m1hujd/vsr_images.zip?rlkey=3w3d8dxbt7xgq64pyh7zosnzm&e=1&st=7ltz5qqh&dl=1"

def load_vsr_dataset(split='test', num_samples=100, image_base_path=None):
    """
    Loads VSR dataset with proper image handling
    :param split: 'train', 'validation', or 'test'
    :param num_samples: Number of samples to load
    :param image_base_path: Local path to downloaded VSR images (optional)
    """
    # First try Hugging Face dataset
    try:
        logger.info("Attempting to load from Hugging Face dataset")
        return load_vsr_from_hf(split, num_samples, image_base_path)
    except Exception as e:
        logger.warning(f"Hugging Face load failed: {e}")
    
    # Then try GitHub metadata
    try:
        logger.info("Falling back to GitHub metadata")
        return load_vsr_from_github(split, num_samples, image_base_path)
    except Exception as e:
        logger.error(f"GitHub load failed: {e}")
        return create_fallback_dataset(num_samples)

def load_vsr_from_hf(split, num_samples, image_base_path):
    """Load dataset from Hugging Face with proper image handling"""
    logger.info(f"Loading VSR {split} split from Hugging Face")
    dataset = load_dataset("cambridgeltl/vsr_random", split=split)
    
    samples = []
    valid_count = 0
    
    for item in tqdm(dataset, desc=f"Processing {split} split"):
        if valid_count >= num_samples:
            break
            
        # Only use true statements (label=1)
        if item.get('label', 1) != 1:
            continue
            
        try:
            image = get_vsr_image(
                image_id=item['image'],
                image_base_path=image_base_path
            )
            
            samples.append({
                "image": image,
                "caption": item['caption'],
                "relation_type": item['relation'].lower(),
                "image_id": item['image']
            })
            valid_count += 1
        except Exception as e:
            logger.warning(f"Failed to load image {item['image']}: {e}")
    
    logger.info(f"Loaded {len(samples)} samples from Hugging Face")
    return samples

def load_vsr_from_github(split, num_samples, image_base_path):
    """Fallback to GitHub metadata if Hugging Face fails"""
    logger.info(f"Loading VSR {split} split from GitHub")
    json_url = f"https://raw.githubusercontent.com/cambridgeltl/visual-spatial-reasoning/master/data/splits/random/{split}.jsonl"
    
    response = requests.get(json_url)
    response.raise_for_status()
    lines = response.text.splitlines()
    
    samples = []
    valid_count = 0
    
    for line in tqdm(lines, desc=f"Processing {split} split", total=min(num_samples, len(lines))):
        if valid_count >= num_samples:
            break
            
        try:
            item = json.loads(line)
            
            # Only use true statements (label=1)
            if item.get('label', 1) != 1:
                continue
                
            image = get_vsr_image(
                image_id=item['image'],
                image_base_path=image_base_path
            )
            
            samples.append({
                "image": image,
                "caption": item['caption'],
                "relation_type": item['relation'].lower(),
                "image_id": item['image']
            })
            valid_count += 1
        except Exception as e:
            logger.warning(f"Failed to process item: {e}")
    
    logger.info(f"Loaded {len(samples)} samples from GitHub")
    return samples

def get_vsr_image(image_id, image_base_path=None):
    """
    Load VSR image from local storage or remote sources
    :param image_id: COCO image ID (e.g., '000000391895.jpg')
    :param image_base_path: Local directory containing VSR images
    """
    # 1. First try local filesystem if path provided
    if image_base_path:
        local_path = os.path.join(image_base_path, image_id)
        if os.path.exists(local_path):
            return Image.open(local_path).convert("RGB")
    
    # 2. Try COCO repositories
    try:
        for prefix in ["train2017", "val2017"]:
            url = f"http://images.cocodataset.org/{prefix}/{image_id}"
            response = requests.get(url, stream=True, timeout=15)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        logger.debug(f"COCO download failed: {e}")
    
    # 3. Try Dropbox mirror as last resort
    try:
        response = requests.get(VSR_IMAGE_BASE, stream=True, timeout=30)
        response.raise_for_status()
        
        # Save zip to temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            zip_path = tmp_file.name
        
        # Extract specific image from zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open(f"vsr_images/{image_id}") as file_in_zip:
                return Image.open(io.BytesIO(file_in_zip.read())).convert("RGB")
    except Exception as e:
        logger.error(f"Dropbox download failed: {e}")
        raise ValueError(f"All image sources failed for {image_id}")

def create_fallback_dataset(num_samples):
    """Final fallback: Create synthetic data"""
    logger.error("All data sources failed. Using synthetic data.")
    return [
        {
            "image": create_synthetic_image("A red ball is to the left of a blue box"), 
            "caption": "A red ball is to the left of a blue box", 
            "relation_type": "left",
            "image_id": "synthetic_1"
        },
        {
            "image": create_synthetic_image("A cat is sitting above the mat"), 
            "caption": "A cat is sitting above the mat", 
            "relation_type": "above",
            "image_id": "synthetic_2"
        }
    ][:num_samples]

def create_synthetic_image(caption, size=(300, 300)):
    """Creates a synthetic image with a visual representation of the caption."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw caption text
    try:
        font = ImageFont.load_default()
        text_position = (10, 10)
    except ImportError:
        font = None
        text_position = (10, 10)
    
    draw.text(text_position, caption, fill=(0, 0, 0), font=font)
    
    # Draw spatial relationships if possible
    if "red ball" in caption and "blue box" in caption:
        ball_pos = (50, 150) if "left" in caption else (200, 150)
        box_pos = (200, 150) if "left" in caption else (50, 150)
        draw.ellipse((ball_pos[0], ball_pos[1], ball_pos[0]+50, ball_pos[1]+50), fill="red")
        draw.rectangle((box_pos[0], box_pos[1], box_pos[0]+50, box_pos[1]+50), fill="blue")
    
    elif "cat" in caption and "mat" in caption:
        cat_pos = (150, 50) if "above" in caption else (150, 200)
        mat_pos = (100, 200) if "above" in caption else (100, 100)
        draw.ellipse((cat_pos[0], cat_pos[1], cat_pos[0]+40, cat_pos[1]+40), fill="orange")
        draw.rectangle((mat_pos[0], mat_pos[1], mat_pos[0]+100, mat_pos[1]+30), fill="brown")
    
    return img