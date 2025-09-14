import os
import json
import requests
from PIL import Image
import io
import logging
import zipfile
import tempfile
from tqdm import tqdm
import shutil
import numpy as np
from datasets import load_dataset


logger = logging.getLogger(__name__)

#global cache directory for lazy initialization
_vsr_cache = {}

def get_vsr_cache_paths():
    """returns paths to VSR cache and image directories, creating them if necessary"""
    if "image_dir" not in _vsr_cache:
        cache_dir = os.path.join(tempfile.gettempdir(), "vsr_cache")
        image_dir = os.path.join(cache_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        _vsr_cache["cache_dir"] = cache_dir
        _vsr_cache["image_dir"] = image_dir
    return _vsr_cache["cache_dir"], _vsr_cache["image_dir"]

def download_vsr_images():
    """download and extract VSR images with progress tracking"""
    cache_dir, image_dir = get_vsr_cache_paths()
    
    #skip if images already exist
    if os.path.exists(image_dir) and len(os.listdir(image_dir)) > 1000:
        logger.info("Using cached VSR images")
        return image_dir
    zip_path = os.path.join(cache_dir, "vsr_images.zip")
    url = "https://www.dropbox.com/scl/fi/efvlqxp4zhxfp60m1hujd/vsr_images.zip?rlkey=3w3d8dxbt7xgq64pyh7zosnzm&e=1&st=7ltz5qqh&dl=1"
    
    logger.info("Downloading VSR images (this may take several minutes)...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    
    #extract the zip file
    logger.info("extracting images...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc="Extracting"):
            try:
                zip_ref.extract(member, cache_dir)
            except zipfile.error as e:
                logger.warning(f"Skipping invalid file {member.filename}: {e}")
    
    #move images to proper directory
    extracted_dir = os.path.join(cache_dir, "vsr_images")
    if os.path.exists(extracted_dir):
        for file_name in os.listdir(extracted_dir):
            shutil.move(os.path.join(extracted_dir, file_name), image_dir)
        shutil.rmtree(extracted_dir)
    
    #verify we have sufficient images
    if len(os.listdir(image_dir)) < 1000:
        raise RuntimeError("Insufficient images downloaded")
    
    logger.info(f"Downloaded {len(os.listdir(image_dir))} VSR images")
    return image_dir


def load_vsr_dataset(split='test', num_samples=100):
    """Main function to load VSR dataset with comprehensive error handling"""
    try:
        #download images first
        image_dir = download_vsr_images()
        
        #load dataset metadata from Hugging Face
        logger.info(f"Loading VSR metadata for {split} split")
        dataset = load_dataset("cambridgeltl/vsr_random", split=split)
        
        #filter and process samples
        samples = []
        valid_count = 0
        skipped = 0
        
        for item in tqdm(dataset, desc="Processing samples"):
            if valid_count >= num_samples:
                break
                
            #only use validated true statements
            if item.get('label', 0) != 1:
                skipped += 1
                continue
                
            image_path = os.path.join(image_dir, item['image'])
            if not os.path.exists(image_path):
                logger.debug(f"Missing image: {item['image']}")
                skipped += 1
                continue
                
            try:
                image = Image.open(image_path).convert("RGB")
                samples.append({
                    "image": image,
                    "caption": item['caption'],
                    "relation_type": item['relation'].lower(),
                    "image_id": item['image']
                })
                valid_count += 1
            except Exception as e:
                logger.warning(f"Failed to process {item['image']}: {e}")
                skipped += 1
        
        logger.info(f"Loaded {len(samples)} samples ({skipped} skipped)")
        return samples
        
    except Exception as e:
        logger.error(f"Critical error loading dataset: {e}")
        return create_fallback_dataset(num_samples)

def create_fallback_dataset(num_samples):
    """create synthetic fallback dataset with more samples"""
    logger.error("Using synthetic fallback dataset")
    samples = []
    relations = ['left', 'right', 'above', 'below', 'front', 'behind']
    
    for i in range(min(num_samples, 50)):
        relation = relations[i % len(relations)]
        caption = f"Sample {i+1}: Object A is {relation} of Object B"
        samples.append({
            "image": create_synthetic_image(caption),
            "caption": caption,
            "relation_type": relation,
            "image_id": f"synthetic_{i+1}"
        })
    
    return samples

def create_synthetic_image(caption, size=(400, 400)):
    """create better synthetic images with visual relationships"""
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.new("RGB", size, (240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    #try to load font
    try:
        font = ImageFont.truetype("Arial", 14)
    except:
        font = None
    
    #draw caption
    draw.rectangle([10, 10, size[0]-10, 40], fill=(220, 220, 255))
    draw.text((20, 15), caption, fill=(0, 0, 0), font=font)
    
    #draw objects based on relation
    obj_a_pos, obj_b_pos = (100, 200), (300, 200)
    
    if 'left' in caption:
        obj_a_pos, obj_b_pos = (100, 200), (300, 200)
    elif 'right' in caption:
        obj_a_pos, obj_b_pos = (300, 200), (100, 200)
    elif 'above' in caption:
        obj_a_pos, obj_b_pos = (200, 100), (200, 300)
    elif 'below' in caption:
        obj_a_pos, obj_b_pos = (200, 300), (200, 100)
    elif 'front' in caption:
        obj_a_pos, obj_b_pos = (200, 200), (250, 200)
    elif 'behind' in caption:
        obj_a_pos, obj_b_pos = (250, 200), (200, 200)
    
    #draw objects
    draw.ellipse(
        [obj_a_pos[0]-30, obj_a_pos[1]-30, obj_a_pos[0]+30, obj_a_pos[1]+30],
        fill=(255, 100, 100)
    )
    draw.rectangle(
        [obj_b_pos[0]-30, obj_b_pos[1]-30, obj_b_pos[0]+30, obj_b_pos[1]+30],
        fill=(100, 100, 255)
    )
    
    #draw labels
    draw.text((obj_a_pos[0]-10, obj_a_pos[1]-40), "A", fill=(0, 0, 0), font=font)
    draw.text((obj_b_pos[0]-10, obj_b_pos[1]-40), "B", fill=(0, 0, 0), font=font)
    
    return img