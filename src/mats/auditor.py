import torch
from tqdm import tqdm
from PIL import Image, ImageDraw
import io
import re
import numpy as np
import requests
from io import BytesIO
from datasets import load_dataset
import logging
import sys
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger(__name__)

def create_synthetic_image(caption, size=(300, 300)):
    """Create synthetic image with visual representation of caption"""
    img = Image.new('RGB', size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw caption
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except ImportError:
        font = None
    
    draw.text((10, 10), caption, fill=(0, 0, 0), font=font)
    
    # Draw objects based on caption
    if "red ball" in caption and "blue box" in caption:
        if "left" in caption:
            draw.ellipse((50, 150, 100, 200), fill='red')  # Left ball
            draw.rectangle((250, 150, 300, 200), fill='blue')  # Right box
        else:
            draw.ellipse((250, 150, 300, 200), fill='red')  # Right ball
            draw.rectangle((50, 150, 100, 200), fill='blue')  # Left box
    
    if "cat" in caption and "mat" in caption:
        if "above" in caption:
            draw.ellipse((150, 50, 200, 100), fill='gray')  # Top cat
            draw.rectangle((125, 200, 225, 250), fill='brown')  # Bottom mat
        else:
            draw.ellipse((150, 200, 200, 250), fill='gray')  # Bottom cat
            draw.rectangle((125, 50, 225, 100), fill='brown')  # Top mat
    
    return img

def load_vsr_dataset():
    """Load VSR dataset with robust error handling"""
    try:
        # Try both configurations without trust_remote_code
        for config in ['random', 'zeroshot']:
            try:
                logger.info(f"Trying configuration: {config}")
                dataset = load_dataset("juletxara/visual-spatial-reasoning", config)
                logger.info("✅ VSR dataset loaded successfully!")
                
                # Process dataset
                if isinstance(dataset, dict) and 'train' in dataset:
                    if 'test' not in dataset:
                        train_data = dataset['train']
                        if hasattr(train_data, 'train_test_split'):
                            split_dataset = train_data.train_test_split(test_size=0.2)
                            test_data = split_dataset['test']
                        else:
                            raise ValueError("Dataset does not support train_test_split")
                    else:
                        test_data = dataset['test']
                else:
                    raise ValueError("Dataset format not recognized")
                
                # Convert byte images to PIL
                def convert_to_pil(item):
                    if isinstance(item['image'], bytes):
                        item['image'] = Image.open(BytesIO(item['image']))
                    return item
                
                test_data = test_data.map(convert_to_pil)
                return test_data
            except Exception as e:
                logger.warning(f"Config '{config}' failed: {str(e)}")
        
        raise RuntimeError("All configurations failed")
        
    except Exception as e:
        logger.error(f"Dataset load failed: {str(e)}")
        logger.info("⚠️ Falling back to synthetic dataset")
        
        # Fallback synthetic data
        return [
            {
                'image': create_synthetic_image("A red ball is to the left of a blue box"),
                'caption': "A red ball is to the left of a blue box",
                'relation_type': 'left'
            },
            {
                'image': create_synthetic_image("A cat is sitting above the mat"),
                'caption': "A cat is sitting above the mat",
                'relation_type': 'above'
            }
        ]

def ensure_pil_image(image):
    """Convert various formats to PIL Image"""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, bytes):
        return Image.open(BytesIO(image))
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    if isinstance(image, str):  # File path
        return Image.open(image)
    return Image.fromarray(np.array(image))  # Last resort

def normalize_response(response):
    """Convert model response to boolean"""
    if not response:
        return None
        
    response = response.lower().strip()
    
    # Handle various true/false patterns
    if re.search(r'\b(true|yes|correct|right|affirmative)\b', response):
        return True
    if re.search(r'\b(false|no|incorrect|wrong|negative)\b', response):
        return False
        
    # Spatial relation detection
    if "left" in response and "right" not in response:
        return True
    if "right" in response and "left" not in response:
        return False
        
    # Positional indicators
    if "above" in response and "below" not in response:
        return True
    if "below" in response and "above" not in response:
        return False
        
    return None  # Indeterminate

def check_logical_consistency(orig_pred, pert_pred, relation_type):
    """Enhanced consistency check with relation awareness"""
    if orig_pred is None or pert_pred is None:
        return 'INDETERMINATE'
    
    # For mutually exclusive relations
    exclusive_relations = ['left', 'right', 'above', 'below', 
                           'in front of', 'behind', 'top', 'bottom']
    
    if any(rel in relation_type for rel in exclusive_relations):
        return 'CONSISTENT' if orig_pred != pert_pred else 'INCONSISTENT'
    
    # For non-exclusive relations
    return 'CONSISTENT' if orig_pred == pert_pred else 'INCONSISTENT'

def perturb_spatial_words(sentence):
    """Perturb spatial relationships in text"""
    swaps = {
        'left': 'right', 'right': 'left',
        'above': 'below', 'below': 'above',
        'top': 'bottom', 'bottom': 'top',
        'in front of': 'behind', 'behind': 'in front of',
        'front': 'back', 'back': 'front'
    }
    
    # Case-insensitive replacement
    for orig, repl in swaps.items():
        pattern = r'\b' + re.escape(orig) + r'\b'
        sentence = re.sub(pattern, repl, sentence, flags=re.IGNORECASE)
    
    return sentence

def rotate_image(image, angle=15):
    """Rotate image while preserving spatial relationships"""
    try:
        img = ensure_pil_image(image).convert("RGB")
        return img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    except Exception as e:
        logger.error(f"Rotation failed: {e}")
        return image

def flip_image_horizontally(image):
    """Flip image left-to-right (changes left/right relationships)"""
    try:
        img = ensure_pil_image(image).convert("RGB")
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    except Exception as e:
        logger.error(f"Flip failed: {e}")
        return image

def get_model_response(model, processor, image, caption):
    """Get model response with error handling"""
    try:
        # Build prompt
        prompt = (
            "USER: <image>\nBased strictly on visual spatial relationships, "
            f"is this true or false? \"{caption}\" "
            "Answer only with 'True' or 'False'.\nASSISTANT:"
        )
        
        # Process inputs
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode response
        response = processor.batch_decode(
            outputs, 
            skip_special_tokens=True
        )[0]
        
        # Extract assistant response
        return response.split("ASSISTANT:")[-1].strip()
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return "ERROR"

def run_text_perturbation_audit(model, processor, dataset):
    """
    Runs a text perturbation audit on a given dataset
    """
    results = []
    
    for item in tqdm(dataset, desc="Text Perturbation Audit"):
        try:
            # Prepare data
            image = ensure_pil_image(item['image']).convert("RGB")
            caption = item['caption']
            relation_type = item.get('relation_type', 'unknown')
            
            # Generate perturbed caption
            pert_caption = perturb_spatial_words(caption)
            
            # Get predictions
            orig_response = get_model_response(model, processor, image, caption)
            pert_response = get_model_response(model, processor, image, pert_caption)
            
            # Normalize responses
            orig_norm = normalize_response(orig_response)
            pert_norm = normalize_response(pert_response)
            
            # Check consistency
            consistency = check_logical_consistency(orig_norm, pert_norm, relation_type)
            
            results.append({
                'caption': caption,
                'perturbed_caption': pert_caption,
                'orig_response': orig_response,
                'pert_response': pert_response,
                'orig_norm': orig_norm,
                'pert_norm': pert_norm,
                'consistency': consistency,
                'relation_type': relation_type
            })
            
        except Exception as e:
            logger.error(f"Text audit failed: {e}")
            results.append({
                'error': str(e),
                'item': item
            })
            
    return results

def run_visual_perturbation_audit(model, processor, dataset):
    """
    Runs visual perturbation audit (rotation)
    """
    results = []
    
    for item in tqdm(dataset, desc="Visual Perturbation Audit"):
        try:
            # Prepare data
            image = ensure_pil_image(item['image']).convert("RGB")
            caption = item['caption']
            relation_type = item.get('relation_type', 'unknown')
            
            # Original prediction
            orig_response = get_model_response(model, processor, image, caption)
            orig_norm = normalize_response(orig_response)
            
            # Rotated prediction
            rotated_img = rotate_image(image, 15)
            rotated_response = get_model_response(model, processor, rotated_img, caption)
            rotated_norm = normalize_response(rotated_response)
            
            # Check consistency
            if orig_norm is None or rotated_norm is None:
                consistency = 'INDETERMINATE'
            elif orig_norm == rotated_norm:
                consistency = 'CONSISTENT'
            else:
                consistency = 'INCONSISTENT'
                
            results.append({
                'caption': caption,
                'orig_response': orig_response,
                'rotated_response': rotated_response,
                'orig_norm': orig_norm,
                'rotated_norm': rotated_norm,
                'consistency': consistency,
                'relation_type': relation_type
            })
            
        except Exception as e:
            logger.error(f"Visual audit failed: {e}")
            results.append({
                'error': str(e),
                'item': item
            })
            
    return results

def run_coordinated_perturbation_audit(model, processor, dataset):
    """
    Runs coordinated visual+text perturbation audit
    """
    results = []
    
    for item in tqdm(dataset, desc="Coordinated Perturbation Audit"):
        try:
            # Prepare data
            image = ensure_pil_image(item['image']).convert("RGB")
            caption = item['caption']
            relation_type = item.get('relation_type', 'unknown')
            
            # Original prediction
            orig_response = get_model_response(model, processor, image, caption)
            orig_norm = normalize_response(orig_response)
            
            # Coordinated perturbations
            flipped_img = flip_image_horizontally(image)
            pert_caption = perturb_spatial_words(caption)
            pert_response = get_model_response(model, processor, flipped_img, pert_caption)
            pert_norm = normalize_response(pert_response)
            
            # Check consistency
            if orig_norm is None or pert_norm is None:
                consistency = 'INDETERMINATE'
            elif orig_norm == pert_norm:
                consistency = 'CONSISTENT'
            else:
                consistency = 'INCONSISTENT'
                
            results.append({
                'orig_caption': caption,
                'pert_caption': pert_caption,
                'orig_response': orig_response,
                'pert_response': pert_response,
                'orig_norm': orig_norm,
                'pert_norm': pert_norm,
                'consistency': consistency,
                'relation_type': relation_type
            })
            
        except Exception as e:
            logger.error(f"Coordinated audit failed: {e}")
            results.append({
                'error': str(e),
                'item': item
            })
            
    return results

def calculate_scs(results):
    """
    Calculate Spatial Consistency Score (SCS)
    """
    if not results:
        return 0.0
        
    consistent = sum(1 for r in results if r.get('consistency') == 'CONSISTENT')
    return consistent / len(results)

def generate_summary_table(results):
    """
    Generate summary statistics from audit results
    """
    if not results:
        return {}
        
    # Basic counts
    total = len(results)
    consistent = sum(1 for r in results if r.get('consistency') == 'CONSISTENT')
    inconsistent = sum(1 for r in results if r.get('consistency') == 'INCONSISTENT')
    indeterminate = sum(1 for r in results if r.get('consistency') == 'INDETERMINATE')
    
    # Failure mode analysis
    failure_modes = {}
    for r in results:
        if r.get('consistency') == 'INCONSISTENT':
            rel_type = r.get('relation_type', 'unknown')
            failure_modes[rel_type] = failure_modes.get(rel_type, 0) + 1
    
    return {
        'total_samples': total,
        'consistent': consistent,
        'inconsistent': inconsistent,
        'indeterminate': indeterminate,
        'consistency_rate': consistent / total if total > 0 else 0,
        'failure_modes': failure_modes
    }