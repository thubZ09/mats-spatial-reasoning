import torch
from tqdm import tqdm
from PIL import Image
import io
import re
import numpy as np
from transformers import AutoProcessor
import logging

#setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_pil_image(image):
    """Convert various formats to PIL Image"""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image))
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    if isinstance(image, str): 
        return Image.open(image)
    raise ValueError(f"Unsupported image type: {type(image)}")

def normalize_response(response):
    """Convert free-text responses to spatial judgments"""
    if response is None:
        return None
        
    response = response.lower().strip()
    
    #handle various true/false patterns
    if re.search(r'\b(true|yes|correct|right)\b', response):
        return True
    if re.search(r'\b(false|no|incorrect|wrong)\b', response):
        return False
        
    #spatial relation detection
    if "left" in response and "right" not in response:
        return True
    if "right" in response and "left" not in response:
        return False
        
    #positional indicators
    if "above" in response and "below" not in response:
        return True
    if "below" in response and "above" not in response:
        return False
        
    return None 

def check_logical_consistency(orig_pred, pert_pred, relation_type):
    """Enhanced consistency check with relation awareness"""
    if orig_pred is None or pert_pred is None:
        return 'INDETERMINATE'
    
    exclusive_relations = ['left', 'right', 'above', 'below', 
                           'in front of', 'behind', 'top', 'bottom']
    
    if any(rel in relation_type for rel in exclusive_relations):
        return 'CONSISTENT' if orig_pred != pert_pred else 'INCONSISTENT'
    
    return 'CONSISTENT' if orig_pred == pert_pred else 'INCONSISTENT'

# ----------------------------
# pertubation functions

def perturb_spatial_words(sentence):
    """Perturbs spatial words in a sentence by swapping opposites"""
    spatial_swaps = {
        'left': 'right',
        'right': 'left',
        'above': 'below',
        'below': 'above',
        'top': 'bottom',
        'bottom': 'top',
        'in front of': 'behind',
        'behind': 'in front of',
        'front': 'back',
        'back': 'front'
    }
    
    perturbed = sentence
    for orig, repl in spatial_swaps.items():
        # Handle multi-word phrases
        if ' ' in orig:
            pattern = r'\b' + re.escape(orig) + r'\b'
            perturbed = re.sub(pattern, repl, perturbed, flags=re.IGNORECASE)
        # Single words
        else:
            pattern = r'\b' + re.escape(orig) + r'\b'
            perturbed = re.sub(pattern, repl, perturbed, flags=re.IGNORECASE)
    
    return perturbed

def rotate_image(image, angle=15):
    """Rotate image by specified angle"""
    try:
        image = ensure_pil_image(image)
        return image.rotate(angle)
    except Exception as e:
        logger.error(f"Rotation failed: {e}")
        return image

def flip_image_horizontally(image):
    """Flip image left-to-right"""
    try:
        image = ensure_pil_image(image)
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    except Exception as e:
        logger.error(f"Flip failed: {e}")
        return image

# ----------------------------
# Audit Functions
# ----------------------------

def get_raw_prediction(model, processor, image, caption_text):
    """Robust prediction function with error handling"""
    try:
        # Convert to PIL if needed
        image = ensure_pil_image(image).convert("RGB")
        
        # Improved prompt for spatial reasoning
        prompt = (
            "USER: <image>\nBased strictly on the visual spatial relationships, "
            f"is this statement true or false? \"{caption_text}\"\n"
            "Answer only with 'True' or 'False'.\nASSISTANT:"
        )
        
        inputs = processor(
            text=prompt, 
            images=image, 
            return_tensors="pt"
        ).to(model.device)
        
        # Handle models without image processing capability
        if 'pixel_values' not in inputs:
            inputs['pixel_values'] = None

        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=15)
        
        response = processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Extract assistant response
        return response.split("ASSISTANT:")[-1].strip()
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "ERROR"

def run_text_perturbation_audit(model, processor, dataset):
    """
    Runs a text perturbation audit on a given dataset using REAL images.
    """
    results = []
    
    for item in tqdm(dataset, desc="Auditing with Text Perturbation"):
        try:
            # Extract and convert image
            original_image = ensure_pil_image(item['image']).convert("RGB")
            original_caption = item['caption']
            
            # Get relation type if available
            relation_type = item.get('relation_type', 'unknown')
            
            # 1. Generate perturbed caption
            perturbed_caption = perturb_spatial_words(original_caption)
            
            # 2. Get raw predictions
            raw_original_pred = get_raw_prediction(model, processor, original_image, original_caption)
            raw_perturbed_pred = get_raw_prediction(model, processor, original_image, perturbed_caption)
            
            # 3. Normalize predictions
            norm_original_pred = normalize_response(raw_original_pred)
            norm_perturbed_pred = normalize_response(raw_perturbed_pred)
            
            # 4. Check logical consistency
            consistency = check_logical_consistency(
                norm_original_pred, 
                norm_perturbed_pred,
                relation_type
            )
            
            results.append({
                'original_caption': original_caption,
                'perturbed_caption': perturbed_caption,
                'raw_original_pred': raw_original_pred,
                'raw_perturbed_pred': raw_perturbed_pred,
                'normalized_original': norm_original_pred,
                'normalized_perturbed': norm_perturbed_pred,
                'consistency': consistency,
                'relation_type': relation_type
            })
            
        except Exception as e:
            logger.error(f"Audit failed for item: {e}")
            results.append({
                'error': str(e),
                'item': item
            })
        
    return results

def run_visual_perturbation_audit(model, processor, dataset):
    """
    Runs a visual perturbation audit (rotation) on a dataset.
    """
    results = []
    
    for item in tqdm(dataset, desc="Auditing with Visual Perturbation"):
        try:
            # Extract and convert image
            original_image = ensure_pil_image(item['image']).convert("RGB")
            caption = item['caption']
            relation_type = item.get('relation_type', 'unknown')
            
            # 1. Create rotated version
            perturbed_image = rotate_image(original_image, angle=15)
            
            # 2. Get raw predictions
            raw_original_pred = get_raw_prediction(model, processor, original_image, caption)
            raw_perturbed_pred = get_raw_prediction(model, processor, perturbed_image, caption)
            
            # 3. Normalize predictions
            norm_original_pred = normalize_response(raw_original_pred)
            norm_perturbed_pred = normalize_response(raw_perturbed_pred)
            
            # 4. Check consistency (should be same)
            if norm_original_pred is None or norm_perturbed_pred is None:
                consistency = 'INDETERMINATE'
            elif norm_original_pred == norm_perturbed_pred:
                consistency = 'CONSISTENT'
            else:
                consistency = 'INCONSISTENT'
                
            results.append({
                'caption': caption,
                'raw_original_pred': raw_original_pred,
                'raw_perturbed_pred': raw_perturbed_pred,
                'normalized_original': norm_original_pred,
                'normalized_perturbed': norm_perturbed_pred,
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
    Runs coordinated visual+text perturbation audit (flip + swap).
    """
    results = []
    
    for item in tqdm(dataset, desc="Coordinated Perturbation Audit"):
        try:
            # Extract and convert image
            original_image = ensure_pil_image(item['image']).convert("RGB")
            original_caption = item['caption']
            relation_type = item.get('relation_type', 'unknown')
            
            # 1. Apply coordinated perturbations
            flipped_image = flip_image_horizontally(original_image)
            perturbed_caption = perturb_spatial_words(original_caption)
            
            # 2. Get raw predictions
            # Original: (original image, original caption)
            raw_original_pred = get_raw_prediction(model, processor, original_image, original_caption)
            
            # Perturbed: (flipped image, perturbed caption)
            raw_perturbed_pred = get_raw_prediction(model, processor, flipped_image, perturbed_caption)
            
            # 3. Normalize predictions
            norm_original_pred = normalize_response(raw_original_pred)
            norm_perturbed_pred = normalize_response(raw_perturbed_pred)
            
            # 4. Check consistency (should be same)
            if norm_original_pred is None or norm_perturbed_pred is None:
                consistency = 'INDETERMINATE'
            elif norm_original_pred == norm_perturbed_pred:
                consistency = 'CONSISTENT'
            else:
                consistency = 'INCONSISTENT'
                
            results.append({
                'original_caption': original_caption,
                'perturbed_caption': perturbed_caption,
                'raw_original_pred': raw_original_pred,
                'raw_perturbed_pred': raw_perturbed_pred,
                'normalized_original': norm_original_pred,
                'normalized_perturbed': norm_perturbed_pred,
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

# ----------------------------
# Analysis Functions
# ----------------------------

def calculate_scs(results, perturbation_type='text'):
    """
    Calculate Spatial Consistency Score (SCS)
    """
    if not results:
        return 0.0
        
    consistent_count = sum(1 for r in results 
                          if r.get('consistency') == 'CONSISTENT')
    return consistent_count / len(results)

def generate_summary_table(audit_results):
    """
    Generate summary statistics from audit results
    """
    if not audit_results:
        return {}
        
    # Basic counts
    total = len(audit_results)
    consistent = sum(1 for r in audit_results if r.get('consistency') == 'CONSISTENT')
    inconsistent = sum(1 for r in audit_results if r.get('consistency') == 'INCONSISTENT')
    indeterminate = sum(1 for r in audit_results if r.get('consistency') == 'INDETERMINATE')
    
    # Failure mode analysis
    failure_modes = {}
    for r in audit_results:
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
