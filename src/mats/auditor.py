# src/mats/auditor.py
import torch
from tqdm import tqdm
import logging
from . import perturbations
from . import metrics

logger = logging.getLogger(__name__)

def ensure_pil_image(image):
    """Compatibility function matching perturbations.py"""
    return perturbations.ensure_pil_image(image)

def get_raw_prediction(model, processor, image, caption_text, model_type):
    """
    Gets a prediction using advanced prompting and generation controls
    to combat pathological truth bias.
    """
    try:
        image = ensure_pil_image(image).convert("RGB")
        
        # Neutral prompt without compliance triggers
        prompt = (
            f"Question: Considering the image, is this statement accurate? "
            f"\"{caption_text}\" Answer only with 'true' or 'false'."
        )

        # Model-specific formatting
        if model_type == "llava":
            prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        elif model_type == "blip":
            prompt = f"Question: {prompt} Answer:"
        
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        
        # Strict generation parameters
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=5,        # Force short answers
                temperature=0.01,        # Minimize randomness
                num_beams=1,             # Disable beam search
                do_sample=False,          # Greedy decoding
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        response = processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Clean up response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        elif "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
            
        return response.strip()
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return ""

def run_text_perturbation_audit(model, processor, dataset, model_type):
    results = []
    for item in tqdm(dataset, desc=f"Text Audit ({model_type})"):
        try:
            # Get perturbed caption and change status
            pert_caption, was_changed = perturbations.perturb_spatial_words(item['caption'])
            
            # Skip if no spatial terms found
            if not was_changed:
                logger.debug(f"Skipping: No spatial terms in '{item['caption']}'")
                continue
                
            # Get predictions
            raw_orig = get_raw_prediction(model, processor, item['image'], item['caption'], model_type)
            raw_pert = get_raw_prediction(model, processor, item['image'], pert_caption, model_type)
            
            # Normalize and check consistency
            norm_orig = metrics.normalize_response(raw_orig)
            norm_pert = metrics.normalize_response(raw_pert)
            consistency = metrics.check_text_consistency(norm_orig, norm_pert)
            
            results.append({
                'caption': item['caption'],
                'perturbed_caption': pert_caption,
                'raw_original_pred': raw_orig,
                'raw_perturbed_pred': raw_pert,
                'consistency': consistency,
                'relation_type': item['relation_type']
            })
        except Exception as e:
            logger.error(f"Item failed in text audit: {e}")
    return results

def run_visual_perturbation_audit(model, processor, dataset, model_type):
    results = []
    for item in tqdm(dataset, desc=f"Visual Audit ({model_type})"):
        try:
            pert_image = perturbations.rotate_image(item['image'])
            raw_orig = get_raw_prediction(model, processor, item['image'], item['caption'], model_type)
            raw_pert = get_raw_prediction(model, processor, pert_image, item['caption'], model_type)
            norm_orig = metrics.normalize_response(raw_orig)
            norm_pert = metrics.normalize_response(raw_pert)
            consistency = metrics.check_visual_consistency(norm_orig, norm_pert)
            results.append({'caption': item['caption'], 'raw_original_pred': raw_orig, 'raw_perturbed_pred': raw_pert, 'consistency': consistency, 'relation_type': item['relation_type']})
        except Exception as e:
            logger.error(f"Item failed in visual audit: {e}")
    return results

def run_coordinated_perturbation_audit(model, processor, dataset, model_type):
    results = []
    for item in tqdm(dataset, desc=f"Coordinated Audit ({model_type})"):
        try:
            pert_image = perturbations.flip_image_horizontally(item['image'])
            pert_caption = perturbations.perturb_spatial_words(item['caption'])
            raw_orig = get_raw_prediction(model, processor, item['image'], item['caption'], model_type)
            raw_pert = get_raw_prediction(model, processor, pert_image, pert_caption, model_type)
            norm_orig = metrics.normalize_response(raw_orig)
            norm_pert = metrics.normalize_response(raw_pert)
            consistency = metrics.check_coordinated_consistency(norm_orig, norm_pert)
            results.append({'caption': item['caption'], 'perturbed_caption': pert_caption, 'raw_original_pred': raw_orig, 'raw_perturbed_pred': raw_pert, 'consistency': consistency, 'relation_type': item['relation_type']})
        except Exception as e:
            logger.error(f"Item failed in coordinated audit: {e}")
    return results