# src/mats/auditor.py

import torch
from tqdm import tqdm
import logging
from . import perturbations
from . import metrics

logger = logging.getLogger(__name__)

def get_raw_prediction(model, processor, image, caption_text, model_type):
    """Gets a prediction, adapting the prompt and logic for the specified model type."""
    # This function is now responsible for getting a raw string output from any model.
    # ... (paste the full get_raw_prediction function from the previous response here) ...
    # This is the one with the if/elif for "clip", "llava", "blip"
    prompt = ""
    if model_type == "clip":
        inputs = processor(text=[caption_text, "a photo"], images=image, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        return str(outputs.logits_per_image[0][0].item())
    elif model_type == "llava":
        prompt = f"USER: <image>\nBased strictly on the visual spatial relationships, is this statement true or false? \"{caption_text}\" Answer only with 'True' or 'False'.\nASSISTANT:"
    elif model_type == "blip":
        prompt = f"Question: Based on the image, is the following statement true or false? \"{caption_text}\" Answer:"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=10)
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    if "ASSISTANT:" in response:
        return response.split("ASSISTANT:")[-1].strip()
    return response.strip()


def run_text_perturbation_audit(model, processor, dataset, model_type):
    results = []
    for item in tqdm(dataset, desc=f"Text Audit ({model_type})"):
        try:
            pert_caption, was_changed = perturbations.perturb_spatial_words(item['caption'])
            if not was_changed:
                logger.debug(f"Skipping sample as no spatial keyword was found: {item['caption']}")
                continue
            raw_orig = get_raw_prediction(model, processor, item['image'], item['caption'], model_type)
            raw_pert = get_raw_prediction(model, processor, item['image'], pert_caption, model_type)
            norm_orig = metrics.normalize_response(raw_orig)
            norm_pert = metrics.normalize_response(raw_pert)
            consistency = metrics.check_text_consistency(norm_orig, norm_pert)
            results.append({'caption': item['caption'], 'perturbed_caption': pert_caption, 'raw_original_pred': raw_orig, 'raw_perturbed_pred': raw_pert, 'consistency': consistency, 'relation_type': item['relation_type']})
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