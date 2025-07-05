# In src/mats/auditor.py
import torch
import json
import logging
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
from . import perturbations, metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuditorConfig:
    """Configuration class for auditor parameters."""
    def __init__(self, 
                 max_new_tokens: int = 5,
                 do_sample: bool = False,
                 batch_size: int = 1,
                 clear_cache_frequency: int = 100):
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.batch_size = batch_size
        self.clear_cache_frequency = clear_cache_frequency

def build_prompt(caption: str, cognitive_map: str = None) -> str:
    """Builds the prompt, optionally including a cognitive map."""
    if cognitive_map:
        return (
            "A ground-truth 'cognitive map' of the scene is provided in JSON format.\n"
            "Based on both the image and this map, is the following statement true or false?\n\n"
            f"Cognitive Map:\n```json\n{cognitive_map}\n```\n\n"
            f"Statement: \"{caption}\"\n\n"
            "Answer with only the single word 'true' or 'false'."
        )
    else:
        return f"Based on the image, is the following statement true or false? \"{caption}\""

def get_generative_prediction(model, processor, image, caption: str, cognitive_map: str = None) -> str:
    """
    Gets a 'true'/'false' prediction from a generative model.
    This version is hardened to handle LLaVA and Qwen model quirks.
    """
    prompt_text = build_prompt(caption, cognitive_map)

    try:
        is_qwen_model = "qwen" in getattr(model.config, 'model_type', '').lower()
        gen_kwargs = {"max_new_tokens": 10, "do_sample": False}

        if is_qwen_model:
            # Qwen-specific formatting and generation arguments
            conversation = [{'image': image}, {'text': prompt_text}]
            # The Qwen processor needs the template applied before tokenization
            full_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(model.device)
            # The Qwen tokenizer handles padding internally and does not need pad_token_id
        else:
            # LLaVA-style formatting
            full_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
            inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(model.device)
            # LLaVA requires pad_token_id for generation
            if hasattr(processor.tokenizer, 'pad_token_id') and processor.tokenizer.pad_token_id is not None:
                gen_kwargs['pad_token_id'] = processor.tokenizer.pad_token_id

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[-1]
        
        # Clean the final output
        return response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response.strip()

    except Exception as e:
        logger.error(f"Error in generative prediction: {e}")
        return "ERROR"
    
def get_contrastive_prediction(model, processor, image, caption1: str, caption2: str) -> str:
    """Gets a prediction from a contrastive model, handling CLIP and BiomedCLIP quirks."""
    try:
        # A more robust check for BiomedCLIP
        is_biomed_clip = "biomed" in getattr(model.config, '_name_or_path', '').lower()
        
        if is_biomed_clip:
            # BiomedCLIP's processor does not accept the 'padding' argument
            inputs = processor(text=[caption1, caption2], images=image, return_tensors="pt").to(model.device)
        else:
            inputs = processor(text=[caption1, caption2], images=image, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        best_match_index = outputs.logits_per_image.argmax().item()
        return caption1 if best_match_index == 0 else caption2
    except Exception as e:
        logger.error(f"Error in contrastive prediction: {e}")
        return "ERROR"
    
def _clear_gpu_cache_if_needed(step: int, config: AuditorConfig) -> None:
    """clear GPU cache periodically to prevent memory issues."""
    if torch.cuda.is_available() and step % config.clear_cache_frequency == 0:
        torch.cuda.empty_cache()
        logger.debug(f"Cleared GPU cache at step {step}")

def _validate_inputs(model, processor, model_type: str, dataset) -> None:
    """Validate inputs to the audit function."""
    if model_type not in ['generative', 'contrastive']:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Must be 'generative' or 'contrastive'")
    
    if not dataset:
        raise ValueError("Dataset cannot be empty")
    
    if model is None or processor is None:
        raise ValueError("Model and processor cannot be None")

def run_comparative_text_audit(model, processor, model_type: str, dataset, config: dict = None, use_maps: bool = False):
    """
    A unified audit function that intelligently handles both generative and contrastive models
    and can optionally run in a "map-assisted" mode.
    """
    # Import locally to avoid circular dependencies
    from . import perturbations, metrics
    
    results = []
    audit_mode = "Map-Assisted" if use_maps else "Baseline"
    
    for item in tqdm(dataset, desc=f"Auditing {audit_mode} ({model_type})"):
        original_caption = item['caption']
        perturbed_caption, was_changed = perturbations.perturb_spatial_words(original_caption)
        if not was_changed:
            continue

        cognitive_map = perturbations.generate_cognitive_map(original_caption) if use_maps else None
        
        if model_type == 'generative':
            orig_resp = get_generative_prediction(model, processor, item['image'], original_caption, cognitive_map)
            pert_resp = get_generative_prediction(model, processor, item['image'], perturbed_caption, cognitive_map)
            
            norm_orig = metrics.normalize_response(orig_resp)
            norm_pert = metrics.normalize_response(pert_resp)
            consistency = metrics.check_text_consistency(norm_orig, norm_pert)

        elif model_type == 'contrastive':
            # Note: Maps are not typically used with contrastive models, so we don't pass them.
            best_caption = get_contrastive_prediction(model, processor, item['image'], original_caption, perturbed_caption)
            consistency = 'CONSISTENT' if best_caption == original_caption else 'INCONSISTENT'
        
        results.append({'consistency': consistency})
        
    return results


def analyze_audit_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the results of an audit and provide summary statistics.
    
    Args:
        results: List of audit results
    
    Returns:
        Dict containing analysis summary
    """
    if not results:
        return {"error": "No results to analyze"}
    
    total = len(results)
    consistent = sum(1 for r in results if r['consistency'] == 'CONSISTENT')
    inconsistent = sum(1 for r in results if r['consistency'] == 'INCONSISTENT')
    errors = sum(1 for r in results if r['consistency'] == 'ERROR')
    
    # Analyze by relation type
    relation_stats = {}
    for result in results:
        rel_type = result.get('relation_type', 'unknown')
        if rel_type not in relation_stats:
            relation_stats[rel_type] = {'total': 0, 'consistent': 0}
        relation_stats[rel_type]['total'] += 1
        if result['consistency'] == 'CONSISTENT':
            relation_stats[rel_type]['consistent'] += 1
    
    # Calculate consistency rates by relation type
    for rel_type in relation_stats:
        stats = relation_stats[rel_type]
        stats['consistency_rate'] = stats['consistent'] / stats['total'] if stats['total'] > 0 else 0
    
    return {
        'total_items': total,
        'consistent': consistent,
        'inconsistent': inconsistent,
        'errors': errors,
        'overall_consistency_rate': consistent / total if total > 0 else 0,
        'error_rate': errors / total if total > 0 else 0,
        'relation_type_analysis': relation_stats
    }