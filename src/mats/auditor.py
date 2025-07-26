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
    """config class for auditor parameters"""
    def __init__(self, 
                 max_new_tokens: int = 5,
                 do_sample: bool = False,
                 batch_size: int = 1,
                 clear_cache_frequency: int = 100):
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.batch_size = batch_size
        self.clear_cache_frequency = clear_cache_frequency

def detect_model_type(model, processor) -> Tuple[str, str]:
    """
    Detect the specific model type and family for proper handling.
    Returns: (model_family, model_architecture)
    """
    model_name = getattr(model.config, '_name_or_path', '').lower()
    model_type = getattr(model.config, 'model_type', '').lower()
    
    # Check for specific model families
    if 'llava' in model_name or 'llava' in model_type:
        return 'llava', 'generative'
    elif 'moondream' in model_name:
        return 'moondream', 'generative'
    elif 'qwen' in model_name or 'qwen' in model_type:
        return 'qwen', 'generative'
    elif 'siglip' in model_name or hasattr(model.config, 'text_config'):
        return 'siglip', 'contrastive'
    elif 'clip' in model_name or 'clip' in model_type:
        return 'clip', 'contrastive'
    elif 'biomed' in model_name:
        return 'biomed_clip', 'contrastive'
    elif hasattr(model, 'get_text_features') and hasattr(model, 'get_image_features'):
        # Generic CLIP-like model
        return 'clip_like', 'contrastive'
    elif hasattr(model, 'generate'):
        # Generic generative model
        return 'generic_generative', 'generative'
    else:
        logger.warning(f"Unknown model type: {model_name}. Attempting generic approach.")
        return 'unknown', 'unknown'

def build_prompt(caption: str, cognitive_map: str = None, model_family: str = 'generic') -> str:
    """builds the prompt, optionally including a cognitive map, tailored for specific models"""
    
    base_question = f"Based on the image, is the following statement true or false? \"{caption}\""
    
    if cognitive_map:
        map_context = (
            "A ground-truth 'cognitive map' of the scene is provided in JSON format.\n"
            "Based on both the image and this map, is the following statement true or false?\n\n"
            f"Cognitive Map:\n```json\n{cognitive_map}\n```\n\n"
            f"Statement: \"{caption}\"\n\n"
        )
    else:
        map_context = base_question + "\n\n"
    
    # Model-specific prompt formatting
    if model_family == 'moondream':
        # Moondream prefers simple, direct questions
        if cognitive_map:
            return map_context + "Answer: true or false"
        else:
            return base_question + " Answer: true or false"
            
    elif model_family == 'llava':
        # LLaVA works well with structured prompts
        if cognitive_map:
            return map_context + "Please answer with only 'true' or 'false'."
        else:
            return base_question + " Please answer with only 'true' or 'false'."
            
    elif model_family == 'qwen':
        # Qwen prefers conversational style
        if cognitive_map:
            return map_context + "Answer with only the single word 'true' or 'false'."
        else:
            return base_question + " Answer with only the single word 'true' or 'false'."
    
    else:
        # Generic fallback
        if cognitive_map:
            return map_context + "Answer with only 'true' or 'false'."
        else:
            return base_question + " Answer with only 'true' or 'false'."

def get_generative_prediction(model, processor, image, caption: str, cognitive_map: str = None, model_family: str = 'generic') -> str:
    """
    Gets a 'true'/'false' prediction from a generative model with model-specific handling
    """
    prompt_text = build_prompt(caption, cognitive_map, model_family)

    try:
        gen_kwargs = {"max_new_tokens": 10, "do_sample": False}
        
        if model_family == 'moondream':
            # Moondream2 uses its own special interface - FIXED
            try:
                # Moondream2 requires: answer_question(image_embeds, prompt, tokenizer)
                if hasattr(model, 'answer_question') and hasattr(model, 'encode_image'):
                    # Encode the image first
                    image_embeds = model.encode_image(image)
                    # Use the answer_question method with correct arguments
                    response = model.answer_question(image_embeds, prompt_text, processor.tokenizer)
                    return clean_moondream_response(response)
                else:
                    logger.warning("Moondream2 methods not found, trying alternative approach")
                    # Alternative: try with processor
                    inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        output_ids = model.generate(**inputs, **gen_kwargs)
                    response = processor.batch_decode(output_ids, skip_special_tokens=True)[-1]
                    return clean_moondream_response(response)
                    
            except Exception as e:
                logger.error(f"Moondream2 native interface failed: {e}")
                # Final fallback - try basic processor approach
                try:
                    inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        output_ids = model.generate(**inputs, **gen_kwargs)
                    response = processor.batch_decode(output_ids, skip_special_tokens=True)[-1]
                    return clean_moondream_response(response)
                except Exception as final_e:
                    logger.error(f"All Moondream2 approaches failed: {final_e}")
                    return "ERROR"
        
        elif model_family == 'qwen':
            # Qwen-specific formatting
            conversation = [{'image': image}, {'text': prompt_text}]
            full_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(model.device)
                
        elif model_family == 'llava':
            # LLaVA-style formatting with proper conversation template
            full_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
            inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(model.device)
            
            # LLaVA requires pad_token_id for generation
            if hasattr(processor.tokenizer, 'pad_token_id') and processor.tokenizer.pad_token_id is not None:
                gen_kwargs['pad_token_id'] = processor.tokenizer.pad_token_id
                
        else:
            # Generic generative model approach
            try:
                # Try with image token placeholder
                full_prompt = f"<image>\n{prompt_text}"
                inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(model.device)
            except:
                # Fallback without image token
                inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(model.device)

        # Generate response (skip if we already got one from Moondream)
        if model_family != 'moondream':
            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)
            
            response = processor.batch_decode(output_ids, skip_special_tokens=True)[-1]
            # Clean the response based on model family
            cleaned_response = clean_generative_response(response, model_family)
            return cleaned_response

    except Exception as e:
        logger.error(f"Error in generative prediction for {model_family}: {e}")
        return "ERROR"

def clean_generative_response(response: str, model_family: str) -> str:
    """Clean and normalize responses from different generative models"""
    
    # Remove common prefixes/suffixes
    response = response.strip()
    
    if model_family == 'llava':
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
    
    elif model_family == 'moondream':
        return clean_moondream_response(response)
        
    elif model_family == 'qwen':
        # Qwen sometimes includes the original prompt
        if "Answer with only" in response:
            response = response.split("Answer with only")[-1].strip()
    
    # Generic cleaning
    response = response.lower().strip()
    
    # Extract true/false from common patterns
    if 'true' in response and 'false' not in response:
        return 'true'
    elif 'false' in response and 'true' not in response:
        return 'false'
    elif response.startswith('true'):
        return 'true'
    elif response.startswith('false'):
        return 'false'
    
    # Handle yes/no responses
    if response.startswith('yes'):
        return 'true'
    elif response.startswith('no'):
        return 'false'
    
    return response.strip()

def clean_moondream_response(response: str) -> str:
    """Specific cleaning for Moondream2 responses"""
    response = response.strip().lower()
    
    # Moondream sometimes gives verbose answers
    # Look for key words in the response
    if 'true' in response and 'false' not in response:
        return 'true'
    elif 'false' in response and 'true' not in response:
        return 'false'
    elif 'yes' in response and 'no' not in response:
        return 'true'
    elif 'no' in response and 'yes' not in response:
        return 'false'
    elif 'correct' in response:
        return 'true'
    elif 'incorrect' in response or 'wrong' in response:
        return 'false'
    
    return response.strip()

def get_contrastive_prediction(model, processor, image, caption1: str, caption2: str, model_family: str = 'clip') -> str:
    """Gets a prediction from a contrastive model with the definitive SigLIP fix."""
    try:
        # --- THE FINAL FIX FOR SIGLIP ---
        if model_family == 'siglip':
            # SigLIP is sensitive to data types. We must ensure the image tensor
            # matches the model's expected dtype (e.g., float16) to prevent the error.
            inputs = processor(text=[caption1, caption2], images=image, return_tensors="pt", padding=True)
            inputs['pixel_values'] = inputs['pixel_values'].to(model.device, dtype=model.dtype)
            inputs['input_ids'] = inputs['input_ids'].to(model.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(model.device)
        else: # Standard logic for CLIP / BiomedCLIP
            inputs = processor(text=[caption1, caption2], images=image, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, 'logits_per_image'): # For CLIP
                best_match_index = outputs.logits_per_image.argmax().item()
            else: # For SigLIP
                best_match_index = outputs.logits.argmax().item()
        
        return caption1 if best_match_index == 0 else caption2

    except Exception as e:
        logger.error(f"Error in contrastive prediction for {model_family}: {e}")
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

def run_comparative_text_audit(model, processor, model_type: str = None, dataset=None, config: dict = None, use_maps: bool = False):
    """
    A unified audit function that intelligently handles both generative and contrastive models
    and can optionally run in a "map-assisted" mode.
    """
    # Import locally to avoid circular dependencies
    from . import perturbations, metrics
    
    # Auto-detect model type if not provided
    if model_type is None:
        model_family, model_type = detect_model_type(model, processor)
        logger.info(f"Auto-detected model: {model_family} ({model_type})")
    else:
        model_family, _ = detect_model_type(model, processor)
        logger.info(f"Using provided model_type: {model_type}, detected family: {model_family}")
    
    _validate_inputs(model, processor, model_type, dataset)
    
    results = []
    audit_mode = "Map-Assisted" if use_maps else "Baseline"
    config = config or AuditorConfig()
    
    logger.info(f"Starting {audit_mode} audit for {model_family} model ({model_type})")
    
    for step, item in enumerate(tqdm(dataset, desc=f"Auditing {audit_mode} ({model_type})")):
        try:
            original_caption = item['caption']
            perturbed_caption, was_changed = perturbations.perturb_spatial_words(original_caption)
            if not was_changed:
                continue

            cognitive_map = item.get("cognitive_map") if use_maps else None
            
            if model_type == 'generative':
                orig_resp = get_generative_prediction(
                    model, processor, item['image'], original_caption, cognitive_map, model_family
                )
                pert_resp = get_generative_prediction(
                    model, processor, item['image'], perturbed_caption, cognitive_map, model_family
                )
                
                norm_orig = metrics.normalize_response(orig_resp)
                norm_pert = metrics.normalize_response(pert_resp)
                consistency = metrics.check_text_consistency(norm_orig, norm_pert)

            elif model_type == 'contrastive':
                # Note: Maps are not typically used with contrastive models
                best_caption = get_contrastive_prediction(
                    model, processor, item['image'], original_caption, perturbed_caption, model_family
                )
                consistency = 'CONSISTENT' if best_caption == original_caption else 'INCONSISTENT'
            
            else:
                logger.error(f"Unknown model_type: {model_type}")
                consistency = 'ERROR'
            
            results.append({
                'consistency': consistency,
                'original_caption': original_caption,
                'perturbed_caption': perturbed_caption,
                'model_family': model_family,
                'model_type': model_type
            })
            
            # Clear cache periodically
            _clear_gpu_cache_if_needed(step, config)
            
        except Exception as e:
            logger.error(f"Error processing item {step}: {e}")
            results.append({
                'consistency': 'ERROR',
                'error': str(e),
                'model_family': model_family,
                'model_type': model_type
            })
    
    logger.info(f"Completed audit with {len(results)} results")
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
    
    # Get model info from results
    model_info = {}
    for result in results:
        if 'model_family' in result:
            model_info['family'] = result['model_family']
            model_info['type'] = result.get('model_type', 'unknown')
            break
    
    # Analyze by relation type if available
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
        'model_info': model_info,
        'total_items': total,
        'consistent': consistent,
        'inconsistent': inconsistent,
        'errors': errors,
        'overall_consistency_rate': consistent / total if total > 0 else 0,
        'error_rate': errors / total if total > 0 else 0,
        'relation_type_analysis': relation_stats
    }