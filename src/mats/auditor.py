# In src/mats/auditor.py mod
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
    """Detect the specific model type and family for proper handling."""
    model_name = getattr(model.config, '_name_or_path', '').lower()
    
    if 'llava' in model_name:
        return 'llava', 'generative'
    elif 'moondream' in model_name:
        return 'moondream', 'generative'
    elif 'qwen' in model_name:
        return 'qwen', 'generative'
    elif 'siglip' in model_name:
        return 'siglip', 'contrastive'
    elif 'clip' in model_name:
        return 'clip', 'contrastive'
    elif hasattr(model, 'get_image_features') and hasattr(model, 'get_text_features'):
        return 'clip_like', 'contrastive'
    elif hasattr(model, 'generate'):
        return 'generic_generative', 'generative'
    else:
        logger.warning(f"Unknown model type: {model_name}. Attempting generic approach.")
        return 'unknown', 'unknown'

def build_prompt(caption: str, cognitive_map: str = None, model_family: str = 'generic') -> str:
    """builds the prompt, optionally including a cognitive map, tailored for specific models"""
    base_question = f"Based on the image, is the following statement true or false? \"{caption}\""
    
    if cognitive_map:
        prompt_text = (
            "A ground-truth 'cognitive map' of the scene is provided in JSON format.\n"
            "Based on both the image and this map, is the following statement true or false?\n\n"
            f"Cognitive Map:\n```json\n{cognitive_map}\n```\n\n"
            f"Statement: \"{caption}\"\n\n"
            "Answer with only the single word 'true' or 'false'."
        )
    else:
        prompt_text = base_question
    
    return prompt_text

def get_generative_prediction(model, processor, image, caption: str, cognitive_map: str = None, model_family: str = 'generic') -> str:
    """gets a 'true'/'false' prediction from a generative model with model-specific handling"""
    prompt_text = build_prompt(caption, cognitive_map, model_family)

    try:
        gen_kwargs = {"max_new_tokens": 10, "do_sample": False}
        
        if model_family == 'moondream':
            # FIXED MOONDREAM2 HANDLING
            # Ensure image is properly prepared and on correct device with correct dtype
            if isinstance(image, Image.Image):
                # Convert PIL image to the format Moondream expects
                image_tensor = processor(image).unsqueeze(0)
            else:
                image_tensor = image
            
            # Ensure correct device and dtype - critical for avoiding Half/Byte dtype errors
            if hasattr(model, 'device'):
                image_tensor = image_tensor.to(model.device)
            if hasattr(model, 'dtype') and model.dtype != torch.uint8:
                image_tensor = image_tensor.to(dtype=model.dtype)
            elif not hasattr(model, 'dtype'):
                # Default to float32 if model dtype is unclear
                image_tensor = image_tensor.to(dtype=torch.float32)
            
            # Use Moondream's specific interface correctly
            try:
                # Encode image first
                image_embeds = model.encode_image(image_tensor)
                # Use answer_question method with proper arguments
                response = model.answer_question(
                    image_embeds=image_embeds, 
                    question=prompt_text, 
                    tokenizer=processor
                )
                return response.strip()
            except Exception as moondream_error:
                logger.warning(f"Moondream answer_question failed: {moondream_error}")
                # Fallback to generate method if available
                if hasattr(model, 'generate'):
                    # Try standard generation as backup
                    inputs = processor(text=prompt_text, images=image, return_tensors="pt")
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        output_ids = model.generate(**inputs, **gen_kwargs)
                    response = processor.batch_decode(output_ids, skip_special_tokens=True)[-1]
                    return response.strip()
                else:
                    raise moondream_error
            
        elif model_family == 'qwen':
            # Qwen-specific formatting
            if processor.tokenizer.chat_template is None:
                processor.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] }}{% endif %}{% endfor %}"
            conversation = [{'image': image}, {'text': prompt_text}]
            full_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(model.device)
            
        else: # LLaVA-style formatting
            full_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
            inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(model.device)
            if hasattr(processor.tokenizer, 'pad_token_id') and processor.tokenizer.pad_token_id is not None:
                gen_kwargs['pad_token_id'] = processor.tokenizer.pad_token_id

        # For non-moondream models, use standard generation
        if model_family != 'moondream':
            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)
            
            response = processor.batch_decode(output_ids, skip_special_tokens=True)[-1]
            return response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response.strip()

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
    """Gets a prediction from a contrastive model with comprehensive SigLIP and CLIP handling."""
    try:
        # Process inputs
        inputs = processor(text=[caption1, caption2], images=image, return_tensors="pt", padding=True)

        # COMPREHENSIVE FIX FOR SIGLIP AND OTHER CONTRASTIVE MODELS
        # Ensure attention_mask exists - this is critical for SigLIP
        if 'attention_mask' not in inputs:
            # Create attention mask based on input_ids if it doesn't exist
            if 'input_ids' in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            else:
                logger.warning("No input_ids found to create attention_mask")
        
        # Enhanced device and dtype casting with error handling
        device = getattr(model, 'device', torch.device('cpu'))
        
        for key in list(inputs.keys()):
            if isinstance(inputs[key], torch.Tensor):
                try:
                    if key == 'pixel_values':
                        # Handle pixel values with proper dtype
                        if hasattr(model, 'dtype') and model.dtype != torch.uint8:
                            inputs[key] = inputs[key].to(device, dtype=model.dtype)
                        else:
                            inputs[key] = inputs[key].to(device, dtype=torch.float32)
                    else:
                        # Handle text inputs (input_ids, attention_mask, etc.)
                        inputs[key] = inputs[key].to(device)
                except Exception as tensor_error:
                    logger.warning(f"Failed to move {key} to device: {tensor_error}")
                    # Remove problematic tensor rather than crash
                    del inputs[key]
        
        # Validate we have minimum required inputs
        required_keys = ['input_ids', 'pixel_values']
        missing_keys = [key for key in required_keys if key not in inputs]
        if missing_keys:
            logger.error(f"Missing required inputs: {missing_keys}")
            return "ERROR"
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Handle different output formats
            if hasattr(outputs, 'logits_per_image'):  
                # CLIP-style output
                logits = outputs.logits_per_image
            elif hasattr(outputs, 'logits'):  
                # SigLIP-style output
                logits = outputs.logits
            elif hasattr(outputs, 'similarity_scores'):
                # Alternative similarity output
                logits = outputs.similarity_scores
            else:
                # Try to find any tensor output that could be logits
                for attr_name in dir(outputs):
                    attr_value = getattr(outputs, attr_name)
                    if isinstance(attr_value, torch.Tensor) and attr_value.dim() >= 1:
                        logits = attr_value
                        break
                else:
                    logger.error("Could not find logits in model output")
                    return "ERROR"
            
            # Get the best match
            if logits.dim() > 1:
                best_match_index = logits.argmax(dim=-1).item()
            else:
                best_match_index = logits.argmax().item()
        
        return caption1 if best_match_index == 0 else caption2

    except Exception as e:
        logger.error(f"Error in contrastive prediction for {model_family}: {e}")
        logger.error(f"Available inputs keys: {list(inputs.keys()) if 'inputs' in locals() else 'N/A'}")
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

def run_comparative_text_audit(model, processor, dataset, use_maps: bool = False):
    model_family, model_type = detect_model_type(model, processor)
    
    results = []
    audit_mode = "Map-Assisted" if use_maps else "Baseline"
    
    for item in tqdm(dataset, desc=f"Auditing {audit_mode} ({model_family})"):
        original_caption = item['caption']
        perturbed_caption, was_changed = perturbations.perturb_spatial_words(original_caption)
        if not was_changed:
            continue

        cognitive_map = item.get("cognitive_map") if use_maps else None
        
        if model_type == 'generative':
            orig_resp = get_generative_prediction(model, processor, item['image'], original_caption, cognitive_map, model_family)
            pert_resp = get_generative_prediction(model, processor, item['image'], perturbed_caption, cognitive_map, model_family)
            
            # Clean responses using the cleaning functions
            norm_orig = clean_generative_response(orig_resp, model_family)
            norm_pert = clean_generative_response(pert_resp, model_family)
            
            consistency = metrics.check_text_consistency(norm_orig, norm_pert)
        elif model_type == 'contrastive':
            best_caption = get_contrastive_prediction(model, processor, item['image'], original_caption, perturbed_caption, model_family)
            consistency = 'CONSISTENT' if best_caption == original_caption else 'INCONSISTENT'
        else:
            consistency = 'ERROR'
            
        results.append({
            'consistency': consistency, 
            'relation_type': item.get('relation_type'),
            'model_family': model_family,
            'model_type': model_type
        })
        
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