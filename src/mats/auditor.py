# In src/mats/auditor.py
import torch
import logging
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

def get_generative_prediction(model, processor, image, caption: str):
    """
    Gets a 'true'/'false' prediction from a generative model.
    Handles Qwen-VL and LLaVA/BLIP style models robustly.
    """
    try:
        # Detect Qwen model by config or class name
        is_qwen_model = "qwen" in getattr(model.config, 'model_type', '').lower() or "qwen" in model.__class__.__name__.lower()
        gen_kwargs = {"max_new_tokens": 10, "do_sample": False}

        if is_qwen_model:
            query = f"Based on the image, is this statement true or false? \"{caption}\""
            # Qwen expects chat format: [{'role':..., 'content':...}]
            conversation = [{"role": "user", "content": query}]

            # Check for tokenizer and if it has chat_template
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer and hasattr(tokenizer, "chat_template") and getattr(tokenizer, "chat_template", None) is None:
                tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\\n' }}"
                    "{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '\\n' }}"
                    "{% endif %}{% endfor %}"
                )
            # Get chat template method (tokenizer or processor)
            if tokenizer and hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            elif hasattr(processor, "apply_chat_template"):
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            else:
                raise RuntimeError("No apply_chat_template method found on processor or tokenizer.")

            # Prepare inputs and move to model device
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        else:
            # LLaVA/BLIP and similar models
            prompt = f"USER: <image>\nBased on the image, is the following statement true or false? \"{caption}\"\nASSISTANT:"
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, 'pad_token_id') and processor.tokenizer.pad_token_id is not None:
                gen_kwargs['pad_token_id'] = processor.tokenizer.pad_token_id

        with torch.no_grad():
            generate_ids = model.generate(**inputs, **gen_kwargs)

        response = processor.batch_decode(generate_ids, skip_special_tokens=True)[-1]
        return response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response.strip()
    except Exception as e:
        logger.error(f"Error in generative prediction: {e}")
        return "ERROR"

def get_contrastive_prediction(model, processor, image, caption1, caption2):
    try:
        is_biomed_clip = "biomedclip" in model.__class__.__name__.lower()

        inputs = processor(text=[caption1, caption2], images=image, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        if is_biomed_clip:
            # For BiomedCLIP, call directly with images and texts
            with torch.no_grad():
                outputs = model(images=image, texts=[caption1, caption2])
                best_match_index = outputs.logits_per_image.argmax().item()
                return caption1 if best_match_index == 0 else caption2
        else:
            # For other contrastive models (CLIP, etc)
            inputs = processor(text=[caption1, caption2], images=image, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                best_match_index = outputs.logits_per_image.argmax().item()
                return caption1 if best_match_index == 0 else caption2
    except Exception as e:
        logger.error(f"Error in contrastive prediction: {e}")
        return "ERROR"
    
def _clear_gpu_cache_if_needed(step: int, config: AuditorConfig) -> None:
    """Clear GPU cache periodically to prevent memory issues."""
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

def run_comparative_text_audit(model, processor, model_type: str, dataset, 
                             config: Optional[AuditorConfig] = None) -> List[Dict[str, Any]]:
    """
    A unified audit function that intelligently handles both generative and contrastive models.
    
    Args:
        model: The model to audit
        processor: The model processor
        model_type: Either 'generative' or 'contrastive'
        dataset: Dataset containing items with 'image', 'caption', and 'relation_type' keys
        config: Configuration object with audit parameters
    
    Returns:
        List[Dict]: Results of the audit with consistency information
    
    Raises:
        ValueError: If inputs are invalid
        Exception: If audit fails completely
    """
    # Validate inputs
    _validate_inputs(model, processor, model_type, dataset)
    
    if config is None:
        config = AuditorConfig()
    
    results = []
    error_count = 0
    skipped_count = 0
    
    logger.info(f"Running Comparative Text Audit for: {model_type.upper()}")
    logger.info(f"Dataset size: {len(dataset)}")
    
    try:
        for step, item in enumerate(tqdm(dataset, desc=f"Auditing ({model_type})")):
            try:
                # Validate item structure
                required_keys = ['image', 'caption', 'relation_type']
                missing_keys = [key for key in required_keys if key not in item]
                if missing_keys:
                    logger.warning(f"Skipping item {step}: missing keys {missing_keys}")
                    skipped_count += 1
                    continue
                
                image = item['image']
                original_caption = item['caption']
                relation_type = item['relation_type']
                
                # Perturb the caption
                perturbed_caption, was_changed = perturbations.perturb_spatial_words(original_caption)
                if not was_changed:
                    skipped_count += 1
                    continue  # Skip samples that can't be perturbed
                
                if model_type == 'generative':
                    # --- Logic for LLaVA
                    orig_response = get_generative_prediction(model, processor, image, original_caption)
                    pert_response = get_generative_prediction(model, processor, image, perturbed_caption)
                    
                    # Handle error responses
                    if orig_response == "ERROR" or pert_response == "ERROR":
                        error_count += 1
                        consistency = 'ERROR'
                    else:
                        norm_orig = metrics.normalize_response(orig_response)
                        norm_pert = metrics.normalize_response(pert_response)
                        consistency = metrics.check_text_consistency(norm_orig, norm_pert)
                    
                elif model_type == 'contrastive':
                    # --- Logic for CLIP ---
                    # We ask CLIP: which of these two sentences better describes the image?
                    best_caption = get_contrastive_prediction(model, processor, image, original_caption, perturbed_caption)
                    
                    if best_caption == "ERROR":
                        error_count += 1
                        consistency = 'ERROR'
                        orig_response = "ERROR"
                        pert_response = "ERROR"
                    else:
                        # For CLIP, "consistency" means it correctly chose the original, true caption.
                        consistency = 'CONSISTENT' if best_caption == original_caption else 'INCONSISTENT'
                        orig_response = "N/A (Contrastive)"
                        pert_response = f"CLIP chose: '{best_caption}'"
                
                results.append({
                    'item_id': step,
                    'caption': original_caption,
                    'perturbed_caption': perturbed_caption,
                    'orig_response': orig_response,
                    'pert_response': pert_response,
                    'consistency': consistency,
                    'relation_type': relation_type
                })
                
                # Periodic memory cleanup
                _clear_gpu_cache_if_needed(step, config)
                
            except Exception as e:
                logger.error(f"Error processing item {step}: {e}")
                error_count += 1
                continue
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log summary statistics
        total_processed = len(results)
        logger.info(f"Audit completed:")
        logger.info(f"  - Total items processed: {total_processed}")
        logger.info(f"  - Items skipped: {skipped_count}")
        logger.info(f"  - Errors encountered: {error_count}")
        
        if total_processed > 0:
            consistent_count = sum(1 for r in results if r['consistency'] == 'CONSISTENT')
            consistency_rate = consistent_count / total_processed
            logger.info(f"  - Consistency rate: {consistency_rate:.2%}")
        
        return results
        
    except Exception as e:
        logger.error(f"Fatal error during audit: {e}")
        raise

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