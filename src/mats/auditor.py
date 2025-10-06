import torch
import json
import logging
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple

from . import perturbations, metrics
from .metrics import MATSMetrics, normalize_response

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
    """to detect the specific model type and family for proper handling"""
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
    """gets a 'T/F' prediction with model-specific handling"""
    prompt_text = build_prompt(caption, cognitive_map, model_family)

    try:
        gen_kwargs = {"max_new_tokens": 10, "do_sample": False}
            #Did not use this (moondream) in the final version
        if model_family == 'moondream':
            try:
                if hasattr(model, 'answer_question'):
                    if isinstance(image, Image.Image):
                        processed_image = processor(image)
                        if isinstance(processed_image, dict) and 'pixel_values' in processed_image:
                            image_tensor = processed_image['pixel_values']
                        else:
                            image_tensor = processed_image
                    else:
                        image_tensor = image
                    
                    if not isinstance(image_tensor, torch.Tensor):
                        image_tensor = torch.tensor(image_tensor)
                    
                    if image_tensor.dim() == 3:
                        image_tensor = image_tensor.unsqueeze(0)
                    
                    device = getattr(model, 'device', torch.device('cpu'))
                    image_tensor = image_tensor.to(device)
                    
                    if image_tensor.dtype == torch.uint8:
                        image_tensor = image_tensor.float() / 255.0
                    elif image_tensor.dtype != torch.float32:
                        image_tensor = image_tensor.to(torch.float32)
                    
                    with torch.no_grad():
                        image_embeds = model.encode_image(image_tensor)
                        
                        if isinstance(prompt_text, list):
                            prompt_text = prompt_text[0]
                        
                        response = model.answer_question(
                            image_embeds=image_embeds, 
                            question=str(prompt_text),  
                            tokenizer=processor.tokenizer if hasattr(processor, 'tokenizer') else processor
                        )
                        return str(response).strip()
                
                #fallback to generate method
                elif hasattr(model, 'generate'):
                    if hasattr(processor, 'tokenizer'):
                        text_inputs = processor.tokenizer(prompt_text, return_tensors="pt")
                        if isinstance(image, Image.Image):
                            image_inputs = processor(images=image, return_tensors="pt")
                        else:
                            image_inputs = {"pixel_values": image}
                        
                        inputs = {**text_inputs, **image_inputs}
                    else:
                        inputs = processor(text=prompt_text, images=image, return_tensors="pt")
                    
                    device = getattr(model, 'device', torch.device('cpu'))
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    
                    if 'pixel_values' in inputs:
                        pixel_values = inputs['pixel_values']
                        if pixel_values.dtype == torch.uint8:
                            inputs['pixel_values'] = pixel_values.float() / 255.0
                        elif pixel_values.dtype != torch.float32:
                            inputs['pixel_values'] = pixel_values.to(torch.float32)
                    
                    with torch.no_grad():
                        output_ids = model.generate(**inputs, **gen_kwargs)
                    
                    response = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True) if hasattr(processor, 'tokenizer') else processor.decode(output_ids[0], skip_special_tokens=True)
                    
                    if prompt_text in response:
                        response = response.replace(prompt_text, "").strip()
                    
                    return response.strip()
                
                else:
                    logger.error("Moondream model has no supported generation method")
                    return "ERROR"
                    
            except Exception as moondream_error:
                logger.error(f"Moondream processing failed: {moondream_error}")
                return "ERROR"
            
        elif model_family == 'qwen':
            #Qwen-specific formatting
            if processor.tokenizer.chat_template is None:
                processor.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] }}{% endif %}{% endfor %}"
            conversation = [{'image': image}, {'text': prompt_text}]
            full_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(model.device)
            
        else: #LLaVA-style formatting
            full_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
            inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(model.device)
            if hasattr(processor.tokenizer, 'pad_token_id') and processor.tokenizer.pad_token_id is not None:
                gen_kwargs['pad_token_id'] = processor.tokenizer.pad_token_id

        #for non-moondream models, we use standard generation
        if model_family != 'moondream':
            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)
            
            response = processor.batch_decode(output_ids, skip_special_tokens=True)[-1]
            return response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response.strip()

    except Exception as e:
        logger.error(f"Error in generative prediction for {model_family}: {e}")
        return "ERROR"

def clean_generative_response(response: str, model_family: str) -> str:
    """clean and normalize responses from different generative models"""
    
    #remove common prefixes/suffixes
    response = response.strip()
    
    if model_family == 'llava':
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
    
    elif model_family == 'moondream':
        return clean_moondream_response(response)
        
    elif model_family == 'qwen':
        #qwen sometimes includes the original prompt
        if "Answer with only" in response:
            response = response.split("Answer with only")[-1].strip()
    
    #generic cleaning
    response = response.lower().strip()
    
    #extract true/false from common patterns
    if 'true' in response and 'false' not in response:
        return 'true'
    elif 'false' in response and 'true' not in response:
        return 'false'
    elif response.startswith('true'):
        return 'true'
    elif response.startswith('false'):
        return 'false'
    
    #handle yes/no responses
    if response.startswith('yes'):
        return 'true'
    elif response.startswith('no'):
        return 'false'
    
    return response.strip()

#Not included in the final experiment
def clean_moondream_response(response: str) -> str:
    """Specific cleaning for Moondream2 responses"""
    response = response.strip().lower()
    
    #moondream sometimes gives verbose answers
    #look for key words in the response
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
    """gets a prediction from a contrastive model with comprehensive SigLIP and CLIP handling."""
    try:
        
        # Get model device and dtype information
        device = getattr(model, 'device', torch.device('cpu'))
        model_dtype = getattr(model, 'dtype', torch.float32)
        
        #process inputs with proper error handling
        try:
            inputs = processor(text=[caption1, caption2], images=image, return_tensors="pt", padding=True)
        except Exception as proc_error:
            logger.error(f"Processor failed: {proc_error}")
            #try alternative processing
            try:
                inputs = processor(text=[caption1, caption2], images=[image], return_tensors="pt", padding=True)
            except Exception:
                return "ERROR"

        #CRITICAL FIX FOR SIGLIP DTYPE ISSUES
        #ensuring attention_mask exists and handle all tensor dtypes properly
        if 'attention_mask' not in inputs and 'input_ids' in inputs:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        
        #enhanced device and dtype casting with comprehensive error handling
        for key in list(inputs.keys()):
            if isinstance(inputs[key], torch.Tensor):
                try:
                    inputs[key] = inputs[key].to(device)
                    
                    #handle dtype conversion based on tensor type
                    if key == 'pixel_values':
                        #img tensors - handle byte to float conversion carefully
                        if inputs[key].dtype == torch.uint8:
                            #convert uint8 to float and normalize
                            inputs[key] = inputs[key].float() / 255.0
                        elif inputs[key].dtype != model_dtype and model_dtype != torch.uint8:
                            #convert to model dtype if it's not uint8
                            if model_dtype == torch.float16:
                                inputs[key] = inputs[key].to(torch.float16)
                            else:
                                inputs[key] = inputs[key].to(torch.float32)
                        #special handling for SigLIP which might expect specific dtypes
                        if model_family == 'siglip' and inputs[key].dtype == torch.float64:
                            inputs[key] = inputs[key].to(torch.float32)
                            
                    elif key in ['input_ids', 'attention_mask']:
                        #text tensors should remain as long integers
                        if inputs[key].dtype != torch.long:
                            inputs[key] = inputs[key].to(torch.long)
                    
                    else:
                        #other tensors - convert to appropriate dtype
                        if inputs[key].dtype == torch.uint8 and key != 'pixel_values':
                            inputs[key] = inputs[key].to(torch.long)
                        elif model_dtype != torch.uint8 and inputs[key].dtype != model_dtype:
                            if inputs[key].dtype in [torch.uint8, torch.int8] and model_dtype in [torch.float16, torch.float32]:
                                inputs[key] = inputs[key].to(model_dtype)
                                
                except Exception as tensor_error:
                    logger.warning(f"Failed to process tensor {key}: {tensor_error}")
                    #for non-critical tensors, we can try to continue without them
                    if key not in ['input_ids', 'pixel_values']:
                        logger.warning(f"Removing problematic tensor: {key}")
                        del inputs[key]
                    else:
                        #for critical tensors, this is an error
                        logger.error(f"Critical tensor {key} failed processing")
                        return "ERROR"
        
        #validate we have minimum required inputs
        required_keys = ['input_ids', 'pixel_values']
        missing_keys = [key for key in required_keys if key not in inputs]
        if missing_keys:
            logger.error(f"Missing required inputs: {missing_keys}")
            return "ERROR"
        
        #additional validation for tensor shapes and dtypes
        try:
            if 'pixel_values' in inputs:
                pv = inputs['pixel_values']
                logger.debug(f"pixel_values shape: {pv.shape}, dtype: {pv.dtype}")
                
            if 'input_ids' in inputs:
                ii = inputs['input_ids']
                logger.debug(f"input_ids shape: {ii.shape}, dtype: {ii.dtype}")
        except Exception as debug_error:
            logger.debug(f"Debug logging failed: {debug_error}")
        
        #model inference with comprehensive error handling
        with torch.no_grad():
            try:
                outputs = model(**inputs)
            except RuntimeError as runtime_error:
                error_msg = str(runtime_error)
                if "dtype" in error_msg.lower() and ("half" in error_msg.lower() or "byte" in error_msg.lower()):
                    logger.error(f"Dtype mismatch error: {error_msg}")
                    #try to fix dtype issues by ensuring all tensors are float32
                    for key in inputs:
                        if isinstance(inputs[key], torch.Tensor) and inputs[key].dtype in [torch.uint8, torch.int8, torch.float16]:
                            if key == 'pixel_values':
                                if inputs[key].dtype == torch.uint8:
                                    inputs[key] = inputs[key].float() / 255.0
                                else:
                                    inputs[key] = inputs[key].to(torch.float32)
                            elif key not in ['input_ids', 'attention_mask']:
                                inputs[key] = inputs[key].to(torch.float32)
                    
                    #retry inference
                    try:
                        outputs = model(**inputs)
                    except Exception as retry_error:
                        logger.error(f"Retry failed: {retry_error}")
                        return "ERROR"
                else:
                    raise runtime_error
            
            #handle different output formats
            if hasattr(outputs, 'logits_per_image'):  
                #CLIP-style output
                logits = outputs.logits_per_image
            elif hasattr(outputs, 'logits'):  
                #SigLIP-style output
                logits = outputs.logits
            elif hasattr(outputs, 'similarity_scores'):
                #alternative similarity output
                logits = outputs.similarity_scores
            else:
                #try to find any tensor output that could be logits
                for attr_name in dir(outputs):
                    if not attr_name.startswith('_'):
                        attr_value = getattr(outputs, attr_name)
                        if isinstance(attr_value, torch.Tensor) and attr_value.dim() >= 1:
                            logits = attr_value
                            logger.debug(f"Using {attr_name} as logits tensor")
                            break
                else:
                    logger.error("Could not find logits in model output")
                    logger.error(f"Available attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
                    return "ERROR"
            
            #get the best match
            if logits.dim() > 1:
                best_match_index = logits.argmax(dim=-1).item()
            else:
                best_match_index = logits.argmax().item()
        
        return caption1 if best_match_index == 0 else caption2

    except Exception as e:
        logger.error(f"Error in contrastive prediction for {model_family}: {e}")
        logger.error(f"Available inputs keys: {list(inputs.keys()) if 'inputs' in locals() else 'N/A'}")
        logger.error(f"Model device: {getattr(model, 'device', 'unknown')}")
        logger.error(f"Model dtype: {getattr(model, 'dtype', 'unknown')}")
        return "ERROR"


def _clear_gpu_cache_if_needed(step: int, config: AuditorConfig) -> None:
    """clear GPU cache periodically to prevent memory issues"""
    if torch.cuda.is_available() and step % config.clear_cache_frequency == 0:
        torch.cuda.empty_cache()
        logger.debug(f"Cleared GPU cache at step {step}")

def _validate_inputs(model, processor, model_type: str, dataset) -> None:
    """validate inputs to the audit function"""
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
            
            #clean responses using the cleaning functions
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

def compute_mats_metrics(audit_results: List[Dict]) -> Dict:
    """convert audit results to metrics"""
    metrics = MATSMetrics()
    
    scs_data = []
    for result in audit_results:
        if 'original_response' in result:
            scs_data.append({
                'relation': result.get('relation_type', 'unknown'),
                'original_raw': result['original_response'],
                'inverted_raw': result['perturbed_response']
            })
    
    eval_result = metrics.evaluate_model('YourModel', scs_data, [])  #compute metrics
    
    return {
        'scs': eval_result.scs * 100,  #convert to percentage
        'per_relation_scs': {k: v*100 for k, v in eval_result.per_relation_scs.items()},
        'coverage': eval_result.coverage * 100
    }

def analyze_audit_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    analyze the results of an audit and provide summary statistics   
    Args:
        results: List of audit results   
    Returns:
        dict containing analysis summary
    """
    if not results:
        return {"error": "No results to analyze"}
    
    total = len(results)
    consistent = sum(1 for r in results if r['consistency'] == 'CONSISTENT')
    inconsistent = sum(1 for r in results if r['consistency'] == 'INCONSISTENT')
    errors = sum(1 for r in results if r['consistency'] == 'ERROR')
    
    #get model info from results
    model_info = {}
    for result in results:
        if 'model_family' in result:
            model_info['family'] = result['model_family']
            model_info['type'] = result.get('model_type', 'unknown')
            break
    
    #analyze by relation type if available
    relation_stats = {}
    for result in results:
        rel_type = result.get('relation_type', 'unknown')
        if rel_type not in relation_stats:
            relation_stats[rel_type] = {'total': 0, 'consistent': 0}
        relation_stats[rel_type]['total'] += 1
        if result['consistency'] == 'CONSISTENT':
            relation_stats[rel_type]['consistent'] += 1
    
    #calculate consistency rates by relation type
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