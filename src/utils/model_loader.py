import os
import tempfile
from transformers import (
    AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig,
    Blip2Processor, Blip2ForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
    CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer,
    SiglipProcessor, SiglipModel
)
import torch
import logging

logger = logging.getLogger(__name__)

def load_llava_1_5_7b():
    logger.info("loading LLaVA-1.5-7B...")
    model_id = "llava-hf/llava-1.5-7b-hf"
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quantization_config, device_map="auto"
    )
    logger.info("LLaVA-1.5-7B loaded successfully.")
    return model, processor, "generative"

def load_clip_vit_b_32():
    logger.info("Loading CLIP-ViT-B/32...")
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, device_map="auto")
    logger.info("CLIP-ViT-B/32 loaded successfully.")
    return model, processor, "contrastive"

def load_siglip_base():
    logger.info("Loading SigLIP-Base-Patch16-224...")   
    try:
        model_id = "google/siglip-base-patch16-224"
        
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            
            processor = SiglipProcessor.from_pretrained(model_id)
            model = SiglipModel.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto"
            )
            
        except Exception as e:
            logger.warning(f"Quantized SigLIP loading failed: {e}. Trying without quantization...")
            
            #fallback without quantization
            processor = SiglipProcessor.from_pretrained(model_id)
            model = SiglipModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        logger.info("SigLIP-Base-Patch16-224 loaded successfully.")
        return model, processor, "contrastive"
        
    except Exception as e:
        logger.error(f"Failed to load SigLIP: {str(e)}")
        raise e

def load_siglip2_base():
    logger.info("Loading SigLIP2-Base-Patch16-224...")
    
    try:
        model_id = "google/siglip2-base-patch16-224"
        
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            
            processor = SiglipProcessor.from_pretrained(model_id)
            model = SiglipModel.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto"
            )
            
        except Exception as e:
            logger.warning(f"Quantized SigLIP2 loading failed: {e}. Trying without quantization...")
            
            #fallback without quantization
            processor = SiglipProcessor.from_pretrained(model_id)
            model = SiglipModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        logger.info("SigLIP2-Base-Patch16-224 loaded successfully.")
        return model, processor, "contrastive"
        
    except Exception as e:
        logger.error(f"Failed to load SigLIP2: {str(e)}")
        raise e


def load_qwen_vl_chat():
    logger.info("Loading Qwen-VL-Chat...")
    
    try:
        try:
            import transformers_stream_generator
        except ImportError:
            import subprocess
            import sys
            logger.info("Installing transformers_stream_generator...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers_stream_generator"])
        
        model_id = "Qwen/Qwen-VL-Chat"
        
        # ceate temporary directory for offloading
        offload_dir = tempfile.mkdtemp(prefix="qwen_offload_")
        logger.info(f"Using offload directory: {offload_dir}")
        
        #try loading with quantization first (uses less memory)
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                offload_folder=offload_dir
            )
            
        except Exception as e:
            logger.warning(f"Quantized loading failed: {e}. Trying with float16...")
            
            # Fallback to float16 without quantization
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                offload_folder=offload_dir,
                low_cpu_mem_usage=True
            )
        
        logger.info("Qwen-VL-Chat loaded successfully.")
        return model, processor, "generative"
        
    except Exception as e:
        logger.error(f"Failed to load Qwen-VL-Chat: {str(e)}")
        raise e

#utility function to get available models
def get_available_models():
    """returns a dictionary of available models and their load functions"""
    return {
        'llava-1.5-7b': load_llava_1_5_7b,
        'clip-vit-b-32': load_clip_vit_b_32,
        'siglip-base': load_siglip_base,
        'siglip2-base': load_siglip2_base,
        'qwen-vl-chat': load_qwen_vl_chat,
    }

#memory usage estimation (approximate)
def get_model_memory_requirements():
    """Returns approximate memory requirements for each model in GB."""
    return {
        'llava-1.5-7b': {'fp16': 14, '4bit': 4.5},
        'clip-vit-b-32': {'fp16': 1.2, '4bit': 0.6},
        'siglip-base': {'fp16': 1.2, '4bit': 0.6},
        'siglip2-base': {'fp16': 1.2, '4bit': 0.6},
        'qwen-vl-chat': {'fp16': 8, '4bit': 4},
    }

def can_load_together_on_colab_free(model_names, use_quantization=True):
    """
    estimates if the given models can be loaded together on Colab free tier
    Assumes ~12-13GB available GPU memory on T4.
    """
    memory_reqs = get_model_memory_requirements()
    total_memory = 0
    quantization_key = '4bit' if use_quantization else 'fp16'
    
    for model_name in model_names:
        if model_name in memory_reqs:
            total_memory += memory_reqs[model_name][quantization_key]
        else:
            logger.warning(f"Unknown model: {model_name}")
            return False
    
    #conservative estimate: 12GB available on Colab free tier
    colab_free_memory = 12.0
    can_fit = total_memory <= colab_free_memory
    
    logger.info(f"Estimated memory usage: {total_memory:.1f}GB")
    logger.info(f"Available memory: {colab_free_memory}GB")
    logger.info(f"Can fit on Colab free tier: {can_fit}")
    
    return can_fit, total_memory, colab_free_memory