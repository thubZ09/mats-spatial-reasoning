# In src/utils/model_loader.py

import os
import tempfile
from transformers import (
    AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig,
    Blip2Processor, Blip2ForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
    CLIPProcessor, CLIPModel, AutoModelForCausalLM
)
import torch
import logging

logger = logging.getLogger(__name__)

def load_llava_1_5_7b():
    logger.info("Loading LLaVA-1.5-7B...")
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

def load_qwen_vl_chat():
    """Loads the Qwen-VL-Chat model with proper disk offloading."""
    logger.info("Loading Qwen-VL-Chat...")
    
    try:
        # Install missing dependency if needed
        try:
            import transformers_stream_generator
        except ImportError:
            import subprocess
            import sys
            logger.info("Installing transformers_stream_generator...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers_stream_generator"])
        
        model_id = "Qwen/Qwen-VL-Chat"
        
        # Create temporary directory for offloading
        offload_dir = tempfile.mkdtemp(prefix="qwen_offload_")
        logger.info(f"Using offload directory: {offload_dir}")
        
        # Try loading with quantization first (uses less memory)
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

def load_biomed_clip():
    """Loads the Microsoft BiomedCLIP model with proper processor handling."""
    logger.info("Loading BiomedCLIP...")
    
    try:
        model_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        
        # Try loading the model first
        model = CLIPModel.from_pretrained(model_id, device_map="auto")
        
        # For BiomedCLIP, we need to use the base CLIP processor
        # since it doesn't have its own preprocessor_config.json
        try:
            processor = CLIPProcessor.from_pretrained(model_id)
        except Exception as e:
            logger.warning(f"Failed to load processor from model_id: {e}")
            logger.info("Using base CLIP processor instead...")
            
            # Fallback to base CLIP processor
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            
        logger.info("BiomedCLIP loaded successfully.")
        return model, processor, "contrastive"
        
    except Exception as e:
        logger.error(f"Failed to load BiomedCLIP: {str(e)}")
        raise e

def load_biomed_clip_alternative():
    """Alternative BiomedCLIP loader using manual processor configuration."""
    logger.info("Loading BiomedCLIP (alternative method)...")
    
    try:
        model_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        
        # Load the model
        model = CLIPModel.from_pretrained(model_id, device_map="auto")
        
        # Create processor manually with compatible configuration
        from transformers import CLIPTokenizer, CLIPImageProcessor
        
        # Use compatible tokenizer and image processor
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        # Combine them into a processor
        processor = CLIPProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer
        )
        
        logger.info("BiomedCLIP loaded successfully with manual processor.")
        return model, processor, "contrastive"
        
    except Exception as e:
        logger.error(f"Failed to load BiomedCLIP (alternative): {str(e)}")
        raise e