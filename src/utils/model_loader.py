# In src/utils/model_loader.py

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
    return model, processor, "generative" # Return model type

def load_clip_vit_b_32():
    logger.info("Loading CLIP-ViT-B/32...")
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, device_map="auto")
    logger.info("CLIP-ViT-B/32 loaded successfully.")
    return model, processor, "contrastive" # Return model type


def load_qwen_vl_chat():
    """Loads the Qwen-VL-Chat model."""
    logger.info("Loading Qwen-VL-Chat...")
    model_id = "Qwen/Qwen-VL-Chat"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    logger.info("Qwen-VL-Chat loaded successfully.")
    return model, processor, "generative" # Return model type

def load_biomed_clip():
    """Loads the Microsoft BiomedCLIP model."""
    logger.info("Loading BiomedCLIP...")
    model_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, device_map="auto")
    logger.info("BiomedCLIP loaded successfully.")
    return model, processor, "contrastive" # Return model type
