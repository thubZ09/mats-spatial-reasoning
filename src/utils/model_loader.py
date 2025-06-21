# In src/utils/model_loader.py

from transformers import (
    AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig,
    Blip2Processor, Blip2ForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
    CLIPProcessor, CLIPModel
)
import torch
import logging

logger = logging.getLogger(__name__)

def load_llava_1_5_7b():
    """Loads the LLaVA 1.5 7B model."""
    logger.info("Loading LLaVA-1.5-7B...")
    model_id = "llava-hf/llava-1.5-7b-hf"
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )
    logger.info("LLaVA-1.5-7B loaded successfully.")
    return model, processor, "llava"

def load_blip2_opt_2_7b():
    """Loads the BLIP-2 OPT 2.7B model."""
    logger.info("Loading BLIP-2-opt-2.7b...")
    model_id = "Salesforce/blip2-opt-2.7b"
    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    logger.info("BLIP-2-opt-2.7b loaded successfully.")
    return model, processor, "blip"

def load_instructblip_vicuna_7b():
    """Loads the InstructBLIP Vicuna 7B model."""
    logger.info("Loading InstructBLIP-vicuna-7b...")
    model_id = "Salesforce/instructblip-vicuna-7b"
    processor = InstructBlipProcessor.from_pretrained(model_id)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    logger.info("InstructBLIP-vicuna-7b loaded successfully.")
    return model, processor, "blip"

def load_clip_vit_b_32():
    """Loads the CLIP ViT-B/32 model."""
    logger.info("Loading CLIP-ViT-B/32...")
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, device_map="auto")
    logger.info("CLIP-ViT-B/32 loaded successfully.")
    return model, processor, "clip"