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
    """
    Loads the Microsoft BiomedCLIP model using OpenCLIP.
    This model requires open_clip_torch package.
    """
    logger.info("Loading BiomedCLIP...")
    
    try:
        # Install open_clip_torch if not available
        try:
            import open_clip
        except ImportError:
            import subprocess
            import sys
            logger.info("Installing open_clip_torch...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "open_clip_torch"])
            import open_clip
        
        # Load the model using OpenCLIP
        model, preprocess = open_clip.create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        tokenizer = open_clip.get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        
        # Create a wrapper to make it compatible with your existing code
        class BiomedCLIPWrapper:
            def __init__(self, model, tokenizer, preprocess):
                self.model = model
                self.tokenizer = tokenizer
                self.preprocess = preprocess
                self.device = next(model.parameters()).device
                
            def to(self, device):
                self.model = self.model.to(device)
                self.device = device
                return self
                
            def eval(self):
                self.model.eval()
                return self
                
            def __call__(self, images, texts):
                # This mimics the OpenCLIP interface
                with torch.no_grad():
                    image_features, text_features, logit_scale = self.model(images, texts)
                    return image_features, text_features, logit_scale
        
        class BiomedCLIPProcessor:
            def __init__(self, tokenizer, preprocess):
                self.tokenizer = tokenizer
                self.preprocess = preprocess
                self.context_length = 256
                
            def __call__(self, text=None, images=None, return_tensors="pt"):
                results = {}
                
                if text is not None:
                    # Handle text input
                    if isinstance(text, str):
                        text = [text]
                    tokens = self.tokenizer(text, context_length=self.context_length)
                    results['input_ids'] = tokens
                    
                if images is not None:
                    # Handle image input
                    if not isinstance(images, list):
                        images = [images]
                    
                    processed_images = []
                    for img in images:
                        if hasattr(img, 'convert'):  # PIL Image
                            processed_img = self.preprocess(img)
                        else:  # Already a tensor
                            processed_img = img
                        processed_images.append(processed_img)
                    
                    if len(processed_images) == 1:
                        results['pixel_values'] = processed_images[0].unsqueeze(0)
                    else:
                        results['pixel_values'] = torch.stack(processed_images)
                
                return results
        
        # Wrap the model and processor
        wrapped_model = BiomedCLIPWrapper(model, tokenizer, preprocess)
        wrapped_processor = BiomedCLIPProcessor(tokenizer, preprocess)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        wrapped_model.to(device)
        wrapped_model.eval()
        
        logger.info("BiomedCLIP loaded successfully.")
        return wrapped_model, wrapped_processor, "contrastive"
        
    except Exception as e:
        logger.error(f"Failed to load BiomedCLIP: {str(e)}")
        raise e

# Alternative: Use a different biomedical model that works with transformers
def load_biomed_clip_alternative():
    """
    Alternative: Load a different biomedical vision model that works with transformers.
    """
    logger.info("Loading alternative biomedical vision model...")
    
    try:
        # Try using a different model that might work better
        model_id = "flaviagiammarino/pubmed-clip-vit-base-patch32"
        
        try:
            processor = CLIPProcessor.from_pretrained(model_id)
            model = CLIPModel.from_pretrained(model_id, device_map="auto")
            
            logger.info("Alternative biomedical CLIP model loaded successfully.")
            return model, processor, "contrastive"
            
        except Exception:
            # If that doesn't work, fall back to regular CLIP
            logger.warning("Alternative biomedical model failed, using regular CLIP...")
            model_id = "openai/clip-vit-base-patch32"
            processor = CLIPProcessor.from_pretrained(model_id)
            model = CLIPModel.from_pretrained(model_id, device_map="auto")
            
            logger.info("Regular CLIP loaded as fallback.")
            return model, processor, "contrastive"
            
    except Exception as e:
        logger.error(f"Failed to load alternative model: {str(e)}")
        raise e
