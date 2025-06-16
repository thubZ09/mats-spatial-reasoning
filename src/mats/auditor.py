import torch
from tqdm import tqdm
from .perturbations import perturb_spatial_words
from .metrics import normalize_response, check_logical_consistency # Import both
from .perturbations import rotate_image

def run_text_perturbation_audit(model, processor, dataset):
    results = []
    
    for item in tqdm(dataset, desc="Auditing dataset"): # Note: now iterates over a simple list
        original_caption = item['caption']
        
        # Helper function for getting raw predictions
        def get_raw_prediction(caption_text):
            from PIL import Image
            dummy_image = Image.new('RGB', (640, 480), color = 'white')
            prompt = f"USER: <image>\nLook at this image and evaluate: {caption_text}\nIs this statement true or false? Answer with just 'True' or 'False'. ASSISTANT:"
            inputs = processor(text=prompt, images=dummy_image, return_tensors="pt").to("cuda")
            with torch.no_grad():
                generate_ids = model.generate(**inputs, max_new_tokens=10)
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return response.split("ASSISTANT:")[-1].strip()

        # 1. Get raw predictions
        raw_original_pred = get_raw_prediction(original_caption)
        
        perturbed_caption = perturb_spatial_words(original_caption)
        raw_perturbed_pred = get_raw_prediction(perturbed_caption)
        
        # 2. Normalize the raw predictions
        norm_original_pred = normalize_response(raw_original_pred)
        norm_perturbed_pred = normalize_response(raw_perturbed_pred)

        # 3. Check for logical consistency
        consistency = check_logical_consistency(norm_original_pred, norm_perturbed_pred)
        
        results.append({
            'original_caption': original_caption,
            'perturbed_caption': perturbed_caption,
            'raw_original_pred': raw_original_pred,
            'raw_perturbed_pred': raw_perturbed_pred,
            'norm_original_pred': norm_original_pred,
            'norm_perturbed_pred': norm_perturbed_pred,
            'consistency': consistency
        })
        
    return results

def run_visual_perturbation_audit(model, processor, dataset):
    """
    Runs a visual perturbation audit (rotation) on a given dataset.

    Args:
        model: The loaded vision-language model.
        processor: The model's processor.
        dataset: A list of dictionaries, where each dict has an 'image' and 'caption'.

    Returns:
        A list of dictionaries, where each dictionary contains an audit result.
    """
    results = []

    # Note: This requires real images, so we'll need to adapt our data loading.
    # For now, we will build the logic assuming we have real PIL images.
    # We will simulate this with a dummy image for the first test run.

    for item in tqdm(dataset, desc="Auditing with Visual Perturbation"):
        caption = item['caption']
        
        # In a real run, this would be: original_image = item['image']
        # For now, let's create a dummy image to test the flow.
        try:
            from PIL import Image
            # A simple dummy image with a distinct feature to see rotation
            original_image = Image.new('RGB', (200, 200), 'blue')
            top_half = Image.new('RGB', (200, 100), 'red')
            original_image.paste(top_half, (0, 0))
        except ImportError:
            print("Pillow is not installed. Skipping image creation.")
            original_image = None


        # 1. Create the perturbed version (rotated image)
        if original_image is None:
            continue
        perturbed_image = rotate_image(original_image, angle=15) # Rotate by 15 degrees

        # Helper for getting predictions
        def get_raw_prediction(img, cap_text):
            prompt = f"USER: <image>\nLook at this image and evaluate: {cap_text}\nIs this statement true or false? Answer with just 'True' or 'False'. ASSISTANT:"
            inputs = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
            with torch.no_grad():
                generate_ids = model.generate(**inputs, max_new_tokens=10)
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return response.split("ASSISTANT:")[-1].strip()

        # 2. Get raw predictions for both images against the SAME caption
        raw_original_pred = get_raw_prediction(original_image, caption)
        raw_perturbed_pred = get_raw_prediction(perturbed_image, caption)
        
        # 3. Normalize the responses
        norm_original_pred = normalize_response(raw_original_pred)
        norm_perturbed_pred = normalize_response(raw_perturbed_pred)

        # 4. Check for consistency (for rotation, predictions SHOULD be the same)
        if norm_original_pred is None or norm_perturbed_pred is None:
            consistency = 'INDETERMINATE'
        elif norm_original_pred == norm_perturbed_pred: # <-- Note the logic change!
            consistency = 'CONSISTENT'
        else:
            consistency = 'INCONSISTENT'
        
        results.append({
            'caption': caption,
            'raw_original_pred': raw_original_pred,
            'raw_perturbed_pred': raw_perturbed_pred,
            'consistency': consistency
        })
        
    return results