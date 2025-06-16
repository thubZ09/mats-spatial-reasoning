import torch
from tqdm import tqdm
from .perturbations import perturb_spatial_words, rotate_image # Consolidate imports
from .metrics import normalize_response, check_logical_consistency

def run_text_perturbation_audit(model, processor, dataset):
    """
    Runs a text perturbation audit on a given dataset using REAL images.

    Args:
        model: The loaded vision-language model.
        processor: The model's processor.
        dataset: A dataset object where each item has an 'image' and 'caption'.

    Returns:
        A list of dictionaries containing audit results.
    """
    results = []
    
    # This helper function now accepts the image as an argument
    def get_raw_prediction(image, caption_text):
        # The prompt for the model
        prompt = f"USER: <image>\nLook at this image and evaluate: {caption_text}\nIs this statement true or false? Answer with just 'True' or 'False'. ASSISTANT:"
        
        # Process the real image and text
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        
        # Generate the response
        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=10)
        
        # Decode and clean the response
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response.split("ASSISTANT:")[-1].strip()

    # Loop through the dataset, now using the real image for each item
    for item in tqdm(dataset, desc="Auditing with Text Perturbation"):
        # Extract the real image and original caption from the dataset item
        original_image = item['image']
        original_caption = item['caption']
        
        # Ensure the image is in RGB format, as LLaVA expects it
        if original_image.mode != "RGB":
            original_image = original_image.convert("RGB")
        
        # 1. Generate the perturbed caption
        perturbed_caption = perturb_spatial_words(original_caption)
        
        # 2. Get raw predictions using the SAME image but DIFFERENT captions
        raw_original_pred = get_raw_prediction(original_image, original_caption)
        raw_perturbed_pred = get_raw_prediction(original_image, perturbed_caption)
        
        # 3. Normalize the raw predictions
        norm_original_pred = normalize_response(raw_original_pred)
        norm_perturbed_pred = normalize_response(raw_perturbed_pred)

        # 4. Check for logical consistency (e.g., True vs. False)
        consistency = check_logical_consistency(norm_original_pred, norm_perturbed_pred)
        
        results.append({
            'original_caption': original_caption,
            'perturbed_caption': perturbed_caption,
            'raw_original_pred': raw_original_pred,
            'raw_perturbed_pred': raw_perturbed_pred,
            'consistency': consistency
        })
        
    return results

def run_visual_perturbation_audit(model, processor, dataset):
    """
    Runs a visual perturbation audit (e.g., rotation) on a dataset using REAL images.
    Compares model response on an image vs. a rotated version of it.

    Args:
        model: The loaded vision-language model.
        processor: The model's processor.
        dataset: A dataset object where each item has an 'image' and 'caption'.

    Returns:
        A list of dictionaries containing audit results.
    """
    results = []

    def get_raw_prediction(image, caption_text):
        prompt = f"USER: <image>\nLook at this image and evaluate: {caption_text}\nIs this statement true or false? Answer with just 'True' or 'False'. ASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=10)
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response.split("ASSISTANT:")[-1].strip()

    for item in tqdm(dataset, desc="Auditing with Visual Perturbation"):
        # Get the real image and caption
        original_image = item['image']
        caption = item['caption']

        if original_image.mode != "RGB":
            original_image = original_image.convert("RGB")
            
        # 1. Create the visually perturbed version (rotated image)
        perturbed_image = rotate_image(original_image, angle=15) # Rotate by 15 degrees

        # 2. Get raw predictions for both images against the SAME caption
        raw_original_pred = get_raw_prediction(original_image, caption)
        raw_perturbed_pred = get_raw_prediction(perturbed_image, caption)
        
        # 3. Normalize the responses
        norm_original_pred = normalize_response(raw_original_pred)
        norm_perturbed_pred = normalize_response(raw_perturbed_pred)

        # 4. Check for consistency (for rotation, predictions SHOULD be the same)
        if norm_original_pred is None or norm_perturbed_pred is None:
            consistency = 'INDETERMINATE'
        elif norm_original_pred == norm_perturbed_pred:
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
