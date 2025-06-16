import torch
from tqdm import tqdm
from .perturbations import perturb_spatial_words
from .metrics import normalize_response, check_logical_consistency # Import both

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