import torch
from tqdm import tqdm
from .perturbations import perturb_spatial_words
from .metrics import check_logical_consistency

def run_text_perturbation_audit(model, processor, dataset):
    """
    Runs the textual perturbation audit on a given dataset.

    Args:
        model: The loaded vision-language model.
        processor: The model's processor.
        dataset: A dictionary containing the data to audit (e.g., from data_loader).

    Returns:
        A list of dictionaries, where each dictionary contains an audit result.
    """
    results = []
    
    # Using tqdm for a nice progress bar
    for item in tqdm(dataset['train'], desc="Auditing dataset"):
        original_caption = item['caption']
        
        # We will deal with real images later. For now, we pass None.
        image = None # Or item['image'] if it's a loaded PIL image
        
        # --- The core audit logic from your notebook ---
        
        # 1. Create the perturbed version
        perturbed_caption = perturb_spatial_words(original_caption)
        
        # 2. Get predictions (using a simplified helper function)
        # Note: A real implementation would handle images properly.
        # For now, this assumes a text-only or dummy-image scenario.
        def get_pred_internal(caption_text):
            prompt = f"USER: <image>\n{caption_text} ASSISTANT:"
            # A real image would be passed here instead of being ignored
            inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
            generate_ids = model.generate(**inputs, max_new_tokens=20)
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return response.split("ASSISTANT:")[-1].strip()

        original_pred = get_pred_internal(original_caption)
        perturbed_pred = get_pred_internal(perturbed_caption)
        
        # 3. Check consistency using the new metrics function
        is_consistent = check_logical_consistency(original_pred, perturbed_pred, original_caption)
        
        results.append({
            'original_caption': original_caption,
            'perturbed_caption': perturbed_caption,
            'original_prediction': original_pred,
            'perturbed_prediction': perturbed_pred,
            'consistency': is_consistent
        })
        
    return results