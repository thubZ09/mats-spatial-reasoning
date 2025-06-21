# In scripts/run_full_evaluation.py
import sys
import json
import os
import torch

# Add project root to path to allow importing from src
sys.path.append('.')

from src.utils import model_loader, data_loader
from src.mats import auditor

MODELS_TO_TEST = {
    "llava-1.5-7b": model_loader.load_llava_1_5_7b,
    "blip2-opt-2.7b": model_loader.load_blip2_opt_2_7b,
    "instructblip-vicuna-7b": model_loader.load_instructblip_vicuna_7b
    # We will skip CLIP for now as it requires different analysis logic.
}

DATASETS_TO_TEST = {
    "VSR": data_loader.load_vsr_dataset,
}

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    for model_name, loader_func in MODELS_TO_TEST.items():
        print(f"--- Loading model: {model_name} ---")
        model, processor, model_type = loader_func()

        for dataset_name, data_func in DATASETS_TO_TEST.items():
            print(f"--- Auditing {model_name} on dataset: {dataset_name} ---")
            dataset = data_func(num_samples=100) # Use 100 samples for the main run

            text_results = auditor.run_text_perturbation_audit(model, processor, dataset, model_type)
            # visual_results = auditor.run_visual_perturbation_audit(model, processor, dataset, model_type)
            # coordinated_results = auditor.run_coordinated_perturbation_audit(model, processor, dataset, model_type)

            output_path = f"results/{model_name}_{dataset_name}_results.json"
            with open(output_path, 'w') as f:
                json.dump({'text_audit': text_results}, f, indent=4) # Add other results here too
            print(f"✅ Results for {model_name} on {dataset_name} saved to {output_path}")
        
        # Unload model to free up memory
        del model
        del processor
        torch.cuda.empty_cache()