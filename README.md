<div align='center'>

# Multimodal Audit for Truthful Spatialization (MATS)
<strong>Pathological Truth Bias in Vision-Language Models</strong>


[![arXiv](https://img.shields.io/badge/arXiv-2509.22674-b31b1b.svg)](https://arxiv.org/abs/2509.22674)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)


</div>

## Abstract

Vision-language models (VLMs) exhibit a concerning failure mode termed *pathological truth bias*: the systematic tendency to affirm visually contradicted statements rather than rejecting them. Using MATS (Multimodal Audit for Truthful Spatialization), we demonstrate that instruction-tuned generative VLMs (LLaVA-1.5, Qwen-VL-chat) show very low Spatial Consistency Scores (SCS ≈ 1–3%) and high Incorrect Agreement Rates (IAR ≈ 75–80%), while contrastive encoders (CLIP, SigLIP) remain substantially more robust (SCS ≈ 57–68%, IAR ≈ 8–12%). Through systematic activation patching across 420 trials, we causally localize these failures to mid-to-late cross-attention layers in generative models and pooled/projection components in contrastive encoders, achieving 23% patch success in restoring correct behavior. Results implicate current instruction-tuning practices that prioritize agreeableness over truthfulness, and identify specific neural loci as targets for intervention-based repairs.

<p align="center">
  <img src=".github/paper.png" alt="MATS Overview" width="800"/>
</p>


## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration)
- 16GB+ GPU memory recommended for generative models

### Setup
```bash
#clone repository
git clone https://github.com/thubZ09/mats-spatial-reasoning.git
cd mats-spatial-reasoning

#create virtual environment
python -m venv venv
source venv/bin/activate  #windows: venv\Scripts\activate

#install dependencies
pip install -r requirements.txt

#verify installation
python -c "from src.mats.metrics import MATSMetrics; print('Installation successful')"
```
## Quick Start  
**1. Behavioral Audit - Compute Spatial Consistency Score (SCS) and Incorrect Agreement Rate (IAR)**
```bash 
from src.mats.metrics import MATSMetrics
from src.mats.auditor import run_comparative_text_audit

#initialize metrics
metrics = MATSMetrics()

#prepare test data
scs_data = [
    {
        'relation': 'left',
        'original_raw': 'TRUE',
        'inverted_raw': 'TRUE'  # Pathological: should flip to FALSE
    },
    # ... more examples
]

iar_data = [
    {
        'category': 'spatial',
        'raw_response': 'TRUE'  # Incorrect: absurd statement
    },
    # ... more examples
]

#evaluate model
result = metrics.evaluate_model('LLaVA-1.5', scs_data, iar_data)
metrics.print_summary(result)
```
**2. Statistical analysis
compute confidence intervals and significance tests**
```bash
from src.mats.statistical_tests import MATSStatisticalTests
import numpy as np

stats = MATSStatisticalTests(alpha=0.05, n_bootstrap=10000)

#bootstrap 95% CI for SCS
scs_flips = np.array([0, 0, 1, 0, 0, ...])  #binary flip indicators
_, ci_lower, ci_upper = stats.bootstrap_confidence_interval(scs_flips)
print(f"SCS 95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")

#compare generative vs contrastive families
comparison = stats.compare_iar_by_family(
    generative_iar=(78, 100),
    contrastive_iar=(12, 100)
)
print(f"p-value: {comparison['proportion_test']['p_value']:.6f}")
print(f"Cohen's h: {comparison['cohens_h']:.3f}") 
```
**3. Activation Patching (Mechanistic Analysis)**  
*CLIP Patching*
```bash 
from src.patching.runner import run_clip_experiment

#define modules to test (pooled/projection components)
modules = [
    "vision_model.encoder.layers.11.mlp",
    "vision_model.post_layernorm",
    "vision_projection"
]

#run experiment
results_path = run_clip_experiment(
    model_name="openai/clip-vit-base-patch32",
    examples=patch_examples,  #donor/target image-text pairs
    modules_to_test=modules,
    out_dir="results/clip_patches"
)

#analyze results
import pandas as pd
df = pd.read_csv(results_path)
print(f"Mean Δcos: {df['dcos'].mean():.4f}")
print(f"Successful repairs: {(df['prefer_after'] == 1).sum()}")
```
*LLaVA Patching*
```bash
from src.patching.runner import run_llava_experiment

#define attention/MLP layers to test
modules = [
    "model.layers.12.self_attn",
    "model.layers.15.self_attn",
    "model.layers.18.cross_attn",
    "model.layers.12.mlp"
]

#run patching with your LLaVA wrapper
results_path = run_llava_experiment(
    model_wrapper=your_llava_wrapper,
    examples=patch_examples,
    modules_to_test=modules,
    out_dir="results/llava_patches"
)

#compute patch success rate
df = pd.read_csv(results_path)
success_rate = df['success'].mean()
print(f"Patch success rate: {success_rate:.1%}") 
```

## Metrics reference  
**Spatial Consistency Score (SCS)**  
Measures whether a model flips its judgment under predicate inversion 
```bash 
SCS = (number of flips) / (total valid pairs)
```

High SCS → Proper spatial reasoning      
Low SCS → Pathological truth bias

**Incorrect Agreement Rate (IAR)**  
Fraction of absurd statements incorrectly marked TRUE  
```bash 
IAR = (incorrect agreements) / (total absurd statements)
```
Low IAR → Proper rejection behavior  
High IAR → Pathological truth bias

## Results

| Model Family | SCS (%) | IAR (%) | Architecture |
|--------------|---------|---------|--------------|
| **Generative VLMs** |
| LLaVA-1.5    | 1.2     | 78.5    | Instruction-tuned |
| Qwen-VL-chat | 2.8     | 76.2    | Instruction-tuned |
| **Contrastive Encoders** |
| CLIP ViT-B/32 | 57.1   | 12.0    | Contrastive pre-training |
| SigLIP-Base  | 68.2    | 8.0     | Contrastive pre-training |

**Mechanistic Findings:** 
- Activation patching successfully restores correct outputs in 23.3% of cases
- Causal effects concentrate in layers 12-18 (mid-to-late cross-attention)
- CLIP pooled/projection patches yield mean Δcos ≈ 0.05–0.07
- All effects statistically significant (p < 0.001)

## Citation
If you find this work useful, please cite:
```bash
@article{thube2025mats,
  title={Pathological Truth Bias in Vision-Language Models},
  author={Thube, Yash},
  journal={arXiv preprint arXiv:2509.22674},
  year={2025}
```
## License
The project is licensed under the MIT License - see [LICENSE](LICENSE.md) file for details.

## Contributing
Contributions are welcomed! Please see [contibutions](CONTRIBUTING.md) for guidelines. 

Areas for contribution:
- Additional model evaluations (Gemini, GPT-4V, etc.)  
- Extended perturbation types (temporal, causal)  
- Alternative patching strategies  
- Mitigation techniques
