from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .clip_patching import load_clip_model, run_patch_batch as clip_run_patch_batch
from .llava_patching import run_patch_batch_llava


def run_clip_experiment(
    model_name: str,
    processor_model: Optional[str],
    examples: Iterable[dict],
    modules_to_test: List[str],
    out_dir: str = "output/clip",
    blends: Iterable[float] = (1.0,),
    max_examples: Optional[int] = None,
    device: Optional[str] = None,
):
    """
    load CLIP (by model_name) and run patch batch. `examples` is the same format
    expected by clip_patching.run_patch_batch (donor_image, target_image, correct_text, absurd_text, example_id)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model, processor, device = load_clip_model(model_name, device=device)
    out_csv = clip_run_patch_batch(model, processor, device, examples, modules_to_test, out_dir=out_dir, blends=blends, max_examples=max_examples)
    print("Saved CLIP results to:", out_csv)
    return out_csv


def run_llava_experiment(
    model_wrapper,
    examples: Iterable[dict],
    modules_to_test: List[str],
    out_dir: str = "output/llava",
    blends: Iterable[float] = (1.0,),
    max_examples: Optional[int] = None,
):
    """
    run LLaVA patching using a provided model_wrapper (must implement the interface).
    `examples` is a list of dicts (donor_image, target_image, clean_prompt, absurd_prompt, example_id)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_csv = Path(out_dir) / "llava_patch_results.csv"
    df = run_patch_batch_llava(model_wrapper, list(examples), modules_to_test, out_csv_path=str(out_csv), blends=list(blends), max_examples=max_examples)
    print("Saved LLaVA results to:", out_csv)
    return str(out_csv)
