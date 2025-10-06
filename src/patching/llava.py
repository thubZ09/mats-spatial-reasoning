from __future__ import annotations
import contextlib
import copy
import math
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

#model-wrapper interface (adapter)
class ModelWrapperInterface:
    """
    required methods (semantic contract)
    - encode_image(image: PIL.Image) -> torch.Tensor
        return a CPU tensor (1 x D) or device tensor (1 x D) that represents pooled image features

    - generate_answer(image: PIL.Image, prompt_text: str, max_new_tokens: int = 32, **gen_kwargs) -> str
        run the model in inference mode with the provided prompt and return the generated string
    - get_module_by_name(dotted_name: str) -> torch.nn.Module

        resolve and return a module/submodule by a dotted path e.g., "vision_model.encoder.layers.12.cross_attn"
    - run_forward_for_activation(image: PIL.Image, prompt_text: str, module_name: str) -> torch.Tensor
        run a forward pass that triggers the module and returns its activation

    LLaVAPatcher uses these methods to record donor activations and to run patched forwards
    """

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        raise NotImplementedError

    def generate_answer(self, image: Image.Image, prompt_text: str, max_new_tokens: int = 32, **gen_kwargs) -> str:
        raise NotImplementedError

    def get_module_by_name(self, dotted_name: str) -> torch.nn.Module:
        raise NotImplementedError

    def run_forward_for_activation(self, image: Image.Image, prompt_text: str, module_name: str) -> torch.Tensor:
        raise NotImplementedError

#try for your model
class ModelWrapperExample(ModelWrapperInterface):
    """
    example adapter showing the minimal wrapper shape
    replace internals with your LLaVA wrapper's real calls (tokenizer, image processor, model.generate, etc)
    """

    def __init__(self, model=None, tokenizer=None, vision_processor=None, device: str = "cuda"):
        #if you already have a LLaVA wrapper class, adapt it here and forward methods
        self.model = model
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.device = device

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        return pooled image representation (1 x D) as CPU tensor
        replace with model-specific image->embedding pipeline
        """
        raise NotImplementedError("please implement encode_image for your model wrapper")

    def generate_answer(self, image: Image.Image, prompt_text: str, max_new_tokens: int = 32, **gen_kwargs) -> str:
        """
        run generation and return the model's generated string
        use your deterministic single-token forcing or greedy generation to get stable TRUE/FALSE outputs
        """
        raise NotImplementedError("please implement generate_answer for your model wrapper")

    def get_module_by_name(self, dotted_name: str) -> torch.nn.Module:
        raise NotImplementedError("please implement get_module_by_name for your model wrapper")

    def run_forward_for_activation(self, image: Image.Image, prompt_text: str, module_name: str) -> torch.Tensor:
        raise NotImplementedError("please implement run_forward_for_activation for your model wrapper")

#safe module resolver & activation recorder (fallback)
def resolve_module_by_name(root_module: torch.nn.Module, dotted: str) -> torch.nn.Module:
    """
    resolve dotted module name against a given root torch.nn.Module instance
    attempts attribute access and numeric indexing for container modules
    """
    parts = dotted.split(".")
    mod = root_module
    for p in parts:
        if hasattr(mod, p):
            mod = getattr(mod, p)
        elif p.isdigit() and hasattr(mod, "__getitem__"):
            mod = mod[int(p)]
        else:
            named = dict(root_module.named_modules())
            if dotted in named:
                return named[dotted]
            raise KeyError(f"could not resolve '{dotted}' under root module '{root_module.__class__.__name__}'")
    return mod


def record_activation_via_hook(module: torch.nn.Module, forward_inputs: Tuple[Any, ...], forward_kwargs: Dict[str, Any],
                               run_fn):
    """
    attach a forward hook to capture the module's output for a single run
    - run_fn is a callable that actually runs the model forward given the inputs/kwargs
    returns the activation as a cpu tensor
    """
    activation = None

    def _hook(m, inp, out):
        nonlocal activation
        activation = out.detach().cpu().clone()

    h = module.register_forward_hook(_hook)
    try:
        run_fn(*forward_inputs, **forward_kwargs)
    finally:
        h.remove()
    if activation is None:
        raise RuntimeError("activation hook did not observe an output. module may not have been invoked")
    return activation


@contextlib.contextmanager
def patch_module_output(module: torch.nn.Module, donor_activation: torch.Tensor, blend: float = 1.0):
    """
    context manager that temporarily replaces module.forward so its output is blended
    with donor_activation
    behavior:
      - if donor_activation shape differs but is broadcastable on batch dim, attempt to expand
      - if shapes incompatible, the wrapper will return the original output (no patch)
      - blend=1.0 -> full donor activation; 0.0 -> original module behavior
    """
    if not isinstance(donor_activation, torch.Tensor):
        raise ValueError("donor_activation must be a torch.Tensor")

    orig_forward = module.forward

    def _patched_forward(*args, **kwargs):
        orig_out = orig_forward(*args, **kwargs)
        try:
            donor_dev = donor_activation.to(orig_out.device)
        except Exception:
            donor_dev = donor_activation.clone().to(orig_out.device)
        if donor_dev.shape != orig_out.shape:
            if donor_dev.dim() == orig_out.dim() and donor_dev.shape[0] == 1 and orig_out.shape[0] > 1:
                donor_dev = donor_dev.expand(orig_out.shape[0], *donor_dev.shape[1:])
            else:
                return orig_out
        if blend >= 1.0:
            return donor_dev
        elif blend <= 0.0:
            return orig_out
        else:
            return (1.0 - blend) * orig_out + blend * donor_dev

    module.forward = _patched_forward
    try:
        yield
    finally:
        module.forward = orig_forward

#parsing helper
def parse_binary_response(generated_text: str) -> str:
    """
    parse model output string and return 'TRUE','FALSE', or 'UNKNOWN'
    the generation should be deterministic and constrained
    """
    if not isinstance(generated_text, str):
        return "UNKNOWN"
    t = generated_text.strip().upper()
    #keep only first token-like element
    first = t.split()[0] if t.split() else ""
    if first in ("TRUE", "T", "YES", "Y"):
        return "TRUE"
    if first in ("FALSE", "F", "NO", "N"):
        return "FALSE"
    #fallback
    if "TRUE" in t:
        return "TRUE"
    if "FALSE" in t:
        return "FALSE"
    return "UNKNOWN"

#llava patcher
@dataclass
class LLaVAPatcher:
    """
    driver for LLaVA patching experiments

    `model_wrapper` must implement the ModelWrapperInterface above

    methods:
      - run_single_patch_trial(donor_image, target_image, clean_prompt, absurd_prompt, module_name, blend)
         -> returns dict with results including whether the final binary answer flipped
    """
    model_wrapper: ModelWrapperInterface

    def run_single_patch_trial(
        self,
        donor_image: Image.Image,
        target_image: Image.Image,
        clean_prompt: str,
        absurd_prompt: str,
        module_name: str,
        blend: float = 1.0,
        max_new_tokens: int = 32,
    ) -> Dict[str, Any]:
        """
        single donor -> target patch trial for LLaVA-style generative models

        procedure:
         → run donor image + clean_prompt and record donor activation at module_name
         → run target image + absurd_prompt to get 'before' generated answer (parse binary)
         → temporarily patch the module with donor activation and re-run target -> get 'after' generated answer
         → return a result dict with before/after parsed labels, any UNKNOWNs, and debug info

        NOTE: generation should be deterministic (greedy or single-token forced) to make comparisons stable
        """
        result = {
            "module": module_name,
            "blend": float(blend),
            "before": None,
            "after": None,
            "before_raw": None,
            "after_raw": None,
            "patch_error": None,
        }
        try:
            #record donor activation using wrapper-provided runner
            donor_act = self.model_wrapper.run_forward_for_activation(
                donor_image, clean_prompt, module_name
            ) 
            #run 'before' generation on corrupted prompt
            before_raw = self.model_wrapper.generate_answer(target_image, absurd_prompt, max_new_tokens=max_new_tokens)
            before_label = parse_binary_response(before_raw)
            #apply patch and run 'after'
            module = self.model_wrapper.get_module_by_name(module_name)
            with patch_module_output(module, donor_act, blend=blend):
                after_raw = self.model_wrapper.generate_answer(target_image, absurd_prompt, max_new_tokens=max_new_tokens)
            after_label = parse_binary_response(after_raw)

            result.update({
                "before": before_label,
                "after": after_label,
                "before_raw": before_raw,
                "after_raw": after_raw,
            })

            #success flag: patch corrected an erroneous TRUE to FALSE (or optionally the reverse)
            #interpret success as flipping from incorrect->correct as defined by the user:
            result["success"] = int(before_label == "TRUE" and after_label == "FALSE")
        except Exception as e:
            tb = traceback.format_exc()
            result["patch_error"] = f"{type(e).__name__}: {str(e)}\n{tb}"
            result["success"] = 0
        return result


#convenience batch-run helper
def run_patch_batch_llava(
    model_wrapper: ModelWrapperInterface,
    examples: list,
    modules_to_test: list,
    out_csv_path: Optional[str] = None,
    blends: list = (1.0,),
    max_examples: Optional[int] = None,
) -> pd.DataFrame:
    """
    run LLaVA patching over a list of examples

    `examples` should be an iterable of dicts with keys:
       - donor_image: PIL.Image (clean)
       - target_image: PIL.Image (corrupted/absurd)
       - clean_prompt: str (textually correct statement)
       - absurd_prompt: str (textually false statement)
       - example_id: optional

    returns a pandas dataframe of results
    """
    patcher = LLaVAPatcher(model_wrapper)
    rows = []
    n = 0
    for ex in tqdm(examples, desc="llava examples"):
        if max_examples is not None and n >= max_examples:
            break
        n += 1
        donor_image = ex["donor_image"]
        target_image = ex["target_image"]
        clean_prompt = ex["clean_prompt"]
        absurd_prompt = ex["absurd_prompt"]
        example_id = ex.get("example_id", f"ex{n}")

        for module_name in modules_to_test:
            for blend in blends:
                res = patcher.run_single_patch_trial(donor_image, target_image, clean_prompt, absurd_prompt, module_name, blend=blend)
                res.update({"example_id": example_id})
                rows.append(res)

    df = pd.DataFrame(rows)
    if out_csv_path:
        df.to_csv(out_csv_path, index=False)
    return df
