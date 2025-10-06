from __future__ import annotations
import contextlib
import copy
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

try:
    from transformers import CLIPModel, CLIPProcessor
except Exception as exc:
    raise ImportError(
        "transformers must be installed. e.g. pip install transformers"
    ) from exc


def load_clip_model(model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
    """
    load a CLIP model + processor, returns (model, processor, device)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor, device


def encode_images(model: CLIPModel, processor: CLIPProcessor, device: str, images: List[Image.Image]):
    """
    encode a list of PIL Images and return normalized image embeddings (N x D) as torch
    """
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    #normalize
    out = F.normalize(out, dim=-1).cpu()
    return out


def encode_texts(model: CLIPModel, processor: CLIPProcessor, device: str, texts: List[str]):
    """
    encode a list of texts and return normalized text embeddings (N x D) as torch
    """
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    out = F.normalize(out, dim=-1).cpu()
    return out


#module resolution & hooks
def resolve_module_by_name(model: torch.nn.Module, dotted: str) -> torch.nn.Module:
    """
    given a model and a dotted module path ("vision_model.encoder.layers.8.mlp"),
    return the nn.Module object, raises KeyError if not found.
    """
    parts = dotted.split(".")
    mod = model
    for p in parts:
        if hasattr(mod, p):
            mod = getattr(mod, p)
        elif p.isdigit() and hasattr(mod, "__getitem__"):
            mod = mod[int(p)]
        else:
            #try named_modules fallback
            named = dict(model.named_modules())
            if dotted in named:
                return named[dotted]
            raise KeyError(f"could not resolve module path '{dotted}' on model.")
    return mod


def record_module_output(model: torch.nn.Module, module: torch.nn.Module, model_inputs: Dict[str, Any], device: str):
    """
    run one forward pass with the given model_inputs and capture the module's output the 'activation'
    returns a CPU tensor of the module's output
    """
    activation = None

    def _hook(module, inp, out):
        nonlocal activation
        activation = out.detach().cpu().clone()

    h = module.register_forward_hook(_hook)
    with torch.no_grad():
        #model_inputs already placed on device by caller
        _ = model(**model_inputs)
    h.remove()
    if activation is None:
        raise RuntimeError("activation hook did not capture an output. module forward may not have been called.")
    return activation


@contextlib.contextmanager
def patch_module_output(module: torch.nn.Module, donor_activation: torch.Tensor, blend: float = 1.0):
    """
    context manager that temporarily wraps module.forward so that its output is
    blended with donor_activation

      - donor_activation is expected to be a tensor compatible with the module's output shape
      - blend = 1.0 -> fully donor; blend = 0.0 -> original behavior
      - we implement wrapper by calling the original forward and mixing:
            out' = (1-blend)*orig_out + blend*donor_activation
      - this approach is pragmatic, it works for many common modules but may fail
        for modules that return tuples, views, or rely on in-place ops. Use with caution
    """
    if not isinstance(donor_activation, torch.Tensor):
        raise ValueError("donor_activation must be a torch.Tensor (cpu or device)")

    orig_forward = module.forward

    def _patched_forward(*args, **kwargs):
        #ccall original to compute original output
        orig_out = orig_forward(*args, **kwargs)
        try:
            donor_deviceed = donor_activation.to(orig_out.device)
        except Exception:
            donor_deviceed = donor_activation.clone().to(orig_out.device)
        #ry to broadcast shapes if necessary
        if donor_deviceed.shape != orig_out.shape:
            #try to expand on batch dim if donor has 1 batch
            if donor_deviceed.dim() == orig_out.dim() and donor_deviceed.shape[0] == 1 and orig_out.shape[0] > 1:
                donor_deviceed = donor_deviceed.expand(orig_out.shape[0], *donor_deviceed.shape[1:])
            else:
                return orig_out
        if blend >= 1.0:
            return donor_deviceed
        elif blend <= 0.0:
            return orig_out
        else:
            return (1.0 - blend) * orig_out + blend * donor_deviceed

    module.forward = _patched_forward
    try:
        yield
    finally:
        module.forward = orig_forward #restore original forward


#metric helpers
def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    cosine similarity between two 1-D PyTorch tensors
    """
    a = a.float()
    b = b.float()
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return float((a * b).sum().item())

def l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.norm(a - b).item())

#sngle-patch experiment
def run_single_patch_trial(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    donor_image: Image.Image,
    target_image: Image.Image,
    correct_text: str,
    absurd_text: str,
    module_name: str,
    blend: float = 1.0
) -> Dict[str, Any]:
    """
    run a single donor->target patch trial on `module_name`

    returns a dict with keys:
      - module_name
      - dcos (float): cos(after, text_correct) - cos(before, text_correct)
      - dL2_to_donor (float): L2(after, donor_emb) - L2(before, donor_emb)
      - prefer_before (int): 1 if before preferred correct_text, else 0
      - prefer_after (int): 1 if after preferred correct_text, else 0
      - additional debug fields..
    """
    device_t = device
    model.to(device_t)

    #prepare inputs (we keep single-example batch semantics)
    donor_inputs = processor(images=[donor_image], text=[correct_text], return_tensors="pt", padding=True)
    donor_inputs = {k: v.to(device_t) for k, v in donor_inputs.items()}
    target_inputs = processor(images=[target_image], text=[absurd_text], return_tensors="pt", padding=True)
    target_inputs = {k: v.to(device_t) for k, v in target_inputs.items()}

    #compute donor activation at module
    module = resolve_module_by_name(model, module_name)
    donor_act = record_module_output(model, module, donor_inputs, device_t)

    #compute 'before' embeddings for target image (without patch)
    with torch.no_grad():
        targ_img_emb_before = model.get_image_features(pixel_values=target_inputs['pixel_values'].to(device_t)).detach().cpu()
        candidate_texts = [correct_text, absurd_text]
        text_embs = encode_texts(model, processor, device_t, candidate_texts)  # N x D, already cpu
    #normalize
    targ_img_before_n = F.normalize(targ_img_emb_before, dim=-1).squeeze(0).cpu()
    text_correct_emb = text_embs[0].cpu()
    sims_before = (targ_img_before_n @ text_embs.T).squeeze(0).numpy()
    prefer_before = int(sims_before[0] >= sims_before[1])

    #apply patch (monkeypatch module.forward) while running corrupted input
    with patch_module_output(module, donor_act, blend=blend):
        with torch.no_grad():
            targ_img_emb_after = model.get_image_features(pixel_values=target_inputs['pixel_values'].to(device_t)).detach().cpu()

    targ_img_after_n = F.normalize(targ_img_emb_after, dim=-1).squeeze(0).cpu()

    #compute metrics
    cos_before = float((targ_img_before_n @ text_correct_emb).item())
    cos_after = float((targ_img_after_n @ text_correct_emb).item())
    dcos = cos_after - cos_before

    #L2 to donor embedding (donor image embedding used as proxy for donor semantics)
    donor_img_emb = encode_images(model, processor, device_t, [donor_image]).squeeze(0)
    l2_before = l2_distance(targ_img_before_n, donor_img_emb)
    l2_after = l2_distance(targ_img_after_n, donor_img_emb)
    dL2 = l2_after - l2_before

    #prefer_after
    sims_after = (targ_img_after_n @ text_embs.T).numpy()
    prefer_after = int(sims_after[0] >= sims_after[1])

    out = {
        "module": module_name,
        "dcos": float(dcos),
        "cos_before": float(cos_before),
        "cos_after": float(cos_after),
        "dL2_to_donor": float(dL2),
        "l2_before": float(l2_before),
        "l2_after": float(l2_after),
        "prefer_before": int(prefer_before),
        "prefer_after": int(prefer_after),
        "blend": float(blend)
    }
    return out

#batch runner
def run_patch_batch(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    examples: Iterable[Dict[str, Any]],
    modules_to_test: Iterable[str],
    out_dir: str = "output",
    blends: Iterable[float] = (1.0,),
    max_examples: Optional[int] = None
) -> str:
    """
    run patch experiments on a batch of examples

    `examples` is an iterable of dicts with keys:
      - donor_image: PIL.Image (clean image)
      - target_image: PIL.Image (corrupted/absurd)
      - correct_text: str
      - absurd_text: str
      - example_id: optional id

    `modules_to_test` is a list of dotted module paths to test.

    returns the saved CSV filepath
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n = 0
    for ex in tqdm(examples, desc="examples"):
        if max_examples is not None and n >= max_examples:
            break
        n += 1
        donor_image = ex["donor_image"]
        target_image = ex["target_image"]
        correct_text = ex["correct_text"]
        absurd_text = ex["absurd_text"]
        example_id = ex.get("example_id", f"ex{n}")

        for module_name in modules_to_test:
            for blend in blends:
                try:
                    r = run_single_patch_trial(
                        model, processor, device,
                        donor_image=donor_image,
                        target_image=target_image,
                        correct_text=correct_text,
                        absurd_text=absurd_text,
                        module_name=module_name,
                        blend=blend
                    )
                    r.update({"example_id": example_id})
                    rows.append(r)
                except Exception as e:
                    rows.append({
                        "example_id": example_id,
                        "module": module_name,
                        "error": str(e),
                    })

    df = pd.DataFrame(rows)
    out_file = out_dir / "clip_proj_patch_results.csv"
    df.to_csv(out_file, index=False)
    return str(out_file)


#small utility, loading a PIL image from path
def load_image(path_or_pil):
    if isinstance(path_or_pil, Image.Image):
        return path_or_pil
    return Image.open(path_or_pil).convert("RGB")
