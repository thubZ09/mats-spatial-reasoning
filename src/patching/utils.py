from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image
import pandas as pd


def load_image(path_or_pil):
    """
    accept either a PIL Image or a path-like object, return a PIL.Image (RGB)
    """
    if isinstance(path_or_pil, Image.Image):
        return path_or_pil.convert("RGB")
    p = Path(path_or_pil)
    return Image.open(str(p)).convert("RGB")


def load_examples_from_jsonl(path: str, image_key_donor="donor_image", image_key_target="target_image") -> List[Dict[str, Any]]:
    """
    load examples from a JSONL file, each line should be a JSON object with keys:
      - donor_image (path)
      - target_image (path)
      - clean_prompt (str)
      - absurd_prompt (str)
      - example_id (optional)
    returns a list of dicts where donor_image/target_image are PIL Images (loaded)
    """
    examples = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            donor_p = obj.get(image_key_donor)
            target_p = obj.get(image_key_target)
            example = {
                "donor_image": load_image(donor_p) if donor_p else None,
                "target_image": load_image(target_p) if target_p else None,
                "clean_prompt": obj.get("clean_prompt"),
                "absurd_prompt": obj.get("absurd_prompt"),
                "example_id": obj.get("example_id"),
            }
            examples.append(example)
    return examples


def save_results_df(df: pd.DataFrame, out_path: str):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(p), index=False)
    return str(p)


def write_jsonl_examples(example_list: Iterable[Dict[str, Any]], out_path: str):
    """
    write an iterable of example dicts to JSONL. NOTE: images must be paths (strings) here
    """
    with open(out_path, "w", encoding="utf8") as f:
        for ex in example_list:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def simple_logger(msg: str):
    print(msg)
