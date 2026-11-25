"""Thin adapter around the base VLM: loading, input building, loss masking, generation.

Kept separate from the data core so that the CPU package imports without torch.
Only `dataset.to_model_example` output (image path + prompt, optionally target) enters here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from PIL import Image

BASE_ID = "Qwen/Qwen3.5-2B"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(model_id: str = BASE_ID, revision: str = BASE_REVISION, device: Optional[str] = None,
         dtype: Optional[torch.dtype] = None, max_pixels: Optional[int] = None):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = device or pick_device()
    if dtype is None:
        dtype = torch.float32 if device == "cpu" else torch.float16
    kwargs = {} if max_pixels is None else {"max_pixels": max_pixels}
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, **kwargs)
    model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, processor, device


def _messages(image: Image.Image, prompt: str, target: Optional[str] = None,
              system: Optional[str] = None) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    if system is not None:  # frozen-base prompt variant only (base_prompt.py); never used in training
        msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
    msgs.append(
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    )
    if target is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return msgs


def _template(processor, messages, add_generation_prompt: bool, template_kwargs: Dict[str, Any]):
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt, return_dict=True,
        return_tensors="pt", **template_kwargs,
    )


BLANK_SIZE = 512  # same pixel budget as the rendered graphs


def build_inference_inputs(processor, image_path: Optional[str], prompt: str,
                           template_kwargs: Optional[Dict[str, Any]] = None, system: Optional[str] = None):
    """`image_path=None` means a blank white image (oracle text-only diagnostic)."""
    if image_path is None:
        image = Image.new("RGB", (BLANK_SIZE, BLANK_SIZE), "white")
    else:
        image = Image.open(image_path).convert("RGB")
    return _template(processor, _messages(image, prompt, system=system), True, template_kwargs or {})


def build_training_inputs(processor, image_path: str, prompt: str, target: str,
                          template_kwargs: Optional[Dict[str, Any]] = None):
    """Full conversation with labels that are -100 everywhere except the assistant's answer.

    The prompt-only tokenisation must be a prefix of the full one; that is asserted, so a
    template that inserts something unexpected cannot silently shift the loss mask.
    """
    image = Image.open(image_path).convert("RGB")
    kw = template_kwargs or {}
    full = _template(processor, _messages(image, prompt, target), False, kw)
    head = _template(processor, _messages(image, prompt), True, kw)
    n = head["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][0, :n], head["input_ids"][0]):
        raise ValueError("prompt tokens are not a prefix of the full conversation; loss mask would be wrong")
    labels = full["input_ids"].clone()
    labels[:, :n] = -100
    full["labels"] = labels
    return full, n


@torch.no_grad()
def generate(model, processor, inputs, max_new_tokens: int = 64) -> Dict[str, Any]:
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0, inputs["input_ids"].shape[1]:]
    return {
        "text": processor.decode(new, skip_special_tokens=True),
        "raw": processor.decode(new, skip_special_tokens=False),
        "new_tokens": int(new.shape[0]),
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "hit_limit": int(new.shape[0]) >= max_new_tokens,
        "ids": [int(x) for x in new.tolist()],
    }
