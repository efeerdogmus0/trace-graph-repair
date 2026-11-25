"""Checkpoint / resume for LoRA training runs.

A checkpoint is written only at an optimizer-update boundary (no partial gradients), into
`<run_dir>/ckpt` via a temporary directory and an atomic rename, so an interrupted write never
leaves a half checkpoint behind. It holds the trainable parameters, optimizer and scheduler
state, the number of samples consumed and the training log. The sample order is a pure
function of (seed, arm, epoch index), so resuming replays the order and skips `seen` samples.
"""
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

CKPT = "ckpt"


def trainable_state(model) -> Dict[str, Any]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def save(run_dir: Path, model, opt, sched, seen: int, log: List[Dict[str, Any]], extra: Dict[str, Any]) -> Path:
    import torch
    run_dir = Path(run_dir)
    tmp, final = run_dir / (CKPT + ".tmp"), run_dir / CKPT
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    torch.save({"params": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict()}, tmp / "state.pt")
    (tmp / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    (tmp / "meta.json").write_text(json.dumps({"seen": seen, **extra}, indent=2))
    old = run_dir / (CKPT + ".old")
    if final.exists():
        if old.exists():
            shutil.rmtree(old)
        os.rename(final, old)
    os.rename(tmp, final)
    if old.exists():
        shutil.rmtree(old)
    return final


def load(run_dir: Path, model, opt, sched, expect: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Restore state in place. Returns {'seen', 'log', **meta} or None if no checkpoint.
    Raises if the checkpoint belongs to a different configuration (keys in `expect`)."""
    import torch
    final = Path(run_dir) / CKPT
    if not (final / "meta.json").exists():
        return None
    meta = json.loads((final / "meta.json").read_text())
    for k, v in expect.items():
        if meta.get(k) != v:
            raise ValueError("checkpoint mismatch on %s: %r != %r" % (k, meta.get(k), v))
    state = torch.load(final / "state.pt", map_location="cpu")
    named = dict(model.named_parameters())
    if set(state["params"]) != {n for n, p in named.items() if p.requires_grad}:
        raise ValueError("checkpoint parameter names differ from the model's trainable parameters")
    with torch.no_grad():
        for n, t in state["params"].items():
            named[n].copy_(t.to(named[n].device, named[n].dtype))
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    log = [json.loads(l) for l in (final / "log.jsonl").read_text().splitlines() if l]
    return {**meta, "log": log}
