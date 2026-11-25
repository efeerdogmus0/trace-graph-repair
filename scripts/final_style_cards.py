"""Locked-test style-pair cards: one question on a root image and on its restyled copy, with every final run's
recorded answers (CPU only, no model, no new estimate).

    PYTHONPATH=src python scripts/final_style_cards.py [--table reports/final/final_table.json]
        [--queue scripts/kaggle/final_queue.json] [--out reports/final/style_cards.json]
        [--out-md reports/final/style_cards.md] [--copy-images reports/final/style_cards]

Brief section 22 asks the demo to show a style change. On dev no response to a restyled copy was recorded
(scripts/dev_gallery.py says so), so these cards can only come from the locked test. Written before any
locked-test output was read (DECISIONS 2026-11-19T23:40Z).

Gate. The script first loads the accepted final table through scripts/final_figures.py's load_table: while
reports/final/final_table.json does not exist, or does not cover exactly the twelve queue runs, it refuses
before opening any prediction or flags file. The frozen base (arm A, jobF13) is not part of that table and is
not shown.

Selection rule (RULE, fixed here before any per-sample outcome was looked at):
  splits  test_id (style seen in training, copy in another seen style) and test_style_ood (held-out styles).
  pool    restyled copies (pair kind invariant_style, role variant) of the split's invariance set whose base
          (the root-image sample with the same question) is in the split's main evaluation set.
  order   ascending sha256("style-cards-v1:" + variant sample_id); no response and no score enters the order.
  slots   per split: the first reachability pair and the first critical-edge pair in that order.
  A slot with no candidate is reported as empty and never filled by another rule.

Every response shown is a line of a final kernel's predictions file
(runs/kaggle/<job>/runs/<run>/predictions_<split>[_inv].jsonl) whose sha256 must equal the one recorded in
the collected flags file (<dev_dir>/flags_<split>[_inv].json); otherwise the script stops. Verdicts are
metrics.score_sample against the generator's graph (via demo/walkthrough.py's verdict). The cards are an
illustration: the invariance rates are in scripts/final_invariance.py's table and no claim rests on a card.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from trace_vr import arms as arms_mod, dataset, tasks

ROOT = Path(__file__).resolve().parents[1]
RULE = "style-cards-v1"
CARD_SPLITS = ("test_id", "test_style_ood")
SLOT_TASKS = (tasks.REACHABILITY, tasks.CRITICAL_EDGE)
KIND = "invariant_style"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


final_figures = _load("final_figures", "scripts/final_figures.py")
dev_gallery = _load("dev_gallery", "scripts/dev_gallery.py")
SPLITS = final_figures.SPLITS


def order_key(sample_id: str) -> str:
    return hashlib.sha256((RULE + ":" + sample_id).encode()).hexdigest()


def _pair(sample: Dict[str, Any], role: str) -> Optional[str]:
    for p in sample["pairs"]:
        if p["kind"] == KIND and p["role"] == role:
            return p["pair_id"]
    return None


def style_pairs(main_rows: List[Dict[str, Any]], inv_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """[(base, variant)] for every restyled copy whose base is in the main rows, in RULE order."""
    base_of = {}
    for b in main_rows:
        pid = _pair(b, "base")
        if pid is not None:
            base_of[pid] = b
    out = []
    for v in inv_rows:
        pid = _pair(v, "variant")
        if pid is not None and pid in base_of:
            out.append({"base": base_of[pid], "variant": v})
    return sorted(out, key=lambda p: order_key(p["variant"]["sample_id"]))


def select(pairs: List[Dict[str, Any]]) -> Dict[str, Optional[Dict[str, Any]]]:
    """First pair per slot task; looks at the task only."""
    return {t: next((p for p in pairs if p["variant"]["task"] == t), None) for t in SLOT_TASKS}


def load_responses(pred: Path, flags: Path) -> Dict[str, Any]:
    raw = pred.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    want = json.loads(flags.read_text()).get("predictions_sha256")
    if want != sha:
        raise SystemExit("%s is not the predictions file recorded in %s" % (pred, flags))
    out = {}
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["sample_id"] in out:
                raise SystemExit("sample %s appears twice in %s" % (r["sample_id"], pred))
            out[r["sample_id"]] = r["text"]
    return {"responses": out, "sha256": sha, "file": str(pred)}


def run_files(entry: Dict[str, Any], split: str, root: Path = ROOT) -> Dict[str, Dict[str, Path]]:
    pred_dir = root / "runs" / "kaggle" / entry["job"] / "runs" / entry["run"]
    dev_dir = root / entry["dev_dir"]
    return {key: {"pred": pred_dir / ("predictions_%s%s.jsonl" % (split, x)), "flags": dev_dir / ("flags_%s%s.json" % (split, x))}
            for key, x in (("main", ""), ("inv", "_inv"))}


def card(pair: Dict[str, Any], runs: List[Dict[str, Any]], styles: Dict[str, str]) -> Dict[str, Any]:
    b, v = pair["base"], pair["variant"]
    rows = []
    for r in runs:
        tb, tv = r["main"]["responses"].get(b["sample_id"]), r["inv"]["responses"].get(v["sample_id"])
        if tb is None or tv is None:
            raise SystemExit("%s has no recorded response for %s / %s" % (r["job"], b["sample_id"], v["sample_id"]))
        vb, vv = dev_gallery.wt.verdict(b, tb), dev_gallery.wt.verdict(v, tv)
        rows.append({"arm": r["arm"], "seed": r["seed"], "job": r["job"], "root_text": tb, "root": vb,
                     "copy_text": tv, "copy": vv, "same_correctness": vb["correct"] == vv["correct"]})
    return {"task": v["task"], "instruction": b["public_input"]["instruction"],
            "same_instruction": b["public_input"]["instruction"] == v["public_input"]["instruction"],
            "true_answer": tasks.format_target(b["training_target"]),
            "root": {"sample_id": b["sample_id"], "image": b["image_path"], "image_sha256": b["image_sha256"],
                     "style_id": styles.get(b["image_id"])},
            "copy": {"sample_id": v["sample_id"], "image": v["image_path"], "image_sha256": v["image_sha256"],
                     "style_id": styles.get(v["image_id"])},
            "rows": rows}


def build(table: Path, queue: Dict[str, Any], root: Path = ROOT, splits=None) -> Dict[str, Any]:
    jobs = queue["jobs"]
    final_figures.load_table(table, [e["job"] for e in jobs], splits)   # refuses before any test file is opened
    order = {a: i for i, a in enumerate(dict.fromkeys(e["arm"] for e in jobs))}
    jobs = sorted(jobs, key=lambda e: (order[e["arm"]], e["seed"]))
    out = {"rule": RULE, "table": final_figures.dev_figures._rel(Path(table)),
           "table_sha256": final_figures.dev_figures._sha256(Path(table)), "splits": {}}
    by_manifest = {}
    for manifest, split, roots in (SPLITS if splits is None else splits):
        if split not in CARD_SPLITS:
            continue
        if manifest not in by_manifest:
            by_manifest[manifest] = dataset.load_dataset(root / "manifests" / (manifest + ".json"))
        man, data_dir, by_split, images = by_manifest[manifest]
        main_rows = arms_mod.eval_samples(by_split[split], roots)
        inv_rows = arms_mod.invariance_samples(by_split[split], roots)
        pairs = style_pairs(main_rows, inv_rows)
        runs = []
        for e in jobs:
            f = run_files(e, split, root)
            runs.append({"job": e["job"], "arm": e["arm"], "seed": e["seed"],
                         "main": load_responses(f["main"]["pred"], f["main"]["flags"]),
                         "inv": load_responses(f["inv"]["pred"], f["inv"]["flags"])})
        styles_by_image = {r["image_id"]: r.get("style_id") for r in images[split]}
        chosen = select(pairs)
        out["splits"][split] = {
            "manifest": manifest, "content_digest": man["content_digest"], "root_graphs": roots,
            "data_dir": str(data_dir), "pool_pairs": len(pairs),
            "sources": [{"job": r["job"], "arm": r["arm"], "seed": r["seed"],
                         "predictions": [final_figures.dev_figures._rel(Path(r[k]["file"])) for k in ("main", "inv")],
                         "sha256": [r["main"]["sha256"], r["inv"]["sha256"]]} for r in runs],
            "slots": [{"task": t, "card": None if chosen[t] is None else card(chosen[t], runs, styles_by_image)}
                      for t in SLOT_TASKS]}
    return out


def copy_images(cards: Dict[str, Any], dest: Path) -> str:
    dest.mkdir(parents=True, exist_ok=True)
    for split in cards["splits"].values():
        for slot in split["slots"]:
            c = slot["card"]
            if c is None:
                continue
            for side in ("root", "copy"):
                src = Path(split["data_dir"]) / c[side]["image"]
                if hashlib.sha256(src.read_bytes()).hexdigest() != c[side]["image_sha256"]:
                    raise SystemExit("image %s does not match the manifest" % src)
                shutil.copyfile(src, dest / src.name)
    return dest.name


def _mark(v: Dict[str, Any]) -> str:
    return dev_gallery._mark(v)


def markdown(cards: Dict[str, Any], image_prefix: Optional[str] = None) -> str:
    cell = dev_gallery._cell
    out = ["# Locked test: style-pair cards", "",
           "Generated by `scripts/final_style_cards.py`; do not edit. One question asked on a root image and on a copy of the",
           "same graph drawn in another style (same answer by construction). Rule `%s`: pairs in a hash order that uses no" % cards["rule"],
           "response or score; per split the first reachability and the first critical-edge pair. Every response is a recorded",
           "line of a final kernel's predictions file whose sha256 matches the collected flags; the verdict is the verifier's.",
           "An illustration, not a measurement: the invariance rates are in the invariance table.", ""]
    for split, s in cards["splits"].items():
        out += ["## %s (%d style pairs in the pool)" % (split, s["pool_pairs"]), ""]
        for slot in s["slots"]:
            c = slot["card"]
            out += ["### %s" % slot["task"].replace("_", " ").capitalize(), ""]
            if c is None:
                out += ["No pair in the pool meets this rule.", ""]
                continue
            if image_prefix is not None:
                out += ["| Root image (style `%s`) | Restyled copy (style `%s`) |" % (c["root"]["style_id"], c["copy"]["style_id"]),
                        "|---|---|",
                        "| ![root](%s/%s) | ![copy](%s/%s) |" % (image_prefix, Path(c["root"]["image"]).name,
                                                               image_prefix, Path(c["copy"]["image"]).name), ""]
            out += ["Samples `%s` (root) and `%s` (copy)%s." % (c["root"]["sample_id"], c["copy"]["sample_id"],
                                                                "" if c["same_instruction"] else "; the two instructions differ"),
                    "", "```", c["instruction"], "```", "",
                    "True answer (from the generator's graph): %s" % cell(c["true_answer"]), "",
                    "| Arm | Seed | Root response | Verifier | Copy response | Verifier | Same correctness |",
                    "|---|---|---|---|---|---|---|"]
            out += ["| %s | %s | %s | %s | %s | %s | %s |" % (r["arm"], r["seed"], cell(r["root_text"]), _mark(r["root"]),
                                                          cell(r["copy_text"]), _mark(r["copy"]),
                                                          "yes" if r["same_correctness"] else "no") for r in c["rows"]]
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", default=str(ROOT / "reports/final/final_table.json"))
    ap.add_argument("--queue", default=str(ROOT / "scripts/kaggle/final_queue.json"))
    ap.add_argument("--out"); ap.add_argument("--out-md"); ap.add_argument("--copy-images")
    a = ap.parse_args(argv)
    cards = build(Path(a.table), json.loads(Path(a.queue).read_text()))
    prefix = copy_images(cards, Path(a.copy_images)) if a.copy_images else None
    for s in cards["splits"].values():
        s.pop("data_dir")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(cards, indent=1, sort_keys=True) + "\n")
    text = markdown(cards, prefix)
    if a.out_md:
        Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_md).write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
