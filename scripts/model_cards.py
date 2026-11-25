"""Generate one model card per downloaded Kaggle LoRA adapter (CPU only, reads files, writes Markdown).

Usage: python scripts/model_cards.py [--check]

Inputs (untracked, from `kaggle kernels output`): runs/kaggle/<job>/runs/<run>/{train_summary.json,metrics_dev.json}.
The adapter hash is taken from the tracked runs/kaggle_artifacts.json, never recomputed here, so a card can only
describe an adapter whose bytes were already recorded by scripts/kaggle/artifact_manifest.py.
Locked test (since 2026-11-22): if reports/final/final_table.json exists, every run named in
scripts/kaggle/final_queue.json (as `run`, or as the `parser` of an arm-E entry) gets a "Locked-test evaluation" section with
its own seed's per-split numbers copied from that table; runs not in the queue keep the dev-only scope text.
External (since 2026-11-23): runs evaluated in the pre-registered VGCure record (reports/external/vgcure_final_seeds_vs_B.json;
a C_desc parser through its arm-E run) get an "External evaluation" section with their own seed's scores from that record.
Output: docs/model_cards/<run>.md and docs/model_cards/README.md (index). Every number is copied from those files.
--check: regenerate in memory and compare with the files on disk (exit 1 on any difference).
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "runs/kaggle"
MANIFEST = ROOT / "runs/kaggle_artifacts.json"
QUEUE = ROOT / "scripts/kaggle/final_queue.json"
FINAL_TABLE = ROOT / "reports/final/final_table.json"
VGCURE = ROOT / "reports/external/vgcure_final_seeds_vs_B.json"
TEST_TASKS = ("minimal_graph_repair", "reachability", "critical_edge", "non_edit_macro_accuracy")
OUT = ROOT / "docs/model_cards"

ARM_TEXT = {
    "B": "QA fine-tuning: answers the non-edit questions (reachability, critical edge) and the repair task directly.",
    "C": "Structural fine-tuning (MCDGraph-inspired): graph description, edge infilling and graph matching.",
    "C_desc": "Graph-description-only fine-tuning; used as the parser of the parse-and-solve arm E.",
    "D": "Repair fine-tuning: proposes one minimal edge edit that makes the reachability goals hold.",
    "D2": "Repair fine-tuning plus an edit-outcome auxiliary task (revision 1, did not beat D).",
}

SCOPE_TESTED = (
    "Research artifact from a pilot-scale study on synthetic drawn directed graphs (TRACE benchmark, "
    "`manifests/pilot.json`), three seeds per arm. Evaluated once on the locked test splits and once on a "
    "pre-registered, adapted VGCure subset (sections below). Not intended for any use outside reproducing the TRACE "
    "experiments; the numbers say nothing about natural images or other graph drawings."
)

SCOPE = (
    "Research artifact from a one-seed-per-run dev pilot on synthetic drawn directed graphs (TRACE benchmark, "
    "`manifests/pilot.json`). Not evaluated on the locked test splits or on any external benchmark yet. "
    "Not intended for any use outside reproducing the TRACE experiments; numbers below are dev-split numbers "
    "and say nothing about natural images or other graph drawings."
)


def adapters(manifest):
    """run name -> (job, adapter record) for every adapter_model.safetensors in the manifest."""
    out = {}
    for f in manifest["files"]:
        parts = f["path"].split("/")
        if parts[-1] == "adapter_model.safetensors" and len(parts) == 5 and parts[1] == "runs":
            run = parts[2]
            if run in out:
                if out[run][1]["sha256"] != f["sha256"]:
                    raise SystemExit(f"two different adapters for run {run}: {out[run][0]} and {parts[0]}")
                continue   # byte-identical copy carried by an evaluation kernel (jobF*/jobV*); keep the first job
            out[run] = (parts[0], f)
    return out


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def locked_test(queue, table):
    """run name -> (final job, arm label, seed, role) for every run the locked-test queue evaluated."""
    out = {}
    for j in queue["jobs"]:
        out[j["run"]] = (j["job"], j["arm"], j["seed"], "evaluated")
        if j.get("parser"):
            out[j["parser"]] = (j["job"], j["arm"], j["seed"], "parser")
    return out


def test_section(entry, table):
    job, arm, seed, role = entry
    lines = ["## Locked-test evaluation", ""]
    if role == "parser":
        lines += [f"Used as the graph parser of arm **E** (parse with this adapter, then the exact solver) in kernel "
                  f"`trace-{job}`, seed {seed}. The numbers are those of arm E, not of this adapter answering directly.", ""]
    else:
        lines += [f"Evaluated once in kernel `trace-{job}` (arm **{arm}**, seed {seed}).", ""]
    lines += [f"Accuracy of this seed per locked split, copied from `reports/final/final_table.json` "
              f"(pooled-over-seed intervals and the pre-registered reading: `reports/final/final_primary.md`).", "",
              "| Split | Roots | " + " | ".join(TEST_TASKS) + " |", "|---|---|" + "---|" * len(TEST_TASKS)]
    for split in table["splits"]:
        t = table["tables"][split]
        rows = [a for a in t["arms"] if a["label"] == arm]
        if len(rows) != 1 or str(seed) not in rows[0]["per_seed"]:
            raise SystemExit(f"final_table.json: no {arm} seed {seed} on {split}")
        v = rows[0]["per_seed"][str(seed)]
        lines.append(f"| {split} | {t['root_graphs']} | " + " | ".join(_fmt(float(v[k])) for k in TEST_TASKS) + " |")
    return lines + [""]


def vgcure_scores(run, record):
    """Own-seed VGCure scores of `run` (a C_desc parser through its arm-E run), or None."""
    if record is None:
        return None
    name = run.replace("C_desc", "E")
    for arm, a in record["arms"].items():
        for path, scores in zip(a["runs"], a["per_seed"]):
            if path.rsplit("/", 1)[1] == name:
                return arm, path, scores, name != run
    return None


def external_section(ext, record):
    arm, path, s, parser = ext
    return [
        "## External evaluation (pre-registered VGCure subset)",
        "",
        f"Evaluated once{' as the parser of arm E' if parser else ''} on {record['graphs']} VGCure graphs "
        f"({record['n']} items), adapted protocol (our prompts, images letterboxed to 512x512, strict JSON scoring; "
        f"not official VGCure numbers). Run `{path}`; record `reports/external/vgcure_final_seeds_vs_B.json`.",
        "",
        "| Edge-existence accuracy | Reachability balanced accuracy | External macro |",
        "|---|---|---|",
        f"| {_fmt(s['edge_exists_accuracy'])} | {_fmt(s['reachability_balanced_accuracy'])} | {_fmt(s['external_macro'])} |",
        "",
    ]


def card(run, job, rec, train, metrics, test=None, table=None, ext=None, record=None):
    arm = train["arm"]
    lines = [
        f"# TRACE adapter `{run}`",
        "",
        f"LoRA adapter for `{train['base_id']}` (revision `{train['revision']}`), arm **{arm}**, seed {train['seed']}.",
        "",
        f"- Training objective: {ARM_TEXT.get(arm, 'see src/trace_vr/arms.py')}",
        f"- Produced by Kaggle kernel `trace-{job}`; file `runs/kaggle/{rec['path']}`",
        f"- `adapter_model.safetensors`: {rec['bytes']} bytes, sha256 `{rec['sha256']}`",
        "",
        "## Scope",
        "",
        SCOPE_TESTED if test else SCOPE,
        "",
        "## Training",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key in ("manifest", "split", "content_digest", "images_digest", "root_graphs", "samples", "epochs",
                "accum", "updates", "lr", "lora_r", "seed", "tokens_in_seen", "target_tokens_seen",
                "train_seconds", "mean_loss_first_100", "mean_loss_last_100", "gpu", "peak_allocated_gib",
                "torch", "platform"):
        if key in train:
            lines.append(f"| {key} | `{_fmt(train[key], 4)}` |")
    lines += ["", f"LoRA target modules (regex): `{train.get('lora_targets', '')}`", ""]
    lines += ["## Dev evaluation", ""]
    if metrics is None:
        lines += ["No dev evaluation of this adapter on its own (it is used inside another arm).", ""]
    else:
        m = metrics["metrics"]
        lines += [
            f"{metrics['eval_root_graphs']} dev root graphs, {metrics['samples']} samples, input `{metrics['input_mode']}`, "
            f"max_new_tokens {metrics['max_new_tokens']}, truncated {metrics['truncated_at_limit']}.",
            "",
            "| Task | n | accuracy | format failures |",
            "|---|---|---|---|",
        ]
        for task, t in m["per_task"].items():
            lines.append(f"| {task} | {t['n']} | {_fmt(t['accuracy'])} | {_fmt(t['format_failure_rate'])} |")
        lines += ["", f"Non-edit macro accuracy: **{_fmt(m['non_edit_macro_accuracy'])}**.", ""]
    if test:
        lines += test_section(test, table)
    if ext:
        lines += external_section(ext, record)
    lines += [
        "## Licence",
        "",
        "Adapter weights: CC BY 4.0 (`LICENSE-DATA.md`); training and evaluation code: MIT (`LICENSE`). The base "
        "model's own licence (Apache-2.0 per its model card) applies to the base weights, which are not included.",
        "",
    ]
    return "\n".join(lines)


def build(kaggle=KAGGLE, manifest_path=MANIFEST, queue_path=QUEUE, table_path=FINAL_TABLE):
    manifest = json.loads(manifest_path.read_text())
    table = json.loads(table_path.read_text()) if table_path and table_path.exists() else None
    tested = locked_test(json.loads(queue_path.read_text()), table) if table else {}
    record = json.loads(VGCURE.read_text()) if VGCURE.exists() else None
    cards, rows = {}, []
    for run, (job, rec) in sorted(adapters(manifest).items()):
        d = kaggle / job / "runs" / run
        train = json.loads((d / "train_summary.json").read_text())
        mpath = d / "metrics_dev.json"
        metrics = json.loads(mpath.read_text()) if mpath.exists() else None
        if metrics is not None and metrics.get("adapter") != f"runs/{run}/adapter":
            raise SystemExit(f"{mpath}: evaluates {metrics.get('adapter')}, not runs/{run}/adapter")
        cards[f"{run}.md"] = card(run, job, rec, train, metrics, tested.get(run), table, vgcure_scores(run, record), record)
        macro = _fmt(metrics["metrics"]["non_edit_macro_accuracy"]) if metrics else "-"
        t = tested.get(run)
        final = f"{t[0]}" + (" (as E parser)" if t[3] == "parser" else "") if t else "-"
        rows.append(f"| [`{run}`]({run}.md) | {train['arm']} | {train['seed']} | {job} | {macro} | {final} | `{rec['sha256'][:12]}` |")
    index = "\n".join([
        "# Model cards (generated)",
        "",
        "Generated by `scripts/model_cards.py` from `runs/kaggle_artifacts.json` and the downloaded Kaggle outputs.",
        "Only adapters trained on Kaggle are listed; the Colab adapters were lost with their VMs.",
        "Macro = dev non-edit macro accuracy of the adapter itself (200 roots, image input).",
        "Locked test = the final kernel that evaluated the run once (per-split numbers in its card).",
        "",
        "| Run | Arm | Seed | Job | Dev macro | Locked test | sha256 (prefix) |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
    ])
    cards["README.md"] = index
    return cards


def main(argv):
    cards = build()
    if "--check" in argv:
        bad = [n for n, t in cards.items() if not (OUT / n).exists() or (OUT / n).read_text() != t]
        if bad:
            print("differs:", ", ".join(bad))
            return 1
        print(f"ok: {len(cards)} files")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for n, t in cards.items():
        (OUT / n).write_text(t)
    print(f"wrote {len(cards)} files to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
