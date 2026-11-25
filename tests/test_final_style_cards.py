"""scripts/final_style_cards.py: style-pair cards from the locked test, gated on the accepted final table.

Everything here runs on the tiny generated dataset with made-up responses, so no locked-test output is involved.
Checked: the gate refuses before any prediction file is needed, the selection ignores responses, every shown
verdict is the verifier's, a predictions file that is not the one recorded in the flags stops the script, and
on the real repository (final table not written yet) the script refuses and writes nothing."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from trace_vr import metrics

from test_dataset import built  # noqa: F401  (module-scoped fixture)

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("final_style_cards", ROOT / "scripts" / "final_style_cards.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sc = _load()
SEEDS = sorted(sc.final_figures.SEEDS)
ARMS = ["B", "D", "E", "C"]
SPLITS = [("tiny", "test_id", 4), ("tiny", "test_style_ood", 3)]
ANSWERS = ('{"answer": true}', '{"answer": false}', 'maybe')


def _queue():
    jobs, i = [], 1
    for seed in SEEDS:
        for arm in ARMS:
            jobs.append({"job": "jobF%02d" % i, "arm": arm, "seed": seed, "run": "run_%s_%d" % (arm, seed),
                         "dev_dir": "runs/dev_%s_%d" % (arm, seed)})
            i += 1
    return {"jobs": jobs}


def _table(root, queue):
    tables = {split: {"split": split, "manifest": m, "root_graphs": n,
                      "arms": [{"label": a, "seeds": SEEDS} for a in dict.fromkeys(e["arm"] for e in queue["jobs"])]}
              for m, split, n in SPLITS}
    p = root / "final_table.json"
    p.write_text(json.dumps({"reference": "B", "jobs": [e["job"] for e in queue["jobs"]],
                             "splits": [s for _, s, _ in SPLITS], "tables": tables}))
    return p


def _write_runs(root, queue, built, salt=""):
    for e in queue["jobs"]:
        for _, split, n in SPLITS:
            files = sc.run_files(e, split, root)
            for key, rows in (("main", sc.arms_mod.eval_samples(built["samples"][split], n)),
                              ("inv", sc.arms_mod.invariance_samples(built["samples"][split], n))):
                lines = []
                for s in rows:
                    h = int(hashlib.sha256((salt + e["job"] + s["sample_id"]).encode()).hexdigest()[:8], 16)
                    lines.append(json.dumps({"sample_id": s["sample_id"], "text": ANSWERS[h % 3]}))
                raw = ("\n".join(lines) + "\n").encode()
                for p in files[key].values():
                    p.parent.mkdir(parents=True, exist_ok=True)
                files[key]["pred"].write_bytes(raw)
                files[key]["flags"].write_text(json.dumps({"predictions_sha256": hashlib.sha256(raw).hexdigest()}))


def test_refuses_without_the_final_table(built, tmp_path):
    with pytest.raises(SystemExit, match="refused"):
        sc.build(tmp_path / "final_table.json", _queue(), built["root"], SPLITS)


def test_cards_show_recorded_answers_scored_by_the_verifier(built):
    root, queue = built["root"], _queue()
    _write_runs(root, queue, built)
    cards = sc.build(_table(root, queue), queue, root, SPLITS)
    assert set(cards["splits"]) == {"test_id", "test_style_ood"}
    shown = 0
    for split, s in cards["splits"].items():
        assert s["pool_pairs"] > 0
        by_id = {x["sample_id"]: x for x in built["samples"][split]}
        for slot in s["slots"]:
            c = slot["card"]
            if c is None:
                continue
            shown += 1
            assert c["task"] == slot["task"] and c["same_instruction"]
            assert c["root"]["style_id"] != c["copy"]["style_id"]
            assert [(r["arm"], r["seed"]) for r in c["rows"]] == [(a, sd) for a in ARMS for sd in SEEDS]
            for r in c["rows"]:
                for side, key in (("root", "root_text"), ("copy", "copy_text")):
                    want = metrics.score_sample(by_id[c[side]["sample_id"]], r[key])
                    assert r[side]["correct"] == bool(want["correct"])
                assert r["same_correctness"] == (r["root"]["correct"] == r["copy"]["correct"])
    assert shown >= 2
    md = sc.markdown(cards)
    assert md.count("| B | %d |" % SEEDS[0]) == shown


def test_selection_ignores_the_responses(built):
    root, queue = built["root"], _queue()
    picked = []
    for salt in ("", "other"):
        _write_runs(root, queue, built, salt)
        cards = sc.build(_table(root, queue), queue, root, SPLITS)
        picked.append([(sl["card"] or {}).get("copy", {}).get("sample_id")
                       for s in cards["splits"].values() for sl in s["slots"]])
    assert picked[0] == picked[1]


def test_unrecorded_predictions_file_stops(built):
    root, queue = built["root"], _queue()
    _write_runs(root, queue, built)
    f = sc.run_files(queue["jobs"][0], "test_id", root)["inv"]["pred"]
    f.write_bytes(f.read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="not the predictions file"):
        sc.build(_table(root, queue), queue, root, SPLITS)


def test_real_repository_refuses_and_writes_nothing(tmp_path):
    if (ROOT / "reports/final/final_table.json").exists():
        pytest.skip("final table exists")
    out = tmp_path / "cards.md"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/final_style_cards.py"), "--out-md", str(out)],
                       capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src")})
    assert r.returncode != 0 and "refused" in r.stderr + r.stdout
    assert not out.exists()
