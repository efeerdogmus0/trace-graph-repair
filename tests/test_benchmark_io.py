"""scripts/benchmark_io.py: the model-agnostic export / score path.

Checks that the export leaks no label, that the scorer's numbers are the library's own
(metrics.evaluate, bootstrap.cluster_bootstrap, analysis.pair_table), that the flags file it
writes is accepted by scripts/bootstrap_ci.py, and that bad prediction files and locked
splits are refused."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from trace_vr import analysis, arms, bootstrap, metrics, tasks
from test_dataset import built  # noqa: F401  (module-scoped fixture)

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "train"  # the tiny build has no dev split; "train" is not locked


def _run(*args, env=None):
    e = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
    e.update(env or {})
    return subprocess.run([sys.executable, "scripts/benchmark_io.py", *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True, env=e)


def _rows(built, roots=10 ** 9):
    return arms.eval_samples(built["samples"][SPLIT], roots)


def _write(path, rows, answer):
    lines = [json.dumps({"sample_id": s["sample_id"], "text": answer(i, s)}) for i, s in enumerate(rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {s["sample_id"]: answer(i, s) for i, s in enumerate(rows)}


def _truth(i, s):
    return tasks.format_target(s["training_target"])


def test_export_has_every_main_sample_and_no_label(built, tmp_path):
    out = tmp_path / "prompts.jsonl"
    r = _run("export", built["manifest_path"], SPLIT, "--out", out)
    assert r.returncode == 0, r.stderr
    recs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    rows = _rows(built)
    assert [x["sample_id"] for x in recs] == [s["sample_id"] for s in rows]
    for x, s in zip(recs, rows):
        assert set(x) == {"sample_id", "task", "image", "prompt"}
        assert x["prompt"] == s["public_input"]["instruction"] and x["task"] == s["task"]
        assert (built["data_dir"] / x["image"]).is_file()
    assert "private_truth" not in out.read_text() and "training_target" not in out.read_text()


def test_roots_option_limits_export(built, tmp_path):
    out = tmp_path / "p.jsonl"
    assert _run("export", built["manifest_path"], SPLIT, "--out", out, "--roots", 2).returncode == 0
    ids = [json.loads(line)["sample_id"] for line in out.read_text().splitlines()]
    assert ids == [s["sample_id"] for s in _rows(built, 2)]
    assert _run("export", built["manifest_path"], SPLIT, "--out", out, "--roots", 999).returncode != 0


def test_score_equals_library_numbers_and_flags_feed_bootstrap_ci(built, tmp_path):
    rows = _rows(built)
    responses = _write(tmp_path / "pred.jsonl", rows, lambda i, s: _truth(i, s) if i % 3 else "not json")
    rep, flags = tmp_path / "rep.json", tmp_path / "flags.json"
    r = _run("score", built["manifest_path"], SPLIT, tmp_path / "pred.jsonl", "--out", rep, "--flags", flags,
             "--boot", 50, "--name", "mine")
    assert r.returncode == 0, r.stderr
    report = json.loads(rep.read_text())
    assert report["metrics"] == json.loads(json.dumps(metrics.evaluate(rows, responses, lenient=True)))
    payload = json.loads(flags.read_text())
    decoded = bootstrap.decode_flags(rows, payload)
    assert [decoded[s["sample_id"]] for s in rows] == [i % 3 != 0 for i in range(len(rows))]
    assert report["twin_pairs"] == analysis.pair_table(rows, decoded)
    assert report["intervals"] == json.loads(json.dumps(bootstrap.cluster_bootstrap(rows, decoded, n_boot=50)))
    assert payload["run"] == "mine" and payload["content_digest"] == built["manifest"]["content_digest"]
    assert "non-edit macro" in r.stdout
    # the same flags file is what the paired-comparison script reads
    ci_out = tmp_path / "ci.json"
    ci = subprocess.run([sys.executable, "scripts/bootstrap_ci.py", str(built["manifest_path"]), str(flags), str(flags),
                         "--boot", "20", "--out", str(ci_out)], cwd=ROOT, capture_output=True, text=True,
                        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert ci.returncode == 0, ci.stderr
    assert json.loads(ci_out.read_text())["a_minus_b"][bootstrap.MACRO]["point"] == 0


def test_oracle_scores_one_and_missing_lines_count_as_wrong(built, tmp_path):
    rows = _rows(built)
    _write(tmp_path / "all.jsonl", rows, _truth)
    rep = tmp_path / "rep.json"
    assert _run("score", built["manifest_path"], SPLIT, tmp_path / "all.jsonl", "--out", rep, "--boot", 20).returncode == 0
    full = json.loads(rep.read_text())
    assert full["metrics"]["non_edit_macro_accuracy"] == 1.0 and full["metrics"]["missing_responses"] == 0
    assert all(c["both_right"] == c["pairs"] > 0 for c in full["twin_pairs"].values())
    _write(tmp_path / "half.jsonl", rows[: len(rows) // 2], _truth)
    assert _run("score", built["manifest_path"], SPLIT, tmp_path / "half.jsonl", "--out", rep, "--boot", 20).returncode == 0
    half = json.loads(rep.read_text())["metrics"]
    assert half["missing_responses"] == len(rows) - len(rows) // 2
    assert sum(round(e["accuracy"] * e["n"]) for e in half["per_task"].values()) == len(rows) // 2


@pytest.mark.parametrize("lines, message", [
    (['{"sample_id": "nope", "text": "x"}'], "not in the exported set"),
    (["DUP", "DUP"], "appears twice"),
    (['{"id": 1}'], "expected"),
    (['{"sample_id": "FIRST", "text": 3}'], "must be a string"),
])
def test_bad_prediction_files_are_refused(built, tmp_path, lines, message):
    first = _rows(built)[0]["sample_id"]
    dup = json.dumps({"sample_id": first, "text": "x"})
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(dup if x == "DUP" else x.replace("FIRST", first) for x in lines) + "\n")
    r = _run("score", built["manifest_path"], SPLIT, path, "--boot", 20)
    assert r.returncode != 0 and message in r.stderr


def test_locked_split_refused_without_unlock(built, tmp_path):
    r = _run("export", built["manifest_path"], "test_id", "--out", tmp_path / "p.jsonl")
    assert r.returncode != 0 and "locked" in r.stderr and not (tmp_path / "p.jsonl").exists()
    r = _run("score", built["manifest_path"], "test_id", tmp_path / "none.jsonl")
    assert r.returncode != 0 and "locked" in r.stderr
