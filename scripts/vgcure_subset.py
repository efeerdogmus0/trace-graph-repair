"""Build the pre-registered VGCure external-evaluation subset.

usage: python scripts/vgcure_subset.py [--dev] <dir with understanding/ and reasoning/ JSONL files> [out_full.json]

Writes reports/external/vgcure_subset.json (selection record: revision, file hashes, subset digest,
sample ids, counts; no question text, VGCure states no licence) and, if given, the full subset with
instructions and answers to out_full.json (keep it out of git).
"""
import json
import sys
from pathlib import Path

from trace_vr import vgcure


def main() -> None:
    dev = "--dev" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--dev"]
    src = Path(sys.argv[1])
    sub = (vgcure.build_subset(src, vgcure.DEV_PER_SOURCE, skip=vgcure.PER_SOURCE) if dev
           else vgcure.build_subset(src))
    record = {k: v for k, v in sub.items() if k != "samples"}
    record["n_samples"] = len(sub["samples"])
    record["counts"] = vgcure.summary(sub)
    record["sample_ids"] = [s["sample_id"] for s in sub["samples"]]
    out = Path(vgcure.DEV_RECORD if dev else vgcure.TEST_RECORD)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(sub, sort_keys=True) + "\n")
    print("VGCURE-SUBSET", json.dumps({"n": record["n_samples"], "sha256": record["subset_sha256"]}))


if __name__ == "__main__":
    main()
