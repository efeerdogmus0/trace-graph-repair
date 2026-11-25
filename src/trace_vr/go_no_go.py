"""Pre-registered go/no-go rule after revision 1 (fixed 2026-11-12, before job11 was read back).

Inputs are paired root-bootstrap entries for the non-edit macro (dicts with point/lo/hi, as written
by `bootstrap.cluster_bootstrap(...)["a_minus_b"][MACRO]`) on the same 200 pilot dev roots:

  machine  = B(Kaggle) - B(Colab)   same recipe, same seed, only the machine differs
  d2_vs_b  = D2(Kaggle) - B(Kaggle) same machine
  d_vs_b   = D(Colab)  - B(Colab)   same machine

Rules (brief section 20 / 20.1: at most two evidence-based revisions, then record H1 as not supported):
  1. The machine effect is "small" when its interval contains 0 and |point| <= MACHINE_TOL.
     Otherwise cross-machine comparisons (D2 vs D) are not trusted, and D is retrained on Kaggle
     before any further decision ("retrain_D_on_kaggle").
  2. If either D-arm interval vs its same-machine B lies entirely above 0, transfer is supported on
     dev at pilot scale ("H1_supported_on_dev"); go to M5 with that arm.
  3. If both D-arm intervals lie entirely below 0 and the larger point gap exceeds GAP_CLOSE,
     a single further revision cannot plausibly close it: record H1 as not supported and move to
     the negative-result path ("close_H1_not_supported"). Revision 2 is not spent.
  4. Otherwise ("revision_2"): run the last allowed revision; its success criterion is fixed now:
     paired D_rev2 - B(same machine) macro interval entirely above 0. Any other outcome closes H1.
"""
from __future__ import annotations

from typing import Any, Dict

MACHINE_TOL = 0.03
GAP_CLOSE = 0.10


def _check(e: Dict[str, Any], name: str) -> None:
    for k in ("point", "lo", "hi"):
        if k not in e:
            raise ValueError("%s lacks %s" % (name, k))
    if not e["lo"] <= e["point"] <= e["hi"]:
        raise ValueError("%s: point outside its interval" % name)


def machine_small(machine: Dict[str, Any]) -> bool:
    _check(machine, "machine")
    return machine["lo"] <= 0.0 <= machine["hi"] and abs(machine["point"]) <= MACHINE_TOL


def decide(machine: Dict[str, Any], d2_vs_b: Dict[str, Any], d_vs_b: Dict[str, Any]) -> Dict[str, Any]:
    for e, n in ((d2_vs_b, "d2_vs_b"), (d_vs_b, "d_vs_b")):
        _check(e, n)
    small = machine_small(machine)
    out: Dict[str, Any] = {"machine_small": small, "machine_tol": MACHINE_TOL, "gap_close": GAP_CLOSE}
    if not small:
        out["decision"] = "retrain_D_on_kaggle"
        out["reason"] = "machine effect not small; D2 vs D and D vs B are not comparable across machines"
        return out
    if d2_vs_b["lo"] > 0 or d_vs_b["lo"] > 0:
        out["decision"] = "H1_supported_on_dev"
        out["reason"] = "a D arm is above its same-machine B on the paired interval"
        return out
    best_gap = max(d2_vs_b["point"], d_vs_b["point"])
    out["best_point_gap"] = best_gap
    if d2_vs_b["hi"] < 0 and d_vs_b["hi"] < 0 and best_gap < -GAP_CLOSE:
        out["decision"] = "close_H1_not_supported"
        out["reason"] = "both D arms below B with gap > %.2f; revision 2 not spent" % GAP_CLOSE
        return out
    out["decision"] = "revision_2"
    out["reason"] = "gap not conclusive; last revision, success = D_rev2 - B interval above 0"
    return out
