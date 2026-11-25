"""Prompt variants for the frozen base (arm A) and the rule that picks one on dev.

Why: with the common prompt the untrained base reasons in prose and rarely ends with the single
JSON object the strict scorer requires (runs/pilot_A/summary.json: 99 of 180 responses cut off at
160 tokens, strict macro 0.10). A floor that fails on format alone says little about what the base
can read from a drawing, so arm A may be given a prompt that asks for the answer directly.

What is and is not changed:
- only arm A's prompt; every trained arm keeps the common prompt (they learn the format in training);
- the image, the question text and the strict scorer are untouched; a variant only adds one fixed
  sentence, either after the question (`direct`) or as a system message (`system`);
- no answer prefill and no stripping of reasoning: those would change the scoring interface.

Selection rule (fixed 2026-11-14, before any variant was run; DECISIONS):
  each variant is run once on the same dev roots with the long token limit; its token limit is the
  one `token_budget.choose` picks, and its score is the strict non-edit macro at that limit. The
  variant with the highest score is used for arm A; on an exact tie the earlier one in VARIANTS
  wins (the common prompt first). Lenient scores and truncation are reported, never used to pick.
  The common-prompt result is always reported next to the chosen one.

Pure Python (no torch), so the rule is tested without a model.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

DIRECT_SENTENCE = "Do not explain your reasoning. Reply with the JSON object only."

COMMON, DIRECT, SYSTEM = "common", "direct", "system"
VARIANTS = (COMMON, DIRECT, SYSTEM)  # order = tie-break order


def apply(variant: str, prompt: str) -> Tuple[Optional[str], str]:
    """(system message or None, user prompt) for a variant. `common` returns the prompt unchanged."""
    if variant == COMMON:
        return None, prompt
    if variant == DIRECT:
        return None, prompt + "\n" + DIRECT_SENTENCE
    if variant == SYSTEM:
        return DIRECT_SENTENCE, prompt
    raise KeyError("unknown base prompt variant: %r" % (variant,))


def check_allowed(variant: str, adapter: str) -> None:
    """Variants other than the common prompt exist for the frozen base only."""
    if variant not in VARIANTS:
        raise SystemExit("unknown prompt variant %r (choose from %s)" % (variant, ", ".join(VARIANTS)))
    if variant != COMMON and adapter != "none":
        raise SystemExit("prompt variant %r is for the frozen base (adapter none) only" % variant)


def run_name(variant: str) -> str:
    if variant not in VARIANTS:
        raise KeyError("unknown base prompt variant: %r" % (variant,))
    return "base_prompt_%s" % variant


def entry(scan_result: Dict[str, Any]) -> Dict[str, Any]:
    """The numbers of one variant's `token_budget.scan` result that the rule and the report use."""
    limit = scan_result.get("chosen_limit")
    if limit is None:
        raise ValueError("token-budget result has no chosen limit")
    at = scan_result["limits"][str(limit)]
    return {
        "chosen_limit": int(limit),
        "non_edit_macro": at["non_edit_macro"],
        "non_edit_macro_lenient": at.get("non_edit_macro_lenient"),
        "max_truncation_rate": at["max_truncation_rate"],
        "per_task_accuracy": at.get("per_task_accuracy"),
        "per_task_format_failure": at.get("per_task_format_failure"),
    }


def choose(scans: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Apply the selection rule to {variant: token_budget.scan result}.

    Every variant must be present and all must describe the same evaluation (manifest content,
    split, number of roots and samples, run limit); otherwise nothing is chosen.
    """
    missing = [v for v in VARIANTS if v not in scans]
    extra = sorted(set(scans) - set(VARIANTS))
    if missing or extra:
        raise ValueError("variants missing %s, unknown %s" % (missing, extra))
    for key in ("content_digest", "split", "eval_root_graphs", "samples", "run_limit"):
        seen = {repr(scans[v].get(key)) for v in VARIANTS}
        if len(seen) != 1:
            raise ValueError("variants were not run on the same evaluation: %s differs" % key)
    if scans[COMMON].get("split") != "dev":
        raise ValueError("the base prompt is chosen on dev only, not on %r" % (scans[COMMON].get("split"),))
    entries = {v: entry(scans[v]) for v in VARIANTS}
    best = max(e["non_edit_macro"] for e in entries.values())
    chosen = next(v for v in VARIANTS if entries[v]["non_edit_macro"] == best)
    return {
        "rule": "highest strict non-edit macro at each variant's own token limit; tie -> earlier of %s" % (VARIANTS,),
        "chosen": chosen,
        "chosen_limit": entries[chosen]["chosen_limit"],
        "variants": entries,
        "split": scans[COMMON].get("split"),
        "eval_root_graphs": scans[COMMON].get("eval_root_graphs"),
        "samples": scans[COMMON].get("samples"),
        "run_limit": scans[COMMON].get("run_limit"),
        "content_digest": scans[COMMON].get("content_digest"),
        "direct_sentence": DIRECT_SENTENCE,
    }
