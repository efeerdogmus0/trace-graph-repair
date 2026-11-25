"""Short English announcement draft and Turkish result summary, generated from the primary statistic (CPU only).

Usage: python scripts/result_summary.py [--primary reports/final/final_primary.json]
                                        [--out-en reports/final/announcement_draft_en.md]
                                        [--out-tr reports/final/sonuc_ozeti_tr.md]

Brief section 23 / PLAN item 39 ask for a short English announcement draft and a Turkish result summary once the
primary reading exists. Rule `summary-v1`, fixed 2026-11-20 before any locked-test number was read:
- the only input is final_primary.json as written by scripts/final_primary.py; no other report is opened, so the
  text cannot pick a nicer secondary number. It refuses (writes nothing) while that file is missing or malformed.
- the deciding comparison is D (TRACE repair) against the reference arm recorded in the file; its reading selects
  one of five fixed sentence pairs below (positive_clear / positive / negative / inconclusive / incomplete_seeds).
  The wording of each pair was written before the reading is known and is tested; do not edit it after reading.
- secondary facts are stated, never used to change the sentence: task_drop, splits_same_sign, E - D, C - reference.
- every text says pilot scale (512 training root graphs, one 2B base model, synthetic drawings) and that it is a
  draft: nothing is posted or uploaded by this script; the owner confirms any publication.
Numbers are copied from the file with fixed formatting (3 decimals, signed differences)."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = "summary-v1"
PRIMARY = "primary_six_cell"
READINGS = ("positive_clear", "positive", "negative", "inconclusive", "incomplete_seeds")

EN = {
    "positive_clear": "Training on minimal repair edits (arm D) beat the reference fine-tuning arm by a clear margin on the "
                      "pre-registered six-cell statistic, with every seed pointing the same way.",
    "positive": "Training on minimal repair edits (arm D) beat the reference fine-tuning arm on the pre-registered six-cell "
                "statistic, but by less than the pre-registered clear margin of 0.05.",
    "negative": "Training on minimal repair edits (arm D) did worse than the reference fine-tuning arm on the "
                "pre-registered six-cell statistic; the hypothesis is not supported at this scale.",
    "inconclusive": "The pre-registered six-cell statistic does not separate training on minimal repair edits (arm D) from "
                    "the reference fine-tuning arm: the interval or the seeds disagree, so we report no effect either way.",
    "incomplete_seeds": "The comparison could not be read under the pre-registered rule because a seed is missing; no "
                        "claim is made.",
}
TR = {
    "positive_clear": "Minimal onarım düzenlemeleriyle eğitim (kol D), önceden kayıtlı altı hücreli ölçüde referans "
                      "ince ayar kolunu açık farkla geçti; üç tohumun hepsi aynı yönde.",
    "positive": "Minimal onarım düzenlemeleriyle eğitim (kol D), önceden kayıtlı altı hücreli ölçüde referans kolu geçti; "
                "fark önceden belirlenen 0,05'lik açık fark eşiğinin altında.",
    "negative": "Minimal onarım düzenlemeleriyle eğitim (kol D), önceden kayıtlı altı hücreli ölçüde referans kolun "
                "gerisinde kaldı; hipotez bu ölçekte desteklenmedi.",
    "inconclusive": "Önceden kayıtlı altı hücreli ölçü kol D ile referans kolu ayıramıyor (aralık sıfırı kapsıyor ya da "
                    "tohumlar ayrışıyor); iki yönde de etki iddia edilmiyor.",
    "incomplete_seeds": "Bir tohum eksik olduğu için karşılaştırma önceden kayıtlı kurala göre okunamadı; iddia yok.",
}
SCALE_EN = ("Pilot scale: adapters were trained on 512 synthetic root graphs with one 2B base model (Qwen3.5-2B, LoRA); "
            "the locked test uses procedurally drawn graphs only. Nothing here measures real diagrams.")
SCALE_TR = ("Pilot ölçek: adaptörler tek bir 2B taban modelde (Qwen3.5-2B, LoRA) 512 sentetik kök graf ile eğitildi; "
            "kilitli test yalnız prosedürel çizilmiş graflardan oluşuyor. Gerçek diyagramlar ölçülmedi.")


def _ci(e, signed=True):
    f = "%+.3f [%+.3f, %+.3f]" if signed else "%.3f [%.3f, %.3f]"
    return f % (e["point"], e["lo"], e["hi"])


def load(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit("result summary refused: %s does not exist (run scripts/final_reports.py first)" % p)
    t = json.loads(p.read_text())
    if t.get("statistic") != PRIMARY or not t.get("reference") or not t.get("arms"):
        raise SystemExit("result summary refused: %s is not a final_primary.py output" % p)
    return t


def parts(t):
    ref = t["reference"]
    diffs = {(d["a"], d["b"]): d for d in t["differences"]}
    if ("D", ref) not in diffs:
        raise SystemExit("result summary refused: no D - %s row in the primary file" % ref)
    d = diffs[("D", ref)]
    if d["reading"] not in READINGS or d["diff"].get(PRIMARY) is None:
        raise SystemExit("result summary refused: unknown reading %r" % d["reading"])
    arms = {a["label"]: a for a in t["arms"]}
    return ref, d, diffs, arms


def render(t, lang):
    ref, d, diffs, arms = parts(t)
    tr = lang == "tr"
    head = ("# Sonuç özeti (TASLAK, paylaşılmadı)" if tr else "# Announcement draft (DRAFT, not posted)")
    out = [head, "", (TR if tr else EN)[d["reading"]], ""]
    lab = "Fark D - %s (altı hücre, üç tohum ortalaması, %%95 aralık)" if tr else "Difference D - %s (six cells, mean of three seeds, 95%% interval)"
    out.append("- %s: %s; okuma `%s`." % (lab % ref, _ci(d["diff"][PRIMARY]), d["reading"]) if tr else
               "- %s: %s; reading `%s`." % (lab % ref, _ci(d["diff"][PRIMARY]), d["reading"]))
    for label in ("D", ref):
        if label in arms:
            out.append("- %s %s: %s" % ("Kol" if tr else "Arm", label, _ci(arms[label]["pooled"][PRIMARY], signed=False)))
    seeds = ", ".join("%s: %+.3f" % (k, v) for k, v in d["seed_matched"].items())
    out.append("- %s: %s" % ("Tohum bazında fark" if tr else "Per-seed difference", seeds or "n/a"))
    if d["task_drop"]:
        out.append("- " + ("Görev bazında 0,02'den fazla düşüş: %s" if tr else "Task mean more than 0.02 below the reference: %s")
                   % ", ".join(d["task_drop"]))
    out.append("- " + (("Üç bölmede fark aynı yönde: %s" % ("evet" if d["splits_same_sign"] else "hayır")) if tr else
                       ("All three splits have the sign of the pooled difference: %s" % ("yes" if d["splits_same_sign"] else "no"))))
    for a, b in (("E", "D"), ("C", ref)):
        if (a, b) in diffs and diffs[(a, b)]["diff"].get(PRIMARY):
            out.append("- %s %s - %s: %s (%s)" % ("Raporlanan" if tr else "Reported", a, b, _ci(diffs[(a, b)]["diff"][PRIMARY]),
                                                 "karar vermez" if tr else "does not decide"))
    out += ["", SCALE_TR if tr else SCALE_EN, "",
            ("Kural `%s`, kaynak `reports/final/final_primary.json`. Yayın/yükleme sahibin onayıyla yapılır." if tr else
             "Rule `%s`, source `reports/final/final_primary.json`. Any posting or upload is confirmed by the owner first.") % RULE]
    return "\n".join(out) + "\n"


def main(argv):
    opts = {"primary": str(ROOT / "reports/final/final_primary.json"), "out-en": str(ROOT / "reports/final/announcement_draft_en.md"),
            "out-tr": str(ROOT / "reports/final/sonuc_ozeti_tr.md")}
    args = list(argv)
    for key in list(opts):
        if "--" + key in args:
            i = args.index("--" + key); opts[key] = args[i + 1]; del args[i:i + 2]
    if args:
        raise SystemExit(__doc__)
    t = load(opts["primary"])
    texts = {"out-en": render(t, "en"), "out-tr": render(t, "tr")}
    for k, text in texts.items():
        Path(opts[k]).parent.mkdir(parents=True, exist_ok=True)
        Path(opts[k]).write_text(text)
        print("wrote %s" % opts[k])


if __name__ == "__main__":
    main(sys.argv[1:])
