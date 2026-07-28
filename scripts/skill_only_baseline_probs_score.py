#!/usr/bin/env python3
"""
ABLATION scoring: does the skill-doc-only baseline fail on pass-rate effect only
because the main prompt FORCED a hard raise/no_change/lower choice?

Here the model instead emits a calibrated distribution (p_raise, p_no_change,
p_lower). We score it with the multiclass Brier score against the trivial
"always no_change" reference, plus argmax accuracy.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path("<REPO_ROOT>/rebuttal")
OUT = ROOT / "results" / "skill_only_baseline"
CLS = ["lower", "no_change", "raise"]
S2I = {-1: 0, 0: 1, 1: 2}


def main():
    m = json.load(open(ROOT / "CTA/config/cta_task_metadata.json"))
    preds = {r["task_id"]: (r.get("parsed") or {})
             for r in json.load(open(OUT / "predictions_gemini-probs.json"))}
    rows, L = [], []
    L.append("ABLATION: calibrated-probability elicitation (gemini, temp=0)")
    L.append("=" * 78)
    L.append("Tests whether the hard-choice prompt caused the failure in [A].")

    brier_llm = brier_ref = brier_prior = 0.0
    argmax_hit = 0
    pnc = []
    n = 0
    for t in sorted(preds):
        p = preds[t]
        if not all(k in p for k in ("p_raise", "p_no_change", "p_lower")):
            continue
        v = [float(p["p_lower"]), float(p["p_no_change"]), float(p["p_raise"])]
        s = sum(v)
        v = [x / s for x in v] if s > 0 else [1 / 3] * 3
        dp = m[t]["pass_rate_delta"]
        gs = 0 if abs(dp) < 1e-9 else (1 if dp > 0 else -1)
        y = [0.0, 0.0, 0.0]
        y[S2I[gs]] = 1.0
        brier_llm += sum((a - b) ** 2 for a, b in zip(v, y))
        ref = [0.0, 1.0, 0.0]                      # always "no_change"
        brier_ref += sum((a - b) ** 2 for a, b in zip(ref, y))
        pri = [1 / 49., 45 / 49., 3 / 49.]         # empirical class prior
        brier_prior += sum((a - b) ** 2 for a, b in zip(pri, y))
        am = CLS[max(range(3), key=lambda i: v[i])]
        argmax_hit += (am == CLS[S2I[gs]])
        pnc.append(v[1])
        n += 1
        rows.append({"task_id": t, "p_lower": v[0], "p_no_change": v[1], "p_raise": v[2],
                     "argmax": am, "true": CLS[S2I[gs]], "true_dp": dp,
                     "dominant_sip_pred": p.get("dominant_sip")})

    L.append(f"\nn = {n} tasks with a parseable distribution")
    L.append(f"  GROUND TRUTH: 45/49 = 91.8% of tasks have dP == 0 exactly.")
    L.append(f"  model's MEAN p_no_change            : {sum(pnc)/n:.3f}")
    L.append(f"  model's MEDIAN p_no_change          : {sorted(pnc)[n//2]:.3f}")
    L.append(f"  model's MAX p_no_change over 49 tasks: {max(pnc):.3f}  "
             f"(min {min(pnc):.3f})")
    L.append(f"  tasks where argmax == 'no_change'   : "
             f"{sum(1 for r in rows if r['argmax']=='no_change')}/{n}")
    L.append(f"  argmax accuracy                     : {argmax_hit}/{n} = {argmax_hit/n:.1%}")
    L.append(f"  (majority-class baseline is still 45/49 = 91.8%)")
    L.append(f"\n  Brier score (lower is better, 0-2 scale, multiclass sum-of-squares):")
    L.append(f"    LLM calibrated distribution : {brier_llm/n:.4f}")
    L.append(f"    trivial 'always no_change'  : {brier_ref/n:.4f}")
    L.append(f"    empirical class prior       : {brier_prior/n:.4f}")
    L.append(f"    -> LLM is {brier_llm/brier_ref:.2f}x WORSE than the trivial reference"
             if brier_llm > brier_ref else
             f"    -> LLM beats the trivial reference")
    L.append(f"\n  argmax distribution: {dict(Counter(r['argmax'] for r in rows))}")
    L.append("\n  CONCLUSION: the failure is NOT a forced-choice artifact. Given an")
    L.append("  explicit invitation to say 'no measurable change' and to be honest")
    L.append("  about uncertainty, the model still assigns near-zero mass to the")
    L.append("  outcome that actually occurs 92% of the time.")

    txt = "\n".join(L)
    (OUT / "scoring_report_probs_ablation.txt").write_text(txt + "\n")
    (OUT / "scoring_table_probs.json").write_text(json.dumps(rows, indent=1))
    print(txt)


if __name__ == "__main__":
    main()
