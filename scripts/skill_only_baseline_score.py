#!/usr/bin/env python3
"""
Score the SKILL-DOCUMENT-ONLY LLM BASELINE against CTA ground truth.

Ground truth:
  - dominant SIP per task: argmax of
    cta_combined_results.json[task].modules.sip_detection.sip_statistics.by_type[t].count
  - dP sign: sign(cta_task_metadata.json[task].pass_rate_delta)

Reported ALWAYS alongside the trivial majority-class baseline, because 45/49
tasks have dP == 0 (always-predict-no_change scores 91.8%).
"""
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("<REPO_ROOT>/rebuttal")
OUT = ROOT / "results" / "skill_only_baseline"
SIPS = ["procedural_scaffolding", "edge_case_prompting", "redundant_exploration",
        "surface_anchoring", "concept_bleed"]
EFFECTS = ["raise", "no_change", "lower"]
EFF2SIGN = {"raise": 1, "no_change": 0, "lower": -1}


def load_truth():
    d = json.load(open(ROOT / "CTA/cta_output/cta_combined_results.json"))
    m = json.load(open(ROOT / "CTA/config/cta_task_metadata.json"))
    truth = {}
    for k in sorted(d):
        bt = d[k]["modules"]["sip_detection"]["sip_statistics"].get("by_type", {}) or {}
        counts = {t: bt[t]["count"] for t in bt}
        total = sum(counts.values())
        if counts:
            mx = max(counts.values())
            arg = sorted(t for t, c in counts.items() if c == mx)
        else:
            arg = []
        dp = m[k]["pass_rate_delta"]
        truth[k] = {
            "sip_counts": counts, "total_sips": total, "dominant_sip_set": arg,
            "dominant_sip": arg[0] if len(arg) == 1 else None,
            "is_tie": len(arg) > 1,
            "pass_rate_delta": dp,
            "dp_sign": 0 if abs(dp) < 1e-9 else (1 if dp > 0 else -1),
            "baseline_pass_rate": m[k]["baseline_pass_rate"],
        }
    return truth


def cohen_kappa(pairs, labels):
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def perm_p(pairs, labels, n_perm=20000, seed=0):
    """P(accuracy >= observed) when predictions are randomly reassigned to tasks."""
    rng = random.Random(seed)
    obs = sum(1 for a, b in pairs if a == b)
    preds = [b for _, b in pairs]
    truths = [a for a, _ in pairs]
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(preds)
        if sum(1 for t, p in zip(truths, preds) if t == p) >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _comb(n, k):
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num, den = 1, 1
    for i in range(k):
        num *= n - i
        den *= i + 1
    return num // den


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant counts b (model right, base wrong)
    and c (base right, model wrong)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(_comb(n, i) for i in range(0, k + 1)) / float(2 ** n)
    return min(1.0, 2 * tail)


def score_model(model, truth, lines, outdir=None):
    pf = (outdir or OUT) / f"predictions_{model}.json"
    preds = {r["task_id"]: (r.get("parsed") or {}) for r in json.load(open(pf))}
    tasks = sorted(preds)
    lines.append(f"\n{'='*78}\nMODEL: {model}   (n_tasks_run = {len(tasks)})\n{'='*78}")

    # ---------------- (A) dP-sign ----------------
    dp_rows = []
    for t in tasks:
        pe = preds[t].get("pass_rate_effect")
        dp_rows.append((t, truth[t]["dp_sign"], EFF2SIGN.get(pe, None), pe))
    n = len(dp_rows)
    correct = sum(1 for _, g, p, _ in dp_rows if p is not None and p == g)
    gt_dist = Counter(g for _, g, _, _ in dp_rows)
    maj = gt_dist.most_common(1)[0]
    maj_correct = maj[1]
    lo, hi = wilson(correct, n)
    lines.append("\n[A] dP-SIGN PREDICTION (does the skill raise / not change / lower pass rate)")
    lines.append(f"  ground-truth sign distribution: "
                 f"{{+1: {gt_dist.get(1,0)}, 0: {gt_dist.get(0,0)}, -1: {gt_dist.get(-1,0)}}}")
    lines.append(f"  MAJORITY-CLASS BASELINE (always predict sign={maj[0]}): "
                 f"{maj_correct}/{n} = {maj_correct/n:.1%}")
    lines.append(f"  LLM (skill-doc only)                        : "
                 f"{correct}/{n} = {correct/n:.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
    lines.append(f"  DELTA vs majority baseline                  : "
                 f"{(correct-maj_correct)/n:+.1%} ({correct-maj_correct:+d} tasks)")
    b = sum(1 for _, g, p, _ in dp_rows if p == g and g != maj[0])
    c = sum(1 for _, g, p, _ in dp_rows if p != g and g == maj[0])
    lines.append(f"  McNemar vs majority baseline (b={b}, c={c}) : p = {mcnemar_exact(b,c):.4g}")
    pred_dist = Counter(p for _, _, p, _ in dp_rows)
    lines.append(f"  LLM predicted sign distribution: "
                 f"{{+1: {pred_dist.get(1,0)}, 0: {pred_dist.get(0,0)}, -1: {pred_dist.get(-1,0)}}}")
    nz = [(t, g, p) for t, g, p, _ in dp_rows if g != 0]
    hit_nz = sum(1 for _, g, p in nz if g == p)
    lines.append(f"  recall on the {len(nz)} tasks that ACTUALLY moved (|dP|>0): {hit_nz}/{len(nz)}")
    for t, g, p in nz:
        lines.append(f"      {t:34s} true_sign={g:+d}  pred_sign={str(p):>4s}  "
                     f"dP={truth[t]['pass_rate_delta']:+.4f}")
    lines.append("  3x3 confusion (rows=true sign, cols=pred sign) order [-1, 0, +1]:")
    for g in (-1, 0, 1):
        row = [sum(1 for _, gg, pp, _ in dp_rows if gg == g and pp == p) for p in (-1, 0, 1)]
        lines.append(f"      true {g:+d} : {row}")

    # ---------------- (B) dominant SIP ----------------
    lines.append("\n[B] DOMINANT-SIP PREDICTION (vs CTA's per-task argmax SIP)")
    evalable = [t for t in tasks if truth[t]["total_sips"] > 0]
    unique = [t for t in evalable if not truth[t]["is_tie"]]
    ties = [t for t in evalable if truth[t]["is_tie"]]
    zeroed = [t for t in tasks if truth[t]["total_sips"] == 0]
    lines.append(f"  tasks run={len(tasks)}; CTA found 0 SIPs on {len(zeroed)} "
                 f"(undefined ground truth, EXCLUDED): {zeroed}")
    lines.append(f"  scored on {len(unique)} tasks with a unique argmax "
                 f"(+{len(ties)} tie task(s) reported separately: {ties})")

    pairs = [(truth[t]["dominant_sip"], preds[t].get("dominant_sip")) for t in unique]
    n2 = len(pairs)
    acc = sum(1 for a, bq in pairs if a == bq)
    gtd = Counter(a for a, _ in pairs)
    prd = Counter(bq for _, bq in pairs)
    majsip, majn = gtd.most_common(1)[0]
    prior_rand = sum((v / n2) ** 2 for v in gtd.values())
    lo2, hi2 = wilson(acc, n2)
    lines.append(f"  CTA ground-truth dominant-SIP distribution: {dict(gtd)}")
    lines.append(f"  LLM predicted dominant-SIP distribution   : {dict(prd)}")
    lines.append(f"  ---- accuracy ----")
    lines.append(f"  UNIFORM-RANDOM baseline (1/5)                       : {1/5:.1%}")
    lines.append(f"  PRIOR-MATCHED RANDOM baseline (sum p_i^2)           : {prior_rand:.1%}")
    lines.append(f"  MAJORITY-CLASS baseline (always '{majsip}'): {majn}/{n2} = {majn/n2:.1%}")
    lines.append(f"  LLM (skill-doc only)                                : "
                 f"{acc}/{n2} = {acc/n2:.1%}   95% CI [{lo2:.1%}, {hi2:.1%}]")
    lines.append(f"  DELTA vs majority-class baseline                    : "
                 f"{(acc-majn)/n2:+.1%} ({acc-majn:+d} tasks)")
    k = cohen_kappa(pairs, SIPS)
    lines.append(f"  Cohen's kappa (LLM vs CTA)                          : {k:+.3f}")
    lines.append(f"  permutation p (acc >= obs under label reshuffle)    : "
                 f"{perm_p(pairs, SIPS):.4g}")
    b2 = sum(1 for a, bq in pairs if a == bq and a != majsip)
    c2 = sum(1 for a, bq in pairs if a != bq and a == majsip)
    lines.append(f"  McNemar vs majority-class (b={b2}, c={c2})            : "
                 f"p = {mcnemar_exact(b2, c2):.4g}")
    for t in ties:
        hit = preds[t].get("dominant_sip") in truth[t]["dominant_sip_set"]
        lines.append(f"  tie task {t}: true={truth[t]['dominant_sip_set']} "
                     f"pred={preds[t].get('dominant_sip')} -> {'HIT' if hit else 'MISS'}")

    lines.append("\n  confusion matrix (rows = CTA truth, cols = LLM prediction)")
    hdr = "".join(f"{s[:4].upper():>6s}" for s in SIPS)
    lines.append(f"      {'':>28s}{hdr}   support")
    for a in SIPS:
        row = [sum(1 for x, y in pairs if x == a and y == s) for s in SIPS]
        lines.append(f"      {a:>28s}" + "".join(f"{v:6d}" for v in row) + f"   {gtd.get(a,0):7d}")
    lines.append("\n  per-class precision / recall (LLM vs CTA)")
    for s in SIPS:
        tp = sum(1 for x, y in pairs if x == s and y == s)
        fp = sum(1 for x, y in pairs if x != s and y == s)
        fn = sum(1 for x, y in pairs if x == s and y != s)
        pr = tp / (tp + fp) if tp + fp else float("nan")
        rc = tp / (tp + fn) if tp + fn else float("nan")
        lines.append(f"      {s:>28s}  P={pr:.2f}  R={rc:.2f}  (tp={tp} fp={fp} fn={fn})")

    # collapsed 3-way: constructive / neutral / destructive
    cat = {"procedural_scaffolding": "constructive", "edge_case_prompting": "constructive",
           "redundant_exploration": "neutral",
           "surface_anchoring": "destructive", "concept_bleed": "destructive"}
    cpairs = [(cat[a], cat[bq]) for a, bq in pairs if bq in cat]
    cacc = sum(1 for a, bq in cpairs if a == bq)
    cg = Counter(a for a, _ in cpairs)
    cmaj = cg.most_common(1)[0]
    lines.append(f"\n  collapsed to 3 valence classes (constructive/neutral/destructive):")
    lines.append(f"      truth dist {dict(cg)}")
    lines.append(f"      majority baseline ('{cmaj[0]}'): {cmaj[1]}/{len(cpairs)} = {cmaj[1]/len(cpairs):.1%}")
    lines.append(f"      LLM: {cacc}/{len(cpairs)} = {cacc/len(cpairs):.1%}   "
                 f"kappa = {cohen_kappa(cpairs, sorted(set(cat.values()))):+.3f}")

    # ---------------- (C) confidence calibration ----------------
    confs = [(preds[t].get("confidence"), truth[t]["dominant_sip"] == preds[t].get("dominant_sip"))
             for t in unique if isinstance(preds[t].get("confidence"), (int, float))]
    if confs:
        mc = sum(c for c, _ in confs) / len(confs)
        ch = [c for c, h in confs if h]
        cm = [c for c, h in confs if not h]
        lines.append(f"\n[C] SELF-REPORTED CONFIDENCE (on dominant_sip)")
        lines.append(f"  mean stated confidence {mc:.3f} vs actual accuracy {acc/n2:.3f}  "
                     f"-> overconfidence {mc - acc/n2:+.3f}")
        lines.append(f"  mean conf when RIGHT {sum(ch)/len(ch):.3f} (n={len(ch)}); "
                     f"when WRONG {sum(cm)/len(cm):.3f} (n={len(cm)})"
                     if ch and cm else "  (degenerate split)")
    return {"model": model, "tasks": tasks, "unique": unique,
            "dp": {"n": n, "llm": correct, "majority": maj_correct},
            "sip": {"n": n2, "llm": acc, "majority": majn, "kappa": k,
                    "prior_random": prior_rand},
            "pairs": pairs, "preds": preds}


def main():
    truth = load_truth()
    lines = []
    lines.append("SKILL-DOCUMENT-ONLY LLM BASELINE vs CTA")
    lines.append("=" * 78)
    lines.append("The model sees ONLY <task>.md + <task>/SKILL.md. It sees NO execution")
    lines.append("trace and NO CTA output. This is therefore a LOWER BOUND on Reviewer 2's")
    lines.append("requested 'simple LLM looking at the two traces' baseline, which needs")
    lines.append("raw traces that are not present in this repo.")
    lines.append(f"\nGround-truth corpus: 49 tasks. dP-sign distribution over all 49: "
                 f"{dict(Counter(truth[t]['dp_sign'] for t in truth))}")

    out = {}
    for model in ["gemini", "claude-opus-4.8"]:
        if (OUT / f"predictions_{model}.json").exists():
            out[model] = score_model(model, truth, lines)

    # cross-model agreement on the shared subset
    if len(out) == 2:
        shared = sorted(set(out["gemini"]["tasks"]) & set(out["claude-opus-4.8"]["tasks"]))
        g, o = out["gemini"]["preds"], out["claude-opus-4.8"]["preds"]
        ag_sip = sum(1 for t in shared if g[t].get("dominant_sip") == o[t].get("dominant_sip"))
        ag_dp = sum(1 for t in shared
                    if g[t].get("pass_rate_effect") == o[t].get("pass_rate_effect"))
        lines.append(f"\n{'='*78}\nCROSS-MODEL AGREEMENT on the {len(shared)} shared tasks\n{'='*78}")
        lines.append(f"  dominant_sip agreement gemini vs opus : {ag_sip}/{len(shared)} = "
                     f"{ag_sip/len(shared):.1%}  kappa="
                     f"{cohen_kappa([(g[t].get('dominant_sip'), o[t].get('dominant_sip')) for t in shared], SIPS):+.3f}")
        lines.append(f"  pass_rate_effect agreement            : {ag_dp}/{len(shared)} = "
                     f"{ag_dp/len(shared):.1%}")
        lines.append("  per-task (task | CTA truth | gemini | opus):")
        for t in shared:
            lines.append(f"    {t:34s} | {str(truth[t]['dominant_sip_set']):50s} | "
                         f"{str(g[t].get('dominant_sip')):24s} | {o[t].get('dominant_sip')}")
        # opus scored on the same 15 tasks that gemini also covers
        sub = [t for t in shared if truth[t]["total_sips"] > 0 and not truth[t]["is_tie"]]
        gp = [(truth[t]["dominant_sip"], g[t].get("dominant_sip")) for t in sub]
        op = [(truth[t]["dominant_sip"], o[t].get("dominant_sip")) for t in sub]
        lines.append(f"\n  head-to-head on the {len(sub)} scoreable shared tasks:")
        lines.append(f"    gemini SIP acc {sum(1 for a,b in gp if a==b)}/{len(sub)}   "
                     f"opus SIP acc {sum(1 for a,b in op if a==b)}/{len(sub)}")

    # ------------------------------------------------------------------ #
    # Headline block + honesty caveats
    # ------------------------------------------------------------------ #
    g = out.get("gemini")
    lines.append(f"\n{'='*78}\nHEADLINE (rebuttal-ready)\n{'='*78}")
    if g:
        lines.append(
            f"  dP-sign, n=49:  skill-doc-only LLM {g['dp']['llm']}/49 = "
            f"{g['dp']['llm']/49:.1%}   vs   always-predict-zero "
            f"{g['dp']['majority']}/49 = {g['dp']['majority']/49:.1%}   "
            f"(LLM is {g['dp']['majority']-g['dp']['llm']} tasks WORSE than trivial)")
        lines.append(
            f"  dominant SIP, n=40:  LLM {g['sip']['llm']}/40 = {g['sip']['llm']/40:.1%}"
            f"   vs   majority-class {g['sip']['majority']}/40 = "
            f"{g['sip']['majority']/40:.1%}   vs   prior-matched random "
            f"{g['sip']['prior_random']:.1%};  kappa = {g['sip']['kappa']:+.3f}")
    lines.append("\nCAVEATS (state these in the rebuttal):")
    lines.append("  1. LOWER BOUND ONLY. This baseline never sees the two execution traces.")
    lines.append("     Reviewer 2 asked for an LLM that READS BOTH TRACES; that experiment")
    lines.append("     needs the raw .jsonl traces, which live only on the author machine.")
    lines.append("     This result bounds how much is recoverable from the skill doc alone;")
    lines.append("     it does NOT by itself establish that a trace-reading LLM would fail.")
    lines.append("  2. The dominant-SIP comparison scores the LLM against CTA, not against")
    lines.append("     human labels. Disagreement therefore CANNOT adjudicate which of the")
    lines.append("     two is correct. It shows only non-recoverability, not CTA validity.")
    lines.append("     The human gold set (meta-review priority 2) remains required.")
    lines.append("  3. The dP-sign result is different in kind: it is scored against MEASURED")
    lines.append("     unit-test pass rates, with no CTA involvement. That half is a clean")
    lines.append("     ground-truth comparison and is the load-bearing number.")
    lines.append("  4. Part of the SIP disagreement is a CTA detector prior, not LLM error:")
    lines.append("     CTA emits procedural_scaffolding on only 11/522 fires (4/49 tasks)")
    lines.append("     because PS is deliberately suppressed on unilateral_action")
    lines.append("     divergences, whereas the document-only LLM predicts PS as dominant on")
    lines.append("     17/40 tasks. Do not present the full gap as purely the LLM's failure.")
    lines.append("  5. n is small (40 scoreable tasks, 1 instance/skill, r=1). CIs are wide.")

    txt = "\n".join(lines)
    (OUT / "scoring_report.txt").write_text(txt + "\n")

    # machine-readable table
    rows = []
    for model, r in out.items():
        for t in r["tasks"]:
            p = r["preds"][t]
            rows.append({
                "model": model, "task_id": t,
                "true_dp": truth[t]["pass_rate_delta"], "true_dp_sign": truth[t]["dp_sign"],
                "pred_effect": p.get("pass_rate_effect"),
                "pred_dp_sign": EFF2SIGN.get(p.get("pass_rate_effect")),
                "dp_correct": EFF2SIGN.get(p.get("pass_rate_effect")) == truth[t]["dp_sign"],
                "true_dominant_sip": truth[t]["dominant_sip"],
                "true_dominant_sip_set": truth[t]["dominant_sip_set"],
                "true_total_sips": truth[t]["total_sips"],
                "pred_dominant_sip": p.get("dominant_sip"),
                "sip_correct": (truth[t]["dominant_sip"] is not None
                                and truth[t]["dominant_sip"] == p.get("dominant_sip")),
                "pred_confidence": p.get("confidence"),
                "baseline_pass_rate": truth[t]["baseline_pass_rate"],
                "reasoning": p.get("reasoning"),
            })
    (OUT / "scoring_table.json").write_text(json.dumps(rows, indent=1))
    hdr = ["model", "task_id", "true_dp", "true_dp_sign", "pred_effect", "dp_correct",
           "true_dominant_sip", "pred_dominant_sip", "sip_correct", "pred_confidence",
           "true_total_sips", "baseline_pass_rate"]
    with open(OUT / "scoring_table.csv", "w") as f:
        f.write(",".join(hdr) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in hdr) + "\n")
    print(txt)


if __name__ == "__main__":
    main()
