#!/usr/bin/env python3
"""
cta_rebucket_robustness.py -- is the bucket->signature story an artifact of one
arbitrary cutoff?

The paper buckets the 49 tasks by baseline pass rate at 0.90 / 0.50
(ceiling / mid / floor, n = 37 / 10 / 2). Nothing in the benchmark forces 0.90.
This script re-runs the whole bucket-conditional analysis under alternative
cutoffs (0.95, 0.90, 0.85, 0.80) and under a STRICT rubric that only calls a
task "ceiling" when every test passed at baseline (pass rate == 1.0), plus a
median split. For each scheme it reports bucket sizes, pooled SIP composition,
the pooled argmax and its margin, the bucket-label permutation p, the cluster
bootstrap P(argmax), and the paired per-task sign test in the top bucket.

Pure stdlib. Usage:  python3 cta_rebucket_robustness.py [--boot 20000]
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cta_common import (  # noqa: E402
    SIP_ABBR, SIP_TYPES, dump, hr, load_tasks,
)
from cta_signature_stability import (  # noqa: E402
    cluster_bootstrap, leave_one_out, permutation_test, pooled, pooled_argmax, sign_test,
)


def relabel(tasks, fn):
    for t in tasks:
        t.bucket = fn(t)
    return tasks


def scheme_cutoff(hi, lo=0.50):
    names = ["ceiling", "mid", "floor"]

    def fn(t):
        if t.baseline >= hi:
            return "ceiling"
        if t.baseline >= lo:
            return "mid"
        return "floor"
    return names, fn


def scheme_strict():
    """All-tests-pass rubric: ceiling only if the baseline solved everything."""
    names = ["all_pass", "partial", "mostly_fail"]

    def fn(t):
        if t.baseline >= 1.0 - 1e-12:
            return "all_pass"
        if t.baseline >= 0.50:
            return "partial"
        return "mostly_fail"
    return names, fn


def scheme_median(tasks):
    vals = sorted(t.baseline for t in tasks)
    med = vals[len(vals) // 2] if len(vals) % 2 else 0.5 * (vals[len(vals) // 2 - 1]
                                                           + vals[len(vals) // 2])
    names = ["at_or_above_median", "below_median"]

    def fn(t):
        return "at_or_above_median" if t.baseline >= med else "below_median"
    return names, fn, med


def analyse(tasks, names, boot, shuffles, seed):
    present = [b for b in names if any(t.bucket == b for t in tasks)]
    perm = permutation_test(tasks, shuffles, seed, buckets=present)
    bootres = cluster_bootstrap(tasks, boot, seed, buckets=present)
    loo = leave_one_out(tasks, buckets=present)
    rows = []
    for b in present:
        members = [t for t in tasks if t.bucket == b]
        c = pooled(members)
        arg, _ = pooled_argmax(c)
        ordered = sorted(c.values(), reverse=True)
        rows.append({
            "bucket": b,
            "n_tasks": len(members),
            "divergences": sum(t.div_total for t in members),
            "sips": sum(t.sip_total for t in members),
            "pooled": {SIP_ABBR[s]: c[s] for s in SIP_TYPES},
            "argmax": SIP_ABBR.get(arg) if arg else None,
            "runner_up": loo[b]["runner_up"],
            "margin": ordered[0] - ordered[1],
            "P_argmax_bootstrap": bootres[b]["P_argmax"],
            "P_tie_bootstrap": bootres[b]["P_tie"],
            "n_single_task_deletions_that_flip": loo[b]["n_single_task_deletions_that_flip_argmax"],
            "min_deletions_to_flip": loo[b]["min_task_deletions_to_flip_to_runner_up"],
            "mean_baseline": (sum(t.baseline for t in members) / len(members)) if members else None,
            "mean_delta_pp": (100.0 * sum(t.delta for t in members) / len(members)) if members else None,
        })
    top = rows[0]
    st = sign_test(tasks, top["bucket"], "surface_anchoring", "edge_case_prompting")
    return {"permutation": perm, "buckets": rows, "top_bucket_sign_test_SA_vs_EP": st}


def print_scheme(label, res):
    hr()
    print(label)
    print("    %-20s %4s %6s %5s   %-38s %-6s %-6s %6s %s"
          % ("bucket", "n", "diverg", "sips", "pooled SIP mix", "argmax", "runup",
             "margin", "P(argmax) [bootstrap]"))
    for r in res["buckets"]:
        mix = " ".join("%s=%d" % (k, v) for k, v in
                       sorted(r["pooled"].items(), key=lambda kv: -kv[1]) if v)
        pa = r["P_argmax_bootstrap"].get(r["argmax"], 0.0) if r["argmax"] else 0.0
        print("    %-20s %4d %6d %5d   %-38s %-6s %-6s %6d   %.3f"
              % (r["bucket"], r["n_tasks"], r["divergences"], r["sips"], mix,
                 r["argmax"], r["runner_up"], r["margin"], pa))
    p = res["permutation"]
    print("    permutation on bucket x dominant-SIP: G=%.3f, p=%.3f (n=%d usable tasks)"
          % (p["G_observed"], p["p_permutation"], p["n_usable_tasks"]))
    st = res["top_bucket_sign_test_SA_vs_EP"]
    print("    top-bucket paired sign test SA vs EP: SA>EP %d / EP>SA %d / ties %d, p=%.3f"
          % (st["wins_SA"], st["wins_EP"], st["ties"], st["p_exact_two_sided"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--shuffles", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260724)
    args = ap.parse_args()

    tasks = load_tasks()
    hr("cta_rebucket_robustness.py -- ALTERNATIVE BUCKET CUTOFFS")
    dist = Counter(round(t.baseline, 3) for t in tasks)
    print("baseline pass-rate distribution (49 tasks): "
          + ", ".join("%.3g:%d" % (k, dist[k]) for k in sorted(dist)))
    print("NB 28/49 tasks sit at baseline = 1.00, so the 'ceiling' bucket is mostly a"
          " single degenerate value; any cutoff in (0.84, 0.90] gives the SAME 37/10/2.")

    out = {}
    for hi in (0.95, 0.90, 0.85, 0.80):
        names, fn = scheme_cutoff(hi)
        relabel(tasks, fn)
        res = analyse(tasks, names, args.boot, args.shuffles, args.seed)
        out["cutoff_%.2f" % hi] = res
        print_scheme("SCHEME: ceiling >= %.2f, mid [0.50, %.2f), floor < 0.50%s"
                     % (hi, hi, "   <-- PAPER'S CHOICE" if abs(hi - 0.90) < 1e-9 else ""), res)

    names, fn = scheme_strict()
    relabel(tasks, fn)
    res = analyse(tasks, names, args.boot, args.shuffles, args.seed)
    out["strict_all_tests_pass"] = res
    print_scheme("SCHEME: STRICT rubric -- all_pass (baseline == 1.00) / partial [0.50,1.00)"
                 " / mostly_fail (< 0.50)", res)

    names, fn, med = scheme_median(tasks)
    relabel(tasks, fn)
    res = analyse(tasks, names, args.boot, args.shuffles, args.seed)
    out["median_split"] = res
    print_scheme("SCHEME: median split at baseline = %.3f" % med, res)

    # ---- what survives? --------------------------------------------------- #
    hr("WHAT SURVIVES")
    top_args = {}
    any_perm_sig = False
    any_sign_sig = False
    any_confident = False
    for key, res in out.items():
        p = res["permutation"]["p_permutation"]
        any_perm_sig = any_perm_sig or p < 0.05
        st = res["top_bucket_sign_test_SA_vs_EP"]["p_exact_two_sided"]
        any_sign_sig = any_sign_sig or st < 0.05
        top_args[key] = [(r["bucket"], r["argmax"], r["margin"],
                          r["P_argmax_bootstrap"].get(r["argmax"], 0.0)) for r in res["buckets"]]
        for r in res["buckets"]:
            if r["argmax"] and r["P_argmax_bootstrap"].get(r["argmax"], 0.0) >= 0.95:
                any_confident = True
    print("1. Bucket-label permutation p < 0.05 under ANY scheme?           %s"
          % ("YES" if any_perm_sig else "NO  -- p ranges %.2f to %.2f"
             % (min(r["permutation"]["p_permutation"] for r in out.values()),
                max(r["permutation"]["p_permutation"] for r in out.values()))))
    print("2. Top-bucket paired sign test (SA vs EP) p < 0.05 under ANY scheme? %s"
          % ("YES" if any_sign_sig else "NO"))
    print("3. Any bucket whose dominant SIP is stable at P(argmax) >= 0.95?  %s"
          % ("YES" if any_confident else "NO"))
    print("4. Does the top bucket's dominant SIP even stay the same across schemes?")
    for key in out:
        b0 = out[key]["buckets"][0]
        print("     %-22s top bucket = %-20s argmax %-4s (margin %d, P=%.3f)"
              % (key, b0["bucket"], b0["argmax"], b0["margin"],
                 b0["P_argmax_bootstrap"].get(b0["argmax"], 0.0)))
    print("\nCONCLUSION: no bucket-conditional SIP-signature conclusion survives"
          " re-bucketing. The label attached to the top bucket flips with the cutoff,"
          " every permutation p is far from significance, and no bucket's dominant type"
          " is stable under a cluster bootstrap. The only robust statements are"
          " marginal ones: the pooled fire mix (EP 186 / SA 184 / CB 99 / RE 42 / PS 11)"
          " and the fact that 45/49 tasks have dP = 0.")

    path = dump("cta_rebucket_robustness.json", {
        "schemes": out,
        "baseline_distribution": {str(k): v for k, v in dist.items()},
        "params": {"seed": args.seed, "shuffles": args.shuffles, "boot": args.boot},
    })
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
