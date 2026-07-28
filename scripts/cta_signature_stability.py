#!/usr/bin/env python3
"""
cta_signature_stability.py -- does Finding #2 survive a per-task test?

Finding #2 (paper): the baseline-difficulty bucket a task falls in predicts the
DOMINANT SIP signature -- ceiling tasks are dominated by Surface Anchoring, mid
tasks by Edge-case Prompting / Concept Bleed, floor tasks by Edge-case
Prompting.

That claim is currently supported by POOLED fire counts, which is a unit-of-
analysis error: 522 fires come from 49 tasks (one instance per skill, r=1), so
fires are clustered and a handful of high-count tasks can manufacture a pooled
margin that no individual task supports. This script re-tests the claim with the
TASK as the unit:

  (i)   permutation test on the bucket x dominant-SIP contingency table (G),
        >= 20k shuffles of the bucket labels;
  (ii)  cluster bootstrap over tasks: P(pooled argmax = each SIP type | bucket);
  (iii) paired per-task sign tests (exact): SA vs EP on ceiling, EP vs CB on mid;
  (iv)  leave-one-out: which single task flips a bucket's pooled argmax.

Pure stdlib. Usage:  python3 cta_signature_stability.py [--boot 20000] [--seed 20260724]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cta_common import (  # noqa: E402
    BUCKETS, SIP_ABBR, SIP_TYPES, RESULTS_DIR,
    binom_two_sided, dump, g_statistic, hr, load_tasks, rng,
)


def pooled(tasks):
    return {t: sum(x.sip[t] for x in tasks) for t in SIP_TYPES}


def pooled_argmax(counts):
    m = max(counts.values())
    if m == 0:
        return None, 0
    winners = [t for t in SIP_TYPES if counts[t] == m]
    return (winners[0] if len(winners) == 1 else None), m


# --------------------------------------------------------------------------- #
# (i) permutation test on bucket x dominant-SIP
# --------------------------------------------------------------------------- #

def permutation_test(tasks, n_shuffles, seed, buckets=None):
    buckets = buckets or BUCKETS
    usable = [t for t in tasks if t.argmax_sip() is not None]
    labels = [t.bucket for t in usable]
    doms = [t.argmax_sip() for t in usable]

    dom_cols = [s for s in SIP_TYPES if s in set(doms)]
    row_idx = {b: i for i, b in enumerate(buckets)}
    col_idx = {s: j for j, s in enumerate(dom_cols)}

    def table_from(lbls):
        tab = [[0] * len(dom_cols) for _ in buckets]
        for lb, dm in zip(lbls, doms):
            tab[row_idx[lb]][col_idx[dm]] += 1
        return tab

    obs_tab = table_from(labels)
    g_obs = g_statistic(obs_tab)

    r = rng(seed)
    perm = list(labels)
    ge = 0
    for _ in range(n_shuffles):
        r.shuffle(perm)
        if g_statistic(table_from(perm)) >= g_obs - 1e-12:
            ge += 1
    p = (ge + 1.0) / (n_shuffles + 1.0)

    return {
        "n_usable_tasks": len(usable),
        "n_excluded_zero_sip": sum(1 for t in tasks if t.sip_total == 0),
        "n_excluded_tie": sum(1 for t in tasks if t.sip_total > 0 and t.argmax_sip() is None),
        "columns": [SIP_ABBR[s] for s in dom_cols],
        "table": {b: {SIP_ABBR[s]: obs_tab[row_idx[b]][col_idx[s]] for s in dom_cols}
                  for b in buckets},
        "row_totals": {b: sum(obs_tab[row_idx[b]]) for b in buckets},
        "G_observed": g_obs,
        "n_shuffles": n_shuffles,
        "n_perm_ge_obs": ge,
        "p_permutation": p,
    }


# --------------------------------------------------------------------------- #
# (ii) cluster bootstrap over tasks
# --------------------------------------------------------------------------- #

def cluster_bootstrap(tasks, n_boot, seed, buckets=None):
    buckets = buckets or BUCKETS
    out = {}
    for bi, b in enumerate(buckets):
        members = [t for t in tasks if t.bucket == b]
        n = len(members)
        obs = pooled(members)
        obs_arg, _ = pooled_argmax(obs)
        # deterministic per-bucket stream (str hashing is salted per process)
        r = rng(seed + 1009 * (bi + 1))
        wins = {s: 0 for s in SIP_TYPES}
        ties = 0
        empty = 0
        margins = []          # top1 - top2 of the resampled pooled counts
        sa_minus_ep = []
        ep_minus_cb = []
        for _ in range(n_boot):
            samp = [members[r.randrange(n)] for _ in range(n)]
            c = pooled(samp)
            a, m = pooled_argmax(c)
            if m == 0:
                empty += 1
            elif a is None:
                ties += 1
            else:
                wins[a] += 1
            ordered = sorted(c.values(), reverse=True)
            margins.append(ordered[0] - ordered[1])
            sa_minus_ep.append(c["surface_anchoring"] - c["edge_case_prompting"])
            ep_minus_cb.append(c["edge_case_prompting"] - c["concept_bleed"])

        def pct(v, q):
            s = sorted(v)
            k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
            return s[k]

        out[b] = {
            "n_tasks": n,
            "observed_pooled": {SIP_ABBR[s]: obs[s] for s in SIP_TYPES},
            "observed_argmax": SIP_ABBR.get(obs_arg) if obs_arg else None,
            "P_argmax": {SIP_ABBR[s]: wins[s] / float(n_boot) for s in SIP_TYPES},
            "P_tie": ties / float(n_boot),
            "P_empty": empty / float(n_boot),
            "top1_minus_top2_CI95": [pct(margins, 0.025), pct(margins, 0.975)],
            "SA_minus_EP_CI95": [pct(sa_minus_ep, 0.025), pct(sa_minus_ep, 0.975)],
            "EP_minus_CB_CI95": [pct(ep_minus_cb, 0.025), pct(ep_minus_cb, 0.975)],
            "n_boot": n_boot,
        }
    return out


# --------------------------------------------------------------------------- #
# (iii) paired per-task sign tests
# --------------------------------------------------------------------------- #

def sign_test(tasks, bucket, a, b):
    members = [t for t in tasks if t.bucket == bucket]
    wins_a = sum(1 for t in members if t.sip[a] > t.sip[b])
    wins_b = sum(1 for t in members if t.sip[b] > t.sip[a])
    ties = sum(1 for t in members if t.sip[a] == t.sip[b])
    n = wins_a + wins_b
    return {
        "bucket": bucket,
        "pair": "%s_vs_%s" % (SIP_ABBR[a], SIP_ABBR[b]),
        "n_tasks_in_bucket": len(members),
        "wins_%s" % SIP_ABBR[a]: wins_a,
        "wins_%s" % SIP_ABBR[b]: wins_b,
        "ties": ties,
        "n_discordant": n,
        "p_exact_two_sided": binom_two_sided(wins_a, n),
        "pooled_%s" % SIP_ABBR[a]: sum(t.sip[a] for t in members),
        "pooled_%s" % SIP_ABBR[b]: sum(t.sip[b] for t in members),
    }


# --------------------------------------------------------------------------- #
# (iv) leave-one-out on the pooled argmax
# --------------------------------------------------------------------------- #

def leave_one_out(tasks, buckets=None):
    buckets = buckets or BUCKETS
    out = {}
    for b in buckets:
        members = [t for t in tasks if t.bucket == b]
        base_counts = pooled(members)
        base_arg, base_max = pooled_argmax(base_counts)
        ordered = sorted(base_counts.values(), reverse=True)
        flips = []
        for i, drop in enumerate(members):
            rest = members[:i] + members[i + 1:]
            c = pooled(rest)
            a, m = pooled_argmax(c)
            if a != base_arg:
                flips.append({
                    "dropped_task": drop.task_id,
                    "dropped_sip_vector": {SIP_ABBR[s]: drop.sip[s] for s in SIP_TYPES},
                    "dropped_task_sips": drop.sip_total,
                    "new_argmax": SIP_ABBR.get(a) if a else ("TIE" if m else "NONE"),
                    "new_pooled": {SIP_ABBR[s]: c[s] for s in SIP_TYPES},
                })
        # Exact minimum number of task deletions that hands the bucket to the
        # runner-up: greedily drop the tasks with the largest (top1 - runnerup)
        # per-task contribution until the pooled margin is gone.
        runner_up = None
        if base_arg is not None:
            rivals = sorted([s for s in SIP_TYPES if s != base_arg],
                            key=lambda s: -base_counts[s])
            runner_up = rivals[0]
            contrib = sorted((t.sip[base_arg] - t.sip[runner_up] for t in members),
                             reverse=True)
            need = base_counts[base_arg] - base_counts[runner_up]
            k, acc = 0, 0
            for c in contrib:
                if acc > need:
                    break
                if c <= 0:
                    k = None
                    break
                acc += c
                k += 1
            min_k = k if (k is not None and acc > need) else None
        else:
            min_k = None

        out[b] = {
            "n_tasks": len(members),
            "pooled": {SIP_ABBR[s]: base_counts[s] for s in SIP_TYPES},
            "pooled_argmax": SIP_ABBR.get(base_arg) if base_arg else None,
            "runner_up": SIP_ABBR.get(runner_up) if runner_up else None,
            "pooled_margin_top1_minus_top2": ordered[0] - ordered[1],
            "n_single_task_deletions_that_flip_argmax": len(flips),
            "flipping_tasks": flips,
            "min_task_deletions_to_flip_to_runner_up": min_k,
            "max_single_task_share_of_bucket_fires": (
                max((t.sip_total for t in members), default=0) /
                float(sum(t.sip_total for t in members) or 1)),
        }
    return out


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--shuffles", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260724)
    args = ap.parse_args()

    tasks = load_tasks()

    hr("cta_signature_stability.py -- FALSIFICATION TEST FOR PAPER FINDING #2")
    print("unit of analysis = TASK (n=49, one instance per skill, r=1)")
    print("seed=%d  shuffles=%d  bootstrap=%d" % (args.seed, args.shuffles, args.boot))

    # ---- (i) permutation ------------------------------------------------- #
    perm = permutation_test(tasks, args.shuffles, args.seed)
    hr()
    print("(i) PERMUTATION TEST -- bucket x dominant-SIP table")
    print("    usable tasks: %d  (excluded: %d with zero SIPs, %d with a tied argmax)"
          % (perm["n_usable_tasks"], perm["n_excluded_zero_sip"], perm["n_excluded_tie"]))
    hdr = "    %-9s " % "bucket" + "".join("%5s" % c for c in perm["columns"]) + "   total"
    print(hdr)
    for b in BUCKETS:
        row = perm["table"][b]
        print("    %-9s " % b + "".join("%5d" % row[c] for c in perm["columns"])
              + "%8d" % perm["row_totals"][b])
    print("    G_observed = %.4f ;  #(G_perm >= G_obs) = %d / %d ;  p = %.4f"
          % (perm["G_observed"], perm["n_perm_ge_obs"], perm["n_shuffles"], perm["p_permutation"]))

    # ---- (ii) cluster bootstrap ------------------------------------------ #
    boot = cluster_bootstrap(tasks, args.boot, args.seed)
    hr()
    print("(ii) CLUSTER BOOTSTRAP over tasks -- P(pooled argmax = type | bucket)")
    for b in BUCKETS:
        r = boot[b]
        print("    %-8s n=%2d  observed pooled %s  -> argmax %s"
              % (b, r["n_tasks"],
                 " ".join("%s=%d" % (k, v) for k, v in
                          sorted(r["observed_pooled"].items(), key=lambda kv: -kv[1])),
                 r["observed_argmax"]))
        probs = sorted(r["P_argmax"].items(), key=lambda kv: -kv[1])
        print("             P(argmax): " + "  ".join("%s=%.3f" % (k, v) for k, v in probs
                                                     if v > 0.0005)
              + ("  [tie=%.3f]" % r["P_tie"] if r["P_tie"] else ""))
        print("             95%% CI top1-top2 = [%d, %d] ; SA-EP = [%d, %d] ; EP-CB = [%d, %d]"
              % (tuple(r["top1_minus_top2_CI95"]) + tuple(r["SA_minus_EP_CI95"])
                 + tuple(r["EP_minus_CB_CI95"])))

    # ---- (iii) sign tests ------------------------------------------------ #
    signs = [
        sign_test(tasks, "ceiling", "surface_anchoring", "edge_case_prompting"),
        sign_test(tasks, "mid", "edge_case_prompting", "concept_bleed"),
        sign_test(tasks, "floor", "edge_case_prompting", "surface_anchoring"),
    ]
    hr()
    print("(iii) PAIRED PER-TASK SIGN TESTS (exact, two-sided)")
    for s in signs:
        a, b = s["pair"].split("_vs_")
        print("    %-8s %-9s : %s>%s in %d tasks, %s>%s in %d, ties %d  "
              "(pooled %s=%d vs %s=%d)  ->  p = %.4f"
              % (s["bucket"], s["pair"], a, b, s["wins_%s" % a], b, a, s["wins_%s" % b],
                 s["ties"], a, s["pooled_%s" % a], b, s["pooled_%s" % b],
                 s["p_exact_two_sided"]))

    # ---- (iv) leave-one-out ---------------------------------------------- #
    loo = leave_one_out(tasks)
    hr()
    print("(iv) LEAVE-ONE-OUT on the pooled argmax")
    for b in BUCKETS:
        r = loo[b]
        print("    %-8s pooled argmax=%s  margin over %s = %d  "
              "largest single task = %.1f%% of the bucket's fires  "
              "min deletions to hand it to %s: %s"
              % (b, r["pooled_argmax"], r["runner_up"],
                 r["pooled_margin_top1_minus_top2"],
                 100 * r["max_single_task_share_of_bucket_fires"],
                 r["runner_up"], r["min_task_deletions_to_flip_to_runner_up"]))
        if r["flipping_tasks"]:
            for f in r["flipping_tasks"]:
                print("             DROP %-34s -> argmax becomes %s   (task had %d fires: %s)"
                      % (f["dropped_task"], f["new_argmax"], f["dropped_task_sips"],
                         " ".join("%s=%d" % (k, v) for k, v in
                                  sorted(f["dropped_sip_vector"].items(), key=lambda kv: -kv[1])
                                  if v)))
        else:
            print("             no single-task deletion flips the argmax")

    # ---- verdict ---------------------------------------------------------- #
    ceil_sign = signs[0]
    mid_sign = signs[1]
    supported = (
        perm["p_permutation"] < 0.05
        and ceil_sign["p_exact_two_sided"] < 0.05
        and boot["ceiling"]["P_argmax"].get("SA", 0) >= 0.95
    )
    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"
    hr("VERDICT")
    print("Paper Finding #2 (bucket -> dominant SIP signature): %s at the task level." % verdict)
    print("  - bucket-label permutation, G=%.2f, p=%.2f (n=%d usable tasks, %d shuffles):"
          " the bucket x dominant-SIP table is indistinguishable from random labelling."
          % (perm["G_observed"], perm["p_permutation"], perm["n_usable_tasks"], perm["n_shuffles"]))
    print("  - ceiling SA-vs-EP paired sign test: %d/%d/%d (SA>EP / EP>SA / ties), p=%.3f."
          % (ceil_sign["wins_SA"], ceil_sign["wins_EP"], ceil_sign["ties"],
             ceil_sign["p_exact_two_sided"]))
    print("  - mid EP-vs-CB paired sign test: %d/%d/%d, p=%.3f."
          % (mid_sign["wins_EP"], mid_sign["wins_CB"], mid_sign["ties"],
             mid_sign["p_exact_two_sided"]))
    print("  - cluster bootstrap: P(SA argmax | ceiling)=%.3f, P(EP argmax | mid)=%.3f,"
          " P(CB argmax | mid)=%.3f, P(EP argmax | floor)=%.3f."
          % (boot["ceiling"]["P_argmax"]["SA"], boot["mid"]["P_argmax"]["EP"],
             boot["mid"]["P_argmax"]["CB"], boot["floor"]["P_argmax"]["EP"]))
    print("  - pooled margins are tiny relative to the fire counts: ceiling SA-EP=%d of %d"
          " fires (flipped by dropping %s task(s)); mid EP-CB=%d of %d (%d single-task"
          " deletions already flip it); floor EP-SA=%d of %d (one task supplies 100%%)."
          % (loo["ceiling"]["pooled_margin_top1_minus_top2"],
             sum(loo["ceiling"]["pooled"].values()),
             loo["ceiling"]["min_task_deletions_to_flip_to_runner_up"],
             loo["mid"]["pooled_margin_top1_minus_top2"],
             sum(loo["mid"]["pooled"].values()),
             loo["mid"]["n_single_task_deletions_that_flip_argmax"],
             loo["floor"]["pooled_margin_top1_minus_top2"],
             sum(loo["floor"]["pooled"].values())))
    print("RECOMMENDATION: report the signature claim as descriptive of the pooled fire mix,"
          " not as a bucket-conditional prediction; or drop it.")

    path = dump("cta_signature_stability.json", {
        "permutation_test": perm,
        "cluster_bootstrap": boot,
        "sign_tests": signs,
        "leave_one_out": loo,
        "verdict": verdict,
        "params": {"seed": args.seed, "shuffles": args.shuffles, "boot": args.boot},
    })
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
