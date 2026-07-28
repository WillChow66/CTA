#!/usr/bin/env python3
"""
cta_destructive_accounting.py -- how much of CTA's "destructive" headline is a
counting-unit artifact, and how much of it is forced by the detector's geometry?

Two problems with the paper's destructive share:

(A) UNIT. The 522 SIPs are (divergence, type) pairs -- module4_detector.py scores
    all five SIP types on EVERY divergence and emits each type that clears its
    own threshold, so one divergence can produce up to 5 fires. "54.2% of SIPs
    are destructive" is therefore a share of FIRES, not of divergent behaviours.
    At the DIVERGENCE level only bounds are computable from the aggregates.

(B) FORCED FIRES. Edge-case Prompting auto-fires on every unilateral_action
    divergence: module4_detector.py ~line 326 adds a flat +0.55 base for
    is_unilateral, and EP's emission threshold is 0.50, so the base alone clears
    it with no edge-case content required. Those EP fires are structural, not
    evidence about the skill document.

This script quantifies both from the shipped aggregates and states plainly what
CANNOT be computed without the per-divergence records (claude_process/**/*.jsonl,
which are gitignored and absent from this checkout).

Pure stdlib. Usage:  python3 cta_destructive_accounting.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cta_common import (  # noqa: E402
    CODE_THRESHOLDS, DESTRUCTIVE_SIPS, DIV_TYPES, SIP_ABBR, SIP_CATEGORY, SIP_TYPES,
    dump, hr, load_tasks,
)


def main():
    tasks = load_tasks()

    total_div = sum(t.div_total for t in tasks)
    total_fires = sum(t.sip_total for t in tasks)
    by_type = {s: sum(t.sip[s] for t in tasks) for s in SIP_TYPES}
    by_cat = {}
    for s in SIP_TYPES:
        by_cat[SIP_CATEGORY[s]] = by_cat.get(SIP_CATEGORY[s], 0) + by_type[s]
    div_by_type = {d: sum(t.div[d] for t in tasks) for d in DIV_TYPES}

    hr("cta_destructive_accounting.py -- FIRE-LEVEL vs DIVERGENCE-LEVEL")
    print("divergences : %d  (%s)" % (
        total_div, ", ".join("%s=%d" % (d, div_by_type[d]) for d in DIV_TYPES)))
    print("SIP fires   : %d  (%s)" % (
        total_fires, ", ".join("%s=%d" % (SIP_ABBR[s], by_type[s]) for s in SIP_TYPES)))
    print("fires per divergence (mean) : %.3f   (a divergence can carry 0..5 fires)"
          % (total_fires / float(total_div)))

    # ---- (1) fire-level shares ------------------------------------------- #
    hr()
    print("(1) FIRE-LEVEL shares -- the number the paper reports")
    for cat in ("destructive", "constructive", "neutral"):
        members = [SIP_ABBR[s] for s in SIP_TYPES if SIP_CATEGORY[s] == cat]
        print("    %-13s %3d / %d = %5.1f%%   (%s)"
              % (cat, by_cat[cat], total_fires, 100.0 * by_cat[cat] / total_fires,
                 "+".join(members)))
    destructive_fires = by_cat["destructive"]

    # ---- (2) divergence-level bounds ------------------------------------- #
    # Each SIP type fires at most ONCE per divergence, so within a task:
    #   #divergences carrying >=1 destructive fire  >= max(SA_i, CB_i)   (max overlap)
    #                                               <= min(SA_i + CB_i, D_i)  (disjoint)
    lo_d = hi_d = 0
    lo_any = hi_any = 0
    over_capacity = []       # tasks where fires > divergences => co-firing is forced
    per_task = []
    for t in tasks:
        sa, cb = t.sip["surface_anchoring"], t.sip["concept_bleed"]
        l = max(sa, cb)
        h = min(sa + cb, t.div_total)
        lo_d += l
        hi_d += h
        la = max(t.sip.values()) if t.sip_total else 0
        ha = min(t.sip_total, t.div_total)
        lo_any += la
        hi_any += ha
        if t.sip_total > t.div_total:
            over_capacity.append((t.task_id, t.sip_total, t.div_total))
        per_task.append({
            "task_id": t.task_id, "bucket": t.bucket, "divergences": t.div_total,
            "fires": t.sip_total, "SA": sa, "CB": cb,
            "destructive_divergences_lower": l, "destructive_divergences_upper": h,
            "unilateral": t.div["unilateral_action"], "EP": t.sip["edge_case_prompting"],
            "forced_EP": min(t.div["unilateral_action"], t.sip["edge_case_prompting"]),
            "unforced_EP": t.sip["edge_case_prompting"] - t.div["unilateral_action"],
        })

    hr()
    print("(2) DIVERGENCE-LEVEL bounds -- what the aggregates actually license")
    print("    A divergence counts as 'destructive' if SA or CB fired on it.")
    print("    lower bound (SA and CB maximally co-fire on the same divergences):"
          " %d / %d = %.1f%%" % (lo_d, total_div, 100.0 * lo_d / total_div))
    print("    upper bound (SA and CB never co-fire, capped by each task's"
          " divergence count): %d / %d = %.1f%%" % (hi_d, total_div, 100.0 * hi_d / total_div))
    print("    => destructive share at the DIVERGENCE level is in [%.1f%%, %.1f%%],"
          " not %.1f%%." % (100.0 * lo_d / total_div, 100.0 * hi_d / total_div,
                            100.0 * destructive_fires / total_fires))
    print("    divergences carrying ANY SIP: in [%d, %d] = [%.1f%%, %.1f%%]"
          % (lo_any, hi_any, 100.0 * lo_any / total_div, 100.0 * hi_any / total_div))
    print("    tasks where fires > divergences (co-firing is arithmetically forced): %d"
          % len(over_capacity))
    for tid, f, d in sorted(over_capacity, key=lambda x: -(x[1] - x[2]))[:8]:
        print("        %-34s %3d fires over %3d divergences" % (tid, f, d))

    # ---- (3) forced-EP decomposition ------------------------------------- #
    uni = div_by_type["unilateral_action"]
    ep = by_type["edge_case_prompting"]
    violations = [p for p in per_task if p["EP"] < p["unilateral"]]
    all_forced = [p for p in per_task if p["unilateral"] > 0 and p["unforced_EP"] == 0]
    re_fires = by_type["redundant_exploration"]

    hr()
    print("(3) FORCED-EP DECOMPOSITION")
    print("    EP threshold = %.2f ; is_unilateral adds a flat +0.55 base"
          " (module4_detector.py ~L326) => every unilateral_action divergence"
          " emits EP with no skill content required."
          % CODE_THRESHOLDS["edge_case_prompting"])
    print("    consistency check: EP_i >= unilateral_i for every task? %s (%d violations)"
          % ("YES" if not violations else "NO", len(violations)))
    print("    unilateral_action divergences        : %d" % uni)
    print("    EP fires                             : %d" % ep)
    print("    -> forced by the unilateral base     : %d = %.1f%% of EP fires,"
          " %.1f%% of all %d fires" % (uni, 100.0 * uni / ep, 100.0 * uni / total_fires,
                                       total_fires))
    print("    -> earned by keyword/write evidence  : %d = %.1f%% of EP fires"
          % (ep - uni, 100.0 * (ep - uni) / ep))
    print("    tasks whose EP fires are 100%% forced : %d / %d with any unilateral divergence"
          % (len(all_forced), sum(1 for p in per_task if p["unilateral"] > 0)))
    print("    RE scoring uses only trace-vs-trace features (intent/content similarity,"
          " event-count ratio, target Jaccard); no skill-document term.")
    print("    => fires that require NO skill-document content at all:"
          " RE %d + forced EP %d = %d = %.1f%% of all fires."
          % (re_fires, uni, re_fires + uni, 100.0 * (re_fires + uni) / total_fires))
    print("    Constructive share net of forced EP: (EP %d - forced %d + PS %d) = %d"
          " = %.1f%% of fires (paper's constructive share: %.1f%%)."
          % (ep, uni, by_type["procedural_scaffolding"],
             ep - uni + by_type["procedural_scaffolding"],
             100.0 * (ep - uni + by_type["procedural_scaffolding"]) / total_fires,
             100.0 * by_cat["constructive"] / total_fires))

    # ---- (4) thresholds --------------------------------------------------- #
    hr()
    print("(4) THRESHOLDS -- code vs paper text")
    print("    code (authoritative, module4_detector.py L112-122): "
          + ", ".join("%s=%.2f" % (SIP_ABBR[s], CODE_THRESHOLDS[s]) for s in SIP_TYPES))
    print("    paper text says a uniform 0.50; config min_detection_confidence is dead code.")
    print("    Consequence for this accounting: SA fires at 0.40 and CB at 0.55, i.e. the two"
          " DESTRUCTIVE types sit on opposite sides of the value the paper quotes, so the"
          " destructive share is threshold-sensitive in a way the text does not disclose."
          " Re-scoring at a true uniform 0.50 needs the raw per-fire scores -> NOT COMPUTABLE"
          " here.")

    # ---- (5) not computable ---------------------------------------------- #
    not_computable = [
        "Exact divergence-level destructive share. Needs the (divergence_id, sip_type) "
        "join; the aggregates only carry per-task per-type counts. Bounds only: "
        "[%d, %d] of %d = [%.1f%%, %.1f%%]." % (lo_d, hi_d, total_div,
                                                100.0 * lo_d / total_div,
                                                100.0 * hi_d / total_div),
        "The co-fire matrix (which SIP types land on the SAME divergence). This is what "
        "decides whether 'destructive' and 'constructive' fires are describing the same "
        "behaviour twice.",
        "Whether the 112 forced-EP fires sit on divergences that ALSO carry SA or CB. If "
        "they largely do, the constructive/destructive split is double-counting one event.",
        "Per-fire confidence distribution and how many fires sit within epsilon of their "
        "threshold. Only a per-task per-type avg_confidence is stored, so threshold "
        "sensitivity (e.g. re-scoring every type at a uniform 0.50) cannot be simulated.",
        "Phase attribution of SIP fires. divergence_statistics carries by_phase, but "
        "sip_statistics does not, so 'SIPs concentrate in phase X' is unverifiable here.",
        "Any per-divergence null baseline (e.g. shuffling skill documents across tasks to "
        "get a false-positive rate for SA/CB). Requires re-running Modules 3-4 on the raw "
        "traces.",
        "Anything about run-to-run variance: r=1, one instance per skill, so there is no "
        "within-task replicate in this data at all.",
    ]
    hr()
    print("(5) NOT COMPUTABLE FROM THE SHIPPED AGGREGATES")
    print("    Raw traces (claude_process/**/*.jsonl) are gitignored and absent here;")
    print("    cta_output holds only per-task aggregates. The following need a re-run")
    print("    on the machine that has the traces:")
    for i, s in enumerate(not_computable, 1):
        print("    %d. %s" % (i, s))

    hr("VERDICT")
    print("The paper's '%.1f%% of SIPs are destructive' is a share of FIRES."
          % (100.0 * destructive_fires / total_fires))
    print("At the level of distinct divergent behaviours the same data supports only"
          " %.1f%%-%.1f%%." % (100.0 * lo_d / total_div, 100.0 * hi_d / total_div))
    print("Separately, %d of %d EP fires (%.1f%%) are emitted by a structural rule that"
          " never inspects the skill document, so the constructive column is inflated:"
          " %.1f%% of ALL fires are forced EP." % (uni, ep, 100.0 * uni / ep,
                                                   100.0 * uni / total_fires))
    print("RECOMMENDATION: report the destructive share at the divergence level with the"
          " bound, and report forced-EP separately from earned EP.")

    path = dump("cta_destructive_accounting.json", {
        "totals": {
            "divergences": total_div, "divergences_by_type": div_by_type,
            "fires": total_fires, "fires_by_type": by_type, "fires_by_category": by_cat,
            "fires_per_divergence": total_fires / float(total_div),
        },
        "fire_level": {
            "destructive": destructive_fires,
            "destructive_share": destructive_fires / float(total_fires),
            "destructive_types": [SIP_ABBR[s] for s in DESTRUCTIVE_SIPS],
        },
        "divergence_level_bounds": {
            "destructive_lower": lo_d, "destructive_upper": hi_d,
            "destructive_share_lower": lo_d / float(total_div),
            "destructive_share_upper": hi_d / float(total_div),
            "any_sip_lower": lo_any, "any_sip_upper": hi_any,
            "tasks_with_fires_exceeding_divergences": [
                {"task_id": a, "fires": b, "divergences": c} for a, b, c in over_capacity],
        },
        "forced_ep": {
            "unilateral_action_divergences": uni, "ep_fires": ep,
            "forced_ep": uni, "earned_ep": ep - uni,
            "forced_share_of_ep": uni / float(ep),
            "forced_share_of_all_fires": uni / float(total_fires),
            "re_fires": re_fires,
            "fires_needing_no_skill_document": re_fires + uni,
            "fires_needing_no_skill_document_share": (re_fires + uni) / float(total_fires),
            "ep_lt_unilateral_violations": [p["task_id"] for p in violations],
            "tasks_with_all_ep_forced": [p["task_id"] for p in all_forced],
        },
        "thresholds_code_authoritative": CODE_THRESHOLDS,
        "not_computable_without_per_divergence_records": not_computable,
        "per_task": per_task,
    })
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
