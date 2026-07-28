#!/usr/bin/env python3
"""
cta_common.py -- shared loaders + self-checks for the CTA rebuttal analyses.

NO-API, pure stdlib (numpy optional and unused). Everything is derived from two
canonical artifacts that ship with the repo:

  CTA/cta_output/cta_combined_results.json   (49 tasks, per-task aggregates)
  CTA/config/cta_task_metadata.json          (49 tasks, pass rates / tokens)

Raw traces (claude_process/**/*.jsonl) are gitignored and ABSENT from this
checkout, so nothing here can look at per-divergence records. Any quantity that
needs a per-divergence join is reported as a bound or as NOT-COMPUTABLE (see
cta_destructive_accounting.py).

The module refuses to import cleanly if the canonical totals do not reproduce:
696 divergences / 522 SIP fires / bucket sizes 37-10-2. That is deliberate --
a silent drift in the inputs must not silently change a rebuttal number.
"""

import json
import math
import os
import random
from collections import Counter

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REBUTTAL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTA_ROOT = os.path.join(REBUTTAL_ROOT, "CTA")
COMBINED_PATH = os.path.join(CTA_ROOT, "cta_output", "cta_combined_results.json")
METADATA_PATH = os.path.join(CTA_ROOT, "config", "cta_task_metadata.json")
SKILLS_DIR = os.path.join(CTA_ROOT, "skills")
RESULTS_DIR = os.path.join(REBUTTAL_ROOT, "results")

# --------------------------------------------------------------------------- #
# Vocabulary (order is load-bearing: the 5-vector is always in this order)
# --------------------------------------------------------------------------- #

SIP_TYPES = [
    "edge_case_prompting",
    "surface_anchoring",
    "concept_bleed",
    "redundant_exploration",
    "procedural_scaffolding",
]
SIP_ABBR = {
    "edge_case_prompting": "EP",
    "surface_anchoring": "SA",
    "concept_bleed": "CB",
    "redundant_exploration": "RE",
    "procedural_scaffolding": "PS",
}
# Category as emitted by the detector itself (module4_detector.py).
SIP_CATEGORY = {
    "edge_case_prompting": "constructive",
    "procedural_scaffolding": "constructive",
    "redundant_exploration": "neutral",
    "surface_anchoring": "destructive",
    "concept_bleed": "destructive",
}
DESTRUCTIVE_SIPS = [t for t in SIP_TYPES if SIP_CATEGORY[t] == "destructive"]

DIV_TYPES = ["target_mismatch", "unilateral_action", "content_mismatch"]

# Per-type emission thresholds actually used by the code
# (module4_detector.py lines ~112-122). The paper text says a uniform 0.50;
# the advisor decision is that the CODE is authoritative.
CODE_THRESHOLDS = {
    "surface_anchoring": 0.40,
    "redundant_exploration": 0.45,
    "edge_case_prompting": 0.50,
    "procedural_scaffolding": 0.55,
    "concept_bleed": 0.55,
}

# Default bucketing used in the paper.
DEFAULT_CUTS = (0.90, 0.50)  # ceiling >= 0.90 ; mid in [0.50, 0.90) ; floor < 0.50
BUCKETS = ["ceiling", "mid", "floor"]


class SelfCheckError(AssertionError):
    """Raised loudly when the canonical totals fail to reproduce."""


# --------------------------------------------------------------------------- #
# Task record
# --------------------------------------------------------------------------- #


class Task:
    __slots__ = (
        "task_id",
        "sip",
        "sip_total",
        "div",
        "div_total",
        "avg_skill_similarity",
        "baseline",
        "with_rate",
        "without_rate",
        "delta",
        "with_tokens",
        "without_tokens",
        "token_ratio",
        "bucket",
    )

    def __init__(self, task_id):
        self.task_id = task_id

    # -- SIP helpers -------------------------------------------------------- #
    def vec(self):
        """5-vector of SIP counts in SIP_TYPES order."""
        return [self.sip[t] for t in SIP_TYPES]

    def argmax_sip(self):
        """Unique dominant SIP type, or None if zero SIPs or a tie."""
        if self.sip_total == 0:
            return None
        m = max(self.sip.values())
        winners = [t for t in SIP_TYPES if self.sip[t] == m]
        return winners[0] if len(winners) == 1 else None

    def destructive_fires(self):
        return sum(self.sip[t] for t in DESTRUCTIVE_SIPS)

    def __repr__(self):
        return "Task(%s, base=%.2f, d=%d, sip=%d, %s)" % (
            self.task_id,
            self.baseline,
            self.div_total,
            self.sip_total,
            self.bucket,
        )


def bucket_of(baseline, cuts=DEFAULT_CUTS):
    hi, lo = cuts
    if baseline >= hi:
        return "ceiling"
    if baseline >= lo:
        return "mid"
    return "floor"


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def load_tasks(cuts=DEFAULT_CUTS, check=True):
    with open(COMBINED_PATH) as fh:
        combined = json.load(fh)
    with open(METADATA_PATH) as fh:
        meta = json.load(fh)

    if set(combined) != set(meta):
        raise SelfCheckError(
            "task id sets differ: combined=%d meta=%d symdiff=%s"
            % (len(combined), len(meta), sorted(set(combined) ^ set(meta))[:10])
        )

    tasks = []
    for tid in sorted(combined):
        rec = combined[tid]
        md = meta[tid]
        t = Task(tid)

        sip_stats = rec["modules"]["sip_detection"]["sip_statistics"]
        t.sip = {k: int(sip_stats["by_type"].get(k, {}).get("count", 0)) for k in SIP_TYPES}
        t.sip_total = int(sip_stats["total_sips"])
        if sum(t.sip.values()) != t.sip_total:
            raise SelfCheckError(
                "%s: by_type SIP counts (%d) != total_sips (%d)"
                % (tid, sum(t.sip.values()), t.sip_total)
            )

        dstats = rec["modules"]["alignment"]["divergence_statistics"]
        t.div = {k: int(dstats["by_type"].get(k, 0)) for k in DIV_TYPES}
        t.div_total = int(dstats["total_divergences"])
        if sum(t.div.values()) != t.div_total:
            raise SelfCheckError(
                "%s: divergence by_type (%d) != total (%d)"
                % (tid, sum(t.div.values()), t.div_total)
            )
        t.avg_skill_similarity = dstats.get("avg_skill_similarity")

        t.baseline = float(md["baseline_pass_rate"])
        t.with_rate = float(md["with_skill_pass_rate"])
        t.without_rate = float(md["without_skill_pass_rate"])
        t.delta = float(md["pass_rate_delta"])
        t.with_tokens = float(md["with_skill_avg_tokens"])
        t.without_tokens = float(md["without_skill_avg_tokens"])
        t.token_ratio = float(md["token_overhead_ratio"])
        t.bucket = bucket_of(t.baseline, cuts)
        tasks.append(t)

    if check:
        self_check(tasks)
    return tasks


# --------------------------------------------------------------------------- #
# Self-check -- fails loudly
# --------------------------------------------------------------------------- #

EXPECT = {
    "n_tasks": 49,
    "divergences": 696,
    "div_by_type": {"target_mismatch": 521, "unilateral_action": 112, "content_mismatch": 63},
    "sips": 522,
    "sip_by_type": {
        "edge_case_prompting": 186,
        "surface_anchoring": 184,
        "concept_bleed": 99,
        "redundant_exploration": 42,
        "procedural_scaffolding": 11,
    },
    "bucket_sizes": {"ceiling": 37, "mid": 10, "floor": 2},
    "bucket_sips": {"ceiling": 415, "mid": 92, "floor": 15},
    "zero_delta_tasks": 45,
}


def self_check(tasks):
    errs = []

    def eq(label, got, want):
        if got != want:
            errs.append("%s: got %r, expected %r" % (label, got, want))

    eq("n_tasks", len(tasks), EXPECT["n_tasks"])

    dv = Counter()
    sp = Counter()
    for t in tasks:
        for k, v in t.div.items():
            dv[k] += v
        for k, v in t.sip.items():
            sp[k] += v
    eq("total_divergences", sum(dv.values()), EXPECT["divergences"])
    eq("divergences_by_type", dict(dv), EXPECT["div_by_type"])
    eq("total_sips", sum(sp.values()), EXPECT["sips"])
    eq("sips_by_type", {k: sp[k] for k in SIP_TYPES}, EXPECT["sip_by_type"])

    bs = Counter(t.bucket for t in tasks)
    eq("bucket_sizes", {b: bs[b] for b in BUCKETS}, EXPECT["bucket_sizes"])
    bsip = Counter()
    for t in tasks:
        bsip[t.bucket] += t.sip_total
    eq("bucket_sips", {b: bsip[b] for b in BUCKETS}, EXPECT["bucket_sips"])

    eq("zero_delta_tasks", sum(1 for t in tasks if abs(t.delta) < 1e-12),
       EXPECT["zero_delta_tasks"])

    mean_base = sum(t.baseline for t in tasks) / len(tasks)
    if abs(mean_base - 0.889) > 0.002:
        errs.append("mean baseline pass rate: got %.6f, expected ~0.889" % mean_base)
    mean_dp_pp = 100.0 * sum(t.delta for t in tasks) / len(tasks)
    if abs(mean_dp_pp - 0.34) > 0.01:
        errs.append("mean dP: got %.4f pp, expected ~+0.34 pp" % mean_dp_pp)
    med = median([t.delta for t in tasks])
    if abs(med) > 1e-12:
        errs.append("median dP: got %r, expected 0" % med)

    # Every SKILL.md must be present (49 skills, one instance each).
    missing = [t.task_id for t in tasks
               if not os.path.isfile(os.path.join(SKILLS_DIR, t.task_id, "SKILL.md"))]
    if missing:
        errs.append("missing SKILL.md for: %s" % missing)

    if errs:
        raise SelfCheckError(
            "CTA SELF-CHECK FAILED (%d problem(s)):\n  - %s" % (len(errs), "\n  - ".join(errs))
        )
    return True


# --------------------------------------------------------------------------- #
# Small stats helpers (stdlib only)
# --------------------------------------------------------------------------- #


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def g_statistic(table):
    """Likelihood-ratio G for a rows x cols contingency table (list of lists)."""
    n = sum(sum(r) for r in table)
    if n == 0:
        return 0.0
    rows = [sum(r) for r in table]
    cols = [sum(table[i][j] for i in range(len(table))) for j in range(len(table[0]))]
    g = 0.0
    for i, r in enumerate(table):
        for j, o in enumerate(r):
            if o <= 0:
                continue
            e = rows[i] * cols[j] / n
            if e > 0:
                g += o * math.log(o / e)
    return 2.0 * g


def comb(n, k):
    """Binomial coefficient (math.comb is 3.8+; this keeps us 3.6-compatible)."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num = 1
    den = 1
    for i in range(k):
        num *= (n - i)
        den *= (i + 1)
    return num // den


def binom_two_sided(k, n):
    """Exact two-sided binomial test against p=0.5 (sign test)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2.0 ** n))


def rng(seed):
    return random.Random(seed)


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def dump(name, obj):
    ensure_results_dir()
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    return path


def hr(title=""):
    if title:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
    else:
        print("-" * 78)


# --------------------------------------------------------------------------- #


def main():
    tasks = load_tasks()
    hr("cta_common.py -- SELF-CHECK PASSED")
    print("tasks                 : %d (one instance per skill, r=1)" % len(tasks))
    dv = Counter()
    sp = Counter()
    for t in tasks:
        for k, v in t.div.items():
            dv[k] += v
        for k, v in t.sip.items():
            sp[k] += v
    print("divergences           : %d  (%s)" % (
        sum(dv.values()), ", ".join("%s=%d" % (k, dv[k]) for k in DIV_TYPES)))
    print("SIP fires             : %d  (%s)" % (
        sum(sp.values()), ", ".join("%s=%d" % (SIP_ABBR[k], sp[k]) for k in SIP_TYPES)))
    bs = Counter(t.bucket for t in tasks)
    bsip = Counter()
    bdiv = Counter()
    for t in tasks:
        bsip[t.bucket] += t.sip_total
        bdiv[t.bucket] += t.div_total
    for b in BUCKETS:
        print("bucket %-8s       : n=%2d  divergences=%3d  sips=%3d" %
              (b, bs[b], bdiv[b], bsip[b]))
    print("mean baseline         : %.4f" % (sum(t.baseline for t in tasks) / len(tasks)))
    print("mean dP               : %+.4f pp   median dP: %.1f   dP==0: %d/%d" % (
        100.0 * sum(t.delta for t in tasks) / len(tasks),
        median([t.delta for t in tasks]),
        sum(1 for t in tasks if abs(t.delta) < 1e-12), len(tasks)))
    print("tasks w/ >=1 SIP      : %d ; with a UNIQUE dominant SIP: %d ; ties: %d" % (
        sum(1 for t in tasks if t.sip_total > 0),
        sum(1 for t in tasks if t.argmax_sip() is not None),
        sum(1 for t in tasks if t.sip_total > 0 and t.argmax_sip() is None)))
    dump("cta_common_selfcheck.json", {
        "self_check": "PASSED",
        "n_tasks": len(tasks),
        "divergences_total": sum(dv.values()),
        "divergences_by_type": dict(dv),
        "sips_total": sum(sp.values()),
        "sips_by_type": {k: sp[k] for k in SIP_TYPES},
        "bucket_sizes": {b: bs[b] for b in BUCKETS},
        "bucket_sips": {b: bsip[b] for b in BUCKETS},
        "bucket_divergences": {b: bdiv[b] for b in BUCKETS},
        "mean_baseline": sum(t.baseline for t in tasks) / len(tasks),
        "mean_delta_pp": 100.0 * sum(t.delta for t in tasks) / len(tasks),
        "median_delta": median([t.delta for t in tasks]),
        "n_zero_delta": sum(1 for t in tasks if abs(t.delta) < 1e-12),
        "code_thresholds_authoritative": CODE_THRESHOLDS,
        "paper_text_claims_uniform_threshold": 0.50,
        "per_task": [
            {"task_id": t.task_id, "baseline": t.baseline, "delta": t.delta,
             "bucket": t.bucket, "divergences": t.div_total, "div_by_type": t.div,
             "sips": t.sip_total, "sip_by_type": t.sip,
             "dominant_sip": t.argmax_sip()}
            for t in tasks
        ],
    })
    print("\nwrote %s" % os.path.join(RESULTS_DIR, "cta_common_selfcheck.json"))


if __name__ == "__main__":
    main()
