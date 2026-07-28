#!/usr/bin/env python3
"""
Corrected, review-passed numbers for the CTA rebuttal.

Supersedes the argmax-collapsed permutation test in cta_signature_stability.py.
Every claim here survived the independent statistical review; claims that did NOT
survive are printed in a WITHDRAWN section so they cannot be quoted by accident.

Run:  /usr/bin/python3.12 scripts/final_rebuttal_numbers.py
"""
import json, math, random, collections, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "CTA")
TYPES = ["procedural_scaffolding", "edge_case_prompting", "redundant_exploration",
         "surface_anchoring", "concept_bleed"]
SHORT = {"procedural_scaffolding": "PS", "edge_case_prompting": "EP",
         "redundant_exploration": "RE", "surface_anchoring": "SA", "concept_bleed": "CB"}


def load():
    comb = json.load(open(f"{C}/cta_output/cta_combined_results.json"))
    meta = json.load(open(f"{C}/config/cta_task_metadata.json"))
    mb = {t["skill_id"]: t for t in (meta if isinstance(meta, list) else list(meta.values()))}
    out = []
    for tid, rec in comb.items():
        m = mb.get(tid)
        if not m or m.get("baseline_pass_rate") is None:
            continue
        by = rec["modules"]["sip_detection"]["sip_statistics"].get("by_type", {})
        v = [by.get(t, {}).get("count", 0) for t in TYPES]
        ds = rec["modules"]["alignment"]["divergence_statistics"]
        b = m["baseline_pass_rate"]
        out.append({"task": tid, "vec": v, "ndiv": ds.get("total_divergences", 0),
                    "bucket": "ceiling" if b >= 0.9 else ("mid" if b >= 0.5 else "floor")})
    return out


def G_of(table):
    n = sum(sum(r) for r in table.values())
    rt = {k: sum(v) for k, v in table.items()}
    ct = [sum(table[k][j] for k in table) for j in range(5)]
    g = 0.0
    for k in table:
        for j in range(5):
            o, e = table[k][j], rt[k] * ct[j] / n if n else 0
            if o > 0 and e > 0:
                g += 2 * o * math.log(o / e)
    return g


def build(tasks, labels):
    t = collections.defaultdict(lambda: [0] * 5)
    for tk, b in zip(tasks, labels):
        for j in range(5):
            t[b][j] += tk["vec"][j]
    return dict(t)


def main():
    tasks = load()
    assert len(tasks) == 49
    labels = [t["bucket"] for t in tasks]
    obs = build(tasks, labels)
    G_obs = G_of(obs)
    tot = [sum(t["vec"][j] for t in tasks) for j in range(5)]
    ndiv = sum(t["ndiv"] for t in tasks)
    S = sum(tot)

    print("=" * 78)
    print("CTA REBUTTAL -- CORRECTED NUMBERS (post independent statistical review)")
    print("=" * 78)
    print(f"self-check: {len(tasks)} tasks | {ndiv} divergences | {S} SIP fires "
          f"| {' '.join(f'{SHORT[TYPES[j]]}={tot[j]}' for j in range(5))}")
    assert ndiv == 696 and S == 522, "data drift -- STOP"

    # ---- A. Finding #2, tested at the level the paper actually claims ----
    print("\n[A] FINDING #2 -- bucket-conditional SIP signature")
    print("    The paper's claim is about TASKS, but its evidence pools FIRES.")
    # invalid fire-level reference
    def chi2_sf(x, k):
        from math import exp, lgamma
        a, xx = k / 2.0, x / 2.0
        if xx < a + 1:
            s = 1.0 / a; term = s
            for i in range(1, 600):
                term *= xx / (a + i); s += term
                if abs(term) < 1e-16 * abs(s): break
            return 1.0 - s * exp(-xx + a * math.log(xx) - lgamma(a))
        tiny = 1e-300; b = xx + 1 - a; c = 1 / tiny; d = 1 / b; h = d
        for i in range(1, 600):
            an = -i * (i - a); b += 2
            d = an * d + b; d = tiny if abs(d) < tiny else d
            c = b + an / c; c = tiny if abs(c) < tiny else c
            d = 1 / d; de = d * c; h *= de
            if abs(de - 1) < 1e-14: break
        return h * exp(-xx + a * math.log(xx) - lgamma(a))

    print(f"    (i)  fire-level chi-square, each of the {S} fires treated as independent:")
    print(f"           G={G_obs:.2f}, df=8  ->  p={chi2_sf(G_obs,8):.2e}   [invalid here: ignores task clustering]")
    random.seed(20260725); N = 50000; ge = 0
    perm = list(labels)
    for _ in range(N):
        random.shuffle(perm)
        if G_of(build(tasks, perm)) >= G_obs - 1e-12:
            ge += 1
    p_task = (ge + 1) / (N + 1)
    print(f"    (ii) task-clustered permutation, full count vectors, all {len(tasks)} tasks:")
    print(f"           G={G_obs:.2f}  ->  p={p_task:.4f}   [correct test]")
    print(f"    => the apparent significance is a UNIT-OF-ANALYSIS artefact, not an effect.")

    # ---- B. ceiling SA vs EP, the only real sign test ----
    ce = [t for t in tasks if t["bucket"] == "ceiling"]
    sa_i, ep_i = TYPES.index("surface_anchoring"), TYPES.index("edge_case_prompting")
    a = sum(1 for t in ce if t["vec"][sa_i] > t["vec"][ep_i])
    b_ = sum(1 for t in ce if t["vec"][ep_i] > t["vec"][sa_i])
    ties = len(ce) - a - b_
    n = a + b_
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(a, b_) + 1)) / 2 ** n)
    print(f"\n[B] CEILING, paired per-task SA vs EP (the ONE adequately-powered sign test)")
    print(f"    SA>EP {a} | EP>SA {b_} | ties {ties} | exact two-sided p={p:.4f}")
    print(f"    pooled counts SA={sum(t['vec'][sa_i] for t in ce)} EP={sum(t['vec'][ep_i] for t in ce)} "
          f"-> the pooled margin is not reproduced at task level")

    # ---- C. destructive accounting, corrected ----
    D = tot[TYPES.index("surface_anchoring")] + tot[TYPES.index("concept_bleed")]
    lo = sum(max(t["vec"][3], t["vec"][4]) for t in tasks)
    hi = sum(min(t["vec"][3] + t["vec"][4], t["ndiv"]) for t in tasks)
    print(f"\n[C] DESTRUCTIVE ACCOUNTING (corrected)")
    print(f"    fire-level share (the paper's figure)      : {D}/{S} = {D/S:.1%}   [UNCHANGED, still correct]")
    print(f"    distinct divergences carrying SA or CB     : {lo}-{hi}")
    print(f"    => SA/CB co-firing inflates the destructive COUNT by "
          f"{D/hi-1:+.1%} to {D/lo-1:+.1%}; it does NOT reduce the share.")

    print("\n" + "=" * 78)
    print("SUPERSEDED FIGURES AND WHY")
    print("=" * 78)
    print("  x 'permutation p = 0.84 / indistinguishable from random'")
    print("      -> from an argmax-collapsed test discarding ~82% of the count information.")
    print(f"      -> correct value is p = {p_task:.2f}. The accurate description is 'not statistically supported'.")
    print("  x 'destructive share drops from 54.2% to 35-39%'")
    print("      -> that compared fires/fires against divergences/divergences (both terms changed).")
    print("      -> like-for-like the share does not drop.")
    print("  x 'the 112 forced-EP fires dilute the destructive share'")
    print("      -> applied consistently it RAISES it to 283/410 = 69.0%. The structural-EP decomposition is")
    print("         a construct-validity point about EP, not a share adjustment.")
    print("  x mid EP-vs-CB and floor EP-vs-SA sign tests as 'corroboration'")
    print("      -> 4 and 1 discordant pairs; smallest attainable two-sided p is 0.125 and 1.000.")
    print("         They cannot reach significance under any data. Only the ceiling test is reported.")
    print("\nSCOPE THE CLAIM: power simulation shows a signature strong enough to match the")
    print("paper's prose (delta>=0.5-0.6) would have been detected with >=95% probability.")
    print("The strong form of the claim is excluded; a moderate signature is not. The accurate statement is 'not supported',")
    print("and do not assert the effect is zero.")


if __name__ == "__main__":
    main()
