#!/usr/bin/env /usr/bin/python3.12
"""
Sec 6.2 test: does blind procedural-vs-declarative rating of the 49 SKILL.md docs
predict the outcomes Sec 6.2 says it predicts?

Reads the blind ratings produced by sec62_rate_skills.py, joins the per-task
outcomes, and runs the pre-registered tests. Nothing here touched the ratings.
"""
import os, json, glob, itertools
import numpy as np
from scipy import stats

RAW = "<REPO_ROOT>/rebuttal/results/sec62_procedural/raw"
OUT = "<REPO_ROOT>/rebuttal/results/sec62_procedural"
COMB = "<REPO_ROOT>/rebuttal/CTA/cta_output/cta_combined_results.json"
META = "<REPO_ROOT>/rebuttal/CTA/config/cta_task_metadata.json"
SKILLS = "<REPO_ROOT>/rebuttal/CTA/skills"

RNG = np.random.default_rng(20260725)
NBOOT, NPERM = 10000, 20000

# ---------------------------------------------------------------- ratings ----
runs = {}
for f in sorted(glob.glob(os.path.join(RAW, "ratings_*.json"))):
    d = json.load(open(f))
    tag = "%s|%s" % (d["model"], d["run_tag"])
    runs[tag] = {r["task_id"]: r["parsed"] for r in d["results"] if r["ok"]}

tasks = sorted(set.intersection(*[set(v) for v in runs.values()]))
assert len(tasks) == 49, len(tasks)

def score(tag, t):
    return float(runs[tag][t]["procedural_score"])

def binary(tag, t):
    return bool(runs[tag][t]["has_terminal_completion_step"])

GEM = ["gemini|runA", "gemini|runB"]
OPU = ["claude-opus-4.8|runA", "claude-opus-4.8|runB"]
ALL = GEM + OPU

gem_mean = np.array([np.mean([score(t_, t) for t_ in GEM]) for t in tasks])
opu_mean = np.array([np.mean([score(t_, t) for t_ in OPU]) for t in tasks])
primary = np.array([np.mean([score(t_, t) for t_ in ALL]) for t in tasks])

# binary: majority across 4 passes (ties -> True only if >=2 of 4 and both raters agree once)
bin_votes = np.array([[binary(t_, t) for t_ in ALL] for t in tasks])
primary_bin = bin_votes.sum(axis=1) >= 3   # strict: >=3/4 passes say yes

# ---------------------------------------------------------------- outcomes ---
comb = json.load(open(COMB))
meta = json.load(open(META))

SIPS = ["edge_case_prompting", "surface_anchoring", "concept_bleed",
        "redundant_exploration", "procedural_scaffolding"]

rows = []
for t in tasks:
    m = comb[t]["modules"]
    by = m.get("sip_detection", {}).get("sip_statistics", {}).get("by_type", {})
    cnt = {s: by.get(s, {}).get("count", 0) for s in SIPS}
    tot = m.get("sip_detection", {}).get("total_sips", 0)
    dv = m.get("alignment", {}).get("divergence_statistics", {}).get("by_type", {})
    md = meta[t]
    rows.append(dict(
        task_id=t,
        procedural_primary=float(primary[tasks.index(t)]),
        procedural_gemini=float(gem_mean[tasks.index(t)]),
        procedural_opus=float(opu_mean[tasks.index(t)]),
        terminal_step=bool(primary_bin[tasks.index(t)]),
        pass_rate_delta=md["pass_rate_delta"],
        baseline_pass_rate=md["baseline_pass_rate"],
        surface_anchoring=cnt["surface_anchoring"],
        concept_bleed=cnt["concept_bleed"],
        total_sips=tot,
        destructive_share=((cnt["surface_anchoring"] + cnt["concept_bleed"]) / tot) if tot else None,
        token_overhead_ratio=md["token_overhead_ratio"],
        total_divergences=m.get("alignment", {}).get("total_divergences", 0),
        unilateral_action=dv.get("unilateral_action", 0),
        doc_chars=len(open(os.path.join(SKILLS, t, "SKILL.md"), encoding="utf-8",
                           errors="replace").read()),
    ))

# ------------------------------------------------------------- statistics ----
def corr(x, y, kind):
    if kind == "spearman":
        return stats.spearmanr(x, y).statistic
    return stats.pearsonr(x, y).statistic

def boot_ci(x, y, kind):
    n = len(x); out = []
    for _ in range(NBOOT):
        i = RNG.integers(0, n, n)
        xs, ys = x[i], y[i]
        if np.std(xs) == 0 or np.std(ys) == 0:
            continue
        out.append(corr(xs, ys, kind))
    if not out:
        return (float("nan"), float("nan"))
    return tuple(float(v) for v in np.percentile(out, [2.5, 97.5]))

def perm_p(x, y, kind, direction):
    obs = corr(x, y, kind)
    xs = x.copy(); ge = 0; le = 0; abs_ge = 0
    for _ in range(NPERM):
        RNG.shuffle(xs)
        c = corr(xs, y, kind)
        if c >= obs: ge += 1
        if c <= obs: le += 1
        if abs(c) >= abs(obs): abs_ge += 1
    two = (abs_ge + 1) / (NPERM + 1)
    one = ((ge + 1) / (NPERM + 1)) if direction == "positive" else ((le + 1) / (NPERM + 1))
    return float(obs), float(one), float(two)

def test(name, x, y, direction, mask=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if mask is not None:
        x, y = x[mask], y[mask]
    res = {"n": int(len(x)), "predicted_direction": direction}
    for kind in ("spearman", "pearson"):
        obs, p1, p2 = perm_p(x, y, kind, direction)
        lo, hi = boot_ci(x, y, kind)
        res[kind] = {"r": obs, "ci95": [lo, hi], "p_one_sided_predicted": p1,
                     "p_two_sided": p2,
                     "sign_matches_prediction": (obs > 0) == (direction == "positive")}
    return name, res

# reliability -----------------------------------------------------------------
rel = {}
rel["between_rater_meanscores"] = {
    "pearson": float(stats.pearsonr(gem_mean, opu_mean).statistic),
    "spearman": float(stats.spearmanr(gem_mean, opu_mean).statistic),
    "mean_abs_diff": float(np.mean(np.abs(gem_mean - opu_mean))),
    "gemini_mean": float(gem_mean.mean()), "opus_mean": float(opu_mean.mean()),
    "gemini_sd": float(gem_mean.std(ddof=1)), "opus_sd": float(opu_mean.std(ddof=1)),
}
for a, b, lbl in [(GEM[0], GEM[1], "intra_gemini_AB"),
                  (OPU[0], OPU[1], "intra_opus_AB")]:
    xa = np.array([score(a, t) for t in tasks]); xb = np.array([score(b, t) for t in tasks])
    rel[lbl] = {"pearson": float(stats.pearsonr(xa, xb).statistic),
                "spearman": float(stats.spearmanr(xa, xb).statistic),
                "mean_abs_diff": float(np.mean(np.abs(xa - xb)))}
# all pairwise run-level
rel["pairwise_runs_pearson"] = {}
for a, b in itertools.combinations(ALL, 2):
    xa = np.array([score(a, t) for t in tasks]); xb = np.array([score(b, t) for t in tasks])
    rel["pairwise_runs_pearson"]["%s vs %s" % (a, b)] = float(stats.pearsonr(xa, xb).statistic)

def kappa(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int)
    n = len(a); po = float((a == b).mean())
    pe = float((a.mean() * b.mean()) + ((1 - a.mean()) * (1 - b.mean())))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po, pe, n

gb = np.array([sum(binary(t_, t) for t_ in GEM) >= 1 for t in tasks], int)
ob = np.array([sum(binary(t_, t) for t_ in OPU) >= 1 for t in tasks], int)
k, po, pe, n = kappa(gb, ob)
rel["binary_terminal_step"] = {
    "cohens_kappa_between_raters": k, "observed_agreement": po, "expected_agreement": pe,
    "gemini_n_true": int(gb.sum()), "opus_n_true": int(ob.sum()),
    "primary_n_true": int(primary_bin.sum()), "n": n,
    "note": "base rate is extremely low; kappa is unstable at this imbalance",
}
kA, poA, _, _ = kappa([binary(GEM[0], t) for t in tasks], [binary(GEM[1], t) for t in tasks])
kB, poB, _, _ = kappa([binary(OPU[0], t) for t in tasks], [binary(OPU[1], t) for t in tasks])
rel["binary_terminal_step"]["intra_gemini_kappa"] = kA
rel["binary_terminal_step"]["intra_opus_kappa"] = kB

r_xx = rel["between_rater_meanscores"]["pearson"]
rel["attenuation_ceiling_sqrt_r_between_raters"] = float(np.sqrt(max(r_xx, 0)))

# DV descriptives -------------------------------------------------------------
dP = np.array([r["pass_rate_delta"] for r in rows])
SA = np.array([r["surface_anchoring"] for r in rows], float)
TOT = np.array([r["total_sips"] for r in rows], float)
TOK = np.array([r["token_overhead_ratio"] for r in rows], float)
dshare_mask = np.array([r["destructive_share"] is not None for r in rows])
DSH = np.array([r["destructive_share"] if r["destructive_share"] is not None else np.nan
                for r in rows])
LEN = np.array([r["doc_chars"] for r in rows], float)

desc = {
    "dP": {"n_zero": int((dP == 0).sum()), "n": len(dP), "mean": float(dP.mean()),
           "sd": float(dP.std(ddof=1)), "min": float(dP.min()), "max": float(dP.max()),
           "n_nonzero": int((dP != 0).sum())},
    "surface_anchoring": {"mean": float(SA.mean()), "sd": float(SA.std(ddof=1)),
                          "min": float(SA.min()), "max": float(SA.max()),
                          "n_zero": int((SA == 0).sum())},
    "total_sips": {"mean": float(TOT.mean()), "sd": float(TOT.std(ddof=1)),
                   "n_zero": int((TOT == 0).sum()), "max": float(TOT.max())},
    "destructive_share": {"n_defined": int(dshare_mask.sum()),
                          "mean": float(np.nanmean(DSH)), "sd": float(np.nanstd(DSH, ddof=1))},
    "token_overhead_ratio": {"mean": float(TOK.mean()), "median": float(np.median(TOK)),
                             "min": float(TOK.min()), "max": float(TOK.max())},
    "procedural_primary": {"mean": float(primary.mean()), "sd": float(primary.std(ddof=1)),
                           "min": float(primary.min()), "max": float(primary.max())},
}

# primary + secondary tests ---------------------------------------------------
tests = {}
for label, y, direction, mask in [
    ("H1_dP", dP, "negative", None),
    ("H2_surface_anchoring", SA, "positive", None),
    ("H3_destructive_share", DSH, "positive", dshare_mask),
    ("E_total_sips", TOT, "positive", None),
    ("E_token_overhead_ratio", TOK, "positive", None),
    ("C_doc_chars_confound", LEN, "positive", None),
]:
    nm, res = test(label, primary, y, direction, mask)
    tests[nm] = res
    for rater, vec in [("gemini", gem_mean), ("opus", opu_mean)]:
        _, rr = test(label, vec, y, direction, mask)
        tests[nm]["per_rater_" + rater] = {
            "spearman_r": rr["spearman"]["r"], "spearman_p_two": rr["spearman"]["p_two_sided"],
            "pearson_r": rr["pearson"]["r"], "pearson_p_two": rr["pearson"]["p_two_sided"]}

# Holm across H1-H3 (spearman, one-sided in predicted direction)
prim = [("H1_dP", tests["H1_dP"]["spearman"]["p_one_sided_predicted"]),
        ("H2_surface_anchoring", tests["H2_surface_anchoring"]["spearman"]["p_one_sided_predicted"]),
        ("H3_destructive_share", tests["H3_destructive_share"]["spearman"]["p_one_sided_predicted"])]
prim.sort(key=lambda z: z[1])
holm = {}
for i, (nm, p) in enumerate(prim):
    holm[nm] = min(1.0, p * (len(prim) - i))
for i in range(1, len(prim)):
    holm[prim[i][0]] = max(holm[prim[i][0]], holm[prim[i - 1][0]])

# H4: terminal completion step (binary) --------------------------------------
h4 = {}
g1 = primary_bin.astype(bool); g0 = ~g1
for nm, y, mask in [("surface_anchoring", SA, None), ("destructive_share", DSH, dshare_mask),
                    ("total_sips", TOT, None), ("pass_rate_delta", dP, None)]:
    a = y[g1 & (mask if mask is not None else True)]
    b = y[g0 & (mask if mask is not None else True)]
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    u = stats.mannwhitneyu(a, b, alternative="two-sided") if len(a) and len(b) else None
    h4[nm] = {"n_true": int(len(a)), "n_false": int(len(b)),
              "median_true": float(np.median(a)) if len(a) else None,
              "median_false": float(np.median(b)) if len(b) else None,
              "mannwhitney_u": float(u.statistic) if u else None,
              "p_two_sided": float(u.pvalue) if u else None}

# power note for H1 -----------------------------------------------------------
nz = [(r["task_id"], r["pass_rate_delta"]) for r in rows if r["pass_rate_delta"] != 0]
power = {
    "n_tasks_with_dP_exactly_zero": int((dP == 0).sum()),
    "nonzero_dP_tasks": [{"task_id": a, "dP": b} for a, b in nz],
    "max_attainable_abs_spearman_given_ties": None,
}
# with 45 ties, the max |spearman| achievable by any ranking of x:
ranks_y = stats.rankdata(dP)
best = max(abs(stats.spearmanr(np.sort(ranks_y), ranks_y).statistic),
           abs(stats.spearmanr(-np.sort(ranks_y), ranks_y).statistic))
power["max_attainable_abs_spearman_given_ties"] = float(best)

report = {
    "n_tasks": len(tasks), "n_rating_passes": len(ALL), "raters": ALL,
    "reliability": rel, "descriptives": desc, "tests": tests,
    "holm_adjusted_p_primary_spearman_onesided": holm,
    "H4_terminal_completion_step": h4, "power": power,
}
json.dump(report, open(os.path.join(OUT, "sec62_correlations.json"), "w"), indent=1)

# per-task table
hdr = ["task_id", "procedural_primary", "procedural_gemini", "procedural_opus",
       "terminal_step", "pass_rate_delta", "baseline_pass_rate", "surface_anchoring",
       "concept_bleed", "total_sips", "destructive_share", "token_overhead_ratio",
       "total_divergences", "unilateral_action", "doc_chars"]
with open(os.path.join(OUT, "sec62_per_task_table.csv"), "w") as fh:
    fh.write(",".join(hdr) + "\n")
    for r in sorted(rows, key=lambda z: -z["procedural_primary"]):
        fh.write(",".join("" if r[h] is None else
                          ("%.4f" % r[h] if isinstance(r[h], float) else str(r[h]))
                          for h in hdr) + "\n")

print(json.dumps({"reliability": rel, "descriptives": desc,
                  "holm": holm, "power": power}, indent=1))
print("\n=== TESTS ===")
for k2, v in tests.items():
    print("%-26s n=%2d  spearman r=%+.3f CI[%+.3f,%+.3f] p1=%.4f p2=%.4f | pearson r=%+.3f p2=%.4f | pred=%s match=%s"
          % (k2, v["n"], v["spearman"]["r"], v["spearman"]["ci95"][0], v["spearman"]["ci95"][1],
             v["spearman"]["p_one_sided_predicted"], v["spearman"]["p_two_sided"],
             v["pearson"]["r"], v["pearson"]["p_two_sided"],
             v["predicted_direction"], v["spearman"]["sign_matches_prediction"]))
print("\n=== H4 ===")
print(json.dumps(h4, indent=1))
