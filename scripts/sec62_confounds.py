#!/usr/bin/env /usr/bin/python3.12
"""
Supplementary to sec62_analyze.py. The headline H2 result (procedural_score
correlates NEGATIVELY with surface_anchoring, i.e. opposite to Sec 6.2) could be
an artefact of document length: declarative reference docs are long, and a long
document simply offers more surface for the detector to flag. Also test the
paper's own known driver, baseline pass rate (the ceiling bucket carries 415 of
522 SIPs).

Reports zero-order, partial (rank), and semi-partial correlations plus a
permutation p for the partials.
"""
import json, os
import numpy as np
from scipy import stats

OUT = "<REPO_ROOT>/rebuttal/results/sec62_procedural"
RNG = np.random.default_rng(7717)
NPERM = 20000

rows = []
with open(os.path.join(OUT, "sec62_per_task_table.csv")) as fh:
    hdr = fh.readline().strip().split(",")
    for line in fh:
        v = line.rstrip("\n").split(",")
        d = dict(zip(hdr, v))
        rows.append(d)

def col(name, f=float):
    return np.array([f(r[name]) if r[name] != "" else np.nan for r in rows])

proc = col("procedural_primary")
SA = col("surface_anchoring")
TOT = col("total_sips")
DSH = col("destructive_share")
LEN = np.log10(col("doc_chars"))
BASE = col("baseline_pass_rate")
TOK = col("token_overhead_ratio")
DIV = col("total_divergences")

def rankz(x):
    return stats.rankdata(x)

def partial(x, y, covs, use_rank=True):
    """Correlation of x,y after linearly removing covs (rank-transformed => partial Spearman)."""
    m = ~(np.isnan(x) | np.isnan(y) | np.any(np.isnan(covs), axis=0))
    xs, ys = x[m], y[m]
    cs = np.vstack([c[m] for c in covs])
    if use_rank:
        xs, ys = rankz(xs), rankz(ys)
        cs = np.vstack([rankz(c) for c in cs])
    A = np.vstack([np.ones(len(xs)), cs]).T
    rx = xs - A @ np.linalg.lstsq(A, xs, rcond=None)[0]
    ry = ys - A @ np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(stats.pearsonr(rx, ry).statistic), int(m.sum()), rx, ry

def perm_partial(x, y, covs, use_rank=True):
    obs, n, rx, ry = partial(x, y, covs, use_rank)
    cnt = 0
    rxs = rx.copy()
    for _ in range(NPERM):
        RNG.shuffle(rxs)
        if abs(float(stats.pearsonr(rxs, ry).statistic)) >= abs(obs):
            cnt += 1
    return obs, n, (cnt + 1) / (NPERM + 1)

report = {}

# zero-order structure of the confounds
zo = {}
for nm, a, b in [
    ("procedural~log_len", proc, LEN),
    ("SA~log_len", SA, LEN),
    ("total_sips~log_len", TOT, LEN),
    ("destructive_share~log_len", DSH, LEN),
    ("SA~baseline_pass_rate", SA, BASE),
    ("procedural~baseline_pass_rate", proc, BASE),
    ("total_sips~baseline_pass_rate", TOT, BASE),
    ("SA~total_divergences", SA, DIV),
    ("procedural~total_divergences", proc, DIV),
    ("total_divergences~log_len", DIV, LEN),
]:
    m = ~(np.isnan(a) | np.isnan(b))
    zo[nm] = {"spearman": float(stats.spearmanr(a[m], b[m]).statistic),
              "pearson": float(stats.pearsonr(a[m], b[m]).statistic),
              "n": int(m.sum())}
report["zero_order_confound_structure"] = zo

# partials for the three pre-registered DVs
part = {}
for nm, y in [("surface_anchoring", SA), ("destructive_share", DSH),
              ("total_sips", TOT), ("token_overhead_ratio", TOK)]:
    part[nm] = {}
    for cname, covs in [("control_log_len", [LEN]),
                        ("control_baseline", [BASE]),
                        ("control_log_len+baseline", [LEN, BASE]),
                        ("control_log_len+baseline+total_divergences", [LEN, BASE, DIV])]:
        r, n, p = perm_partial(proc, y, covs)
        part[nm][cname] = {"partial_spearman": r, "n": n, "perm_p_two_sided": p}
report["partial_correlations_procedural_vs_outcome"] = part

# Is procedural_score doing anything beyond length? Compare nested rank models.
m = ~np.isnan(SA)
def r2(X, y):
    A = np.vstack([np.ones(len(y))] + [rankz(c) for c in X]).T if X else np.ones((len(y), 1))
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ b
    return 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
ySA = rankz(SA[m])
report["nested_rank_models_SA"] = {
    "R2_len_only": r2([LEN[m]], ySA),
    "R2_len_plus_procedural": r2([LEN[m], proc[m]], ySA),
    "R2_procedural_only": r2([proc[m]], ySA),
    "R2_len_baseline": r2([LEN[m], BASE[m]], ySA),
    "R2_len_baseline_plus_procedural": r2([LEN[m], BASE[m], proc[m]], ySA),
}

# Robustness: does the wrong-direction H2 survive dropping the most extreme tasks?
sens = {}
order = np.argsort(-SA)
for k in (0, 1, 2, 3):
    keep = np.ones(len(SA), bool)
    keep[order[:k]] = False
    sens["drop_top%d_SA" % k] = {
        "spearman": float(stats.spearmanr(proc[keep], SA[keep]).statistic),
        "pearson": float(stats.pearsonr(proc[keep], SA[keep]).statistic), "n": int(keep.sum())}
# and restricted to the ceiling bucket only (baseline >= 0.9), where 415/522 SIPs live
cm = BASE >= 0.9
sens["ceiling_bucket_only"] = {
    "spearman": float(stats.spearmanr(proc[cm], SA[cm]).statistic),
    "pearson": float(stats.pearsonr(proc[cm], SA[cm]).statistic), "n": int(cm.sum())}
report["sensitivity_H2"] = sens

json.dump(report, open(os.path.join(OUT, "sec62_confounds.json"), "w"), indent=1)
print(json.dumps(report, indent=1))
