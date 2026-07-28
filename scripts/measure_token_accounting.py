#!/usr/bin/env python3
"""
EXPERIMENT: measure the token-accounting discrepancy, and price the Sonnet null.

Run this on the machine that has the raw traces (./claude_process/**/claude_thinking/*.jsonl).
No API calls. Pure stdlib. Takes seconds.

It answers three questions with MEASUREMENTS, not estimates:

  Q1. Holding the RECORD SET FIXED (assistant records only, exactly what
      module1_parser keeps), does the paper's FIELD SET (input_tokens +
      output_tokens) differ materially from full accounting (all four usage
      fields)? Usage found on non-assistant records is reported separately so
      the field-set comparison is not contaminated by a record-set change.

  Q2. What do the corrected token_overhead_ratio numbers look like, i.e. what
      should Sec 4.5 / Table 3 actually say?  These are printed ONLY if the
      metric-replication gate PASSES (see below); otherwise they are withheld,
      or emitted with an UNVALIDATED banner under --allow-unvalidated.

  Q3. What is the TRUE token volume of one agent run, split BY CONDITION?
      The Sonnet null baseline is a no-skill design, so it is priced off
      WITHOUT-skill runs only. With-skill runs are priced separately, for the
      r=3 both-conditions regeneration design.

HARD GATE (do not remove): before any corrected Sec 4.5 number is printed, the
script recomputes the paper's own metric from the traces and compares it to the
shipped cta_task_metadata.json. If the reproduction rate or the task coverage is
below threshold -- the signature of being pointed at the wrong metadata file or
a different trace tree -- the run is declared FAILED and the corrected numbers
are withheld.

Usage:
    /usr/bin/python3.12 scripts/measure_token_accounting.py \
        --traces ./claude_process \
        --metadata ./config/cta_task_metadata.json \
        --out ./token_accounting_report

Outputs <out>.json and <out>.txt

Exit codes:
    0  replication gate PASSED
    1  replication gate FAILED (corrected Sec 4.5 numbers withheld/unvalidated)
    2  no trace files found under --traces
    3  traces found but nothing scoreable (diagnosed, not a traceback)
"""

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Filename matching.
#
# Two reference parsers exist in the paper's own codebase and they do NOT agree:
#
#   CTA/src/cta/module1_parser.py  _FILENAME_RE   (the parser that produced the
#       paper's token metric): non-greedy task id, and a trailing ``.*`` so any
#       suffix after the timestamp is allowed; strips ``.jsonl`` OR ``.json``.
#
#   CTA/scripts/analyze_tokens.py  parse_filename : greedy task id, anchored
#       ``$`` right after the timestamp, so ANY trailing suffix is rejected;
#       strips ``.jsonl`` only.
#
# The pre-fix version of this script copied the STRICT (analyze_tokens) pattern
# and dropped non-matching traces silently. We now accept whatever EITHER
# reference parser accepts (i.e. the module1_parser superset), and we REPORT
# every file that is skipped, plus every file the strict pattern would have
# silently discarded, plus any file where the two parsers disagree on task id.
# ---------------------------------------------------------------------------

# module1_parser.py::_FILENAME_RE  (permissive; superset of the strict one)
PAPER_RE = re.compile(
    r"^claude_(?P<task>.+?)"
    r"_use-agent-(?P<agent>true|false)"
    r"_use-skill-(?P<skill>true|false)"
    r"_(?P<date>\d{8})_(?P<time>\d{6})(?P<tail>.*)$"
)

# analyze_tokens.py::parse_filename  (strict; what this script used to use)
STRICT_RE = re.compile(
    r"^claude_(?P<task>.+)"
    r"_use-agent-(?:true|false)"
    r"_use-skill-(?P<skill>true|false)"
    r"_\d{8}_\d{6}$"
)

TRACE_SUFFIXES = (".jsonl", ".json")


def strip_suffix(name: str) -> str:
    for suf in TRACE_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


# ---------------------------------------------------------------------------
# Prices.  ASSUMPTIONS, printed in the report so a reader can check them.
# Public list prices, USD per 1,000,000 tokens. Cache-write price is the 5-minute
# TTL tier (the 1-hour TTL tier is 2x the base input price).
# Verify against https://www.anthropic.com/pricing before quoting a dollar figure;
# every number is overridable with --price-input/--price-output/--price-cache-*.
# ---------------------------------------------------------------------------
PRICE_TABLE = {
    ("claude-sonnet-4-5", "standard"): {
        "input": 3.00, "cache_write": 3.75, "cache_read": 0.30, "output": 15.00},
    ("claude-sonnet-4-5", "long-context"): {   # prompts > 200K tokens
        "input": 6.00, "cache_write": 7.50, "cache_read": 0.60, "output": 22.50},
    ("claude-opus-4-1", "standard"): {
        "input": 15.00, "cache_write": 18.75, "cache_read": 1.50, "output": 75.00},
    ("claude-haiku-4-5", "standard"): {
        "input": 1.00, "cache_write": 1.25, "cache_read": 0.10, "output": 5.00},
}
PRICE_SOURCE_NOTE = ("public list prices, USD per 1M tokens; cache-write = 5-minute TTL tier; "
                     "assumed as of 2026-07 -- re-check anthropic.com/pricing before quoting")


# ---------------------------------------------------------------------------
# Small safe-stat helpers (a StatisticsError traceback is not a diagnosis).
# ---------------------------------------------------------------------------
def _mean(xs):
    return st.mean(xs) if xs else None


def _median(xs):
    return st.median(xs) if xs else None


def _max(xs):
    return max(xs) if xs else None


def fmt(v, spec="{:.3f}", na="   n/a"):
    return na if v is None else spec.format(v)


# ---------------------------------------------------------------------------
def parse_trace(path):
    """Return the full accounting for one trace file, split by record type.

    The paper's metric (module1_parser) keeps ONLY records with type=="assistant"
    and sums message.usage.input_tokens + output_tokens. To compare field sets
    honestly we keep that record set fixed and additionally sum the two cache
    fields over the SAME records. Usage seen on any other record type is
    accumulated separately and never mixed into the comparison.
    """
    a_in = a_out = a_cw = a_cr = 0          # assistant records (paper's record set)
    n_in = n_out = n_cw = n_cr = 0          # every other record type
    n_assistant = 0
    n_nonassistant_usage = 0
    nonassistant_types = Counter()
    lines = bad_json = usage_records = 0

    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            lines += 1
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            if not isinstance(rec, dict):
                bad_json += 1
                continue
            msg = rec.get("message") or {}
            usage = msg.get("usage") if isinstance(msg, dict) else None
            if not isinstance(usage, dict) or not usage:
                continue
            usage_records += 1
            i = int(usage.get("input_tokens", 0) or 0)
            o = int(usage.get("output_tokens", 0) or 0)
            cw = int(usage.get("cache_creation_input_tokens", 0) or 0)
            cr = int(usage.get("cache_read_input_tokens", 0) or 0)
            if rec.get("type") == "assistant":       # module1_parser's filter
                a_in += i; a_out += o; a_cw += cw; a_cr += cr
                n_assistant += 1
            else:
                n_in += i; n_out += o; n_cw += cw; n_cr += cr
                n_nonassistant_usage += 1
                nonassistant_types[str(rec.get("type"))] += 1

    return {
        # paper field set, paper record set
        "paper_total": a_in + a_out,
        # full field set, SAME (assistant) record set  <- the clean comparison
        "a_input": a_in, "a_output": a_out, "a_cache_write": a_cw, "a_cache_read": a_cr,
        "assistant_full_total": a_in + a_out + a_cw + a_cr,
        # everything else, reported separately, never silently merged
        "na_input": n_in, "na_output": n_out, "na_cache_write": n_cw, "na_cache_read": n_cr,
        "nonassistant_full_total": n_in + n_out + n_cw + n_cr,
        "nonassistant_usage_records": n_nonassistant_usage,
        "nonassistant_types": dict(nonassistant_types),
        "allrec_total": a_in + a_out + a_cw + a_cr + n_in + n_out + n_cw + n_cr,
        "assistant_msgs": n_assistant,
        "lines": lines, "bad_json_lines": bad_json, "usage_records": usage_records,
    }


def run_cost_usd(t, price, include_nonassistant=False):
    """Price ONE run. Billing follows real API responses, which are the assistant
    records; non-assistant usage blocks are frequently restatements of the same
    call, so they are excluded by default and reported as a separate risk."""
    i = t["a_input"] + (t["na_input"] if include_nonassistant else 0)
    cw = t["a_cache_write"] + (t["na_cache_write"] if include_nonassistant else 0)
    cr = t["a_cache_read"] + (t["na_cache_read"] if include_nonassistant else 0)
    o = t["a_output"] + (t["na_output"] if include_nonassistant else 0)
    return (i * price["input"] + cw * price["cache_write"]
            + cr * price["cache_read"] + o * price["output"]) / 1_000_000


# ---------------------------------------------------------------------------
def collect_files(root: Path):
    """Find trace files; return (scored_files, layout_diagnostics)."""
    files, diag = [], {}
    for pat in ("**/claude_thinking/*.jsonl", "**/claude_thinking/*.json"):
        files.extend(root.glob(pat))
    files = sorted(set(files))
    diag["n_under_claude_thinking"] = len(files)
    if not files:
        stray = sorted(root.rglob("*.jsonl"))[:2000]
        diag["n_jsonl_anywhere_in_tree"] = len(stray)
        diag["stray_examples"] = [str(p) for p in stray[:8]]
    return files, diag


def match_files(files):
    """Apply the permissive (module1_parser) pattern; report everything skipped."""
    runs = defaultdict(lambda: {"with": [], "without": []})
    skipped, strict_only_drops, task_id_disagreements, accepted = [], [], [], []
    for fp in files:
        stem = strip_suffix(fp.name)
        m = PAPER_RE.match(stem)
        if not m:
            skipped.append(fp.name)
            continue
        s = STRICT_RE.match(stem)
        if not s:
            strict_only_drops.append(fp.name)
        elif s.group("task") != m.group("task"):
            task_id_disagreements.append(
                f"{fp.name}  [module1='{m.group('task')}' vs analyze_tokens='{s.group('task')}']")
        task = m.group("task")
        cond = "with" if m.group("skill") == "true" else "without"
        rec = parse_trace(fp)
        rec["file"] = str(fp)
        rec["task"] = task
        rec["cond"] = cond
        runs[task][cond].append(rec)
        accepted.append(rec)
    return runs, accepted, skipped, strict_only_drops, task_id_disagreements


def crosscheck_metadata(rows, metadata_path, tol):
    """Recompute the paper's own metric and compare against shipped metadata."""
    out = {"metadata_path": str(metadata_path), "loaded": False, "error": None,
           "n_metadata_entries": 0, "checked": 0, "reproduced": 0, "mismatched": 0,
           "missing_from_metadata": [], "worst": []}
    try:
        md = json.load(open(metadata_path))
    except Exception as e:                                  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["loaded"] = True
    if isinstance(md, list):
        entries = {str(t.get("skill_id")): t for t in md if isinstance(t, dict)}
    elif isinstance(md, dict):
        entries = {}
        for k, v in md.items():
            if isinstance(v, dict):
                entries[str(v.get("skill_id", k))] = v
    else:
        out["error"] = f"metadata is a {type(md).__name__}, expected list/dict"
        return out
    out["n_metadata_entries"] = len(entries)

    diffs = []
    for r in rows:
        s = entries.get(r["task"])
        if not s or s.get("token_overhead_ratio") is None:
            out["missing_from_metadata"].append(r["task"])
            continue
        out["checked"] += 1
        shipped = float(s["token_overhead_ratio"])
        d = abs(shipped - r["paper_ratio"])
        diffs.append((d, r["task"], shipped, r["paper_ratio"]))
        if d > tol:
            out["mismatched"] += 1
        else:
            out["reproduced"] += 1
    diffs.sort(reverse=True)
    out["worst"] = [{"task": t, "shipped": sh, "recomputed": ours, "abs_diff": d}
                    for d, t, sh, ours in diffs[:6]]
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="./claude_process")
    ap.add_argument("--metadata", default="./config/cta_task_metadata.json")
    ap.add_argument("--out", default="./token_accounting_report")
    ap.add_argument("--model", default="claude-sonnet-4-5",
                    choices=sorted({m for m, _ in PRICE_TABLE}))
    ap.add_argument("--price-tier", default="standard", choices=["standard", "long-context"])
    ap.add_argument("--price-input", type=float, default=None)
    ap.add_argument("--price-output", type=float, default=None)
    ap.add_argument("--price-cache-write", type=float, default=None)
    ap.add_argument("--price-cache-read", type=float, default=None)
    ap.add_argument("--repro-tol", type=float, default=0.05,
                    help="abs tolerance when matching shipped token_overhead_ratio")
    ap.add_argument("--min-repro-rate", type=float, default=0.90,
                    help="gate: fraction of cross-checked tasks that must reproduce")
    ap.add_argument("--min-coverage", type=float, default=0.50,
                    help="gate: fraction of paired tasks that must be present in metadata")
    ap.add_argument("--min-metadata-coverage", type=float, default=0.80,
                    help="gate: fraction of the metadata's tasks that must have been measured "
                         "(catches --traces pointing at a partial trace tree). Lower it "
                         "deliberately if you really mean to score a subset.")
    ap.add_argument("--allow-unvalidated", action="store_true",
                    help="on gate FAIL, still print Sec 4.5 numbers, marked UNVALIDATED")
    ap.add_argument("--max-examples", type=int, default=8)
    a = ap.parse_args()

    price = dict(PRICE_TABLE[(a.model, a.price_tier)])
    for k, v in (("input", a.price_input), ("output", a.price_output),
                 ("cache_write", a.price_cache_write), ("cache_read", a.price_cache_read)):
        if v is not None:
            price[k] = v
    price_overridden = any(v is not None for v in
                           (a.price_input, a.price_output, a.price_cache_write, a.price_cache_read))

    root = Path(a.traces)
    if not root.exists():
        print(f"[FATAL] --traces path does not exist: {root}\n"
              f"        (resolved to {root.resolve()})")
        sys.exit(2)

    files, layout = collect_files(root)
    if not files:
        msg = [f"[FATAL] No traces under {root.resolve()}/**/claude_thinking/*.jsonl(.json)"]
        if layout.get("n_jsonl_anywhere_in_tree"):
            msg.append(f"        BUT the tree does contain {layout['n_jsonl_anywhere_in_tree']} "
                       f"*.jsonl file(s) outside any claude_thinking/ directory, e.g.:")
            msg += [f"          {p}" for p in layout["stray_examples"]]
            msg.append("        -> the trace tree has a different layout than the paper's "
                       "claude_process/<model>/<batch>/claude_thinking/. Point --traces at the "
                       "root that CONTAINS claude_thinking/, or move/symlink the files.")
        else:
            msg.append("        The tree contains no *.jsonl at all -- run this on the machine "
                       "holding the raw traces.")
        print("\n".join(msg))
        sys.exit(2)

    runs, accepted, skipped, strict_only, disagree = match_files(files)

    # ---------------- per-task rows (paired tasks only) ---------------------
    def agg(lst, key):
        return _mean([r[key] for r in lst]) or 0.0

    rows, paired_runs = [], []
    only_with, only_without = [], []
    for task, c in sorted(runs.items()):
        if not c["with"] or not c["without"]:
            (only_with if c["with"] else only_without).append(task)
            continue
        paired_runs.extend(c["with"] + c["without"])
        pw, po = agg(c["with"], "paper_total"), agg(c["without"], "paper_total")
        fw, fo = agg(c["with"], "assistant_full_total"), agg(c["without"], "assistant_full_total")
        aw, ao = agg(c["with"], "allrec_total"), agg(c["without"], "allrec_total")
        rows.append({
            "task": task,
            "n_with": len(c["with"]), "n_without": len(c["without"]),
            "paper_with": pw, "paper_without": po,
            "paper_ratio": (pw + 1) / (po + 1),            # module1's formula
            "full_with": fw, "full_without": fo,
            "full_ratio": (fw + 1) / (fo + 1),             # assistant records, all 4 fields
            "allrec_with": aw, "allrec_without": ao,
            "allrec_ratio": (aw + 1) / (ao + 1),           # sensitivity only
            "cache_share_with": ((agg(c["with"], "a_cache_read") + agg(c["with"], "a_cache_write")) / fw
                                 if fw else None),
            "cache_share_without": ((agg(c["without"], "a_cache_read") + agg(c["without"], "a_cache_write")) / fo
                                    if fo else None),
        })

    tot_bad_json = sum(r["bad_json_lines"] for r in accepted)
    files_with_bad_json = [Path(r["file"]).name for r in accepted if r["bad_json_lines"]]
    tot_usage_records = sum(r["usage_records"] for r in accepted)
    na_records = sum(r["nonassistant_usage_records"] for r in accepted)
    na_types = Counter()
    for r in accepted:
        na_types.update(r["nonassistant_types"])

    L = []
    P = L.append
    P("=" * 78)
    P("TOKEN ACCOUNTING EXPERIMENT")
    P("=" * 78)
    P(f"trace root                 : {root.resolve()}")
    P(f"trace files found          : {len(files)}   (glob: **/claude_thinking/*.jsonl|*.json)")
    P(f"filename-matched (scored)  : {len(accepted)}")
    P(f"filename-SKIPPED (dropped) : {len(skipped)}")
    for n in skipped[: a.max_examples]:
        P(f"    skipped: {n}")
    if len(skipped) > a.max_examples:
        P(f"    ... and {len(skipped) - a.max_examples} more")
    if skipped:
        P("    ^ these matched NEITHER module1_parser._FILENAME_RE nor "
          "analyze_tokens.parse_filename")
    P(f"accepted by module1_parser but REJECTED by the strict analyze_tokens regex : "
      f"{len(strict_only)}")
    for n in strict_only[: a.max_examples]:
        P(f"    would have been silently dropped: {n}")
    if len(strict_only) > a.max_examples:
        P(f"    ... and {len(strict_only) - a.max_examples} more")
    if disagree:
        P(f"task-id DISAGREEMENT between the two reference parsers : {len(disagree)} "
          f"(module1 non-greedy wins here)")
        for n in disagree[: a.max_examples]:
            P(f"    {n}")
    P(f"malformed JSON lines       : {tot_bad_json} in {len(set(files_with_bad_json))} file(s)"
      + (f"  e.g. {sorted(set(files_with_bad_json))[:3]}" if files_with_bad_json else ""))
    P(f"records carrying usage     : {tot_usage_records} "
      f"({tot_usage_records - na_records} assistant, {na_records} non-assistant"
      + (f" {dict(na_types)})" if na_types else ")"))
    P(f"tasks seen                 : {len(runs)}   "
      f"(paired {len(rows)}, with-skill only {len(only_with)}, without-skill only {len(only_without)})")
    for label, lst in (("with-skill only", only_with), ("without-skill only", only_without)):
        if lst:
            P(f"    UNPAIRED [{label}]: {', '.join(lst[: a.max_examples])}"
              + (f" ... +{len(lst) - a.max_examples}" if len(lst) > a.max_examples else ""))
            P("      ^ excluded from Q1/Q2 ratios (no counterfactual); still priced in Q3.")
    P("")

    # ---------------- B5: nothing scoreable -> diagnose, do not crash -------
    if not rows:
        P("!" * 78)
        P("NOTHING SCOREABLE -- no task has BOTH a with-skill and a without-skill run.")
        P("No ratio, mean or median can be computed. Likely cause, in order:")
        if not accepted:
            P("  1. EVERY trace filename was skipped by the filename pattern (see the")
            P("     'filename-SKIPPED' list above). The pattern the paper's own parsers")
            P("     accept is: claude_<task>_use-agent-<bool>_use-skill-<bool>_YYYYMMDD_HHMMSS[...]")
            P("     If your traces use a different convention, rename them or extend PAPER_RE.")
        elif len(runs) and not rows:
            P("  1. Traces parsed fine, but every task has only ONE condition "
              f"(with-only {len(only_with)}, without-only {len(only_without)}).")
            P("     Either half the runs are missing, or the use-skill-<bool> field in the")
            P("     filenames is not varying. Check the UNPAIRED list above.")
        if tot_usage_records == 0:
            P("  2. Not one record in any accepted file carried message.usage -- this is not")
            P("     a Claude Code stream-json trace tree, or usage was stripped.")
        P("  3. --traces may point at the wrong tree entirely.")
        P("!" * 78)
        txt = "\n".join(L)
        print(txt)
        Path(a.out + ".txt").write_text(txt)
        Path(a.out + ".json").write_text(json.dumps({
            "status": "NOTHING_SCOREABLE",
            "files_found": len(files), "files_matched": len(accepted),
            "files_skipped": skipped, "strict_regex_would_drop": strict_only,
            "tasks_seen": len(runs), "unpaired_with_only": only_with,
            "unpaired_without_only": only_without,
            "usage_records": tot_usage_records, "bad_json_lines": tot_bad_json,
            "per_task": [], "summary": None,
        }, indent=2))
        print(f"\nwrote {a.out}.txt and {a.out}.json")
        sys.exit(3)

    # ---------------- B1: hard replication gate ----------------------------
    cc = crosscheck_metadata(rows, a.metadata, a.repro_tol)
    coverage = cc["checked"] / len(rows) if rows else 0.0
    md_coverage = (cc["checked"] / cc["n_metadata_entries"]) if cc["n_metadata_entries"] else 0.0
    rate = cc["reproduced"] / cc["checked"] if cc["checked"] else 0.0
    cc["coverage"] = coverage
    cc["metadata_coverage"] = md_coverage
    cc["rate"] = rate

    reasons = []
    if not cc["loaded"]:
        reasons.append(f"metadata could not be loaded ({cc['error']})")
    elif cc["checked"] == 0:
        reasons.append("ZERO task ids are shared between the traces and the metadata file "
                       "-- almost certainly the wrong --metadata or the wrong --traces tree")
    else:
        if coverage < a.min_coverage:
            reasons.append(f"metadata covers only {coverage:.0%} of paired tasks "
                           f"(< --min-coverage {a.min_coverage:.0%})")
        if md_coverage < a.min_metadata_coverage:
            reasons.append(f"only {cc['checked']}/{cc['n_metadata_entries']} "
                           f"({md_coverage:.0%}) of the metadata's tasks were measured "
                           f"(< --min-metadata-coverage {a.min_metadata_coverage:.0%}) -- "
                           f"--traces looks like a PARTIAL trace tree, so any rewritten "
                           f"Sec 4.5 number would describe a subset, not the 49-task benchmark")
        if rate < a.min_repro_rate:
            reasons.append(f"reproduction rate {rate:.0%} < --min-repro-rate {a.min_repro_rate:.0%}")
    gate_pass = not reasons

    P("--- GATE: did we replicate the PAPER's OWN metric? ---")
    P(f"  metadata file      : {cc['metadata_path']}  "
      f"({cc['n_metadata_entries']} entries)" if cc["loaded"] else
      f"  metadata file      : {cc['metadata_path']}  [UNREADABLE: {cc['error']}]")
    P(f"  paired tasks       : {len(rows)}   cross-checked: {cc['checked']} "
      f"(coverage {coverage:.0%}, threshold {a.min_coverage:.0%})")
    P(f"  metadata measured  : {cc['checked']}/{cc['n_metadata_entries']} "
      f"({md_coverage:.0%}, threshold {a.min_metadata_coverage:.0%}) "
      f"-- guards against a partial trace tree")
    P(f"  reproduced         : {cc['reproduced']}/{cc['checked']} "
      f"(rate {rate:.0%}, threshold {a.min_repro_rate:.0%}, tol {a.repro_tol})")
    P(f"  mismatched         : {cc['mismatched']}")
    if cc["missing_from_metadata"]:
        miss = cc["missing_from_metadata"]
        P(f"  absent from metadata: {len(miss)}  e.g. {', '.join(miss[: a.max_examples])}")
    if cc["worst"]:
        P("  largest |shipped - recomputed| :")
        for w in cc["worst"]:
            flag = "MISMATCH" if w["abs_diff"] > a.repro_tol else "ok"
            P(f"    {w['task']:34} shipped {w['shipped']:9.4f}  ours {w['recomputed']:9.4f}"
              f"  d={w['abs_diff']:8.4f}  [{flag}]")
    if gate_pass:
        P("  RESULT: *** PASS *** -- we reproduce the paper's metric; corrected numbers below "
          "are trustworthy.")
    else:
        P("  RESULT: *** FAIL ***")
        for r in reasons:
            P(f"    - {r}")
        P("    Corrected Sec 4.5 / Table 3 numbers are " +
          ("PRINTED BUT UNVALIDATED (--allow-unvalidated)." if a.allow_unvalidated
           else "WITHHELD. Fix the inputs and re-run, or pass --allow-unvalidated."))
        P("    DO NOT rewrite the paper from a FAILED run.")
    P("")

    pr = [r["paper_ratio"] for r in rows]
    fr = [r["full_ratio"] for r in rows]
    ar = [r["allrec_ratio"] for r in rows]
    cs = [r["cache_share_with"] for r in rows if r["cache_share_with"] is not None]

    # ---------------- B4: field-set change with the record set held fixed --
    na_tokens = sum(r["nonassistant_full_total"] for r in paired_runs)
    all_tokens = sum(r["allrec_total"] for r in paired_runs)
    P("--- Q1: does the metric change?  (RECORD SET HELD FIXED = assistant records only) ---")
    P(f"  paper field set (input+output)      mean ratio {fmt(_mean(pr),'{:6.3f}')}   "
      f"median {fmt(_median(pr),'{:6.3f}')}")
    P(f"  full field set  (all four fields)   mean ratio {fmt(_mean(fr),'{:6.3f}')}   "
      f"median {fmt(_median(fr),'{:6.3f}')}")
    P(f"  tasks with ratio < 1.0 : paper {sum(1 for x in pr if x < 1):3d}/{len(pr)}"
      f"   -> full {sum(1 for x in fr if x < 1):3d}/{len(fr)}")
    P(f"  tasks with ratio < 0.5 : paper {sum(1 for x in pr if x < .5):3d}/{len(pr)}"
      f"   -> full {sum(1 for x in fr if x < .5):3d}/{len(fr)}")
    P(f"  mean share of with-skill ASSISTANT volume that is CACHED: "
      f"{fmt(_mean(cs), '{:.1%}')}" + ("" if cs else "   (no non-zero with-skill volume)"))
    P("  (if this share is high, the paper's field set was blind to most of the volume)")
    P("  NON-ASSISTANT usage records, reported separately so the field-set comparison stays clean:")
    P(f"    records {na_records}  tokens {na_tokens:,}  "
      f"({(na_tokens / all_tokens if all_tokens else 0):.1%} of all-record volume)  "
      f"types {dict(na_types) if na_types else '{}'}")
    P(f"    all-record ratio (record set CHANGED -- sensitivity only, not the headline): "
      f"mean {fmt(_mean(ar),'{:.3f}')}  median {fmt(_median(ar),'{:.3f}')}")
    P("")

    # ---------------- Q2 (gated) -------------------------------------------
    show_q2 = gate_pass or a.allow_unvalidated
    tag = "" if gate_pass else "UNVALIDATED "
    P("--- Q2: corrected Sec 4.5 / Table 3 inputs ---")
    if not show_q2:
        P("  " + "#" * 70)
        P("  #  WITHHELD: the metric-replication gate FAILED (see above).")
        P("  #  Publishing these numbers would mean rewriting Sec 4.5 from a run that")
        P("  #  provably does not reproduce the paper's own metric.")
        P("  " + "#" * 70)
    else:
        if not gate_pass:
            P("  " + "!" * 70)
            P("  !  UNVALIDATED -- gate FAILED. These numbers must NOT enter the paper.")
            P("  " + "!" * 70)
        P(f"  {tag}corrected mean overhead   {fmt(_mean(fr), '{:.2f}')}x")
        P(f"  {tag}corrected median overhead {fmt(_median(fr), '{:.2f}')}x")
        best = max(rows, key=lambda r: r["full_ratio"])
        P(f"  {tag}corrected max overhead    {best['full_ratio']:.2f}x  ({best['task']})")
        P(f"  {tag}largest DIRECTION changes (paper ratio -> corrected ratio):")
        for r in sorted(rows, key=lambda r: -abs(r["full_ratio"] - r["paper_ratio"]))[:10]:
            P(f"    {r['task']:34} {r['paper_ratio']:7.3f} -> {r['full_ratio']:7.3f}")
    P("")

    # ---------------- B3: Q3 priced per condition --------------------------
    with_runs = [r for r in accepted if r["cond"] == "with"]
    without_runs = [r for r in accepted if r["cond"] == "without"]
    cost_w = [run_cost_usd(r, price) for r in with_runs]
    cost_o = [run_cost_usd(r, price) for r in without_runs]
    vol_w = [r["allrec_total"] for r in with_runs]
    vol_o = [r["allrec_total"] for r in without_runs]
    paper_o = [r["paper_total"] for r in without_runs]

    P(f"--- Q3: true cost of ONE agent run ({a.model}, {a.price_tier} tier) ---")
    P(f"  PRICE ASSUMPTIONS ({PRICE_SOURCE_NOTE}"
      + (", OVERRIDDEN on the command line" if price_overridden else "") + "):")
    P(f"    input ${price['input']:.2f}/M   cache-write ${price['cache_write']:.2f}/M   "
      f"cache-read ${price['cache_read']:.2f}/M   output ${price['output']:.2f}/M")
    P("    billed volume = assistant records only (non-assistant usage blocks are usually")
    P("    restatements of the same API call; see the delta line below if any exist).")
    P(f"  runs parsed          : {len(accepted)}  ({len(with_runs)} with-skill, "
      f"{len(without_runs)} without-skill)")
    for label, vol, cost in (("with-skill   ", vol_w, cost_w), ("without-skill", vol_o, cost_o)):
        if not vol:
            P(f"  {label} : NONE PARSED -- cannot price this condition.")
            continue
        P(f"  {label} : tokens median {fmt(_median(vol), '{:,.0f}')}  mean {fmt(_mean(vol), '{:,.0f}')}"
          f"  max {fmt(_max(vol), '{:,.0f}')}")
        P(f"  {label} : cost   median ${fmt(_median(cost), '{:.3f}')}  mean ${fmt(_mean(cost), '{:.3f}')}"
          f"  max ${fmt(_max(cost), '{:.3f}')}")
    if na_records:
        alt = sum(run_cost_usd(r, price, include_nonassistant=True) for r in accepted)
        base = sum(run_cost_usd(r, price) for r in accepted)
        P(f"  RISK: {na_records} non-assistant usage record(s) are excluded above. If they are in "
          f"fact billable, the total over the {len(accepted)} parsed runs rises "
          f"${base:.2f} -> ${alt:.2f} ({(alt / base - 1) if base else 0:+.1%}); scale every "
          f"projection below by that factor.")
    if without_runs and paper_o:
        mo, mp = _median(vol_o), _median(paper_o)
        uf = f"{mo / mp:.1f}x" if mp else "undefined (paper metric is zero on these traces)"
        P(f"  paper-metric per without-skill run : median {fmt(mp, '{:,.0f}')}"
          f"  (undercount factor {uf})")
    P("")
    P("  PROJECTED COST OF THE SONNET NULL BASELINE -- no-skill runs ONLY:")
    if not cost_o:
        P("    CANNOT PRICE: zero without-skill runs were parsed. The null baseline is a")
        P("    no-skill design; pricing it off with-skill runs would mis-state the bill in an")
        P("    unknown direction. Fix the trace selection and re-run.")
    else:
        for label, n in [("17 tasks x 2 runs", 34), ("17 tasks x 3 runs", 51),
                         ("49 tasks x 2 runs", 98), ("49 tasks x 3 runs", 147)]:
            P(f"    {label:20} = {n:3d} runs : median-based ${_median(cost_o)*n:8.2f}"
              f"   mean-based ${_mean(cost_o)*n:8.2f}   worst ${_max(cost_o)*n:9.2f}")
        P(f"    (priced from {len(cost_o)} WITHOUT-skill runs; with-skill runs excluded by design)")
    P("")
    P("  FOR REFERENCE -- r=3 regeneration of BOTH conditions (49 tasks x 3 runs x 2 arms = 294 runs):")
    if cost_o and cost_w:
        P(f"    without-skill leg 147 runs : median-based ${_median(cost_o)*147:8.2f}"
          f"   mean-based ${_mean(cost_o)*147:8.2f}")
        P(f"    with-skill    leg 147 runs : median-based ${_median(cost_w)*147:8.2f}"
          f"   mean-based ${_mean(cost_w)*147:8.2f}")
        P(f"    TOTAL                      : median-based "
          f"${(_median(cost_o) + _median(cost_w))*147:8.2f}"
          f"   mean-based ${(_mean(cost_o) + _mean(cost_w))*147:8.2f}")
    else:
        P("    CANNOT PRICE: need both conditions parsed to price a two-arm design.")
    P("  (add ~30% for retries/failures; buy the next round number up)")
    if not gate_pass:
        P("")
        P("  NOTE: these dollar figures are raw token volume and do not depend on the metric")
        P("  replication, but they DO depend on --traces pointing at the right tree, which the")
        P("  FAILED gate calls into question. Treat as provisional.")
    P("=" * 78)
    P(f"STATUS: {'PASS' if gate_pass else 'FAIL'}"
      + ("" if gate_pass else "  (exit code 1; Sec 4.5 numbers "
                              + ("marked UNVALIDATED)" if a.allow_unvalidated else "withheld)")))
    P("=" * 78)

    txt = "\n".join(L)
    print(txt)
    Path(a.out + ".txt").write_text(txt)
    Path(a.out + ".json").write_text(json.dumps({
        "status": "PASS" if gate_pass else "FAIL",
        "gate": {"pass": gate_pass, "failure_reasons": reasons,
                 "repro_rate": rate, "min_repro_rate": a.min_repro_rate,
                 "coverage": coverage, "min_coverage": a.min_coverage,
                 "metadata_coverage": md_coverage,
                 "min_metadata_coverage": a.min_metadata_coverage,
                 "tolerance": a.repro_tol, **cc},
        "prices": {"model": a.model, "tier": a.price_tier, "usd_per_1m_tokens": price,
                   "overridden_on_cli": price_overridden, "note": PRICE_SOURCE_NOTE},
        "files": {"found": len(files), "matched": len(accepted),
                  "skipped": skipped, "strict_regex_would_have_dropped": strict_only,
                  "parser_task_id_disagreements": disagree,
                  "bad_json_lines": tot_bad_json,
                  "files_with_bad_json": sorted(set(files_with_bad_json))},
        "records": {"usage_records": tot_usage_records,
                    "nonassistant_usage_records": na_records,
                    "nonassistant_types": dict(na_types),
                    "nonassistant_tokens": na_tokens,
                    "all_record_tokens": all_tokens},
        "tasks": {"seen": len(runs), "paired": len(rows),
                  "with_skill_only": only_with, "without_skill_only": only_without},
        "per_task": rows,
        "q1_record_set_fixed": {
            "paper_mean_ratio": _mean(pr), "paper_median_ratio": _median(pr),
            "full_mean_ratio": _mean(fr), "full_median_ratio": _median(fr),
            "allrecord_mean_ratio_sensitivity": _mean(ar),
            "allrecord_median_ratio_sensitivity": _median(ar),
            "mean_cache_share_with_skill": _mean(cs),
        },
        "q2_corrected_sec45": (
            {"mean_overhead": _mean(fr), "median_overhead": _median(fr),
             "max_overhead": _max(fr), "validated": gate_pass}
            if show_q2 else
            {"withheld": True, "reason": reasons}),
        "q3_cost": {
            "n_runs_with_skill": len(with_runs), "n_runs_without_skill": len(without_runs),
            "without_skill_median_cost_usd": _median(cost_o),
            "without_skill_mean_cost_usd": _mean(cost_o),
            "without_skill_median_tokens": _median(vol_o),
            "with_skill_median_cost_usd": _median(cost_w),
            "with_skill_mean_cost_usd": _mean(cost_w),
            "with_skill_median_tokens": _median(vol_w),
            "null_baseline_49x3_median_usd": (_median(cost_o) * 147) if cost_o else None,
            "both_arms_49x3_median_usd": ((_median(cost_o) + _median(cost_w)) * 147)
            if (cost_o and cost_w) else None,
        },
    }, indent=2))
    print(f"\nwrote {a.out}.txt and {a.out}.json")
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
