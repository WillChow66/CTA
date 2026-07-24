#!/usr/bin/env python3
"""G6 One-Command Audit Runner (generalized).

Reproduces the full CTA skill audit from committed raw session data.
An external reviewer can run this without having written the skill:

    python scripts/run_audit.py --config configs/<skill>.yaml
    python scripts/run_audit.py --config configs/xurl.yaml --captures-dir data/captures

Reads:  <captures_dir>/*/state.db + result.json
Writes: data/audit_report.json
Prints: Markdown summary suitable for PR body.

No network access, no API keys, no container runtime required.
"""

import argparse
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cta.hermes_adapter import hermes_session_to_trace
from cta.data_models import EventType
from cta.audit_config import AuditConfig, load_config
from cta.skill_rules import run_detectors


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_session(db_path: Path) -> dict:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    row = cur.execute("SELECT id, model FROM sessions LIMIT 1").fetchone()
    if not row:
        conn.close()
        return {}
    session_id = row["id"]
    rows = cur.execute(
        """SELECT id, role, content, tool_call_id, tool_calls, tool_name,
                  reasoning, reasoning_content, finish_reason
           FROM messages WHERE session_id=? AND active=1 AND compacted=0 ORDER BY id""",
        (session_id,),
    ).fetchall()
    conn.close()

    messages = []
    for r in rows:
        msg = {"role": r["role"], "content": r["content"] or "", "tool_call_id": r["tool_call_id"]}
        if r["role"] == "assistant":
            msg["reasoning"] = r["reasoning"] or r["reasoning_content"] or ""
            msg["tool_calls"] = json.loads(r["tool_calls"]) if r["tool_calls"] else []
        messages.append(msg)

    return {"session_id": session_id, "model": row["model"], "messages": messages}


def extract_tool_calls(messages: list) -> list:
    tool_results = {m["tool_call_id"]: m.get("content", "") for m in messages if m["role"] == "tool"}
    calls = []
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {}) or {}
            name = fn.get("name", "") or ""
            raw = fn.get("arguments", "{}")
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            cid = tc.get("id") or tc.get("call_id") or ""
            calls.append({"name": name, "args": args, "observation": tool_results.get(cid, "")})
    return calls


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def entropy(counts: dict) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def analyze_run(run_dir: Path, config: AuditConfig) -> dict:
    db = run_dir / "state.db"
    if not db.exists():
        return {"run_id": run_dir.name, "error": "missing state.db"}

    session = load_session(db)
    if not session:
        return {"run_id": run_dir.name, "error": "empty database"}

    result_file = run_dir / "result.json"
    try:
        result = json.loads(result_file.read_text()) if result_file.exists() else {}
    except (json.JSONDecodeError, ValueError):
        result = {}

    messages = session["messages"]
    calls = extract_tool_calls(messages)

    # Derive condition from result.json (authoritative) with dir-name fallback
    condition = result.get("condition")
    if not condition:
        condition = "treatment" if "treatment" in run_dir.name else "baseline"
    with_skill = condition == "treatment"

    trace, report = hermes_session_to_trace(session, with_skill=with_skill)
    tool_counts = Counter(c["name"] for c in calls)

    tf = config.tool_filters
    dc = tf.delegation_call
    delegation_calls = [c for c in calls
                        if c["name"] == dc.tool_name and dc.command_contains in c["args"].get("command", "")]
    manual_writes = [c for c in calls if c["name"] in tf.manual_writes]
    br = tf.binary_resolution
    binary_res = [c for c in calls
                  if c["name"] == br.tool_name
                  and re.search(br.command_regex, c["args"].get("command", ""))
                  and br.command_contains in c["args"].get("command", "")]
    skill_views = [c for c in calls if c["name"] == tf.skill_view]
    pty_on_delegation = [c for c in delegation_calls if c["args"].get(tf.pty_arg) is True]
    pty_missing = [c for c in delegation_calls if not c["args"].get(tf.pty_arg)]

    # Derive task_id from result.json (authoritative) with dir-name fallback
    task_id = result.get("task_id", run_dir.name.split("-")[0])

    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "task_id": task_id,
        "condition": condition,
        "session_id": session["session_id"],
        "model": session["model"],
        "exit_code": result.get("exit_code"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "messages": len(messages),
        "tool_calls": len(calls),
        "unique_tools": len(tool_counts),
        "tool_counts": dict(tool_counts.most_common()),
        "entropy": round(entropy(tool_counts), 3),
        "manual_writes": len(manual_writes),
        "delegation_calls": len(delegation_calls),
        "skill_views": len(skill_views),
        "binary_resolution": len(binary_res),
        "pty_set": len(pty_on_delegation),
        "pty_missing": len(pty_missing),
        "adapter_events": report["event_count"],
        "adapter_unmapped": report["unmapped_tools"],
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Hypothesis evaluation (config-driven)
# ---------------------------------------------------------------------------

def evaluate_hypotheses(runs: list, config: AuditConfig) -> dict:
    T = [r for r in runs if r["condition"] == "treatment" and "error" not in r]
    B = [r for r in runs if r["condition"] == "baseline" and "error" not in r]
    pos_T = [r for r in T if r["task_id"].startswith("P")]
    pos_B = [r for r in B if r["task_id"].startswith("P")]

    avg = lambda lst, k: sum(r[k] for r in lst) / len(lst) if lst else 0

    results = {}
    for hid, hcfg in config.hypotheses.items():
        if hcfg.note and "untestable" in (hcfg.note or "").lower():
            results[f"{hid}_{hcfg.name.lower().replace(' ', '_')}"] = {
                "status": f"UNTESTABLE ({hcfg.note})",
            }
            continue

        if hcfg.metric == "tool_call_compression":
            status = "UNTESTABLE"
            ev = {}
            if pos_T and pos_B:
                t_calls, b_calls = avg(pos_T, "tool_calls"), avg(pos_B, "tool_calls")
                t_writes, b_writes = avg(pos_T, "manual_writes"), avg(pos_B, "manual_writes")
                write_comp = b_writes / max(t_writes, 0.1)
                disconfirm = hcfg.disconfirm_threshold or 1.2
                confirm = hcfg.confirm_threshold or 1.5
                if t_calls > b_calls * disconfirm:
                    status = "DISCONFIRMED"
                elif b_calls / max(t_calls, 1) > confirm:
                    status = "CONFIRMED"
                else:
                    status = "PARTIALLY CONFIRMED"
                ev = {"treatment_tool_calls": round(t_calls, 1), "baseline_tool_calls": round(b_calls, 1),
                      "write_compression": f"{write_comp:.1f}x"}
            results[f"{hid}_{hcfg.name.lower().replace(' ', '_')}"] = {"status": status, "evidence": ev}

        elif hcfg.metric == "pty_compliance":
            total_missing = sum(r["pty_missing"] for r in T)
            total_set = sum(r["pty_set"] for r in T)
            confirm = hcfg.confirm_threshold or 1.0
            if confirm >= 1.0:
                status = "CONFIRMED" if total_missing == 0 else "DISCONFIRMED"
            else:
                compliance = total_set / max(total_set + total_missing, 1)
                status = "CONFIRMED" if compliance >= confirm else "DISCONFIRMED"
            results[f"{hid}_{hcfg.name.lower().replace(' ', '_')}"] = {
                "status": status,
                "evidence": {"pty_set": total_set, "pty_missing": total_missing},
            }

        elif hcfg.metric == "binary_resolution_rate":
            with_binary = sum(1 for r in T if r["binary_resolution"] > 0)
            confirm = hcfg.confirm_threshold or 0.67
            status = "CONFIRMED" if len(T) > 0 and with_binary >= confirm * len(T) else "DISCONFIRMED"
            results[f"{hid}_{hcfg.name.lower().replace(' ', '_')}"] = {
                "status": status,
                "evidence": {"traces_with_resolution": with_binary, "total_treatments": len(T)},
            }

        elif hcfg.metric == "interactive_blockade":
            results[f"{hid}_{hcfg.name.lower().replace(' ', '_')}"] = {
                "status": f"UNTESTABLE ({hcfg.note or 'deferred'})",
            }

        else:
            results[f"{hid}_{hcfg.name.lower().replace(' ', '_')}"] = {"status": "UNTESTABLE (unknown metric)"}

    return results


# ---------------------------------------------------------------------------
# SIP detection (config-driven via registry)
# ---------------------------------------------------------------------------

def detect_sips(runs: list, config: AuditConfig) -> list:
    sips = []
    for r in runs:
        if r["condition"] != "treatment" or "error" in r:
            continue
        context = {
            "skill_views": r["skill_views"],
            "binary_resolution": r["binary_resolution"],
            "delegation_calls": r["delegation_calls"],
            "task_id": r["task_id"],
            "task_type": "negative_control" if r["task_id"].startswith("N") else "positive",
        }
        git_diff = Path(r.get("run_dir", "")) / "git_diff.txt"
        if git_diff.exists():
            context["git_diff_path"] = str(git_diff)
        events = r.get("trace").events if r.get("trace") else []
        findings = run_detectors(events, config.sip_detectors, context=context)
        # Collapse per-event findings into one entry per (type, run)
        by_type = {}
        for f in findings:
            if f.sip_type not in by_type:
                by_type[f.sip_type] = {"type": f.sip_type, "valence": f.valence, "run": r["run_id"], "count": 0}
            by_type[f.sip_type]["count"] += 1
        for entry in by_type.values():
            if entry["count"] == 1:
                del entry["count"]
            sips.append(entry)
    return sips


# ---------------------------------------------------------------------------
# Controls validation (config-driven)
# ---------------------------------------------------------------------------

def validate_controls(runs: list, config: AuditConfig) -> dict:
    results = {}
    ctrl = config.controls

    if ctrl.negative:
        tid = ctrl.negative.task_id
        t_runs = [r for r in runs if r["task_id"] == tid and r["condition"] == "treatment" and "error" not in r]
        if ctrl.negative.pass_criteria == "zero_delegation_calls":
            results[f"{tid}_zero_delegation"] = all(r["delegation_calls"] == 0 for r in t_runs) if t_runs else None
        else:
            results[f"{tid}_pass"] = None

    if ctrl.edge_case:
        tid = ctrl.edge_case.task_id
        t_runs = [r for r in runs if r["task_id"] == tid and r["condition"] == "treatment" and "error" not in r]
        if ctrl.edge_case.pass_criteria == "zero_writes":
            results[f"{tid}_zero_writes"] = all(r["manual_writes"] == 0 for r in t_runs) if t_runs else None
        else:
            results[f"{tid}_pass"] = None

    # Meta-check: metric not trivially constructive (only if controls were actually validated)
    neg_results = [v for k, v in results.items() if "delegation" in k and v is not None]
    edge_results = [v for k, v in results.items() if "writes" in k and v is not None]
    if neg_results or edge_results:
        results["metric_not_trivially_constructive"] = all(neg_results) and all(edge_results)

    return results


# ---------------------------------------------------------------------------
# Markdown report (config-driven)
# ---------------------------------------------------------------------------

def markdown_report(runs: list, hypotheses: dict, sips: list, controls: dict, config: AuditConfig) -> str:
    lines = []
    title = config.report.title.format(skill_name=config.skill_name)
    lines.append(f"## {title} (M2 Counterfactual Evidence)")
    lines.append("")
    lines.append(f"**Sessions:** {len([r for r in runs if 'error' not in r])} | "
                 f"**Design:** {config.report.design} | "
                 f"**Model:** {config.report.model}")
    lines.append("")

    # Hypotheses
    lines.append("### Pre-Registered Hypotheses")
    lines.append("")
    lines.append("| # | Hypothesis | Verdict |")
    lines.append("|---|---|---|")
    for hid, h in hypotheses.items():
        lines.append(f"| {hid.split('_')[0]} | {hid.split('_', 1)[1].replace('_', ' ').title()} | **{h['status']}** |")
    lines.append("")

    # Structural comparison
    lines.append("### Structural Comparison (Treatment vs Baseline)")
    lines.append("")
    lines.append("| Task | T msgs | B msgs | T tools | B tools | T writes | B writes | Compression |")
    lines.append("|------|--------|--------|---------|---------|----------|----------|-------------|")

    by_task = defaultdict(lambda: {"treatment": [], "baseline": []})
    for r in runs:
        if "error" not in r:
            by_task[r["task_id"]][r["condition"]].append(r)

    for task_id in sorted(by_task):
        t = by_task[task_id]["treatment"]
        b = by_task[task_id]["baseline"]
        if not t or not b:
            continue
        avg = lambda lst, k: sum(r[k] for r in lst) / len(lst)
        comp = avg(b, "tool_calls") / max(avg(t, "tool_calls"), 1)
        lines.append(f"| {task_id} | {avg(t, 'messages'):.0f} | {avg(b, 'messages'):.0f} | "
                     f"{avg(t, 'tool_calls'):.0f} | {avg(b, 'tool_calls'):.0f} | "
                     f"{avg(t, 'manual_writes'):.0f} | {avg(b, 'manual_writes'):.0f} | {comp:.2f}x |")
    lines.append("")

    # SIPs
    lines.append("### Skill Influence Patterns")
    lines.append("")
    lines.append("| SIP | Valence | Count |")
    lines.append("|-----|---------|-------|")
    sip_counts = Counter((s["type"], s["valence"]) for s in sips)
    for (stype, valence), count in sorted(sip_counts.items()):
        lines.append(f"| {stype} | {valence} | {count} |")
    lines.append("")

    # Controls
    lines.append("### Controls")
    lines.append("")
    for key, val in controls.items():
        label = key.replace("_", " ").title()
        status = "PASS" if val else "FAIL" if val is False else "N/A"
        lines.append(f"- {label}: **{status}**")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="G6 CTA Audit Runner")
    parser.add_argument("--config", type=Path, required=True,
                        help="Per-skill YAML/JSON config (e.g., configs/xurl.yaml)")
    parser.add_argument("--captures-dir", type=Path, default=None,
                        help="Override captures directory from config")
    parser.add_argument("--structural", action="store_true",
                        help="Compute structural metric matrix (treatment×baseline pairs)")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config, captures_dir_override=args.captures_dir)

    captures_dir = ROOT / config.captures_dir if not config.captures_dir.is_absolute() else config.captures_dir
    if not captures_dir.exists():
        print(f"ERROR: {captures_dir} not found. Run capture harness first.")
        sys.exit(1)

    run_dirs = sorted(d for d in captures_dir.iterdir() if d.is_dir())
    print(f"G6 Audit Runner — {len(run_dirs)} sessions from {captures_dir.name}/\n")

    runs = [analyze_run(d, config) for d in run_dirs]
    valid = [r for r in runs if "error" not in r]
    errors = [r for r in runs if "error" in r]

    if errors:
        print(f"WARNING: {len(errors)} sessions failed to load:")
        for e in errors:
            print(f"  {e['run_id']}: {e['error']}")

    hypotheses = evaluate_hypotheses(valid, config)
    sips = detect_sips(valid, config)
    controls = validate_controls(valid, config)

    # Strip non-serializable trace objects before writing JSON
    serializable_runs = []
    for r in runs:
        sr = {k: v for k, v in r.items() if k not in ("trace", "run_dir")}
        serializable_runs.append(sr)

    report = {
        "audit_version": "2.0.0",
        "skill": config.skill_name,
        "skill_version": config.skill_version,
        "design": config.report.design,
        "sessions": serializable_runs,
        "hypotheses": hypotheses,
        "sips": sips,
        "controls": controls,
    }

    if args.structural or "structural_scorer" in config.metrics:
        from cta.structural_scorer import score_batch
        t_dirs = [d for d in run_dirs if "treatment" in d.name and (d / "state.db").exists()]
        b_dirs = [d for d in run_dirs if "baseline" in d.name and (d / "state.db").exists()]
        if t_dirs and b_dirs:
            structural = score_batch(t_dirs, b_dirs)
            report["structural_metrics"] = structural
            agg = structural.get("aggregate")
            if agg:
                print(f"Structural metrics: {agg['valid_pairs']} pairs | "
                      f"ECR={agg['event_count_ratio']['mean']:.3f} | "
                      f"WC={agg['write_compression']['mean']:.1f}x")

    output_path = ROOT / "data" / "audit_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Report written: {output_path}\n")

    md = markdown_report(valid, hypotheses, sips, controls, config)
    md_path = ROOT / "data" / "audit_report.md"
    md_path.write_text(md)
    print(f"Markdown written: {md_path}\n")
    print(md)


if __name__ == "__main__":
    main()
