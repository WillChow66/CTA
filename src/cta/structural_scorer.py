"""Standalone structural metric scorer for counterfactual trace pairs.

Usage:
    python -m cta.structural_scorer treatment.db baseline.db
    python -m cta.structural_scorer treatment.db baseline.db --output metrics.json
    python -m cta.structural_scorer data/m3_captures/ --pair-by-task
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .data_models import Trace
from .hermes_adapter import hermes_session_to_trace
from .structural_metrics import StructuralComparison, compare


def load_session(db_path: Path) -> Dict[str, Any]:
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


def score_pair(treatment_db: Path, baseline_db: Path) -> Dict[str, Any]:
    t_session = load_session(treatment_db)
    b_session = load_session(baseline_db)

    if not t_session or not b_session:
        return {"error": "empty session", "treatment": str(treatment_db), "baseline": str(baseline_db)}

    t_trace, t_report = hermes_session_to_trace(t_session, with_skill=True)
    b_trace, b_report = hermes_session_to_trace(b_session, with_skill=False)

    comparison = compare(t_trace, b_trace)

    return {
        "treatment": str(treatment_db),
        "baseline": str(baseline_db),
        "treatment_model": t_session.get("model", ""),
        "baseline_model": b_session.get("model", ""),
        "treatment_messages": len(t_session.get("messages", [])),
        "baseline_messages": len(b_session.get("messages", [])),
        "metrics": asdict(comparison),
    }


def discover_pairs(captures_dir: Path) -> List[Tuple[Path, Path]]:
    """Auto-discover treatment/baseline pairs by task prefix in a captures directory."""
    pairs = []
    dirs = sorted(d for d in captures_dir.iterdir() if d.is_dir())
    treatment_dirs = [d for d in dirs if "treatment" in d.name and (d / "state.db").exists()]

    for t_dir in treatment_dirs:
        b_name = t_dir.name.replace("treatment", "baseline")
        b_dir = captures_dir / b_name
        if b_dir.exists() and (b_dir / "state.db").exists():
            pairs.append((t_dir / "state.db", b_dir / "state.db"))

    return pairs


def score_batch(treatment_dirs: List[Path], baseline_dirs: List[Path]) -> Dict[str, Any]:
    """Score all treatment/baseline pairs by directory name matching.

    Expected by run_audit.py. Returns dict with 'pairs' and 'aggregate' keys.
    """
    b_by_name = {d.name.replace("baseline", "treatment"): d for d in baseline_dirs}
    results = []
    for t_dir in treatment_dirs:
        b_dir = b_by_name.get(t_dir.name)
        if b_dir and (t_dir / "state.db").exists() and (b_dir / "state.db").exists():
            results.append(score_pair(t_dir / "state.db", b_dir / "state.db"))

    valid = [r for r in results if "error" not in r]
    output: Dict[str, Any] = {"pairs": results, "valid_pairs": len(valid)}

    if valid:
        ecrs = [r["metrics"]["event_count_ratio"] for r in valid]
        wcs = [r["metrics"]["write_compression"] for r in valid]
        entropies = [r["metrics"]["entropy_ratio"] for r in valid]
        output["aggregate"] = {
            "valid_pairs": len(valid),
            "event_count_ratio": {
                "mean": sum(ecrs) / len(ecrs),
                "min": min(ecrs),
                "max": max(ecrs),
            },
            "write_compression": {
                "mean": sum(wcs) / len(wcs),
                "min": min(wcs),
                "max": max(wcs),
            },
            "entropy_ratio": {
                "mean": sum(entropies) / len(entropies),
            },
        }

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Structural metric scorer for counterfactual trace pairs"
    )
    parser.add_argument("paths", nargs="*", help="treatment.db baseline.db (or captures dir with --pair-by-task)")
    parser.add_argument("--output", "-o", type=str, default="", help="Write JSON to file")
    parser.add_argument("--pair-by-task", action="store_true",
                        help="Auto-discover pairs in a captures directory")
    args = parser.parse_args()

    results: List[Dict[str, Any]] = []

    if args.pair_by_task:
        if len(args.paths) != 1:
            parser.error("--pair-by-task requires exactly one captures directory")
        captures_dir = Path(args.paths[0])
        if not captures_dir.is_dir():
            parser.error(f"Not a directory: {captures_dir}")
        pairs = discover_pairs(captures_dir)
        if not pairs:
            print(f"No treatment/baseline pairs found in {captures_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Discovered {len(pairs)} pair(s) in {captures_dir}", file=sys.stderr)
        for t_db, b_db in pairs:
            results.append(score_pair(t_db, b_db))
    else:
        if len(args.paths) != 2:
            parser.error("Provide exactly 2 paths: treatment.db baseline.db")
        t_db, b_db = Path(args.paths[0]), Path(args.paths[1])
        for p in (t_db, b_db):
            if not p.exists():
                parser.error(f"Not found: {p}")
        results.append(score_pair(t_db, b_db))

    output = {"pairs": results, "count": len(results)}
    if len(results) > 1:
        valid = [r for r in results if "error" not in r]
        if valid:
            ecrs = [r["metrics"]["event_count_ratio"] for r in valid]
            wcs = [r["metrics"]["write_compression"] for r in valid]
            output["aggregate"] = {
                "mean_ecr": sum(ecrs) / len(ecrs),
                "mean_wc": sum(wcs) / len(wcs),
                "min_ecr": min(ecrs),
                "max_ecr": max(ecrs),
            }

    rendered = json.dumps(output, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
        print(f"Written: {args.output}", file=sys.stderr)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
