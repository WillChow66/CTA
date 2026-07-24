#!/usr/bin/env python3
"""G1+ Semantic Validation: conservation, alternation, and vocabulary checks.

Goes beyond the structural "zero None" gate (G1) to verify semantic correctness
of the Hermes→CTA adapter mapping.

Checks:
  0. Degenerate detection: classify empty sessions (infra failure, not adapter bug)
  1. Conservation: every tool_call has a matching tool response (no orphans)
  2. Alternation: assistant→tool message ordering is well-formed
  3. Vocabulary coverage: all tool names are in the explicit map
  4. CTA mapping: adapter produces zero None-typed events

Verdicts:
  PASS  — all checks pass, session is valid for analysis
  SKIP  — session is degenerate (empty due to infra failure); excluded from analysis
  FAIL  — adapter bug detected (None events, orphans, unmapped tools)

Usage:
    python scripts/validate_g1_plus.py <session.json|state.db> [--verbose]
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cta.hermes_adapter import (
    HERMES_TOOL_MAP,
    hermes_session_to_trace,
)


def load_session(path: str) -> Dict[str, Any]:
    """Load a session from JSON file or SQLite state.db."""
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    if p.suffix == ".db" or p.name == "state.db":
        conn = sqlite3.connect(str(p))
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT id, model, end_reason FROM sessions LIMIT 1").fetchone()
        if not row:
            conn.close()
            raise ValueError("No sessions found in database")
        session_id = row["id"]
        rows = conn.execute(
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

        return {
            "session_id": session_id,
            "model": row["model"],
            "end_reason": row["end_reason"],
            "messages": messages,
        }
    raise ValueError(f"Unsupported file type: {p.suffix}")


def check_degenerate(session: Dict[str, Any]) -> Dict[str, Any]:
    """Classify whether a session is degenerate (empty due to infra failure).

    Guardrail: distinguishes "nothing to translate" from "translator failed."
    Degenerate sessions are SKIPped, not FAILed — they represent infrastructure
    problems (credit exhaustion, timeout, crash), not adapter bugs.

    Classifications:
      - no_model_response: 0 assistant messages (API never responded: 402, timeout, crash)
      - no_actions: assistant responded but produced 0 tool calls (trivial Q&A or refusal)
      - not_degenerate: session has tool calls; proceed with full validation
    """
    messages = session.get("messages", [])
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    has_tool_calls = any(
        m.get("tool_calls") for m in assistant_msgs
    )

    if not assistant_msgs:
        classification = "no_model_response"
        cause = session.get("end_reason") or "unknown (likely credit exhaustion or API failure)"
    elif not has_tool_calls:
        classification = "no_actions"
        cause = "model responded but took no actions"
    else:
        classification = "not_degenerate"
        cause = ""

    return {
        "check": "degenerate_detection",
        "degenerate": classification != "not_degenerate",
        "classification": classification,
        "cause": cause,
        "assistant_messages": len(assistant_msgs),
        "has_tool_calls": has_tool_calls,
    }


def check_conservation(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Every tool_call must have a matching tool response."""
    sent_ids = set()
    received_ids = set()

    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                cid = tc.get("id") or tc.get("call_id") or ""
                if cid:
                    sent_ids.add(cid)
        elif msg.get("role") == "tool":
            cid = msg.get("tool_call_id", "")
            if cid:
                received_ids.add(cid)

    orphaned = sent_ids - received_ids
    return {
        "check": "conservation",
        "passed": len(orphaned) == 0,
        "sent": len(sent_ids),
        "received": len(received_ids),
        "orphaned": sorted(orphaned),
    }


def check_alternation(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check assistant→tool ordering. Adjacent assistant messages are a known
    Hermes pattern (reasoning + tool-call in same API response), not a failure."""
    violations = []
    prev_role = None

    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "tool" and prev_role == "tool":
            pass  # consecutive tool responses are fine (parallel calls)
        elif role == "tool" and prev_role not in ("assistant", "tool", None):
            violations.append({"index": i, "role": role, "prev_role": prev_role})
        prev_role = role

    assistant_adjacencies = 0
    prev_role = None
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and prev_role == "assistant":
            assistant_adjacencies += 1
        prev_role = role

    return {
        "check": "alternation",
        "passed": len(violations) == 0,
        "ordering_violations": violations,
        "assistant_adjacencies": assistant_adjacencies,
        "note": "assistant→assistant adjacencies are expected Hermes pattern (reasoning + action)",
    }


def check_vocabulary(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """All tool names must be in the explicit map."""
    seen_tools = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {}) or {}
                name = fn.get("name", "")
                if name:
                    seen_tools.add(name)

    unmapped = seen_tools - set(HERMES_TOOL_MAP.keys())
    return {
        "check": "vocabulary",
        "passed": len(unmapped) == 0,
        "tools_seen": sorted(seen_tools),
        "unmapped": sorted(unmapped),
        "coverage": f"{len(seen_tools - unmapped)}/{len(seen_tools)}",
    }


def check_cta_mapping(session: Dict[str, Any]) -> Dict[str, Any]:
    """Adapter must produce zero None-typed events.

    Guardrail: separates adapter failure (none_events > 0) from degenerate
    input (event_count == 0 with none_events == 0). A degenerate session
    produces 0 events because there were 0 actions — the adapter is faithfully
    translating nothing into nothing. That's not a bug.
    """
    trace, report = hermes_session_to_trace(session, with_skill=False)
    adapter_failed = report["none_events"] > 0
    return {
        "check": "cta_mapping",
        "passed": not adapter_failed,
        "adapter_failure": adapter_failed,
        "event_count": report["event_count"],
        "none_events": report["none_events"],
        "unmapped_tools": report["unmapped_tools"],
        "type_counts": report["type_counts"],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="G1+ Semantic Validation")
    parser.add_argument("session", help="Path to session JSON or state.db")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    session = load_session(args.session)
    messages = session.get("messages", [])

    print(f"G1+ Semantic Validation")
    print(f"{'='*50}")
    print(f"Session: {session.get('session_id', 'unknown')}")
    print(f"Messages: {len(messages)}")
    print()

    # Guardrail: detect degenerate sessions before running adapter checks
    degenerate = check_degenerate(session)
    if degenerate["degenerate"]:
        print(f"  [SKIP] {degenerate['check']}")
        print(f"         classification: {degenerate['classification']}")
        print(f"         cause: {degenerate['cause']}")
        print(f"         assistant_messages: {degenerate['assistant_messages']}")
        print()
        print(f"{'='*50}")
        print(f"Verdict: G1+ SKIPPED (degenerate session — not an adapter failure)")
        print(f"         Exclude from aggregate metrics; do not flow into SIP analysis.")
        return 0

    checks = [
        check_conservation(messages),
        check_alternation(messages),
        check_vocabulary(messages),
        check_cta_mapping(session),
    ]

    all_pass = True
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        if not c["passed"]:
            all_pass = False
        print(f"  [{status}] {c['check']}")
        if args.verbose or not c["passed"]:
            for k, v in c.items():
                if k not in ("check", "passed"):
                    print(f"         {k}: {v}")
        print()

    verdict = "G1+ PASSED" if all_pass else "G1+ FAILED"
    print(f"{'='*50}")
    print(f"Verdict: {verdict}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
