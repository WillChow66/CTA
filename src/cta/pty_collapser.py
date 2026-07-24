"""PTYCollapser — preprocessing pass for interactive-mode traces.

Operates on raw Hermes messages (preserving full tool_call arguments) to
collapse scattered process(poll/write/log/kill) calls into composite EXECUTE
events representing coherent interactive PTY sessions.

A PTY session starts with terminal(pty=true, background=true) and ends with
process(action="kill") or trace end. All process() calls matching the
session_id are collapsed into the parent EXECUTE event's content as a
structured JSON sub-trace.

This is a drop-in superset of hermes_session_to_trace for interactive traces:
it produces a Module-2-ready Trace plus PTYSession metadata. For traces
without PTY sessions, output is identical to the adapter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .data_models import Event, EventType, EventOutcome, Trace
from .hermes_adapter import (
    ToolCallRecord,
    extract_tool_records,
    map_tool,
    _infer_outcome,
)


@dataclass
class PTYSession:
    """A collapsed interactive PTY session."""
    parent_event_id: int
    command: str
    session_id: str | None = None
    actions: list[dict[str, Any]] = field(default_factory=list)
    total_polls: int = 0
    total_writes: int = 0
    total_logs: int = 0
    killed: bool = False
    trust_dialog_handled: bool = False
    child_call_ids: list[str] = field(default_factory=list)
    has_error: bool = False


def collapse_pty_sessions(
    messages: list[dict],
    with_skill: bool = False,
    trace_id: str = "unknown",
) -> tuple[Trace, list[PTYSession], dict]:
    """Collapse PTY process() scatter from raw Hermes messages.

    Returns (trace, pty_sessions, report) where trace is Module-2-ready
    with PTY regions collapsed into composite EXECUTE events.
    """
    records = extract_tool_records(messages)
    pty_sessions: list[PTYSession] = []

    # Index process records by session_id for robust child collection
    # (handles interleaved non-process calls like read_file/ls)
    process_by_session: dict[str, list[int]] = {}
    for idx, rec in enumerate(records):
        if rec.name == "process":
            sid = rec.args.get("session_id", "")
            if sid:
                process_by_session.setdefault(sid, []).append(idx)

    # Identify PTY parents and claim their children
    consumed: set[int] = set()
    parent_map: dict[int, PTYSession] = {}  # record index -> session

    for idx, rec in enumerate(records):
        if not _is_pty_parent(rec):
            continue

        session_id = _extract_session_id(rec.observation)
        session = PTYSession(
            parent_event_id=rec.event.event_id,
            command=rec.args.get("command", ""),
            session_id=session_id,
        )
        parent_map[idx] = session
        consumed.add(idx)

        # Collect children by session_id
        child_indices = process_by_session.get(session_id, []) if session_id else []
        if not child_indices:
            # Fallback: collect forward until kill or next non-process
            child_indices = _collect_forward(records, idx + 1)

        for ci in child_indices:
            if ci in consumed:
                continue
            consumed.add(ci)
            child_rec = records[ci]
            action_info = _process_action(child_rec)
            session.actions.append(action_info)
            session.child_call_ids.append(child_rec.call_id)

            action = action_info["action"]
            if action == "poll" or action == "wait":
                session.total_polls += 1
            elif action == "write":
                session.total_writes += 1
                if _is_trust_response(action_info):
                    session.trust_dialog_handled = True
            elif action == "log":
                session.total_logs += 1
            elif action == "kill":
                session.killed = True

            if _infer_outcome(child_rec.observation) == EventOutcome.FAILURE:
                session.has_error = True

        pty_sessions.append(session)

    # Build output events: REASON turns + tool records (collapsed or passthrough)
    events: list[Event] = []
    event_id = 0
    tool_counts: dict[str, int] = {}
    unmapped: set[str] = set()

    # Emit REASON events for pure-reasoning assistant turns
    reason_events = _extract_reason_events(messages)

    # Merge reason events and tool records in message order
    reason_iter = iter(reason_events)
    next_reason = next(reason_iter, None)

    for idx, rec in enumerate(records):
        # Emit any reason events that precede this tool record
        while next_reason is not None and next_reason[0] <= rec.event.event_id:
            ev = next_reason[1]
            ev.event_id = event_id
            events.append(ev)
            event_id += 1
            next_reason = next(reason_iter, None)

        if idx in parent_map:
            # Emit composite event for PTY parent
            session = parent_map[idx]
            composite = _build_composite_event(session, rec, event_id)
            events.append(composite)
            event_id += 1
            tool_counts[rec.name] = tool_counts.get(rec.name, 0) + 1
        elif idx in consumed:
            # Child process() call — consumed into composite, skip
            continue
        else:
            # Passthrough: emit the adapter-produced event with renumbered id
            ev = rec.event
            ev.event_id = event_id
            events.append(ev)
            event_id += 1
            tool_counts[rec.name] = tool_counts.get(rec.name, 0) + 1
            if rec.name not in _KNOWN_TOOLS:
                unmapped.add(rec.name)

    # Emit trailing reason events
    while next_reason is not None:
        ev = next_reason[1]
        ev.event_id = event_id
        events.append(ev)
        event_id += 1
        next_reason = next(reason_iter, None)

    trace = Trace(
        trace_id=trace_id,
        events=events,
        task_id="",
        with_skill=with_skill,
    )

    none_events = sum(1 for e in events if e.type is None)
    report = {
        "trace_id": trace_id,
        "event_count": len(events),
        "none_events": none_events,
        "unmapped_tools": sorted(unmapped),
        "tool_counts": dict(sorted(tool_counts.items(), key=lambda kv: -kv[1])),
        "type_counts": _type_counts(events),
        "pty_sessions": len(pty_sessions),
        "pty_collapsed_calls": sum(len(s.actions) for s in pty_sessions),
    }

    return trace, pty_sessions, report


# --------------------------------------------------------------------------- #
# PTY parent detection (from args, not observation strings)
# --------------------------------------------------------------------------- #

def _is_pty_parent(rec: ToolCallRecord) -> bool:
    """Detect a background PTY launch from tool_call arguments."""
    if rec.name != "terminal":
        return False
    args = rec.args
    return (
        args.get("pty") is True
        and args.get("background") is True
    )


def _extract_session_id(observation: str) -> str | None:
    """Extract session_id from the background launch receipt observation."""
    if not observation:
        return None
    try:
        parsed = json.loads(observation)
        if isinstance(parsed, dict):
            return parsed.get("session_id")
    except (json.JSONDecodeError, TypeError):
        pass
    return None


# --------------------------------------------------------------------------- #
# Child process() action parsing (from args directly)
# --------------------------------------------------------------------------- #

def _process_action(rec: ToolCallRecord) -> dict[str, Any]:
    """Extract action type and data from a process() tool call's arguments."""
    args = rec.args
    action = args.get("action", "unknown")
    # Normalize: model uses "submit", skill documents "write"
    if action == "submit":
        action = "write"
    return {
        "action": action,
        "data": args.get("data"),
        "session_id": args.get("session_id"),
        "observation": rec.observation,
        "call_id": rec.call_id,
    }


def _is_trust_response(action_info: dict[str, Any]) -> bool:
    """Detect if a write action is the folder-trust dialog response."""
    data = action_info.get("data", "")
    if isinstance(data, str):
        return data.strip() in ("1", "1\n", "1\\n", "y", "yes")
    return False


def _collect_forward(records: list[ToolCallRecord], start: int) -> list[int]:
    """Fallback: collect process() records forward until kill or non-process gap."""
    indices = []
    for i in range(start, len(records)):
        rec = records[i]
        if rec.name == "process":
            indices.append(i)
            if rec.args.get("action") == "kill":
                break
        elif rec.event.type == EventType.REASON:
            continue
        else:
            # Allow one non-process gap (interleaved file checks)
            # but break on two consecutive non-process calls
            if i + 1 < len(records) and records[i + 1].name != "process":
                break
    return indices


# --------------------------------------------------------------------------- #
# Composite event construction
# --------------------------------------------------------------------------- #

def _build_composite_event(
    session: PTYSession, rec: ToolCallRecord, event_id: int
) -> Event:
    """Build a single composite EXECUTE event for a collapsed PTY session."""
    sub_trace = {
        "session_id": session.session_id,
        "command": session.command,
        "total_polls": session.total_polls,
        "total_writes": session.total_writes,
        "total_logs": session.total_logs,
        "trust_dialog_handled": session.trust_dialog_handled,
        "terminated": "kill" if session.killed else "implicit",
        "actions": [
            {"action": a["action"], "data": a.get("data")}
            for a in session.actions
        ],
    }

    if session.has_error:
        outcome = EventOutcome.FAILURE
    elif session.killed or session.total_polls > 0:
        outcome = EventOutcome.SUCCESS
    else:
        outcome = EventOutcome.PARTIAL

    return Event(
        event_id=event_id,
        type=EventType.EXECUTE,
        target=rec.args.get("command", ""),
        content=json.dumps(sub_trace),
        reasoning=rec.event.reasoning,
        outcome=outcome,
    )


# --------------------------------------------------------------------------- #
# REASON event extraction (mirrors adapter's pure-reasoning turn handling)
# --------------------------------------------------------------------------- #

def _extract_reason_events(messages: list[dict]) -> list[tuple[int, Event]]:
    """Extract REASON events from pure-reasoning turns, keyed by approximate position.

    Returns list of (position_hint, Event) where position_hint is the count of
    tool calls seen before this reasoning turn (used for ordering).
    """
    results = []
    tool_call_count = 0

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            tool_call_count += len(tool_calls)
            continue

        reasoning = (msg.get("reasoning") or "").strip()
        content = (msg.get("content") or "").strip()
        text = reasoning or content
        if text:
            results.append((tool_call_count, Event(
                event_id=0,  # renumbered later
                type=EventType.REASON,
                target="assistant",
                content=content,
                reasoning=reasoning,
                outcome=EventOutcome.SUCCESS,
            )))

    return results


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_KNOWN_TOOLS = {
    "read_file", "write_file", "patch", "replace",
    "terminal", "execute_code", "command",
    "search_files", "session_search", "web_search",
    "skill_view", "skills_list", "skill_manage", "skill_patch",
    "memory", "clarify", "todo",
    "web_extract", "browser_navigate", "browser_console", "browser_snapshot",
    "delegate_task", "process", "mcp_json_read_resource",
    "mcp_honcho_get_queue_status",
}


def _type_counts(events: list[Event]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in events:
        key = e.type.value if e.type is not None else "NONE"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------- #
# H3 hypothesis evaluation (unchanged interface)
# --------------------------------------------------------------------------- #

def h3_verdict(pty_sessions: list[PTYSession]) -> dict[str, Any]:
    """Evaluate H3 (Interactive Blockade Resolution) from collapsed sessions.

    H3: Hermes detects folder trust prompt and sends `1\\n`.
    Disconfirmation: >=5 consecutive polls without resolution.
    """
    if not pty_sessions:
        return {"status": "UNTESTABLE", "reason": "No PTY sessions found in trace"}

    results = []
    for s in pty_sessions:
        blockade = s.total_polls >= 5 and s.total_writes == 0
        results.append({
            "session_id": s.session_id,
            "trust_dialog_handled": s.trust_dialog_handled,
            "total_polls": s.total_polls,
            "total_writes": s.total_writes,
            "blockade_detected": blockade,
            "killed": s.killed,
        })

    any_handled = any(r["trust_dialog_handled"] for r in results)
    any_blockade = any(r["blockade_detected"] for r in results)

    if any_blockade:
        status = "DISCONFIRMED"
    elif any_handled:
        status = "CONFIRMED"
    else:
        status = "INCONCLUSIVE (trust dialog may not have appeared)"

    return {"status": status, "sessions": results}
