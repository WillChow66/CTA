"""
Hermes -> CTA trace adapter (G1 probe).

Converts a Hermes CLI session JSON document (~/.hermes/sessions/session_*.json)
into a CTA ``Trace`` of ``Event`` records so the WillChow66/CTA alignment and
rule-based SIP detectors can operate on Hermes executions.

Scope: G1 only -- prove the Hermes tool envelope flattens cleanly onto CTA's
``EventType`` enum with zero unmapped tool types. No trained classifiers
(G3); this module is pure structural translation.

Hermes session shape (verified):
    {session_id, model, system_prompt, tools, messages[]}
    assistant msg: {role, content, reasoning, finish_reason, tool_calls[]}
        tool_call: {id, type:"function", function:{name, arguments}}
    tool msg:      {role:"tool", content, tool_call_id}

CTA target (src/cta/data_models.py):
    EventType: READ, WRITE, EXECUTE, SEARCH, REASON, ERROR, TOOL_CALL
    Event(event_id, type, target, content, reasoning, outcome, ...)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .data_models import Event, EventType, EventOutcome, Trace


# --------------------------------------------------------------------------- #
# Vocabulary normalization: Hermes tool name -> CTA EventType
# --------------------------------------------------------------------------- #
# Every tool observed across 120 recent sessions is mapped explicitly. Anything
# unseen lands on TOOL_CALL (recorded by the validator), never a silent None.

HERMES_TOOL_MAP: Dict[str, EventType] = {
    # File I/O
    "read_file": EventType.READ,
    "write_file": EventType.WRITE,
    "patch": EventType.WRITE,
    "replace": EventType.WRITE,
    # Execution
    "terminal": EventType.EXECUTE,
    "execute_code": EventType.EXECUTE,
    "command": EventType.EXECUTE,
    # Search
    "search_files": EventType.SEARCH,
    # Reasoning / context injection
    "skill_view": EventType.REASON,
    "skills_list": EventType.REASON,
    "skill_manage": EventType.REASON,
    "skill_patch": EventType.REASON,
    "memory": EventType.REASON,
    "session_search": EventType.SEARCH,
    "clarify": EventType.REASON,
    "todo": EventType.REASON,
    # Web / browser
    "web_search": EventType.SEARCH,
    "web_extract": EventType.READ,
    "browser_navigate": EventType.TOOL_CALL,
    "browser_console": EventType.TOOL_CALL,
    "browser_snapshot": EventType.TOOL_CALL,
    # Orchestration / external
    "delegate_task": EventType.TOOL_CALL,
    "process": EventType.TOOL_CALL,
    "mcp_json_read_resource": EventType.READ,
    "mcp_honcho_get_queue_status": EventType.TOOL_CALL,
}


def map_tool(name: str) -> EventType:
    """Map a Hermes tool name to a CTA EventType. Never returns None."""
    return HERMES_TOOL_MAP.get(name or "", EventType.TOOL_CALL)


def _is_unmapped(name: str) -> bool:
    return (name or "") not in HERMES_TOOL_MAP


# --------------------------------------------------------------------------- #
# Argument / target extraction
# --------------------------------------------------------------------------- #

def _parse_arguments(raw: Any) -> Dict[str, Any]:
    """tool_call arguments may be a JSON string or already a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_raw": parsed}
        except (json.JSONDecodeError, ValueError):
            return {"_raw": raw}
    return {}


def _target_for(name: str, args: Dict[str, Any]) -> str:
    """Pick the most meaningful 'target' field for a CTA Event."""
    for key in ("path", "file_path", "command", "query", "pattern", "url", "name"):
        val = args.get(key)
        if isinstance(val, str) and val:
            return val
    if name == "execute_code":
        code = args.get("code") or args.get("_raw") or ""
        if isinstance(code, str) and code:
            return extract_synthetic_target(code)
    return name


def extract_synthetic_target(code_string: str) -> str:
    """Extract file paths or module names from execute_code arguments.

    Provides a meaningful target for EXECUTE events whose raw target would
    otherwise fall back to the tool name, breaking TF-IDF intent matching.
    """
    targets = []
    for match in re.finditer(r'(?:from|import)\s+([\w.]+)', code_string):
        targets.append(match.group(1))
    for match in re.finditer(r'(?:open|Path)\s*\(\s*["\']([^"\']+)["\']', code_string):
        targets.append(match.group(1))
    for match in re.finditer(r'["\']([\w/.-]+\.\w{1,4})["\']', code_string):
        targets.append(match.group(1))

    if targets:
        return ",".join(sorted(set(targets))[:3])
    return "execute_code"


# --------------------------------------------------------------------------- #
# Outcome heuristic (rule-based, G3-compliant)
# --------------------------------------------------------------------------- #

_ERROR_MARKERS = (
    "traceback (most recent call last)",
    "error:",
    "command failed",
    "no such file",
    "permission denied",
    "exception",
)


def _infer_outcome(observation: str) -> EventOutcome:
    low = (observation or "").lower()
    if any(marker in low for marker in _ERROR_MARKERS):
        return EventOutcome.FAILURE
    return EventOutcome.SUCCESS


# --------------------------------------------------------------------------- #
# Core flattening: session messages -> ordered Events
# --------------------------------------------------------------------------- #

def _index_tool_results(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map tool_call_id -> tool result content for observation pairing."""
    results: Dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "tool":
            cid = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            if isinstance(content, list):  # tolerate block form
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            results[cid] = content or ""
    return results


def hermes_session_to_trace(
    session: Dict[str, Any],
    with_skill: bool = False,
) -> Tuple[Trace, Dict[str, Any]]:
    """Flatten a Hermes session into a CTA Trace.

    Returns (trace, report) where report carries G1 gate metrics:
        unmapped_tools: sorted list of tool names not in the explicit map
        tool_counts:    per-tool occurrence counts
        none_events:    count of events with a None type (must be 0)
    """
    messages = session.get("messages", [])
    tool_results = _index_tool_results(messages)

    events: List[Event] = []
    tool_counts: Dict[str, int] = {}
    unmapped: set = set()
    event_id = 0

    for msg in messages:
        role = msg.get("role")
        if role != "assistant":
            continue

        reasoning = (msg.get("reasoning") or "").strip()
        content = (msg.get("content") or "").strip()
        tool_calls = msg.get("tool_calls") or []

        # Pure-reasoning assistant turn (no tool calls): emit a REASON event so
        # phase segmentation still sees the thought.
        if not tool_calls:
            text = reasoning or content
            if text:
                events.append(Event(
                    event_id=event_id,
                    type=EventType.REASON,
                    target="assistant",
                    content=content,
                    reasoning=reasoning,
                    outcome=EventOutcome.SUCCESS,
                ))
                event_id += 1
            continue

        # Action turn: attach reasoning to the first action of this turn.
        first = True
        for tc in tool_calls:
            fn = tc.get("function", {}) or {}
            name = fn.get("name", "") or ""
            args = _parse_arguments(fn.get("arguments"))
            cid = tc.get("id") or tc.get("call_id") or ""

            tool_counts[name] = tool_counts.get(name, 0) + 1
            if _is_unmapped(name):
                unmapped.add(name)

            etype = map_tool(name)
            observation = tool_results.get(cid, "")
            outcome = _infer_outcome(observation)

            events.append(Event(
                event_id=event_id,
                type=etype,
                target=_target_for(name, args),
                content=observation,
                reasoning=reasoning if first else "",
                outcome=outcome,
            ))
            event_id += 1
            first = False

    none_events = sum(1 for e in events if e.type is None)

    trace = Trace(
        trace_id=session.get("session_id", "unknown"),
        events=events,
        task_id="",
        with_skill=with_skill,
    )

    report = {
        "trace_id": trace.trace_id,
        "event_count": len(events),
        "none_events": none_events,
        "unmapped_tools": sorted(unmapped),
        "tool_counts": dict(sorted(tool_counts.items(), key=lambda kv: -kv[1])),
        "type_counts": _type_counts(events),
    }
    return trace, report


def _type_counts(events: List[Event]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in events:
        key = e.type.value if e.type is not None else "NONE"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------- #
# Full-args extraction (for PTYCollapser and other raw-message consumers)
# --------------------------------------------------------------------------- #

@dataclass
class ToolCallRecord:
    """A single tool call with full arguments preserved."""
    name: str
    args: Dict[str, Any]
    observation: str
    call_id: str
    event: Event


def extract_tool_records(messages: List[Dict[str, Any]]) -> List[ToolCallRecord]:
    """Extract tool call records with full args from raw Hermes messages.

    Unlike hermes_session_to_trace (which flattens args into a single target
    string), this preserves the complete arguments dict for each tool call.
    Pure-reasoning turns (no tool_calls) are excluded.
    """
    tool_results = _index_tool_results(messages)
    records: List[ToolCallRecord] = []
    event_id = 0

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            continue

        reasoning = (msg.get("reasoning") or "").strip()
        first = True
        for tc in tool_calls:
            fn = tc.get("function", {}) or {}
            name = fn.get("name", "") or ""
            args = _parse_arguments(fn.get("arguments"))
            cid = tc.get("id") or tc.get("call_id") or ""

            observation = tool_results.get(cid, "")
            etype = map_tool(name)
            outcome = _infer_outcome(observation)

            event = Event(
                event_id=event_id,
                type=etype,
                target=_target_for(name, args),
                content=observation,
                reasoning=reasoning if first else "",
                outcome=outcome,
            )
            records.append(ToolCallRecord(
                name=name,
                args=args,
                observation=observation,
                call_id=cid,
                event=event,
            ))
            event_id += 1
            first = False

    return records


# --------------------------------------------------------------------------- #
# Gate evaluation
# --------------------------------------------------------------------------- #

def evaluate_gate(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-session reports into a single G1 pass/fail verdict."""
    all_unmapped: set = set()
    total_none = 0
    total_events = 0
    for r in reports:
        all_unmapped.update(r["unmapped_tools"])
        total_none += r["none_events"]
        total_events += r["event_count"]

    passed = (not all_unmapped) and total_none == 0 and total_events > 0
    return {
        "passed": passed,
        "sessions": len(reports),
        "total_events": total_events,
        "total_none_events": total_none,
        "unmapped_tools": sorted(all_unmapped),
    }
