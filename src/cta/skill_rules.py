"""SIP detector registry and example detectors.

Skill Influence Patterns (SIPs) are behavioral signatures that a skill leaves
on agent traces. This module provides the detector dispatch interface and
example detectors that are general (not skill-specific).

To add a detector:
  1. Write a function: detect_my_sip(events, context=None) -> List[SIPFinding]
  2. Register it: DETECTOR_REGISTRY["my_sip"] = detect_my_sip
  3. Reference by name in your skill config's sip_detectors list

Detectors are isolated (try/except) — one failure doesn't crash the pipeline.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from .data_models import Event, EventType


@dataclass
class SIPFinding:
    """A single SIP detection result."""
    sip_type: str
    valence: str  # constructive | neutral | destructive
    event_id: int
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)


def detect_concept_bleed(events: List[Event], context: Dict[str, Any] | None = None) -> List[SIPFinding]:
    """Flag skill tool usage on negative-control tasks where it should NOT trigger."""
    if not context:
        return []
    if context.get("task_type") == "negative_control" and context.get("delegation_calls", 0) > 0:
        return [SIPFinding(
            sip_type="CONCEPT_BLEED",
            valence="destructive",
            event_id=0,
            description="Skill tool invoked on negative-control task (scope leak)",
            evidence={"task_id": context.get("task_id", ""), "delegation_calls": context["delegation_calls"]},
        )]
    return []


def detect_false_success(events: List[Event], context: Dict[str, Any] | None = None) -> List[SIPFinding]:
    """Flag cases where the agent reports success despite tool errors in the trace.

    Recovery-aware: if the agent acknowledges the error (fallback language,
    retry, explicit failure mention), it is NOT flagged.
    """
    import re

    findings = []
    error_events = [
        e for e in events
        if e.type == EventType.EXECUTE and e.outcome is not None and e.outcome.value == "error"
    ]
    if not error_events:
        return []

    final_reasons = [
        e.content for e in events
        if e.type == EventType.REASON and e.content
    ]
    if not final_reasons:
        return []

    last_reason = final_reasons[-1]
    acknowledgment = re.search(
        r"fell back|fall.?back|manual(?:ly)?|failed|failure|unable|could not|"
        r"permission.{0,20}(denied|block|error)|timed? ?out|timeout|stuck|"
        r"not logged in|error.{0,20}(occur|encounter|happen)",
        last_reason,
        re.IGNORECASE,
    )
    if acknowledgment:
        return []

    success_claim = re.search(
        r"(?:successfully|completed?|done|finished?|created?|implemented?|added?)",
        last_reason,
        re.IGNORECASE,
    )
    if success_claim:
        findings.append(SIPFinding(
            sip_type="FALSE_SUCCESS",
            valence="destructive",
            event_id=error_events[0].event_id,
            description="Agent claims success despite tool errors in trace",
            evidence={
                "error_count": len(error_events),
                "claim_snippet": last_reason[:200],
            },
        ))
    return findings


DETECTOR_REGISTRY: Dict[str, Callable] = {
    "concept_bleed": detect_concept_bleed,
    "false_success": detect_false_success,
}


def run_detectors(
    events: List[Event],
    detector_names: List[str],
    context: Dict[str, Any] | None = None,
) -> List[SIPFinding]:
    """Run named detectors from the registry. Unknown names are skipped."""
    findings = []
    for name in detector_names:
        fn = DETECTOR_REGISTRY.get(name)
        if fn is None:
            continue
        try:
            sig = inspect.signature(fn)
            if "context" in sig.parameters:
                findings.extend(fn(events, context=context))
            else:
                findings.extend(fn(events))
        except Exception:
            continue
    return findings
