"""Structural metrics for print-mode trace comparison.

DTW alignment is semantically dubious when treatment traces are ~5 events and
baseline traces are ~50. These simpler structural metrics capture the real
signal: the skill collapses N granular operations into 1 delegation.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .data_models import Event, EventType, Trace


@dataclass
class StructuralComparison:
    """Result of comparing a treatment trace against a baseline trace."""
    event_count_ratio: float
    tool_vocabulary_entropy_treatment: float
    tool_vocabulary_entropy_baseline: float
    entropy_ratio: float
    write_compression: float
    unilateral_actions: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def event_count_ratio(treatment: Trace, baseline: Trace) -> float:
    """Treatment events / baseline events. Values <1 mean compression."""
    t = len(treatment.events)
    b = len(baseline.events)
    if b == 0:
        return float(t) if t > 0 else 1.0
    return t / b


def tool_vocabulary_entropy(events: List[Event]) -> float:
    """Shannon entropy (bits) over the distribution of event types."""
    if not events:
        return 0.0
    counts = Counter(e.type.value for e in events)
    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def write_compression(treatment: Trace, baseline: Trace) -> float:
    """Baseline WRITE events / treatment WRITE events. Higher = more offloaded."""
    t_writes = sum(1 for e in treatment.events if e.type == EventType.WRITE)
    b_writes = sum(1 for e in baseline.events if e.type == EventType.WRITE)
    if t_writes == 0:
        return float(b_writes) if b_writes > 0 else 1.0
    return b_writes / t_writes


def unilateral_actions(treatment: Trace, baseline: Trace) -> List[str]:
    """WRITE targets present in baseline but absent in treatment.

    These represent actions the baseline took that the skill made unnecessary
    (work offloaded into the delegation).
    """
    t_targets = {e.target for e in treatment.events if e.type == EventType.WRITE}
    b_targets = {e.target for e in baseline.events if e.type == EventType.WRITE}
    return sorted(b_targets - t_targets)


def compare(treatment: Trace, baseline: Trace) -> StructuralComparison:
    """Compute all structural metrics for a treatment/baseline pair."""
    ecr = event_count_ratio(treatment, baseline)
    h_t = tool_vocabulary_entropy(treatment.events)
    h_b = tool_vocabulary_entropy(baseline.events)
    h_ratio = h_t / h_b if h_b > 0 else 1.0
    wc = write_compression(treatment, baseline)
    ua = unilateral_actions(treatment, baseline)

    return StructuralComparison(
        event_count_ratio=ecr,
        tool_vocabulary_entropy_treatment=h_t,
        tool_vocabulary_entropy_baseline=h_b,
        entropy_ratio=h_ratio,
        write_compression=wc,
        unilateral_actions=ua,
        summary={
            "treatment_events": len(treatment.events),
            "baseline_events": len(baseline.events),
            "treatment_writes": sum(1 for e in treatment.events if e.type == EventType.WRITE),
            "baseline_writes": sum(1 for e in baseline.events if e.type == EventType.WRITE),
            "treatment_unique_tools": len({e.type.value for e in treatment.events}),
            "baseline_unique_tools": len({e.type.value for e in baseline.events}),
        },
    )
