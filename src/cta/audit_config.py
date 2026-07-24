"""Audit configuration loader for the generalized CTA runner.

Loads a per-skill config (YAML or JSON) that drives tool filters, SIP detectors,
hypothesis evaluation, control validation, and report generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DelegationFilter:
    tool_name: str
    command_contains: str


@dataclass
class BinaryResolutionFilter:
    tool_name: str
    command_regex: str
    command_contains: str


@dataclass
class ToolFilters:
    delegation_call: DelegationFilter
    manual_writes: List[str]
    skill_view: str
    binary_resolution: BinaryResolutionFilter
    pty_arg: str = "pty"


@dataclass
class HypothesisConfig:
    name: str
    metric: str
    confirm_threshold: Optional[float] = None
    disconfirm_threshold: Optional[float] = None
    note: Optional[str] = None


@dataclass
class ControlConfig:
    task_id: str
    pass_criteria: str


@dataclass
class ControlsConfig:
    negative: Optional[ControlConfig] = None
    edge_case: Optional[ControlConfig] = None


@dataclass
class TaskConfig:
    id: str
    prompt: str
    expect: str = ""
    category: str = "positive"


@dataclass
class ReportConfig:
    title: str = "CTA Skill Audit: {skill_name}"
    model: str = ""
    design: str = ""


@dataclass
class AuditConfig:
    skill_name: str
    skill_version: str
    skill_path: str
    captures_dir: Path
    tool_filters: ToolFilters
    sip_detectors: List[str]
    hypotheses: Dict[str, HypothesisConfig]
    controls: ControlsConfig
    report: ReportConfig
    metrics: List[str] = field(default_factory=list)
    tasks: List[TaskConfig] = field(default_factory=list)
    fixture: str = "fixture/"
    runs_per_condition: int = 3


def load_config(path: Path, captures_dir_override: Optional[Path] = None) -> AuditConfig:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)

    skill = raw["skill"]
    tf_raw = raw["tool_filters"]

    tool_filters = ToolFilters(
        delegation_call=DelegationFilter(
            tool_name=tf_raw["delegation_call"]["tool_name"],
            command_contains=tf_raw["delegation_call"]["command_contains"],
        ),
        manual_writes=tf_raw["manual_writes"],
        skill_view=tf_raw["skill_view"],
        binary_resolution=BinaryResolutionFilter(
            tool_name=tf_raw["binary_resolution"]["tool_name"],
            command_regex=tf_raw["binary_resolution"]["command_regex"],
            command_contains=tf_raw["binary_resolution"]["command_contains"],
        ),
        pty_arg=tf_raw.get("pty_arg", "pty"),
    )

    hypotheses = {}
    for hid, h in raw.get("hypotheses", {}).items():
        hypotheses[hid] = HypothesisConfig(
            name=h["name"],
            metric=h["metric"],
            confirm_threshold=h.get("confirm_threshold"),
            disconfirm_threshold=h.get("disconfirm_threshold"),
            note=h.get("note"),
        )

    controls_raw = raw.get("controls", {})
    controls = ControlsConfig(
        negative=ControlConfig(**controls_raw["negative"]) if "negative" in controls_raw else None,
        edge_case=ControlConfig(**controls_raw["edge_case"]) if "edge_case" in controls_raw else None,
    )

    report_raw = raw.get("report", {})
    report = ReportConfig(
        title=report_raw.get("title", "CTA Skill Audit: {skill_name}"),
        model=report_raw.get("model", ""),
        design=report_raw.get("design", ""),
    )

    captures_dir = captures_dir_override or Path(raw.get("captures_dir", "data/m2_captures"))

    tasks: List[TaskConfig] = []
    tasks_raw = raw.get("tasks", {})
    for category, task_list in tasks_raw.items():
        for t in task_list or []:
            tasks.append(TaskConfig(
                id=t["id"],
                prompt=t["prompt"],
                expect=t.get("expect", ""),
                category=category,
            ))

    return AuditConfig(
        skill_name=skill["name"],
        skill_version=skill.get("version", ""),
        skill_path=skill.get("path", ""),
        captures_dir=captures_dir,
        tool_filters=tool_filters,
        sip_detectors=raw.get("sip_detectors", []),
        hypotheses=hypotheses,
        controls=controls,
        report=report,
        metrics=raw.get("metrics", []),
        tasks=tasks,
        fixture=raw.get("fixture", "fixture/"),
        runs_per_condition=raw.get("runs_per_condition", 3),
    )
