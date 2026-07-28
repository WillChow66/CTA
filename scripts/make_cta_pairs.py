#!/usr/bin/env python3
"""
Build per-pair trace roots + CTA configs so the UNMODIFIED CTA pipeline can be
run on (a) same-condition NULL pairs and (b) treatment pairs.

Why this is needed
------------------
src/cta/pipeline.py::_run_module1 globs ``<trace_logs_dir>/**/claude_thinking/*.jsonl``
and buckets each trace by the ``use-skill-(true|false)`` field in its FILENAME
(src/cta/module1_parser.py::_FILENAME_RE). _run_module3 then aligns
``plus[i]`` against ``minus[i]`` for i in range(min(len(plus), len(minus))).

So:
  * a NULL pair = two runs of the SAME condition, one of which is *renamed* to
    the opposite use-skill flag. The pipeline then audits noise-vs-noise.
  * only ONE pair may live under a given trace root, otherwise index-wise
    pairing picks an arbitrary combination.

Usage
-----
  python3 make_cta_pairs.py --mode null-minus  --tasks tdd-workflow,xlsx,...
  python3 make_cta_pairs.py --mode null-plus   --tasks ...
  python3 make_cta_pairs.py --mode treatment   --tasks ...

Then, for every config it prints:
  python run_cta_analysis.py --config <cfg> --all -o json

Run from the CTA/ directory.
"""
import argparse
import itertools
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

FILENAME_RE = re.compile(
    r"^claude_(?P<task_id>.+?)"
    r"_use-agent-(?P<use_agent>true|false)"
    r"_use-skill-(?P<use_skill>true|false)"
    r"_(?P<ts>\d{8}_\d{6}).*\.jsonl$"
)


def discover(trace_root, tasks):
    """task -> {with_skill: [paths sorted by timestamp]}"""
    found = {}  # type: Dict[str, Dict[bool, List[Path]]]
    for p in sorted(trace_root.glob("**/claude_thinking/*.jsonl")):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        task = m.group("task_id")
        if tasks and task not in tasks:
            continue
        ws = m.group("use_skill") == "true"
        found.setdefault(task, {True: [], False: []})[ws].append(p)
    for task in found:
        for ws in (True, False):
            found[task][ws].sort(key=lambda q: FILENAME_RE.match(q.name).group("ts"))
    return found


def flip(name, to_true):
    want = "true" if to_true else "false"
    other = "false" if to_true else "true"
    return name.replace(f"_use-skill-{other}_", f"_use-skill-{want}_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["null-minus", "null-plus", "treatment"])
    ap.add_argument("--trace-root", default="claude_process/claude-opus-4.8/batch1")
    ap.add_argument("--tasks", default="", help="comma-separated; empty = all found")
    ap.add_argument("--base-config", default="cta_config.yaml")
    ap.add_argument("--out-root", default=None,
                    help="default: cta_pairs/<mode>")
    ap.add_argument("--max-pairs", type=int, default=3,
                    help="max pairs per task (default 3 = all pairs of 3 runs)")
    args = ap.parse_args()

    tasks = [t for t in args.tasks.split(",") if t]
    trace_root = Path(args.trace_root)
    out_root = Path(args.out_root or f"cta_pairs/{args.mode}")
    base_cfg = yaml.safe_load(open(args.base_config))

    found = discover(trace_root, tasks)
    if not found:
        raise SystemExit(f"No traces matched under {trace_root}")

    # pair index -> list of (src_path, dest_name)
    buckets = {}  # type: Dict[int, List[Tuple[Path, str]]]
    report = []  # type: List[str]

    for task, byc in sorted(found.items()):
        if args.mode == "treatment":
            plus, minus = byc[True], byc[False]
            combos = list(itertools.product(range(len(plus)), range(len(minus))))[
                : args.max_pairs
            ]
            if not combos:
                report.append(f"  SKIP {task}: need >=1 run in each condition "
                              f"(+{len(plus)} / -{len(minus)})")
                continue
            for k, (i, j) in enumerate(combos):
                buckets.setdefault(k, []).append((plus[i], plus[i].name))
                buckets.setdefault(k, []).append((minus[j], minus[j].name))
            report.append(f"  {task}: {len(combos)} treatment pair(s) "
                          f"from +{len(plus)} / -{len(minus)}")
        else:
            same = byc[False] if args.mode == "null-minus" else byc[True]
            combos = list(itertools.combinations(range(len(same)), 2))[: args.max_pairs]
            if not combos:
                report.append(f"  SKIP {task}: need >=2 same-condition runs, have {len(same)}")
                continue
            to_true = args.mode == "null-minus"  # relabel member A to the opposite flag
            for k, (i, j) in enumerate(combos):
                buckets.setdefault(k, []).append((same[i], flip(same[i].name, to_true)))
                buckets.setdefault(k, []).append((same[j], same[j].name))
            report.append(f"  {task}: {len(combos)} null pair(s) from {len(same)} runs")

    cmds = []  # type: List[str]
    for k, items in sorted(buckets.items()):
        pair_dir = out_root / f"pair{k}"
        dest = pair_dir / "traces" / "claude_thinking"
        dest.mkdir(parents=True, exist_ok=True)
        for src, name in items:
            shutil.copy2(src, dest / name)
        cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
        cfg["data"]["trace_logs_dir"] = str(pair_dir / "traces")
        cfg["data"]["output_dir"] = str(pair_dir / "cta_output")
        cfg_path = out_root / f"cta_config_{args.mode}_pair{k}.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        cmds.append(f"python run_cta_analysis.py --config {cfg_path} --all -o json")

    print(f"\nmode={args.mode}  trace_root={trace_root}")
    print("\n".join(report))
    print(f"\nBuilt {len(buckets)} pair root(s) under {out_root}")
    print("\nRun these (no API cost, pure re-analysis):")
    for c in cmds:
        print("  " + c)
    print(f"\nThen aggregate: {out_root}/pair*/cta_output/cta_combined_results.json")
    print("NOTE: Module 5 (quality prediction) reads config/cta_task_metadata.json,")
    print("      which holds the ORIGINAL Sonnet pass rates. Ignore Module 5 output")
    print("      for these pairs, or regenerate metadata with scripts/cta_prepare_data.py.")


if __name__ == "__main__":
    main()
