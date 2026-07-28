#!/usr/bin/env python3
"""
SKILL-DOCUMENT-ONLY LLM BASELINE for Counterfactual Trace Auditing (CTA).

Question: how much of CTA's per-task output is recoverable by an LLM that reads
ONLY the skill document + the task description, with NO trace analysis at all?

The model is blind to every CTA artifact. It predicts, per task:
  (a) whether the skill raises / does not change / lowers unit-test pass rate
  (b) the dominant Skill Influence Pattern (SIP) among the 5 CTA categories
  (c) a 0-1 confidence

Usage:
  python3 skill_only_baseline_run.py --model gemini --out <dir> [--tasks t1,t2,...]

Reads credentials from <REPO_ROOT>/rebuttal/.nairr_env (never printed).
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("<REPO_ROOT>/rebuttal")
CTA = ROOT / "CTA"
SKILLS = CTA / "skills"
TASKS = CTA / "tasks" / "batch1"

SIP_TYPES = [
    "procedural_scaffolding",
    "edge_case_prompting",
    "redundant_exploration",
    "surface_anchoring",
    "concept_bleed",
]

SYSTEM_PROMPT = """You are an expert analyst of AI coding-agent behaviour.

BACKGROUND. A "skill" is a markdown document injected into a coding agent's
context before it attempts a software-engineering task. Researchers run the same
task twice with the same model: once WITH the skill document in context, once
WITHOUT it. They then compare the two execution traces and label the ways the
skill changed the agent's behaviour. Each behavioural change is classified into
exactly one of five Skill Influence Patterns (SIPs):

1. procedural_scaffolding (PS, constructive) - the skill supplies concrete steps
   the agent would not otherwise have taken, so the with-skill run performs MORE
   implementation work in service of the SAME goal: extra edits, extra checks,
   more steps toward the same targets.

2. edge_case_prompting (EP, constructive) - the skill pushes the agent to produce
   an artifact or guard the baseline run never produced at all: an extra test
   file, a defensive branch, error handling, a null/version check, a validation
   script. Structurally it shows up as the with-skill agent acting on a target
   the baseline never touched.

3. redundant_exploration (RE, neutral) - the skill sends the agent on a detour
   that lands in the same place: same intent, same final content, but more events
   spent getting there. Re-reading, re-checking, restating what it already knew.

4. surface_anchoring (SA, destructive) - the agent copies literal surface tokens
   verbatim out of the skill document into its own output: version numbers,
   file paths, string literals, SHOUT_CASE constants, config names that came from
   the skill's examples rather than from the actual repository.

5. concept_bleed (CB, destructive) - a broad skill convinces the agent to widen
   the task: it introduces substantially MORE new files/targets than the baseline,
   and the extras trace back to skill sections that are not relevant to this task.

You will be shown a skill document and the task it was used on. You will NOT be
shown any execution trace. Predict the outcome anyway, from the document alone.

Answer with a single strict JSON object and nothing else. No markdown fences, no
prose before or after. Schema:

{"pass_rate_effect": "raise" | "no_change" | "lower",
 "dominant_sip": "procedural_scaffolding" | "edge_case_prompting" | "redundant_exploration" | "surface_anchoring" | "concept_bleed",
 "confidence": <float 0.0-1.0>,
 "reasoning": "<at most 40 words>"}

"pass_rate_effect" is whether adding this skill document changes the unit-test
pass rate of the agent on this task relative to running without it.
"dominant_sip" is the SIP you expect to occur MOST OFTEN in the with-vs-without
trace comparison for this task.
"confidence" is your confidence in dominant_sip."""

# ---------------------------------------------------------------------------
# ABLATION VARIANT: elicit a calibrated probability distribution instead of a
# forced hard choice, to test whether the baseline's failure on pass-rate
# effect is merely an artifact of forcing a direction. Reveals no ground truth.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_PROBS = SYSTEM_PROMPT.split("Answer with a single strict JSON")[0] + """\
Answer with a single strict JSON object and nothing else. No markdown fences, no
prose before or after. Schema:

{"p_raise": <float>, "p_no_change": <float>, "p_lower": <float>,
 "dominant_sip": "procedural_scaffolding" | "edge_case_prompting" | "redundant_exploration" | "surface_anchoring" | "concept_bleed",
 "confidence": <float 0.0-1.0>,
 "reasoning": "<at most 40 words>"}

p_raise / p_no_change / p_lower are CALIBRATED probabilities that must sum to
1.0. p_no_change is the probability that adding this skill document produces NO
MEASURABLE CHANGE in the unit-test pass rate on this task. Be honest about
uncertainty: if you cannot tell from the document alone, say so with your
probabilities. Do not inflate p_raise or p_lower to seem decisive.
"dominant_sip" is the SIP you expect to occur MOST OFTEN in the with-vs-without
trace comparison. "confidence" is your confidence in dominant_sip."""

USER_TMPL = """## TASK DESCRIPTION (`{task_id}`)

{task_md}

## SKILL DOCUMENT INJECTED FOR THIS TASK (`{task_id}/SKILL.md`)

{skill_md}

## YOUR PREDICTION

Emit the strict JSON object now."""


def load_env(path=ROOT / ".nairr_env"):
    env = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def call_model(base, key, model, system, user, temperature=0.0, max_tok=3000,
               retries=4):
    url = base.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_completion_tokens": max_tok,
    }
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=data,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            last = e
            detail = ""
            if isinstance(e, urllib.error.HTTPError):
                try:
                    detail = e.read().decode()[:300]
                except Exception:  # noqa: BLE001
                    pass
            sys.stderr.write(f"[retry {attempt+1}] {type(e).__name__} {detail}\n")
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"all retries failed: {last}")


def parse_json(txt):
    """Best-effort strict-JSON extraction from a model reply."""
    if not txt:
        return None
    t = txt.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tok", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--variant", default="hard", choices=["hard", "probs"])
    args = ap.parse_args()

    env = load_env()
    base, key = env["NAIRR_BASE"], env["NAIRR_KEY"]

    SYS = SYSTEM_PROMPT if args.variant == "hard" else SYSTEM_PROMPT_PROBS
    tag = args.model if args.variant == "hard" else f"{args.model}-probs"
    outdir = Path(args.out)
    rawdir = outdir / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)

    task_ids = sorted(p.name for p in SKILLS.iterdir() if (p / "SKILL.md").exists())
    if args.tasks:
        want = set(args.tasks.split(","))
        task_ids = [t for t in task_ids if t in want]
    print(f"[info] model={args.model} tasks={len(task_ids)} temp={args.temperature}")

    # persist the exact prompt template used
    (outdir / f"prompt_template_{tag}.txt").write_text(
        "=== SYSTEM ===\n" + SYS + "\n\n=== USER TEMPLATE ===\n" + USER_TMPL
    )

    def work(tid):
        skill_md = (SKILLS / tid / "SKILL.md").read_text()
        tmd_p = TASKS / f"{tid}.md"
        task_md = tmd_p.read_text() if tmd_p.exists() else "(no task description file)"
        user = USER_TMPL.format(task_id=tid, task_md=task_md, skill_md=skill_md)
        t0 = time.time()
        resp = call_model(base, key, args.model, SYS, user,
                          args.temperature, args.max_tok)
        dt = time.time() - t0
        content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
        usage = resp.get("usage", {}) or {}
        rec = {
            "task_id": tid,
            "model": tag,
            "temperature": args.temperature,
            "latency_s": round(dt, 2),
            "usage": usage,
            "raw_content": content,
            "parsed": parse_json(content),
            "user_prompt_chars": len(user),
        }
        (rawdir / f"{tag}__{tid}.json").write_text(json.dumps(rec, indent=1))
        p = rec["parsed"] or {}
        eff = p.get("pass_rate_effect")
        if eff is None and "p_no_change" in p:
            eff = "pnc=%.2f" % p["p_no_change"]
        print(f"  {tid:42s} {str(eff):10s} "
              f"{str(p.get('dominant_sip')):24s} conf={p.get('confidence')} "
              f"[{usage.get('prompt_tokens')}+{usage.get('completion_tokens')}tok {dt:.0f}s]")
        return rec

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, t): t for t in task_ids}
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001
                print(f"  [FAIL] {futs[f]}: {e}")
                results.append({"task_id": futs[f], "model": tag,
                                "error": str(e), "parsed": None, "usage": {}})

    results.sort(key=lambda r: r["task_id"])
    (outdir / f"predictions_{tag}.json").write_text(json.dumps(results, indent=1))

    pt = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in results)
    ct = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in results)
    ok = sum(1 for r in results if r.get("parsed"))
    print(f"[done] parsed_ok={ok}/{len(results)} prompt_tokens={pt} completion_tokens={ct}")
    (outdir / f"usage_{tag}.json").write_text(json.dumps(
        {"model": tag, "n_calls": len(results), "parsed_ok": ok,
         "prompt_tokens": pt, "completion_tokens": ct,
         "total_tokens": pt + ct}, indent=1))


if __name__ == "__main__":
    main()
