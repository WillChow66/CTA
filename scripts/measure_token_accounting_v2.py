import json, re, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path("scan")
PAPER = ("input_tokens","output_tokens")
ALL4  = ("input_tokens","output_tokens","cache_creation_input_tokens","cache_read_input_tokens")

def totals(fp):
    p = a = 0
    with open(fp, errors="ignore") as f:
        for line in f:
            if "usage" not in line: continue
            try: obj = json.loads(line)
            except Exception: continue
            for u in find_usage(obj):
                p += sum(int(u.get(k,0) or 0) for k in PAPER)
                a += sum(int(u.get(k,0) or 0) for k in ALL4)
    return p, a

def find_usage(o):
    if isinstance(o, dict):
        if "usage" in o and isinstance(o["usage"], dict): yield o["usage"]
        for v in o.values(): yield from find_usage(v)
    elif isinstance(o, list):
        for v in o: yield from find_usage(v)

runs = defaultdict(dict)   # (arm, task, cond) -> list of (paper, all4)
for fp in ROOT.rglob("*.jsonl"):
    parts = fp.parts
    if "traces" not in parts or "analysis" in parts: continue
    i = parts.index("traces")
    arm, task, cond = parts[i-1], parts[i+1], parts[i+2]
    rep = parts[i+3] if len(parts) > i+3 else "r?"
    runs.setdefault((arm,task),{}).setdefault(cond,[]).append(totals(fp))

print(f"{'task':<30}{'paper ratio':>12}{'true ratio':>12}{'paper capt%':>12}")
print("-"*66)
rows=[]
for (arm,task),cond in sorted(runs.items()):
    if arm != "claude-sonnet-4.5-nudged": continue
    w, wo = cond.get("with",[]), cond.get("without",[])
    if not w or not wo: continue
    pw = sum(x[0] for x in w)/len(w); aw = sum(x[1] for x in w)/len(w)
    po = sum(x[0] for x in wo)/len(wo); ao = sum(x[1] for x in wo)/len(wo)
    if po<=0 or ao<=0: continue
    rows.append((task, pw/po, aw/ao, 100*(pw+po)/(aw+ao)))
for t,pr,tr,cap in rows:
    print(f"{t:<30}{pr:>12.2f}{tr:>12.2f}{cap:>12.2f}")
import statistics as st
print("-"*66)
print(f"{'MEAN':<30}{st.mean(r[1] for r in rows):>12.2f}{st.mean(r[2] for r in rows):>12.2f}{st.mean(r[3] for r in rows):>12.2f}")
print(f"{'MEDIAN':<30}{st.median(r[1] for r in rows):>12.2f}{st.median(r[2] for r in rows):>12.2f}{st.median(r[3] for r in rows):>12.2f}")
print(f"\nratios below 1.0:  paper method {sum(1 for r in rows if r[1]<1)}/{len(rows)}   full accounting {sum(1 for r in rows if r[2]<1)}/{len(rows)}")
