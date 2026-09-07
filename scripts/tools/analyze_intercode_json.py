#!/usr/bin/env python3
"""Where the missing intercode reward actually is.

A mean reward tells you how far you are from 1.0; it never tells you which
mistake to fix.  This buckets every scored task by the FIRST thing that went
wrong and prices each bucket in mean-reward terms, so the biggest number is
the thing worth working on next.

    python3 scripts/tools/analyze_intercode_json.py eval_out/ab_capbaru.json
    python3 scripts/tools/analyze_intercode_json.py eval_out/ab_capbaru.json eval_out/ab_caplama.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

# Substrings a shell prints when the COMMAND ITSELF is malformed, as opposed to
# a well-formed command that found nothing.  Keep this list literal: a regex
# over stderr is how "No such file or directory" (a legitimate answer to a
# legitimate question) ends up miscounted as a syntax failure.
SYNTAX_MARKERS = (
    "paths must precede expression",
    "unknown predicate",
    "invalid option",
    "unrecognized option",
    "syntax error",
    "unexpected token",
    "missing argument",
    "Try 'find --help'",
    "usage:",
    "Usage:",
    "illegal option",
    "invalid argument",
    "command not found",
)


def bucket(row: dict) -> str:
    # A JSON written before traces were recorded has no "trace" key at all.
    # Guessing from the final observation alone is much weaker -- it cannot
    # tell a silent turn from no turn -- so say so rather than reporting
    # every task as "no turn at all".
    if "trace" not in row:
        obs = (row.get("agent_obs") or "")
        sim = row.get("similarity", 0.0)
        if sim > 0.999:
            return "solved"
        if obs == "Invalid tool call.":
            return "never emitted a valid tool call"
        if any(m in obs for m in SYNTAX_MARKERS):
            return "bash was malformed"
        if not obs.strip():
            return "ran, printed nothing"
        if sim >= 0.5:
            return "near miss (sim >= 0.5)"
        return "ran cleanly, wrong output"

    trace = row.get("trace") or []
    cmds = [t for t in trace if t.get("cmd")]
    sim = row.get("similarity", 0.0)

    if sim > 0.999:
        return "solved"
    if not trace:
        return "no turn at all"
    if not cmds:
        return "never emitted a valid tool call"
    if any(t.get("obs") == "Invalid tool call." for t in trace):
        return "some turns unparseable"
    if any(m in (t.get("obs") or "") for t in cmds for m in SYNTAX_MARKERS):
        return "bash was malformed"
    if not (row.get("agent_obs") or "").strip():
        return "ran, printed nothing"
    if sim >= 0.5:
        return "near miss (sim >= 0.5)"
    return "ran cleanly, wrong output"


def report(path: Path) -> dict:
    rows = json.loads(path.read_text()).get("tasks", [])
    if not rows:
        print(f"{path}: no tasks")
        return {}
    n = len(rows)
    if not any("trace" in r for r in rows):
        print(f"\n  CATATAN: {path.name} ditulis sebelum trace direkam, jadi")
        print( "           bucket di bawah ditebak dari observasi terakhir saja.")
        print( "           Skor ulang (2 menit) untuk analisis penuh.")
    counts: Counter = Counter()
    lost: Counter = Counter()
    for r in rows:
        b = bucket(r)
        counts[b] += 1
        # What this task leaves on the table in p3, the only part still moving.
        lost[b] += 0.33 * (1.0 - r.get("similarity", 0.0))

    mean = sum(r["reward"] for r in rows) / n
    print(f"\n=== {path.name} ===")
    print(f"  {n} task, mean reward {mean:.4f}")
    print(f"  {'bucket':<32} {'n':>4} {'%':>6} {'p3 hilang':>11}")
    for b, c in counts.most_common():
        print(f"  {b:<32} {c:>4} {100*c/n:>5.1f}% {lost[b]/n:>10.4f}")
    print(f"  {'TOTAL p3 yang bisa direbut':<32} {'':>4} {'':>6} {sum(lost.values())/n:>10.4f}")

    # The commands the model reaches for, and how often each one ends badly.
    verbs: Counter = Counter()
    verb_bad: Counter = Counter()
    for r in rows:
        for t in (r.get("trace") or []):
            cmd = (t.get("cmd") or "").strip()
            if not cmd:
                continue
            v = re.split(r"[\s|;&]+", cmd)[0].split("/")[-1]
            verbs[v] += 1
            if any(m in (t.get("obs") or "") for m in SYNTAX_MARKERS):
                verb_bad[v] += 1
    if verbs:
        print(f"\n  perintah terbanyak (dan berapa yang sintaksnya rusak):")
        for v, c in verbs.most_common(8):
            bad = verb_bad[v]
            flag = "  <-- sering rusak" if c and bad / c > 0.25 else ""
            print(f"    {v:<12} {c:>4}x   rusak {bad:>3} ({100*bad/c:>4.0f}%){flag}")

    turns = [r.get("turns", 0) for r in rows]
    if turns:
        print(f"\n  giliran per task: rata-rata {sum(turns)/len(turns):.1f}, "
              f"maks {max(turns)}, yang 0 giliran: {sum(1 for t in turns if t == 0)}")
    return {"n": n, "mean": mean, "counts": counts, "rows": rows}


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 1
    seen = [(p, report(p)) for p in paths if p.exists()]
    for p in paths:
        if not p.exists():
            print(f"\n  (lewati {p}: tidak ada)")
    if len(seen) == 2 and all(s[1] for s in seen):
        (pa, a), (pb, b) = seen
        print(f"\n=== {pa.name} -> {pb.name} ===")
        print(f"  mean reward {a['mean']:.4f} -> {b['mean']:.4f}   ({b['mean']-a['mean']:+.4f})")
        keys = set(a["counts"]) | set(b["counts"])
        for k in sorted(keys, key=lambda k: -(b["counts"].get(k, 0) + a["counts"].get(k, 0))):
            ca, cb = a["counts"].get(k, 0), b["counts"].get(k, 0)
            if ca or cb:
                print(f"  {k:<32} {ca:>4} -> {cb:<4} ({cb-ca:+d})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
