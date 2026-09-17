#!/usr/bin/env python3
"""Recompute an EnvTask group score from local measurements.

The validator's environment score, reverse-engineered and then CONFIRMED
against the published numbers for group 3 of tourn_4f605297e6d85428_20260907:

    per environment: rank the miners (ties take the AVERAGE rank),
                     normalise to (N - rank) / (N - 1),
    task score:      the mean of those across environments.

This reads the per-model intercode JSONs and the per-pair othello JSONs that
tools/run_replay_group.sh produces, and prints the table.

--expect group3 checks the result against what the validator actually
published. That check is the point: a replay that cannot reproduce a known
outcome cannot be trusted on an unknown one, so run it before reading
anything into a counterfactual.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# What the validator published for tourn_4f605297e6d85428_20260907 round 1
# group 3 (task ce540755). Source: api.gradients.io/tournament/latest/details.
TRUTH = {
    "group3": {
        "intercode": {"5CcwdezW": 0.7745, "5E6p1x3e": 0.7250,
                      "5EeEWvKf": 0.8005, "5GU4Xkd3": 0.7350},
        # a_wins / draws / b_wins over the games the validator played
        "othello_pairs": {
            ("5CcwdezW", "5EeEWvKf"): (4, 0, 12),
            ("5E6p1x3e", "5GU4Xkd3"): (9, 0, 7),
            ("5CcwdezW", "5GU4Xkd3"): (9, 2, 5),
            ("5CcwdezW", "5E6p1x3e"): (4, 0, 12),
            ("5E6p1x3e", "5EeEWvKf"): (9, 0, 9),
            ("5EeEWvKf", "5GU4Xkd3"): (9, 0, 7),
        },
        "final": {"5CcwdezW": 0.3333333333333333, "5E6p1x3e": 0.4166666666666667,
                  "5EeEWvKf": 0.9166666666666667, "5GU4Xkd3": 0.3333333333333333},
    }
}


def ranks_with_ties(scores: dict) -> dict:
    """Competition ranking where tied entries share the AVERAGE rank. This is
    the detail that makes the formula reproduce the published numbers: two
    miners tied at the top take rank 1.5, not 1 and 2."""
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    out: dict = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        avg = (i + j) / 2 + 1          # 1-based, averaged over the tied block
        for k in range(i, j + 1):
            out[ordered[k][0]] = avg
        i = j + 1
    return out


def normalise(ranks: dict) -> dict:
    n = len(ranks)
    if n < 2:
        return {k: 1.0 for k in ranks}
    return {k: (n - r) / (n - 1) for k, r in ranks.items()}


def othello_scores(pairs: dict, tags: list) -> dict:
    """Head-to-head record -> (wins + 0.5*draws) / games played, per miner."""
    w = {t: 0.0 for t in tags}
    n = {t: 0 for t in tags}
    for (a, b), (aw, dr, bw) in pairs.items():
        if a not in w or b not in w:
            continue
        total = aw + dr + bw
        w[a] += aw + 0.5 * dr
        w[b] += bw + 0.5 * dr
        n[a] += total
        n[b] += total
    return {t: (w[t] / n[t] if n[t] else 0.0) for t in tags}


def load_measured(d: Path, tags: list):
    inter: dict = {}
    for t in tags:
        p = d / f"intercode_{t}.json"
        if p.is_file():
            rows = json.loads(p.read_text()).get("tasks", [])
            if rows:
                inter[t] = sum(r["reward"] for r in rows) / len(rows)
    pairs: dict = {}
    for p in sorted(d.glob("othello_*__*.json")):
        j = json.loads(p.read_text())
        a, b = p.stem[len("othello_"):].split("__", 1)
        pairs[(a, b)] = (j["a_wins"], j["draws"], j["b_wins"])
    return inter, pairs


def table(title: str, inter: dict, pairs: dict, tags: list, me: str = None):
    oth = othello_scores(pairs, tags)
    ni = normalise(ranks_with_ties(inter))
    no = normalise(ranks_with_ties(oth))
    final = {t: (ni.get(t, 0.0) + no.get(t, 0.0)) / 2 for t in tags}
    order = sorted(tags, key=lambda t: -final[t])
    print(f"\n=== {title} ===")
    print(f"  {'miner':<12} {'intercode':>10} {'norm':>7} {'othello':>9} {'norm':>7} {'SKOR':>8}  peringkat")
    for i, t in enumerate(order, 1):
        star = "  <-- Anda" if me and t == me else ""
        print(f"  {t:<12} {inter.get(t,float('nan')):>10.4f} {ni.get(t,0):>7.3f} "
              f"{oth.get(t,0):>9.4f} {no.get(t,0):>7.3f} {final[t]:>8.4f}  {i}{star}")
    return final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="eval_out/replay")
    ap.add_argument("--tags", nargs="*", default=["5CcwdezW", "5E6p1x3e", "5EeEWvKf", "5GU4Xkd3"])
    ap.add_argument("--me", default="5CcwdezW")
    ap.add_argument("--expect", default=None, choices=sorted(TRUTH))
    args = ap.parse_args()

    if args.expect:
        t = TRUTH[args.expect]
        got = table(f"KEBENARAN validator ({args.expect})",
                    t["intercode"], t["othello_pairs"], args.tags, args.me)
        bad = [k for k in t["final"] if abs(got[k] - t["final"][k]) > 1e-6]
        if bad:
            print(f"\n  RUMUSNYA SALAH untuk {bad} -- jangan percaya angka replay di bawah.")
        else:
            print("\n  Rumus cocok dengan angka resmi sampai 1e-6.")

    d = Path(args.dir)
    inter, pairs = load_measured(d, args.tags)
    missing_i = [t for t in args.tags if t not in inter]
    want_pairs = len(args.tags) * (len(args.tags) - 1) // 2
    if missing_i or len(pairs) < want_pairs:
        print(f"\n  Pengukuran lokal belum lengkap: intercode hilang {missing_i or 'tidak ada'}, "
              f"pasangan othello {len(pairs)}/{want_pairs}.")
        if not inter or not pairs:
            return 0
    measured = table("REPLAY lokal", inter, pairs, args.tags, args.me)

    if args.expect:
        t = TRUTH[args.expect]
        print(f"\n=== lokal vs validator ===")
        print(f"  {'miner':<12} {'intercode lokal':>16} {'validator':>11} {'selisih':>9}")
        for tag in args.tags:
            if tag in inter:
                v = t["intercode"].get(tag)
                print(f"  {tag:<12} {inter[tag]:>16.4f} {v:>11.4f} {inter[tag]-v:>+9.4f}")
        print(f"\n  {'miner':<12} {'SKOR lokal':>11} {'SKOR resmi':>11} {'selisih':>9}")
        for tag in args.tags:
            if tag in measured:
                print(f"  {tag:<12} {measured[tag]:>11.4f} {t['final'][tag]:>11.4f} "
                      f"{measured[tag]-t['final'][tag]:>+9.4f}")
        print("\n  Selisih intercode yang KONSISTEN arahnya = offset alat ukur, bukan")
        print("  kesalahan: geser semua angka lokal lama dengan offset itu. Selisih")
        print("  yang acak = alatnya tidak mengukur hal yang sama.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
