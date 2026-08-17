"""Offline parallel multi-seed MCCFR prebuild for the Liar's Dice blueprint.

Runs many external-sampling MCCFR workers from DIFFERENT seeds in parallel, SUMS
their per-infoset strategy accumulators (strat_sum), normalizes, and atomic-writes
ONE merged blueprint into the RUNTIME cache dir (default /cache/ld_mccfr) — NOT a
repo bundle. The runtime expert (liar_dice_cfr._get_blueprint) then LOADS it in ms
instead of each gen worker doing a thin cold solve. Merging N seeds' strat_sum is
equivalent to one ~N*iters MCCFR run (CFR+ average strategies sum), so the teacher
is markedly better converged. Offline, pure-Python, best-effort: any failure just
leaves the per-worker in-process build as the runtime fallback.

The cache (de)serialization uses liar_dice_cfr's SHARED helpers, so the format the
prebuild writes is byte-identical to what the runtime reader expects (no drift).

Run standalone:
    python -m our_envs.prebuild_liar_dice_blueprint --total-dice 10 --workers 20 \
        --iters-per-worker 6000 --cache-dir /cache/ld_mccfr
(or it is invoked automatically at gen-start by generate_trajectories).
"""
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import liar_dice_cfr as e  # noqa: E402

# Small chunk so the wall-time cap is checked frequently: at total=10 each MCCFR
# iteration is ~0.28s, so a 200-iter chunk would be ~56s — too coarse to bound the
# cap tightly. 50 iters (~14s at total=10, instant at small totals) keeps the
# overshoot past the deadline small while the per-chunk time-check cost is nil.
_TRAIN_CHUNK = 50


def _train_worker(args):
    """One MCCFR seed -> {(hand, current|None): list(strat_sum)}. Picklable result.

    Runs a CHUNKED, WALL-TIME-CAPPED solve: a full `iters` solve at total=10 takes
    ~28 min, so without an internal cap every worker runs to completion and the
    merge silently degrades to one seed (the cap is enforced here, NOT by the
    parent cancelling already-running futures, which the executor cannot do). CFR+
    average strategies are anytime-valid, so a partial solve is still a usable
    blueprint."""
    total_dice, iters, seed, max_seconds = args
    solver = e.DudoMCCFR(total_dice // 2)
    rng = random.Random(seed)
    start = time.monotonic()
    done = 0
    while done < iters:
        step = min(_TRAIN_CHUNK, iters - done)
        solver.train(step, rng)
        done += step
        if max_seconds and (time.monotonic() - start) >= max_seconds:
            break
    return {key: list(nd.strat_sum) for key, nd in solver.nodes.items()}


def build_parallel(total_dice: int, workers: int, iters_per_worker: int,
                   base_seed: int = 0, max_seconds: "float | None" = None) -> dict:
    """Run `workers` seeds in parallel (each self-capped at `max_seconds` wall
    time), SUM strat_sum per infoset, normalize over each infoset's legal actions
    -> blueprint {(hand, current|None): {aid: prob}}. Workers all stop near the
    cap and return their partial (anytime-valid) accumulators; we merge EVERY
    completed future via as_completed (no submission-order blocking), so all N
    seeds contribute and the total wall time is ~max_seconds, not N*per-worker."""
    workers = max(1, workers)
    args = [(total_dice, iters_per_worker, base_seed + s, max_seconds) for s in range(workers)]
    merged: dict = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_train_worker, a) for a in args]
        for fut in as_completed(futures):
            try:
                part = fut.result()
            except Exception:
                continue
            for key, ss in part.items():
                acc = merged.get(key)
                if acc is None:
                    merged[key] = list(ss)
                else:
                    for i in range(len(ss)):
                        acc[i] += ss[i]
    bp: dict = {}
    for (hand, current), ss in merged.items():
        acts = e.legal_action_ids_for(current, total_dice)
        s = sum(ss)
        if s > 0.0 and len(acts) == len(ss):
            bp[(hand, current)] = {acts[i]: ss[i] / s for i in range(len(acts))}
        # audit P2: zero-mass infosets (opponent-only visits) used to be written
        # as fabricated uniform entries, outranking the epsilon-band at runtime.
        # Omit them so the band rescues those states.
    return bp


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--total-dice", type=int, default=10)
    p.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 8))
    p.add_argument("--iters-per-worker", type=int, default=6000)
    p.add_argument("--cache-dir", default=os.environ.get("LD_MCCFR_CACHE_DIR", "/cache/ld_mccfr"))
    p.add_argument("--max-seconds", type=float, default=0.0)
    args = p.parse_args()
    try:
        max_seconds = args.max_seconds if args.max_seconds > 0 else None
        t0 = time.time()
        bp = build_parallel(args.total_dice, args.workers, args.iters_per_worker, max_seconds=max_seconds)
        path = e.cache_file_path(args.total_dice, args.cache_dir)
        if path and bp:
            e._atomic_write_json(path, e._serialize_blueprint(bp))
            print(f"[prebuild_ld] {len(bp)} infosets in {time.time() - t0:.1f}s "
                  f"({args.workers}x{args.iters_per_worker} iters) -> {path}", flush=True)
        else:
            print(f"[prebuild_ld] no cache path or empty blueprint (path={path}, n={len(bp)})", flush=True)
    except Exception as exc:
        print(f"[prebuild_ld] prebuild failed (non-fatal): {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
