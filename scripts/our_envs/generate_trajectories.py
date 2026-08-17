"""Generate expert game trajectories against env servers and save as
an HF DatasetDict (train / validation splits) ready for train_sft_env.py.

Analogous to tokenize_instruct.py but for environment SFT tasks.

Per-game MCTS opponent strength matches what the validator uses for
env evaluation (G.O.D/validator/core/constants.py:ENVIRONMENTS). This
keeps the expert trajectories aligned with the playstyle the model
will be graded against.

Run from /workspace/scripts/:
  python -m our_envs.generate_trajectories --environment_name liars_dice \\
      --output_path /path/to/dataset --num_games 5000

  python -m our_envs.generate_trajectories --environment_name gin_rummy \\
      --output_path /path/to/dataset --num_games 1000 --max_turn 200
"""

import argparse
import os
import random
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from concurrent.futures.process import BrokenProcessPool

from datasets import Dataset, DatasetDict

from our_envs.shared_env import GAMES_TO_TASK_ID_RANGE, init_env_pool
from our_envs.sft_env_configs import get_sft_trajectory_generator, inprocess_kind, runs_inprocess


# Authoritative per-game MCTS opponent config (mirrors
# G.O.D/validator/core/constants.py:ENVIRONMENTS[<env>].eval_payload_extra).
_OPPONENT_CONFIG_PER_GAME: dict[str, dict] = {
    "liars_dice":  {"opponent": "mcts", "mcts_max_simulations": 225, "mcts_num_rollouts": 1},
    "leduc_poker": {"opponent": "mcts", "mcts_max_simulations": 50,  "mcts_num_rollouts": 1},
    "gin_rummy":   {"opponent": "mcts", "mcts_max_simulations": 50,  "mcts_num_rollouts": 1},
    # othello (PR #1201): EVAL baseline = mcts@50 (ENVIRONMENTS["othello"]
    # .eval_payload_extra); the GRPO training rollout uses a weaker mcts@25.
    # We train at eval strength (50) so win-biased data matches eval difficulty.
    "othello":     {"opponent": "mcts", "mcts_max_simulations": 50,  "mcts_num_rollouts": 1},
    # goofspiel (PR #1217): generate vs the PvP MCTS baseline. SMOKE-TEST GAP —
    # if the env-server rejects mcts for goofspiel, change to {"opponent": "random"}.
    "goofspiel":   {"opponent": "mcts", "mcts_max_simulations": 50,  "mcts_num_rollouts": 1},
    # clobber (#1243): EVAL baseline = mcts@50 (ENVIRONMENTS["clobber"]
    # .eval_payload_extra). The GRPO training rollout uses a weaker mcts@25.
    "clobber":     {"opponent": "mcts", "mcts_max_simulations": 50,  "mcts_num_rollouts": 1},
}


# Process-pool worker: each process loads the expert generator once via
# _worker_init, then handles multiple games sequentially. Using processes
# (not threads) gives each worker its own GIL so CPU-bound expert
# computation runs truly in parallel. --num_workers controls how many
# concurrent env-server connections are open.

_GENERATE_FN = None


def _worker_init(env_name: str) -> None:
    global _GENERATE_FN
    _GENERATE_FN = get_sft_trajectory_generator(env_name)


def _worker_play(game_id: int, endpoint: str, max_turn: int):
    """Return either list[dict] (legacy expert generators) or
    tuple[list[dict], float] (winner-style random + score generators)."""
    return _GENERATE_FN(game_id, endpoint, max_turn)


VALIDATION_RATIO    = 0.01
MIN_ASSISTANT_TURNS = 1


def _sliding_windows(conv: list[dict], window_turns: int, window_step: int) -> list[list[dict]]:
    """Split a conversation into overlapping sub-conversations.

    Each window: [system] + window_turns * (user, assistant) pairs.
    Short games (fewer than window_turns pairs) are kept as one window.
    """
    system = [m for m in conv if m["role"] == "system"]
    turns  = [m for m in conv if m["role"] != "system"]

    pairs = []
    i = 0
    while i + 1 < len(turns):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            pairs.append((turns[i], turns[i + 1]))
            i += 2
        else:
            i += 1

    if not pairs:
        return []

    windows = []
    for start in range(0, len(pairs), window_step):
        chunk = pairs[start : start + window_turns]
        if not chunk:
            break
        window_conv = system[:]
        for user_msg, asst_msg in chunk:
            window_conv.extend([user_msg, asst_msg])
        windows.append(window_conv)

    return windows


def _clean(messages: "list[dict] | None") -> "list[dict] | None":
    if not messages:
        return None
    # Preserve STRUCTURED assistant tool_calls (fix #3): an assistant move is now
    # {"content": None, "tool_calls": [...]}, so DON'T str()-coerce content (None must
    # stay None, not become "None") and DON'T drop tool_calls. Non-assistant / text
    # messages keep str(content) as before. arguments inside tool_calls are already
    # JSON strings (Arrow-safe); the tokenizer decodes them back to dicts.
    cleaned: "list[dict]" = []
    for m in messages:
        c = m.get("content")
        nm = {"role": m["role"], "content": (None if c is None else str(c))}
        if m.get("tool_calls"):
            nm["tool_calls"] = m["tool_calls"]
        cleaned.append(nm)
    messages = cleaned
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if not messages:
        return None
    if sum(1 for m in messages if m["role"] == "assistant") < MIN_ASSISTANT_TURNS:
        return None
    return messages


def _per_turn_windows(conv: list[dict]) -> list[list[dict]]:
    """Split a per-turn-fresh tool_call episode (multiple system messages, one
    per turn + a reflect block) into independent per-turn SFT examples. Each
    block runs from one system message to the next; tool/role + multi-message
    turns are preserved. Mirrors the #1168 eval, where each turn is a fresh
    [system(rules+memory), user, ...inner tool loop...] conversation.

    Used instead of _sliding_windows when a conversation has >1 system message.
    """
    blocks: list[list[dict]] = []
    cur: list[dict] = []
    for m in conv:
        if m.get("role") == "system" and cur:
            blocks.append(cur)
            cur = []
        cur.append(m)
    if cur:
        blocks.append(cur)

    out: list[list[dict]] = []
    for b in blocks:
        while b and b[-1]["role"] != "assistant":
            b.pop()
        if b and sum(1 for m in b if m["role"] == "assistant") >= MIN_ASSISTANT_TURNS:
            out.append(b)
    return out


def _stats(conversations: list[list[dict]]) -> dict:
    turn_counts = [sum(1 for m in c if m["role"] == "assistant") for c in conversations]
    return {
        "total": len(conversations),
        "avg_assistant_turns": round(sum(turn_counts) / len(turn_counts), 2),
        "turn_distribution": dict(sorted(Counter(turn_counts).items())),
    }


def _ld_writable_cache_dir() -> "str | None":
    """First WRITABLE dir for the LD blueprint cache, or None if caching is off.

    The tournament container mounts /cache READ-ONLY. With the old hardcoded
    /cache/ld_mccfr default that silently cost us the whole expert: the parallel
    prebuild SOLVED the blueprint fine, then `_atomic_write_json` raised
    `OSError: [Errno 30] Read-only file system`, the converged blueprint (a local)
    was discarded, and every gen worker fell back to its own bounded cold solve —
    which `_MCCFR_MIN_USABLE_ITERS` correctly rejects (~180 iters vs the 5000 gate),
    so `get_expert_action` served the weaker epsilon-band heuristic and the DBR
    layer, which only runs on top of a non-empty blueprint, never executed at all.

    Probe order keeps the old preference: /cache first, so a genuinely writable
    /cache still shares one blueprint across rounds AND across containers (the
    env-server path reads the same cache); a per-run temp dir otherwise, which is
    all an in-process run needs since the workers only read it back within the
    same run. An explicit LD_MCCFR_CACHE_DIR always wins — including an empty
    value, which disables caching (`cache_file_path` returns None)."""
    explicit = os.environ.get("LD_MCCFR_CACHE_DIR")
    if explicit is not None:
        return explicit.strip() or None
    for cand in ("/cache/ld_mccfr", os.path.join(tempfile.gettempdir(), "ld_mccfr")):
        try:
            os.makedirs(cand, exist_ok=True)
            probe = os.path.join(cand, f".w{os.getpid()}")
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write("")
            os.remove(probe)
            return cand
        except Exception:
            continue  # read-only mount / no permission / missing parent
    return None


def _maybe_prebuild_ld_blueprint(env_name: str, num_workers: int) -> None:
    """Warm the Liar's Dice MCCFR blueprint ONCE before forking gen workers: a
    parallel multi-seed solve merged into a WRITABLE runtime cache (see
    _ld_writable_cache_dir), so every worker LOADS it (ms) instead of each doing a
    thin cold in-process solve that the usability gate would reject. Idempotent
    (skips if the cache file already exists, e.g. a warm cache across rounds).
    Best-effort — any failure leaves the per-worker in-process build as the runtime
    fallback. Helps BOTH the env-server and self-play LD paths (both consult the
    blueprint). Runs BEFORE the gen pool forks, so the resolved cache dir is
    exported into the environment the workers inherit."""
    if env_name != "liars_dice":
        return
    try:
        cache_dir = _ld_writable_cache_dir()
        if cache_dir:
            # Export so the forked gen workers resolve the SAME path we write to.
            os.environ["LD_MCCFR_CACHE_DIR"] = cache_dir
            print(f"[generate_trajectories] LD blueprint cache dir: {cache_dir}", flush=True)
        else:
            print("[generate_trajectories] LD blueprint cache DISABLED (no writable dir) — "
                  "workers fall back to the bounded in-process solve", flush=True)
        from our_envs import liar_dice_cfr as ldc
        from our_envs import prebuild_liar_dice_blueprint as pb
        total = 10  # symmetric solve total (== liar_dice_cfr._MCCFR_MAX_TOTAL)
        path = ldc.cache_file_path(total)
        if path and os.path.exists(path):
            print(f"[generate_trajectories] LD blueprint cache present, skip prebuild: {path}", flush=True)
            return
        cores = os.cpu_count() or 8
        workers = max(1, min(num_workers or cores, cores))
        # 6000 -> 28000/worker (2026-07-04): with the typical 8-16 gen workers the
        # merged multi-seed solve totals ~224k-448k iterations ~= the intended
        # 280k-class convergence — paid ONCE here in parallel before workers
        # fork, NOT per-worker on the cold path (the in-process fallback stays
        # at the bounded 60k/90s safety net).
        iters = int(os.environ.get("LD_PREBUILD_ITERS", "28000"))
        max_s = float(os.environ.get("LD_PREBUILD_SECONDS", "600"))
        print(f"[generate_trajectories] LD prebuild: {workers}x{iters} iters (per-worker cap {max_s:.0f}s)...", flush=True)
        bp = pb.build_parallel(total, workers, iters, max_seconds=(max_s if max_s > 0 else None))
        if path and bp:
            ldc._atomic_write_json(path, ldc._serialize_blueprint(bp))
            print(f"[generate_trajectories] LD prebuild wrote {len(bp)} infosets -> {path}", flush=True)
    except Exception as exc:
        print(f"[generate_trajectories] LD prebuild skipped (non-fatal): {type(exc).__name__}: {exc}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--environment_name", required=True)
    p.add_argument("--output_path",      required=True)
    p.add_argument("--num_games",   type=int, default=5000)
    p.add_argument("--max_turn",    type=int, default=30)
    p.add_argument("--window_turns", type=int, default=10,
                   help="Split each game into sub-conversations of this many (user,assistant) "
                        "pairs. Games shorter than this are kept whole. Default 10.")
    p.add_argument("--window_step", type=int, default=0,
                   help="Slide window by this many pairs (default: window_turns // 2).")
    p.add_argument("--num_workers", type=int, default=0,
                   help="Number of worker processes. Default 0 = num_servers.")
    p.add_argument("--seed", type=int, default=42)
    # Score-bias sampling. Env returns terminal reward in [0, 1] with
    # 0.5 = draw. Two filter modes:
    #
    # --wins-only: hard cutoff, drops score <= 0.5 (pure expert demos
    #   only). Simple but loses state-coverage diversity.
    #
    # --sample-by-score: probabilistic keep by score (tournament-winner
    #   recipe). Keep each scored game with probability
    #   clamp(score, 0, 1) ** score_power. No quartiles, no global pivot --
    #   a smooth bias toward winning games. At power=3.0 a 0.5 game survives
    #   ~12.5%, a 1.0 game always survives. Only leduc_poker enables this;
    #   liars_dice and gin_rummy keep all games (their experts already
    #   produce strong demos and need full state-coverage).
    p.add_argument("--wins-only", action="store_true",
                   help="Discard games with terminal score <= 0.5 (env returns [0,1] "
                        "where 0.5 = draw). Only when generator returns score.")
    p.add_argument("--sample-by-score", action="store_true",
                   help="Probabilistic score-power sampling: keep each game with "
                        "probability clamp(score,0,1) ** --score-power.")
    p.add_argument("--score-power", type=float, default=3.0,
                   help="Score-power exponent (default 3.0). keep_prob = "
                        "clamp(score,0,1) ** power; higher = stronger bias to wins.")
    p.add_argument("--per-game-timeout", type=int, default=120,
                   help="Per-game worker timeout in seconds. Protects against stuck env-server "
                        "or pyspiel hang. 0 disables (default 120s).")
    p.add_argument("--max-gen-seconds", type=int, default=0,
                   help="Hard time cap on the generation loop (seconds). When elapsed time exceeds "
                        "this, stop submitting new games and let in-flight games finish. 0 = no cap "
                        "(run all num_games regardless of time).")
    args = p.parse_args()
    if args.window_step == 0:
        args.window_step = args.window_turns // 2 or 1

    is_inprocess = runs_inprocess(args.environment_name)
    task_id_min, task_id_max = GAMES_TO_TASK_ID_RANGE[args.environment_name]

    if is_inprocess:
        # In-process pyspiel gen (teacher-vs-MCTS default, or teacher-vs-teacher
        # self-play): no env server, no HTTP. Play is CPU-bound in-process, so
        # parallelism is over CPU cores (override with --num_workers / GEN_WORKERS).
        # Force any score filter OFF: the teacher seat's moves are strong labels
        # regardless of the game outcome (and self-play keeps both seats), so we want
        # full state coverage, not a win-biased subset.
        opponent_payload = None
        env_pool = None
        num_servers = 0
        num_workers = args.num_workers or max(1, int(os.environ.get("GEN_WORKERS", str(os.cpu_count() or 8))))
        args.wins_only = False
        args.sample_by_score = False
    else:
        if args.environment_name not in _OPPONENT_CONFIG_PER_GAME:
            raise ValueError(
                f"Unknown environment {args.environment_name!r}. "
                f"Supported: {sorted(_OPPONENT_CONFIG_PER_GAME)}"
            )
        opponent_payload = _OPPONENT_CONFIG_PER_GAME[args.environment_name]
        reset_payload = {"task_id": task_id_min, "seed": 42, **opponent_payload}
        _, env_pool, num_servers, _, _ = init_env_pool(reset_payload)
        num_workers = args.num_workers or max(1, num_servers)

    # Pre-filter game IDs that deterministically crash the env-server's
    # MCTS opponent (pyspiel C++ edge case, no Python traceback). Each
    # occurrence wastes 5-10 seconds before the worker times out, so a
    # known-bad allowlist saves real time on large num_games sweeps.
    KNOWN_BAD_GAMES = {363100463}  # gin_rummy hand=8 knock=8 seed

    random.seed(args.seed)
    game_ids = random.sample(range(task_id_min + 1, task_id_max), args.num_games)
    game_ids = [g for g in game_ids if g not in KNOWN_BAD_GAMES]
    tasks = [
        (gid, None if is_inprocess else env_pool[i % num_servers]["base_url"], args.max_turn)
        for i, gid in enumerate(game_ids)
    ]

    # Progress printing cadence: aim for ~20 progress lines total over the
    # run, so logs stay readable for both small (1000-game) and large
    # (15000-game) batches. Always also prints on the final game and on
    # time-cap / error events regardless of interval.
    progress_every = max(1, args.num_games // 20)

    _kind = inprocess_kind(args.environment_name)
    if _kind == "selfplay":
        mode_desc, opp_desc = "self-play(in-process)", "teacher-self-play"
    elif _kind == "mcts":
        mode_desc, opp_desc = "teacher-vs-mcts(in-process)", "mcts(in-process)"
    elif _kind == "heuristic":
        mode_desc, opp_desc = "teacher-vs-heuristic(in-process)", "heuristic(random/mirror/sampled)"
    else:
        mode_desc = "env-server"
        opp_desc = f"{opponent_payload['opponent']}@{opponent_payload.get('mcts_max_simulations', 'N/A')}sims"
    print(
        f"[generate_trajectories] start env={args.environment_name} "
        f"mode={mode_desc} "
        f"num_games={args.num_games} max_turn={args.max_turn} "
        f"num_workers={num_workers} num_servers={num_servers} "
        f"opp={opp_desc}",
        flush=True,
    )
    print(
        f"[generate_trajectories] progress_every={progress_every} games (5% intervals)",
        flush=True,
    )

    # Warm the LD blueprint cache once (parallel multi-seed) BEFORE forking the
    # gen pool, so every worker loads it instead of cold-solving. No-op for other
    # envs; best-effort (never blocks generation).
    _maybe_prebuild_ld_blueprint(args.environment_name, num_workers)

    # Stage 1: stream collection (per-game timeout + exception handling).
    # We collect ALL results first regardless of score so Stage 2 has the
    # full distribution to compute quartile thresholds from. Counters are
    # updated for live telemetry; final filter happens after the loop.
    raw_results: list = []  # list of (messages, score) or (messages, None)
    skipped = 0
    worker_failed = 0
    timeout_failed = 0
    wins = 0
    losses = 0
    draws = 0
    no_score = 0
    start_time = time.time()
    time_cap_hit = False
    cancelled_pending = 0
    per_game_timeout = args.per_game_timeout if args.per_game_timeout > 0 else None
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(args.environment_name,),
    ) as pool:
        futures = {pool.submit(_worker_play, gid, ep, mt): gid for gid, ep, mt in tasks}
        completed = 0
        # audit P1: the old per-future result(timeout=...) was DEAD CODE —
        # as_completed only yields ALREADY-completed futures, so no timeout ever
        # fired and one hung worker (in-process pyspiel/search) stalled the whole
        # 3h task (nothing upstream has a timeout either). The real bound is this
        # OVERALL deadline on as_completed itself; on stall we cancel + terminate
        # and fall through to Stage 2 with everything collected so far.
        gen_deadline = None
        if args.max_gen_seconds > 0:
            gen_deadline = args.max_gen_seconds + max(per_game_timeout or 0, 300)
        stalled = False
        try:
          for future in as_completed(futures, timeout=gen_deadline):
            # Hard time cap: stop accepting new games once we hit the budget.
            # We let already-running games finish (cancelling them would lose
            # in-flight work and orphan env-server episodes). Pending futures
            # that haven't started yet are cancelled so the pool drains fast.
            if (
                args.max_gen_seconds > 0
                and not time_cap_hit
                and (time.time() - start_time) > args.max_gen_seconds
            ):
                time_cap_hit = True
                for f in futures:
                    if not f.done() and f.cancel():
                        cancelled_pending += 1
                print(
                    f"[generate_trajectories] TIME CAP {args.max_gen_seconds}s reached "
                    f"at {completed}/{args.num_games} games; cancelled "
                    f"{cancelled_pending} pending futures, draining in-flight",
                    flush=True,
                )
            if future.cancelled():
                continue

            game_id = futures[future]
            # Per-game timeout + worker exception handling so a single stuck
            # game or pyspiel crash does not abort the batch.
            try:
                result = future.result(timeout=per_game_timeout)
            except FutureTimeoutError:
                print(
                    f"[generate_trajectories] timeout (>{per_game_timeout}s) for game {game_id}",
                    flush=True,
                )
                timeout_failed += 1
                completed += 1
                continue
            except BrokenProcessPool:
                # audit P2: a segfaulting worker (pyspiel C++) breaks the whole
                # pool; every remaining future would raise this. Break LOUDLY and
                # keep everything collected so far instead of silently counting
                # hundreds of worker_failed.
                print(
                    f"[generate_trajectories] BROKEN POOL at game {game_id} "
                    f"({completed} done) — keeping collected results",
                    flush=True,
                )
                worker_failed += 1
                break
            except Exception as exc:
                print(
                    f"[generate_trajectories] worker exception for game {game_id}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                worker_failed += 1
                completed += 1
                continue

            score_repr = "n/a"
            if isinstance(result, list) and result and isinstance(result[0], tuple):
                # Self-play multi-view: [(rows, score), ...] — one view per seat,
                # rows already per-turn SFT examples ({"messages": [...]}). Flatten
                # each seat's rows into raw_results; BOTH seats are kept (no filter,
                # so ~50% per-seat win rate is expected for teacher-vs-teacher).
                for rows, score in result:
                    sc = score if isinstance(score, (int, float)) else None
                    if sc is not None:
                        if sc > 0.5:
                            wins += 1
                        elif sc < 0.5:
                            losses += 1
                        else:
                            draws += 1
                    for row in (rows or []):
                        msgs = row.get("messages") if isinstance(row, dict) else None
                        if msgs:
                            raw_results.append((game_id, msgs, sc))
            elif isinstance(result, tuple) and len(result) == 2:
                messages_raw, score = result
                score_repr = f"{score:.3f}"
                if score > 0.5:
                    wins += 1
                elif score < 0.5:
                    losses += 1
                else:
                    draws += 1
                raw_results.append((game_id, messages_raw, score))
            else:
                messages_raw = result
                no_score += 1
                raw_results.append((game_id, messages_raw, None))

            completed += 1
            # Print only on the cadence interval, on the final game, or
            # immediately after a time-cap event (so the cap is visible in
            # context with surrounding progress lines). Per-game spam was
            # readable for 2500-game GR but became thousands of lines for
            # 15000-game LP, drowning out useful events.
            should_print = (
                completed % progress_every == 0
                or completed == args.num_games
                or time_cap_hit
            )
            if should_print:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = args.num_games - completed
                eta_sec = remaining / rate if rate > 0 else 0.0
                scored = wins + losses + draws
                win_pct = (100.0 * wins / scored) if scored > 0 else 0.0
                print(
                    f"[generate_trajectories] game "
                    f"{completed}/{args.num_games} "
                    f"({100.0 * completed / args.num_games:.1f}%) "
                    f"wins={wins} losses={losses} draws={draws} win_rate={win_pct:.1f}% "
                    f"raw_total={len(raw_results)} timeout={timeout_failed} worker_fail={worker_failed} "
                    f"elapsed={elapsed:.0f}s "
                    f"rate={rate:.2f}games/s "
                    f"eta={eta_sec:.0f}s",
                    flush=True,
                )

        except FutureTimeoutError:
            stalled = True
            print(
                f"[generate_trajectories] STALL: overall gen deadline "
                f"{gen_deadline}s hit with {completed} games done — cancelling "
                f"pending, terminating workers, KEEPING collected results",
                flush=True,
            )
            for f2 in futures:
                f2.cancel()
            try:
                pool.shutdown(wait=False, cancel_futures=True)
                for proc in list(getattr(pool, "_processes", {}).values()):
                    try:
                        proc.terminate()
                    except Exception:
                        pass
            except Exception:
                pass

    # Stage 2: apply final filter.
    #   --sample-by-score -> probabilistic keep by score (tournament-winner
    #                        recipe): keep prob = clamp(score, 0, 1) ** power.
    #                        Higher power = steeper bias toward winning games.
    #                        Only leduc_poker uses this; LD/GR keep all.
    #   --wins-only       -> hard cutoff at 0.5 (legacy compat)
    #   neither           -> keep all raw_results
    conversations: list[list[dict]] = []
    filtered_by_score = 0
    score_kept = 0
    score_seen = 0
    # audit P2 determinism: as_completed yields in completion order (varies per
    # run), so sort by game id and use a seeded RNG for the keep-decisions —
    # identical --seed now reproduces the exact same dataset + split membership.
    raw_results.sort(key=lambda t: t[0])
    _stage2_rng = random.Random(args.seed)
    all_scores = [s for _, _, s in raw_results if s is not None]

    if args.sample_by_score and all_scores:
        # Probabilistic score-power sampling. For each scored game, keep it
        # with probability clamp(score, 0, 1) ** power. At power=3.0 a
        # 0.5-score game survives ~12.5% of the time and a 1.0-score game
        # always survives -> strongly biases the SFT set toward winning
        # demonstrations, without the hard cliff that --wins-only imposes.
        power = max(0.0, float(args.score_power))
        print(
            f"[generate_trajectories] score-power sampling: power={power} "
            f"n_scored={len(all_scores)} "
            f"score_min={min(all_scores):.3f} score_max={max(all_scores):.3f}",
            flush=True,
        )

        for _gid, messages_raw, score in raw_results:
            if score is None:
                cleaned = _clean(messages_raw)
                if cleaned is None:
                    skipped += 1
                else:
                    conversations.append(cleaned)
                continue
            score_seen += 1
            keep_prob = max(0.0, min(1.0, score)) ** power
            if _stage2_rng.random() >= keep_prob:
                filtered_by_score += 1
                continue
            cleaned = _clean(messages_raw)
            if cleaned is None:
                skipped += 1
            else:
                conversations.append(cleaned)
                score_kept += 1
    elif args.wins_only:
        # Legacy hard cutoff: drop score <= 0.5.
        for _gid, messages_raw, score in raw_results:
            if score is not None and score <= 0.5:
                filtered_by_score += 1
                continue
            cleaned = _clean(messages_raw)
            if cleaned is None:
                skipped += 1
            else:
                conversations.append(cleaned)
    else:
        # No filter — keep everything.
        for _gid, messages_raw, _ in raw_results:
            cleaned = _clean(messages_raw)
            if cleaned is None:
                skipped += 1
            else:
                conversations.append(cleaned)

    total_elapsed = time.time() - start_time
    scored_total = wins + losses + draws
    win_pct_total = (100.0 * wins / scored_total) if scored_total > 0 else 0.0
    cap_status = (
        f"time_cap_hit=True cancelled_pending={cancelled_pending}"
        if time_cap_hit
        else "time_cap_hit=False"
    )
    score_summary = ""
    if args.sample_by_score and all_scores:
        score_summary = f" score_kept={score_kept}/{score_seen}"

    print(
        f"[generate_trajectories] DONE collected={len(conversations)} "
        f"skipped={skipped} filtered_by_score={filtered_by_score} "
        f"timeout_failed={timeout_failed} worker_failed={worker_failed} "
        f"wins={wins} losses={losses} draws={draws} no_score={no_score} "
        f"win_rate={win_pct_total:.1f}% "
        f"{cap_status}{score_summary} "
        f"total_elapsed={total_elapsed:.0f}s",
        flush=True,
    )

    if not conversations:
        raise RuntimeError(
            f"No valid conversations generated (mode={'in-process' if is_inprocess else 'env-server'}). "
            + ("Check pyspiel + the in-process generator (teacher-vs-MCTS / self-play) "
               "+ per-game formatters."
               if is_inprocess else "Check ENVIRONMENT_SERVER_URLS.")
        )

    windowed: list[list[dict]] = []
    for conv in conversations:
        # Per-turn-fresh tool_call episodes carry one system message per turn;
        # split them per turn rather than into (user,assistant) sliding windows.
        if sum(1 for m in conv if m.get("role") == "system") > 1:
            windows = _per_turn_windows(conv)
        else:
            windows = _sliding_windows(conv, args.window_turns, args.window_step)
        windowed.extend(windows if windows else [conv])
    conversations = windowed

    print(f"[generate_trajectories] windows total={len(conversations)}", flush=True)
    print(f"[generate_trajectories] stats={_stats(conversations)}", flush=True)

    dataset = Dataset.from_list([{"messages": c} for c in conversations])
    splits = dataset.train_test_split(test_size=VALIDATION_RATIO, seed=args.seed)
    dd = DatasetDict({"train": splits["train"], "validation": splits["test"]})

    dd.save_to_disk(args.output_path)
    print(f"[generate_trajectories] saved -> {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
