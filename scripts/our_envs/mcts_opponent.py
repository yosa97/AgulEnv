"""In-process OpenSpiel MCTS opponent for teacher-vs-MCTS SFT trajectory gen.

Replaces the env-server (mcts-api) HTTP opponent with the SAME MCTS run directly
on the live pyspiel state — no network round-trip per turn, so gen throughput
jumps ~1-2 orders of magnitude (the HTTP mcts-api path managed well under 1 game/s
on the long-game envs othello/gin, starving them of data; in-process play is
CPU-bound and scales across every core).

The opponent is an OpenSpiel MCTSBot with a RandomRolloutEvaluator(n_rollouts=1),
which mirrors the validator eval opponent: G.O.D env eval / the PvP MCTS baseline
use mcts_num_rollouts=1 (see _OPPONENT_CONFIG_PER_GAME in generate_trajectories.py).
So the teacher trains against the same-strength MCTS it is graded against — the
eval-aligned state distribution that fixed the clobber forfeit — just without the
HTTP transport.

Never raises: an unsupported game / bad state returns None from mcts_step_or_none
so the caller falls back to a random legal move (keeps gen robust, never zeroes a
dataset).
"""

import numpy as np


# Per-env MCTS opponent simulation band. The eval opponent strength (mirrored from
# G.O.D/validator ENVIRONMENTS[<env>].eval_payload_extra) is liars_dice 225, and 50
# for leduc / gin / othello / clobber. We sample within a band that BRACKETS the
# eval value rather than pinning one fixed count, for two reasons: (a) state-
# distribution variety — a spread of opponent strengths visits a wider set of boards
# across games than a single opponent; (b) throughput — the long-game envs
# (gin/othello) are dominated by opponent search time, so the lower end keeps games
# fast while the upper end still reaches eval strength. leduc is short + cheap, so it
# is pinned at the exact eval value.
_MCTS_SIMS_BAND: dict = {
    "othello":     (25, 75),
    "gin_rummy":   (25, 50),
    "leduc_poker": (50, 50),
    "clobber":     (25, 75),
    "liars_dice":  (150, 225),
}
_DEFAULT_BAND = (50, 50)


def mcts_sims_for(env_name: str, rng) -> int:
    """Sample this game's opponent simulation count from the env's band (seeded by
    the caller's per-game rng, so identical game_id -> identical opponent)."""
    lo, hi = _MCTS_SIMS_BAND.get(env_name, _DEFAULT_BAND)
    return rng.randint(lo, hi) if hi > lo else lo


def make_mcts_bot(game, simulations: int, seed: int):
    """Build an in-process OpenSpiel MCTSBot (RandomRolloutEvaluator, n_rollouts=1)
    for the OPPONENT seat — the same rollout count as the validator eval opponent.
    solve=False so it stays robust on the imperfect-information envs (leduc/gin/
    liars) where proven-win solving is ill-defined."""
    from open_spiel.python.algorithms import mcts
    rs = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=rs)
    return mcts.MCTSBot(
        game,
        2.0,                # uct_c exploration constant (OpenSpiel default)
        int(simulations),   # max_simulations
        evaluator,
        solve=False,
        random_state=rs,
        verbose=False,
    )


def mcts_step_or_none(bot, state):
    """Return the MCTS action id for `state`, or None on any failure (unsupported
    node type, pyspiel C++ edge case). The caller falls back to a random legal
    move so a single bad state never aborts the game or the batch."""
    try:
        return int(bot.step(state))
    except Exception:
        return None
