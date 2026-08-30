"""Adaptive opponent league for GRPO game environments.

Every environment in this repo trains against a fixed opponent: a single MCTS
simulation budget, hardcoded per module (``_MCTS_SIMS = 25`` in
``gin_rummy_opponent_modeling.py``, ``mcts_max_simulations: 50`` everywhere
else).  A fixed opponent is the wrong training signal for two reasons.

  * If it is weaker than the evaluation baseline, the policy is optimised
    against something easier than what scores it.
  * Once the policy reliably beats it, every episode in a GRPO group returns
    the same terminal reward.  Group-relative advantage is then identically
    zero and the batch contributes no gradient — the run keeps burning rollouts
    and stops learning.

This module replaces that with prioritised fictitious self-play (PFSP)
matchmaking: a pool of opponents of differing strength, a running win-rate
estimate per opponent, and sampling weighted toward the opponents that keep the
agent nearest a target win rate — the matches that actually carry information.

The pool is deliberately anchored on the evaluation baseline (MCTS @ 50
simulations) and extends above it, so the policy is never optimised against a
strictly easier distribution than the one that scores it.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    """Env-var float that never raises at import time."""
    try:
        return float(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name) or default))
    except (TypeError, ValueError):
        return default


# The PvP eval baseline is MCTS @ 50 simulations (generate_trajectories.py:39).
# The pool brackets it rather than sitting under it.
DEFAULT_SIMS_LADDER = (50, 100, 200, 400)

# Win rate the matchmaker steers toward.  0.5 maximises the variance of the
# terminal reward inside a GRPO group, which is exactly what the advantage
# estimate needs.
TARGET_WIN_RATE = _env_float("LEAGUE_TARGET_WIN_RATE", 0.5)

# Width of the preference bell around the target, in win-rate units.  Wider =
# flatter sampling over the pool; narrower = concentrate on the informative band.
BAND_WIDTH = max(1e-3, _env_float("LEAGUE_BAND_WIDTH", 0.22))

# Pseudo-counts, so an opponent with no games yet is treated as being at the
# target rather than at 0% or 100%.
PRIOR_GAMES = max(1, _env_int("LEAGUE_PRIOR_GAMES", 4))

# Floor on every opponent's sampling probability, so no opponent is ever fully
# starved and the win-rate estimates stay fresh.
EXPLORE_FLOOR = min(0.5, max(0.0, _env_float("LEAGUE_EXPLORE_FLOOR", 0.10)))


@dataclass
class Opponent:
    """One entry in the league."""

    key: str
    payload: dict
    wins: float = 0.0
    games: float = 0.0

    def win_rate(self, prior_games: int = PRIOR_GAMES, prior_rate: float = TARGET_WIN_RATE) -> float:
        """Beta-smoothed win rate, so a fresh opponent starts at the target."""
        return (self.wins + prior_rate * prior_games) / (self.games + prior_games)


@dataclass
class OpponentLeague:
    """Sampling pool of opponents with PFSP-style matchmaking."""

    opponents: list[Opponent]
    target_win_rate: float = TARGET_WIN_RATE
    band_width: float = BAND_WIDTH
    explore_floor: float = EXPLORE_FLOOR
    prior_games: int = PRIOR_GAMES
    total_games: int = field(default=0)

    # ---------------------------------------------------------------- build
    @classmethod
    def from_sims_ladder(
        cls,
        base_payload: dict,
        sims_ladder: "tuple[int, ...]" = DEFAULT_SIMS_LADDER,
        **kwargs,
    ) -> "OpponentLeague":
        """Build a league from a ladder of MCTS simulation budgets.

        ``base_payload`` supplies every field the env-server expects other than
        ``mcts_max_simulations`` (typically ``{"opponent": "mcts",
        "mcts_num_rollouts": 1}``).
        """
        env_ladder = os.environ.get("LEAGUE_SIMS_LADDER")
        if env_ladder:
            try:
                sims_ladder = tuple(
                    int(x) for x in env_ladder.replace(",", " ").split() if x.strip()
                )
            except ValueError:
                pass
        if not sims_ladder:
            sims_ladder = DEFAULT_SIMS_LADDER

        opponents = []
        for sims in sims_ladder:
            payload = dict(base_payload)
            payload["mcts_max_simulations"] = int(sims)
            opponents.append(Opponent(key=f"mcts@{int(sims)}", payload=payload))
        return cls(opponents=opponents, **kwargs)

    # --------------------------------------------------------------- sample
    def weights(self) -> list[float]:
        """Sampling weight per opponent: a bell centred on the target win rate.

        An opponent the agent beats 95% of the time and one it loses to 95% of
        the time both produce zero-variance groups; both are down-weighted.  The
        explore floor keeps every opponent reachable so stale estimates recover.
        """
        raw = []
        for opp in self.opponents:
            gap = opp.win_rate(self.prior_games, self.target_win_rate) - self.target_win_rate
            raw.append(math.exp(-(gap * gap) / (2.0 * self.band_width * self.band_width)))
        total = sum(raw)
        n = len(raw)
        if n == 0:
            return []
        if total <= 0.0:
            return [1.0 / n] * n
        floor = self.explore_floor / n
        return [(1.0 - self.explore_floor) * (w / total) + floor for w in raw]

    def sample(self, rng: "random.Random | None" = None) -> Opponent:
        """Draw an opponent. ``rng`` is required for reproducible rollouts."""
        if not self.opponents:
            raise ValueError("OpponentLeague is empty")
        r = rng or random
        return r.choices(self.opponents, weights=self.weights(), k=1)[0]

    # --------------------------------------------------------------- update
    def record(self, key: str, score: float) -> None:
        """Record one finished game. ``score`` in [0, 1]: 1 win, 0.5 draw, 0 loss."""
        try:
            s = float(score)
        except (TypeError, ValueError):
            return
        s = max(0.0, min(1.0, s))
        for opp in self.opponents:
            if opp.key == key:
                opp.wins += s
                opp.games += 1.0
                self.total_games += 1
                return

    # ----------------------------------------------------------------- logs
    def status(self) -> dict:
        w = self.weights()
        return {
            "total_games": self.total_games,
            "opponents": [
                {
                    "key": o.key,
                    "games": int(o.games),
                    "win_rate": round(o.win_rate(self.prior_games, self.target_win_rate), 3),
                    "p_sample": round(wi, 3),
                }
                for o, wi in zip(self.opponents, w)
            ],
        }

    def format_status(self) -> str:
        parts = [
            f"{o['key']}(n={o['games']} wr={o['win_rate']:.2f} p={o['p_sample']:.2f})"
            for o in self.status()["opponents"]
        ]
        return "[LEAGUE] " + " ".join(parts)
