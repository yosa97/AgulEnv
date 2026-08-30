"""GRPO rollout driver for othello.

Othello already had a task-id range (``shared_env.GAMES_TO_TASK_ID_RANGE``), a
self-contained board parser and expert, and a working env-server contract used
by the SFT trajectory generator — but no GRPO driver and no ``_REGISTRY`` entry,
so an ENVIRONMENT task naming ``othello`` could only ever raise ``ValueError``.
This module supplies that layer.

Two things differ from the older per-game drivers on purpose:

* Rollouts run through ``batched_rollout.run_cohort``, which advances every
  episode in lockstep and issues ONE batched generation per turn instead of one
  single-prompt generation per episode per turn behind a size-1 semaphore.
* The opponent is drawn from ``OpponentLeague`` — a pool anchored on the
  evaluation baseline (MCTS @ 50) and extending above it, re-weighted from the
  agent's running win rate — instead of a hardcoded simulation budget.

Reward is the terminal game result, normalised to [-1, 1], minus a small
penalty per illegal action.  There is no dense positional shaping: othello disc
counts are famously anti-correlated with winning in the midgame, and the audit's
own finding on the other envs was that shaping had grown large enough to
outweigh the result.
"""

from our_envs.batched_rollout import BoardGameEnv, GameSpec
from our_envs.othello_trajectories import _OthelloObsTransform, _SYSTEM_PROMPT
from our_envs.shared_env import rollout_reward_func  # re-exported for the registry

# Othello runs at most 60 placements plus passes; 70 covers the tail safely.
_MAX_TURN = 70

_ENV = BoardGameEnv(
    GameSpec(
        name="othello",
        system_prompt=_SYSTEM_PROMPT,
        obs_transform_factory=_OthelloObsTransform,
        max_turn=_MAX_TURN,
        max_prompt_len=8192,
    ),
    task_key="othello",
)


def rollout_full_prompt_and_completion_parallelized_curriculum(prompts, trainer, max_turns=_MAX_TURN):
    """Whole-game rollout with action masking over every model turn."""
    return _ENV.rollout_full(prompts, trainer, max_turns=max_turns)


def rollout_last_prompt_and_completion_parallelized_curriculum(prompts, trainer, max_turns=_MAX_TURN):
    """Whole-game rollout, training on the final turn's tokens only."""
    return _ENV.rollout_last(prompts, trainer, max_turns=max_turns)


__all__ = [
    "rollout_full_prompt_and_completion_parallelized_curriculum",
    "rollout_last_prompt_and_completion_parallelized_curriculum",
    "rollout_reward_func",
]
