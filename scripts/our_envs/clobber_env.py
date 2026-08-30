"""GRPO rollout driver for clobber.

Same gap as othello: a task-id range, a self-contained parser and expert, and a
working env-server contract on the SFT side, but no GRPO driver and no
``_REGISTRY`` entry.  See ``othello_env`` for the rationale behind the batched
cohort loop and the opponent league; both games share the engine.

Clobber is a last-move-wins combinatorial game, so the terminal result is the
only honest signal — there is no meaningful midgame material count to shape on.
Reward is the normalised game result minus a small penalty per illegal action.
"""

from our_envs.batched_rollout import BoardGameEnv, GameSpec
from our_envs.clobber_trajectories import _ClobberObsTransform, _SYSTEM_PROMPT
from our_envs.shared_env import rollout_reward_func  # re-exported for the registry

# Clobber games end when a player is stuck; boards here run well under 70 plies.
_MAX_TURN = 70

_ENV = BoardGameEnv(
    GameSpec(
        name="clobber",
        system_prompt=_SYSTEM_PROMPT,
        obs_transform_factory=_ClobberObsTransform,
        max_turn=_MAX_TURN,
        max_prompt_len=8192,
    ),
    task_key="clobber",
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
