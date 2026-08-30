"""
Environment training configuration registry.

Each entry in ``_REGISTRY`` describes everything ``train_grpo_env.py`` needs to
set up training for a given game environment:

- Which rollout / reward callables to use.
- A ``curriculum_factory`` callable that receives ``training_args`` (already
  adjusted for the training mode) and returns the env's ``CurriculumScheduler``
  subclass, including any extra env-specific constructor params.
- Per-mode overrides via ``ModeConfig`` instances (``reasoning``, ``no_mask``,
  ``full_prompt``).  Only specify fields that deviate from the mode default.
- ``vllm_max_model_length`` — env-specific initial context window size; used by
  ``grpo_env_config.py`` to build the subprocess CLI command.

See REFACTOR_PLAN.md §Part 2 for the full two-layer config flow.

Adding a new per-mode field
---------------------------
1. Add it to ``ModeConfig`` with ``None`` default.
2. Read and apply it in the mode-dispatch block of ``train_grpo_env.py``.
3. Set it in whichever registry entries need a non-default value.
Zero changes to ``EnvTrainingConfig`` required.

Adding a new per-env (but not per-mode) field
----------------------------------------------
1. Add it to ``EnvTrainingConfig`` with the current shared default.
2. Set it in whichever registry entries need a non-default value.
"""

from dataclasses import dataclass, field, fields
from typing import Callable

from our_envs.alf_world_env import (
    alfworld_rollout_first_prompt_and_completion_parallelized as _alf_rollout_last,
    alfworld_rollout_full_prompt_and_completion_parallelized  as _alf_rollout_full,
    alfworld_rollout_reward_func                              as _alf_reward,
)
from our_envs.clobber_env import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _clobber_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _clobber_rollout_last,
    rollout_reward_func                                        as _clobber_reward,
)
from our_envs.othello_env import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _othello_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _othello_rollout_last,
    rollout_reward_func                                        as _othello_reward,
)
from our_envs.gin_rummy_env import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _gin_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _gin_rollout_last,
    rollout_reward_func                                        as _gin_reward,
    _curriculum_factory                                        as _gin_curriculum,
)
from our_envs.gin_rummy_opponent_modeling import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _gin_opp_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _gin_opp_rollout_last,
    rollout_reward_func                                        as _gin_opp_reward,
    _curriculum_factory                                        as _gin_opp_curriculum,
)
from our_envs.liar_dice_opponent_modeling import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _liar_opp_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _liar_opp_rollout_last,
    rollout_reward_func                                        as _liar_opp_reward,
    _curriculum_factory                                        as _liar_opp_curriculum,
)
from our_envs.leduc_poker_opponent_modeling import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _leduc_opp_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _leduc_opp_rollout_last,
    rollout_reward_func                                        as _leduc_opp_reward,
    _curriculum_factory                                        as _leduc_opp_curriculum,
)
from our_envs.goof_spiel_env import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _goof_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _goof_rollout_last,
    rollout_reward_func                                        as _goof_reward,
    _curriculum_factory                                        as _goof_curriculum,
)
from our_envs.leduc_poker_env import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _leduc_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _leduc_rollout_last,
    rollout_reward_func                                        as _leduc_reward,
    _curriculum_factory                                        as _leduc_curriculum,
)
from our_envs.liar_dice_env import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _liar_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _liar_rollout_last,
    rollout_reward_func                                        as _liar_reward,
    _curriculum_factory                                        as _liar_curriculum,
)


# ---------------------------------------------------------------------------
# ModeConfig — overrides for a single training mode
# ---------------------------------------------------------------------------

@dataclass
class ModeConfig:
    """
    Per-training-mode overrides for one environment.

    All fields default to ``None``, meaning "use the mode-level default
    from ``train_grpo_env.py``".  Only populate fields that need to deviate.

    To add a new per-mode configurable: add it here with ``None`` default,
    then read + apply it in the mode-dispatch block of ``train_grpo_env.py``.
    """
    initial_max_turn:    int | None  = None
    rollouts_per_stage:  int | None  = None
    # None → mode default (GRPOTrainer for reasoning/no_mask,
    #                       ActionMaskedGRPOTrainer for full_prompt)
    # reasoning always uses GRPOTrainer regardless of this field.
    trainer_class:       "type | None" = None
    # None → mode default (2048 for reasoning, 16 for no_mask/full_prompt)
    max_completion_length: int | None = None

    # None → fall back to the env-level ``EnvTrainingConfig.num_generations``.
    num_generations: int | None = None

    # Fully-specified hyperparams for one (env, mode, size).  A size_label
    # present here wins over ``DEFAULT_HYPERPARAMS`` in train_grpo_env.py.
    # Annotated as a string: SizeHyperparams is defined further down.
    per_size: "dict[str, SizeHyperparams]" = field(default_factory=dict)

    def apply_scalars(self, args) -> None:
        """Apply the scalar overrides that are not part of SizeHyperparams.

        Called by train_grpo_env.py *after* ``SizeHyperparams.apply``, so these
        win over both the CLI defaults and the per-size table.
        """
        if self.initial_max_turn is not None:
            args.initial_max_turn = self.initial_max_turn
        if self.rollouts_per_stage is not None:
            args.rollouts_per_stage = self.rollouts_per_stage


# ---------------------------------------------------------------------------
# SizeHyperparams — co-tuned VRAM/dynamics params for one (env, mode, size)
# ---------------------------------------------------------------------------
@dataclass
class SizeHyperparams:
    """Co-tuned VRAM/dynamics params for one (env, mode, size). All fields required."""
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_generations:             int
    vllm_gpu_memory_utilization: float
    beta:                        float

    def apply(self, args) -> None:
        for f in fields(self):
            setattr(args, f.name, getattr(self, f.name))


# ---------------------------------------------------------------------------
# EnvTrainingConfig — full config for one environment
# ---------------------------------------------------------------------------

@dataclass
class EnvTrainingConfig:
    rollout_full: Callable
    rollout_last: Callable
    reward_func:  Callable

    # Curriculum factory.  Receives training_args (already adjusted for the
    # training mode) and returns a CurriculumScheduler (or subclass).
    # None = this env uses no curriculum scheduler.
    curriculum_factory: Callable | None = None

    # Initial vllm context window size for this env.
    # Used by grpo_env_config.py to build --vllm_max_model_length in the CLI.
    # (Reasoning mode adds 2048 on top of this at runtime.)
    vllm_max_model_length: int = 5248

    # Per-env generation parameters.
    num_generations: int   = 4
    temperature:     float = 1.0
    top_k:           int   = 0

    # GRPO inner-loop iterations (mu).  Applied in train_grpo_env.py.
    num_iterations:  int   = 2

    # Per-mode overrides.  Omit or leave fields as None to use mode defaults.
    reasoning:  ModeConfig = field(default_factory=ModeConfig)
    no_mask:    ModeConfig = field(default_factory=ModeConfig)
    full_prompt: ModeConfig = field(default_factory=ModeConfig)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, EnvTrainingConfig] = {
    # Key must match GAMES_TO_TASK_ID_RANGE in shared_env.py exactly.
    "goofspiel": EnvTrainingConfig(
        rollout_full=_goof_rollout_full,
        rollout_last=_goof_rollout_last,
        reward_func=_goof_reward,
        curriculum_factory=_goof_curriculum,
        reasoning=ModeConfig(initial_max_turn=1),
        no_mask=ModeConfig(initial_max_turn=1),
    ),
    "gin_rummy": EnvTrainingConfig(
        rollout_full=_gin_rollout_full,
        rollout_last=_gin_rollout_last,
        reward_func=_gin_reward,
        curriculum_factory=_gin_curriculum,
        reasoning=ModeConfig(initial_max_turn=8),
        no_mask=ModeConfig(initial_max_turn=4, rollouts_per_stage=512),
        full_prompt=ModeConfig(initial_max_turn=8),
    ),
    "gin_rummy_opponent_modeling": EnvTrainingConfig(
        # Matches round-1 winner config (GR): no vllm_max_model_length override,
        # no num_generations override, ModeConfig overrides per mode.
        rollout_full=_gin_opp_rollout_full,
        rollout_last=_gin_opp_rollout_last,
        reward_func=_gin_opp_reward,
        curriculum_factory=_gin_opp_curriculum,
        reasoning=ModeConfig(initial_max_turn=8),
        no_mask=ModeConfig(initial_max_turn=4, rollouts_per_stage=512),
        full_prompt=ModeConfig(initial_max_turn=8),
    ),
    "liars_dice": EnvTrainingConfig(
        rollout_full=_liar_rollout_full,
        rollout_last=_liar_rollout_last,
        reward_func=_liar_reward,
        curriculum_factory=_liar_curriculum,
        reasoning=ModeConfig(rollouts_per_stage=2048, initial_max_turn=1),
        no_mask=ModeConfig(rollouts_per_stage=2048, initial_max_turn=1),
        full_prompt=ModeConfig(rollouts_per_stage=2048, initial_max_turn=1),
        num_generations=8,
        temperature=2.0,
        top_k=5,
    ),
    "liars_dice_opponent_modeling": EnvTrainingConfig(
        # Matches round-2 winner config (LD). ModeConfig is intentionally
        # empty so per-bucket `rollouts_per_stage` / `initial_max_turn` from
        # LIARS_DICE_GRPO_CONFIG in grpo_env_config.py flow through CLI args
        # untouched. Forcing fixed values here would clobber the bucket-level
        # `rollouts_per_stage=512` for 6_9_b (7B models) since the trainer
        # applies ModeConfig overrides *after* CLI parse.
        rollout_full=_liar_opp_rollout_full,
        rollout_last=_liar_opp_rollout_last,
        reward_func=_liar_opp_reward,
        curriculum_factory=_liar_opp_curriculum,
        num_generations=8,
    ),
    "leduc_poker": EnvTrainingConfig(
        rollout_full=_leduc_rollout_full,
        rollout_last=_leduc_rollout_last,
        reward_func=_leduc_reward,
        curriculum_factory=_leduc_curriculum,
        num_generations=8,
        temperature=2.0,
        top_k=5,
    ),
    "leduc_poker_opponent_modeling": EnvTrainingConfig(
        # Matches latest tournament position-1 winner config (LP):
        # num_generations=8 (not 4), temperature=2.0, top_k=5, no
        # ModeConfig overrides (uses registry defaults per-mode).
        rollout_full=_leduc_opp_rollout_full,
        rollout_last=_leduc_opp_rollout_last,
        reward_func=_leduc_opp_reward,
        curriculum_factory=_leduc_opp_curriculum,
        num_generations=8,
        temperature=2.0,
        top_k=5,
    ),
    # Board games: whole-game rollouts through the batched cohort engine, with
    # an opponent league in place of a fixed MCTS budget.  No curriculum_factory
    # -- the league adapts opponent strength, which is the curriculum that
    # matters here; truncating a board game by turn count would just remove the
    # terminal result these rewards are built on.
    "othello": EnvTrainingConfig(
        rollout_full=_othello_rollout_full,
        rollout_last=_othello_rollout_last,
        reward_func=_othello_reward,
        vllm_max_model_length=9216,
        num_generations=8,
        reasoning=ModeConfig(max_completion_length=2048),
        no_mask=ModeConfig(max_completion_length=24),
        full_prompt=ModeConfig(max_completion_length=24),
    ),
    "clobber": EnvTrainingConfig(
        rollout_full=_clobber_rollout_full,
        rollout_last=_clobber_rollout_last,
        reward_func=_clobber_reward,
        vllm_max_model_length=9216,
        num_generations=8,
        reasoning=ModeConfig(max_completion_length=2048),
        no_mask=ModeConfig(max_completion_length=24),
        full_prompt=ModeConfig(max_completion_length=24),
    ),
    "alfworld": EnvTrainingConfig(
        rollout_full=_alf_rollout_full,
        rollout_last=_alf_rollout_last,
        reward_func=_alf_reward,
    ),
}


# ---------------------------------------------------------------------------
# Variant routing
# ---------------------------------------------------------------------------

# Change this to select a non-default variant for a base environment name.
_VARIANT_OVERRIDES: dict[str, str] = {
     "gin_rummy":   "gin_rummy_opponent_modeling",
     "liars_dice":  "liars_dice_opponent_modeling",
     "leduc_poker": "leduc_poker_opponent_modeling",
}


def get_env_config(name: str) -> EnvTrainingConfig:
    """Look up the training config for a named environment.

    If ``name`` has an entry in ``_VARIANT_OVERRIDES``, that registry key is
    used instead — allowing a single code-level switch between implementations
    without changing the caller's environment name.

    Raises ``ValueError`` with a helpful message if the name is unknown.
    """
    resolved = _VARIANT_OVERRIDES.get(name, name)
    if resolved not in _REGISTRY:
        raise ValueError(
            f"Unknown environment: {name!r} (resolved to {resolved!r}). "
            f"Known environments: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[resolved]
