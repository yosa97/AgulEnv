"""Registry mapping env names to their SFT expert trajectory generator.

When text_trainer sees an ENVIRONMENTTASK whose `environment_name` is
present here, it dispatches to the SFT pipeline (generate trajectories ->
tokenize -> train_sft_env). Envs not in this registry fall back to the
existing GRPO env training path.
"""

from typing import Callable

import os

# DEFAULT generator = TEACHER-vs-MCTS via the env-server (the per-game expert plays
# ONE seat against the env-server's MCTS@50 opponent — the SAME opponent the PvP eval
# uses). This aligns the training STATE DISTRIBUTION to eval, which is what fixed the
# clobber forfeit: teacher-vs-teacher SELF-PLAY produced off-distribution states, so at
# eval (vs MCTS) the LoRA hit unfamiliar boards → escaped to the base memory-writing
# prior → forfeit. Training vs the MCTS opponent removes that shift (clobber went 0-forfeit
# and from 0-10 to 44-44 vs the champion). The tournament trainer spins up one MCTS_API
# server per env (G.O.D trainer/runtime.py + core.constants.environments), so
# ENVIRONMENT_SERVER_URLS is populated at tournament time and this default holds there.
#
# The SAME per-game expert (get_expert_action, defined in each <game>_trajectories.py)
# drives our teacher on BOTH paths — the MCTS is only ever the OPPONENT, never our teacher.
#
# SELF-PLAY (envs/pvp_selfplay.py, in-process pyspiel, both seats, ~2-4x rows/game, no
# server) is now OPT-IN: set SELFPLAY_DISABLE=0 for A/B, and it is also the automatic
# fallback when no env-server URLs are present (bare local run) or pyspiel is missing —
# so a missing server/dependency degrades gracefully instead of failing the task.
_SELFPLAY_ENVS: frozenset = frozenset(
    {"liars_dice", "leduc_poker", "gin_rummy", "othello", "goofspiel", "clobber"}
)

# Envs whose IN-PROCESS gen uses the teacher-vs-MCTS recipe (OUR teacher on one
# seat, an in-process OpenSpiel MCTS opponent on the other). These are the
# sequential games MCTSBot supports; goofspiel (simultaneous move) is NOT here — it
# stays on in-process self-play under the in-process default (self-play is also its
# prior-winning gen).
_INPROCESS_MCTS_ENVS: frozenset = frozenset(
    {"liars_dice", "leduc_poker", "gin_rummy", "othello", "clobber"}
)

# env-server (MCTS-opponent) generators — the fallback path. Each is the SAME
# expert wrapped in an env-server rollout; self-play wraps the same expert in
# both-seats local play.
from our_envs.liar_dice_trajectories import generate_expert_episode as _liar_gen
from our_envs.leduc_poker_trajectories import generate_expert_episode as _leduc_gen
from our_envs.gin_rummy_trajectories import generate_expert_episode as _gin_gen
from our_envs.othello_trajectories import generate_expert_episode as _othello_gen
from our_envs.goofspiel_trajectories import generate_expert_episode as _goof_gen
from our_envs.clobber_trajectories import generate_expert_episode as _clobber_gen

_ENV_SERVER_REGISTRY: dict[str, Callable] = {
    "liars_dice":  _liar_gen,
    "leduc_poker": _leduc_gen,
    "gin_rummy":   _gin_gen,
    "othello":     _othello_gen,
    "goofspiel":   _goof_gen,
    "clobber":     _clobber_gen,
}


def _pyspiel_available() -> bool:
    try:
        import pyspiel  # noqa: F401
        return True
    except Exception:
        return False


def _gen_mode() -> str:
    """Select the SFT trajectory-gen path. DEFAULT = in-process teacher-vs-MCTS (the
    tournament-winner recipe): OUR per-game teacher plays one seat against an
    in-process OpenSpiel MCTS opponent on the other, on the live pyspiel state. This
    is the SAME eval-aligned teacher-vs-MCTS state distribution the old env-server
    path produced (which fixed the clobber forfeit), but with NO HTTP round-trip per
    turn — the env-server mcts-api managed well under 1 game/s on the long-game envs,
    starving othello/gin of data; in-process play is ~1-2 orders faster and scales
    across every core.

    A/B overrides:
      GEN_INPROCESS_MCTS=0  -> env-server HTTP teacher-vs-MCTS (the legacy path; also
                               the automatic fallback when pyspiel/open_spiel is absent)
      SELFPLAY_DISABLE=0    -> in-process teacher-vs-TEACHER self-play (both seats)

    Returns one of: "inprocess_mcts" | "selfplay" | "env_server".
    """
    if os.environ.get("SELFPLAY_DISABLE", "").strip().lower() in ("0", "false", "no"):
        return "selfplay" if _pyspiel_available() else "env_server"
    if os.environ.get("GEN_INPROCESS_MCTS", "1").strip().lower() in ("0", "false", "no", "off"):
        return "env_server"
    if not _pyspiel_available():
        # LOUD: a silently-missing pyspiel would revert to the SLOW env-server + MCTS
        # (HTTP) path without obvious notice — surface it in the logs.
        print(
            "[sft_env_configs] WARNING: pyspiel (open_spiel) not importable — in-process "
            "teacher-vs-MCTS unavailable, falling back to env-server + MCTS (HTTP) "
            "generators. Install open_spiel in the trainer image for the fast in-process "
            "default.",
            flush=True,
        )
        return "env_server"
    return "inprocess_mcts"


_REGISTRY_MODE = "env_server"  # set by _build_registry: inprocess_mcts | selfplay | env_server


def _build_registry() -> "dict[str, Callable]":
    global _REGISTRY_MODE
    mode = _gen_mode()
    _REGISTRY_MODE = mode
    if mode == "env_server":
        return dict(_ENV_SERVER_REGISTRY)
    from our_envs.pvp_selfplay import make_selfplay_generator
    if mode == "selfplay":
        return {name: make_selfplay_generator(name) for name in _SELFPLAY_ENVS}
    # inprocess_mcts: teacher-vs-MCTS for the sequential envs; goofspiel (simultaneous-
    # move) has no valid in-process MCTS opponent (MCTSBot is sequential-only — the
    # winner found the mcts opponent "flatly wrong" for goofspiel), so it runs
    # teacher-vs-HEURISTIC (random/mirror/sampled opponent per game_id), matching the
    # winner's goofspiel recipe. Any other non-MCTS env falls back to self-play.
    from our_envs.pvp_selfplay import make_teacher_vs_mcts_generator
    from our_envs.pvp_selfplay import make_teacher_vs_heuristic_goofspiel_generator
    reg: "dict[str, Callable]" = {}
    for name in _SELFPLAY_ENVS:
        if name in _INPROCESS_MCTS_ENVS:
            reg[name] = make_teacher_vs_mcts_generator(name)
        elif name == "goofspiel":
            reg[name] = make_teacher_vs_heuristic_goofspiel_generator()
        else:
            reg[name] = make_selfplay_generator(name)
    return reg


_SFT_REGISTRY: dict[str, Callable] = _build_registry()


def uses_selfplay(env_name: str) -> bool:
    """Back-compat: True iff `env_name` is generated via teacher-vs-TEACHER
    self-play (both seats). Distinct from the teacher-vs-MCTS in-process default."""
    return _REGISTRY_MODE == "selfplay" and env_name in _SELFPLAY_ENVS


def runs_inprocess(env_name: str) -> bool:
    """True iff `env_name`'s gen runs fully IN-PROCESS (no env-server HTTP): either
    teacher-vs-MCTS (default) or teacher-vs-teacher self-play. generate_trajectories
    uses this to skip init_env_pool and pass endpoint=None. Tracks the REAL registry
    mode (a pyspiel-less fallback reverts to env_server, where an endpoint IS
    needed)."""
    return _REGISTRY_MODE in ("inprocess_mcts", "selfplay") and env_name in _SFT_REGISTRY


def inprocess_kind(env_name: str) -> str:
    """For LOGGING only: which generator actually drives `env_name` — "mcts"
    (in-process teacher-vs-MCTS), "heuristic" (goofspiel teacher-vs-random/mirror/
    sampled), "selfplay" (in-process teacher-vs-teacher), or "env_server" (HTTP). In
    inprocess_mcts mode goofspiel runs teacher-vs-heuristic (MCTS is invalid for the
    simultaneous-move game), so keying the log purely on the mode would mislabel it."""
    if _REGISTRY_MODE == "env_server":
        return "env_server"
    if _REGISTRY_MODE == "inprocess_mcts":
        if env_name in _INPROCESS_MCTS_ENVS:
            return "mcts"
        if env_name == "goofspiel":
            return "heuristic"
    return "selfplay"

# Envs that route to SFT but NOT via a per-episode rollout generator. Intercode
# (NL2Bash) has no opponent: its gold bash commands are the expert, so
# envs.intercode_trajectories formats the mounted dataset into ReAct examples
# instead of rolling out games. These have no _SFT_REGISTRY entry (which feeds
# generate_trajectories); supports_sft still routes them to the SFT pipeline.
_SFT_NON_ROLLOUT_ENVS = {"intercode", "swe_infinite"}


def supports_sft(env_name: str) -> bool:
    return env_name in _SFT_REGISTRY or env_name in _SFT_NON_ROLLOUT_ENVS


def get_sft_trajectory_generator(env_name: str) -> Callable:
    if env_name not in _SFT_REGISTRY:
        raise ValueError(f"No SFT trajectory generator for env: {env_name!r}")
    return _SFT_REGISTRY[env_name]
