"""Per-turn-fresh tool_call episode runner for PvP SFT distillation.

VERIFIED upstream (core/pvp/bot.py:6-7,180-184): the #1168 PvP eval rebuilds the
conversation FRESH every turn — on turn N the model sees only
[system(rules + CURRENT memory render + tool guidance), user(state)] plus the
turn-local inner tool-call loop; the ONLY cross-turn state is SlotMemory.

So a PvP game is distilled into N INDEPENDENT per-turn SFT examples (one per
move) + one post-game reflect example, NOT one growing conversation. This
runner plays a game vs the env-server MCTS opponent with the per-game expert
policy and emits that structure. Each per-turn example:

    system: <rules> + <memory render AS OF this turn> + _TOOL_GUIDANCE
    user:   Current state:\n<board>\n\nYou are Player N.\nLegal actions:\n<...>
    [assistant <tool_call> memory write -> {role:tool} ok-result]*   (policy)
    assistant <tool_call> game_action(action_id)                     (ends turn)

It returns ONE flat message list = concat of all per-turn blocks + the reflect
block, each block starting with a system message, so
generate_trajectories._per_turn_windows() splits them back into independent
examples. Per-game memory content comes from a MemoryPolicy (rich, deterministic,
never-raises). Everything here is never-raise: a policy failure degrades to no
note for that turn, never a crashed game.
"""

import re

import requests

from our_envs.pvp_tool_calling import (
    MemoryState,
    WORKING,
    LONG_TERM,
    build_system_prompt,
    toolcall_user_prompt_from_reformatted,
    memory_write_fragment,
    assistant_turn_message,
    reflection_user_prompt,
    memory_writes_enabled,
)


class MemoryPolicy:
    """Per-game memory-note policy. Deterministic + NEVER-RAISES. Subclasses
    override the hooks; the base writes nothing (memory stays empty — still a
    valid per-turn-fresh episode, just without the opponent-modelling edge).

    Methods receive the *reformatted* observation (old envelope) so policies can
    regex out game signals; they return lists of (area, op, slot, content)
    writes where area in {WORKING, LONG_TERM}, op in {"rewrite","append"}.
    """

    def observe(self, reformatted_obs: str) -> None:
        """Accumulate cross-turn signals (e.g. opponent's actions). Called once
        per turn before turn_writes."""

    def turn_writes(self, reformatted_obs: str, action_id: str, state: MemoryState, turn_idx: int) -> list:
        return []

    def reflect_writes(self, outcome: str, state: MemoryState) -> list:
        return []


def _safe(fn, *args, default=None):
    try:
        return fn(*args)
    except Exception:
        return default


_BODY_RE = re.compile(r"Current State:\n(.*?)\n\n(?:You are Player|Legal Actions:)", re.DOTALL)


def _state_body(reformatted_obs: str) -> str:
    """Extract just the board/state body from a reformatted observation (for the
    reflect 'Final state:' line). Falls back to the whole obs."""
    m = _BODY_RE.search(reformatted_obs or "")
    return m.group(1) if m else (reformatted_obs or "")


def run_toolcall_episode(
    *,
    game_name: str,
    game_id: int,
    env_endpoint: str,
    opponent_payload: dict,
    max_turn: int,
    rules_prompt: str,
    expert_action_fn,
    obs_transform,
    policy: "MemoryPolicy | None" = None,
    request_timeout: int = 2400,
) -> "tuple[list[dict], float] | None":
    """Play one game and emit the per-turn-fresh SFT structure.

    expert_action_fn(messages) -> bare action-id string (reuses the game's
        existing get_expert_action, fed a 1-message [{"role":"user",...}] list).
    obs_transform(raw_obs) -> reformatted observation string (game-specific
        extract + reformat_to_pvp envelope).
    Returns (flat_messages, final_reward) or None on env-server failure.
    """
    policy = policy or MemoryPolicy()
    # train_omit_memory guard (audit P2): PVP_MEMORY_WRITES=1 WITHOUT
    # PVP_TRAIN_NO_MEMORY=0 would teach memory tool-calls that the row's own
    # system prompt / tools list omits — writes require the memory surface.
    from our_envs.pvp_tool_calling import train_omit_memory
    writes_on = memory_writes_enabled() and not train_omit_memory()
    state = MemoryState()

    try:
        res = requests.post(
            f"{env_endpoint}/reset",
            json={"task_id": game_id, "seed": game_id, **opponent_payload},
            timeout=request_timeout,
        )
        res.raise_for_status()
        block = res.json()["result"]
        episode_id = block.get("episode_id", "")
        raw_obs = block.get("observation", "")
    except Exception as exc:
        print(f"[env] Reset failed (game {game_id}): {exc}")
        return None

    flat: list[dict] = []
    final_reward = 0.0
    last_reformatted = ""

    for turn_idx in range(max_turn):
        reformatted = _safe(obs_transform, raw_obs, default="") or ""
        last_reformatted = reformatted
        _safe(policy.observe, reformatted)
        action = expert_action_fn([{"role": "user", "content": reformatted}])

        # System prompt renders memory AS OF the start of this turn (before this
        # turn's writes); writes then mutate `state` so the NEXT turn's system
        # shows them — exactly the eval's per-turn re-render of live SlotMemory.
        # #1201 single-call protocol: the whole turn is ONE assistant response
        # holding the memory-edit tool calls AND the terminating game_action
        # (tool results are discarded by the eval — no role=tool messages).
        flat.append({"role": "system", "content": build_system_prompt(rules_prompt, state)})
        flat.append({"role": "user", "content": toolcall_user_prompt_from_reformatted(reformatted)})
        fragments: list[str] = []
        if writes_on:
            for area, op, slot, content in (_safe(policy.turn_writes, reformatted, action, state, turn_idx, default=[]) or []):
                fragments.append(memory_write_fragment(state, area, op, slot, content))
        flat.append(assistant_turn_message(action, fragments))

        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action, "episode_id": episode_id},
                timeout=request_timeout,
            )
            step_res.raise_for_status()
            step_block = step_res.json()["result"]
            raw_obs = step_block.get("observation", "")
            done = step_block.get("done", False)
            if done:
                sr = step_block.get("reward")
                if isinstance(sr, (int, float)):
                    final_reward = float(sr)
        except Exception as exc:
            print(f"[env] Step failed (game {game_id}): {exc}")
            return None

        if done:
            # Use the TERMINAL observation for the reflect "Final state:" (the
            # eval reflects on format_state(terminal_state)).
            last_reformatted = (_safe(obs_transform, raw_obs, default="") or last_reformatted)
            break
    else:
        print(f"[env] max_turn={max_turn} reached (game {game_id})")

    # Post-game reflect: consolidate a durable long-term opponent read.
    # Eval reflection is also ONE model call (memory tools only, no game_action),
    # so the reflect example is one assistant message of memory tool calls.
    if writes_on:
        outcome = "win" if final_reward > 0.5 else ("loss" if final_reward < 0.5 else "draw")
        rwrites = _safe(policy.reflect_writes, outcome, state, default=[]) or []
        if rwrites:
            flat.append({"role": "system", "content": build_system_prompt(rules_prompt, state, reflection=True)})
            flat.append({"role": "user", "content": reflection_user_prompt(outcome, _state_body(last_reformatted))})
            fragments = [
                memory_write_fragment(state, area, op, slot, content)
                for area, op, slot, content in rwrites
            ]
            flat.append(assistant_turn_message(None, fragments))

    return flat, final_reward
