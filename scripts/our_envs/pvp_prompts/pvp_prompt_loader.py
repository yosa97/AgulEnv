"""Validator PvP prompt template loader.

Reads pvp_game_prompts.yml (byte-copied from the validator at
G.O.D/core/config/pvp_game_prompts.yml) and reproduces the same
`generate_system_prompt` / `generate_user_prompt` shape the validator's
BaseGameAgent uses at PvP eval time:

  system: template.format(game_name=..., rules=<game_rules>)
  user:   "Current State:\n{state}\n\n"
          "You are Player {pid}.\n"
          "Legal Actions:\n0 -> roll\n1 -> bid_a1\n...\n\n"
          "Your choice (ID only):"

The trainer-side trajectory generators feed these strings into the
expert pipeline so the SFT data lives in the same prompt space the
model will be evaluated in. Any divergence (extra sections, reordered
blocks, missing/extra whitespace) shows up as a measurable accuracy
drop at PvP eval time — LD games are 1-2 turns so a single wrong
output loses the whole match.
"""

import os
import re
from functools import lru_cache

import yaml

from our_envs.pvp_tool_calling import move_encoding, build_system_prompt, MemoryState


_PROMPT_YAML = os.path.join(os.path.dirname(__file__), "pvp_game_prompts.yml")
# Byte-copy of PR #1201's core/config/pvp_game_prompts.yml (the post-tool-calling
# validator: no "# Output Format" block; canonical othello_rules). Loaded for
# tool_call mode so template folding + rules are EXACTLY what the eval's
# BaseGameAgent.generate_system_prompt() produces.
_PROMPT_YAML_TOOLCALL = os.path.join(
    os.path.dirname(__file__), "pvp_game_prompts_toolcall.yml"
)


@lru_cache(maxsize=1)
def load_prompts() -> dict:
    """Load and cache the PvP prompts YAML once per process."""
    with open(_PROMPT_YAML, "r") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_prompts_toolcall() -> dict:
    with open(_PROMPT_YAML_TOOLCALL, "r") as f:
        return yaml.safe_load(f)


_RULES_KEY = {
    "liars_dice":  "liars_dice_rules",
    "leduc_poker": "leduc_poker_rules",
    "gin_rummy":   "gin_rummy_rules",
    "othello":     "othello_rules",
    "goofspiel":   "goofspiel_rules",
    "clobber":     "clobber_rules",
}


def get_game_rules_prompt(game_name: str) -> str:
    """The static per-game agent prompt — byte-exact equivalent of the eval's
    BaseGameAgent.generate_system_prompt() in the post-#1201 validator.

    Built from the canonical #1201 YAML (template + rules), so YAML folding is
    identical to eval: the parsed template is
    'You are playing {game_name}.\\n# Game Rules {rules}\\n' (SINGLE newline,
    rules inline after '# Game Rules '). This is what build_system_prompt()
    prepends to the per-turn memory block + tool guidance.
    """
    prompts = load_prompts_toolcall()
    rules_key = _RULES_KEY.get(game_name)
    if rules_key is None:
        raise ValueError(
            f"Unknown game_name {game_name!r}; supported: {sorted(_RULES_KEY)}"
        )
    return prompts["system_prompt_template"].format(
        game_name=game_name, rules=prompts[rules_key],
    )


def get_system_prompt(game_name: str) -> str:
    """Return the system prompt for ``game_name`` in the active move encoding.

    bare_id mode (current/old validator): byte-identical to what the PvP eval
    container constructs via BaseGameAgent.generate_system_prompt() — the
    YAML template with the "respond with ONLY the action ID" output format.

    tool_call mode (post-#1168 validator): agent rules + an EMPTY memory block
    + _TOOL_GUIDANCE (mirrors core/pvp LLMBot._system_prompt at game start). In
    memory-writes mode the generators rebuild this per turn via
    build_system_prompt(get_game_rules_prompt(game), current_memory_state); this
    static form is the game-start / no-writes fallback.
    """
    if move_encoding() == "tool_call":
        return build_system_prompt(get_game_rules_prompt(game_name), MemoryState())
    prompts = load_prompts()
    rules_key = _RULES_KEY.get(game_name)
    if rules_key is None:
        raise ValueError(
            f"Unknown game_name {game_name!r}; supported: {sorted(_RULES_KEY)}"
        )
    rules = prompts[rules_key]
    return prompts["system_prompt_template"].format(
        game_name=game_name, rules=rules,
    )


def get_user_prompt(
    state_desc: str,
    player_id: int,
    legal_actions_with_labels: list,
) -> str:
    """Build the user prompt string the validator passes to the model.

    legal_actions_with_labels: iterable of (action_id, label_text) pairs.
    """
    actions_desc = [f"{aid} -> {label}" for aid, label in legal_actions_with_labels]
    return (
        f"Current State:\n{state_desc}\n\n"
        f"You are Player {player_id}.\n"
        "Legal Actions:\n" + "\n".join(actions_desc) + "\n\n"
        "Your choice (ID only):"
    )


_LEGAL_ACTIONS_HEADER_RE = re.compile(r"^\s*Legal\s+Actions\s*:\s*$", re.MULTILINE)
_LEGAL_ACTION_LINE_RE = re.compile(r"^\s*(\d+)\s*->\s*(.+?)\s*$", re.MULTILINE)


def reformat_to_pvp_user_prompt(raw_observation: str, player_id: int = 0) -> str:
    """Wrap a raw env-server observation into the validator's user prompt.

    The env-server returns observation text that already contains a
    "Legal Actions:" block but lacks the validator's framing:

      Current State:
      <state desc>

      You are Player {pid}.
      Legal Actions:
      0 -> ...

      Your choice (ID only):

    This helper splits the raw observation at the "Legal Actions:" header,
    re-parses the action labels, and reassembles the text inside that
    template so the prompt the model sees during SFT distillation is
    byte-aligned with what the PvP eval container constructs at inference
    time via BaseGameAgent.generate_user_prompt().

    Falls back to the raw observation unchanged when the header or any
    action lines cannot be parsed, so the pipeline degrades gracefully
    on unexpected env-server outputs rather than silently drops actions.
    """
    if not raw_observation:
        return raw_observation
    match = _LEGAL_ACTIONS_HEADER_RE.search(raw_observation)
    if not match:
        return raw_observation
    state_desc = raw_observation[: match.start()].rstrip()
    legal_block = raw_observation[match.end():]
    legal_pairs: list = []
    for m in _LEGAL_ACTION_LINE_RE.finditer(legal_block):
        legal_pairs.append((int(m.group(1)), m.group(2).strip()))
    if not legal_pairs:
        return raw_observation
    return get_user_prompt(state_desc, player_id, legal_pairs)
