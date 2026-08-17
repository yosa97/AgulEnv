"""Shared tool-calling + memory surface for PvP env trajectory distillation.

Background (G.O.D PR #1168 "Env memory system" + feat/othello)
--------------------------------------------------------------
The PvP eval (liars_dice / leduc_poker / gin_rummy / othello) was rewritten so
the model drives each turn by emitting a native OpenAI tool call to
``game_action(action_id)`` (plus optional memory-edit tools). VERIFIED upstream
facts that this module reproduces (core/pvp/bot.py, memory.py, tools.py):

  * The transcript is REBUILT FRESH EVERY TURN (bot.py:6-7, :180-184). On turn N
    the model sees ONLY [system(rules + CURRENT memory render + tool guidance),
    user(state + legal actions)] plus the turn-local inner tool-call loop. The
    ONLY cross-turn state is SlotMemory, rendered into the system prompt.
  * Memory: WORKING (4 slots × 128 tok, reset each game) + LONG_TERM (8 slots ×
    128 tok, persists across a matchup = opponent model). Edited via tools
    {working,long_term}_{rewrite,append}({slot:int, content:str}); rewrite
    head-truncates, append is FIFO tail-truncate. render() byte-format:
    "{AREA} (your notes):\n  [1] <text or (empty)>\n  [2] ...".
  * Tool result message: {"role":"tool","content":"ok:.../error:...","tool_call_id":id}.
  * After each game: reflect() — system(rules+memory+_REFLECTION_GUIDANCE) +
    user("The game is over. Result for you: WIN. ...") -> memory tools only.

⚠️ RELEASE TIMING — this is UNRELEASED upstream (absent from our submodule pin).
This branch (athena) defaults to the NEW protocol; submitting to a pre-#1168
validator FORFEITS. Set ``_DEFAULT_MOVE_ENCODING="bare_id"`` to target the old
validator. Memory-writes default ON here (winner recipe; PVP_MEMORY_WRITES=0 to
disable): each env's *MemoryPolicy writes a SHORT note on only ~25% of turns
(crc32 rarity gate), memory-FIRST then game_action in ONE response, so the model
learns game_action ALWAYS follows a memory write. The OLD failure (dense EVERY-turn
writes -> "emit memory first" habit that dropped game_action, 100% forfeit,
2026-06-22) is why the writes are RARE (25%), not why they are off. train_omit_memory()
defaults OFF (offer the surface) so those memory-write targets are valid + the train
prefix byte-matches the eval (bot.py:220 always renders the memory tools + block +
guidance). This mirrors the reference recipe (audited: offers memory globally, writes
in gin/goofspiel/liar_dice, game_action-only in clobber/othello/leduc, no no-memory
toggle). A first memory-write run (517 steps) forfeited via dead-end (wrote memory, no
game_action) — but that was UNDERTRAINING (our 45-min gen cap left ~35 min training);
sft_env_config now uses a size-scaled gen budget (~14 min gen -> ~75 min train for a
7B/1.5h) to train ~2x longer and learn the continuation. PVP_TRAIN_NO_MEMORY=1 /
PVP_MEMORY_WRITES=0 restore game_action-only for A/B.

Encoding: a tool call is emitted as Qwen ``<tool_call>{json}</tool_call>`` TEXT
inside the assistant ``content`` (and tool results as {role:tool, content} str),
so every message stays a plain {role, content} dict — HF-Arrow-safe and
maskable inside the chat-template assistant span without per-row struct drift.
The tokenizer passes ``tools=PVP_TOOLS`` to apply_chat_template so the tools
block matches eval. Pure + never-raises throughout.
"""

import hashlib
import json
import os
import re


# --- encoding + feature switches -------------------------------------------
_DEFAULT_MOVE_ENCODING = "tool_call"   # "tool_call" (#1168) | "bare_id" (old validator)
_DEFAULT_MEMORY_WRITES = True          # ON (winner recipe): each env's *MemoryPolicy writes a
                                       # SHORT factual note on only ~25% of turns (crc32 rarity
                                       # gate), memory-FIRST then game_action in ONE response, so
                                       # the model learns game_action ALWAYS follows a memory write.
                                       # First attempt (517 steps) dead-ended (wrote memory, no
                                       # game_action) — but that was UNDERTRAINING: our fixed 45-min
                                       # gen cap left only ~35 min training. With the size-scaled
                                       # gen budget (sft_env_config: ~14 min gen for a 7B/1.5h ->
                                       # ~75 min train) the model should now learn the continuation.
                                       # PVP_MEMORY_WRITES=0 restores game_action-only targets.


def move_encoding() -> str:
    enc = os.environ.get("PVP_MOVE_ENCODING", _DEFAULT_MOVE_ENCODING).strip().lower()
    return enc if enc in ("tool_call", "bare_id") else _DEFAULT_MOVE_ENCODING


def tool_calling_enabled() -> bool:
    return move_encoding() == "tool_call"


def memory_writes_enabled() -> bool:
    """Per-turn-fresh restructure + memory writes. Only meaningful in tool_call
    mode (bare_id is the old growing-conversation protocol)."""
    if not tool_calling_enabled():
        return False
    v = os.environ.get("PVP_MEMORY_WRITES")
    if v is None:
        return _DEFAULT_MEMORY_WRITES
    return v.strip().lower() not in ("0", "false", "no", "off", "")


# --- upstream constants (mirror core/pvp/constants.py + bot.py guidance) ----
GAME_ACTION_TOOL_NAME = "game_action"
PVP_WORKING_MEM_SLOTS = 4
PVP_WORKING_SLOT_TOKENS = 128
PVP_LONGTERM_MEM_SLOTS = 8
PVP_LONGTERM_SLOT_TOKENS = 128

WORKING = "working_memory"
LONG_TERM = "long_term_memory"

# verbatim from PR #1201 core/pvp/bot.py:44-53 (single-call turn protocol)
_TOOL_GUIDANCE = (
    "You get ONE response this turn. In it, optionally edit your memory notes, and "
    "then call game_action with a legal action id to commit your move. If you do not "
    "call game_action, you forfeit the turn — so always include it."
)
# No-memory training variant (PVP_TRAIN_NO_MEMORY): drops the memory mention so the
# guidance matches a game_action-only tools list.
_TOOL_GUIDANCE_NO_MEM = (
    "You get ONE response this turn: call game_action with a legal action id to "
    "commit your move. If you do not call game_action, you forfeit the turn — so "
    "always include it."
)
_REFLECTION_GUIDANCE = (
    "The game is over. Use the memory tools to update your long-term notes on "
    "this opponent for future games — keep durable, generalisable reads (their "
    "tendencies, your counter-strategy) and drop move-by-move detail. There is "
    "no move to make."
)

_AREA_N = {WORKING: PVP_WORKING_MEM_SLOTS, LONG_TERM: PVP_LONGTERM_MEM_SLOTS}
_AREA_BUDGET = {WORKING: PVP_WORKING_SLOT_TOKENS, LONG_TERM: PVP_LONGTERM_SLOT_TOKENS}


# --- whitespace token counting (mirrors WhitespaceTokenCounter default) -----
# Production injects a real tokenizer per player; at train time we use the
# dependency-free whitespace counter. Notes are short (<<128 words) so
# truncation rarely fires, and the token-count integer only appears in the
# (loss-masked) tool-result string, so any counter mismatch vs eval is cosmetic.
def _wc(text: str) -> int:
    return len(text.split())


def _truncate(text: str, max_tokens: int, keep: str) -> "tuple[str, bool]":
    words = text.split()
    if len(words) <= max_tokens:
        return text, False
    kept = words[:max_tokens] if keep == "head" else words[-max_tokens:]
    return " ".join(kept), True


# --- memory state (mirrors core/pvp/memory.py SlotMemory, never-raises) ------
class MemoryState:
    """Tracks WORKING + LONG_TERM slot memory exactly as the eval would, so the
    rendered memory block + tool-result strings in our SFT data match what the
    model sees at eval. Mutated as the expert writes notes during a game."""

    def __init__(self):
        self._slots = {
            WORKING: {i: "" for i in range(1, PVP_WORKING_MEM_SLOTS + 1)},
            LONG_TERM: {i: "" for i in range(1, PVP_LONGTERM_MEM_SLOTS + 1)},
        }

    def reset_working(self) -> None:
        self._slots[WORKING] = {i: "" for i in range(1, PVP_WORKING_MEM_SLOTS + 1)}

    def _valid(self, area: str, slot) -> bool:
        return (
            area in self._slots
            and isinstance(slot, int)
            and not isinstance(slot, bool)
            and 1 <= slot <= _AREA_N[area]
        )

    def rewrite(self, area: str, slot: int, content: str) -> str:
        if not self._valid(area, slot):
            return f"error: slot {slot} out of range (1-{_AREA_N.get(area, 0)})"
        text, trunc = _truncate(str(content), _AREA_BUDGET[area], "head")
        self._slots[area][slot] = text
        note = " (truncated to budget)" if trunc else ""
        return f"ok: slot {slot} rewritten, {_wc(text)} tokens{note}"

    def append(self, area: str, slot: int, content: str) -> str:
        if not self._valid(area, slot):
            return f"error: slot {slot} out of range (1-{_AREA_N.get(area, 0)})"
        existing = self._slots[area][slot]
        combined = (existing + "\n" + str(content)) if existing else str(content)
        text, trunc = _truncate(combined, _AREA_BUDGET[area], "tail")
        self._slots[area][slot] = text
        note = " (oldest dropped)" if trunc else ""
        return f"ok: slot {slot} appended, {_wc(text)} tokens{note}"

    def render(self) -> str:
        """Byte-exact LLMBot._memory_block(): WORKING then LONG_TERM, each
        "{AREA} (your notes):\\n  [i] <text or (empty)>", areas joined by \\n\\n."""
        blocks = []
        for area in (WORKING, LONG_TERM):
            title = f"{area.upper()} (your notes):"
            lines = "\n".join(
                f"  [{i}] {self._slots[area][i] or '(empty)'}"
                for i in range(1, _AREA_N[area] + 1)
            )
            blocks.append(f"{title}\n{lines}")
        return "\n\n".join(blocks)


def render_memory_block() -> str:
    """Empty memory block (game start) — used by the system prompt when no
    notes have been written yet (and by the memory-aware prompt in the
    no-writes path)."""
    return MemoryState().render()


# --- OpenAI tool schemas (passed to apply_chat_template(tools=) by tokenizer) -
# Byte-matched to PR #1201/#1217 core/pvp/tools.py (build_memory_tools /
# build_game_action_tool) + core/models/pvp_models.py field descriptions.
#
# ⚠️ KEY ORDER IS LEAK-SENSITIVE. The eval builds these via Pydantic
# `model_json_schema()` (+ _params_schema), and the Qwen3 chat template renders
# each tool with `{{ tool | tojson }}` — which PRESERVES dict key order. So the
# rendered <tools> block, and thus the model's prefix, depends on key order. If
# our dict order differs from the eval's pydantic order, the train-time tools
# text differs from eval and the model decodes from an off-distribution prefix
# (root cause of the goofspiel memory-leak-at-greedy, 2026-06-22). Mirror the
# EXACT pydantic v2 order: a `parameters` dict is {properties, required, type};
# each property is {description, type, <minimum/maximum | enum>}.
_AREA_PURPOSE = {
    WORKING: "notes for THIS game, reset each game",
    LONG_TERM: "notes on THIS opponent, persist across games",
}
_OP_PHRASING = {
    "rewrite": ("Overwrite", "replaces the slot's previous content"),
    "append": ("Append to", "oldest text drops if the slot is full"),
}


def _mem_tool_schema(area: str, op: str) -> dict:
    verb, effect = _OP_PHRASING[op]
    return {
        "type": "function",
        "function": {
            "name": f"{area}_{op}",
            "description": (
                f"{verb} a {area} slot (slots 1-{_AREA_N[area]}; "
                f"{_AREA_PURPOSE[area]}); {effect}."
            ),
            "parameters": {
                "properties": {
                    "slot": {
                        "description": "Target slot number.",
                        "type": "integer",
                        "minimum": 1,
                        "maximum": _AREA_N[area],
                    },
                    "content": {"description": "Text content for the slot.", "type": "string"},
                },
                "required": ["slot", "content"],
                "type": "object",
            },
        },
    }


def build_game_action_schema(legal_actions: "list[int] | None" = None) -> dict:
    """game_action schema mirroring tools.py build_game_action_tool: per-turn
    legal-ids hint in the description + advisory enum on action_id."""
    action_id = {
        "description": "A legal action id for the current state.",
        "type": "integer",
    }
    if legal_actions:
        action_id["enum"] = list(legal_actions)
        legal_hint = "Legal action ids: " + ", ".join(str(a) for a in legal_actions) + "."
    else:
        legal_hint = "Legal action ids: see the legal actions list."
    return {
        "type": "function",
        "function": {
            "name": GAME_ACTION_TOOL_NAME,
            "description": f"Commit your move and end your turn. {legal_hint}",
            "parameters": {
                "properties": {"action_id": action_id},
                "required": ["action_id"],
                "type": "object",
            },
        },
    }


# Order mirrors the eval (bot.py:194): memory tools FIRST (working rewrite,
# working append, long_term rewrite, long_term append), game_action LAST.
MEMORY_TOOL_SCHEMAS = [
    _mem_tool_schema(WORKING, "rewrite"),
    _mem_tool_schema(WORKING, "append"),
    _mem_tool_schema(LONG_TERM, "rewrite"),
    _mem_tool_schema(LONG_TERM, "append"),
]


def train_omit_memory() -> bool:
    """Training-only: whether to OMIT the memory tools + memory block from the prompt.

    DEFAULTS OFF (2026-07-13) — i.e. training OFFERS the memory surface (4 memory tools
    + memory block + _TOOL_GUIDANCE), matching the eval prefix (G.O.D bot.py:220) AND
    required for the winner recipe: with the surface offered, ~25% of targets write a
    SHORT note FIRST then game_action (PVP_MEMORY_WRITES ON), teaching game_action ALWAYS
    follows a memory write. This is what the reference recipe does (audited: memory offered
    GLOBALLY, writes in gin/goofspiel/liar_dice; game_action-only targets in
    clobber/othello/leduc; NO PVP_TRAIN_NO_MEMORY flag anywhere).

    Why not memory-OMITTED: a 4-GPU memory-write run (517 steps) forfeited — the model
    wrote memory then DEAD-ENDED (no game_action). BUT that was UNDERTRAINING, not the
    recipe: our fixed 45-min gen cap left only ~35 min training, so the model half-learned
    (writes memory) but never learned the game_action continuation. The winner spends only
    ~12% of the budget on gen (we now match this in sft_env_config with a size-scaled gen
    budget -> ~75 min training for a 7B/1.5h), so it trains ~2x longer and learns the
    continuation. NOTE a SEPARATE deeper factor: our R2 base (clobber-mcts-merged) has a
    memory-writing PRIOR that the winner's RAW Qwen2.5-7B base lacks — even a memory-OFF
    adapter wrote memory+dead-ended on othello; a cleaner base may still be needed.

    Set PVP_TRAIN_NO_MEMORY=1 to OMIT the surface (game_action-only; also gates
    PVP_MEMORY_WRITES off). The `... or "0"` maps BOTH unset AND empty-string to the
    memory-OFFERED default so a launcher forwarding an unset host var as "" can't flip
    it."""
    return (os.environ.get("PVP_TRAIN_NO_MEMORY") or "0").strip().lower() in ("1", "true", "yes", "on")


def pvp_tools_for_turn(legal_actions: "list[int] | None" = None) -> list:
    """The per-turn tools list (memory + dynamic game_action), eval order.
    With PVP_TRAIN_NO_MEMORY set, the memory tools are omitted (game_action only)."""
    game = [build_game_action_schema(legal_actions)]
    return game if train_omit_memory() else (MEMORY_TOOL_SCHEMAS + game)


_LEGAL_ID_RE = re.compile(r"^(\d+)\s*->", re.MULTILINE)


def legal_ids_from_user(user_content: str) -> "list[int]":
    """Parse the legal action ids from a per-turn user prompt
    ("Legal actions:\n<id> -> ..."). Single-sourced here so BOTH the generator (per-row
    tools column) and the tokenizer (fallback reconstruction) key on the same format."""
    block = str(user_content).split("Legal actions:\n", 1)
    return [int(x) for x in _LEGAL_ID_RE.findall(block[1])] if len(block) == 2 else []


def pvp_tools_json(legal_actions: "list[int] | None" = None) -> str:
    """Per-row ``tools`` column value: ``json.dumps`` of this turn's tools list.

    Stored as a JSON STRING (not a list-of-dicts): datasets/Arrow unifies struct schemas
    across a list column's elements, which corrupts heterogeneous tool
    ``parameters.properties`` (e.g. a no-arg memory tool's ``{}`` vs game_action's
    ``{action_id}``). The tokenizer ``json.loads`` it back before apply_chat_template.
    Single-sourcing the eval ``<tools>`` block AT GENERATION means tokenize no longer
    reconstructs it from a mutable env flag -> no gen/tokenize drift (fix #2)."""
    return json.dumps(pvp_tools_for_turn(legal_actions))


def pvp_tools_for_reflection() -> list:
    """Reflection offers ONLY the memory tools (bot.py:176 — no game_action)."""
    return list(MEMORY_TOOL_SCHEMAS)


# Static fallback (tokenizer rows where legal ids can't be recovered).
PVP_TOOLS = pvp_tools_for_turn(None)


# --- message builders ------------------------------------------------------
def _tool_call_content(name: str, arguments: dict) -> str:
    """Qwen <tool_call> TEXT surface for one call. LEGACY — kept for the bare_id path
    and reference; the tool_call path now emits STRUCTURED tool_calls (below) so
    apply_chat_template renders each model family's surface natively at tokenize."""
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


def _structured_tool_call(name: str, arguments: dict) -> dict:
    """One OpenAI-format tool_calls[] entry. `arguments` is a JSON STRING (Arrow-safe
    across heterogeneous envs: datasets/Arrow unifies struct schemas across a column, so
    a dict `arguments` with different keys per env corrupts; the tokenizer json.loads it
    back to a dict before apply_chat_template so the template's tojson renders it once)."""
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _game_action_tool_call(action_id: "str | int") -> "dict | None":
    """Structured game_action tool_calls[] entry; None if the id is not int-castable
    (caller should DROP the row, not emit an untooled/forfeiting target)."""
    try:
        aid_int = int(str(action_id).strip())
    except (TypeError, ValueError):
        return None
    return _structured_tool_call(GAME_ACTION_TOOL_NAME, {"action_id": aid_int})


def assistant_move_message(action_id: "str | int") -> dict:
    """Assistant message that commits a move.

    tool_call mode -> {"role":"assistant","content":None,"tool_calls":[game_action]}
        STRUCTURED (fix #3): apply_chat_template renders the Qwen `<tool_call>` OR the
        Llama surface natively at tokenize, so BOTH pool families parse it at eval (the
        old hand-baked Qwen text forfeited on the 2 Llama models).
    bare_id  mode  -> {"role":"assistant","content": '<action_id>'}
    The value POSTed to the env-server /step is ALWAYS the bare id string (unchanged)."""
    aid = str(action_id).strip()
    if move_encoding() == "tool_call":
        tc = _game_action_tool_call(aid)
        if tc is None:
            # Non-int id: never SILENTLY emit bare text (an untooled row forfeits at
            # eval). Dormant (experts return numerics) but log; caller should drop.
            print(f"[pvp_tool_calling] WARNING assistant_move_message: non-int action_id "
                  f"{aid!r} — cannot emit game_action tool_call; row should be DROPPED", flush=True)
            return {"role": "assistant", "content": aid}
        return {"role": "assistant", "content": None, "tool_calls": [tc]}
    return {"role": "assistant", "content": aid}


def memory_write_fragment(state: MemoryState, area: str, op: str, slot: int, content: str) -> dict:
    """Apply a memory write to `state` and return the STRUCTURED memory tool_calls[]
    entry teaching it (fix #3 — was a <tool_call> TEXT fragment). `op` in
    {"rewrite","append"}. Never raises.

    #1201 single-call protocol: the turn is ONE model response containing all memory-edit
    tool calls plus the game_action call; tool RESULTS are discarded by the eval, so we
    only mutate the state (for the NEXT turn's memory render) and emit the call.
    assistant_turn_message collects these fragments + the game_action into ONE message's
    tool_calls list (memory-first)."""
    if op == "rewrite":
        state.rewrite(area, slot, content)
    else:
        state.append(area, slot, content)
    return _structured_tool_call(f"{area}_{op}", {"slot": slot, "content": content})


# The clobber+gin task winner emits memory-write tool_calls FIRST, game_action
# LAST — matching the eval tool-schema order (memory tools first) AND Qwen3's
# natural memory-first prior. This teaches "game_action ALWAYS follows a memory
# write", so at eval the strong 4B memory prior writes memory then RELIABLY
# continues to game_action instead of dead-ending at memory-then-EOS. Our old
# game_action-first ordering fights the prior. Default ON here (empty = on) since
# the memory-inoculation recipe requires it; PVP_MEMORY_FIRST=0 restores the old
# game_action-first order for A/B. Only matters when a turn actually writes memory
# (rare, signal-gated) — most turns are game_action-only either way.
_MEMORY_FIRST = (os.environ.get("PVP_MEMORY_FIRST") or "1").strip().lower() not in ("0", "false", "no", "off")


def assistant_turn_message(action_id: "str | int | None", fragments: "list[dict]") -> dict:
    """ONE assistant message for the whole turn, STRUCTURED tool_calls (fix #3). With
    _MEMORY_FIRST (default): the memory-write tool_calls (`fragments`) FIRST, then the
    game_action tool_call (winner order — game_action is the learned continuation after a
    memory write). Legacy (_MEMORY_FIRST off): game_action first. The eval (_run_turn)
    applies every memory edit and takes the FIRST game_action, so order is scoring-
    irrelevant but decisive for whether the 4B reliably reaches game_action after its
    prior's memory write. action_id=None (reflection) emits only the memory fragments.
    `fragments` are STRUCTURED memory tool_calls[] dicts (from memory_write_fragment)."""
    game_tcs: "list[dict]" = []
    if action_id is not None:
        tc = _game_action_tool_call(action_id)
        if tc is None:
            # Non-int id: skip game_action rather than emit bare text (untooled = forfeit).
            # Dormant (experts return numerics) but log; caller should drop the row.
            print(f"[pvp_tool_calling] WARNING assistant_turn_message: non-int action_id "
                  f"{str(action_id).strip()!r} — no game_action tool_call; row should be DROPPED",
                  flush=True)
        else:
            game_tcs.append(tc)
    tool_calls = (list(fragments) + game_tcs) if _MEMORY_FIRST else (game_tcs + list(fragments))
    if not tool_calls:
        return {"role": "assistant", "content": ""}
    return {"role": "assistant", "content": None, "tool_calls": tool_calls}


# --- per-turn user prompt (new tool_call eval format) ----------------------
# core/pvp/bot.py:254-261 _user_prompt = "Current state:\n{state_desc}\n\n
#   You are Player {id}.\nLegal actions:\n{action_lines}"  (lowercase headers,
# NO "Your choice (ID only):" trailer — unlike the old bare-id env-server wrap).
_OLD_STATE_RE = re.compile(r"Current State:\n(.*?)\n\nYou are Player\s+(\d+)", re.DOTALL)
# Lookahead trailer-strip (tolerates a single OR double newline before the
# env-server "Your choice ..." trailer), mirroring pvp_state_format._LEGAL_BLOCK_RE.
# The old `(?:\n\nYour choice|\Z)` only stripped a \n\n-separated trailer, so a
# single-\n GR/othello trailer would leak "Your choice (action ID only):" into
# the action block (off-distribution vs the validator, which emits no trailer).
_OLD_LEGAL_RE = re.compile(r"Legal Actions:\n(.*?)(?=\n\s*Your choice|\Z)", re.DOTALL)


def toolcall_user_prompt_from_reformatted(reformatted: str, player_id: int = 0) -> str:
    """Convert an old-format reformatted observation ("Current State:.../Legal
    Actions:.../Your choice (ID only):") into the #1168 per-turn user prompt.
    Falls back to the input unchanged if it can't be parsed (never raises)."""
    sm = _OLD_STATE_RE.search(reformatted)
    state_desc = sm.group(1) if sm else reformatted
    pid = int(sm.group(2)) if sm else player_id
    # KEEP Leduc betting-history lines in the TRAINING prompt: G.O.D #1240
    # (2026-06-24) RE-ADDED the within-game betting sequence to the eval prompt
    # ("Round 1/2 actions: Fold, Call, Raise"), reversing the #1217 stripping. So
    # the model SHOULD see betting (it lifts the leduc partial-observability ceiling
    # our leduc_cfr keys on), and our reformatter now normalises those lines to the
    # validator's exact token format (pvp_state_format._normalize_lp_betting). The
    # old strip is therefore removed. No-op for non-Leduc games (no such line).
    lm = _OLD_LEGAL_RE.search(reformatted)
    action_lines = lm.group(1).rstrip() if lm else ""
    # core/pvp builds action lines flush-left ("{id} -> {str}"); the env-server
    # indents them ("  {id} -> ...") and _extract_state leaves them indented, so
    # dedent to match the eval user prompt byte-for-byte.
    action_lines = re.sub(r"^[ \t]+", "", action_lines, flags=re.MULTILINE)
    return (
        f"Current state:\n{state_desc}\n\n"
        f"You are Player {pid}.\n"
        f"Legal actions:\n{action_lines}"
    )


def build_system_prompt(rules_and_agent_prompt: str, state: "MemoryState | None" = None,
                        reflection: bool = False) -> str:
    """Per-turn system prompt: agent rules + CURRENT memory render + guidance
    (verbatim _TOOL_GUIDANCE, or _REFLECTION_GUIDANCE for the post-game turn).
    Mirrors core/pvp/bot.py _system_prompt / _reflection_system_prompt.
    With PVP_TRAIN_NO_MEMORY (train only), drops the memory block + memory-mention
    guidance so the whole prompt is memory-free (matches the omitted memory tools)."""
    if train_omit_memory() and not reflection:
        return "\n\n".join([rules_and_agent_prompt, _TOOL_GUIDANCE_NO_MEM])
    mem = (state or MemoryState()).render()
    guidance = _REFLECTION_GUIDANCE if reflection else _TOOL_GUIDANCE
    return "\n\n".join([rules_and_agent_prompt, mem, guidance])


def _semantic_tiebreak_enabled() -> bool:
    """R1 optional tie-break (PVP_SEMANTIC_TIEBREAK, default OFF). When OFF,
    ``canonical_argmax`` keeps its obs-hash tie-break byte-for-byte."""
    v = os.environ.get("PVP_SEMANTIC_TIEBREAK")
    if v is None:
        return False
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def canonical_argmax(legal_ids: "list[int]", weights: "list[float]", obs: str,
                     secondary: "list[float] | None" = None) -> int:
    """Deterministic CANONICAL pick for Tier-1 DAgger label decoupling (the SFT
    game_action label is a pure function of the observation, not a sampled/dual-
    teacher action). Returns the max-weight legal id; ties are broken by a STABLE
    hash of the observation string (hashlib SHA1, NOT Python's salted hash()/an RNG,
    so it is identical across worker processes regardless of PYTHONHASHSEED) — giving
    near-uniform (e.g. Nash) infosets a fixed-but-unpredictable choice with no
    readable smallest-id tell. Never raises (degenerate input -> min/0).

    Optional R1 semantic tie-break: pass ``secondary`` (a per-legal-action value
    list, index-aligned with ``legal_ids``) and set ``PVP_SEMANTIC_TIEBREAK`` to
    break a top-weight tie by the LARGEST secondary value (deterministic) before
    falling back to the obs-hash among any still-tied ids. With the flag OFF or
    ``secondary`` unset the obs-hash path is taken UNCHANGED, so existing callers
    (three positional args) keep their exact current behaviour."""
    if not legal_ids:
        return 0
    if not weights or len(weights) != len(legal_ids):
        return min(legal_ids)
    top = max(weights)
    tied = [a for a, w in zip(legal_ids, weights) if w >= top - 1e-12]
    if len(tied) == 1:
        return tied[0]
    # R1 opt-in: narrow the top-weight tie by a secondary per-action value first.
    if (secondary is not None and len(secondary) == len(legal_ids)
            and _semantic_tiebreak_enabled()):
        sec = {a: s for a, s in zip(legal_ids, secondary)}
        best_sec = max(sec[a] for a in tied)
        narrowed = [a for a in tied if sec[a] >= best_sec - 1e-12]
        if len(narrowed) == 1:
            return narrowed[0]
        tied = narrowed  # still tied -> deterministic obs-hash among the rest
    h = int(hashlib.sha1(obs.encode("utf-8", "ignore")).hexdigest(), 16)
    return tied[h % len(tied)]


def reflection_user_prompt(outcome: str, final_state_desc: str) -> str:
    """core/pvp/bot.py:259-266 _reflection_user_prompt. `outcome` in
    {win,loss,draw} (any case) -> upper-cased."""
    return (
        f"The game is over. Result for you: {str(outcome).upper()}.\n\n"
        f"Final state:\n{final_state_desc}\n\n"
        "Update your long-term notes on this opponent for future games."
    )


# ---------------------------------------------------------------------------
# VERIFY when #1168 is live (untestable until then):
#   1. Qwen2.5 apply_chat_template(tools=PVP_TOOLS) renders the tools block and
#      an assistant message whose content == our '<tool_call>\n{json}\n</tool_call>'
#      to the SAME tokens the eval emits for a structured tool_call. Re-check the
#      exact newlines/spacing Qwen uses, and CodeLlama/Hermes in the lineup.
#   2. role=tool message renders identically (Qwen wraps as <tool_response>); our
#      {role:tool,content} (no tool_call_id) matches eval's rendering.
#   3. The per-turn user prompt ("Current state:" lowercase, no trailer) matches
#      core/pvp _user_prompt for each game's format_state.
#   4. Tool-result token-count integers differ (whitespace vs real tokenizer) —
#      cosmetic (masked), but confirm it doesn't perturb the model.
# ---------------------------------------------------------------------------
