"""Local pyspiel teacher-vs-teacher self-play SFT data generation (DEFAULT).

WHY: the env-server
(mcts-api) loop POSTs ONE seat's actions over HTTP to a server that auto-plays a
server-side MCTS opponent — the dominant cost on the long-game envs (gin/othello,
well under 1 game/s) and a single-seat, expert-vs-MCTS state distribution. Local
self-play loads the OpenSpiel game IN-PROCESS and has our per-game TEACHER play
BOTH seats, which: removes all HTTP + server-MCTS, harvests EVERY decision node of
both seats (~2-4x rows/game, strong-vs-strong = the peer-vs-peer distribution the
eval actually visits), and builds prompts from the SAME local pyspiel engine the
PvP eval scores on (validator/evaluation/pvp/game_runner.py).

This directly attacks our documented worst problems: data-starvation (<0.2 epoch,
othello/gin 91/9 imbalance) and train/eval distribution drift. It reuses our
EXISTING experts (get_expert_action) as the both-seats teacher and our EXISTING
tool-calling row format (pvp_tool_calling), so the rows stay byte-aligned to eval.

STATUS — LIVE / DEFAULT. self-play is the default SFT generator for all 6 PvP
game envs, wired in sft_env_configs._build_registry (each env ->
make_selfplay_generator). generate_trajectories consumes the MULTI-VIEW
(rows, score) output of BOTH seats and keeps every row (no wins-only filter).
Control: self-play is ON whenever `pyspiel` is importable; set SELFPLAY_DISABLE=1
(or ship without open_spiel in dockerfiles/standalone-text-trainer.dockerfile) to
fall back to the env-server + MCTS path. The per-game formatters below are
byte-aligned to what our experts parse AND what the PvP eval renders — goofspiel
routes through the eval reformatter (reformat_to_pvp), the rest are
ported/passthrough of observation_string. pyspiel is imported lazily so importing
this module (for inspection) never requires it. A GPU smoke-test (per-game
byte-verify of a self-play row's user prompt vs a real eval prompt) remains the
recommended final check before a large run.
"""

import importlib
import os
import random
import re
import zlib


# Population / league self-play: when enabled (default ON), the two seats are
# driven by DIFFERENT teacher personas (seat A = get_expert_action, seat B =
# get_expert_action_b) so the distilled SFT data spans a broader, less-exploitable
# state distribution. A game whose env module exposes no get_expert_action_b falls
# back to the single-teacher behaviour (both seats = persona A), so this is a no-op
# for games without a second persona. Disable entirely with SELFPLAY_DUAL_TEACHER=0.
# Both seats are kept unfiltered (no wins-only): each persona's deviation from the
# first teacher is bounded/near-optimal by construction, so the losing seat's rows
# are still strong demonstrations (and outcome-filtering stochastic games would
# only inject survivorship bias).
# Default DUAL-teacher (persona A + B) — the league-diversity config that WON the
# prior goofspiel/LD/LP tasks. Kept as the global default so this commit does NOT
# change those winners. SELFPLAY_DUAL_TEACHER=0 forces single globally (for tests).
_DUAL_TEACHER = os.environ.get("SELFPLAY_DUAL_TEACHER", "1").strip().lower() not in (
    "0", "false", "no", "off",
)
# ...BUT clobber+gin FORCE single-teacher regardless: the Qwen3-4B game_action
# emission needs low-entropy labels (v12a trained DUAL by a forwarding bug and
# dropped game_action 11/11; v12c single-teacher = clean med-6). Scoped to these
# two envs so it cannot touch the dual-teacher winners of other games (this repo
# may be trained for several tasks). SELFPLAY_DUAL_TEACHER=1 does NOT override the
# scope (the 4B forfeits without it); set PVP_FORCE_DUAL_CLOBBER_GIN=1 to opt out.
_SINGLE_TEACHER_ENVS = frozenset({"clobber", "gin_rummy"})
_FORCE_DUAL_CG = os.environ.get("PVP_FORCE_DUAL_CLOBBER_GIN", "0").strip().lower() in (
    "1", "true", "yes", "on",
)

# Tier-1 DAgger decouple (SELFPLAY_CONSISTENT_LABELS, default ON). The two personas
# still DRIVE both seats (state + opponent-style diversity preserved), but every SFT
# game_action LABEL is produced by ONE deterministic CANONICAL labeler that is a pure
# function of the observation string — so identical states map to identical labels
# across seats, seeds, and RNG. This removes the dual-teacher (+ persona-sampling)
# (state->action) CONTRADICTION that made the SFT target multi-valued and collapsed
# the model to "None"/memory; it matches the boss winner's single-consistent-teacher
# label property while keeping BOTH experts active as drivers. The env still advances
# by the diverse PLAYED action; only the supervised label is canonicalised. Set
# SELFPLAY_CONSISTENT_LABELS=0 to A/B against the old played-action labelling.
_CONSISTENT_LABELS = os.environ.get("SELFPLAY_CONSISTENT_LABELS", "1").strip().lower() not in (
    "0", "false", "no", "off",
)

_EXPERT_MODULE = {
    "liars_dice": "our_envs.liar_dice_trajectories",
    "leduc_poker": "our_envs.leduc_poker_trajectories",
    "gin_rummy": "our_envs.gin_rummy_trajectories",
    "othello": "our_envs.othello_trajectories",
    "clobber": "our_envs.clobber_trajectories",
    "goofspiel": "our_envs.goofspiel_trajectories",
}


# task-id ranges + config-id divisor mirror the validator (game_eval.config_id_for_seed)
_TASK_ID_RANGE = {
    "goofspiel": (0, 99_999_999),
    "liars_dice": (100_000_000, 199_999_999),
    "leduc_poker": (200_000_000, 299_999_999),
    "gin_rummy": (300_000_000, 399_999_999),
    "othello": (400_000_000, 499_999_999),
    "clobber": (700_000_000, 799_999_999),
}
_PVP_CONFIG_ID_DIVISOR = 100_000_000


def _config_id_for_seed(seed: int, env_name: str) -> int:
    lo, hi = _TASK_ID_RANGE[env_name]
    return random.Random(seed).randint(lo, hi) % _PVP_CONFIG_ID_DIVISOR


# --- eval-faithful per-env state formatting (ported from core/pvp/agents.py) ---
def _card_name(card_id: int) -> str:
    ranks = ["J", "Q", "K", "A"]
    suits = ["♠", "♥"]
    rank_idx, suit_idx = card_id // 2, card_id % 2
    return f"{ranks[rank_idx]}{suits[suit_idx]}" if rank_idx < len(ranks) else f"Card_{card_id}"


def _re_extract(info_str: str, pattern: str) -> str:
    m = re.search(pattern, info_str)
    return m.group(1) if m else ""


def _parse_betting(seq: str) -> str:
    amap = {0: "Fold", 1: "Call", 2: "Raise"}
    nums = [int(x) for x in seq.split() if x.isdigit()]
    return ", ".join(amap.get(a, f"Action{a}") for a in nums) if nums else "(none)"


def _format_liars_dice(state, player_id: int) -> str:
    info_str = state.information_state_string(player_id)
    if not info_str:
        return str(state)
    parts = info_str.split()
    dice = [int(d) for d in parts[0] if d.isdigit()]
    bid_parts = [p for p in parts[1:] if "-" in p]
    num_dice = len(dice)
    lines = [
        f"Your dice: {dice} (showing: {', '.join(map(str, dice))})",
        f"Dice per player: {num_dice}",
        f"Total dice in game: {num_dice * state.num_players()}",
        f"Players: {state.num_players()}",
        f"Current player: Player {state.current_player()}",
    ]
    if bid_parts:
        q, f = bid_parts[-1].split("-")
        lines.append(f'\nCurrent bid: "{q}-{f}" (at least {q} dice showing {f} across all players)')
        lines.append("You can: (1) Make a higher bid, or (2) Call 'Liar'")
    else:
        lines.append("No bid yet - you must make the first bid")
    return "\n".join(lines)


def _format_leduc(state, player_id: int) -> str:
    info_str = state.information_state_string(player_id)
    private_card = _re_extract(info_str, r"\[Private: (-?\d+)\]")
    round_num = _re_extract(info_str, r"\[Round (\d+)\]")
    pot = _re_extract(info_str, r"\[Pot: (\d+)\]")
    money = _re_extract(info_str, r"\[Money: ([\d ]+)\]")
    public_card = _re_extract(info_str, r"\[Public: (-?\d+)\]")
    r1 = _re_extract(info_str, r"\[Round1: ([^\]]*)\]")
    r2 = _re_extract(info_str, r"\[Round2: ([^\]]*)\]")
    lines: list = []
    if private_card and private_card != "-10000":
        lines.append(f"Your card: {_card_name(int(private_card))}")
    else:
        lines.append("Your card: (not dealt yet)")
    if public_card and public_card != "-10000":
        lines.append(f"Public card: {_card_name(int(public_card))}")
        if private_card and private_card != "-10000" and int(private_card) // 2 == int(public_card) // 2:
            lines.append("Hand: PAIR")
    lines.append(f"Round: {round_num}/2")
    lines.append(f"Pot: {pot} chips")
    if money:
        chips = money.split()
        if len(chips) >= 2:
            lines.append(f"Your chips: {chips[player_id]}")
            lines.append(f"Opponent chips: {chips[1 - player_id]}")
    if r1:
        lines.append(f"Round 1 actions: {_parse_betting(r1)}")
    if r2:
        lines.append(f"Round 2 actions: {_parse_betting(r2)}")
    return "\n".join(lines)


def _format_gin_rummy(state, player_id: int) -> str:
    return state.observation_string(player_id)


def _format_othello(state, player_id: int) -> str:
    # Othello: P0 = x (Black, moves first), P1 = o (White) — matches OthelloAgent.
    colour = "x (Black)" if player_id == 0 else "o (White)"
    return f"You play {colour}.\n{state.observation_string(player_id)}"


def _format_clobber(state, player_id: int) -> str:
    # Clobber: P0 = o (White, moves first), P1 = x (Black) — matches ClobberAgent.
    colour = "o (White)" if player_id == 0 else "x (Black)"
    return f"You play {colour}.\n{state.observation_string(player_id)}"


def _format_goofspiel(state, player_id: int) -> str:
    return state.observation_string(player_id)


_FORMATTERS = {
    "liars_dice": _format_liars_dice,
    "leduc_poker": _format_leduc,
    "gin_rummy": _format_gin_rummy,
    "othello": _format_othello,
    "clobber": _format_clobber,
    "goofspiel": _format_goofspiel,
}

_OTHELLO_OPENING_PLIES = (2, 6)
_CLOBBER_OPENING_PLIES = (2, 6)
_GOOFSPIEL_NUM_CARDS = (5, 8, 10, 13)
_CLOBBER_BOARD_SIZES = ((4, 5), (5, 5), (5, 6))


def _load_game(env_name: str, config_id: int):
    import pyspiel  # lazy: only needed when self-play is actually enabled
    if env_name == "liars_dice":
        return pyspiel.load_game("liars_dice", {"players": 2, "numdice": 5})
    if env_name == "leduc_poker":
        return pyspiel.load_game("leduc_poker", {"players": 2})
    if env_name == "gin_rummy":
        hand_var = (config_id // 3) % 3
        knock_var = config_id % 3
        return pyspiel.load_game("gin_rummy", {"hand_size": 7 + hand_var, "knock_card": 10 - knock_var})
    if env_name == "othello":
        return pyspiel.load_game("othello")
    if env_name == "clobber":
        rows, cols = _CLOBBER_BOARD_SIZES[config_id % len(_CLOBBER_BOARD_SIZES)]
        return pyspiel.load_game("clobber", {"rows": rows, "columns": cols})
    if env_name == "goofspiel":
        # imp_info=True matches the validator (GoofspielAgent: opponent hand hidden).
        num_cards = _GOOFSPIEL_NUM_CARDS[config_id % len(_GOOFSPIEL_NUM_CARDS)]
        return pyspiel.load_game("goofspiel", {
            "num_cards": num_cards, "players": 2, "imp_info": True,
            "points_order": "random", "returns_type": "win_loss",
        })
    raise ValueError(f"No pyspiel self-play spec for env {env_name!r}")


def _setup_initial_state(env_name: str, state, seed: int) -> None:
    """Seeded random opening plies for the perfect-info board games (othello,
    clobber) — mirrors agents.py setup_initial_state, for state-distribution
    variety. Chance-node games deal via the chance-node branch in the play loop."""
    plies = {"othello": _OTHELLO_OPENING_PLIES, "clobber": _CLOBBER_OPENING_PLIES}.get(env_name)
    if plies is None:
        return
    rng = random.Random(seed)
    for _ in range(rng.randint(*plies)):
        if state.is_terminal():
            break
        legal = state.legal_actions()
        if not legal:
            break
        state.apply_action(rng.choice(legal))


def _expert(env_name: str, persona: str = "a"):
    """Lazy-import OUR per-env teacher and wrap it as
    choose_action(observation, player_id, legal_ids, rng) -> int.

    persona='a' -> get_expert_action (the shipped first teacher). persona='b' ->
    get_expert_action_b if the module defines it, ELSE get_expert_action (so a
    game without a second persona transparently reuses the first teacher for both
    seats). Never raises at selection time: any failure falls back to a legal id."""
    mod_name = _EXPERT_MODULE.get(env_name)
    if mod_name is None:
        raise ValueError(f"No expert for env {env_name!r}")
    mod = importlib.import_module(mod_name)
    fn = None
    if persona == "b":
        fn = getattr(mod, "get_expert_action_b", None)
    if fn is None:
        fn = mod.get_expert_action

    def choose_action(observation: str, player_id: int, legal_ids: list, rng=None) -> int:
        try:
            return int(fn([{"role": "user", "content": observation}]))
        except Exception:
            return (rng or random).choice(legal_ids)
    return choose_action


def _seat_teachers(env_name: str, seed: int):
    """Return {0: teacher, 1: teacher} for the two seats. With dual-teacher off (or
    no persona B for this env) both seats use persona A (legacy behaviour). With it
    on, persona A and B are assigned to the seats, SWAPPED on odd seeds so each
    persona plays both player positions across game_ids (no persona x first-move
    bias)."""
    t_a = _expert(env_name, "a")
    # clobber+gin force single-teacher (4B emission); every other env honours the
    # global _DUAL_TEACHER default, so the prior dual-teacher winners are untouched.
    force_single = env_name in _SINGLE_TEACHER_ENVS and not _FORCE_DUAL_CG
    if force_single or not _DUAL_TEACHER:
        return {0: t_a, 1: t_a}
    mod = importlib.import_module(_EXPERT_MODULE[env_name])
    if getattr(mod, "get_expert_action_b", None) is None:
        return {0: t_a, 1: t_a}            # no second persona -> single teacher
    t_b = _expert(env_name, "b")
    return {0: t_a, 1: t_b} if seed % 2 == 0 else {0: t_b, 1: t_a}


def _canonical_labeler(env_name: str):
    """Tier-1 DAgger decouple: return label(observation, legal_ids) -> int that
    produces the ONE deterministic SFT game_action LABEL for a state, as a PURE
    FUNCTION OF THE OBSERVATION (independent of which driver persona stepped the
    node). Binds the env's ``get_label_action`` (the deterministic argmax of the
    expert's OWN distribution + obs-hashed tie-break — added to the SAMPLING experts
    goofspiel/liars_dice/leduc_poker) if defined, ELSE ``get_expert_action`` (already
    a deterministic argmax for the perfect-info experts othello/clobber/gin). Never
    raises: any failure or out-of-legal value clamps to min(legal_ids) — NOT to the
    driver's played action, which would re-inject the persona-B/seat-swap
    contradiction on exactly the hard rows."""
    mod = importlib.import_module(_EXPERT_MODULE[env_name])
    fn = getattr(mod, "get_label_action", None) or mod.get_expert_action

    def _clamp(legal_ids: list) -> int:
        # leduc: min(legal) is FOLD (id 0) in facing-bet states — a certain loss
        # under the PvP eval's sign scoring (fold = guaranteed 0.0 while call
        # always retains showdown equity). Prefer call/check (1), then raise (2).
        # OpenSpiel leduc_poker action ids: 0=Fold, 1=Call/Check, 2=Raise.
        if env_name == "leduc_poker":
            for pref in (1, 2):
                if pref in legal_ids:
                    return pref
        return min(legal_ids)

    def label(observation: str, legal_ids: list) -> int:
        try:
            v = int(fn([{"role": "user", "content": observation}]))
        except Exception:
            return _clamp(legal_ids)
        if v not in legal_ids:
            return _clamp(legal_ids)
        # leduc id 0 is ALWAYS Fold; the label policy is never-fold (WL-CFR), so a
        # 0 here is a fallback artifact (e.g. the module's unparseable-obs "0"
        # default passing the legality check because fold happens to be legal).
        if env_name == "leduc_poker" and v == 0:
            return _clamp(legal_ids)
        return v
    return label


def _teacher_obs(state_desc: str, player_id: int, legal_actions: list) -> str:
    """The 'Current State: ... Legal Actions: <id> -> <label>' envelope our
    get_expert_action / reformatters parse (and that toolcall_user_prompt rebuilds
    the per-turn model prompt from).

    The DOUBLE newline before "You are Player" is REQUIRED: pvp_tool_calling
    ._OLD_STATE_RE matches ``Current State:\\n(.*?)\\n\\nYou are Player (\\d+)``. With
    a single newline the regex fails, so toolcall_user_prompt_from_reformatted
    falls back to treating the WHOLE envelope as the state (legal-actions/"Your
    choice" leak into the rendered state) AND the player id defaults to 0 for BOTH
    seats — an off-distribution, seat-mislabelled prompt that the model never sees
    at eval (the 100/100-forfeit class of bug). Keep the ``\\n\\n``."""
    action_lines = "\n".join(f"{aid} -> {label}" for aid, label in legal_actions)
    return (
        f"Current State:\n{state_desc}\n\n"
        f"You are Player {player_id}.\n"
        f"Legal Actions:\n{action_lines}\n"
        f"Your choice (ID only):"
    )


_SYSTEM_PROMPT_CACHE: dict = {}


def _system_prompt(env_name: str) -> str:
    sp = _SYSTEM_PROMPT_CACHE.get(env_name)
    if sp is None:
        from our_envs.pvp_prompts.pvp_prompt_loader import get_game_rules_prompt
        from our_envs.pvp_tool_calling import build_system_prompt, MemoryState
        sp = build_system_prompt(get_game_rules_prompt(env_name), MemoryState())
        _SYSTEM_PROMPT_CACHE[env_name] = sp
    return sp


_LABEL_COT_ENABLED = (os.environ.get("PVP_LABEL_COT") or "0").strip().lower() in ("1", "true", "yes", "on")
_LABEL_COMMENT_FN_CACHE: dict = {}


def _label_comment(env_name: str, reformatted_obs: str) -> str:
    """Optional per-env one-sentence rule-grounded CoT for the assistant label
    (the winner's proven learnability device: naming the visible feature that
    fired makes the policy imitable at higher fidelity). An env opts in by
    defining ``get_label_comment(messages) -> str`` (deterministic, never-raises,
    "" on failure) — currently othello only. GATED OFF by default
    (PVP_LABEL_COT=1): it changes the othello SFT row format (comment text before
    the tool call), which must be probe-validated end-to-end before default-on."""
    if not _LABEL_COT_ENABLED:
        return ""
    # The comment narrates the CANONICAL label's choice (get_label_comment mirrors
    # get_label_action). With SELFPLAY_CONSISTENT_LABELS=0 the row's action is the
    # persona's SAMPLED move, which can differ — a rationale naming a different
    # move than the tool call it precedes would actively teach announce-one-move-
    # play-another. Comment only when the canonical labeler stamps the rows.
    if not _CONSISTENT_LABELS:
        return ""
    fn = _LABEL_COMMENT_FN_CACHE.get(env_name, False)
    if fn is False:
        try:
            fn = getattr(importlib.import_module(_EXPERT_MODULE[env_name]), "get_label_comment", None)
        except Exception:
            fn = None
        _LABEL_COMMENT_FN_CACHE[env_name] = fn
    if fn is None:
        return ""
    try:
        return (fn([{"role": "user", "content": reformatted_obs}]) or "").strip()
    except Exception:
        return ""


# --- per-seat memory wiring (winner recipe, wired into SELF-PLAY) -----------
# run_toolcall_episode (env-server path) already threads MemoryPolicy writes into
# the SFT rows, but the DEFAULT gin/clobber generator is THIS self-play path,
# which used to bypass memory entirely (game_action-only rows, always-empty
# memory block). Two consequences at eval on the Qwen3-4B: (a) the model never
# learned the [memory-write -> game_action] continuation, so its own prior
# memory writes dead-end without a move; (b) it never saw a POPULATED memory
# block, so once its own verbose notes landed in the block every later turn was
# off-distribution -> runaway generation past PVP_TURN_MAX_TOKENS -> forfeit.
# This wiring fixes both IN the real generator:
#   * PVP_MEMORY_WRITES=1 -> per-seat MemoryState + the env's *MemoryPolicy
#     rare short notes, memory-FIRST then game_action (assistant_turn_message).
#   * PVP_MEM_AUG (default ON) -> deterministically inject a model-style
#     VERBOSE state-dump note into the memory state early in ~half the games,
#     so the system prompts of later turns render junk-populated memory blocks
#     while the LABEL stays [short note?] + game_action. Eval-aligned: the eval
#     block really does fill with the model's own dumps.
#   * PVP_MEM_AUG (default ON, but GATED OFF under no-memory via train_omit_memory,
#     so it has no effect on the clobber+gin no-memory path) -> inject a verbose
#     state-dump note into the memory state early in ~half the games.
_MEM_AUG = (os.environ.get("PVP_MEM_AUG") or "1").strip().lower() not in ("0", "false", "no", "off")

# Per-env memory WRITES, aligned to the winner's proven recipe (verified in the
# winner repo: gin_rummy/goof_spiel/liar_dice generators emit a working_memory_rewrite
# tool_call; othello/leduc_poker/clobber generators emit game_action ONLY, and the
# clobber generator's own comment states why: "Clobber's full state is always fully
# visible — no opponent-hand tracking or memory-worthy hidden information, unlike e.g.
# liar_dice/gin_rummy"). So memory is WRITTEN only where the label needs HIDDEN info:
# the imperfect-information envs. The memory SURFACE (tools + an empty memory block)
# is still OFFERED for every env via the cached empty-memory system prompt (train ==
# eval), but a perfect-information env never writes -> its target stays game_action-
# only and cannot dead-end at eval (write memory -> stop). Override with
# PVP_MEMORY_WRITE_ENVS="a,b,c".
_MEMORY_WRITE_ENVS: frozenset = frozenset(
    (os.environ.get("PVP_MEMORY_WRITE_ENVS") or "gin_rummy,goofspiel,liars_dice")
    .replace(" ", "").split(",")
)
_RULES_CACHE: dict = {}
_MEM_POLICY_CACHE: dict = {}
_AUG_BODY_RE = re.compile(r"Current State:\n(.*?)\n\n(?:You are Player|Legal Actions:)", re.DOTALL)


def _rules_prompt(env_name: str) -> str:
    rp = _RULES_CACHE.get(env_name)
    if rp is None:
        from our_envs.pvp_prompts.pvp_prompt_loader import get_game_rules_prompt
        rp = get_game_rules_prompt(env_name)
        _RULES_CACHE[env_name] = rp
    return rp


def _memory_policy_for(env_name: str):
    """Fresh per-game instance of the env's *MemoryPolicy (never-raises; base
    no-op policy when the env module doesn't define one)."""
    cls = _MEM_POLICY_CACHE.get(env_name, False)
    if cls is False:
        cls = None
        try:
            mod = importlib.import_module(_EXPERT_MODULE[env_name])
            from our_envs.pvp_episode import MemoryPolicy
            for name in dir(mod):
                if name.endswith("MemoryPolicy"):
                    cand = getattr(mod, name)
                    # cand is not MemoryPolicy: every env module imports the BASE
                    # class by name, and dir() is sorted — "MemoryPolicy" sorts
                    # BEFORE "OthelloMemoryPolicy", so without this exclusion
                    # othello silently got the no-op base policy (audit P1).
                    if isinstance(cand, type) and issubclass(cand, MemoryPolicy) and cand is not MemoryPolicy:
                        cls = cand
                        break
        except Exception:
            cls = None
        _MEM_POLICY_CACHE[env_name] = cls
    if cls is None:
        from our_envs.pvp_episode import MemoryPolicy
        return MemoryPolicy()
    try:
        return cls()
    except Exception:
        from our_envs.pvp_episode import MemoryPolicy
        return MemoryPolicy()


def _mem_wiring_on(env_name: str) -> bool:
    """Memory WRITES + populated-memory aug are active only for the imperfect-
    information envs in _MEMORY_WRITE_ENVS (winner recipe). For a perfect-information
    env (othello/leduc/clobber) this returns False, so no memory is ever written and
    the target stays game_action-only — but the memory SURFACE is still offered via
    the cached empty-memory system prompt (_system_prompt), so the training prompt
    still matches the eval's (memory tools + an empty block). bare_id mode has no
    memory surface at all -> wiring stays off (audit P2)."""
    if env_name not in _MEMORY_WRITE_ENVS:
        return False
    from our_envs.pvp_tool_calling import memory_writes_enabled, train_omit_memory, tool_calling_enabled
    if not tool_calling_enabled() or train_omit_memory():
        return False
    return memory_writes_enabled() or _MEM_AUG


def _mem_aug_inject(mem_state, env_name: str, seed: int, seat: int, turn: int, reformatted_obs: str) -> None:
    """Deterministically (crc32 — stable across workers) inject a model-style
    verbose state-dump into a working slot at ONE early turn in ~half the games
    (plus occasional long-term junk), simulating the model's own eval-time
    writes. Mutates mem_state only; labels are untouched. Never raises."""
    try:
        h = zlib.crc32(f"{seed}:{seat}:memaug".encode())
        if h % 2:
            return
        if turn != ((h >> 3) % 4) + 1:
            return
        m = _AUG_BODY_RE.search(reformatted_obs or "")
        body = " ".join((m.group(1) if m else reformatted_obs or "").split())
        if not body:
            return
        from our_envs.pvp_tool_calling import WORKING, LONG_TERM
        mem_state.rewrite(WORKING, 1 + ((h >> 5) % 2), f"Note on this game: {body}")
        if (h >> 7) % 3 == 0:
            mem_state.rewrite(LONG_TERM, 1, f"Opponent notes: {body}")
    except Exception:
        pass


def _seat_fragments(policy, mem_state, reformatted_obs: str, label_id: int, turn: int) -> list:
    """The seat's memory-write tool-call fragments for this turn (mutates
    mem_state so the NEXT turn's system shows them). Never raises."""
    from our_envs.pvp_tool_calling import memory_writes_enabled, memory_write_fragment
    if not memory_writes_enabled():
        return []
    frags: list = []
    try:
        for area, op, slot, content in (policy.turn_writes(reformatted_obs, str(label_id), mem_state, turn) or []):
            frags.append(memory_write_fragment(mem_state, area, op, slot, content))
    except Exception:
        return frags
    return frags


def _attach_row_meta(row: dict) -> dict:
    """R3 Q-gap side-channel: pop the labeler's ``(qgap, forced)`` metadata (set
    via ``envs.row_meta.set_meta`` just before the action was chosen) and attach
    it to the emitted row as NEUTRAL side columns ``_qgap`` / ``_forced``. Called
    exactly once per emitted row (``_build_row`` is the single per-row choke).

    This does NOT touch the row's ``messages`` / label — it only adds two side
    columns. Harmless when the labeler set nothing: ``pop_meta`` returns
    ``{None, False}`` (neutral -> weight 1.0 downstream, no rows dropped). The
    merge/train weighting only READS these columns when ``QGAP_WEIGHTING`` is on.

    IMPORTANT: the columns are ADDED ONLY when QGAP_WEIGHTING is on. When off
    (default) the row is byte-identical to before this change — no extra columns.
    This matters because the SINGLE-ENV pipeline SKIPS merge_trajectories (which is
    where the columns would otherwise be stripped), so an un-stripped ``_qgap`` /
    ``_forced`` would reach ``mix_miner_datasets`` -> ``concatenate_datasets`` and
    raise on the schema mismatch vs a mounted miner dataset. ``pop_meta`` is ALWAYS
    called (even when off) to clear the thread-local slot so a labeler's qgap can
    never bleed into the NEXT row. Never raises."""
    try:
        from our_envs.row_meta import pop_meta
        meta = pop_meta()  # always pop -> clear the slot (prevent stale bleed)
    except Exception:
        meta = {"qgap": None, "forced": False}
    if os.environ.get("QGAP_WEIGHTING", "").strip().lower() not in ("", "0", "false", "no", "off"):
        row["_qgap"] = meta.get("qgap")
        row["_forced"] = bool(meta.get("forced"))
    return row


def _build_row(
    env_name: str,
    reformatted_obs: str,
    action_id: int,
    system_content: "str | None" = None,
    fragments: "list | None" = None,
) -> dict:
    """One per-turn SFT example in OUR eval tool-calling format: system (rules +
    memory render AS OF this turn + guidance) / user (rebuilt per-turn prompt) /
    assistant ([memory writes ->] game_action tool call). Mirrors
    run_toolcall_episode's per-turn window. Without memory wiring the system is
    the cached empty-memory prompt and the assistant is game_action-only.
    bare_id mode keeps the old assistant_move_message surface (audit P1: the
    turn-message builder is tool_call-only; fragments are impossible in bare_id)."""
    from our_envs.pvp_tool_calling import (
        toolcall_user_prompt_from_reformatted, assistant_turn_message,
        assistant_move_message, move_encoding,
    )
    if move_encoding() != "tool_call":
        assistant = assistant_move_message(str(action_id))
    else:
        assistant = assistant_turn_message(str(action_id), list(fragments or []))
    # CoT comment only on the tool_call surface (audit P2: "comment\n<id>" would
    # be off-format for the legacy bare_id validator). Fix #3: the assistant is now
    # STRUCTURED (content:None + tool_calls), so the comment becomes the `content`
    # thought and apply_chat_template renders it BEFORE the tool_call — byte-equivalent
    # to the old "comment\n<tool_call>" on Qwen, and correct on Llama for free.
    comment = _label_comment(env_name, reformatted_obs) if move_encoding() == "tool_call" else ""
    if comment and "tool_calls" in assistant:
        assistant = dict(assistant)
        assistant["content"] = comment
    row = {"messages": [
        {"role": "system", "content": system_content if system_content is not None else _system_prompt(env_name)},
        {"role": "user", "content": toolcall_user_prompt_from_reformatted(reformatted_obs)},
        assistant,
    ]}
    return _attach_row_meta(row)


def _emit_row(env_name: str, seed: int, seat_mem, p: int, obs: str, label_id: int, turn: int, seat_turn: int) -> dict:
    """Build the seat's per-turn row, threading the seat's memory context when
    wiring is on (observe -> aug-inject -> system snapshot BEFORE this turn's
    writes -> fragments mutate state for the NEXT turn — mirroring
    run_toolcall_episode's ordering and the eval's per-turn re-render).
    seat_turn = the seat's OWN decision index (audit P2: the global ply counter
    only ever shows ONE parity to each seat in alternating games, so parity-
    mismatched seats would NEVER receive the aug injection); `turn` keeps the
    global round semantics for turn_writes (goofspiel notes use rounds)."""
    if seat_mem is None:
        return _build_row(env_name, obs, label_id)
    st, pol = seat_mem[p]
    try:
        pol.observe(obs)
    except Exception:
        pass
    if _MEM_AUG:
        _mem_aug_inject(st, env_name, seed, p, seat_turn, obs)
    from our_envs.pvp_tool_calling import build_system_prompt
    sys_c = build_system_prompt(_rules_prompt(env_name), st)
    frags = _seat_fragments(pol, st, obs, label_id, turn)
    return _build_row(env_name, obs, label_id, system_content=sys_c, fragments=frags)


def _score(returns: list, player_id: int) -> float:
    me, opp = returns[player_id], returns[1 - player_id]
    return 1.0 if me > opp else (0.0 if me < opp else 0.5)


# Games whose EVAL reformatter REBUILDS the state body (not a passthrough of
# observation_string), so the self-play row must apply the SAME reformatter to
# stay byte-aligned with the eval prompt AND with what the per-game expert parses.
# goofspiel: reformat_goofspiel_to_pvp rebuilds a 7-line body + a "[P{pid}]Bid: {c}"
# legal block from observation_string — raw observation_string would drift (the
# PR #1217 forfeit-100/100 failure mode). For these, the reformatter returns the
# COMPLETE PvP envelope, so _teacher_obs is NOT applied on top. The other games
# are passthrough/ported (observation_string is the eval body), so they go through
# _teacher_obs as before.
_REFORMAT_ENVELOPE_ENVS = {"goofspiel"}


def _build_obs(env_name: str, state, player_id: int, legal_ids: list) -> str:
    """Build the per-turn observation envelope that BOTH the expert and the SFT
    row consume. Guarded: a bad observation_string / formatter for ONE env must
    never propagate out of play_selfplay_game and zero the whole dataset (a single
    raise would fail every worker for that env -> empty set -> 'No valid
    conversations'). On failure, fall back to a degraded str(state) envelope so the
    game still progresses and yields rows."""
    try:
        if env_name in _REFORMAT_ENVELOPE_ENVS:
            from our_envs.pvp_prompts.pvp_state_format import reformat_to_pvp
            return reformat_to_pvp(state.observation_string(player_id), env_name, player_id=player_id)
        labels = [(a, state.action_to_string(player_id, a)) for a in legal_ids]
        return _teacher_obs(_FORMATTERS[env_name](state, player_id), player_id, labels)
    except Exception:
        try:
            return _teacher_obs(str(state), player_id, [(a, str(a)) for a in legal_ids])
        except Exception:
            return _teacher_obs("", player_id, [])


def play_selfplay_game(env_name: str, seed: int, max_turn: int, rng=None):
    """Play one teacher-vs-teacher game via local pyspiel. Returns a list of
    (rows, score) VIEWS — one per seat — where rows is that seat's per-turn SFT
    examples and score is its terminal outcome in [0,1] (win=1/draw=0.5/loss=0).
    An unfinished game (max_turn hit) scores 0.5 so filters neither keep nor drop."""
    if rng is None:
        rng = random.Random(seed)
    seat_teacher = _seat_teachers(env_name, seed)   # {0: persona, 1: persona} DRIVERS
    labeler = _canonical_labeler(env_name) if _CONSISTENT_LABELS else None
    seat_mem = None
    if _mem_wiring_on(env_name):
        from our_envs.pvp_tool_calling import MemoryState
        seat_mem = {p: (MemoryState(), _memory_policy_for(env_name)) for p in (0, 1)}
    game = _load_game(env_name, _config_id_for_seed(seed, env_name))
    state = game.new_initial_state()
    _setup_initial_state(env_name, state, seed)

    per_seat: dict = {0: [], 1: []}
    turns = 0
    while not state.is_terminal() and turns < max_turn:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            state.apply_action(rng.choices(list(actions), weights=list(probs), k=1)[0])
            continue
        if state.is_simultaneous_node():
            joint: list = []
            for p in range(game.num_players()):
                p_legal = state.legal_actions(p)
                if not p_legal:
                    joint.append(0)
                    continue
                obs = _build_obs(env_name, state, p, p_legal)
                p_action = seat_teacher[p](obs, p, p_legal, rng=rng)   # DRIVER (diverse) steps the game
                if p_action not in p_legal:
                    p_action = rng.choice(p_legal)
                # CANONICAL label = pure function of obs (consistent across seats/seeds);
                # the env still advances by the diverse p_action below.
                label_id = labeler(obs, p_legal) if labeler else p_action
                per_seat[p].append(_emit_row(env_name, seed, seat_mem, p, obs, label_id, turns, len(per_seat[p]) + 1))
                joint.append(p_action)
            state.apply_actions(joint)
            turns += 1
            continue
        cp = state.current_player()
        if cp < 0:
            break
        legal_ids = state.legal_actions(cp)
        if not legal_ids:
            break
        obs = _build_obs(env_name, state, cp, legal_ids)
        action_id = seat_teacher[cp](obs, cp, legal_ids, rng=rng)   # DRIVER (diverse) steps the game
        if action_id not in legal_ids:
            action_id = rng.choice(legal_ids)
        # CANONICAL label = pure function of obs (consistent across seats/seeds);
        # the env still advances by the diverse action_id below.
        label_id = labeler(obs, legal_ids) if labeler else action_id
        per_seat[cp].append(_emit_row(env_name, seed, seat_mem, cp, obs, label_id, turns, len(per_seat[cp]) + 1))
        state.apply_action(action_id)
        turns += 1

    if state.is_terminal():
        returns = state.returns()
        scores = {0: _score(returns, 0), 1: _score(returns, 1)}
    else:
        scores = {0: 0.5, 1: 0.5}
    return [(per_seat[p], scores[p]) for p in (0, 1) if per_seat[p]]


def play_teacher_vs_mcts_game(env_name: str, seed: int, max_turn: int, rng=None):
    """Play one game with OUR teacher on ONE (randomised) seat against an in-process
    OpenSpiel MCTS opponent on the other seat — the winner's data-gen recipe run
    fully in-process (no env-server HTTP). Only the TEACHER seat's decisions become
    SFT rows (its moves are the strong labels); the MCTS seat just advances the game
    toward the eval-realistic boards the teacher is graded on. Returns a single-view
    ``[(rows, score)]`` (rows = teacher-seat per-turn SFT examples, score = teacher's
    terminal outcome in [0,1]); ``[]`` if the teacher never moved. Never raises out
    (per-node failures degrade to a random legal move).

    Reuses the SAME obs / row / memory machinery as play_selfplay_game so the rows
    stay byte-aligned to the PvP eval prompt; the ONLY change is that the opponent
    seat is an MCTS bot (eval-aligned strength) instead of a second teacher, which
    removes the self-play distribution shift while keeping gen in-process-fast."""
    from our_envs.mcts_opponent import make_mcts_bot, mcts_step_or_none, mcts_sims_for
    if rng is None:
        rng = random.Random(seed)
    teacher_seat = rng.choice((0, 1))
    # Consistent-labels: the canonical labeler is BOTH the played move and the SFT
    # label for the teacher seat (identical), so the game advances by the teacher's
    # strongest move and the row records that same move. Without it, persona-a drives.
    labeler = _canonical_labeler(env_name) if _CONSISTENT_LABELS else None
    teacher = _expert(env_name, "a")
    seat_mem = None
    if _mem_wiring_on(env_name):
        from our_envs.pvp_tool_calling import MemoryState
        seat_mem = {teacher_seat: (MemoryState(), _memory_policy_for(env_name))}
    game = _load_game(env_name, _config_id_for_seed(seed, env_name))
    opponent_bot = make_mcts_bot(game, mcts_sims_for(env_name, rng), seed)
    state = game.new_initial_state()
    _setup_initial_state(env_name, state, seed)

    rows: list = []
    turns = 0
    while not state.is_terminal() and turns < max_turn:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            state.apply_action(rng.choices(list(actions), weights=list(probs), k=1)[0])
            continue
        if state.is_simultaneous_node():
            # MCTSBot is sequential-only; the simultaneous env (goofspiel) is NOT
            # routed here (it stays on self-play). Defensive: degrade every seat to a
            # legal move so an unexpected simultaneous node can never hang gen.
            joint = []
            for p in range(game.num_players()):
                p_legal = state.legal_actions(p)
                joint.append(rng.choice(p_legal) if p_legal else 0)
            state.apply_actions(joint)
            turns += 1
            continue
        cp = state.current_player()
        if cp < 0:
            break
        legal_ids = state.legal_actions(cp)
        if not legal_ids:
            break
        if cp == teacher_seat:
            obs = _build_obs(env_name, state, cp, legal_ids)
            label_id = labeler(obs, legal_ids) if labeler else teacher(obs, cp, legal_ids, rng=rng)
            rows.append(_emit_row(env_name, seed, seat_mem, cp, obs, label_id, turns, len(rows) + 1))
            played = label_id if label_id in legal_ids else rng.choice(legal_ids)
            state.apply_action(played)
        else:
            action = mcts_step_or_none(opponent_bot, state)
            if action is None or action not in legal_ids:
                action = rng.choice(legal_ids)
            state.apply_action(action)
        turns += 1

    if state.is_terminal():
        score = _score(state.returns(), teacher_seat)
    else:
        score = 0.5
    return [(rows, score)] if rows else []


def make_teacher_vs_mcts_generator(env_name: str):
    """Return a teacher-vs-in-process-MCTS generator (winner recipe, no env server).
    Drop-in for the env-server generators: same ``(game_id, endpoint, max_turn)``
    signature (endpoint ignored) and the same multi-view ``[(rows, score), ...]``
    return shape generate_trajectories already consumes (here a single seat view)."""
    def _gen(game_id: int, env_endpoint=None, max_turn: int = 70):
        return play_teacher_vs_mcts_game(env_name, seed=game_id, max_turn=max_turn, rng=random.Random(game_id))
    return _gen


# --- goofspiel: teacher-vs-HEURISTIC-opponent (simultaneous-move, NO MCTS) ------
# Goofspiel is simultaneous-move: both players bid at once, so an OpenSpiel MCTSBot
# (sequential-only) cannot drive the opponent seat correctly — the winner's own code
# records the mcts opponent as "flatly wrong for this game" (it flattens the
# simultaneous move). So, like the winner, the opponent seat plays a HEURISTIC bid
# chosen per game_id from a small set, for variety in what the teacher must beat:
#   - "random":  uniform-random legal bid (matches what the old mcts request produced)
#   - "mirror":  our persona-B prize-MATCHING softmax (disciplined, exploitable)
#   - "sampled": our persona-A sampled teacher (the hardest opponent)
# Only the teacher seat is recorded; its label is the canonical deterministic mode.
_GOOFSPIEL_OPPONENTS = ("random", "mirror", "sampled")


def _goofspiel_opponent(kind: str, rng):
    """Return choose(observation, player_id, legal_ids, rng) -> int for the opponent
    seat. Never raises at selection time (falls back to a legal id)."""
    if kind == "mirror":
        return _expert("goofspiel", "b")     # prize-matching softmax
    if kind == "sampled":
        return _expert("goofspiel", "a")     # our sampled teacher

    def _rand(observation: str, player_id: int, legal_ids: list, rng=None) -> int:
        return (rng or random).choice(legal_ids)
    return _rand


def play_teacher_vs_heuristic_goofspiel(seed: int, max_turn: int, rng=None):
    """Goofspiel (simultaneous-move) teacher-vs-heuristic-opponent gen (winner
    recipe). The teacher seat plays + records the canonical label; the opponent seat
    plays a per-game heuristic (random/mirror/sampled). Only the teacher seat is
    emitted. Returns ``[(rows, score)]`` (``[]`` if the teacher never moved). Reuses
    the same obs/row/memory machinery as the self-play path (byte-aligned to eval).
    Never raises out (per-node failures degrade to a legal move)."""
    env_name = "goofspiel"
    if rng is None:
        rng = random.Random(seed)
    teacher_seat = rng.choice((0, 1))
    labeler = _canonical_labeler(env_name) if _CONSISTENT_LABELS else None
    teacher = _expert(env_name, "a")
    opp_fn = _goofspiel_opponent(rng.choice(_GOOFSPIEL_OPPONENTS), rng)
    seat_mem = None
    if _mem_wiring_on(env_name):
        from our_envs.pvp_tool_calling import MemoryState
        seat_mem = {teacher_seat: (MemoryState(), _memory_policy_for(env_name))}
    game = _load_game(env_name, _config_id_for_seed(seed, env_name))
    state = game.new_initial_state()

    rows: list = []
    turns = 0
    while not state.is_terminal() and turns < max_turn:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            state.apply_action(rng.choices(list(actions), weights=list(probs), k=1)[0])
            continue
        if state.is_simultaneous_node():
            joint = [0] * game.num_players()
            for p in range(game.num_players()):
                p_legal = state.legal_actions(p)
                if not p_legal:
                    continue
                obs = _build_obs(env_name, state, p, p_legal)
                if p == teacher_seat:
                    label_id = labeler(obs, p_legal) if labeler else teacher(obs, p, p_legal, rng=rng)
                    rows.append(_emit_row(env_name, seed, seat_mem, p, obs, label_id, turns, len(rows) + 1))
                    joint[p] = label_id if label_id in p_legal else rng.choice(p_legal)
                else:
                    a = opp_fn(obs, p, p_legal, rng=rng)
                    joint[p] = a if a in p_legal else rng.choice(p_legal)
            state.apply_actions(joint)
            turns += 1
            continue
        # Goofspiel has no sequential decision nodes; defensive fall-through.
        cp = state.current_player()
        if cp < 0:
            break
        legal_ids = state.legal_actions(cp)
        if not legal_ids:
            break
        state.apply_action(rng.choice(legal_ids))
        turns += 1

    if state.is_terminal():
        score = _score(state.returns(), teacher_seat)
    else:
        score = 0.5
    return [(rows, score)] if rows else []


def make_teacher_vs_heuristic_goofspiel_generator():
    """Return the goofspiel teacher-vs-heuristic generator. Same signature/return
    shape as the other in-process generators (endpoint ignored, single seat view)."""
    def _gen(game_id: int, env_endpoint=None, max_turn: int = 30):
        return play_teacher_vs_heuristic_goofspiel(seed=game_id, max_turn=max_turn, rng=random.Random(game_id))
    return _gen


def make_selfplay_generator(env_name: str):
    """Return a self-play generator. ``env_endpoint`` is ignored (no env server).
    Drop-in for the env-server generators EXCEPT the return shape is the
    multi-view [(rows, score), ...] (both seats), so generate_trajectories must be
    adapted to consume per-turn rows directly (and drop wins_only) before wiring
    this into _SFT_REGISTRY. Not auto-wired yet — enable per the module docstring."""
    def _gen(game_id: int, env_endpoint=None, max_turn: int = 70):
        return play_selfplay_game(env_name, seed=game_id, max_turn=max_turn, rng=random.Random(game_id))
    return _gen
