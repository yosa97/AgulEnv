"""Goofspiel (GOPS, Game of Pure Strategy) expert SFT trajectory generator.

Goofspiel is the 5th PvP game (G.O.D PR #1217: EvalType.PVP,
EnvironmentName.GOOFSPIEL, task-id 0-99,999,999). Trainer-side analogue of
othello_trajectories.py: roll out a strong-but-cheap expert vs the env-server
opponent and distill the winning play into per-turn SFT trajectories.

Why a MIXED (sampled) expert, NOT a deterministic argmax — verified literature
(2026-06-20 deep-research, see reference_goofspiel_pvp_research):
  * Goofspiel has NO winning deterministic strategy: any pure/argmax bidder is
    crushed 78-13 by the "bid one rank higher" counter (Rhoads-Bartholdi 2012,
    Dror-Kendall 2013). Nash play REQUIRES mixing at every non-final round.
  * The naive "matching" trap (bid the card equal to the prize value) is the
    best response ONLY vs a uniform-random opponent; in symmetric PvP it is a
    pure strategy and exploitable. We must NOT distil it (it is what the legacy
    goof_spiel_env.py hard-codes).
  * Our objective is WIN/LOSS, so margin-optimal solved tables are also the
    wrong equilibrium (provably exploitable under win/loss). Regret-Matching is
    the evidence-backed win/loss policy; exact solving is infeasible for N>=8.
This v1.1 expert is therefore a RANK-RELATIVE mixed bidder that SAMPLES from a
broad Gaussian in RANK space (each card's rank within our remaining hand vs the
prize's strength), with the mode deliberately SHIFTED OFF the matching bid toward
the winning "overbid-one" — the cheap, offline, rules-clean "mixed" lever that
escapes BOTH the argmax sweep AND the bid==prize matching trap. (v1 was a Gaussian
centred on the prize VALUE, i.e. the 78-13 matching trap with additive noise: its
mode == prize for every prize, so a greedy student collapsed straight back onto
the matching line. v1.1 fixes that: rank-relative + mode-shifted + broad mixing.)
A baked offline CFR/Regret-Matching table for N=5 and an MCCFR-with-abstraction
policy for N=8/10/13 are the planned v2 upgrades (the principled Nash policy).

OpenSpiel / env encoding (load-bearing):
  * Each turn a point card (the "prize", value 1..N) is revealed; both players
    SIMULTANEOUSLY and secretly bid one card from their hand (1..N). Higher bid
    wins the prize's points; tie discards it. imp_info: you see the prize, your
    own hand, scores and the win sequence, NOT the opponent's hand/bid.
  * action id = bid card VALUE - 1 (0-indexed card); to bid card c return c-1.
    Legal-action lines render as "{id} -> ..."; reply with the ID NUMBER only.
  * num_cards in {5, 8, 10, 13} (config_id % 4); points_order="random";
    returns_type="win_loss". Terminal reward normalized to [0,1]: WIN=1.0,
    TIE=0.5, LOSS=0.0.

Self-contained (no pyspiel — not installed in the trainer image).
Never-raises: any parse/scoring failure falls back to the smallest legal id.
"""

import math
import os
import random
import re
import zlib

import requests

from our_envs.pvp_prompts.pvp_prompt_loader import get_system_prompt, get_game_rules_prompt
from our_envs.pvp_prompts.pvp_state_format import reformat_to_pvp
from our_envs.pvp_tool_calling import assistant_move_message, tool_calling_enabled, WORKING, LONG_TERM, canonical_argmax
from our_envs.pvp_episode import run_toolcall_episode, MemoryPolicy
try:
    from our_envs.row_meta import set_meta  # R3 qgap side-channel (optional)
except Exception:  # never block generation if the side-channel module is absent
    def set_meta(*_a, **_k):
        return None


_REQUEST_TIMEOUT_SECONDS = 2400

# MUST match generate_trajectories._OPPONENT_CONFIG_PER_GAME["goofspiel"].
# The PvP eval baseline is an in-harness MCTS (PR #1217 game_eval/baseline), so
# we generate vs mcts@50 like the other PvP games. NOTE (smoke-test gap): the
# subagent god-validator-alignment doc once listed goofspiel as "random" — if
# the env-server rejects mcts for goofspiel, switch this to {"opponent": "random"}.
_OPPONENT_PAYLOAD = {"opponent": "mcts", "mcts_max_simulations": 50, "mcts_num_rollouts": 1}

_SYSTEM_PROMPT = get_system_prompt("goofspiel")

# Rank-relative mixed-bid kernel (v1.1 — escapes the matching trap). We weight
# each remaining card by a broad Gaussian in RANK space (the card's rank within
# our remaining hand vs the prize's strength), with the mode deliberately SHIFTED
# above the rank-match toward the winning "overbid-one" — NOT centred on bid==prize
# (the 78-13 trap). _RANK_SPREAD is the std-dev in [0,1] rank units (broad => more
# mixing / less exploitable, tune [0.25, 0.45]); _OVERBID_SHIFT moves the peak off
# the matching rank toward the cheap winning overbid (tune [0.08, 0.18]).
# Both are env-overridable so the OVERBID_SHIFT can be A/B-ed WITHOUT a code
# change: a prize-MATCHING bidder (shift 0.0) is a proven-strong baseline — our
# overbid shift bets against an "overbid-one Nash counter" that may be ABSENT from
# the LLM-vs-LLM PvP lineup. Set GOOFSPIEL_OVERBID_SHIFT=0.0 to test the
# matching-centred policy (keep the broad GOOFSPIEL_RANK_SPREAD either way). The
# validated lesson = SAMPLE around a sensible centre (we do); the centre offset is
# the open question.
# Guarded floats (audit P1): bare float() crashed the module import on
# empty/garbage A/B env values -> zero goofspiel rows. Import must never raise.
try:
    _RANK_SPREAD = float(os.environ.get("GOOFSPIEL_RANK_SPREAD") or "0.32")
except ValueError:
    _RANK_SPREAD = 0.32
_RANK_SPREAD = max(_RANK_SPREAD, 1e-6)  # 0 would ZeroDivision -> min(legal) fallback every turn
try:
    _OVERBID_SHIFT = float(os.environ.get("GOOFSPIEL_OVERBID_SHIFT") or "0.12")
except ValueError:
    _OVERBID_SHIFT = 0.12
# Explicit epsilon-uniform floor blended into the sampling distribution so EVERY
# legal card always keeps probability >= _EPSILON_FLOOR/m — anti-degenerate
# coverage on top of the broad Gaussian mixing (so no card ever starves of
# demonstrations). GOOFSPIEL_EPSILON_FLOOR=0.0 reproduces the pre-floor policy
# bit-for-bit (A-side for A/B). Env-overridable.
try:
    _EPSILON_FLOOR = float(os.environ.get("GOOFSPIEL_EPSILON_FLOOR") or "0.08")
except ValueError:
    _EPSILON_FLOOR = 0.08
_EPSILON_FLOOR = min(max(_EPSILON_FLOOR, 0.0), 1.0)
# P0 ENDGAME MATCHING-COLLAPSE FIX: with m legal cards the rank grid step is
# 1/(m-1), so a constant 0.12 shift is smaller than HALF a grid step whenever
# m <= 5 — the canonical_argmax label then rounds back onto the EXACT
# prize-matching card (verified: at N=5 game start every label == prize), i.e.
# the 78-13-exploitable pattern the winner trains a counter for. Floor the
# shift at 0.6/(m-1) (just past half a step, so the mode genuinely sits one
# rank above the match) — but ONLY for m >= 4: at m <= 3 the un-floored kernel
# is verified-correct (a floored m=2 would ALWAYS bid the higher card and m=3
# would waste mid cards on low prizes), so keep the plain _OVERBID_SHIFT there.
# GOOFSPIEL_ENDGAME_SHIFT_NUM=0 restores the pre-fix labels (A/B lever;
# empty string = default 0.6).
try:
    _ENDGAME_SHIFT_NUM = float(os.environ.get("GOOFSPIEL_ENDGAME_SHIFT_NUM") or "0.6")
except ValueError:
    _ENDGAME_SHIFT_NUM = 0.6  # garbage env value -> default (import must never raise)
# REMAINING-PRIZE RANK MODE (default ON): score the prize's strength by its
# RANK among the REMAINING prizes (current point card + future point cards from
# the "Remaining Point Cards:" line) instead of the value scale
# (prize-1)/(n_est-1). n_est = max(hand_max, prize) is corrupted once high
# cards are spent (verified: hand [2,5,9] vs prize 6 was labeled the
# guaranteed-losing bid 5 because n_est=9 read prize 6 as merely mid-strength
# even when 6 was the HIGHEST prize left). On ANY parse failure the old n_est
# formula is used (never-raise). GOOFSPIEL_PRIZE_RANK=0 restores the old
# scale everywhere (A/B lever; empty string = ON).
_PRIZE_RANK_MODE = (os.environ.get("GOOFSPIEL_PRIZE_RANK") or "1").strip().lower() in ("1", "true", "yes", "on")


def _flag_on(name: str) -> bool:
    """True iff env var ``name`` is set to a truthy value. Unset / '' / '0' /
    'false' / 'no' / 'off' => False (default OFF). Read at CALL time so an A/B run
    or a unit test can toggle the flag without a reimport. Never raises."""
    v = os.environ.get(name)
    if v is None:
        return False
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def _lockin_action_id(last_user: str, legal: "list[int]") -> "int | None":
    """GOOF_LOCKIN: return the LOWEST legal action id (dump the lowest card) IFF the
    game result is already DECIDED by the remaining prize points, else None (a
    contested state -> current sampling/label is UNCHANGED).

    Decided = |my_score - opp_score| > sum(remaining prize values, incl. the
    current point card): one side then cannot be caught / cannot catch up whatever
    the remaining bids are. CRITICAL GUARD — the lock-in NEVER fires in a contested
    state, so it can never purify toward the exploitable bid==prize matching line
    (goofspiel has a repo-proven overbid-one counter); it only dumps the lowest card
    in states whose win/loss is already sealed. Deterministic; never raises."""
    try:
        if not legal:
            return None
        if not re.search(r"Current point card:\s*\d+", last_user):
            return None
        sm = re.search(r"You are Player (\d+)", last_user)
        pid = int(sm.group(1)) if sm else 0
        ptm = re.search(r"Points:\s*(\d+)\s+(\d+)", last_user)
        if not ptm:
            return None
        p0, p1 = int(ptm.group(1)), int(ptm.group(2))
        my_score, opp_score = (p0, p1) if pid == 0 else (p1, p0)
        remaining = _parse_remaining_prizes(last_user)
        if not remaining:
            return None
        if abs(my_score - opp_score) > sum(remaining):
            return min(legal)  # already decided -> dump the lowest card
        return None
    except Exception:
        return None


def _goof_qgap(weights: "list[float]") -> "float | None":
    """Row-meta qgap for the canonical label = the bid-weight gap (teacher's OWN
    units) between the argmax card and the runner-up: 0.0 on a genuine tie, larger
    when the top bid is clearly ahead. None when < 2 weights. Never raises."""
    try:
        if not weights or len(weights) < 2:
            return None
        s = sorted(weights, reverse=True)
        return max(0.0, float(s[0] - s[1]))
    except Exception:
        return None


# --- state parsing ---------------------------------------------------------

def _legal_ids(last_user: str) -> list[int]:
    return [int(x) for x in re.findall(r"^\s*(\d+)\s*->", last_user, re.MULTILINE)]


def _parse_state(obs: str) -> "tuple[int, int] | None":
    """Parse (prize_value, player_id) from a goofspiel observation.

    prize_value = the point card currently up for bid (1..N). player_id = our
    seat (0/1), needed only for memory notes. Returns None if the prize cannot
    be recovered (the one field the bid policy needs). Never raises.
    """
    pm = re.search(r"Current point card:\s*(\d+)", obs)
    if not pm:
        return None
    prize = int(pm.group(1))
    sm = re.search(r"You are Player (\d+)", obs)
    pid = int(sm.group(1)) if sm else 0
    return prize, pid


def _parse_hand(obs: str, player_id: int) -> list[int]:
    """Our remaining bid-card VALUES (1..N), parsed from "P{pid} hand: v v v".
    Falls back to P0/P1 then to an empty list. Never raises."""
    for pat in (rf"P{player_id} hand:\s*([\d ]+)", r"P0 hand:\s*([\d ]+)", r"P1 hand:\s*([\d ]+)"):
        m = re.search(pat, obs)
        if m:
            try:
                return [int(x) for x in m.group(1).split()]
            except ValueError:
                return []
    return []


def _split_concat_ascending(digits: str, count: int) -> "list[int] | None":
    """Split OpenSpiel's separator-less ascending concatenation ("1345",
    "10111213") into exactly ``count`` strictly-ascending distinct values
    (1..13, no leading zeros) via deterministic DFS (1-digit token tried
    first). Returns None if no exact split exists. Never raises."""
    if not digits.isdigit():
        return None

    def rec(i: int, prev: int, left: int) -> "list[int] | None":
        if left == 0:
            return [] if i == len(digits) else None
        for width in (1, 2):
            j = i + width
            if j > len(digits):
                break
            tok = digits[i:j]
            if tok[0] == "0":
                continue
            val = int(tok)
            if val <= prev or val > 13:
                continue
            rest = rec(j, val, left - 1)
            if rest is not None:
                return [val] + rest
        return None

    return rec(0, 0, count)


def _parse_remaining_prizes(obs: str) -> "list[int] | None":
    """Recover the sorted REMAINING prizes (current point card + future point
    cards) from a goofspiel observation. reformat_goofspiel_to_pvp emits
    "Remaining Point Cards: {future prize values, ascending, CONCATENATED with
    no separators}" (OpenSpiel native: {1,3,4,5} -> "1345"), so double-digit
    decks are ambiguous — we disambiguate with the count invariant
    len(future) == len(our hand) - 1 (one prize + one hand card leave play per
    round). A separator-delimited value (defensive, raw env-server form) is
    accepted when its token count matches. Returns None on ANY inconsistency
    (caller falls back to the old n_est formula). Never raises."""
    try:
        pm = re.search(r"Current point card:\s*(\d+)", obs)
        if not pm:
            return None
        prize = int(pm.group(1))
        sm = re.search(r"You are Player (\d+)", obs)
        pid = int(sm.group(1)) if sm else 0
        hand = _parse_hand(obs, pid)
        m = len(hand)
        if m == 0:
            return None
        # Same-line anchor as reformat_goofspiel_to_pvp ([ \t]*/[^\n]*): the
        # value is EMPTY on the final round; \s* would swallow the newline.
        rm = re.search(r"Remaining [Pp]oint [Cc]ards:[ \t]*([^\n]*)", obs)
        if not rm:
            return None
        value = rm.group(1).strip()
        tokens = re.findall(r"\d+", value)
        if not value:
            future = [] if m == 1 else None
        elif len(tokens) > 1:
            future = sorted(int(t) for t in tokens) if len(tokens) == m - 1 else None
        else:
            future = _split_concat_ascending(value, m - 1)
        if future is None or prize in future:
            return None  # unparseable / deck values must be distinct
        remaining = sorted(future + [prize])
        if len(remaining) != m:
            return None
        return remaining
    except Exception:
        return None


# --- goofspiel mechanics (mixed bidder, self-contained, no pyspiel) --------

def _bid_weights(prize: int, legal: list[int], remaining: "list[int] | None" = None) -> list[float]:
    """Rank-relative mixed sampling weights over legal action ids (v1.1).

    The legal set IS our remaining hand (card value = id + 1). We score each card
    by how well its RANK WITHIN OUR REMAINING HAND matches the prize's strength
    (rank-relative — so spent high cards correctly shift bids, unlike the old
    absolute bid==prize centring), with the peak deliberately SHIFTED above the
    match toward the cheap winning "overbid-one" and a broad spread for genuine
    mixing. This escapes both the argmax sweep AND the matching trap (78-13
    exploitable). Weights are strictly positive so every legal card keeps non-zero
    probability; the caller SAMPLES (never argmax).

    ``remaining`` (optional) = the parsed remaining prizes INCLUDING the current
    one; when valid (and _PRIZE_RANK_MODE) the prize's strength is its rank among
    them instead of the n_est value scale. The overbid shift is floored at
    _ENDGAME_SHIFT_NUM/(m-1) for m >= 4 so the label mode clears half a rank-grid
    step (endgame matching-collapse fix); m <= 3 keeps the plain shift.
    """
    cards = sorted(aid + 1 for aid in legal)
    m = len(cards)
    if m <= 1:
        return [1.0] * len(legal)
    if _PRIZE_RANK_MODE and remaining and prize in remaining:
        rem_sorted = sorted(remaining)
        if len(rem_sorted) == 1:
            prize_frac = 0.5
        else:
            prize_frac = rem_sorted.index(prize) / (len(rem_sorted) - 1)  # rank among remaining prizes
    else:
        n_est = max(cards[-1], prize)                   # deck size N (exact at game start)
        prize_frac = (prize - 1) / max(n_est - 1, 1)    # prize strength in [0,1]
    rank_frac = {c: i / (m - 1) for i, c in enumerate(cards)}  # card's rank within our hand
    if m >= 4:
        effective_shift = max(_OVERBID_SHIFT, _ENDGAME_SHIFT_NUM / (m - 1))
    else:
        effective_shift = _OVERBID_SHIFT
    weights = []
    for aid in legal:
        gap = rank_frac[aid + 1] - prize_frac
        weights.append(math.exp(-((gap - effective_shift) ** 2) / (2.0 * _RANK_SPREAD ** 2)))
    # epsilon-uniform floor: (1-eps) * normalized(weights) + eps/m uniform. The
    # returned weights are now a proper distribution with min mass eps/m per card.
    total = sum(weights)
    m_legal = len(legal)
    if total <= 0.0 or m_legal == 0:
        return [1.0] * len(legal)
    floor = _EPSILON_FLOOR / m_legal
    return [(1.0 - _EPSILON_FLOOR) * (w / total) + floor for w in weights]


def _expert_action_id(last_user: str) -> "int | None":
    """Pick a legal action id by SAMPLING a value-proportional mixed bid."""
    legal = _legal_ids(last_user)
    if not legal:
        return None

    parsed = _parse_state(last_user)
    if parsed is None:
        return min(legal)  # prize unreadable -> smallest legal id (concede cheaply)
    prize, _pid = parsed

    remaining = _parse_remaining_prizes(last_user) if _PRIZE_RANK_MODE else None
    weights = _bid_weights(prize, legal, remaining)
    total = sum(weights)
    if total <= 0.0:
        return random.choice(legal)  # degenerate weights -> uniform mix
    return random.choices(legal, weights=weights, k=1)[0]


def get_expert_action(messages: list[dict]) -> str:
    """Return the chosen action id as a string. Never raises."""
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    raw = re.findall(r"^\s*(\d+)\s*->", last_user, re.MULTILINE)
    forced = len(raw) == 1
    # GOOF_LOCKIN (default OFF): in an already-DECIDED state dump the LOWEST card
    # deterministically. Never fires in a contested state, so the sampler below is
    # untouched there. OFF => the sampler is unchanged everywhere.
    if _flag_on("GOOF_LOCKIN"):
        try:
            legal = _legal_ids(last_user)
            lock = _lockin_action_id(last_user, legal) if legal else None
        except Exception:
            lock = None
        if lock is not None:
            set_meta(qgap=None, forced=forced)
            return str(lock)
    try:
        chosen = _expert_action_id(last_user)
    except Exception:
        chosen = None
    if chosen is None:
        set_meta(qgap=None, forced=forced)
        return min(raw, key=int) if raw else "0"
    set_meta(qgap=None, forced=forced)  # sampled -> no deterministic gap
    return str(chosen)


def get_label_action(messages: list[dict]) -> str:
    """Tier-1 CANONICAL label (deterministic, pure function of the observation):
    the ARGMAX of persona-A's off-center rank-relative bid weights, ties broken by an
    obs-hash. Reuses the SAME _bid_weights kernel the sampling expert builds, so the
    label sits at the OFF-CENTER (overbid-shifted) mode — NOT bid==prize — and thus
    avoids the 78-13 matching trap while giving the SFT a single-valued (state->id)
    target. Never raises."""
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    try:
        legal = _legal_ids(last_user)
        if not legal:
            raise ValueError
        forced = len(legal) == 1
        # GOOF_LOCKIN (default OFF): in an already-DECIDED state dump the LOWEST
        # card deterministically. NEVER fires in a contested state, so it cannot
        # purify toward the exploitable bid==prize matching line. OFF => the
        # canonical off-center label below is unchanged.
        if _flag_on("GOOF_LOCKIN"):
            lock = _lockin_action_id(last_user, legal)
            if lock is not None:
                set_meta(qgap=None, forced=forced)
                return str(lock)
        parsed = _parse_state(last_user)
        if parsed is None:
            set_meta(qgap=None, forced=forced)
            return str(min(legal))
        prize, _pid = parsed
        remaining = _parse_remaining_prizes(last_user) if _PRIZE_RANK_MODE else None
        weights = _bid_weights(prize, legal, remaining)
        choice = canonical_argmax(legal, weights, last_user)
        set_meta(qgap=_goof_qgap(weights), forced=forced)
        return str(choice)
    except Exception:
        raw = re.findall(r"^\s*(\d+)\s*->", last_user, re.MULTILINE)
        set_meta(qgap=None, forced=(len(raw) == 1))
        return min(raw, key=int) if raw else "0"


# --- SECOND TEACHER (persona B) — absolute prize-MATCHING softmax -----------
# League/population self-play diversity (SELFPLAY_DUAL_TEACHER): seat 1 plays a
# DIFFERENT-but-equally-strong bidder. Persona B is the tournament-winning boss
# policy: a softmax over the ABSOLUTE distance |card_value - prize| (centred AT
# the prize, shift 0), temperature 0.5, sharing the SAME epsilon-uniform floor.
# It diverges from persona A (rank-relative gap + 0.12 overbid-shift + broad
# Gaussian) at the modal action (match-prize vs overbid-one) and increasingly in
# mid/late game (absolute distance does not rescale as high cards deplete), yet is
# NOT weaker: matching is the exact best response vs uniform-random play (Ross
# 1971, +28 at N=13) and is empirically the winning PvP policy. The 0.08 floor +
# temp-0.5 mixing are LOAD-BEARING — argmax-matching re-exposes the 78-13
# "overbid-one" counter (Rhoads-Bartholdi 2012), so we SAMPLE, never argmax.
try:
    _MATCH_TEMP = float(os.environ.get("GOOFSPIEL_MATCH_TEMP") or "0.5")
except ValueError:
    _MATCH_TEMP = 0.5
_MATCH_TEMP = max(_MATCH_TEMP, 1e-6)


def _bid_weights_matching(prize: int, legal: list[int]) -> list[float]:
    """Persona-B sampling weights: softmax(-|card_value - prize| / temp) over legal
    ids, with the same epsilon-uniform floor as persona A. Strictly positive proper
    distribution; the caller SAMPLES (never argmax). Never raises."""
    m_legal = len(legal)
    if m_legal <= 1:
        return [1.0] * m_legal
    scores = [-abs((aid + 1) - prize) / _MATCH_TEMP for aid in legal]
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]            # stable softmax
    total = sum(exps)
    if total <= 0.0:
        return [1.0] * m_legal
    floor = _EPSILON_FLOOR / m_legal
    return [(1.0 - _EPSILON_FLOOR) * (e / total) + floor for e in exps]


def _expert_action_id_b(last_user: str) -> "int | None":
    """Persona-B pick: SAMPLE a prize-matching softmax bid."""
    legal = _legal_ids(last_user)
    if not legal:
        return None
    parsed = _parse_state(last_user)
    if parsed is None:
        return min(legal)
    prize, _pid = parsed
    weights = _bid_weights_matching(prize, legal)
    total = sum(weights)
    if total <= 0.0:
        return random.choice(legal)
    return random.choices(legal, weights=weights, k=1)[0]


def get_expert_action_b(messages: list[dict]) -> str:
    """Persona-B (prize-matching softmax) action id as a string. Never raises."""
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    raw = re.findall(r"^\s*(\d+)\s*->", last_user, re.MULTILINE)
    forced = len(raw) == 1
    # GOOF_LOCKIN (default OFF): decided-state lock-in -> dump the LOWEST card
    # deterministically (never in contested states). OFF => the sampler is unchanged.
    if _flag_on("GOOF_LOCKIN"):
        try:
            legal = _legal_ids(last_user)
            lock = _lockin_action_id(last_user, legal) if legal else None
        except Exception:
            lock = None
        if lock is not None:
            set_meta(qgap=None, forced=forced)
            return str(lock)
    try:
        chosen = _expert_action_id_b(last_user)
    except Exception:
        chosen = None
    if chosen is None:
        set_meta(qgap=None, forced=forced)
        return min(raw, key=int) if raw else "0"
    set_meta(qgap=None, forced=forced)  # sampled -> no deterministic gap
    return str(chosen)


# --- memory policy (tool_call / #1201 mode) --------------------------------

class GoofspielMemoryPolicy(MemoryPolicy):
    """Goofspiel is imperfect-info, so the edge is hand/tempo tracking. working =
    this round's prize + my remaining hand + my bid (so the model learns to
    reason over its own dwindling cards); long_term = a durable reminder to keep
    high cards for high prizes and mix bids (the unexploitability lesson). We
    cannot see the opponent's hand/bid, so notes stay grounded in our own state
    and the published strategy, never fabricated opponent reads."""

    def __init__(self):
        self._max_prize = 0

    def observe(self, reformatted_obs: str) -> None:
        parsed = _parse_state(reformatted_obs)
        if parsed:
            self._max_prize = max(self._max_prize, parsed[0])

    def turn_writes(self, reformatted_obs, action_id, state, turn_idx):
        # Winner-recipe rarity gate (audit P1): memory notes fire on ~25% of
        # turns (stable crc32 - not every turn: dense per-turn writes trained
        # the memory-first habit that drowned game_action, 2026-06-22).
        if zlib.crc32(f"{action_id}:{turn_idx}:{len(reformatted_obs)}".encode()) % 4 != 0:
            return []
        parsed = _parse_state(reformatted_obs)
        if not parsed:
            return []
        prize, pid = parsed
        hand = _parse_hand(reformatted_obs, pid)
        try:
            bid_card = int(action_id) + 1
        except (TypeError, ValueError):
            bid_card = action_id
        note = (
            f"round {turn_idx + 1}: prize {prize}; my hand {sorted(hand)}; "
            f"bid card {bid_card}"
        )
        return [(WORKING, "rewrite", 1, note)]

    def reflect_writes(self, outcome, state):
        note = (
            f"goofspiel {outcome}: spend high cards on high prizes, sacrifice the "
            f"lowest card on prizes you concede, and mix bids so you cannot be read"
        )
        return [(LONG_TERM, "append", 1, note)]


# --- episode generation ----------------------------------------------------

def generate_expert_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 30,
) -> "tuple[list[dict], float] | None":
    """Run one Goofspiel game vs the env-server opponent using the mixed expert.

    Returns ``(messages, final_reward)`` on success (final_reward in [0,1],
    0.5 = draw, > 0.5 = win), or None on env-server failure.

    max_turn=30: goofspiel is at most N<=13 bidding rounds (we are prompted once
    per round in the convert_to_turn_based wrapper), plus invalid-retry slack —
    well above a full game so the terminal reward is never dropped by the loop
    cap. In tool_call mode (#1201) the game is distilled as per-turn-fresh
    examples with memory writes via run_toolcall_episode; bare_id mode keeps the
    legacy single growing-conversation form below.
    """
    if tool_calling_enabled():
        return run_toolcall_episode(
            game_name="goofspiel",
            game_id=game_id,
            env_endpoint=env_endpoint,
            opponent_payload=_OPPONENT_PAYLOAD,
            max_turn=max_turn,
            rules_prompt=get_game_rules_prompt("goofspiel"),
            expert_action_fn=get_expert_action,
            obs_transform=lambda raw: reformat_to_pvp(raw, "goofspiel"),
            policy=GoofspielMemoryPolicy(),
        )

    reset_payload = {"task_id": game_id, "seed": game_id, **_OPPONENT_PAYLOAD}
    try:
        res = requests.post(
            f"{env_endpoint}/reset", json=reset_payload, timeout=_REQUEST_TIMEOUT_SECONDS
        )
        res.raise_for_status()
        block = res.json()["result"]
        episode_id = block.get("episode_id", "")
        observation = block.get("observation", "")
    except Exception as exc:
        print(f"[env] Reset failed (game {game_id}): {exc}")
        return None

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": reformat_to_pvp(observation, "goofspiel")},
    ]

    final_reward = 0.0
    for _ in range(max_turn):
        action = get_expert_action(messages)
        messages.append(assistant_move_message(action))

        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action, "episode_id": episode_id},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            step_res.raise_for_status()
            step_block = step_res.json()["result"]
            observation = step_block.get("observation", "")
            done = step_block.get("done", False)
            if done:
                step_reward = step_block.get("reward")
                if isinstance(step_reward, (int, float)):
                    final_reward = float(step_reward)
        except Exception as exc:
            print(f"[env] Step failed (game {game_id}): {exc}")
            return None

        if done:
            break
        messages.append({"role": "user", "content": reformat_to_pvp(observation, "goofspiel")})
    else:
        print(f"[env] max_turn={max_turn} reached (game {game_id})")

    return messages, final_reward
