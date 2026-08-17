"""Expert trajectory generator for Liar's Dice SFT.

Expert policy: analytic bid-probability heuristic. For each legal action,
estimate the probability that the bid is true given our private dice and
the total dice in play; for "Liar" actions, estimate 1 - P(current_bid).
Bid actions on a face we hold more of get a small semi-bluff bonus
(LD-C edge); the action is then sampled via a near-greedy softmax.

The Bayesian opponent-inference second opinion was REMOVED (LD-A): a
deterministic analytic line was already readable by a strong peer (lost
0-200 at eval temp=0), and the 50/50 Bayesian mix on LIAR only added
inconsistency to the call demonstrations without fixing predictability.
Published tournament winners use the pure analytic policy with no
opponent model.

Opponent during play: MCTS at the simulation count that the validator uses
for env evaluation (see ENVIRONMENTS["liars_dice"] in
G.O.D/validator/core/constants.py: mcts_max_simulations=225, num_rollouts=1).
Matching the validator's eval opponent means the resulting expert
trajectories teach the model to handle exactly the playstyle it will be
graded against.
"""

import hashlib
import math
import os
import random
import re
import zlib

import requests

from our_envs.liar_dice_opponent_modeling import parse_game_state, bid_probability, GameState
from our_envs.liar_dice_cfr import mccfr_distribution
from our_envs.liar_dice_dbr import dbr_distribution, sharpen
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
# R1 fix (2026-06-20): the old near-greedy temp 0.02 made the teacher a PURE
# strategy -> exploitable (the 0-200 LD loss signature). Mix over equity-
# equivalent bids instead: keep every action within _EPSILON_BAND of the top
# P(true), then softmax-sample WITHIN that band. One dominant move -> band is a
# singleton (still played); several near-equal -> randomized => unreadable, no
# equity sacrificed. (reference_goofspiel_pvp_research / pvp_strength_strategy.)
_EPSILON_BAND = 0.06       # P(true) margin defining "equity-equivalent" actions
_BAND_TEMPERATURE = 0.05   # softmax temp WITHIN the band (mild preference, real mixing)

# Label entropy. The blueprint's raw mix is the right POLICY but a poor TARGET: the
# student sees one state answered several ways and reproduces neither the mix nor the
# mode. <1 sharpens (p^(1/T) renormalised).
#
# The sharpening is SELECTIVE, which is why it is safe to do at all — measured on a
# real blueprint row (top prob after sharpening | Shannon entropy):
#
#              blueprint CONFIDENT (top .625)   blueprint INDIFFERENT (top .300)
#   T=1.0 raw          .625  H=1.54                     .300  H=1.98
#   T=0.6              .827  H=0.93                     .335  H=1.96   <- default
#   T=0.35             .968  H=0.25                     .399  H=1.89
#
# Where the blueprint genuinely mixes, sharpening barely moves it (H 1.98 -> 1.96):
# the equilibrium's indifference — the part that makes it unexploitable — survives.
# Only spots the blueprint already prefers get crisp enough to imitate. 0.35 was
# rejected: .968 on a confident spot is argmax in all but name, and a readable
# deterministic demonstrator is what lost R1 0-400. LD_LABEL_TEMP=1 restores the raw
# mix for A/B.
_LABEL_TEMPERATURE = float(os.environ.get("LD_LABEL_TEMP", "0.6") or "0.6")
_HELD_FACE_BONUS = 0.04  # LD-C: per-matching-die prob bonus for bids on a face we hold (semi-bluff)

# LD-D (audit 2026-06-13): opponent-reply EV value-betting — DEFAULT OFF after a
# probe A/B showed it REGRESSES the greedy policy vs MCTS@225 (82.7% -> 74.0%,
# 2026-06-13). The greedy P(bid literally true) policy already wins ~80% vs the
# eval-strength MCTS (the earlier "LD 25%" was a log misread: draws=0 means that
# 78.8% row was liars_dice, not leduc). Value-betting backfires vs a strong
# caller that declines the bait and raises us into a worse spot. Kept behind the
# flag (LD_OPPONENT_EV=1) for any FUTURE PvP-validated experiment — the gen
# win-rate vs MCTS is not the PvP score, so LD's PvP readability is still open —
# but production ships greedy. EV models opp reply: P(opp call)=1-prior_plaus,
# value = P(opp call)*P(bid true) + P(opp raise)*_V_CONTINUE.
_LD_OPPONENT_EV = os.environ.get("LD_OPPONENT_EV", "0").strip() not in ("0", "false", "no", "")
_V_CONTINUE = 0.42  # estimated win-prob when the opponent raises instead of calling our bid

# Truthful-ladder opponent persona (2026-07-04): the tournament winner's LD expert
# is a Bayesian truth-maximizer — it never bluffs, deterministically climbs "safe"
# bid ladders (bids whose P(literally true | own dice) is high, wild 6s counted)
# and calls Liar only when no safe rung remains. Persona B (the seat-B driver in
# league self-play) plays that policy in ~50% of GAMES — chosen deterministically
# per game from a stable hash of OUR OWN DICE + total dice (constant within a
# hand; the driver passes only the current obs, so obs-text keys re-flip per
# turn) — so the exact state
# distribution winner-family distilled models produce lands in OUR training data
# with OUR canonical blueprint labels stamping the punishes. DRIVER-SIDE ONLY:
# get_label_action is untouched. Gate: LD_TRUTHFUL_LADDER (default on;
# empty-string-safe).
_TRUTHFUL_LADDER_ENABLED = (
    (os.environ.get("LD_TRUTHFUL_LADDER") or "1").strip().lower() not in ("0", "false", "no", "off")
)
_TL_TEMPERATURES = (0.1, 0.2, 0.3, 0.5)  # per-game softmax temp, hash-selected
_TL_SAFE_P = 0.5  # a bid is "safe" when P(bid literally true | own dice) >= this


class LiarDiceMemoryPolicy(MemoryPolicy):
    """LD is imperfect-info, so opponent reads matter. working = my dice + the
    opponent's standing bid + my move; long_term = how aggressively the opponent
    bids (the key LD tell). Regex-only, never-raises."""

    def __init__(self):
        self._max_q = 0
        self._max_f = 0
        self._bids = 0

    def observe(self, obs: str) -> None:
        m = re.search(r'Current bid:\s*"?(\d+)-(\d+)"?', obs)
        if m:
            q, f = int(m.group(1)), int(m.group(2))
            self._bids += 1
            if (q, f) > (self._max_q, self._max_f):
                self._max_q, self._max_f = q, f

    def turn_writes(self, obs, action_id, state, turn_idx):
        # APPEND one short line per turn, never rewrite. The expert's label is scored
        # against a posterior built from the opponent's WHOLE bid sequence, but the
        # eval's state text shows only the single standing bid — so the history has to
        # live somewhere the model can see, or the label depends on information the row
        # it labels does not contain (i.e. it is not learnable at all). A rewrite kept
        # only the latest turn and destroyed exactly that history; a rarity gate left
        # holes in it. Appending every turn accumulates the bid transcript the label
        # actually conditions on.
        #
        # The forfeit risk this replaces (dense writes training a memory-first habit
        # that drops game_action) is handled structurally instead: assistant_turn_message
        # always emits [memory fragments..., game_action], so game_action is present in
        # every turn we write. Notes stay one short line to keep the surface small.
        bm = re.search(r'Current bid:\s*"?(\d+)-(\d+)"?', obs)
        opp = f"opp bid {bm.group(1)}-{bm.group(2)}" if bm else "opp opened"
        return [(WORKING, "append", 1, f"t{turn_idx}: {opp}; I play {action_id}")]

    def reflect_writes(self, outcome, state):
        if self._bids == 0:
            tend = "made no bid (I opened / called early)"
        else:
            tend = f"bid up to {self._max_q}-{self._max_f} across {self._bids} bids"
        return [(LONG_TERM, "append", 1, f"opp {tend}")]

# Authoritative opponent strength: match validator's eval config.
_OPPONENT_PAYLOAD = {
    "opponent": "mcts",
    "mcts_max_simulations": 225,
    "mcts_num_rollouts": 1,
}


# System prompt loaded from the validator's YAML template so that the
# string the model sees during SFT distillation matches the one the PvP
# eval container produces via BaseGameAgent.generate_system_prompt().
# The previous hand-written prompt added a "# Strategy Tips" section the
# validator never emits; that single divergence was enough to lose every
# eval game (LD ends in 1-2 turns so a single wrong output is terminal).
_SYSTEM_PROMPT = get_system_prompt("liars_dice")


def _softmax_weights(probs: list[float], temperature: float) -> list[float]:
    """Convert raw probabilities to softmax weights."""
    if temperature <= 0:
        best = max(range(len(probs)), key=lambda i: probs[i])
        return [1.0 if i == best else 0.0 for i in range(len(probs))]
    scaled = [p / temperature for p in probs]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def _own_face_support(our_dice: list[int], face: int) -> int:
    """Count own dice matching face with wild-6 (face != 6 counts both)."""
    if face == 6:
        return sum(1 for d in our_dice if d == 6)
    return sum(1 for d in our_dice if d == face or d == 6)


def _prior_plausibility(bid, total_dice: int) -> float:
    """P(bid true) from a neutral observer with NO known dice — the opponent's
    marginal view of how believable the bid is (we cannot see their dice, so the
    no-information prior is our best estimate of their average perspective)."""
    if bid is None:
        return 1.0
    return bid_probability(bid, GameState(our_dice=[], total_dice=total_dice, current_bid=None, actions=[]))


def _action_ev(action, gs: GameState) -> float:
    """Opponent-reply EV of an action (win-probability in [0,1]).

    LIAR is terminal: its prob already equals P(win by calling) = 1-P(current
    bid true). For a BID: the opponent either CALLS (we win iff the bid is
    literally true given our dice = action.prob) or RAISES (round continues,
    valued at _V_CONTINUE). The opponent calls implausible bids, so model
    P(opp calls) = 1 - prior_plausibility(bid). This rewards TRUE-but-aggressive
    value bets (baited calls we win) over minimal readable bids that just get
    raised.
    """
    if action.is_liar or action.bid is None:
        return action.prob
    p_true = action.prob  # P(bid true | our dice) = win prob if the opponent calls
    opp_call = min(max(1.0 - _prior_plausibility(action.bid, gs.total_dice), 0.0), 1.0)
    return opp_call * p_true + (1.0 - opp_call) * _V_CONTINUE


def get_expert_action(messages: list[dict], _use_blueprint: bool = True) -> str:
    """Pick an action via opponent-reply EV (LD-D) with a mild semi-bluff edge.

    LD-D: score each action by _action_ev (P(opp call)*P(bid true) +
    P(opp raise)*V_continue) instead of greedy P(bid literally true), so the
    demonstrator value-bets true-but-implausible bids and stops opening with the
    readable minimal-truthful raise that a strong peer exploits. LD-C: a small
    per-held-die bonus remains as a semi-bluff tiebreak on bids. Set
    LD_OPPONENT_EV=0 to fall back to the legacy greedy-truth score.
    """
    try:
        gs = parse_game_state(messages)
    except Exception:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        raw = re.findall(r"^(\d+)\s*->", last_user, re.MULTILINE)
        return min(raw, key=int) if raw else "0"

    if not gs.actions:
        return "0"

    # PRIMARY: the memoryless MCCFR Nash blueprint (~0.47 best-response
    # exploitability in the self-check vs ~0.79 for the analytic band below and
    # ~1.56 uniform). Keyed on (own dice, current bid) = exactly what the agent
    # observes, so it is reconstructable at eval. Falls through to the epsilon-band
    # when the blueprint has no usable mass (asymmetric/odd configs or a cold-build
    # miss) — the band is a stronger fallback than a plain EV policy. SAMPLE the
    # mix. Disable the whole MCCFR path via LD_CFR_EXPERT=0.
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if _use_blueprint:
        _dist = mccfr_distribution(last_user, [a.action_id for a in gs.actions])
        if _dist:
            # Blueprint = the floor, never dropped. Two layers on top:
            #  1. dbr_distribution tilts it toward a posterior-scored response in
            #     proportion to how many opponent bids we have actually seen
            #     (Pconf = Pmax*n/(s+n); n=0 leaves the blueprint untouched). This is
            #     the metered middle of the Nash<->best-response axis rather than the
            #     brittle best-response end — see liar_dice_dbr for the evidence.
            #  2. sharpen lowers the label's entropy so the student can imitate it. A
            #     raw Nash mix answers the same state differently every time and the
            #     student learns neither the mix nor the mode.
            _dist = dbr_distribution(messages, _dist)
            _dist = sharpen(_dist, _LABEL_TEMPERATURE)
            _ids = list(_dist)
            return str(random.choices(_ids, weights=[_dist[a] for a in _ids], k=1)[0])

    # Two separate quantities (R1 band-bug fix): equity defines BAND MEMBERSHIP
    # (so the band stays a true equity-equivalence set), and a capped held-face
    # semi-bluff nudge biases only the WITHIN-BAND softmax weight. Previously the
    # nudge was folded into the membership score and added to bids but NOT to
    # Liar — on a many-held face (held up to 5 -> +0.20, dwarfing the 0.06 band)
    # it deterministically over-bid that face (a readable tell) and could exclude
    # a higher-equity Liar call from the band. Capping at _EPSILON_BAND and
    # keeping it out of membership guarantees it can never drop a stronger action.
    equities: list[float] = []
    biases: list[float] = []
    for action in gs.actions:
        equities.append(_action_ev(action, gs) if _LD_OPPONENT_EV else action.prob)
        if action.bid is not None:
            held = _own_face_support(gs.our_dice, action.bid.face)
            biases.append(min(_HELD_FACE_BONUS * held, _EPSILON_BAND))
        else:
            biases.append(0.0)  # Liar carries no semi-bluff bias by design.

    # Epsilon-band mixing (R1 fix): band membership is PURE equity (within
    # _EPSILON_BAND of the best equity); softmax-sample within the band using
    # equity + the capped semi-bluff bias as the weight only, so the
    # demonstration is a MIXED strategy, not the readable argmax that lost R1.
    top = max(equities)
    band = [(a, eq, b) for a, eq, b in zip(gs.actions, equities, biases) if eq >= top - _EPSILON_BAND]
    band_actions = [a for a, _, _ in band]
    band_weights = _softmax_weights([eq + b for _, eq, b in band], _BAND_TEMPERATURE)
    chosen = random.choices(band_actions, weights=band_weights, k=1)[0]
    return str(chosen.action_id)


# --------------------------------------------------------------------------- #
# R1 LABEL PURIFICATION (LD_PURIFY, default OFF) — Ganzfried-Sandholm.
#
# A near-uniform mixed rollout/equity distribution SAMPLED per state produces
# pattern-inconsistent labels ("stochastic-label disease") that hurt a distilled
# student at greedy eval. Purify = threshold the distribution (MCCFR blueprint, or
# the equity band on a blueprint miss) to its SUPPORT at LD_ROLLOUT_EPS of the max
# mass, then pick DETERMINISTICALLY by action EQUITY (P(bid true) / call-EV — the
# value the proven greedy ~80% policy trusts) rather than by obs-hash — so similar
# infosets get the SAME label. Flag OFF => get_label_action is byte-identical to
# the canonical-argmax path below (greedy ~80% behaviour stays reachable OFF).
# --------------------------------------------------------------------------- #
def _ld_flag_on(name: str) -> bool:
    """OFF for unset / 0 / false / no / off / empty; ON otherwise. Read at CALL
    time so the flag is reversible by simply unsetting it (no re-import)."""
    return os.environ.get(name, "").strip().lower() not in ("0", "false", "no", "off", "")


def _ld_flag_float(name: str, default: float) -> float:
    """Parse a float env var (empty/unset/garbage -> default). Never raises."""
    try:
        return float(os.environ.get(name, "").strip() or default)
    except Exception:
        return default


def _purify_pick(ids: list, weights: dict, value: dict, eps: float):
    """Ganzfried-Sandholm purify of a mixed strategy. Drop candidate `ids` whose
    mixed-strategy `weight` is below `eps` of the max weight (keep the support,
    discard the noise tail), then pick the survivor with the highest `value`
    (id -> action equity / call-EV), the mixed weight as a deterministic secondary
    key. Returns (chosen_id, qgap) where qgap = the prob gap between the top-2
    candidate weights (>=0.0). Pure + deterministic (no RNG, no obs-hash)."""
    ids = list(ids)
    if not ids:
        return None, None
    max_w = max(weights[i] for i in ids)
    thresh = max_w * eps
    survivors = [i for i in ids if weights[i] >= thresh] or ids
    chosen = max(survivors, key=lambda i: (value.get(i, float("-inf")), weights[i]))
    ws = sorted((weights[i] for i in ids), reverse=True)
    qgap = float(ws[0] - ws[1]) if len(ws) >= 2 else 0.0
    return chosen, qgap


def _ld_purified_label(gs, last_user: str) -> str:
    """LD_PURIFY label: threshold the MCCFR blueprint (or the equity band on a
    blueprint miss) to its support, then pick DETERMINISTICALLY by action equity.
    Sets qgap/forced right before returning. Never raises (shares get_label_action's
    text fallbacks)."""
    if gs is None or not gs.actions:
        set_meta(qgap=None, forced=False)
        raw = re.findall(r"^(\d+)\s*->", last_user, re.MULTILINE)
        return min(raw, key=int) if raw else "0"
    forced = (len(gs.actions) == 1)
    # value per action_id = the equity the greedy policy trusts (call-EV under
    # LD_OPPONENT_EV, else P(bid literally true) / P(win by calling Liar)).
    value = {a.action_id: (_action_ev(a, gs) if _LD_OPPONENT_EV else a.prob) for a in gs.actions}
    try:
        _dist = mccfr_distribution(last_user, [a.action_id for a in gs.actions])
    except Exception:
        _dist = None
    try:
        if _dist:
            ids = [i for i in _dist if _dist[i] > 0.0] or list(_dist)
            weights = {i: _dist[i] for i in ids}
        else:
            # equity epsilon-band (same construction as the canonical fallback),
            # then purify its softmax weights.
            equities: list[float] = []
            biases: list[float] = []
            for action in gs.actions:
                equities.append(value[action.action_id])
                if action.bid is not None:
                    held = _own_face_support(gs.our_dice, action.bid.face)
                    biases.append(min(_HELD_FACE_BONUS * held, _EPSILON_BAND))
                else:
                    biases.append(0.0)
            top = max(equities)
            band = [(a, eq, b) for a, eq, b in zip(gs.actions, equities, biases) if eq >= top - _EPSILON_BAND]
            ids = [a.action_id for a, _, _ in band]
            band_w = _softmax_weights([eq + b for _, eq, b in band], _BAND_TEMPERATURE)
            weights = {aid: w for aid, w in zip(ids, band_w)}
        chosen, qgap = _purify_pick(ids, weights, value, _ld_flag_float("LD_ROLLOUT_EPS", 0.15))
        if chosen is None:
            raise ValueError("no candidate")
        set_meta(qgap=qgap, forced=forced)
        return str(chosen)
    except Exception:
        set_meta(qgap=None, forced=forced)
        raw = re.findall(r"^(\d+)\s*->", last_user, re.MULTILINE)
        return min(raw, key=int) if raw else "0"


def get_label_action(messages: list[dict]) -> str:
    """Tier-1 CANONICAL label (deterministic, pure function of the observation): the
    ARGMAX of the SAME MCCFR Nash blueprint distribution get_expert_action SAMPLES,
    ties broken by an obs-hash; falls back to the argmax of the equity epsilon-band
    weights when the blueprint has no mass. A single-valued (state->action_id)
    Nash-support target — not the readable minimal-truthful raise that lost R1, and
    not a per-game sample. Never raises.

    LD_PURIFY (default OFF): route to the Ganzfried-Sandholm purified pick instead.
    OFF => byte-identical to the canonical path below."""
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    try:
        gs = parse_game_state(messages)
    except Exception:
        gs = None
    if _ld_flag_on("LD_PURIFY"):
        return _ld_purified_label(gs, last_user)
    if gs is None or not gs.actions:
        raw = re.findall(r"^(\d+)\s*->", last_user, re.MULTILINE)
        return min(raw, key=int) if raw else "0"
    try:
        _dist = mccfr_distribution(last_user, [a.action_id for a in gs.actions])
        if _dist:
            ids = list(_dist)
            return str(canonical_argmax(ids, [_dist[a] for a in ids], last_user))
        equities: list[float] = []
        biases: list[float] = []
        for action in gs.actions:
            equities.append(_action_ev(action, gs) if _LD_OPPONENT_EV else action.prob)
            if action.bid is not None:
                held = _own_face_support(gs.our_dice, action.bid.face)
                biases.append(min(_HELD_FACE_BONUS * held, _EPSILON_BAND))
            else:
                biases.append(0.0)
        top = max(equities)
        band = [(a, eq, b) for a, eq, b in zip(gs.actions, equities, biases) if eq >= top - _EPSILON_BAND]
        band_ids = [a.action_id for a, _, _ in band]
        band_w = _softmax_weights([eq + b for _, eq, b in band], _BAND_TEMPERATURE)
        return str(canonical_argmax(band_ids, band_w, last_user))
    except Exception:
        raw = re.findall(r"^(\d+)\s*->", last_user, re.MULTILINE)
        return min(raw, key=int) if raw else "0"


def _stable_obs_hash(text: str) -> int:
    """Stable (process- and run-invariant) 64-bit hash of an observation string.
    hashlib, NOT built-in hash(): PYTHONHASHSEED randomises the latter per
    process, which would break the per-game persona determinism."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8", "replace")).digest()[:8], "big")


def _truthful_ladder_action(messages: list[dict], game_hash: int) -> "str | None":
    """Winner-mimic driver policy (persona B, ~50% of self-play games): score each
    legal BID by P(bid literally true | own dice) via the existing bid_probability
    helper (6s wild), softmax-sample the bids at a per-game temperature drawn
    deterministically from _TL_TEMPERATURES by the game hash. Liar is played ONLY
    when no bid is safe (all P(true) < _TL_SAFE_P) — the never-bluff ladder-climber
    shape of the winner family. Deterministic given (own dice, current obs): the
    sampler RNG is seeded from game_hash (dice-keyed) ^ hash(current obs) — no wall clock, no
    global RNG, no cross-turn state. Returns None when unusable so the caller
    falls back to the epsilon-band persona; parse errors propagate to the caller's
    try/except (never-raise preserved there)."""
    gs = parse_game_state(messages)
    if not gs.actions:
        return None
    bids = [a for a in gs.actions if a.bid is not None]
    liar = gs.liar_action
    if not bids:
        return str(liar.action_id) if liar is not None else None
    best_p = max(a.prob for a in bids)
    if liar is not None and best_p < _TL_SAFE_P:
        # Never bluff: no safe rung left on the ladder -> call Liar.
        return str(liar.action_id)
    temperature = _TL_TEMPERATURES[(game_hash >> 1) % len(_TL_TEMPERATURES)]
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    rng = random.Random((game_hash ^ _stable_obs_hash(last_user)) & 0xFFFFFFFFFFFFFFFF)
    # audit P2: sample over SAFE rungs only when any qualify — the ladder is the
    # "never-bluff truth-maximizer" persona, but softmax over ALL bids kept ~1/3
    # weight on p~0.2 overbids at temp 0.5, diluting exactly the distribution
    # this persona exists to reproduce. (Liar-call rule above is unchanged.)
    safe = [a for a in bids if a.prob >= _TL_SAFE_P]
    pool = safe if safe else bids
    weights = _softmax_weights([a.prob for a in pool], temperature)
    chosen = rng.choices(pool, weights=weights, k=1)[0]
    return str(chosen.action_id)


def get_expert_action_b(messages: list[dict]) -> str:
    """SECOND TEACHER (persona B) for league self-play (SELFPLAY_DUAL_TEACHER).
    Per-GAME deterministic 50/50 persona split, keyed on OUR OWN DICE + total
    dice (constant within a hand, re-rolled across hands) — NOT the message
    text: the selfplay driver passes only the CURRENT observation, so an
    obs-text key would re-flip the persona every turn (LD_TRUTHFUL_LADDER
    gate, default on):

    - ~50% TRUTHFUL-LADDER: the winner-family Bayesian truth-maximizer
      (_truthful_ladder_action) — puts the winner's state distribution into our
      training data so the canonical labels stamp the punishes.
    - ~50% (and any fallback): the analytic equity epsilon-band policy with the
      MCCFR blueprint skipped. A genuinely DIFFERENT strong policy — equity +
      capped semi-bluff, mixed within an equity-equivalence band (never argmax, so
      it keeps the unreadable-mix property that beats the R1 0-400 exploit) — that
      disagrees with the Nash blueprint at most infosets, broadening SFT coverage.
      It is somewhat more exploitable than the blueprint vs a best-responder
      (self-check ~0.79 vs ~0.51), but our PvP opponents are weak LLMs (not
      best-responders), so the deviation is bounded in practice -> both-seats.

    Never raises (the ladder branch is wrapped; the band branch shares
    get_expert_action's fallbacks)."""
    if _TRUTHFUL_LADDER_ENABLED:
        try:
            # PER-HAND persona key: our own dice + total dice are constant within
            # a hand and re-rolled across hands — a stable per-game key needing no
            # cross-turn state. (Hashing the obs text re-flips per TURN: the obs
            # changes with every bid.) Same-dice hands share a persona; over the
            # dice distribution the split stays ~50/50.
            gs = parse_game_state(messages)
            game_key = f"TLKEY:{sorted(gs.our_dice)}:{gs.total_dice}"
            game_hash = _stable_obs_hash(game_key)
            if game_hash % 2 == 0:  # ~50% of games -> truthful-ladder persona
                action = _truthful_ladder_action(messages, game_hash)
                if action is not None:
                    return action
        except Exception:
            pass  # never-raise: fall through to the epsilon-band persona
    return get_expert_action(messages, _use_blueprint=False)


def generate_expert_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 30,
) -> "tuple[list[dict], float] | None":
    """Run one Liar's Dice game vs MCTS@225 using the analytic expert policy.

    Returns ``(messages, final_reward)`` on success, or None on env-server
    failure (caller will skip). ``final_reward`` is the env's terminal
    reward in [0, 1] where 0.5 = draw, > 0.5 = win, < 0.5 = loss (verified
    by probe + the ``(step_reward - 0.5) * 100`` shift used in
    liar_dice_env.py). Used by generate_trajectories.py for --wins-only /
    --sample-by-score filters, which threshold around 0.5 rather than 0.
    """
    if tool_calling_enabled():
        return run_toolcall_episode(
            game_name="liars_dice",
            game_id=game_id,
            env_endpoint=env_endpoint,
            opponent_payload=_OPPONENT_PAYLOAD,
            max_turn=max_turn,
            rules_prompt=get_game_rules_prompt("liars_dice"),
            expert_action_fn=get_expert_action,
            obs_transform=lambda raw: reformat_to_pvp(raw, "liars_dice"),
            policy=LiarDiceMemoryPolicy(),
        )

    reset_payload = {"task_id": game_id, "seed": game_id, **_OPPONENT_PAYLOAD}
    try:
        res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_REQUEST_TIMEOUT_SECONDS)
        res.raise_for_status()
        block = res.json()["result"]
        episode_id = block.get("episode_id", "")
        observation = block.get("observation", "")
    except Exception as exc:
        print(f"[env] Reset failed (game {game_id}): {exc}")
        return None

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": reformat_to_pvp(observation, "liars_dice")},
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
        messages.append({"role": "user", "content": reformat_to_pvp(observation, "liars_dice")})
    else:
        print(f"[env] max_turn={max_turn} reached (game {game_id})")

    return messages, final_reward
