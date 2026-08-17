"""Expert trajectory generator for Leduc Poker SFT.

Expert policy: hand-strength heuristic encoded from the strategy hints
already documented in leduc_poker_env.py (_HINT_PROMPT). The hints
describe near-optimal play for the 2-player 6-card variant:

  Round 1:
    K or Q hand -> Raise first; Call if opponent raises
    J hand     -> Fold against a Raise; Check if unchallenged
  Round 2 (public card revealed):
    Pair            -> always Raise; never Fold
    K (no pair)     -> Raise first; Call if opponent raises
    Q + public K    -> Raise first; Call if opponent raises
    Q + public J    -> Check; Fold if opponent raises
    J (no pair)     -> Check; Fold if opponent raises

Opponent during play: MCTS at the simulation count the validator uses
for env evaluation (see ENVIRONMENTS["leduc_poker"] in
G.O.D/validator/core/constants.py: mcts_max_simulations=50,
mcts_num_rollouts=1). Matching the validator's eval opponent means the
expert trajectories teach the model the playstyle it will be graded against.
"""

import math
import os
import random
import re
import zlib

import requests

from our_envs.leduc_poker_opponent_modeling import parse_game_state
from our_envs.leduc_cfr import get_strategy, infoset_key as _cfr_infoset_key, RANKS as _CFR_RANKS
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

# Authoritative opponent strength: match validator's eval config.
_OPPONENT_PAYLOAD = {
    "opponent": "mcts",
    "mcts_max_simulations": 50,
    "mcts_num_rollouts": 1,
}


# Use the validator's YAML system prompt instead of a hand-written one so
# the SFT prompt is byte-identical to what the PvP eval container
# produces. The previous prompt embedded a "# Strategy Tips" section
# the validator never emits, so the model saw a different string at
# eval time than at training time.
_SYSTEM_PROMPT = get_system_prompt("leduc_poker")


def _find_action_id(legal_actions: dict, *keywords: str) -> "int | None":
    """Return the action_id whose label matches any keyword (case-insensitive).

    Tries keywords in order; returns the first match. Keywords are matched
    against the action label as substrings (e.g. "raise" matches "Raise").
    """
    for kw in keywords:
        kw_l = kw.lower()
        for aid, label in legal_actions.items():
            if kw_l in label.lower():
                return aid
    return None


def _fallback_action_from_text(text: str) -> str:
    """NEVER-FOLD final fallback (P0 2026-07-04): parse the '<id> -> <label>'
    action menu from the raw observation and prefer call/check, then raise/bet,
    then the smallest id. The old min(legal) fallback resolved to FOLD in Leduc
    facing-bet states (fold = action id 0 = min legal) — the worst action under
    the PvP eval's sign scoring (a fold is a certain 0.0). Deterministic pure
    function of the text; NEVER returns "0" while any alternative exists
    (review fix: the keyword-miss and no-parse paths previously still emitted
    "0" = Fold, e.g. on bare-digit menu labels from a degraded observation).
    Never raises."""
    try:
        pairs = re.findall(r"^\s*(\d+)\s*->\s*(.+)$", text, re.MULTILINE)
        if pairs:
            la = {int(aid): label for aid, label in pairs}
            aid = _find_action_id(la, "call", "check")
            if aid is None:
                aid = _find_action_id(la, "raise", "bet")
            if aid is None:
                # Keyword miss: OpenSpiel leduc ids are 0=Fold/1=Call/2=Raise —
                # prefer 1, then 2, then the smallest NON-ZERO id; 0 only if it
                # is literally the only parsed action (cannot happen in leduc:
                # call is always legal).
                for pref in (1, 2):
                    if pref in la:
                        aid = pref
                        break
                if aid is None:
                    non_fold = [a for a in la if a != 0]
                    aid = min(non_fold) if non_fold else min(la)
            return str(aid)
        raw = re.findall(r"^\s*(\d+)\s*->", text, re.MULTILINE)
        if raw:
            ids = sorted({int(x) for x in raw})
            if 1 in ids:
                return "1"
            non_fold = [a for a in ids if a != 0]
            return str(non_fold[0]) if non_fold else str(ids[0])
        # Nothing parses at all: call/check (id 1) is ALWAYS legal in leduc,
        # while "0" is Fold — the one action the label policy must never emit.
        return "1"
    except Exception:
        return "1"


# AGGRESSION TIE-BREAK (P0 2026-07-04, label path only): after the WL-CFR solve
# many infosets are near-indifferent call-vs-raise. Under sign scoring raise is
# weakly dominant: it never worsens the showdown result, converts opponent folds
# into wins, and pushes the opponent into facing-raise states its MCTS-shaped
# data underrepresents. MUST match leduc_cfr._TIEBREAK_EPS (gen-start self-check).
# Single source of truth (audit P2): the gen-start self-check mirrors the label
# policy using leduc_cfr's own constant — a drifted local copy would keep the
# self-check green while real labels diverge from what it certified.
from our_envs.leduc_cfr import _TIEBREAK_EPS as _LP_TIEBREAK_EPS


def _aggression_tiebreak(items: list, legal_actions: dict) -> "int | None":
    """If the top-2 label probabilities in `items` [(action_id, prob), ...] are
    within _LP_TIEBREAK_EPS and a raise id is among them, return the raise id.
    Deterministic (no RNG). None = tie-break does not apply (caller argmaxes)."""
    if len(items) < 2:
        return None
    ordered = sorted(items, key=lambda x: -x[1])
    if ordered[0][1] - ordered[1][1] > _LP_TIEBREAK_EPS:
        return None
    raise_id = _find_action_id(legal_actions, "raise", "bet")
    if raise_id is not None and raise_id in (ordered[0][0], ordered[1][0]):
        return raise_id
    return None


def _expert_action_id_from_state(gs) -> "int | None":
    """Pure-Python policy: per-signal aligned action selection.

    Strategy mirrors the signed signals in the env's shaped reward (see
    leduc_poker_env.RewardCalculator.SIGNALS) so each decision matches
    the +EV branch:

      PAIR  (round 2):
        opp_raised -> CALL (call_pair_r2_raise = +0.5, stronger than
                            raising into known-aggression)
        else       -> RAISE (raise_pair_r2 = +0.3)
        NEVER fold a pair.

      K (rank 3, strongest non-pair):
        round 1 -> RAISE (raise_k_r1 = +0.25)
        round 2 -> RAISE (raise_k_r2 = +0.2)

      Q (rank 2):
        round 1 opp_raised -> CALL (call_kq_r1_raise = +0.2; folding Q
                                    to R1 aggression bleeds equity)
        round 2 pub=J      -> RAISE (raise_q_r2_pubj = +0.15; we
                                    dominate)
        round 2 pub=K opp_raised -> CALL (fold_q_pubk_raise = -0.5, so
                                    calling beats folding here)
        else               -> RAISE

      J (rank 1, weakest):
        round 1 opp_raised -> FOLD (fold_j_r1_raise = +0.3)
        round 2 opp_raised -> FOLD (fold_j_r2_raise = +0.2)
        round 2 pub=Q opp_raised -> FOLD (fold_q_pubj_raise = +0.2,
                                    transposing the +EV pattern to J)
        else (no opp aggression) -> CALL

    Falls back to legacy ladder if action_id mapping fails.
    """
    if not gs.legal_actions:
        return None

    fold = _find_action_id(gs.legal_actions, "fold")
    call = _find_action_id(gs.legal_actions, "call", "check")
    raise_ = _find_action_id(gs.legal_actions, "raise")
    default = call if call is not None else (
        raise_ if raise_ is not None else min(gs.legal_actions.keys())
    )

    opp_raised = gs.opp_raised_this_round
    strength = gs.hand_strength  # 4 = Pair, 3 = K, 2 = Q, 1 = J

    # PAIR — NEVER fold; call vs aggression in R2, raise otherwise.
    if gs.has_pair:
        if gs.round == 2 and opp_raised and call is not None:
            return call
        return raise_ if raise_ is not None else (call if call is not None else default)

    # K (rank 3) — raise both rounds.
    if strength == 3:
        if gs.round == 1 and raise_ is not None:
            return raise_
        if gs.round == 2 and raise_ is not None:
            return raise_
        return call if call is not None else default

    # Q (rank 2) — context-dependent. Don't fold to R1 aggression.
    if strength == 2:
        if gs.round == 1 and opp_raised:
            return call if call is not None else default
        if gs.round == 2 and gs.public_card_rank == 1 and raise_ is not None:
            return raise_
        if (
            gs.round == 2
            and gs.public_card_rank == 3
            and opp_raised
        ):
            # fold_q_pubk_raise = -0.5 -> call beats fold
            return call if call is not None else default
        return raise_ if raise_ is not None else (call if call is not None else default)

    # J (rank 1, weakest) — fold to aggression.
    if strength == 1:
        if gs.round == 1 and opp_raised and fold is not None:
            return fold
        if gs.round == 2 and opp_raised and fold is not None:
            return fold
        if (
            gs.round == 2
            and gs.public_card_rank == 2
            and opp_raised
            and fold is not None
        ):
            return fold
        return call if call is not None else default

    return default


# LP Step 2 (audit 2026-06-13): equity + pot-odds expert. Replaces the
# hand-strength decision tree with the EXACT Leduc equity engine
# (_compute_hand_equity) + true pot odds from the GameState (to_call vs pot),
# plus a small calibrated semi-bluff fraction — the heuristic never bluffs and
# is therefore deterministically exploitable. The equity rule reproduces the
# heuristic's sound calls (pairs/K value-bet, J folds to aggression) but adds
# precise pot-odds calling and unpredictability. Toggle LP_EQUITY=0 to fall
# back to the hand-strength heuristic for A/B.
_LP_EQUITY_EXPERT = (os.environ.get("LP_EQUITY", "1").strip() or "1").strip() not in ("0", "false", "no", "")

# LP CFR+ Nash expert (default ON). Solves Leduc in-process at gen-start and
# SAMPLES from the near-Nash mixed strategy. Replaces the exploitable equity
# heuristic (verified research 2026-06-22: equity/pot-odds is exploitable
# because the Leduc Nash is MIXED; MCTS is unsound for imperfect-info, so a Nash
# strategy beats it). NO file/bundle: the strategy is computed in RAM by
# leduc_cfr.get_strategy (the solver ALGORITHM ships, not a table). LP_CFR=0
# falls back to the equity expert for A/B.
_LP_CFR_EXPERT = (os.environ.get("LP_CFR", "1").strip() or "1").strip() not in ("0", "false", "no", "")
_LP_RAISE_EQUITY = 0.62   # value-raise hands that dominate (pairs, K, strong Q)
_LP_BET_EQUITY = 0.55     # value-bet when unraised (above coin-flip)
_LP_BLUFF_PCT = 0.15      # fraction of weak hands we semi-bluff
_LP_BLUFF_EQ_MAX = 0.40   # only bluff hands below this equity
_LP_MAX_RAISES = 2        # Leduc caps raises per round
# R1 fix (2026-06-20): the value side was 100% deterministic (equity>=0.62 ALWAYS
# raised) -> a strong peer reads "raise = strong" and exploits us (the 0-400 LP
# loss). Make value betting STOCHASTIC: raise frequency ramps with equity but is
# capped below 1, so strong hands sometimes slow-play -> balanced, unreadable.
_LP_VALUE_RAISE_CAP = 0.85  # max value-raise frequency (slow-play the rest)
_LP_RAMP_WIDTH = 0.10       # logistic width of the equity->P(raise) ramp


def _ramp(equity: float, center: float, width: float) -> float:
    """Logistic 0->1 ramp of P(aggress) around `center` over scale `width`.
    Replaces a hard equity threshold with a smooth probability so the value
    decision is MIXED (sampled) rather than a readable step function."""
    return 1.0 / (1.0 + math.exp(-(equity - center) / max(width, 1e-6)))


def _equity_action_id_from_state(gs) -> "int | None":
    """Equity + pot-odds policy, MIXED on the value side (R1 fix).

    Facing a bet: if equity beats the pot odds, raise with a probability that
    ramps with equity but is capped below 1 (strong hands sometimes slow-play),
    else call; below the price, CALL (semi-bluff-raising a small fraction of weak
    hands) — P0 2026-07-04: PvP eval scores the SIGN of the return, so folding is
    a certain 0.0 and weakly dominated by calling. Unraised: value-bet with the
    same ramped probability, occasionally bluff-bet, else check. The stochastic
    value decision balances our betting range so a strong peer cannot read
    "raise = strong" — the determinism that lost R1. Uses the exact equity engine
    so all board textures are priced right.
    """
    if not gs.legal_actions:
        return None
    from our_envs.leduc_poker_opponent_modeling import _compute_hand_equity

    call = _find_action_id(gs.legal_actions, "call", "check")
    raise_ = _find_action_id(gs.legal_actions, "raise")
    default = call if call is not None else (
        raise_ if raise_ is not None else min(gs.legal_actions.keys())
    )

    equity = _compute_hand_equity(gs.private_card_rank, gs.public_card_rank)
    to_call = max(0, gs.opp_invested - gs.our_invested)
    can_reraise = raise_ is not None and gs.raises_this_round < _LP_MAX_RAISES

    if to_call > 0:  # facing a bet/raise
        pot_odds = to_call / (gs.pot + to_call) if (gs.pot + to_call) > 0 else 0.5
        if equity >= pot_odds:
            # continue: mix raise vs call (ramped, capped -> sometimes slow-play)
            p_raise = (
                _LP_VALUE_RAISE_CAP * _ramp(equity, _LP_RAISE_EQUITY, _LP_RAMP_WIDTH)
                if can_reraise else 0.0
            )
            if random.random() < p_raise:
                return raise_
            return call if call is not None else default
        # below the price: CALL (fold = certain loss under sign scoring),
        # keeping the occasional semi-bluff raise as-is (P0 2026-07-04)
        if equity < _LP_BLUFF_EQ_MAX and can_reraise and random.random() < _LP_BLUFF_PCT:
            return raise_
        return call if call is not None else default
    # unraised: mix value-bet vs check (+ occasional bluff bet)
    p_bet = (
        _LP_VALUE_RAISE_CAP * _ramp(equity, _LP_BET_EQUITY, _LP_RAMP_WIDTH)
        if raise_ is not None else 0.0
    )
    if random.random() < p_bet:
        return raise_  # value bet (stochastic)
    if equity < _LP_BLUFF_EQ_MAX and raise_ is not None and random.random() < _LP_BLUFF_PCT:
        return raise_  # bluff bet
    return call if call is not None else default  # check


# --------------------------------------------------------------------------- #
# LP CFR+ Nash expert: map our parsed GameState -> the solver's infoset key,
# look up the near-Nash mixed strategy, and SAMPLE (mixed, NOT argmax — argmax
# kills the bluffing/balancing that makes Nash unexploitable). The strategy is
# solved in RAM once per process; nothing is written to disk.
# --------------------------------------------------------------------------- #
def _betting_to_tokens(betting: list) -> str:
    """Chronological betting list (['Raise','Call',...]) -> CFR tokens
    ('r'=bet/raise, 'c'=check/call, 'f'=fold)."""
    out = []
    for a in betting:
        al = str(a).lower()
        if "fold" in al:
            out.append("f")
        elif "raise" in al or "bet" in al:
            out.append("r")
        elif "call" in al or "check" in al:
            out.append("c")
    return "".join(out)


def _canon_infoset_key(gs) -> "str | None":
    """Build the solver's infoset key from a GameState (None if the private rank
    is unknown). board is set only once revealed (round 2)."""
    pr = gs.private_card_rank          # 1=J, 2=Q, 3=K (0 = unknown)
    if pr not in (1, 2, 3):
        return None
    card = _CFR_RANKS[pr - 1]
    board = None
    if gs.round == 2 and gs.public_card_rank in (1, 2, 3):
        board = _CFR_RANKS[gs.public_card_rank - 1]
    return _cfr_infoset_key(card, board,
                            _betting_to_tokens(gs.r1_betting),
                            _betting_to_tokens(gs.r2_betting))


def _cfr_action_id_from_state(gs) -> "int | None":
    """Sample an action from the near-Nash CFR strategy for this infoset.
    Returns None (caller falls back to the equity expert) on any miss — parse
    gap, infoset absent, or no mappable legal action. Never raises."""
    if not gs.legal_actions:
        return None
    try:
        key = _canon_infoset_key(gs)
        if key is None:
            return None
        dist = get_strategy().get(key)   # lazy solve, cached per process
        if not dist:
            return None
        # Map CFR tokens to this state's legal action ids; keep only mappable.
        # NEVER-FOLD OVERRIDE (P0 2026-07-04): PvP eval scores the SIGN of the
        # return, so a fold is a certain 0.0 (weakly dominated by call/check).
        # Drop 'f' whenever a call/check id exists — required even with the
        # LP_WL solve because the ~59 UNREACHED infosets average to uniform
        # (fold mass 0.5).
        can_call = _find_action_id(gs.legal_actions, "call", "check") is not None
        items = []
        for tok, prob in dist.items():
            if prob <= 0:
                continue
            if tok == "f":
                if can_call:
                    continue               # never-fold: skip the fold token
                aid = _find_action_id(gs.legal_actions, "fold")
            elif tok == "r":
                aid = _find_action_id(gs.legal_actions, "raise", "bet")
            else:  # 'c'
                aid = _find_action_id(gs.legal_actions, "call", "check")
            if aid is not None:
                items.append((aid, prob))
        if not items:
            return None
        # Weighted sample — MIXED play preserves the unexploitable balancing.
        total = sum(p for _, p in items)
        r = random.random() * total
        acc = 0.0
        for aid, p in items:
            acc += p
            if r <= acc:
                return aid
        return items[-1][0]
    except Exception:
        return None


def get_expert_action(messages: list[dict], _use_cfr: bool = None) -> str:
    """Pick action via CFR+ Nash (default), equity+pot-odds (LP_CFR=0), or
    hand-strength heuristic (LP_CFR=0 + LP_EQUITY=0). Falls back to smallest
    legal ID on parse/selection failure. _use_cfr overrides the LP_CFR gate
    per-call (persona B passes False to force the equity policy)."""
    use_cfr = _LP_CFR_EXPERT if _use_cfr is None else _use_cfr
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    try:
        gs = parse_game_state(last_user)
    except Exception:
        gs = None

    if gs is None:
        return _fallback_action_from_text(last_user)   # never-fold: call/check first

    chosen = _cfr_action_id_from_state(gs) if use_cfr else None
    if chosen is None:
        chosen = (
            _equity_action_id_from_state(gs) if _LP_EQUITY_EXPERT
            else _expert_action_id_from_state(gs)
        )
    if chosen is None:
        # Last-resort fallback: call/check id first, never fold (was min legal)
        return _fallback_action_from_text(last_user)
    return str(chosen)


def _cfr_label_id_from_state(gs, obs: str) -> "int | None":
    """Deterministic CANONICAL CFR move: the ARGMAX of the near-Nash CFR strategy for
    this infoset with an obs-hashed tie-break, instead of the weighted sample in
    _cfr_action_id_from_state. Returns None on any miss (caller -> min legal)."""
    if not gs.legal_actions:
        return None
    try:
        key = _canon_infoset_key(gs)
        if key is None:
            return None
        dist = get_strategy().get(key)
        if not dist:
            return None
        # NEVER-FOLD OVERRIDE — see _cfr_action_id_from_state (same rationale;
        # the label path must never emit a fold either).
        can_call = _find_action_id(gs.legal_actions, "call", "check") is not None
        items = []
        for tok, prob in dist.items():
            if prob <= 0:
                continue
            if tok == "f":
                if can_call:
                    continue               # never-fold: skip the fold token
                aid = _find_action_id(gs.legal_actions, "fold")
            elif tok == "r":
                aid = _find_action_id(gs.legal_actions, "raise", "bet")
            else:  # 'c'
                aid = _find_action_id(gs.legal_actions, "call", "check")
            if aid is not None:
                items.append((aid, prob))
        if not items:
            return None
        # AGGRESSION TIE-BREAK (deterministic) before the canonical argmax.
        tb = _aggression_tiebreak(items, gs.legal_actions)
        if tb is not None:
            return tb
        return canonical_argmax([a for a, _ in items], [p for _, p in items], obs)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# R1 LABEL PURIFICATION (LP_PURIFY, default OFF) — Ganzfried-Sandholm.
#
# A near-uniform mixed CFR strategy that is SAMPLED per state produces
# pattern-inconsistent labels ("stochastic-label disease") that hurt a distilled
# student at greedy eval. Purify = threshold the CFR+ average strategy to its
# SUPPORT (drop actions below LP_PURIFY_EPS of the max prob, discarding the noise
# tail), then pick DETERMINISTICALLY by a SEMANTIC win-utility ordering (never
# fold; raise weakly dominates call under the PvP SIGN scoring) instead of the
# obs-hash tie-break — so similar infosets get the SAME label. Flag OFF =>
# get_label_action is byte-identical to the current canonical-argmax path below.
# --------------------------------------------------------------------------- #
def _lp_flag_on(name: str) -> bool:
    """OFF for unset / 0 / false / no / off / empty; ON otherwise. Read at CALL
    time so the flag is reversible by simply unsetting it (no re-import)."""
    return os.environ.get(name, "").strip().lower() not in ("0", "false", "no", "off", "")


def _lp_flag_float(name: str, default: float) -> float:
    """Parse a float env var (empty/unset/garbage -> default). Never raises."""
    try:
        return float(os.environ.get(name, "").strip() or default)
    except Exception:
        return default


def _lp_action_value(aid: int, legal_actions: dict) -> int:
    """Semantic win-utility rank for the purified tie-break (higher = better under
    PvP SIGN scoring): raise/bet (2) weakly dominates call/check (1); both dominate
    fold (0) — a fold is a certain 0.0 loss the eval scores as a hand loss even
    when it is chip-EV-neutral. Deterministic; NOT an obs-hash, NOT a smallest-id."""
    label = str(legal_actions.get(aid, "")).lower()
    if "raise" in label or "bet" in label:
        return 2
    if "call" in label or "check" in label:
        return 1
    return 0


def _purify_pick(ids: list, weights: dict, value: dict, eps: float):
    """Ganzfried-Sandholm purify of a mixed strategy. From candidate action `ids`
    with mixed-strategy `weights` (id->prob), drop those below `eps` of the max
    weight (keep the Nash SUPPORT, discard the noise tail), then pick the survivor
    with the highest `value` (id->win-utility), the mixed weight as a deterministic
    secondary key. Returns (chosen_id, qgap) where qgap = the prob gap between the
    top-2 candidate weights (>=0.0). Pure + deterministic (no RNG, no obs-hash)."""
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


def _lp_cfr_items(gs) -> "list | None":
    """Map the CFR+ strategy tokens for this infoset to the state's legal action
    ids with the NEVER-FOLD override (drop the fold token whenever a call/check id
    exists), returning [(action_id, prob), ...] with prob > 0, or None on any miss.
    Mirrors _cfr_label_id_from_state's item construction exactly so the purified
    label maps identically to the canonical one."""
    if not gs.legal_actions:
        return None
    key = _canon_infoset_key(gs)
    if key is None:
        return None
    dist = get_strategy().get(key)
    if not dist:
        return None
    can_call = _find_action_id(gs.legal_actions, "call", "check") is not None
    items = []
    for tok, prob in dist.items():
        if prob <= 0:
            continue
        if tok == "f":
            if can_call:
                continue                       # never-fold: skip the fold token
            aid = _find_action_id(gs.legal_actions, "fold")
        elif tok == "r":
            aid = _find_action_id(gs.legal_actions, "raise", "bet")
        else:  # 'c'
            aid = _find_action_id(gs.legal_actions, "call", "check")
        if aid is not None:
            items.append((aid, prob))
    return items or None


def _lp_purified_label(gs, obs: str) -> str:
    """LP_PURIFY label: threshold the CFR+ average strategy to its support, then
    pick DETERMINISTICALLY by win-utility (never fold; raise > call). Sets
    qgap/forced metadata right before returning. Never raises (falls back to the
    never-fold call/check id on any miss)."""
    forced = bool(gs is not None and gs.legal_actions and len(gs.legal_actions) == 1)
    try:
        items = _lp_cfr_items(gs) if gs is not None else None
        if not items:
            set_meta(qgap=None, forced=forced)
            return _fallback_action_from_text(obs)   # never-fold: call/check first
        ids = [aid for aid, _ in items]
        weights = {aid: p for aid, p in items}
        value = {aid: _lp_action_value(aid, gs.legal_actions) for aid in ids}
        chosen, qgap = _purify_pick(ids, weights, value, _lp_flag_float("LP_PURIFY_EPS", 0.10))
        if chosen is None:
            set_meta(qgap=None, forced=forced)
            return _fallback_action_from_text(obs)
        set_meta(qgap=qgap, forced=forced)
        return str(chosen)
    except Exception:
        set_meta(qgap=None, forced=forced)
        return _fallback_action_from_text(obs)


def get_label_action(messages: list[dict]) -> str:
    """Tier-1 CANONICAL label (deterministic, pure function of the observation): the
    ARGMAX of the SAME near-Nash CFR strategy _cfr_action_id_from_state SAMPLES, ties
    broken by an obs-hash — a single-valued (state->action_id) Nash-support target,
    not a per-game weighted draw and not the equity persona. Label policy adds the
    NEVER-FOLD override + deterministic AGGRESSION TIE-BREAK (P0 2026-07-04: the
    PvP eval scores the sign of the return, so a fold label is a certain loss).
    Falls back to the call/check id (never fold, was min legal) on a CFR miss
    (rare: Leduc is fully solved). Never raises.

    LP_PURIFY (default OFF): route to the Ganzfried-Sandholm purified pick instead.
    OFF => byte-identical to the canonical path below."""
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    try:
        gs = parse_game_state(last_user)
    except Exception:
        gs = None
    if _lp_flag_on("LP_PURIFY"):
        return _lp_purified_label(gs, last_user)
    chosen = _cfr_label_id_from_state(gs, last_user) if (gs is not None and _LP_CFR_EXPERT) else None
    if chosen is None:
        return _fallback_action_from_text(last_user)   # never-fold: call/check first
    return str(chosen)


def get_expert_action_b(messages: list[dict]) -> str:
    """SECOND TEACHER (persona B) for league self-play (SELFPLAY_DUAL_TEACHER):
    the equity + pot-odds / mixed-bluff policy promoted to PRIMARY (the CFR+ Nash
    solver is skipped). A genuinely DIFFERENT strong policy that mixes value-raises
    and capped semi-bluffs and disagrees with the Nash blueprint at most infosets,
    broadening SFT coverage. NOTE: this is the MOST MARGINAL of the six personas —
    an equity policy is more exploitable than Nash vs a best-responder, but our PvP
    opponents are weak LLMs (not best-responders), so it stays bounded -> both-seats
    (A/B-watch this one most closely; the heavier MCRNR exploit-tilt teacher is a
    wins-only follow-up). Never raises (shares get_expert_action's fallbacks)."""
    return get_expert_action(messages, _use_cfr=False)


def generate_expert_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 20,
) -> "tuple[list[dict], float] | None":
    """Run one Leduc Poker game using the hand-strength heuristic expert.

    Mirrors generate_random_episode's structure (tool_call + bare-id paths,
    returns (messages, terminal_score)) but selects actions via the sound
    +EV-aligned heuristic (get_expert_action / _expert_action_id_from_state)
    instead of uniform random. Audit 2026-06-13 found leduc_poker was wired to
    the RANDOM generator: its 78.8% gen win-rate was score-filter survivorship,
    so the kept "wins" demonstrated lucky random actions rather than skill. The
    heuristic produces wins from a real strategy; the score-power filter is
    still applied downstream. generate_random_episode is kept for A/B + revert.
    """
    if tool_calling_enabled():
        return run_toolcall_episode(
            game_name="leduc_poker",
            game_id=game_id,
            env_endpoint=env_endpoint,
            opponent_payload=_OPPONENT_PAYLOAD,
            max_turn=max_turn,
            rules_prompt=get_game_rules_prompt("leduc_poker"),
            expert_action_fn=get_expert_action,
            obs_transform=lambda raw: reformat_to_pvp(raw, "leduc_poker"),
            policy=LeducMemoryPolicy(),
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
        {"role": "user", "content": reformat_to_pvp(observation, "leduc_poker")},
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
        messages.append({"role": "user", "content": reformat_to_pvp(observation, "leduc_poker")})
    else:
        print(f"[env] max_turn={max_turn} reached (game {game_id})")

    return messages, final_reward


# --- Winner-recipe alternative: random + score-bias ----------------------
# Leduc Poker's tiny state space lets MCTS@50 play near-optimally, so even a
# strong heuristic player rarely wins head-to-head. Random play paired with
# score-based sampling (win_strength = clamp((score - 0.5) * 2, 0, 1) ** power
# with power=3.0) biases the training set toward winning trajectories without
# needing a strong handcrafted policy. Threshold pivots on 0.5 because the
# env-server returns terminal reward in [0, 1] where 0.5 = draw.

def _random_action(observation: str) -> str:
    """Uniformly random legal action ID parsed from the observation text."""
    raw = re.findall(r"^\s*(\d+)\s*->", observation, re.MULTILINE)
    if not raw:
        return "1"
    return random.choice(raw)


def _random_action_msgs(messages) -> str:
    """Runner adapter: pull the last user observation and pick a random legal id."""
    obs = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return _random_action(obs)


class LeducMemoryPolicy(MemoryPolicy):
    """working = my card + pot + my move; long_term = the opponent's observed
    betting actions (raise/check/fold tendency). Regex-only, never-raises."""

    def __init__(self):
        self._opp_actions: list[str] = []

    def observe(self, obs: str) -> None:
        for label in ("Round 1 actions", "Round 2 actions"):
            m = re.search(rf"{label}:\s*(.+)", obs)
            if m:
                acts = m.group(1).strip()
                if acts and acts not in self._opp_actions:
                    self._opp_actions.append(acts)

    def turn_writes(self, obs, action_id, state, turn_idx):
        # Winner-recipe rarity gate (audit P1): memory notes fire on ~25% of
        # turns (stable crc32 - not every turn: dense per-turn writes trained
        # the memory-first habit that drowned game_action, 2026-06-22).
        if zlib.crc32(f"{action_id}:{turn_idx}:{len(obs)}".encode()) % 4 != 0:
            return []
        cm = re.search(r"Your card:\s*(\S+)", obs)
        pm = re.search(r"Pot:\s*(\d+)", obs)
        card = cm.group(1) if cm else "?"
        pot = pm.group(1) if pm else "?"
        return [(WORKING, "rewrite", 1, f"my card {card}; pot {pot}; play action {action_id}")]

    def reflect_writes(self, outcome, state):
        summary = "; ".join(self._opp_actions[-3:]) if self._opp_actions else "few actions seen"
        return [(LONG_TERM, "append", 1, f"opp betting: {summary}")]


def generate_random_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 20,
) -> "tuple[list[dict], float] | None":
    """Play one LP game with uniform random actions, return (messages, score).

    ``score`` is the env's terminal reward in [0, 1] where 0.5 = draw,
    > 0.5 = win, < 0.5 = loss (verified by probe + the
    ``(step_reward - 0.5) * 100`` shift used in leduc_poker_env.py).
    Callers downstream-filter via --wins-only or sample proportionally to
    this score to bias the dataset toward winning play. Returns None on
    env-server failure (caller skips the episode).
    """
    if tool_calling_enabled():
        return run_toolcall_episode(
            game_name="leduc_poker",
            game_id=game_id,
            env_endpoint=env_endpoint,
            opponent_payload=_OPPONENT_PAYLOAD,
            max_turn=max_turn,
            rules_prompt=get_game_rules_prompt("leduc_poker"),
            expert_action_fn=_random_action_msgs,
            obs_transform=lambda raw: reformat_to_pvp(raw, "leduc_poker"),
            policy=LeducMemoryPolicy(),
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
        {"role": "user", "content": reformat_to_pvp(observation, "leduc_poker")},
    ]

    final_reward = 0.0
    for _ in range(max_turn):
        action = _random_action(observation)
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
        messages.append({"role": "user", "content": reformat_to_pvp(observation, "leduc_poker")})
    else:
        print(f"[env] max_turn={max_turn} reached (game {game_id})")

    return messages, final_reward
