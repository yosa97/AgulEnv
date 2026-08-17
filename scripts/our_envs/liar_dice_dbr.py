"""Confidence-gated opponent exploitation layered on the MCCFR Nash blueprint.

WHY THIS SHAPE (and not the reigning expert's shape). The reigning Liar's Dice
expert scores every action against a Bayesian posterior over the opponent's dice and
plays near-argmax of it. That is structurally a best response to an ESTIMATED model,
which the published record puts at the maximally-exploitable end of the spectrum:

  * A best response to a model crushes THAT model and loses to everything else. In
    full Texas Hold'em (Johanson 2007, NIPS07-rnash Table 1) each frequentist best
    response beat its own target (e.g. +2170 mb/h) but posted a NEGATIVE row average
    (-142 mb/h) and lost to every opponent it was not fitted to — including a
    near-identical variant of its own target. "Best response is, in practice, a
    brittle computation, and can perform poorly when the model is wrong."
  * The equilibrium is the opposite trade: it never loses (CFR5 lost to none of seven
    opponents, +56 mb/h average) but under-exploits badly (+93 mb/h against an
    opponent worth 2170).
  * Best response and Nash are two ends of ONE axis, indexed by the set of priors you
    stay robust to (Johanson & Bowling, AISTATS 2009, Theorem 1): a singleton prior
    yields exactly a best response, all priors yield exactly Nash. Data Biased
    Response sits in between by making model-trust a function of how much you have
    actually OBSERVED: Pconf(I) = Pmax * n_I / (s + n_I), and crucially Pconf = 0 when
    n_I = 0, so an unobserved spot falls back to equilibrium instead of to a guess.
  * That gate matters most exactly where we are: a Liar's Dice hand is a handful of
    bids, so any posterior is fit on a few observations. Restricted Nash Response
    built on too-few observations gets simultaneously MORE exploitable and LESS
    exploitive — the "exploiter" can go net-negative. The reigning expert trusts its
    posterior fully from the first bid, which is that failure mode.

So: the blueprint stays the floor, and exploitation is metered by evidence. With no
observed opponent bid we play the Nash mix (never worse than the game value); as the
opponent's bids accumulate we blend in the posterior-scored response in proportion to
Pconf. We never reach a pure best response (Pmax < 1), which is the knee of the
published exploitation-vs-exploitability curve rather than its brittle endpoint.

CAVEAT, stated honestly: every number above is heads-up limit Texas Hold'em, not
Liar's Dice. The mechanism (model-trust proportional to evidence) transfers by
argument; the magnitudes do not transfer, and we have not measured them here.

All state is derived from the message history passed in, so a turn's label depends
only on what the row itself shows — no cross-episode module state to leak or reset.
"""

import math
import os
import re

from our_envs.liar_dice_cfr import (
    NUM_FACES,
    _hand_probability,
    _hands,
    _matching_count,
    decode_bid,
    encode_bid,
    legal_action_ids_for,
    liar_id,
)

# Pconf(n) = PMAX * n / (S + n). S is the AISTATS-2009 "s-Curve" strength; PMAX caps
# how far we ever move off the blueprint. PMAX<1 keeps us off the pure-best-response
# endpoint. Defaults are deliberately conservative because a hand yields few bids.
_DBR_PMAX = float(os.environ.get("LD_DBR_PMAX", "0.5") or "0.5")
_DBR_S = float(os.environ.get("LD_DBR_S", "1") or "1")

# Candidate opponent rationality temperatures, marginalized with a uniform prior: we
# do not know how sharp the opponent is, and committing to one value is the "singleton
# prior" that Theorem 1 equates with a plain best response.
_OPP_TEMPS = (0.05, 0.15, 0.4, 1.0)

_RE_BID = re.compile(r'Current bid:\s*"?(\d+)\s*-\s*(\d+)"?')
_RE_DICE = re.compile(r"Your dice:\s*\[?([0-9,\s]+?)\]?\s*(?:\(|$)", re.MULTILINE)
_RE_TOTAL = re.compile(r"Total dice in game:\s*(\d+)")


def dbr_confidence(n_obs: int) -> float:
    """Model-trust in [0, PMAX] from the observation count. n_obs=0 -> 0.0 (pure Nash)."""
    if n_obs <= 0:
        return 0.0
    return _DBR_PMAX * (n_obs / (_DBR_S + n_obs))


def observed_opponent_bids(messages: "list[dict]") -> list:
    """Reconstruct [(context_bid_id, opponent_bid_id), ...] from the conversation.

    The eval's state text shows only the single standing bid, but every past turn is
    still in `messages`, so the opponent's whole bid sequence is recoverable from the
    row itself. Deriving it here (instead of accumulating module state across calls)
    means a label never depends on a previous hand that happened to share a worker.
    """
    bids: list = []
    for m in messages:
        if m.get("role") != "user":
            continue
        found = _RE_BID.search(m.get("content") or "")
        if not found:
            continue
        bid_id = encode_bid(int(found.group(1)), int(found.group(2)))
        if not bids or bids[-1] != bid_id:
            bids.append(bid_id)
    # Each standing bid we faced is one opponent action; its context is the bid that
    # stood before it (None for their opening bid).
    return [(bids[i - 1] if i else None, b) for i, b in enumerate(bids)]


def _prob_bid_true_given_hand(bid_id: int, opp_hand: tuple, our_counts: tuple) -> float:
    """1.0/0.0 — with BOTH hands known a bid is simply true or false."""
    quantity, face = decode_bid(bid_id)
    ours = our_counts[NUM_FACES - 1] if face == NUM_FACES else our_counts[face - 1] + our_counts[NUM_FACES - 1]
    return 1.0 if ours + _matching_count(opp_hand, face) >= quantity else 0.0


def _softmax(scores: "list[float]", temperature: float) -> "list[float]":
    if not scores:
        return []
    t = max(temperature, 1e-6)
    top = max(scores)
    exps = [math.exp((s - top) / t) for s in scores]
    total = sum(exps)
    return [e / total for e in exps] if total > 0 else [1.0 / len(scores)] * len(scores)


def _our_counts(our_dice: "list[int]") -> tuple:
    counts = [0] * NUM_FACES
    for d in our_dice:
        if 1 <= d <= NUM_FACES:
            counts[d - 1] += 1
    return tuple(counts)


def posterior_opponent_hands(observations: list, n_opp_dice: int, our_counts: tuple, total_dice: int) -> list:
    """P(opponent hand | their bid history), as [(hand, probability), ...].

    Likelihood of an observed bid under a candidate hand = how a temperature-softmax
    player holding that hand would have weighted it, marginalized over _OPP_TEMPS.
    With no observations this returns the plain chance prior, which makes the caller's
    blend collapse to the blueprint.
    """
    hands = [(h, _hand_probability(h)) for h in _hands(n_opp_dice)]
    if not observations:
        return hands

    lid = liar_id(total_dice)
    posterior = []
    for hand, prior in hands:
        per_obs = []
        for context_id, chosen_id in observations:
            legal = legal_action_ids_for(context_id, total_dice)
            if chosen_id not in legal:
                continue
            scores = [
                (1.0 - _prob_bid_true_given_hand(context_id, hand, our_counts))
                if a == lid and context_id is not None
                else _prob_bid_true_given_hand(a, hand, our_counts)
                for a in legal
            ]
            per_obs.append((scores, legal.index(chosen_id)))
        if not per_obs:
            posterior.append((hand, prior))
            continue
        mixed = 0.0
        for temp in _OPP_TEMPS:
            likelihood = 1.0
            for scores, idx in per_obs:
                likelihood *= _softmax(scores, temp)[idx]
            mixed += likelihood
        posterior.append((hand, prior * (mixed / len(_OPP_TEMPS))))

    total = sum(w for _, w in posterior)
    return [(h, w / total) for h, w in posterior] if total > 0 else hands


def _prob_bid_true_under_posterior(bid_id: int, our_counts: tuple, posterior: list) -> float:
    quantity, face = decode_bid(bid_id)
    ours = our_counts[NUM_FACES - 1] if face == NUM_FACES else our_counts[face - 1] + our_counts[NUM_FACES - 1]
    needed = quantity - ours
    if needed <= 0:
        return 1.0
    return sum(p for hand, p in posterior if _matching_count(hand, face) >= needed)


def exploit_distribution(legal_ids: list, our_counts: tuple, posterior: list,
                         total_dice: int, current_bid_id: "int | None") -> dict:
    """The posterior-scored response, as a distribution over legal ids.

    Sharp (this IS the exploitative pole); the caller decides how much of it to use.
    """
    if not legal_ids:
        return {}
    lid = liar_id(total_dice)
    scores = []
    for action_id in legal_ids:
        if action_id == lid and current_bid_id is not None:
            scores.append(1.0 - _prob_bid_true_under_posterior(current_bid_id, our_counts, posterior))
        else:
            scores.append(_prob_bid_true_under_posterior(action_id, our_counts, posterior))
    weights = _softmax(scores, 0.02)
    return {a: w for a, w in zip(legal_ids, weights)}


def blend(nash: dict, exploit: dict, n_obs: int) -> dict:
    """(1 - Pconf) * blueprint + Pconf * exploit, over the blueprint's legal support.

    n_obs = 0 -> Pconf = 0 -> exactly the blueprint. The blueprint is never dropped,
    only tilted, which is what keeps the worst case anchored to the equilibrium.
    """
    conf = dbr_confidence(n_obs)
    if conf <= 0.0 or not exploit:
        return nash
    mixed = {a: (1.0 - conf) * p + conf * exploit.get(a, 0.0) for a, p in nash.items()}
    total = sum(mixed.values())
    return {a: p / total for a, p in mixed.items()} if total > 0 else nash


def parse_context(messages: "list[dict]") -> "tuple[list, int, int | None] | None":
    """(our_dice, total_dice, current_bid_id) from the latest user turn, or None."""
    last_user = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
    dice_match = _RE_DICE.search(last_user)
    if not dice_match:
        return None
    our_dice = [int(x) for x in re.findall(r"\d+", dice_match.group(1))]
    if not our_dice:
        return None
    total_match = _RE_TOTAL.search(last_user)
    total_dice = int(total_match.group(1)) if total_match else len(our_dice) * 2
    bid_match = _RE_BID.search(last_user)
    current_bid_id = encode_bid(int(bid_match.group(1)), int(bid_match.group(2))) if bid_match else None
    return our_dice, total_dice, current_bid_id


def dbr_distribution(messages: "list[dict]", nash: dict) -> dict:
    """Blueprint tilted toward the posterior response, gated by observation count.

    Never raises: any parse/shape problem returns the blueprint untouched, so the
    exploitation layer can only ever add to a working Nash floor.
    """
    try:
        if not nash:
            return nash
        ctx = parse_context(messages)
        if ctx is None:
            return nash
        our_dice, total_dice, current_bid_id = ctx
        n_opp_dice = total_dice - len(our_dice)
        if n_opp_dice <= 0:
            return nash
        observations = observed_opponent_bids(messages)
        if not observations:
            return nash  # Pconf(0) = 0; skip the posterior work entirely.
        our_counts = _our_counts(our_dice)
        posterior = posterior_opponent_hands(observations, n_opp_dice, our_counts, total_dice)
        exploit = exploit_distribution(list(nash), our_counts, posterior, total_dice, current_bid_id)
        return blend(nash, exploit, len(observations))
    except Exception:
        return nash


def sharpen(dist: dict, temperature: float) -> dict:
    """Lower-entropy copy of `dist` (p^(1/T) renormalised); T>=1 returns it unchanged.

    The student has to IMITATE this label. A high-entropy target teaches the same state
    a different action each time, and the student reproduces neither the mix nor the
    mode. Sharpening keeps the blueprint's ordering and its genuine indifference (ties
    stay ties) while making the dominant action dominant enough to actually learn.
    """
    if not dist or temperature >= 1.0:
        return dist
    t = max(temperature, 1e-3)
    powered = {a: (p ** (1.0 / t)) for a, p in dist.items() if p > 0}
    total = sum(powered.values())
    return {a: p / total for a, p in powered.items()} if total > 0 else dist
