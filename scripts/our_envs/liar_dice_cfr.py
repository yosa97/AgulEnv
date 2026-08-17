"""In-process memoryless external-sampling MCCFR blueprint for 2-player Liar's
Dice (Dudo) — pure Python, no deps.

WHY: our liars_dice teacher was an analytic P(true)/P(false) + epsilon-band
policy (anchored to literal-truth probability — a deterministic function of state
a memory-equipped PvP peer can model). A memoryless MCCFR Nash blueprint is
instead measured at ~0.51 best-response exploitability vs ~0.79 for the EV policy
and ~1.56 for uniform — materially less exploitable against a strong head-to-head
opponent. This implements that MCCFR core.

HOW (no-bundle, offline): the agent only ever observes (own dice, current bid),
NOT the full bid history, so we solve the game under an (own-dice, current-bid)
INFORMATION ABSTRACTION via external-sampling MCCFR (one deal + one sampled
trajectory per iteration -> scales to any dice count). The average strategy is a
near-equilibrium of the abstracted game. Held in a process-memoized dict (no disk
artifact ships in the repo); CFR+ is anytime-valid so a time-capped solve still
yields a usable blueprint. Pure Python -> works under `--network none`.
Optionally amortise the cold solve across ProcessPool workers within ONE run by
pointing LD_MCCFR_CACHE_DIR at a writable RUNTIME dir (e.g. /cache/ld_mccfr) —
that is a train-time-computed cache, NOT a pre-saved repo bundle.

VARIANT (verified = OpenSpiel `liars_dice`, matches the env-server action ids):
6 faces, 6s WILD (count for any face except a bid on face 6), bid id
`(quantity-1)*6 + (face-1)`, challenge ("Liar") id `total_dice*6`. A bid is higher
iff its id is strictly greater.
"""

import math
import os
import random
import time
from collections import defaultdict
from itertools import combinations_with_replacement

NUM_FACES = 6
WILD_FACE = 6

# Convergence budget (2026-07-04, review-corrected): the IN-PROCESS solve here is
# only the per-worker FALLBACK when the shared prebuild cache is missing (e.g.
# /cache unwritable) — there is NO cross-process lock, so EVERY gen worker pays
# this budget concurrently on the cold path. Keep it at the bounded 90s/60k
# safety-net level; the REAL convergence raise lives in the gen-start PARALLEL
# prebuild (generate_trajectories LD_PREBUILD_ITERS, workers x 28k iters merged
# ~= 280k+ total), which runs once and is what workers actually load.
_MCCFR_ITERS = int(os.environ.get("LIARS_DICE_MCCFR_ITERS", "60000") or "60000")
_MCCFR_MAX_TOTAL = 10           # solve symmetric configs with total dice <= this
_MCCFR_MAX_SECONDS = float(os.environ.get("LIARS_DICE_MCCFR_SECONDS", "90") or "90")
_MCCFR_CHUNK = 50   # audit P1: 200 * ~0.5s/iter blew past the seconds cap by ~50s
_MCCFR_MIN_USABLE_ITERS = int(os.environ.get("LIARS_DICE_MCCFR_MIN_ITERS", "5000") or "5000")
_BLUEPRINT_MEM: "dict[int, dict | None]" = {}   # in-process: total_dice -> blueprint


# --------------------------------------------------------------------------- #
# Action-id encoding (matches the env-server / OpenSpiel liars_dice ids)
# --------------------------------------------------------------------------- #
def decode_bid(action_id: int) -> "tuple[int, int]":
    q, f = divmod(action_id, NUM_FACES)
    return q + 1, f + 1


def encode_bid(quantity: int, face: int) -> int:
    return (quantity - 1) * NUM_FACES + (face - 1)


def liar_id(total_dice: int) -> int:
    return total_dice * NUM_FACES


def legal_action_ids_for(current_id: "int | None", total_dice: int) -> list:
    lid = liar_id(total_dice)
    if current_id is None:
        return list(range(lid))
    return list(range(current_id + 1, lid)) + [lid]


# --------------------------------------------------------------------------- #
# Dice / hand helpers
# --------------------------------------------------------------------------- #
def _hands(dice_per_player: int) -> list:
    return [tuple(c) for c in combinations_with_replacement(range(1, NUM_FACES + 1), dice_per_player)]


def _hand_probability(hand: tuple) -> float:
    n = len(hand)
    counts: dict = defaultdict(int)
    for d in hand:
        counts[d] += 1
    perms = math.factorial(n)
    for c in counts.values():
        perms //= math.factorial(c)
    return perms / (NUM_FACES ** n)


def _matching_count(hand: tuple, face: int) -> int:
    if face == WILD_FACE:
        return sum(1 for d in hand if d == WILD_FACE)
    return sum(1 for d in hand if d == face or d == WILD_FACE)


# --------------------------------------------------------------------------- #
# External-sampling MCCFR over 1v1 Dudo with (hand, current-bid) info sets
# --------------------------------------------------------------------------- #
class _MccfrNode:
    __slots__ = ("regret", "strat_sum", "n")

    def __init__(self, n: int):
        self.regret = [0.0] * n
        self.strat_sum = [0.0] * n
        self.n = n

    def strategy(self) -> list:
        pos = [r if r > 0.0 else 0.0 for r in self.regret]
        s = sum(pos)
        return [p / s for p in pos] if s > 0.0 else [1.0 / self.n] * self.n

    def average(self) -> list:
        s = sum(self.strat_sum)
        return [x / s for x in self.strat_sum] if s > 0.0 else [1.0 / self.n] * self.n


class DudoMCCFR:
    def __init__(self, dice_per_player: int):
        self.dpp = dice_per_player
        self.total = 2 * dice_per_player
        self.liar = liar_id(self.total)
        self.hands = _hands(dice_per_player)
        self.hp = [_hand_probability(h) for h in self.hands]
        self.nodes: dict = {}

    def _node(self, key, n):
        nd = self.nodes.get(key)
        if nd is None:
            nd = _MccfrNode(n)
            self.nodes[key] = nd
        return nd

    def _challenge_u0(self, hands, bid_id, challenger):
        q, f = decode_bid(bid_id)
        count = _matching_count(hands[0], f) + _matching_count(hands[1], f)
        winner = challenger if count < q else 1 - challenger
        return 1.0 if winner == 0 else -1.0

    def _traverse(self, hands, history, traverser, rng):
        player = len(history) % 2
        last = history[-1] if history else None
        if last == self.liar:
            challenger = (len(history) - 1) % 2
            u0 = self._challenge_u0(hands, history[-2], challenger)
            return u0 if traverser == 0 else -u0
        actions = legal_action_ids_for(last, self.total)
        node = self._node((hands[player], last), len(actions))  # MEMORYLESS key
        strat = node.strategy()
        if player == traverser:
            util = [0.0] * len(actions)
            nu = 0.0
            for i, a in enumerate(actions):
                util[i] = self._traverse(hands, history + (a,), traverser, rng)
                nu += strat[i] * util[i]
            for i in range(len(actions)):
                node.regret[i] = max(0.0, node.regret[i] + util[i] - nu)  # CFR+
                node.strat_sum[i] += strat[i]
            return nu
        r = rng.random()  # external sampling: one opponent action
        acc = 0.0
        idx = len(actions) - 1
        for i, p in enumerate(strat):
            acc += p
            if r <= acc:
                idx = i
                break
        return self._traverse(hands, history + (actions[idx],), traverser, rng)

    def train(self, iters, rng):
        for _ in range(iters):
            h0 = rng.choices(self.hands, weights=self.hp, k=1)[0]
            h1 = rng.choices(self.hands, weights=self.hp, k=1)[0]
            self._traverse((h0, h1), (), 0, rng)
            self._traverse((h0, h1), (), 1, rng)

    def blueprint(self) -> dict:
        bp = {}
        for (hand, current), nd in self.nodes.items():
            # audit P2: zero-mass infosets (visited only as the opponent) used to
            # export a fabricated uniform entry, which outranked the stronger
            # epsilon-band fallback at runtime. Omit them -> band rescues.
            if sum(nd.strat_sum) <= 0.0:
                continue
            acts = legal_action_ids_for(current, self.total)
            avg = nd.average()
            bp[(hand, current)] = {acts[i]: avg[i] for i in range(len(acts))}
        return bp


def build_mccfr_blueprint(total_dice: int, iters: "int | None" = None,
                          seed: int = 0, max_seconds: "float | None" = None) -> "tuple[dict, int]":
    iters = iters if iters is not None else _MCCFR_ITERS
    if max_seconds is None:
        max_seconds = _MCCFR_MAX_SECONDS if _MCCFR_MAX_SECONDS > 0 else None
    solver = DudoMCCFR(total_dice // 2)
    rng = random.Random(seed)
    start = time.monotonic()
    done = 0
    while done < iters:
        chunk = min(_MCCFR_CHUNK, iters - done)
        solver.train(chunk, rng)
        done += chunk
        if max_seconds is not None and (time.monotonic() - start) >= max_seconds:
            break
    return solver.blueprint(), done


# Shared cache (de)serialization helpers — ONE implementation used by BOTH the
# runtime reader (_get_blueprint) AND the offline parallel prebuild
# (prebuild_liar_dice_blueprint.py), so the on-disk JSON format can never drift
# between writer and reader (the only real failure mode of a shared cache). The
# cache dir is a train-time RUNTIME dir (default /cache/ld_mccfr), NEVER a
# repo-committed bundle.
# Bump when the MCCFR algorithm / info-set abstraction / serialization changes, so
# a warm /cache written by an OLDER code version is NOT silently reused (the
# filename, not the content, gates the load + the prebuild skip). Reuse across
# ROUNDS of the SAME submission is fine and desired (the blueprint is
# game-deterministic) — the tag only invalidates cross-version caches in dev.
# r2 (2026-07-04): rev bump so the better-converged prebuild (28k iters/worker
# merged) actually REPLACES warm r1 caches written by the old 6k/worker build —
# without the bump, both the loader and the prebuild skip whenever the old file
# exists and the convergence raise silently never lands.
_BLUEPRINT_REV = "r2"


def cache_file_path(total_dice: int, cache_dir: "str | None" = None) -> "str | None":
    d = (cache_dir if cache_dir is not None
         else os.environ.get("LD_MCCFR_CACHE_DIR", "/cache/ld_mccfr")).strip()
    if not d:
        return None
    return os.path.join(d, f"ld_mccfr_total{total_dice}_{_BLUEPRINT_REV}.json")


def _serialize_blueprint(bp: dict) -> dict:
    """{(hand_tuple, current|None): {aid: prob}} -> JSON-safe {str: {str: float}}."""
    return {
        ",".join(str(d) for d in hand) + "|" + ("" if cur is None else str(cur)):
            {str(a): p for a, p in dist.items()}
        for (hand, cur), dist in bp.items()
    }


def _deserialize_blueprint(raw: dict) -> dict:
    """Inverse of _serialize_blueprint: rebuild {(hand_tuple, current|None): {aid: prob}}."""
    bp = {}
    for k, dist in raw.items():
        hand_s, cur_s = k.split("|", 1)
        hand = tuple(int(x) for x in hand_s.split(",") if x != "")
        current = None if cur_s == "" else int(cur_s)
        bp[(hand, current)] = {int(a): float(p) for a, p in dist.items()}
    return bp


def _atomic_write_json(path: str, obj) -> None:
    """Atomic write (tmp + os.replace) so concurrent workers never read a partial
    file. Creates the parent dir. Caller wraps in try/except (best-effort)."""
    import json
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def _cache_path(total_dice: int) -> "str | None":
    """Runtime cache path (train-time-computed, not a repo bundle). Default dir
    /cache/ld_mccfr; override or disable via LD_MCCFR_CACHE_DIR."""
    return cache_file_path(total_dice)


def _get_blueprint(total_dice: int):
    """Lazy per-total-dice blueprint, process-memoized. Returns None for
    unsupported configs (odd / > _MCCFR_MAX_TOTAL). Optionally loads/saves a
    runtime cache to amortise the cold solve across workers in one run."""
    if total_dice in _BLUEPRINT_MEM:
        return _BLUEPRINT_MEM[total_dice]
    if total_dice < 2 or total_dice > _MCCFR_MAX_TOTAL or total_dice % 2 != 0:
        _BLUEPRINT_MEM[total_dice] = None
        return None
    bp = None
    path = _cache_path(total_dice)
    if path and os.path.exists(path):
        try:
            import json
            with open(path, "r", encoding="utf-8") as fh:
                bp = _deserialize_blueprint(json.load(fh))
        except Exception:
            bp = None
    if bp is None:
        try:
            bp, _done = build_mccfr_blueprint(total_dice)
            # audit P1 quality gate: the seconds cap can deliver only a few
            # hundred iters (~0.5s/iter pure Python) -> near-uniform average
            # strategy, STRICTLY WORSE than the epsilon-band fallback it would
            # outrank (band ~0.79 vs uniform ~1.56 exploitability). Serve the
            # blueprint only when the solve actually converged usefully; the
            # multi-worker prebuild merge (written to cache) bypasses this path.
            if _done < _MCCFR_MIN_USABLE_ITERS:
                bp = None
            elif path:
                try:
                    _atomic_write_json(path, _serialize_blueprint(bp))
                except Exception:
                    pass  # best-effort; in-process copy still usable
        except Exception:
            bp = None
    _BLUEPRINT_MEM[total_dice] = bp
    return bp


# --------------------------------------------------------------------------- #
# Observation parse + public API
# --------------------------------------------------------------------------- #
import re  # noqa: E402

_RE_DICE = re.compile(r"Your dice:\s*\[([^\]]+)\]")
_RE_TOTAL = re.compile(r"Total dice in game:\s*(\d+)")
_RE_BID = re.compile(r'Current bid:\s*"(\d+)-(\d+)"')


def _parse_observation(observation: str) -> "tuple[list, int, int | None]":
    dm = _RE_DICE.search(observation)
    our_dice = [int(x.strip()) for x in dm.group(1).split(",")] if dm else []
    tm = _RE_TOTAL.search(observation)
    total_dice = int(tm.group(1)) if tm else len(our_dice) * 2
    bm = _RE_BID.search(observation)
    current_bid_id = encode_bid(int(bm.group(1)), int(bm.group(2))) if bm else None
    return our_dice, total_dice, current_bid_id


def cfr_enabled() -> bool:
    return (os.environ.get("LD_CFR_EXPERT", "1").strip() or "1").strip().lower() not in ("0", "false", "no", "")


def mccfr_distribution(observation: str, legal_action_ids: list) -> "dict | None":
    """Return {legal_action_id: prob} from the MCCFR blueprint (restricted to and
    renormalised over the legal ids), or None when the blueprint has no usable
    mass for this (own-dice, current-bid) info set — the caller then falls back to
    our epsilon-band policy. The key is EXACTLY what the agent observes, so it is
    reconstructable at eval from a single observation."""
    if not cfr_enabled() or not legal_action_ids:
        return None
    our_dice, total_dice, current_bid_id = _parse_observation(observation)
    if not our_dice or len(our_dice) != total_dice // 2:
        return None  # asymmetric/odd configs -> fallback
    bp = _get_blueprint(total_dice)
    if bp is None:
        return None
    entry = bp.get((tuple(sorted(our_dice)), current_bid_id))
    if not entry:
        return None
    legal = set(legal_action_ids)
    kept = {a: p for a, p in entry.items() if a in legal and p > 0.0}
    tot = sum(kept.values())
    if tot <= 0.0:
        return None
    return {a: p / tot for a, p in kept.items()}


# --------------------------------------------------------------------------- #
# Exploitability self-check (full-history best response; for __main__ only)
# --------------------------------------------------------------------------- #
def _dudo_br(total_dice, sigma_fn, br_player, br_hand, history, opp_reach):
    liar = liar_id(total_dice)
    player = len(history) % 2
    last = history[-1] if history else None
    if last == liar:
        challenger = (len(history) - 1) % 2
        q, f = decode_bid(history[-2])
        tot = 0.0
        for oh, w in opp_reach.items():
            hands = (br_hand, oh) if br_player == 0 else (oh, br_hand)
            count = _matching_count(hands[0], f) + _matching_count(hands[1], f)
            winner = challenger if count < q else 1 - challenger
            u0 = 1.0 if winner == 0 else -1.0
            tot += w * (u0 if br_player == 0 else -u0)
        return tot
    actions = legal_action_ids_for(last, total_dice)
    if player == br_player:
        return max(_dudo_br(total_dice, sigma_fn, br_player, br_hand, history + (a,), opp_reach)
                   for a in actions)
    tot = 0.0
    for ai, a in enumerate(actions):
        new_reach = {}
        for oh, w in opp_reach.items():
            p = sigma_fn(oh, last, actions)[ai]
            if p > 0.0:
                new_reach[oh] = w * p
        if new_reach:
            tot += _dudo_br(total_dice, sigma_fn, br_player, br_hand, history + (a,), new_reach)
    return tot


def dudo_exploitability(total_dice: int, sigma_fn) -> float:
    hs = _hands(total_dice // 2)
    hp = [_hand_probability(h) for h in hs]
    prior = {h: hp[i] for i, h in enumerate(hs)}
    total = 0.0
    for bp_player in (0, 1):
        for i, bh in enumerate(hs):
            total += hp[i] * _dudo_br(total_dice, sigma_fn, bp_player, bh, (), prior)
    return total


if __name__ == "__main__":
    # Self-check on the tractable 1-die-each game (total=2): the MCCFR average
    # strategy should reach LOW best-response exploitability (~0.5, vs ~1.56
    # uniform) — verifying the solver converges. Real configs (total=10) build a
    # time-capped blueprint at gen time.
    t0 = time.monotonic()
    bp, done = build_mccfr_blueprint(2, iters=20000, max_seconds=60)
    dt = time.monotonic() - t0
    print(f"[liar_dice_cfr] total=2 solved {done} iters in {dt:.1f}s, {len(bp)} info sets")

    def sigma(hand, current, actions):
        e = bp.get((tuple(sorted(hand)), current))
        if e:
            return [e.get(a, 0.0) for a in actions]
        return [1.0 / len(actions)] * len(actions)

    def unif(hand, current, actions):
        return [1.0 / len(actions)] * len(actions)

    print(f"[liar_dice_cfr] exploitability  MCCFR={dudo_exploitability(2, sigma):+.4f}  "
          f"uniform={dudo_exploitability(2, unif):+.4f}  (lower = less exploitable)")
