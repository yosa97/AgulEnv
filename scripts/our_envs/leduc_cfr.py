"""In-process CFR+ Nash solver for 2-player Leduc Hold'em — pure Python, no deps.

WHY: the Leduc PvP expert was an equity/pot-odds heuristic that loses to MCTS
(~38.7% gen win-rate). Verified research (NeurIPS/AAAI/Science/UAI, 2026-06-22)
is unambiguous: equity heuristics are EXPLOITABLE because the Leduc Nash is MIXED
(bluffing/balancing), and MCTS is UNSOUND for imperfect-information games (the
worst Leduc algorithm). The fix is a near-Nash CFR strategy, which is
unexploitable (zero-sum -> guarantees >= the game value) and trivially beats an
unsound MCTS opponent.

HOW (no-bundle, offline): Leduc is tiny (~936 infosets), so we SOLVE it in RAM
at gen-start with CFR+ (full-tree, deterministic — converges far faster than
MCCFR on a small game). The result is a dict {infoset_key -> {action: prob}}
held in memory for the run. NOTHING is written to disk: only this ALGORITHM
ships in the repo (like any heuristic), so there is no pre-saved table/JSON for a
repo-diff to flag as a "bundle". Pure Python -> works under `--network none`.

VARIANT (verified = OpenSpiel `leduc_poker`): 6-card deck (J/Q/K x 2 suits),
2 rounds, 1 private card R1, 1 public card R2, raises 2 (R1) / 4 (R2), a 2-raise
cap per round, 1 ante each (pot starts at 2). Suits are strategically irrelevant
(only rank matters at showdown) -> we solve in RANK space (3 ranks) with the
correct chance weights, which is exact AND faster than enumerating 6 cards.

Action tokens: 'c' = check/call (passive), 'r' = bet/raise, 'f' = fold.
Infoset key = own_rank + board_rank(or '_') + per-round betting strings:
``f"{card}{board}.{r1}.{r2}"`` — built identically here and at gen-time lookup
(leduc_poker_trajectories._canon_infoset_key) so the lookup always hits.

PAYOFF MODE (LP_WL, default ON): PvP eval scores pure win/loss/draw on the SIGN
of the return, not chips — under sign scoring a fold is a CERTAIN loss and is
weakly dominated by call/check. With LP_WL on, all terminal payoffs are
+1/-1/0 (sign of the chips outcome) so the CFR fixed point is the Nash of the
game the eval ACTUALLY scores: fold vanishes at every reached infoset (chips-EV
Nash folds J to a raise with prob 0.807 -> auto-concedes ~8-15% of hands).
LP_WL=0 restores the chips-EV solve for A/B.
"""

import os

# Empty-string-safe flag: LP_WL="" (set-but-empty) falls back to the default ON.
_LP_WL = (os.environ.get("LP_WL", "1").strip() or "1").lower() not in ("0", "false", "no", "off")

RANKS = ["J", "Q", "K"]          # rank index 0,1,2
R1_BET, R2_BET = 2, 4            # raise sizes per round
MAX_BETS = 2                     # 2-raise cap per round
_START_COUNTS = (2, 2, 2)       # 2 of each rank in the deck


# --------------------------------------------------------------------------- #
# Betting state machine (one round's action string a in {'c','r','f'})
# --------------------------------------------------------------------------- #
def _round_status(a: str) -> str:
    """'fold' (someone folded), 'closed' (betting matched), or 'open'."""
    if a.endswith("f"):
        return "fold"
    if a == "cc":
        return "closed"
    if a.endswith("c") and "r" in a:
        return "closed"
    return "open"


def _legal(a: str) -> list:
    """Legal action tokens at an OPEN round given its action string so far."""
    if a == "" or a[-1] == "c":          # not facing a bet -> check or bet
        return ["c", "r"]
    acts = ["f", "c"]                      # facing a bet -> fold or call
    if a.count("r") < MAX_BETS:           # ... and raise if under the cap
        acts.append("r")
    return acts


def _committed(hist: str) -> list:
    """Total chips each player has put in (antes + bets) for a full history."""
    c = [1, 1]                            # antes
    for ri, rstr in enumerate(hist.split("/")):
        bet = R1_BET if ri == 0 else R2_BET
        for k, a in enumerate(rstr):
            player = k % 2                # player 0 acts first each round
            to_call = c[1 - player] - c[player]
            if a == "c":
                c[player] += to_call      # call (or check when to_call == 0)
            elif a == "r":
                c[player] += to_call + bet
            # 'f' adds nothing (folder forfeits what is already in)
    return c


def _showdown_u0(a: int, b: int, board: int, hist: str) -> float:
    """Player-0 utility at a showdown (both rounds closed, no fold).
    LP_WL on: sign only (+1 win / -1 loss / 0 tie) instead of +/-stake."""
    # com[0] == com[1] at a called showdown
    stake = 1.0 if _LP_WL else float(_committed(hist)[0])
    p0_pair = a == board
    p1_pair = b == board
    if p0_pair and not p1_pair:
        return stake
    if p1_pair and not p0_pair:
        return -stake
    if a > b:
        return stake
    if b > a:
        return -stake
    return 0.0                            # equal rank, no pair -> tie


def infoset_key(card: str, board, r1: str, r2: str) -> str:
    """Canonical infoset key. `card`/`board` are rank chars ('J'/'Q'/'K');
    `board` is None when the board is not yet revealed. r1/r2 are the per-round
    betting strings ('' allowed). MUST match leduc_poker_trajectories lookup."""
    return f"{card}{board if board is not None else '_'}.{r1}.{r2}"


# --------------------------------------------------------------------------- #
# Chance weights (rank space, accounting for 2-of-each-rank without replacement)
# --------------------------------------------------------------------------- #
def _deal_weight(a: int, b: int) -> float:
    # ordered deal prob over 6*5=30 outcomes; same rank -> 2/30, else 4/30
    return (2.0 if a == b else 4.0) / 30.0


def _board_weights(a: int, b: int) -> list:
    counts = list(_START_COUNTS)
    counts[a] -= 1
    counts[b] -= 1                        # 4 cards remain
    return [counts[c] / 4.0 for c in range(3)]


class _Node:
    __slots__ = ("tokens", "regret", "strat_sum")

    def __init__(self, tokens):
        self.tokens = tokens
        n = len(tokens)
        self.regret = [0.0] * n
        self.strat_sum = [0.0] * n

    def strategy(self):
        n = len(self.tokens)
        pos = [r if r > 0.0 else 0.0 for r in self.regret]
        s = sum(pos)
        if s > 0.0:
            return [p / s for p in pos]
        return [1.0 / n] * n

    def average(self):
        n = len(self.tokens)
        s = sum(self.strat_sum)
        if s > 0.0:
            return [x / s for x in self.strat_sum]
        return [1.0 / n] * n


class LeducCFR:
    """CFR+ solver. Call solve(iters) -> {key: {token: prob}}."""

    def __init__(self):
        self.nodes = {}

    def _node(self, key, tokens):
        node = self.nodes.get(key)
        if node is None:
            node = _Node(list(tokens))
            self.nodes[key] = node
        return node

    def _cfr(self, a, b, board, hist, r0, r1, rc, t):
        rounds = hist.split("/")
        cur = rounds[-1]
        round_no = len(rounds)
        status = _round_status(cur)

        if status == "fold":
            folder = (len(cur) - 1) % 2
            if _LP_WL:                        # sign payoff: a fold is a certain loss
                return 1.0 if folder == 1 else -1.0
            com = _committed(hist)
            return com[1] if folder == 1 else -com[0]

        if status == "closed":
            if round_no == 1:                     # board chance node
                val = 0.0
                for c, q in enumerate(_board_weights(a, b)):
                    if q > 0.0:
                        val += q * self._cfr(a, b, c, hist + "/", r0, r1, rc * q, t)
                return val
            return _showdown_u0(a, b, board, hist)  # round 2 closed -> showdown

        # decision node
        player = len(cur) % 2
        card = a if player == 0 else b
        board_char = RANKS[board] if (round_no == 2 and board is not None) else None
        r2s = rounds[1] if len(rounds) > 1 else ""
        key = infoset_key(RANKS[card], board_char, rounds[0], r2s)
        tokens = _legal(cur)
        node = self._node(key, tokens)
        strat = node.strategy()

        util = [0.0] * len(tokens)
        node_util = 0.0
        for i, act in enumerate(tokens):
            if player == 0:
                util[i] = self._cfr(a, b, board, hist + act, r0 * strat[i], r1, rc, t)
            else:
                util[i] = self._cfr(a, b, board, hist + act, r0, r1 * strat[i], rc, t)
            node_util += strat[i] * util[i]

        cf = rc * (r1 if player == 0 else r0)     # counterfactual = chance * opp reach
        sign = 1.0 if player == 0 else -1.0       # p1 minimizes p0's utility
        for i in range(len(tokens)):
            node.regret[i] = max(node.regret[i] + cf * sign * (util[i] - node_util), 0.0)
        reach = r0 if player == 0 else r1
        for i in range(len(tokens)):
            node.strat_sum[i] += t * reach * strat[i]   # CFR+ linear averaging
        return node_util

    def solve(self, iters: int) -> dict:
        for t in range(1, iters + 1):
            for a in range(3):
                for b in range(3):
                    w = _deal_weight(a, b)
                    if w > 0.0:
                        self._cfr(a, b, None, "", 1.0, 1.0, w, t)
        return {key: dict(zip(node.tokens, node.average()))
                for key, node in self.nodes.items()}

    # ----- fixed-strategy expected value (for self-checks; full enumeration) ---
    def _value(self, a, b, board, hist, pol0, pol1):
        rounds = hist.split("/")
        cur = rounds[-1]
        round_no = len(rounds)
        status = _round_status(cur)
        if status == "fold":
            folder = (len(cur) - 1) % 2
            if _LP_WL:                        # sign payoff: a fold is a certain loss
                return 1.0 if folder == 1 else -1.0
            com = _committed(hist)
            return com[1] if folder == 1 else -com[0]
        if status == "closed":
            if round_no == 1:
                val = 0.0
                for c, q in enumerate(_board_weights(a, b)):
                    if q > 0.0:
                        val += q * self._value(a, b, c, hist + "/", pol0, pol1)
                return val
            return _showdown_u0(a, b, board, hist)
        player = len(cur) % 2
        card = a if player == 0 else b
        board_char = RANKS[board] if (round_no == 2 and board is not None) else None
        r2s = rounds[1] if len(rounds) > 1 else ""
        key = infoset_key(RANKS[card], board_char, rounds[0], r2s)
        tokens = _legal(cur)
        probs = (pol0 if player == 0 else pol1)(key, tokens)
        v = 0.0
        for i, act in enumerate(tokens):
            v += probs[i] * self._value(a, b, board, hist + act, pol0, pol1)
        return v

    def value(self, pol0, pol1) -> float:
        v = 0.0
        for a in range(3):
            for b in range(3):
                w = _deal_weight(a, b)
                if w > 0.0:
                    v += w * self._value(a, b, None, "", pol0, pol1)
        return v


# --------------------------------------------------------------------------- #
# True information-set best-response exploitability (self-check diagnostic).
# Walks PHYSICAL cards (6-card deck) so the deal/board chance is exact
# (without-replacement + board-blocking); the opponent's action probabilities are
# looked up in RANK space (suits are strategically irrelevant). A near-Nash
# strategy has exploitability ~0 chips/game; uniform-random is ~4.75. This is a
# DIAGNOSTIC ONLY — it never changes the teacher strategy.
# --------------------------------------------------------------------------- #
_PHYS_DECK = (0, 0, 1, 1, 2, 2)   # rank index (0=J,1=Q,2=K) of each physical card
_BR_ANTE = 1


def _br_showdown(br: int, opp: int, board: int) -> int:
    """+1 br wins, -1 opp wins, 0 tie (rank space; a board pair beats non-pair)."""
    br_pair = (br == board)
    opp_pair = (opp == board)
    if br_pair and not opp_pair:
        return 1
    if opp_pair and not br_pair:
        return -1
    if br > opp:
        return 1
    if opp > br:
        return -1
    return 0


def _opp_strategy_probs(strategy, opp_rank, board_rank, rnd, r1, r2, acts):
    board_char = RANKS[board_rank] if rnd == 2 else None
    key = infoset_key(RANKS[opp_rank], board_char, r1, r2 if rnd == 2 else "")
    d = strategy.get(key)
    if d is None:
        return [1.0 / len(acts)] * len(acts)
    return [d.get(a, 0.0) for a in acts]


def _br_value(strategy, br_player, br_card, rnd, r1, r2, board, pot_br, pot_opp, opp_reach):
    """Info-set best-response EV (chips) for br_player holding physical card
    br_card. The opponent's hidden card is integrated via opp_reach (physical-card
    -> reach weight). A TRUE best response (no clairvoyance) -> Nash exploits ~0."""
    cur = r1 if rnd == 1 else r2
    is_br = (len(cur) % 2 == br_player)
    acts = _legal(cur)
    facing = cur.endswith("r")
    if is_br:
        best = None
        for a in acts:
            v = _br_apply(strategy, br_player, br_card, rnd, r1, r2, board,
                          pot_br, pot_opp, opp_reach, a, True, facing)
            best = v if best is None or v > best else best
        return best if best is not None else 0.0
    total = 0.0
    board_rank = _PHYS_DECK[board] if board is not None else 0
    for ai, a in enumerate(acts):
        new_reach = {}
        for oc, w in opp_reach.items():
            p = _opp_strategy_probs(strategy, _PHYS_DECK[oc], board_rank, rnd, r1, r2, acts)[ai]
            if p > 0.0:
                new_reach[oc] = w * p
        if new_reach:
            total += _br_apply(strategy, br_player, br_card, rnd, r1, r2, board,
                               pot_br, pot_opp, new_reach, a, False, facing)
    return total


def _br_apply(strategy, br_player, br_card, rnd, r1, r2, board,
              pot_br, pot_opp, opp_reach, a, is_br, facing):
    nr1, nr2 = (r1 + a, r2) if rnd == 1 else (r1, r2 + a)
    if a == "f":
        wsum = sum(opp_reach.values())
        if _LP_WL:  # sign payoff: a fold is a certain loss/win regardless of pot
            return (-1.0 if is_br else 1.0) * wsum
        return (-float(pot_br) if is_br else float(pot_opp)) * wsum
    if a == "r":
        amt = R1_BET if rnd == 1 else R2_BET
        if is_br:
            diff = (pot_opp - pot_br) if facing else 0
            return _br_value(strategy, br_player, br_card, rnd, nr1, nr2, board,
                             pot_br + diff + amt, pot_opp, opp_reach)
        diff = (pot_br - pot_opp) if facing else 0
        return _br_value(strategy, br_player, br_card, rnd, nr1, nr2, board,
                         pot_br, pot_opp + diff + amt, opp_reach)
    # check / call ('c')
    if facing:
        hi = max(pot_br, pot_opp)
        return _br_close(strategy, br_player, br_card, rnd, nr1, nr2, board, hi, hi, opp_reach)
    pre = r1 if rnd == 1 else r2
    if len(pre) % 2 == 0:
        # opener checked, the other player is still to act -> round not closed
        return _br_value(strategy, br_player, br_card, rnd, nr1, nr2, board,
                         pot_br, pot_opp, opp_reach)
    return _br_close(strategy, br_player, br_card, rnd, nr1, nr2, board,
                     pot_br, pot_opp, opp_reach)


def _br_close(strategy, br_player, br_card, rnd, r1, r2, board, pot_br, pot_opp, opp_reach):
    if rnd == 1:
        # reveal the public card uniformly over the 4 cards held by neither player
        total = 0.0
        for b in range(len(_PHYS_DECK)):
            if b == br_card:
                continue
            sub = {oc: w * 0.25 for oc, w in opp_reach.items() if oc != b}
            if sub:
                total += _br_value(strategy, br_player, br_card, 2, r1, "", b,
                                   pot_br, pot_opp, sub)
        return total
    total = 0.0
    for oc, w in opp_reach.items():
        res = _br_showdown(_PHYS_DECK[br_card], _PHYS_DECK[oc], _PHYS_DECK[board])
        if _LP_WL:  # sign payoff: +/-1/0 on the showdown result, not the stake
            total += w * float(res)
        else:
            total += w * (pot_opp if res == 1 else (-pot_br if res == -1 else 0.0))
    return total


def compute_exploitability(strategy: dict) -> float:
    """True exploitability (~0 at Nash) in the ACTIVE utility mode: chips/game
    when LP_WL=0 (~4.75 for uniform-random), SIGN units (win-prob edge) when
    LP_WL is on — the BR walk's terminal payoffs are gated by _LP_WL exactly
    like the solver's, so the number is a real convergence guard in BOTH
    modes (review fix: the old chips-only walk made the under-convergence
    WARNING unreachable on the default LP_WL path)."""
    total = 0.0
    n = len(_PHYS_DECK)
    for br_player in (0, 1):
        for br_card in range(n):
            opp_reach = {c: 1.0 / 5.0 for c in range(n) if c != br_card}
            total += _br_value(strategy, br_player, br_card, 1, "", "", None,
                               _BR_ANTE, _BR_ANTE, opp_reach)
    return total / 6.0


# --------------------------------------------------------------------------- #
# Module singleton — solve ONCE per process, hold the strategy in RAM.
# --------------------------------------------------------------------------- #
_STRATEGY = None
_DEFAULT_ITERS = int(os.environ.get("LP_CFR_ITERS", "2000") or "2000")  # empty-safe: import must never raise
_EXPL_WARN = 0.10                # warn if best-response exploitability exceeds this
_TIEBREAK_EPS = 0.05             # MUST match leduc_poker_trajectories._LP_TIEBREAK_EPS


def _label_token_for_check(dist: dict) -> str:
    """Token-level mirror of the leduc_poker_trajectories LABEL policy (argmax +
    never-fold override + aggression tie-break) — used ONLY by the gen-start
    self-check in get_strategy. 'c' (check/call) is legal at every Leduc infoset,
    so the never-fold override always applies there."""
    items = [(t, p) for t, p in dist.items() if p > 0.0]
    if "c" in dist:
        items = [(t, p) for t, p in items if t != "f"]   # never-fold override
    if not items:
        return "c" if "c" in dist else min(dist)          # fallback prefers call
    ordered = sorted(items, key=lambda x: -x[1])
    if (len(ordered) >= 2 and ordered[0][1] - ordered[1][1] <= _TIEBREAK_EPS
            and "r" in (ordered[0][0], ordered[1][0])):
        return "r"                                        # aggression tie-break
    return ordered[0][0]


def get_strategy(iters: int = None) -> dict:
    """Return the cached near-Nash strategy {key: {token: prob}}, solving on the
    first call (lazy singleton). Never writes to disk.

    Convergence guard: a correct near-Nash Leduc solve has game value ~0
    (symmetric zero-sum). If |value| is large the solver under-converged (e.g.
    LP_CFR_ITERS set too low) and the teacher would be a weak/biased strategy
    rather than Nash — we warn loudly so a degraded teacher isn't silently
    distilled into the SFT set."""
    global _STRATEGY
    if _STRATEGY is None:
        solver = LeducCFR()
        _STRATEGY = solver.solve(iters if iters is not None else _DEFAULT_ITERS)
        try:
            # True best-response exploitability is a strictly stronger convergence
            # signal than game-value~=0 (a biased strategy can still have value ~0
            # yet be very exploitable). Keep game-value as a secondary line.
            expl = compute_exploitability(_STRATEGY)

            def _avg(key, tokens):
                d = _STRATEGY.get(key)
                return [d.get(t, 0.0) for t in tokens] if d else [1.0 / len(tokens)] * len(tokens)
            gv = solver.value(_avg, _avg)
            # The BR walk's terminal payoffs are gated by _LP_WL like the
            # solver's, so `expl` is measured in the SAME game the solve
            # optimizes — the threshold is a REAL under-convergence guard in
            # BOTH modes (review fix: the old chips-only walk made this
            # warning unreachable on the default LP_WL path).
            unit = "sign-units/game" if _LP_WL else "chips/game"
            if expl > _EXPL_WARN:
                print(
                    f"[leduc_cfr] WARNING: exploitability {expl:+.4f} {unit} "
                    f"(>{_EXPL_WARN}) at {len(_STRATEGY)} infosets - CFR under-converged; "
                    f"raise LP_CFR_ITERS (current default {_DEFAULT_ITERS}). Teacher is "
                    f"exploitable. (game value {gv:+.4f})",
                    flush=True,
                )
            else:
                print(
                    f"[leduc_cfr] solved {len(_STRATEGY)} infosets, exploitability "
                    f"{expl:+.4f} {unit} (near-Nash), game value {gv:+.4f}.",
                    flush=True,
                )
        except Exception as e:  # never block generation on the self-check
            print(f"[leduc_cfr] convergence self-check skipped: {e!r}", flush=True)
        # Gen-start LABEL self-check (never-raise): mirror the trajectories label
        # policy (argmax + never-fold + aggression tie-break) at token level over
        # EVERY solved infoset and verify it emits zero 'f' tokens.
        try:
            counts = {"c": 0, "r": 0, "f": 0}
            for dist in _STRATEGY.values():
                counts[_label_token_for_check(dist)] += 1
            n_keys = len(_STRATEGY)
            if counts["f"]:
                print(
                    f"[leduc_cfr] WARNING: label self-check found {counts['f']}/{n_keys} "
                    f"FOLD labels (never-fold override should make this 0)",
                    flush=True,
                )
            else:
                print(
                    f"[leduc_cfr] label self-check OK: 0 fold labels over {n_keys} "
                    f"infosets (raise {counts['r']} / call {counts['c']})",
                    flush=True,
                )
        except Exception as e:  # never block generation on the self-check
            print(f"[leduc_cfr] label self-check skipped: {e!r}", flush=True)
    return _STRATEGY


if __name__ == "__main__":
    import time

    iters = int(os.environ.get("LP_CFR_ITERS", "2000"))
    t0 = time.time()
    solver = LeducCFR()
    strat = solver.solve(iters)
    dt = time.time() - t0
    print(f"[leduc_cfr] solved {iters} iters in {dt:.2f}s -> {len(strat)} infosets")
    print(f"[leduc_cfr] exploitability (BR): {compute_exploitability(strat):+.4f} chips/game "
          f"(near-Nash ~0; uniform ~4.75)")
    _unif_strat = {k: {t: 1.0 / len(v) for t in v} for k, v in strat.items()}
    print(f"[leduc_cfr] exploitability of UNIFORM (sanity, expect ~4.75): "
          f"{compute_exploitability(_unif_strat):+.4f}")

    def avg_pol(key, tokens):
        d = strat.get(key)
        if d is None:
            return [1.0 / len(tokens)] * len(tokens)
        return [d.get(tok, 0.0) for tok in tokens]

    def unif(key, tokens):
        return [1.0 / len(tokens)] * len(tokens)

    def passive(key, tokens):   # always check/call; fold only if call illegal
        return [1.0 if tok == "c" else 0.0 for tok in tokens] if "c" in tokens \
            else [1.0 / len(tokens)] * len(tokens)

    def aggro(key, tokens):     # always raise if legal, else call
        if "r" in tokens:
            return [1.0 if tok == "r" else 0.0 for tok in tokens]
        return [1.0 if tok == "c" else 0.0 for tok in tokens]

    gv = solver.value(avg_pol, avg_pol)
    print(f"[leduc_cfr] game value (avg vs avg, P0): {gv:+.4f}  (near 0 = balanced)")
    print(f"[leduc_cfr] avg vs random   P0 EV: {solver.value(avg_pol, unif):+.4f} "
          f"| random vs avg P1 EV: {-solver.value(unif, avg_pol):+.4f}")
    print(f"[leduc_cfr] avg vs passive  P0 EV: {solver.value(avg_pol, passive):+.4f} "
          f"| avg vs aggro P0 EV: {solver.value(avg_pol, aggro):+.4f}")
    print("[leduc_cfr] sample strategies (should bluff/balance, not pure):")
    for key in ["J_..", "K_..", "Q_..", "J_.r.", "K_.r.", "JJ..", "KJ..", "QK.."]:
        if key in strat:
            pretty = {k: round(v, 2) for k, v in strat[key].items()}
            print(f"    {key:10} -> {pretty}")
