"""Othello (Reversi) expert SFT trajectory generator.

Othello is the 4th PvP game on the G.O.D feat/othello branch (EvalType.PVP,
EnvironmentName.OTHELLO, task-id 400M-499M). The trainer-side analogue of
gin_rummy_trajectories.py: roll out a strong-but-cheap heuristic expert vs the
env-server MCTS opponent and distill the winning play into SFT trajectories.

Why a heuristic expert (not the LP random+score-filter recipe): Othello is
perfect-information & deterministic with branching ~4-13 and ~58-move games,
so uniform-random play is extremely weak and harvesting wins by rejection is
wasteful. The expert is a corner-first positional+mobility blend searched with
depth-2 negamax (+alpha-beta; depth-3 in the <=12-empties endgame). The R1
smoke (2026-06-13) measured plain 1-ply greedy at only ~33% win-rate vs the
eval-strength MCTS@50 — the published "heuristic crushes MCTS" numbers come
from depth-5 negamax, so lookahead is load-bearing, not optional.

OpenSpiel encoding (load-bearing):
  * action = row*8 + col  (row-major, 0..63); PASS = 64.
  * observation_string = ToString(): header "Black (x) to play:" / "White (o)
    to play:", a "  a b c d e f g h  " label line, then 8 rows
    "{r+1} <c0> <c1> ... <c7> {r+1}" with cells '-'/'x'/'o', then labels again.
  * Legal-action lines render as "  {id} -> {algebraic}" e.g. "19 -> d3"; the
    model must reply with the ID NUMBER (server regex-extracts the int).
  * Terminal reward normalized to [0,1]: WIN=1.0, TIE=0.5, LOSS=0.0.

Self-contained flip logic (no pyspiel — not installed in the trainer image).
Never-raises: any parse/scoring failure falls back to the smallest legal id.
"""

import hashlib
import math
import os
import random
import re
import zlib
import time

import requests

from our_envs.pvp_prompts.pvp_prompt_loader import get_system_prompt, get_game_rules_prompt
from our_envs.pvp_prompts.pvp_state_format import reformat_to_pvp
from our_envs.pvp_tool_calling import assistant_move_message, tool_calling_enabled, WORKING, LONG_TERM
from our_envs.pvp_episode import run_toolcall_episode, MemoryPolicy


_REQUEST_TIMEOUT_SECONDS = 2400

# MUST match generate_trajectories._OPPONENT_CONFIG_PER_GAME["othello"].
# PR #1201: the EVAL baseline opponent is mcts@50 (ENVIRONMENTS["othello"]
# .eval_payload_extra); the upstream GRPO rollout uses a weaker mcts@25. We
# train against 50 so the win-biased data is calibrated to eval difficulty.
_OPPONENT_PAYLOAD = {"opponent": "mcts", "mcts_max_simulations": 50, "mcts_num_rollouts": 1}

_SYSTEM_PROMPT = get_system_prompt("othello")

_PASS_ACTION = 64

# 8-directional offsets for flank-and-flip.
_DIRS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

# Standard 8x8 weighted-square (disc-square) table: corners +120, X-squares
# -40 (most negative), C-squares -20, stable edge midpoints +20, center ~+3.
# Community-standard ordering corner >> edge > 0 > C-square > X-square.
_POS_WEIGHTS = (
    (120, -20, 20, 5, 5, 20, -20, 120),
    (-20, -40, -5, -5, -5, -5, -40, -20),
    (20, -5, 15, 3, 3, 15, -5, 20),
    (5, -5, 3, 3, 3, 3, -5, 5),
    (5, -5, 3, 3, 3, 3, -5, 5),
    (20, -5, 15, 3, 3, 15, -5, 20),
    (-20, -40, -5, -5, -5, -5, -40, -20),
    (120, -20, 20, 5, 5, 20, -20, 120),
)

_CORNERS = ((0, 0), (0, 7), (7, 0), (7, 7))


# --- board parsing ---------------------------------------------------------

def _parse_board(obs: str) -> "tuple[list[list[str]], str] | None":
    """Parse the 8x8 board + side-to-move from an observation string.

    Returns (board, me) where board[r][c] in {'-','x','o'} and me in {'x','o'},
    or None if the grid / side-to-move cannot be recovered.
    """
    me = None
    if re.search(r"Black\s*\(x\)\s*to\s*play", obs):
        me = "x"
    elif re.search(r"White\s*\(o\)\s*to\s*play", obs):
        me = "o"

    board: list[list[str]] = []
    for line in obs.splitlines():
        m = re.match(r"^\s*[1-8]\s+([-xo](?:\s+[-xo]){7})\s+[1-8]\s*$", line)
        if m:
            cells = re.split(r"\s+", m.group(1).strip())
            if len(cells) == 8:
                board.append(cells)

    if me is None or len(board) != 8:
        return None
    return board, me


def _legal_ids(last_user: str) -> list[int]:
    return [int(x) for x in re.findall(r"^\s*(\d+)\s*->", last_user, re.MULTILINE)]


def _extract_state(obs_text: str) -> str:
    """Strip the env-server "Game: ... / You are Player N" wrapper down to the
    validator's "Current State: ... Legal Actions: ..." body, mirroring
    gin_rummy_env.extract_and_format_observation (generic, no GR specifics).

    The env-server returns "...Current State:\\n{board}\\n\\nLegal Actions:\\n
    ...\\nYour choice...". This normalizes it so the subsequent
    reformat_to_pvp("othello") (wrap_pvp_format) is idempotent (returns as-is)
    instead of double-wrapping a second "Current State:" header. Falls back to
    the raw text when the anchors are absent so it degrades gracefully.
    """
    if not obs_text:
        return obs_text
    if "Invalid action:" in obs_text and "Legal Actions:" in obs_text:
        return obs_text
    m = re.search(r"Current State:\n(.*)", obs_text, re.DOTALL)
    if not m:
        return obs_text
    state_text = m.group(0)
    pm = re.search(r"You are Player (\d+)", obs_text)
    pid = int(pm.group(1)) if pm else 0
    if "Legal Actions:" not in state_text:
        return state_text
    before, after = state_text.split("Legal Actions:", 1)
    return before + f"You are Player {pid}.\nLegal Actions:" + after


# --- othello mechanics (self-contained, no pyspiel) ------------------------

def _opp(me: str) -> str:
    return "o" if me == "x" else "x"


def _apply_move(board: list[list[str]], r: int, c: int, me: str) -> list[list[str]]:
    """Place `me` at (r,c) and flip flanked opponent runs. Returns a new board."""
    opp = _opp(me)
    nb = [row[:] for row in board]
    nb[r][c] = me
    for dr, dc in _DIRS:
        run: list[tuple[int, int]] = []
        rr, cc = r + dr, c + dc
        while 0 <= rr < 8 and 0 <= cc < 8 and nb[rr][cc] == opp:
            run.append((rr, cc))
            rr += dr
            cc += dc
        if run and 0 <= rr < 8 and 0 <= cc < 8 and nb[rr][cc] == me:
            for fr, fc in run:
                nb[fr][fc] = me
    return nb


def _flips_any(board: list[list[str]], r: int, c: int, color: str) -> bool:
    """True if placing `color` at empty (r,c) flips >=1 opponent disc."""
    if board[r][c] != "-":
        return False
    opp = _opp(color)
    for dr, dc in _DIRS:
        rr, cc = r + dr, c + dc
        seen = 0
        while 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == opp:
            seen += 1
            rr += dr
            cc += dc
        if seen and 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == color:
            return True
    return False


def _legal_moves(board: list[list[str]], color: str) -> "list[tuple[int, int]]":
    return [
        (r, c)
        for r in range(8)
        for c in range(8)
        if board[r][c] == "-" and _flips_any(board, r, c, color)
    ]


def _legal_count(board: list[list[str]], color: str) -> int:
    return len(_legal_moves(board, color))


def _stable_edge_count(board: list[list[str]], color: str) -> int:
    """Count corner-anchored stable discs along the edges (cannot be flipped).

    A contiguous run of own discs starting at an owned corner and walking along
    either edge is permanently stable. This is the cheap core of the stability
    feature — the #2-ranked Othello eval term after corners — which the blend
    previously lacked entirely (MCTS rollouts price stability implicitly by
    playing to the end; an eval without it overvalues flippable material).
    """
    stable: set = set()
    edges = {
        (0, 0): ((0, 1), (1, 0)),
        (0, 7): ((0, -1), (1, 0)),
        (7, 0): ((0, 1), (-1, 0)),
        (7, 7): ((0, -1), (-1, 0)),
    }
    for (cr, cc), dirs in edges.items():
        if board[cr][cc] != color:
            continue
        stable.add((cr, cc))
        for dr, dc in dirs:
            r, c = cr + dr, cc + dc
            while 0 <= r < 8 and 0 <= c < 8 and board[r][c] == color:
                stable.add((r, c))
                r += dr
                c += dc
    return len(stable)


def _frontier_count(board: list[list[str]], color: str) -> int:
    """Count `color` discs adjacent (8-dir) to >=1 empty cell (fewer = better)."""
    cnt = 0
    for r in range(8):
        for c in range(8):
            if board[r][c] != color:
                continue
            for dr, dc in _DIRS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == "-":
                    cnt += 1
                    break
    return cnt


def _score_board(board: list[list[str]], me: str, my_mobility: "int | None" = None) -> float:
    """Phase-tapered linear blend, all terms from MY perspective.

    Opening/midgame (>12 empties): corner + mobility + positional - frontier;
    disc count is IGNORED (mid-game disc lead is anti-correlated with result).
    Endgame (<=12 empties): switch to disc maximization (mobility decays to 0).

    ``my_mobility``: the mover's already-computed legal-move count (the _negamax
    leaf has it in hand from its own move generation), so only the OPPONENT's
    mobility is recomputed here. Both sides' frontier counts are fused into the
    single 64-cell positional pass. Score-identical to the two-scan original.
    """
    opp = _opp(me)
    empties = sum(row.count("-") for row in board)
    mid = empties > 12

    pos = 0
    my_disc = opp_disc = 0
    my_front = opp_front = 0
    for r in range(8):
        for c in range(8):
            cell = board[r][c]
            if cell == "-":
                continue
            mine = cell == me
            if mine:
                pos += _POS_WEIGHTS[r][c]
                my_disc += 1
            else:
                pos -= _POS_WEIGHTS[r][c]
                opp_disc += 1
            if mid:
                for dr, dc in _DIRS:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == "-":
                        if mine:
                            my_front += 1
                        else:
                            opp_front += 1
                        break

    # Corners are ALREADY valued in `pos` (the piece-square table weights each
    # corner +120) AND in `stable_diff` (an owned corner anchors a stable edge),
    # so an additional explicit `corner_diff` term triple-counted corners and
    # over-emphasised them. We count corners exactly twice (piece-square +
    # stability) with no separate corner term. NOTE: the remaining lever
    # (rebalancing the `pos` coefficient so
    # corner-via-piece-square doesn't dwarf mobility) is a TUNING change that
    # needs a `test_othello_probe.sh` win-rate vs MCTS@50 to set safely — left
    # for that probe rather than guessed blind.
    stable_diff = _stable_edge_count(board, me) - _stable_edge_count(board, opp)

    if mid:
        my_mob = my_mobility if my_mobility is not None else _legal_count(board, me)
        mob_diff = my_mob - _legal_count(board, opp)
        front_diff = my_front - opp_front
        return (
            15.0 * stable_diff + 20.0 * mob_diff + 10.0 * pos - 8.0 * front_diff
        )
    return 15.0 * stable_diff + 10.0 * (my_disc - opp_disc) + 5.0 * pos


# Search depth ladder: R1 smoke measurements vs the eval-strength MCTS@50 —
# 1-ply greedy 32.8%, depth-2 38.8%, depth-3+stability 52.3% (2026-06-13).
# Othello games are routinely decided by endgame counting, so the ladder ends
# in an EXACT solve: with <=_SOLVE_EMPTIES empties the search depth equals the
# empty count and reaches terminal positions (perfect play to the end, scored
# by true disc count). Branching collapses late so alpha-beta keeps the solve
# cheap. Fewer games/sec is fine — per-turn distillation needs QUALITY wins.
_TERMINAL_SCORE = 100000.0

# Two regimes (mirrors the clobber engine; each disc-placing move fills exactly
# one empty, so `empties` is the endgame progress measure):
#   * ENDGAME exact solve (empties <= _ENDGAME_EMPTIES): negamax to terminal with
#     an exact-only transposition table + D4 symmetry canonicalisation. Othello is
#     routinely decided by exact endgame counting, so this is the strength lever.
#     Node-budgeted so a too-hard 13-16-empty solve in pure Python can't stall
#     bulk gen — it falls back to the shallow search. The <=12 floor always
#     finishes (a generous budget); 13-16 is best-effort.
#   * MID / opening (empties > _ENDGAME_EMPTIES): a cheap NO-TT shallow search
#     (transpositions are rare on a busy mid-board, so the TT would be overhead).
_SEARCH_DEPTH_MID = 3        # opening (> 16 empties) shallow depth
_SEARCH_DEPTH_END = 4        # 13-16 empties + endgame-overflow fallback depth
_ENDGAME_EMPTIES = 12        # <= this: ATTEMPT an exact solve to terminal
_PASS_SLACK = 2              # extra plies so a pass doesn't truncate the exact solve
# A node count does NOT bound pure-Python WALL time (a 12-empty exact solve can be
# 7-15s), so the exact endgame solve is bounded by a WALL-CLOCK deadline (checked
# every 2048 nodes); on timeout it falls back to the shallow depth-4 search. The
# node cap is only a coarse secondary guard. _ENDGAME_EMPTIES is 12 (not 16) so we
# only ATTEMPT the exact solve where it usually finishes within the deadline —
# >=13 empties is reliably too slow in pure Python and would just burn the
# deadline before falling back anyway.
try:
    _ENDGAME_TIME_BUDGET_S = float(os.environ.get("OTHELLO_ENDGAME_SECONDS") or "1.0")
except ValueError:
    _ENDGAME_TIME_BUDGET_S = 1.0  # garbage env value -> default (import must never raise)
_ENDGAME_NODE_BUDGET = 4_000_000
_MID_NODE_BUDGET = 8_000_000
_DEADLINE_CHECK_MASK = 2047  # probe the wall clock every 2048 nodes


class _BudgetExceeded(Exception):
    """Raised when a search exceeds its node budget OR wall-clock deadline (caught
    -> shallow fallback)."""


def _negamax(board, player, depth, alpha, beta, tt, budget):
    """Negamax + alpha-beta over _score_board, with an OPTIONAL exact-only
    transposition table (tt=dict caches only non-fail-high/low values keyed on the
    D4-canonical board + side-to-move, so every hit is the true minimax value and
    the argmax matches a no-TT search; tt=None disables it for the cheap mid
    search). `budget` is [node_count, cap]; exceeding the cap raises
    _BudgetExceeded. A pass consumes a depth tick; both-sides-stuck is terminal,
    scored by true disc count."""
    budget[0] += 1
    if budget[0] > budget[1] or (
        budget[2] is not None
        and (budget[0] & _DEADLINE_CHECK_MASK) == 0
        and time.monotonic() > budget[2]
    ):
        raise _BudgetExceeded

    moves = _legal_moves(board, player)
    opp = _opp(player)
    if not moves:
        if not _legal_moves(board, opp):
            my = sum(row.count(player) for row in board)
            op = sum(row.count(opp) for row in board)
            sign = 1.0 if my > op else (-1.0 if my < op else 0.0)
            return _TERMINAL_SCORE * sign + (my - op)
        return -_negamax(board, opp, depth - 1, -beta, -alpha, tt, budget)
    if depth <= 0:
        # Leaf-eval reuse: the mover's move list is already generated above, so
        # hand its count to _score_board (only the opponent's is recomputed).
        return _score_board(board, player, my_mobility=len(moves))

    key = None
    if tt is not None:
        # Cheap EXACT key: flat board string + side-to-move. The old D4 canonical
        # key (removed as dead code) did 8 board transforms +
        # string-join + min PER NODE, but symmetric endgame transpositions are
        # vanishingly rare, so it was pure overhead (measured 2-4x node-rate win
        # from dropping it). Same-position transpositions still hit; every cached
        # value is still the exact minimax value (TT semantics unchanged).
        key = ("".join("".join(r) for r in board), player)
        ent = tt.get(key)
        if ent is not None and ent[0] >= depth:
            return ent[1]

    # Order moves by static cell weight (corners first, X-squares last) to prune.
    moves.sort(key=lambda rc: -_POS_WEIGHTS[rc[0]][rc[1]])
    alpha_orig = alpha
    best = float("-inf")
    for r, c in moves:
        v = -_negamax(_apply_move(board, r, c, player), opp, depth - 1, -beta, -alpha, tt, budget)
        if v > best:
            best = v
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    if tt is not None and alpha_orig < best < beta:  # exact value -> cache
        prev = tt.get(key)
        if prev is None or prev[0] < depth:
            tt[key] = (depth, best)
    return best


def _search_root(board, me, legal, depth, tt, budget) -> "int | None":
    """Full-window negamax per legal root move sharing tt+budget; smallest-id
    tie-break (deterministic). Raises _BudgetExceeded if the node cap is hit."""
    opp = _opp(me)
    best_id = None
    best_score = None
    for aid in legal:
        if aid == _PASS_ACTION:
            continue
        r, c = divmod(aid, 8)
        if not (0 <= r < 8 and 0 <= c < 8):
            continue
        child = _apply_move(board, r, c, me)
        score = -_negamax(child, opp, depth - 1, float("-inf"), float("inf"), tt, budget)
        if best_score is None or score > best_score or (score == best_score and aid < best_id):
            best_score = score
            best_id = aid
    return best_id


def _search_depth_mid(empties: int) -> int:
    return _SEARCH_DEPTH_END if empties <= _ENDGAME_EMPTIES else _SEARCH_DEPTH_MID


def _expert_action_id(last_user: str) -> "int | None":
    """Pick the best legal action id. Endgame (<=16 empties) = exact solve with a
    transposition table (node-budgeted); otherwise a cheap shallow negamax. Never
    raises; on a budget overflow / parse failure it falls back to a shallow search
    then the smallest legal id."""
    legal = _legal_ids(last_user)
    if not legal:
        return None
    if legal == [_PASS_ACTION]:
        return _PASS_ACTION  # pass is the only legal move (no flips available)

    parsed = _parse_board(last_user)
    if parsed is None:
        return min(legal)  # board unreadable -> smallest legal id
    board, me = parsed
    empties = sum(row.count("-") for row in board)

    if empties <= _ENDGAME_EMPTIES:  # ATTEMPT an exact endgame solve with TT
        deadline = time.monotonic() + _ENDGAME_TIME_BUDGET_S
        try:
            best_id = _search_root(board, me, legal, empties + _PASS_SLACK, {},
                                   [0, _ENDGAME_NODE_BUDGET, deadline])
            if best_id is not None:
                return best_id
        except _BudgetExceeded:
            pass  # exact solve too slow in pure Python -> shallow fallback

    # opening / mid (or endgame timeout): cheap shallow no-TT search.
    try:
        best_id = _search_root(board, me, legal, _search_depth_mid(empties), None,
                               [0, _MID_NODE_BUDGET, None])
    except _BudgetExceeded:
        best_id = None
    return best_id if best_id is not None else min(legal)


def get_expert_action(messages: list[dict]) -> str:
    """Return the chosen action id as a string. Never raises."""
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    try:
        chosen = _expert_action_id(last_user)
    except Exception:
        chosen = None
    if chosen is None:
        raw = re.findall(r"^\s*(\d+)\s*->", last_user, re.MULTILINE)
        return min(raw, key=int) if raw else "0"
    return str(chosen)


def _label_action_id(last_user: str) -> "int | None":
    """Deterministic, LOAD-INDEPENDENT variant of _expert_action_id for the Tier-1
    canonical label: identical argmax EXCEPT the endgame exact solve is bounded by the
    NODE budget ONLY (deadline=None), never the wall-clock _ENDGAME_TIME_BUDGET_S — so
    the same board yields the same move regardless of CPU load on a multi-worker box
    (a wall-clock deadline makes the endgame argmax non-reproducible -> contradictory
    labels). Opening/mid is already fixed-depth deterministic."""
    legal = _legal_ids(last_user)
    if not legal:
        return None
    if legal == [_PASS_ACTION]:
        return _PASS_ACTION
    parsed = _parse_board(last_user)
    if parsed is None:
        return min(legal)
    board, me = parsed
    empties = sum(row.count("-") for row in board)
    if empties <= _ENDGAME_EMPTIES:
        try:
            best_id = _search_root(board, me, legal, empties + _PASS_SLACK, {},
                                   [0, _ENDGAME_NODE_BUDGET, None])   # NODE-only, no wall-clock deadline
            if best_id is not None:
                return best_id
        except _BudgetExceeded:
            pass
    try:
        best_id = _search_root(board, me, legal, _search_depth_mid(empties), None,
                               [0, _MID_NODE_BUDGET, None])
    except _BudgetExceeded:
        best_id = None
    return best_id if best_id is not None else min(legal)


def get_label_action(messages: list[dict]) -> str:
    """Tier-1 CANONICAL label: deterministic, load-independent persona-A argmax
    (endgame node-bounded, no wall-clock deadline) so the same board always yields the
    same SFT game_action label across worker processes. Never raises."""
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    try:
        chosen = _label_action_id(last_user)
    except Exception:
        chosen = None
    if chosen is None:
        raw = re.findall(r"^\s*(\d+)\s*->", last_user, re.MULTILINE)
        return min(raw, key=int) if raw else "0"
    return str(chosen)


# Square-id classes shared by the CoT label comment (below) and the persona-B
# anti-heuristic mix: action id = row*8 + col.
_CORNER_IDS = frozenset((0, 7, 56, 63))
_X_IDS = frozenset((9, 14, 49, 54))            # diagonal corner neighbours
_C_IDS = frozenset((1, 6, 8, 15, 48, 55, 57, 62))  # edge corner neighbours
_X_TO_CORNER = {9: 0, 14: 7, 49: 56, 54: 63}   # X-square -> the corner it gifts


def get_label_comment(messages: list[dict]) -> str:
    """One-sentence, rule-grounded CoT rationale for the CANONICAL label action.

    Deterministic (pure function of the observation: reuses get_label_action's
    node-only-bounded argmax + features recomputed from the parsed board),
    <= 140 chars, and NEVER raises — returns "" on any failure. NOT wired into
    the self-play row builder here; pvp_selfplay may pick it up via
    getattr(module, "get_label_comment", None).
    """
    try:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        legal = _legal_ids(last_user)
        if not legal:
            return ""
        parsed = _parse_board(last_user)
        if parsed is None:
            return ""
        board, me = parsed
        chosen = int(get_label_action(messages))
        if chosen not in legal:
            return ""
        if chosen == _PASS_ACTION:
            return "No flipping move available; passing."
        alg = _action_alg(str(chosen))
        if chosen in _CORNER_IDS:
            return f"Taking corner {alg}."

        empties = sum(row.count("-") for row in board)
        if empties <= _ENDGAME_EMPTIES:
            # Deterministic node-only-bounded solve of the chosen line (same
            # bound style as _label_action_id, so the verdict is reproducible).
            r, c = divmod(chosen, 8)
            try:
                v = -_negamax(
                    _apply_move(board, r, c, me), _opp(me),
                    empties + _PASS_SLACK - 1, float("-inf"), float("inf"),
                    {}, [0, _ENDGAME_NODE_BUDGET, None],
                )
            except _BudgetExceeded:
                v = None
            if v is not None:
                if v > _TERMINAL_SCORE / 2:
                    return f"Endgame solved: {alg} wins the final count."[:140]
                if v >= 0.0:
                    return f"Endgame solved: {alg} secures at least a draw."[:140]
                return f"Endgame: {alg} is the best remaining line."[:140]
            return f"Endgame: playing {alg} for the best final count."[:140]

        # Mid/opening: X-square avoidance (only meaningful while the gifted
        # corner is still open), then opponent-mobility restriction, then a
        # positional fallback — all recomputed from the observation alone.
        open_x = [
            x for x in legal
            if x in _X_IDS and board[_X_TO_CORNER[x] // 8][_X_TO_CORNER[x] % 8] == "-"
        ]
        if chosen not in _X_IDS and open_x:
            x = min(open_x)
            cx = _X_TO_CORNER[x]
            return (
                f"Avoiding X-square {_action_alg(str(x))}; "
                f"corner {_action_alg(str(cx))} is still open."
            )[:140]
        opp = _opp(me)
        opp_mob = {}
        for aid in legal:
            if aid == _PASS_ACTION:
                continue
            r, c = divmod(aid, 8)
            if not (0 <= r < 8 and 0 <= c < 8):
                continue
            opp_mob[aid] = _legal_count(_apply_move(board, r, c, me), opp)
        if chosen in opp_mob and len(opp_mob) > 1:
            alt = min(v for a, v in opp_mob.items() if a != chosen)
            if opp_mob[chosen] < alt:
                return (
                    f"Limits opponent to {opp_mob[chosen]} moves "
                    f"({alt - opp_mob[chosen]} fewer than the best alternative)."
                )[:140]
        return f"Playing {alg} for the best positional balance."[:140]
    except Exception:
        return ""


# --- SECOND TEACHER (persona B) — Boltzmann root-sampling (quantal response) -
# League/population self-play diversity (SELFPLAY_DUAL_TEACHER): seat B replaces
# persona A's deterministic argmax + smallest-id tie-break (the 0-400-exploitable
# R1 line) with a temperature-controlled SAMPLE over near-optimal opening/midgame
# moves -- de-telegraphing the line and broadening SFT state coverage. The DECISIVE
# ENDGAME (<=_ENDGAME_EMPTIES) keeps the exact argmax solver UNCHANGED -- perfect
# play there is non-negotiable. To stay both-seats-safe the deviation is BOUNDED:
# we only sample among moves within _BAND eval-units of the best (a tight near-
# optimal band), softmax at _TAU. A wide band would deliberately play moves the
# eval rates worse (heuristic-tie != true-value-tie) and weaken the data; keep it
# tight (A/B-tune via OTHELLO_SAMPLE_BAND / _TAU, probe before widening).
try:
    _OTH_SAMPLE_TAU = float(os.environ.get("OTHELLO_SAMPLE_TAU") or "8.0")
except ValueError:
    _OTH_SAMPLE_TAU = 8.0
_OTH_SAMPLE_TAU = max(_OTH_SAMPLE_TAU, 1e-6)
try:
    _OTH_SAMPLE_BAND = float(os.environ.get("OTHELLO_SAMPLE_BAND") or "12.0")
except ValueError:
    _OTH_SAMPLE_BAND = 12.0

# --- persona-B ANTI-HEURISTIC MIX -------------------------------------------
# With probability ~_STATIC_MIX_P (env OTHELLO_STATIC_MIX_P, default 0.3) the
# persona-B seat plays a board-blind static-weight-greedy policy for the WHOLE
# game: the upstream examples/mock policy CLASS (public knowledge, independently
# implemented) — corner-first, then X-/C-square avoidance while alternatives
# remain, then argmax over a static weight table. This approximates what the
# tournament winner's distilled 3-rule heuristic model plays, so our teacher's
# canonical labels (get_label_action, UNCHANGED) get stamped on the punish-lines
# against that policy class. The pick is DETERMINISTIC per game: a sha256 of the
# FIRST persona-B observation of the game (persona B has no seed channel — the
# selfplay driver calls it with the observation only and module-level RNG — so
# game boundaries are detected by the empty-count RISING: empties only ever fall
# within one othello game). Driver-only diversity; the label path is untouched.
try:
    _STATIC_MIX_P = float(os.environ.get("OTHELLO_STATIC_MIX_P", "0.3") or "0.3")
except ValueError:
    _STATIC_MIX_P = 0.3
_STATIC_MIX_P = min(max(_STATIC_MIX_P, 0.0), 1.0)

# Board-blind weight table (indexed by action id = row*8 + col): corner 100,
# X-square -50, C-square -20, edges positive, interior small.
_STATIC_WEIGHTS_ROWS = (
    (100, -20, 10, 5, 5, 10, -20, 100),
    (-20, -50, -2, -2, -2, -2, -50, -20),
    (10, -2, 1, 1, 1, 1, -2, 10),
    (5, -2, 1, 1, 1, 1, -2, 5),
    (5, -2, 1, 1, 1, 1, -2, 5),
    (10, -2, 1, 1, 1, 1, -2, 10),
    (-20, -50, -2, -2, -2, -2, -50, -20),
    (100, -20, 10, 5, 5, 10, -20, 100),
)
_STATIC_WEIGHTS = tuple(w for row in _STATIC_WEIGHTS_ROWS for w in row)

_PB_GAME_STATE = {"last_empties": None, "static": False}


def _static_seat_decision(first_obs: str) -> bool:
    """Pure deterministic per-game coin: sha256(first observation) < mix prob.
    Process-independent (no salted builtin hash), so the SAME game key always
    picks the same persona across workers and re-runs."""
    digest = hashlib.sha256(first_obs.encode("utf-8", "replace")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64 < _STATIC_MIX_P


def _static_mix_active(last_user: str, empties: int) -> bool:
    """Track persona-B game boundaries and hold one static/search decision per
    game. A NEW game is a call whose empty count EXCEEDS the last one seen
    (each othello move fills an empty, so empties are non-increasing within a
    game); the decision re-keys on that first observation and then sticks."""
    if _STATIC_MIX_P <= 0.0:
        return False
    st = _PB_GAME_STATE
    if st["last_empties"] is None or empties > st["last_empties"]:
        st["static"] = _static_seat_decision(last_user)
    st["last_empties"] = empties
    return st["static"]


def _static_greedy_id(legal: list) -> int:
    """Board-blind static-weight-greedy move: corner-first; else drop X-squares
    then C-squares while alternatives remain; else argmax static weight
    (smallest id on ties). Deterministic; always returns a member of `legal`."""
    cand = [a for a in legal if a != _PASS_ACTION and 0 <= a < 64]
    if not cand:
        return min(legal)
    corners = [a for a in cand if a in _CORNER_IDS]
    if corners:
        return min(corners)
    non_x = [a for a in cand if a not in _X_IDS]
    if non_x:
        cand = non_x
    non_c = [a for a in cand if a not in _C_IDS]
    if non_c:
        cand = non_c
    return max(cand, key=lambda a: (_STATIC_WEIGHTS[a], -a))


def _search_root_scored(board, me, legal, depth, tt, budget) -> list:
    """Like _search_root but returns [(aid, score), ...] for every playable root
    move, so the caller can softmax-sample. Raises _BudgetExceeded on cap."""
    opp = _opp(me)
    scored = []
    for aid in legal:
        if aid == _PASS_ACTION:
            continue
        r, c = divmod(aid, 8)
        if not (0 <= r < 8 and 0 <= c < 8):
            continue
        child = _apply_move(board, r, c, me)
        score = -_negamax(child, opp, depth - 1, float("-inf"), float("inf"), tt, budget)
        scored.append((aid, score))
    return scored


def _expert_action_id_b(last_user: str) -> "int | None":
    """Persona-B: bounded Boltzmann sampling in opening/mid; exact argmax in the
    endgame (delegates to persona A there) — EXCEPT in static-mix games, which
    DELIBERATELY stay board-blind through the endgame: the winner-family model
    this persona mimics has no endgame counting at all, and the punish-lines our
    exact solve stamps on its thrown endgames are precisely the training signal
    the mix exists to generate (labels are canonical regardless of the driver,
    so a blundered endgame costs nothing but produces the coverage). Assumes the
    sequential one-game-at-a-time selfplay driver (the empties-rising game
    detector in _PB_GAME_STATE would re-key on interleaved games). Never
    raises."""
    legal = _legal_ids(last_user)
    if not legal:
        return None
    if legal == [_PASS_ACTION]:
        return _PASS_ACTION
    parsed = _parse_board(last_user)
    if parsed is None:
        return min(legal)
    board, me = parsed
    empties = sum(row.count("-") for row in board)
    if _static_mix_active(last_user, empties):
        return _static_greedy_id(legal)       # anti-heuristic game: board-blind greedy
    if empties <= _ENDGAME_EMPTIES:
        return _expert_action_id(last_user)   # decisive endgame stays EXACT argmax

    try:
        scored = _search_root_scored(board, me, legal, _search_depth_mid(empties), None,
                                     [0, _MID_NODE_BUDGET, None])
    except _BudgetExceeded:
        scored = None
    if not scored:
        return _expert_action_id(last_user)   # fallback to persona A
    best = max(s for _, s in scored)
    band = [(aid, s) for aid, s in scored if best - s <= _OTH_SAMPLE_BAND]
    if len(band) <= 1:
        return band[0][0] if band else _expert_action_id(last_user)
    top = max(s for _, s in band)             # stable softmax over the tight band
    weights = [math.exp((s - top) / _OTH_SAMPLE_TAU) for _, s in band]
    total = sum(weights)
    if total <= 0.0:
        return min(aid for aid, _ in band)
    return random.choices([aid for aid, _ in band], weights=weights, k=1)[0]


def get_expert_action_b(messages: list[dict]) -> str:
    """Persona-B (bounded Boltzmann root-sampling) action id. Never raises."""
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    try:
        chosen = _expert_action_id_b(last_user)
    except Exception:
        chosen = None
    if chosen is None:
        raw = re.findall(r"^\s*(\d+)\s*->", last_user, re.MULTILINE)
        return min(raw, key=int) if raw else "0"
    return str(chosen)


# --- memory policy (tool_call / #1168 mode) --------------------------------

def _action_alg(action_id: str) -> str:
    """OpenSpiel action id -> algebraic square ("19"->"d3", 64->"pass")."""
    try:
        aid = int(action_id)
    except (TypeError, ValueError):
        return str(action_id)
    if aid == _PASS_ACTION:
        return "pass"
    r, c = divmod(aid, 8)
    if 0 <= r < 8 and 0 <= c < 8:
        return f"{'abcdefgh'[c]}{r + 1}"
    return str(action_id)


class _OthelloObsTransform:
    """Stateful obs transform: extract + PvP-wrap, then inject the #1201 colour
    prefix. OthelloAgent.format_state (PR #1201) prepends "You play x (Black).\\n"
    / "You play o (White).\\n" to observation_string (added because small models
    played the wrong colour). We detect our colour from the side-to-play header
    (the env only prompts us when it is our move) and remember it so the
    terminal observation (reflect "Final state:") gets the prefix too."""

    def __init__(self):
        self._colour = None

    def __call__(self, raw: str) -> str:
        text = reformat_to_pvp(_extract_state(raw), "othello")
        m = re.search(r"(Black \(x\)|White \(o\))\s+to\s+play", text)
        if m:
            self._colour = "x (Black)" if "Black" in m.group(1) else "o (White)"
        if self._colour:
            prefix = f"You play {self._colour}.\n"
            if "Current State:\n" in text:
                text = text.replace("Current State:\n", "Current State:\n" + prefix, 1)
            else:
                text = prefix + text
        return text


class OthelloMemoryPolicy(MemoryPolicy):
    """Othello opponent-modelling notes. Perfect-info, so the edge is positional:
    working = our move + corner/mobility picture this turn; long-term = whether
    the opponent contests corners (the single biggest Othello signal)."""

    def __init__(self):
        self._opp_corners = 0

    def observe(self, reformatted_obs: str) -> None:
        parsed = _parse_board(reformatted_obs)
        if parsed:
            board, me = parsed
            opp = _opp(me)
            self._opp_corners = max(
                self._opp_corners, sum(1 for r, c in _CORNERS if board[r][c] == opp)
            )

    def turn_writes(self, reformatted_obs, action_id, state, turn_idx):
        # Winner-recipe rarity gate (audit P1): memory notes fire on ~25% of
        # turns (stable crc32 - not every turn: dense per-turn writes trained
        # the memory-first habit that drowned game_action, 2026-06-22).
        if zlib.crc32(f"{action_id}:{turn_idx}:{len(reformatted_obs)}".encode()) % 4 != 0:
            return []
        parsed = _parse_board(reformatted_obs)
        if not parsed:
            return []
        board, me = parsed
        opp = _opp(me)
        my_c = sum(1 for r, c in _CORNERS if board[r][c] == me)
        opp_c = sum(1 for r, c in _CORNERS if board[r][c] == opp)
        note = (
            f"move {_action_alg(action_id)}; corners me/opp {my_c}/{opp_c}; "
            f"my legal moves {_legal_count(board, me)}"
        )
        return [(WORKING, "rewrite", 1, note)]

    def reflect_writes(self, outcome, state):
        tendency = "contests corners" if self._opp_corners >= 2 else "weak on corners"
        return [(LONG_TERM, "append", 1, f"opp held {self._opp_corners} corners; {tendency}")]


# --- episode generation ----------------------------------------------------

def generate_expert_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 70,
) -> "tuple[list[dict], float] | None":
    """Run one Othello game vs MCTS@25 using the heuristic expert policy.

    Returns ``(messages, final_reward)`` on success (final_reward in [0,1],
    0.5 = draw, > 0.5 = win), or None on env-server failure.

    max_turn=70: othello is ~58-60 plies and the LLM moves on roughly half,
    plus invalid-retry slack — well above a clean full game so the terminal
    reward is never dropped by hitting the loop cap. The assistant move is
    emitted via assistant_move_message (bare-id or <tool_call>, switchable),
    while the bare id string is always what is POSTed to /step.

    In tool_call mode (#1168) the game is distilled as per-turn-fresh examples
    with memory writes via run_toolcall_episode; bare_id mode keeps the legacy
    single growing-conversation form below.
    """
    if tool_calling_enabled():
        return run_toolcall_episode(
            game_name="othello",
            game_id=game_id,
            env_endpoint=env_endpoint,
            opponent_payload=_OPPONENT_PAYLOAD,
            max_turn=max_turn,
            rules_prompt=get_game_rules_prompt("othello"),
            expert_action_fn=get_expert_action,
            obs_transform=_OthelloObsTransform(),
            policy=OthelloMemoryPolicy(),
        )

    reset_payload = {"task_id": game_id, "seed": game_id, **_OPPONENT_PAYLOAD}
    try:
        res = requests.post(
            f"{env_endpoint}/reset", json=reset_payload, timeout=_REQUEST_TIMEOUT_SECONDS
        )
        res.raise_for_status()
        block = res.json()["result"]
        episode_id = block.get("episode_id", "")
        observation = _extract_state(block.get("observation", ""))
    except Exception as exc:
        print(f"[env] Reset failed (game {game_id}): {exc}")
        return None

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": reformat_to_pvp(observation, "othello")},
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
            observation = _extract_state(step_block.get("observation", ""))
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
        messages.append({"role": "user", "content": reformat_to_pvp(observation, "othello")})
    else:
        print(f"[env] max_turn={max_turn} reached (game {game_id})")

    return messages, final_reward
