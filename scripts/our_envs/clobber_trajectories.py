"""Clobber expert SFT trajectory generator.

Clobber is the 6th PvP game (G.O.D #1243: EvalType.PVP, EnvironmentName.CLOBBER,
task-id 700M-799M). Trainer-side analogue of othello_trajectories.py: roll out a
strong-but-cheap negamax expert vs the env-server MCTS opponent and distill the
winning play into per-turn SFT trajectories.

Why a DETERMINISTIC argmax expert (no mixing): clobber is perfect-information &
deterministic, so — like othello, unlike the imperfect-info games — a strong
search-optimal line is NOT exploitable in the game-theoretic sense; the eval's
2-6 seeded random opening plies prevent the opponent pre-computing a fixed line.
Boards are tiny (20-30 cells) and every move removes exactly one piece, so the
game is short and a deep / exact-to-terminal negamax is cheap.

OpenSpiel encoding (verified against open_spiel/games/clobber/clobber.cc):
  * Players: P0 = 'o' (White, moves first), P1 = 'x' (Black). Cells 'o'/'x'/'.'.
  * action = (start_row * columns + start_column) * 4 + direction, directions
    0=up 1=right 2=down 3=left (row offsets {-1,0,1,0}, col offsets {0,1,0,-1}).
  * ActionToString = "{col}{row}{col}{row}" from-square then to-square, e.g.
    "a5a4" (col letter 'a'+column; row number 1+(rows-1-internal_row), so the
    TOP board line carries the HIGHEST number). Legal lines render "{id} -> {label}".
  * observation_string = ToString(): rows "{rownum}{cells}" (no spaces), then a
    " {col letters}" footer. There is NO "to play" header — colour comes from the
    validator's format_state prefix "You play o (White)." / "You play x (Black).".
  * A move places your piece on the orthogonally-adjacent OPPONENT cell (capturing
    it); your origin becomes empty. No diagonals / empty / own. Make the LAST
    legal move: if it is your turn and you have no legal move, you LOSE.
  * Terminal reward normalized to [0,1]: WIN=1.0, TIE=0.5, LOSS=0.0.

Self-contained (no pyspiel, no CGSuite, no bundled tables). The engine is
negamax + alpha-beta in two regimes (see Claessen 2011; Griebel & Uiterwijk
2016): on a SMALL / FRAGMENTED board it solves EXACTLY to terminal with an
exact-only transposition table + opponent-mobility move ordering + symmetry
canonicalisation (component decomposition makes this the literature's decisive
endgame lever); on a DENSE board, where the stones form one big connected blob
with few transpositions, it runs a cheap NO-TT shallow mobility search (the table
would be pure overhead, and mobility-eval depth has diminishing returns — the
gains come from exact endgame solving, not a deeper static eval). Full CGT
atomic-weight valuation is deliberately NOT used (NP-hard, needs CGSuite,
collapses past 4x4); MCTS is deliberately NOT used (alpha-beta beats it ~84-16 in
the literature). Never-raises: any parse/scoring failure (or a node-budget
overflow) falls back to the cheap shallow search or the smallest legal id.

CLOBBER_LABEL selects the label policy (NEW default = "mobility"): "mobility"
DISABLES the exact endgame solver and labels every ply with a smooth,
time-budgeted iterative-deepening MOBILITY negamax (leaf eval =
my_legal_count - opp_legal_count) and a DETERMINISTIC SEMANTIC tie-break
(most-reduces-opponent-mobility at depth-1, then the engine's natural move
order) — the published winner's fix for the exact-solver + smallest-id-tiebreak
labels that distilled students to a 0-10 loss (our result). The "mobility" path
also writes the R3 (qgap, forced) row-weight side-channel via envs.row_meta.
"legacy" keeps the exact code path below UNCHANGED (byte-for-byte) for A/B.
"""

import hashlib
import os
import random
import re

import requests

from our_envs.pvp_prompts.pvp_prompt_loader import get_system_prompt, get_game_rules_prompt
from our_envs.pvp_prompts.pvp_state_format import reformat_to_pvp
from our_envs.pvp_tool_calling import assistant_move_message, tool_calling_enabled, WORKING, LONG_TERM
from our_envs.pvp_episode import run_toolcall_episode, MemoryPolicy

try:  # shared R3 (qgap, forced) side-channel; no-op fallback keeps never-raise
    from our_envs.row_meta import set_meta
except Exception:  # pragma: no cover - the shared contract module is always present
    def set_meta(qgap=None, forced=False):
        return None


_REQUEST_TIMEOUT_SECONDS = 2400

# EVAL baseline opponent = mcts@50 (G.O.D constants ENVIRONMENTS["clobber"]
# .eval_payload_extra). The training rollout helper uses a weaker mcts@25; we
# train at eval strength so the win-biased data matches eval difficulty.
_OPPONENT_PAYLOAD = {"opponent": "mcts", "mcts_max_simulations": 50, "mcts_num_rollouts": 1}

_SYSTEM_PROMPT = get_system_prompt("clobber")

# 4 orthogonal directions, OpenSpiel order (0=up,1=right,2=down,3=left).
_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))

_TERMINAL_SCORE = 100000.0
# Terminal / proven values are exact at any depth, so they store with a depth
# that dominates every real-search entry in the transposition table.
_TERMINAL_DEPTH = 1_000_000

# Depth ladder + EXACT endgame solving over connected components (Tier 2).
#
# Each move removes exactly one piece, so total_pieces bounds the remaining plies
# and depth == total_pieces is an exact solve to terminal. The literature's single
# highest-ROI lever for Clobber is exact endgame SOLVING (Claessen 2011: +72%
# relative win-rate from the "solving boards" enhancement alone), NOT a fancier
# static eval and NOT MCTS (alpha-beta beats MCTS ~84-16). Empty cells are
# permanent barriers, so the board fragments into independent connected
# components that only ever split — once each component is small, an exact solve
# of the whole remaining position is cheap because the transposition table folds
# the move-order transpositions of the disjunctive sum. So we solve exactly not
# just when FEW pieces remain (hard trigger) but also when the board has
# FRAGMENTED into small components (soft trigger). Mid-game we cap depth and
# score by mobility (the side that runs out of moves loses, so more legal moves
# than the opponent is the core signal).
# WIDENED exact envelope (15/20/10 pieces, 200k nodes) is GATED OFF by default:
# the implementation bench measured 0.15-3.4s PER MOVE on the newly-routed
# 16-20-piece positions — EVERY clobber game passes through that range, so
# default-on would collapse the ~77 games/s gen rate (the othello depth-5
# lesson). CLOBBER_EXACT_WIDE=1 turns it on for an A/B where the gen-rate
# budget allows. NOTE: the flag changes LABELS on 14-20-piece positions, so
# keep it consistent within one training run.
_EXACT_WIDE = (os.environ.get("CLOBBER_EXACT_WIDE") or "0").strip().lower() in ("1", "true", "yes", "on")
_SOLVE_TOTAL_HARD = 15 if _EXACT_WIDE else 13   # <= this many pieces total -> always solve exactly
_SOLVE_COMP_MAX = 10 if _EXACT_WIDE else 8      # largest connected component this small ...
_SOLVE_TOTAL_SOFT = 20 if _EXACT_WIDE else 18   # ... and total <= this -> solve exactly (fragmented board)

# Mid (non-exact) depth LADDER keyed on piece count. Clobber's branching factor
# scales with the stone count (~80 at the full 30-stone opening, ~15-25 by the
# mid-size endgame), so a deep search is CHEAP exactly where it is most decisive
# (few stones, close to terminal — blunders are unrecoverable) and ruinously
# expensive on the dense opening (far from terminal, precise play matters least).
# So we search DEEP when few stones remain and shallow on the opening. Each entry
# is (max_total, depth); piece counts above the last entry use _MID_DEPTH_DENSE.
_MID_DEPTH_LADDER = ((18, 6), (22, 4))
_MID_DEPTH_DENSE = 2

# Two regimes, two cost profiles:
#  * EXACT solve (small / fragmented board): a transposition table pays for itself
#    because the fragmenting disjunctive sum produces many move-order
#    transpositions; capped by _EXACT_NODE_BUDGET so a tangled position can never
#    stall bulk trajectory generation.
#  * MID / dense board: the stones form one big connected blob with FEW
#    transpositions, so the TT (and its per-node canonicalisation) is pure
#    overhead -> we run a cheap NO-TT shallow search instead. Mobility-eval DEPTH
#    has diminishing returns (the literature's strength gains come from exact
#    endgame solving, not a deeper static eval), and bulk generation needs the
#    throughput, so the mid depth is deliberately shallow. _MID_NODE_BUDGET is
#    large (the shallow depth bounds the tree anyway) so the mid search always
#    completes rather than degrading to a near-random legal move.
#
# The exact envelope/budget widening rides the CLOBBER_EXACT_WIDE gate above
# (root alpha-carry + fail-soft child windows make the bigger solves complete,
# but their absolute per-move wall cost is still seconds — gen-rate A/B first).
_EXACT_NODE_BUDGET = 200_000 if _EXACT_WIDE else 80_000
_MID_NODE_BUDGET = 4_000_000


class _BudgetExceeded(Exception):
    """Raised when a search exceeds its node budget (caught -> safe fallback)."""


# --- board / state parsing -------------------------------------------------

_BOARD_LINE_RE = re.compile(r"^\s*(\d+)\s*([ox.]+)\s*$")
_LEGAL_LINE_RE = re.compile(r"^\s*(\d+)\s*->\s*([a-z]\d+[a-z]\d+)\s*$", re.MULTILINE)
_LABEL_RE = re.compile(r"^([a-z])(\d+)([a-z])(\d+)$")


def _parse_board(obs: str) -> "tuple[list[list[str]], str] | None":
    """Parse the board grid + our colour from a clobber observation.

    Returns (board, me) where board[r][c] in {'-internal '} actually {'o','x','.'}
    with internal row 0 = the TOP rendered line, and me in {'o','x'}, or None if
    the grid / colour cannot be recovered. Never raises.
    """
    me = None
    pm = re.search(r"You play ([ox])\b", obs)
    if pm:
        me = pm.group(1)
    else:
        sm = re.search(r"You are Player (\d+)", obs)
        if sm:
            me = "o" if int(sm.group(1)) == 0 else "x"  # P0='o' (White, first)

    board: list[list[str]] = []
    width = None
    for line in obs.splitlines():
        m = _BOARD_LINE_RE.match(line)
        if not m:
            continue
        cells = list(m.group(2))
        if width is None:
            width = len(cells)
        if len(cells) == width:
            board.append(cells)

    if me is None or not board or width is None or width == 0:
        return None
    return board, me


def _legal_label_map(last_user: str) -> dict[int, str]:
    return {int(m.group(1)): m.group(2) for m in _LEGAL_LINE_RE.finditer(last_user)}


def _decode_label(label: str, rows: int) -> "tuple[int, int, int, int] | None":
    """"a5a4" -> (from_r, from_c, to_r, to_c) in INTERNAL row indices. The row
    number n maps to internal row r = rows - n (RowLabel = 1+(rows-1-r))."""
    m = _LABEL_RE.match(label)
    if not m:
        return None
    fc = ord(m.group(1)) - ord("a")
    fr = rows - int(m.group(2))
    tc = ord(m.group(3)) - ord("a")
    tr = rows - int(m.group(4))
    return fr, fc, tr, tc


# --- clobber mechanics (self-contained) ------------------------------------

def _opp(me: str) -> str:
    return "x" if me == "o" else "o"


def _apply_move(board: list[list[str]], fr: int, fc: int, tr: int, tc: int, me: str) -> list[list[str]]:
    """Move `me`'s piece from (fr,fc) onto opponent at (tr,tc): capture + occupy,
    origin becomes empty. Returns a new board."""
    nb = [row[:] for row in board]
    nb[tr][tc] = me
    nb[fr][fc] = "."
    return nb


def _legal_moves(board: list[list[str]], color: str) -> "list[tuple[int, int, int, int]]":
    """Every legal clobber move for `color`: a piece onto an orthogonally-adjacent
    OPPONENT piece. Returns (fr, fc, tr, tc) tuples."""
    rows = len(board)
    cols = len(board[0])
    opp = _opp(color)
    moves: list[tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if board[r][c] != color:
                continue
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == opp:
                    moves.append((r, c, nr, nc))
    return moves


def _piece_count(board: list[list[str]]) -> int:
    return sum(1 for row in board for cell in row if cell != ".")


def _canonical_key(board: list[list[str]]) -> str:
    """Lexicographically-smallest flattened board over the board's
    dimension-preserving symmetry group: the Klein four-group (identity,
    horizontal flip, vertical flip, 180° rotation) for any rectangle, extended to
    the full dihedral group D4 (adds the transpose-based maps) when the board is
    square. Geometric symmetries preserve Clobber values — orthogonal adjacency is
    isotropic and the o/x labels and side-to-move are untouched — so symmetric
    positions safely share one transposition-table slot. Board dimensions are
    fixed for the whole search (a fresh TT per decision), so the flattened string
    is an unambiguous key."""
    hflip = [row[::-1] for row in board]            # mirror columns
    vflip = board[::-1]                             # mirror rows
    rot180 = [row[::-1] for row in board[::-1]]     # 180° rotation
    variants = (board, hflip, vflip, rot180)
    if len(board) == len(board[0]):                 # square -> add transpose maps (D4)
        tr = [list(col) for col in zip(*board)]
        variants += (tr, [r[::-1] for r in tr], tr[::-1], [r[::-1] for r in tr[::-1]])
    return min("".join("".join(row) for row in v) for v in variants)


def _negamax(
    board: list[list[str]],
    player: str,
    depth: int,
    alpha: float,
    beta: float,
    tt: "dict | None",
    budget: list,
) -> float:
    """Negamax with alpha-beta and an OPTIONAL exact-only transposition table.
    Value is from `player`'s perspective; a player to move with NO legal move
    LOSES. The mobility eval (my_moves - opp_moves) is antisymmetric, so negating
    across turns is exact.

    When `tt` is a dict it caches ONLY exact values (positions whose search did not
    fail high or low), keyed on the symmetry-canonical board + side-to-move, so
    every TT hit returns the true minimax value and the deterministic argmax is
    identical to a no-TT search. When `tt` is None the search runs with no table
    and no canonicalisation — used on dense boards where transpositions are rare
    and the TT would be pure overhead. `budget` is [node_count, cap]; exceeding the
    cap raises _BudgetExceeded for a safe fallback. Leaves (depth<=0) are the most
    numerous nodes and almost never re-hit, so they are never canonicalised or
    stored."""
    budget[0] += 1
    if budget[0] > budget[1]:
        raise _BudgetExceeded

    if depth <= 0:
        moves = _legal_moves(board, player)
        if not moves:
            return -_TERMINAL_SCORE  # terminal at the horizon is still a loss
        return float(len(moves) - len(_legal_moves(board, _opp(player))))

    key = None
    if tt is not None:
        key = (_canonical_key(board), player)
        ent = tt.get(key)
        if ent is not None and ent[0] >= depth:
            return ent[1]

    moves = _legal_moves(board, player)
    if not moves:
        if tt is not None:
            tt[key] = (_TERMINAL_DEPTH, -_TERMINAL_SCORE)  # exact at any depth
        return -_TERMINAL_SCORE
    opp = _opp(player)

    alpha_orig = alpha
    best = float("-inf")
    if depth >= 3:
        # Move ordering (opponent-mobility ascending) pays off only with a real
        # subtree below the children, so it is gated to depth>=3; at depth 2 the
        # children are near-leaves and the ordering scan costs more than it saves.
        ordered = sorted(
            (_apply_move(board, fr, fc, tr, tc, player) for fr, fc, tr, tc in moves),
            key=lambda ch: len(_legal_moves(ch, opp)),
        )
        for child in ordered:
            v = -_negamax(child, opp, depth - 1, -beta, -alpha, tt, budget)
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
    else:
        for fr, fc, tr, tc in moves:
            child = _apply_move(board, fr, fc, tr, tc, player)
            v = -_negamax(child, opp, depth - 1, -beta, -alpha, tt, budget)
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


def _components_max_size(board: list[list[str]]) -> int:
    """Size of the largest connected component of OCCUPIED cells (4-adjacency).
    Empty cells are permanent barriers, so each maximal stone-connected region is
    an independent subgame; the largest one bounds how hard the remaining position
    is to solve exactly."""
    rows = len(board)
    cols = len(board[0])
    seen = [[False] * cols for _ in range(rows)]
    best = 0
    for sr in range(rows):
        for sc in range(cols):
            if board[sr][sc] == "." or seen[sr][sc]:
                continue
            stack = [(sr, sc)]
            seen[sr][sc] = True
            size = 0
            while stack:
                r, c = stack.pop()
                size += 1
                for dr, dc in _DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr][nc] and board[nr][nc] != ".":
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            if size > best:
                best = size
    return best


def _is_exact(board: list[list[str]], total: int) -> bool:
    """Whether to solve EXACTLY to terminal: few pieces remain (hard trigger) OR
    the board has fragmented into small independent components (soft trigger)."""
    if total <= _SOLVE_TOTAL_HARD:
        return True
    return total <= _SOLVE_TOTAL_SOFT and _components_max_size(board) <= _SOLVE_COMP_MAX


def _legacy_deep_opening() -> bool:
    """CLOBBER_LEGACY_DEEP (default OFF): deepen the dense-OPENING search beyond the
    depth-2 that lost 0-10, WITHOUT touching what makes legacy emittable. Legacy plays
    legal (0 forfeit) on every model — including the LoRA-tuned >4B ones the mobility
    label forfeits on — because its target is LOW-ENTROPY: an exact-solved endgame plus
    a SMALLEST-ID tie-break, both trivially reproducible so the student confidently
    emits game_action. Deepening only the dense-regime SEARCH picks stronger opening
    moves while keeping that exact-endgame + smallest-id structure intact, so it stays
    emittable. Flag-gated + env-tunable so it can be A/B'd against the shallow legacy
    before becoming the default."""
    return (os.environ.get("CLOBBER_LEGACY_DEEP") or "").strip().lower() in ("1", "true", "yes", "on")


def _mid_depth(total: int) -> int:
    """Shallow-search depth for the non-exact (dense) regime, from the ladder."""
    if _legacy_deep_opening():
        # Deeper, still-monotone f(piece_count) ladder for the dense regime. Endgame
        # (exact solve) + smallest-id tie-break are UNCHANGED, so the label stays
        # low-entropy/emittable; only the dense-opening moves get stronger. Env-tunable.
        if total <= int(os.environ.get("CLOBBER_DEEP_MID_MAX", "18")):
            return int(os.environ.get("CLOBBER_DEEP_MID_DEPTH", "6"))
        if total <= int(os.environ.get("CLOBBER_DEEP_HI_MAX", "22")):
            return int(os.environ.get("CLOBBER_DEEP_HI_DEPTH", "5"))
        return int(os.environ.get("CLOBBER_DEEP_DENSE_DEPTH", "4"))
    for cap, depth in _MID_DEPTH_LADDER:
        if total <= cap:
            return depth
    return _MID_DEPTH_DENSE


def _search_depth(board: list[list[str]]) -> int:
    """The depth the engine will actually use: an exact solve to terminal
    (depth == total_pieces, since each ply removes one piece) when small /
    fragmented, otherwise the piece-count-laddered mid depth."""
    total = _piece_count(board)
    if _is_exact(board, total):
        return total
    return _mid_depth(total)


def _search_root(
    board: list[list[str]], me: str, legal: dict, depth: int, tt: "dict | None", budget: list
) -> "int | None":
    """One full-width root search at `depth`, sharing `tt` + `budget` across all
    candidates; returns the best action id (smallest id among the optimal moves —
    the deterministic tie-break) or None if no candidate is playable.

    ROOT ALPHA-CARRY: candidates are ordered ONCE by a cheap 1-ply key (opponent
    mobility on the child, ascending = likely-best first; aid as the secondary
    key so the search order — and therefore budget consumption — is fully
    deterministic). The first child is searched with a full window; every later
    child with the fail-soft window (best_score - 0.5, +inf). _negamax is
    FAIL-SOFT (it accumulates `best` across the window bounds and returns it on
    both fail-high and fail-low), so per Knuth-Moore a child search that comes
    back <= best_score - 0.5 is an UPPER bound on that child's true value — the
    move provably cannot beat OR tie best_score — while any result above
    best_score - 0.5 is the exact minimax value. All evals are integer-valued
    (mobility diffs / +-_TERMINAL_SCORE), so the 0.5 margin cleanly separates
    "ties best" (returned exact) from "worse" (pruned): the argmax AND the
    smallest-id tie-break are identical to a full-window search, and the explicit
    (score, aid) comparison keeps the winner independent of iteration order.
    Raises _BudgetExceeded if the node cap is hit."""
    rows = len(board)
    cols = len(board[0])
    opp = _opp(me)
    cands: "list[tuple[int, int, list[list[str]]]]" = []
    for aid, label in legal.items():
        mv = _decode_label(label, rows)
        if mv is None:
            continue
        fr, fc, tr, tc = mv
        if not (0 <= fr < rows and 0 <= tr < rows and 0 <= fc < cols and 0 <= tc < cols):
            continue
        if board[fr][fc] != me or board[tr][tc] != opp:
            continue  # label inconsistent with our parsed board -> skip
        child = _apply_move(board, fr, fc, tr, tc, me)
        cands.append((len(_legal_moves(child, opp)), aid, child))
    cands.sort(key=lambda t: (t[0], t[1]))
    best_id = None
    best_score = None
    for _, aid, child in cands:
        beta_child = float("inf") if best_score is None else 0.5 - best_score
        score = -_negamax(child, opp, depth - 1, float("-inf"), beta_child, tt, budget)
        if best_score is None or score > best_score or (score == best_score and aid < best_id):
            best_score = score
            best_id = aid
    return best_id


def _best_id_at_depth(
    board: list[list[str]], me: str, legal: dict, depth: int
) -> "int | None":
    """Self-contained exact-TT root search at `depth` (fresh table + budget).
    Retained as the unit-test entry point; production routes through
    _expert_action_id, which selects the table/budget per regime."""
    return _search_root(board, me, legal, depth, {}, [0, _EXACT_NODE_BUDGET])


def _expert_action_id(last_user: str) -> "int | None":
    """Pick the best legal action id. Two regimes:

      * small / fragmented board -> EXACT negamax to terminal WITH the exact-only
        transposition table (the disjunctive-sum transpositions make it cheap) —
        the literature's decisive endgame-solving lever;
      * dense board -> a cheap NO-TT shallow mobility search (the TT is overhead
        when the stones form one big connected blob with few transpositions).

    The exact solve is node-capped; a tangled position that blows the cap falls
    back to the cheap shallow search, which always completes. Never raises."""
    legal = _legal_label_map(last_user)
    if not legal:
        return None

    parsed = _parse_board(last_user)
    if parsed is None:
        return min(legal)  # board unreadable -> smallest legal id
    board, me = parsed

    total = _piece_count(board)
    if _is_exact(board, total):  # exact-solve regime (small / fragmented)
        try:
            best_id = _search_root(board, me, legal, total, {}, [0, _EXACT_NODE_BUDGET])
            if best_id is not None:
                return best_id
        except _BudgetExceeded:
            pass  # tangled exact solve -> fall through to the cheap shallow search

    # dense board (or rare exact overflow): cheap shallow no-TT laddered search.
    depth = _mid_depth(total)
    try:
        best_id = _search_root(board, me, legal, depth, None, [0, _MID_NODE_BUDGET])
    except _BudgetExceeded:
        best_id = None
    return best_id if best_id is not None else min(legal)


# --- CLOBBER_LABEL = "mobility" (NEW default label policy) ------------------
# The published clobber+gin winner's ablation proved the LEGACY label policy
# UNLEARNABLE: an EXACT endgame solver (proven +-terminal values) plus a
# smallest-action-id tie-break hands the student a target with (a) a cliff-shaped
# value surface — most positions score a flat +-_TERMINAL_SCORE, so there is no
# smooth gradient to imitate — and (b) a systematic top-left / low-id bias in the
# tie set. Students distilled on it LOSE 0-10 (exactly our result). The winner's
# fix, reproduced here, is a SMOOTH deterministic MOBILITY negamax label at every
# ply:
#   * NO exact endgame solver -> always a bounded-depth mobility-eval negamax, so
#     the leaf signal is my_legal_count - opp_legal_count (a continuous, learnable
#     gradient) rather than a +-terminal cliff on most positions;
#   * a NODE-BUDGETED iterative-deepening search (per-decision (node_budget,
#     max_depth) sampled REPRODUCIBLY from the observation) replaces the
#     piece-count depth ladder -> depth adapts to how tangled THIS board is, and a
#     depth that times out mid-search is DISCARDED (keep the last complete depth);
#   * a DETERMINISTIC + SEMANTIC tie-break (most-reduces-opponent-mobility at
#     depth-1, then the engine's natural scan order) replaces smallest-id / obs
#     hash -> ties resolve to a move the value function can explain, killing the
#     unlearnable positional bias.
# CLOBBER_LABEL=legacy restores the exact solver path above unchanged for A/B.

_MOBILITY_DEADLINE_CHECK = 256  # poll the wall-clock deadline every N search nodes


def _clobber_label_mode() -> str:
    """'legacy' iff CLOBBER_LABEL is explicitly set to 'legacy'; otherwise the NEW
    approved default 'mobility'. Unset / anything else -> 'mobility'."""
    return "legacy" if (os.environ.get("CLOBBER_LABEL") or "").strip().lower() == "legacy" else "mobility"


class _TimeUp(Exception):
    """Raised when a mobility search passes its wall-clock deadline; caught by the
    iterative-deepening loop, which then keeps the last FULLY-completed depth."""


def _order_key(child: "list[list[str]]", mover: str, opp: str) -> tuple:
    """1-ply move-ordering key for a child position (opp to move, `mover` just
    played): proven-win child FIRST (opp has no reply), likely-loss child LAST
    (`mover` has no mobility left in the child), else ascending
    opp_mobility - mover_mobility. Pure alpha-beta ordering -> it only speeds the
    search; the minimax value returned is identical for any ordering."""
    opp_mob = len(_legal_moves(child, opp))
    my_mob = len(_legal_moves(child, mover))
    if opp_mob == 0:
        return (0, 0)          # opponent stuck in the child -> we win -> search first
    if my_mob == 0:
        return (2, 0)          # our mobility gone -> likely loss -> search last
    return (1, opp_mob - my_mob)


# Transposition-table entry flags. A per-decision TT memoises internal-node results
# so the DENSE-OPENING candidate subtrees (which overlap heavily) are searched once,
# not N times — the real gen-rate lever. EXACT = the true value; LOWER/UPPER = a
# fail-high/fail-low BOUND (only usable when it produces a cutoff for the probing
# window). The TT NEVER changes the returned value, only the node cost, so the label
# is unchanged (proven in tests/batch_a/test_clobber_pvs.py).
_TT_EXACT = 0
_TT_LOWER = 1  # fail-high: true value >= stored
_TT_UPPER = 2  # fail-low:  true value <= stored


def _board_str(board: "list[list[str]]") -> str:
    """Flat cell string used as the TT key. Board dimensions are constant within a
    single decision's search, so the flattened cells identify the position uniquely
    (no per-decision collision)."""
    return "".join(map("".join, board))


def _negamax_timed(
    board: "list[list[str]]", player: str, depth: int, alpha: float, beta: float,
    clock: list, tt: "dict | None" = None
) -> float:
    """Negamax + alpha-beta with a MOBILITY leaf eval (my_legal - opp_legal), a
    NODE-COUNT deadline, and an OPTIONAL per-decision transposition table `tt`.

    Value is from `player`'s perspective; a player to move with no legal move LOSES.
    Every node bumps `clock` = [nodes_visited, node_budget] and raises _TimeUp once the
    budget is hit (a NODE budget, not wall-clock, keeps the label a pure function of the
    board -> deterministic on any machine). When `tt` is a dict it is probed (returning
    early on an EXACT hit at >= depth, or on a bound that cuts off) and stored on exit;
    the TT is a pure memoisation, so the returned value is identical to the no-TT search
    and the distilled label is unchanged. `tt=None` (the A/B legacy path + the reference
    scan) runs the plain search. Children ordered by the 1-ply _order_key heuristic."""
    clock[0] += 1
    if clock[0] >= clock[1]:
        raise _TimeUp
    opp = _opp(player)
    if depth <= 0:
        moves = _legal_moves(board, player)
        if not moves:
            return -_TERMINAL_SCORE  # terminal at the horizon is still a loss
        return float(len(moves) - len(_legal_moves(board, opp)))
    key = None
    if tt is not None:
        key = (_board_str(board), player)
        e = tt.get(key)
        if e is not None and e[2] >= depth:
            ev, flag, _ed = e
            if flag == _TT_EXACT:
                return ev
            if flag == _TT_LOWER:
                if ev > alpha:
                    alpha = ev
            elif ev < beta:  # _TT_UPPER
                beta = ev
            if alpha >= beta:
                return ev
    alpha_orig = alpha
    moves = _legal_moves(board, player)
    if not moves:
        if tt is not None:
            tt[key] = (-_TERMINAL_SCORE, _TT_EXACT, depth)
        return -_TERMINAL_SCORE
    children = [_apply_move(board, fr, fc, tr, tc, player) for fr, fc, tr, tc in moves]
    children.sort(key=lambda ch: _order_key(ch, player, opp))
    best = float("-inf")
    for child in children:
        v = -_negamax_timed(child, opp, depth - 1, -beta, -alpha, clock, tt)
        if v > best:
            best = v
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    if tt is not None:
        if best <= alpha_orig:
            flag = _TT_UPPER
        elif best >= beta:
            flag = _TT_LOWER
        else:
            flag = _TT_EXACT
        e = tt.get(key)
        if e is None or e[2] <= depth:  # keep the deepest (or newest-equal) entry
            tt[key] = (best, flag, depth)
    return best


def _natural_order_key(label: str, rows: int) -> tuple:
    """Engine natural move-order key for a legal label: (from_row, from_col, dir) in
    the SAME internal scan order _legal_moves enumerates. Used ONLY as the FINAL
    residual tie-break — geometric board order, deliberately NOT the integer action
    id and NOT an observation hash."""
    mv = _decode_label(label, rows)
    if mv is None:
        return (1 << 30, 0, 0)
    fr, fc, tr, tc = mv
    try:
        d = _DIRS.index((tr - fr, tc - fc))
    except ValueError:
        d = 4
    return (fr, fc, d)


def _mobility_root_search(cands: list, me: str, opp: str, depth: int, clock: list) -> "tuple":
    """SINGLE-PASS alpha-beta (PVS) root search at `depth`. Replaces the old
    N-independent-full-window-per-candidate scan (which shared no alpha-beta bound
    across root children and cost O(N x full subtree) — the ~155x slowdown on dense
    openings). Here candidate 0 is searched full-window to establish `best`; every
    later candidate is searched with the parent window (best-0.5, +inf), so it returns
    its EXACT value when >= best (detecting ties and new bests) and FAILS LOW — pruning
    its whole subtree — when < best. Cost is ~O(one strong search).

    Candidates are pre-ordered best-first by the 1-ply heuristic so the best is found
    early (maximal pruning). CRUCIALLY the returned TIE SET is order-independent and
    identical to the full-window version's {aid : value == top} (integer/half-integer
    leaves), so the distilled LABEL is unchanged by this optimisation — only the
    wall-clock/node cost drops. Returns (top_value, tied_ids, second_value); `second`
    is exact when a candidate is demoted from best, else the best fail-low upper bound
    (used only for the default-OFF qgap weight). Raises _TimeUp on node-budget
    exhaustion (caller keeps the last fully-completed depth).

    A FRESH transposition table is shared across the N candidate searches at this
    depth. On a dense opening the candidates differ by one move and their subtrees
    overlap almost completely, so candidate 1 populates the table and candidates 2..N
    mostly hit it — this is what PVS alone could NOT do (tied move values -> no window
    pruning). The TT is per-call (never crosses decisions), so the label stays a pure
    function of the board."""
    tt: dict = {}
    ordered = sorted(cands, key=lambda ac: _order_key(ac[1], me, opp))
    best = None
    tied: list = []
    second = float("-inf")
    for aid, child in ordered:
        if best is None:
            best = -_negamax_timed(child, opp, depth - 1, float("-inf"), float("inf"), clock, tt)
            tied = [aid]
            continue
        # Parent window (best-0.5, +inf) -> child called with (-inf, 0.5-best); the
        # 0.5 margin (matching the exact solver) cleanly separates "== best" (exact,
        # a tie) from "< best" (fail-low, pruned) for integer/half-integer scores.
        v = -_negamax_timed(child, opp, depth - 1, float("-inf"), 0.5 - best, clock, tt)
        if v > best:
            second = best        # exact runner-up: the demoted best
            best = v
            tied = [aid]
        elif v == best:
            tied.append(aid)
        elif v > second:
            second = v           # fail-low upper bound (approximate; qgap is default-OFF)
    return best, tied, second


def _mobility_seed(last_user: str) -> int:
    """Deterministic per-decision seed from the observation text (stable across
    processes, unlike the PYTHONHASHSEED-salted builtin hash) so the sampled
    (node_budget, max_depth) — and therefore the label — is reproducible for a
    given board on ANY machine (a node budget, unlike a wall-clock budget, is
    machine-speed-independent)."""
    return int.from_bytes(hashlib.sha256(last_user.encode("utf-8", "ignore")).digest()[:8], "big")


def _mobility_depth_mode() -> str:
    """'ladder' (NEW default) = deterministic depth from PIECE COUNT (a visible board
    feature) -> a learnable, symmetry-consistent label. 'sampled' = the OLD per-
    observation sha256 (node_budget, depth_cap) sampling, kept only for A/B.

    WHY the switch: the winner's ablation + our finder analysis pinned the per-obs
    sampling as the top roughness source. sha256(obs) avalanches, so a mirror/rotation-
    symmetric or near-duplicate board (different obs text -> different seed) drew a
    different depth -> a different, non-mirror move. The student saw the 'expert' play
    symmetric positions inconsistently — pure label noise it cannot explain from the
    board — so on eval states it reverted to the base Qwen memory-writing prior and
    emitted working_memory_rewrite instead of game_action (= the forfeit). A depth that
    is a pure function of the piece count gives symmetric/near-dup boards the SAME depth
    and the SAME move: a consistent, imitable target."""
    return os.environ.get("CLOBBER_DEPTH_MODE", "ladder").strip().lower()


def _mobility_depth_for(total: int) -> int:
    """Deterministic MONOTONE (fewer pieces -> deeper, then a fixed cap) mobility-search
    depth ladder, a pure function of the piece count. Endgames get the deepest FIXED cap
    (leaves are still mobility evals — terminal is reached only on genuinely short lines,
    NOT a systematic exact-solve regime, which is the brittle ±cliff artifact the
    winner's ablation warned against); dense openings stay shallow (fast gen, still far
    deeper than the legacy depth-2 that lost 0-10). Applied at EVERY ply as one uniform
    smooth mobility family. effective_cap = min(this, total) elsewhere, so tiny boards
    just search their whole (small) tree. Env-tunable."""
    if total <= int(os.environ.get("CLOBBER_LADDER_ENDGAME_MAX", "12")):
        return int(os.environ.get("CLOBBER_LADDER_ENDGAME_DEPTH", "10"))
    if total <= int(os.environ.get("CLOBBER_LADDER_MID_MAX", "16")):
        return int(os.environ.get("CLOBBER_LADDER_MID_DEPTH", "8"))
    if total <= int(os.environ.get("CLOBBER_LADDER_HI_MAX", "22")):
        return int(os.environ.get("CLOBBER_LADDER_HI_DEPTH", "6"))
    return int(os.environ.get("CLOBBER_LADDER_DENSE_DEPTH", "4"))


def _expert_action_id_mobility(last_user: str, seed: int) -> "int | None":
    """NEW default label policy: a smooth deterministic mobility negamax at every
    ply. Writes the R3 (qgap, forced) side-channel via row_meta.set_meta right
    before returning. NEVER raises: single-legal shortcut, and the FIRST legal id
    (natural order, with a logged warning) only as an absolute last resort. Returns
    None ONLY when there is no legal action at all."""
    legal = _legal_label_map(last_user)
    if not legal:
        return None
    forced = len(legal) == 1  # exactly one legal action -> no strategic signal

    parsed = _parse_board(last_user)
    if parsed is None:
        print("[clobber] mobility: board unparseable; falling back to first legal id")
        set_meta(qgap=0.0, forced=forced)
        return next(iter(legal))
    board, me = parsed
    opp = _opp(me)
    rows = len(board)
    cols = len(board[0])

    cands: "list[tuple[int, list[list[str]]]]" = []
    for aid, label in legal.items():
        mv = _decode_label(label, rows)
        if mv is None:
            continue
        fr, fc, tr, tc = mv
        if not (0 <= fr < rows and 0 <= tr < rows and 0 <= fc < cols and 0 <= tc < cols):
            continue
        if board[fr][fc] != me or board[tr][tc] != opp:
            continue  # label inconsistent with the parsed board -> skip
        cands.append((aid, _apply_move(board, fr, fc, tr, tc, me)))

    if not cands:
        print("[clobber] mobility: no legal label matched the board; using first legal id")
        set_meta(qgap=0.0, forced=forced)
        return next(iter(legal))

    if len(cands) == 1:  # single playable move -> nothing to search
        set_meta(qgap=0.0, forced=forced)
        return cands[0][0]

    # Each ply removes exactly one piece, so no line is deeper than total pieces.
    total = _piece_count(board)
    if _mobility_depth_mode() == "sampled":
        # OLD per-observation sha256 sampling — the RANK-1 roughness (mirror/near-dup
        # boards drew different depths -> non-mirror moves -> unlearnable -> forfeit).
        # Kept ONLY for A/B against the deterministic ladder. node 10k-30k / depth 6-8
        # were the validated fast-gen values (~26 g/s at node 3k-8k on the H100 box).
        rng = random.Random(seed)
        _nmin = int(os.environ.get("CLOBBER_NODE_MIN", "10000"))
        _nmax = int(os.environ.get("CLOBBER_NODE_MAX", "30000"))
        _dmin = int(os.environ.get("CLOBBER_DEPTH_MIN", "6"))
        _dmax = int(os.environ.get("CLOBBER_DEPTH_MAX", "8"))
        node_budget = rng.randint(min(_nmin, _nmax), max(_nmin, _nmax))
        depth_cap = rng.randint(min(_dmin, _dmax), max(_dmin, _dmax))
        effective_cap = max(1, min(depth_cap, total))
        clock = [0, node_budget]  # [nodes_visited, node_budget] — deterministic deadline
    else:
        # NEW default 'ladder': depth = a deterministic MONOTONE function of the piece
        # count (a feature the student reads off the board), so symmetric/near-duplicate
        # boards get the SAME depth and the SAME move -> a consistent, learnable label.
        # The node budget is now a large FIXED anti-hang safety only; it is deterministic,
        # so even if it binds on a pathological dense board it binds identically on the
        # board's mirrors -> the label stays a pure function of the board (unlike the
        # obs-hash node budget, which broke symmetry). At ladder depths 4-8 it never
        # binds in practice, so the reached depth is exactly the ladder value.
        effective_cap = max(1, min(_mobility_depth_for(total), total))
        clock = [0, int(os.environ.get("CLOBBER_NODE_SAFETY", "2000000"))]

    best_val = None
    tied = None
    second = float("-inf")
    for depth in range(1, effective_cap + 1):
        try:
            best_val, tied, second = _mobility_root_search(cands, me, opp, depth, clock)
        except _TimeUp:
            break  # discard this partial depth, keep the last fully-completed one

    if best_val is None:  # unreachable safety: even depth-1 didn't complete -> redo it
        best_val, tied, second = _mobility_root_search(cands, me, opp, 1, [0, float("inf")])

    # DETERMINISTIC + SEMANTIC tie-break. Primary: the move that most reduces the
    # opponent's reply mobility at depth-1 (smallest opp mobility in the child) — a
    # genuine game feature the value function can explain. Final residual tie: the
    # engine's natural scan order (board geometry). WHY not smallest-id / obs-hash:
    # the winner's ablation showed an id/hash tie-break bakes a systematic
    # (top-left, low-id) bias the student cannot learn (0-10); a mobility-semantic
    # tie-break removes that bias while staying fully deterministic.
    child_by_aid = dict(cands)
    best_id = min(
        tied,
        key=lambda aid: (
            len(_legal_moves(child_by_aid[aid], opp)),
            _natural_order_key(legal[aid], rows),
        ),
    )

    # qgap = best - second-best (negamax units): 0 on a genuine tie / single move /
    # forced turn. `second` comes from the root search (exact when a candidate was
    # demoted from best, else a fail-low upper bound — fine for the default-OFF weight).
    qgap = max(0.0, float(best_val - second)) if second > float("-inf") else 0.0
    set_meta(qgap=qgap, forced=forced)
    return best_id


def get_expert_action(messages: list[dict]) -> str:
    """Return the chosen action id as a string. Never raises.

    CLOBBER_LABEL selects the label policy:
      * 'mobility' (NEW default): smooth deterministic mobility-negamax label at
        every ply; no exact solver, no id/obs-hash tie-break; writes the R3
        (qgap, forced) side-channel before returning.
      * 'legacy': the exact-solver + piece-count depth-ladder + smallest-id
        tie-break path, kept byte-for-byte for A/B."""
    last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    if _clobber_label_mode() == "legacy":
        try:
            chosen = _expert_action_id(last_user)
        except Exception:
            chosen = None
        if chosen is None:
            raw = re.findall(r"^\s*(\d+)\s*->", last_user, re.MULTILINE)
            return min(raw, key=int) if raw else "0"
        return str(chosen)

    seed = _mobility_seed(last_user)
    try:
        chosen = _expert_action_id_mobility(last_user, seed)
    except Exception:
        chosen = None
    if chosen is None:
        # absolute last resort: first legal id in natural order (never smallest-id).
        raw = re.findall(r"^\s*(\d+)\s*->", last_user, re.MULTILINE)
        set_meta(qgap=0.0, forced=(len(raw) == 1))
        if raw:
            print("[clobber] mobility labeler produced no move; using first legal id")
            return raw[0]
        return "0"
    return str(chosen)


# --- SECOND TEACHER (persona B) — value-tied optimal-move sampler -----------
# League/population self-play diversity (SELFPLAY_DUAL_TEACHER): seat 1 samples
# UNIFORMLY among the proven-equal optimal moves instead of persona A's
# smallest-id tie-break. Confined to the EXACT regime where depth == piece-count
# reaches true terminals, so every root child scores exactly +/-_TERMINAL_SCORE:
# all tied moves share an IDENTICAL game-theoretic value (a proven win, or in a
# lost position an equally-proven loss — binary perfect-info has no "less-losing"
# move). Uniform sampling among them therefore CONSERVES game value (minimax
# tautology) -> poison-safe -> both-seats. It only removes the systematic top-left
# (low-id) bias and spreads SFT data across the optimal-move manifold. CRITICAL:
# we gate on a COMPLETED exact search (no _BudgetExceeded). On exact overflow OR
# the dense/heuristic regime we DEFER to persona A's deterministic choice -- never
# sample among mobility-heuristic "ties", which are NOT proven-equal and could
# shave strength.
def _search_root_ties(
    board: list[list[str]], me: str, legal: dict, depth: int, tt: "dict | None", budget: list
) -> "tuple[float, list] | None":
    """Like _search_root but returns (best_score, [all aids whose score==best_score])
    so the caller can sample among the value-tied set. Uses the same root
    alpha-carry + fail-soft (best_score - 0.5, +inf) windows as _search_root; the
    tie SET is preserved exactly because every eval is integer-valued (in the
    exact regime, +-_TERMINAL_SCORE): a child whose true value EQUALS best_score
    sits strictly inside the window (opponent-view value -best_score is below
    beta_child = 0.5 - best_score) so it returns its exact value and is detected
    as a tie, while a strictly-worse child fails low to an upper bound
    <= best_score - 0.5 and can never fake a tie. The tied list is sorted so the
    output is independent of the search order. Raises _BudgetExceeded if the
    node cap is hit (caller must treat that as 'no completed exact result')."""
    rows = len(board)
    cols = len(board[0])
    opp = _opp(me)
    cands: "list[tuple[int, int, list[list[str]]]]" = []
    for aid, label in legal.items():
        mv = _decode_label(label, rows)
        if mv is None:
            continue
        fr, fc, tr, tc = mv
        if not (0 <= fr < rows and 0 <= tr < rows and 0 <= fc < cols and 0 <= tc < cols):
            continue
        if board[fr][fc] != me or board[tr][tc] != opp:
            continue
        child = _apply_move(board, fr, fc, tr, tc, me)
        cands.append((len(_legal_moves(child, opp)), aid, child))
    cands.sort(key=lambda t: (t[0], t[1]))
    best_score = None
    tied: list = []
    for _, aid, child in cands:
        beta_child = float("inf") if best_score is None else 0.5 - best_score
        score = -_negamax(child, opp, depth - 1, float("-inf"), beta_child, tt, budget)
        if best_score is None or score > best_score:
            best_score = score
            tied = [aid]
        elif score == best_score:
            tied.append(aid)
    if best_score is None:
        return None
    tied.sort()
    return best_score, tied


def _expert_action_id_b(last_user: str) -> "int | None":
    """Persona-B: uniform-sample among value-tied optimal moves in the EXACT
    regime; deterministic (persona A) everywhere else. Never raises."""
    legal = _legal_label_map(last_user)
    if not legal:
        return None
    parsed = _parse_board(last_user)
    if parsed is None:
        return min(legal)
    board, me = parsed
    total = _piece_count(board)
    if _is_exact(board, total):
        try:
            res = _search_root_ties(board, me, legal, total, {}, [0, _EXACT_NODE_BUDGET])
            if res is not None and res[1]:
                return random.choice(res[1])  # all tied -> identical terminal value
        except _BudgetExceeded:
            pass  # exact overflow -> deterministic fallback (no heuristic-tie sampling)
    return _expert_action_id(last_user)  # dense / overflow -> persona A, poison-safe


def get_expert_action_b(messages: list[dict]) -> str:
    """Persona-B action id as a string. Never raises.

    In 'mobility' mode the value-tied sampler has no exact regime to sample in (the
    exact solver is disabled), so persona B uses the SAME deterministic mobility
    label as persona A — a consistent single teacher. In 'legacy' mode it is the
    unchanged value-tied optimal-move sampler, for A/B."""
    if _clobber_label_mode() != "legacy":
        return get_expert_action(messages)
    last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    try:
        chosen = _expert_action_id_b(last_user)
    except Exception:
        chosen = None
    if chosen is None:
        raw = re.findall(r"^\s*(\d+)\s*->", last_user, re.MULTILINE)
        return min(raw, key=int) if raw else "0"
    return str(chosen)


# --- memory policy (tool_call / #1201 mode) --------------------------------

class _ClobberObsTransform:
    """Extract + PvP-wrap, then inject the colour prefix the validator's
    ClobberAgent.format_state prepends: "You play o (White).\\n" (P0) /
    "You play x (Black).\\n" (P1). Clobber's observation_string has no
    side-to-move header, so we recover our seat from the env-server's
    "You are Player N" wrapper and remember it for the terminal observation."""

    def __init__(self):
        self._colour = None  # "o (White)" / "x (Black)"

    def __call__(self, raw: str) -> str:
        sm = re.search(r"You are Player (\d+)", raw)
        if sm:
            self._colour = "o (White)" if int(sm.group(1)) == 0 else "x (Black)"
        text = reformat_to_pvp(raw, "clobber")
        if self._colour:
            prefix = f"You play {self._colour}.\n"
            if "Current State:\n" in text:
                text = text.replace("Current State:\n", "Current State:\n" + prefix, 1)
            else:
                text = prefix + text
        return text


class ClobberMemoryPolicy(MemoryPolicy):
    """Clobber is perfect-info, so the edge is positional/tempo: working = our move
    + mobility picture this turn; long_term = the durable tempo lesson (keep your
    own moves alive, strand the opponent). No fabricated opponent reads."""

    def turn_writes(self, reformatted_obs, action_id, state, turn_idx):
        # The clobber+gin task winner clobber NEVER writes memory — every clobber
        # assistant turn is game_action-only, content-free memory block. (Regime-3
        # short-note writes were tried v13 2026-07-06 and did NOT fix the clobber
        # forfeit while breaking gin — reverted to the no-memory champion path.)
        return []

    def reflect_writes(self, outcome, state):
        # Consistent with turn_writes: clobber is fully memory-free.
        return []


# --- episode generation ----------------------------------------------------

def generate_expert_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 70,
) -> "tuple[list[dict], float] | None":
    """Run one Clobber game vs the env-server MCTS opponent using the negamax
    expert. Returns ``(messages, final_reward)`` (final_reward in [0,1],
    0.5 = draw, > 0.5 = win), or None on env-server failure.

    max_turn=70: boards are 20-30 cells and each move removes a piece, so games
    are short; the cap is well above a full game plus invalid-retry slack so the
    terminal reward is never dropped. In tool_call mode (#1201) the game is
    distilled as per-turn-fresh examples via run_toolcall_episode.
    """
    if tool_calling_enabled():
        return run_toolcall_episode(
            game_name="clobber",
            game_id=game_id,
            env_endpoint=env_endpoint,
            opponent_payload=_OPPONENT_PAYLOAD,
            max_turn=max_turn,
            rules_prompt=get_game_rules_prompt("clobber"),
            expert_action_fn=get_expert_action,
            obs_transform=_ClobberObsTransform(),
            policy=ClobberMemoryPolicy(),
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

    transform = _ClobberObsTransform()
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": transform(observation)},
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
        messages.append({"role": "user", "content": transform(observation)})
    else:
        print(f"[env] max_turn={max_turn} reached (game {game_id})")

    return messages, final_reward
