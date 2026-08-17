"""Validator-style observation reformatters.

The env-server (phoenixbeaudry/game:mcts-api) returns observation strings
that match the validator's PvP eval format for gin_rummy but diverge for
liars_dice (missing "Dice per player", "Players", "Current player",
"You can:" lines) and leduc_poker (different field names: "Pot size" vs
"Pot", "Current round" vs "Round", "Round X betting" vs "Round X
actions"). These helpers transform the env-server string into the exact
multi-line layout that BaseGameAgent.format_state() produces validator-
side, so the user message the model sees during SFT distillation is
byte-aligned with what the validator hands it at inference time.

Helpers below are pure string transforms (no pyspiel dependency). For
GR the body already matches validator's observation_string output, so
only the envelope wrap is applied.

Use ``reformat_to_pvp(raw_obs, game_name, player_id)`` as the entry
point; the dispatcher selects the right per-game reformatter.
"""

import re


_LEGAL_ACTIONS_RE = re.compile(r"(\n*Legal Actions:)", re.IGNORECASE)
_LEGAL_BLOCK_RE = re.compile(
    r"Legal Actions:\s*\n(.*?)(?=\n\s*Your choice|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def wrap_pvp_format(observation: str, player_id: int = 0) -> str:
    """Wrap a raw env-server observation in the validator's PvP envelope.

    Splits at the "Legal Actions:" anchor, treats everything before it as
    the state-description body, and inserts the validator's framing:

      Current State:
      {body}

      You are Player {player_id}.
      Legal Actions:
      {legal actions block}

      Your choice (ID only):

    Idempotent — already-wrapped observations are returned unchanged. The
    fallback case (no "Legal Actions:" anchor parseable) returns the raw
    observation so the pipeline degrades gracefully rather than dropping
    actions.
    """
    if not observation:
        return observation
    if observation.startswith("Current State:"):
        return observation

    match = _LEGAL_ACTIONS_RE.search(observation)
    if not match:
        return observation

    body = observation[:match.start()].rstrip()
    # rest = "Legal Actions:\n  N -> ...\nYour choice (action ID only):".
    # Dedent the action lines and normalize the trailer so the wrapped GR
    # prompt matches the validator byte-for-byte; the state body (the ASCII
    # hand grid) is left untouched since the validator uses observation_string
    # verbatim for gin_rummy.
    rest = observation[match.start():].lstrip("\n")
    rest = _normalize_trailer(_dedent_action_lines(rest))

    return (
        f"Current State:\n{body}\n\n"
        f"You are Player {player_id}.\n"
        f"{rest}"
    )


def _dedent_action_lines(block: str) -> str:
    """Strip leading whitespace from each line so action lines read
    ``N -> label`` exactly like the validator's generate_user_prompt output.

    The env server indents action lines as ``  N -> label`` (two spaces);
    the validator emits them flush-left. (Validator: pvp/agents.py builds
    ``f"{action} -> {action_str}"`` with no indent.)
    """
    return re.sub(r"^[ \t]+", "", block, flags=re.MULTILINE)


def _normalize_trailer(text: str) -> str:
    """Match the validator's choice suffix byte-for-byte.

    The env server emits ``Your choice (action ID only):``; the validator's
    pvp/agents.py emits ``Your choice (ID only):``.
    """
    return text.replace("Your choice (action ID only):", "Your choice (ID only):")


def _extract_legal_and_trailer(raw_obs: str) -> "tuple[str, str] | None":
    """Pull (legal_actions_block, trailer) from raw env-server observation.

    Returns None when the "Legal Actions:" anchor is missing — callers
    treat that as "use raw observation unchanged" rather than crashing.
    The action block is dedented and the trailer normalized so both match
    the validator's PvP user-prompt byte-for-byte.
    """
    match = _LEGAL_BLOCK_RE.search(raw_obs)
    if not match:
        return None
    legal_block = _dedent_action_lines(match.group(1).rstrip())
    trailer = raw_obs[match.end():].strip()
    if not trailer:
        trailer = "Your choice (ID only):"
    else:
        trailer = _normalize_trailer(trailer)
    return legal_block, trailer


# ---------------------------------------------------------------------------
# Liar's Dice
# ---------------------------------------------------------------------------

def reformat_ld_to_pvp(raw_obs: str, player_id: int = 0) -> str:
    """Reformat env-server LD observation to validator's LiarsDiceAgent body.

    Validator emits richer state lines than the env-server does. This
    helper recovers the missing lines from values that ARE present in
    the raw observation:

      - "Dice per player: N"          (derived from len(your_dice))
      - "Players: N"                  (total_dice // dice_per_player)
      - "Current player: Player {pid}"
      - guidance: "You can: ..." or "No bid yet ..."

    Bid line is reformatted from `'Current bid: "2-4"'` to the validator's
    `'Current bid: "2-4" (at least 2 dice showing 4 across all players)'`.
    """
    if not raw_obs:
        return raw_obs

    dice_match = re.search(r"Your dice:\s*\[([^\]]+)\]", raw_obs)
    total_match = re.search(r"Total dice in game:\s*(\d+)", raw_obs)
    bid_match = re.search(r'Current bid:\s*"(\d+)-(\d+)"', raw_obs)
    legal_trailer = _extract_legal_and_trailer(raw_obs)

    if not dice_match or not total_match or legal_trailer is None:
        return raw_obs

    legal_block, trailer = legal_trailer

    try:
        dice = [int(x.strip()) for x in dice_match.group(1).split(",")]
    except ValueError:
        return raw_obs

    total_dice = int(total_match.group(1))
    num_dice = len(dice)
    num_players = max(1, total_dice // num_dice) if num_dice else 2

    lines = [
        f"Your dice: {dice} (showing: {', '.join(map(str, dice))})",
        f"Dice per player: {num_dice}",
        f"Total dice in game: {total_dice}",
        f"Players: {num_players}",
        f"Current player: Player {player_id}",
    ]
    if bid_match:
        quantity, face = bid_match.group(1), bid_match.group(2)
        lines.append(
            f'\nCurrent bid: "{quantity}-{face}" '
            f"(at least {quantity} dice showing {face} across all players)"
        )
        lines.append("You can: (1) Make a higher bid, or (2) Call 'Liar'")
    else:
        lines.append("No bid yet - you must make the first bid")

    body = "\n".join(lines)
    return (
        f"Current State:\n{body}\n\n"
        f"You are Player {player_id}.\n"
        f"Legal Actions:\n{legal_block}\n\n"
        f"{trailer}"
    )


# ---------------------------------------------------------------------------
# Leduc Poker
# ---------------------------------------------------------------------------

_LP_SUITS = ("♠", "♥")  # spades, hearts — matches validator agents.py


def _lp_card_name(raw_card: str) -> str:
    """Return the env-server card token in the validator's {rank}{suit} form.

    The env server already renders the card with its suit glyph (e.g.
    "K♠"/"K♥"), derived from the same pyspiel card_id the
    validator uses, so we PRESERVE it verbatim to stay byte-identical with
    LeducPokerAgent._card_name (suit = card_id % 2 -> ♠/♥). Only
    when the glyph is absent (unexpected env-server variant) do we fall back
    to spades; suit does not affect Leduc hand strength so the rank-only
    fallback is harmless for play while keeping the prompt well-formed.

    The previous version stripped to the rank char and ALWAYS appended
    spades, corrupting every hearts card (K♥ -> K♠) relative to
    what the validator feeds the model at eval time.
    """
    if not raw_card or raw_card.startswith("(not"):
        return "(not dealt yet)"
    rank_char = raw_card[0].upper()
    if rank_char not in {"J", "Q", "K", "A"}:
        return raw_card
    if any(suit in raw_card for suit in _LP_SUITS):
        return raw_card
    return f"{rank_char}{_LP_SUITS[0]}"


_LP_BET_TOKEN = {
    "0": "Fold", "f": "Fold", "fold": "Fold",
    "1": "Call", "c": "Call", "call": "Call", "check": "Call",
    "2": "Raise", "r": "Raise", "raise": "Raise", "bet": "Raise",
}


def _normalize_lp_betting(raw: str) -> str:
    """Normalise the env-server's Round-N betting field into the validator's
    token format. G.O.D #1240 (2026-06-24) re-added betting history to the eval
    prompt as ``Round 1 actions: Fold, Call, Raise`` (LeducPokerAgent._parse_betting:
    0->Fold, 1->Call, 2->Raise, comma-joined). We map whatever the env-server emits
    (numeric / letter / word) onto those exact tokens so our SFT betting lines are
    byte-aligned with eval; unknown tokens pass through. Returns '' for an empty
    sequence (the caller then omits the line). NOTE: the env-server's raw betting
    token format should be probe-confirmed against a real LP observation."""
    toks = re.split(r"[,\s]+", raw.strip())
    out = [_LP_BET_TOKEN.get(t.lower(), t) for t in toks if t]
    return ", ".join(out)


def reformat_lp_to_pvp(raw_obs: str, player_id: int = 0) -> str:
    """Reformat env-server LP observation to validator's LeducPokerAgent body.

    Same fields, different field labels:
      - "Pot size: N"          -> "Pot: N chips"
      - "Current round: X/2"   -> "Round: X/2"
      - "Round X betting: ..." -> "Round X actions: ..."
      - "Hand: Pair"           -> "Hand: PAIR"

    Cards reformatted via _lp_card_name (J/Q/K/A + spades glyph). All
    extraction is regex-tolerant so a missing optional field (no public
    card yet, no Round 2 actions) does not abort the transform.
    """
    if not raw_obs:
        return raw_obs

    legal_trailer = _extract_legal_and_trailer(raw_obs)
    if legal_trailer is None:
        return raw_obs
    legal_block, trailer = legal_trailer

    def _find(pattern: str, default=None):
        m = re.search(pattern, raw_obs)
        return m.group(1) if m else default

    private_raw = _find(r"Your card:\s*(\S+)")
    if private_raw is None:
        return raw_obs

    public_raw = _find(r"Public card:\s*(\S+)")
    has_pair = bool(re.search(r"Hand:\s*Pair\b", raw_obs, re.IGNORECASE))
    round_str = _find(r"(?:Current\s+round|Round):\s*(\d+)\s*/\s*\d+", "1")
    pot_str = _find(r"(?:Pot\s+size|Pot):\s*(\d+)", "0")
    our_chips = _find(r"Your chips:\s*(\d+)", "100")
    opp_chips = _find(r"Opponent chips:\s*(\d+)", "100")

    def _betting(label_alias: str) -> str:
        m = re.search(
            rf"(?:{label_alias}\s+betting|{label_alias}\s+actions):\s*(.+)",
            raw_obs,
        )
        if not m:
            return ""
        return m.group(1).strip().rstrip("\n")

    r1_actions = _normalize_lp_betting(_betting("Round 1"))
    r2_actions = _normalize_lp_betting(_betting("Round 2"))

    lines: list[str] = []
    lines.append(f"Your card: {_lp_card_name(private_raw)}")
    if public_raw and public_raw != "(none)":
        lines.append(f"Public card: {_lp_card_name(public_raw)}")
        if has_pair:
            lines.append("Hand: PAIR")
    lines.append(f"Round: {round_str}/2")
    lines.append(f"Pot: {pot_str} chips")
    lines.append(f"Your chips: {our_chips}")
    lines.append(f"Opponent chips: {opp_chips}")
    if r1_actions:
        lines.append(f"Round 1 actions: {r1_actions}")
    if r2_actions:
        lines.append(f"Round 2 actions: {r2_actions}")

    body = "\n".join(lines)
    return (
        f"Current State:\n{body}\n\n"
        f"You are Player {player_id}.\n"
        f"Legal Actions:\n{legal_block}\n\n"
        f"{trailer}"
    )


# ---------------------------------------------------------------------------
# Gin Rummy
# ---------------------------------------------------------------------------

def reformat_gr_to_pvp(raw_obs: str, player_id: int = 0) -> str:
    """Reformat env-server GR observation.

    The GR body already matches validator's observation_string output
    (validator's GinRummyAgent.format_state returns that string directly).
    Only the validator envelope wrap is needed; see wrap_pvp_format.
    """
    return wrap_pvp_format(raw_obs, player_id=player_id)


# ---------------------------------------------------------------------------
# Othello
# ---------------------------------------------------------------------------

def reformat_othello_to_pvp(raw_obs: str, player_id: int = 0) -> str:
    """Reformat env-server Othello observation.

    Like gin_rummy, the Othello body is OpenSpiel observation_string()
    (= ToString(): the "Black (x) to play:" header, the "  a b c d e f g h  "
    label line, and the 8 board rows) verbatim — the validator's OthelloAgent
    falls back to observation_string for format_state — so only the validator
    PvP envelope wrap is needed.
    """
    return wrap_pvp_format(raw_obs, player_id=player_id)


def reformat_clobber_to_pvp(raw_obs: str, player_id: int = 0) -> str:
    """Reformat env-server Clobber observation.

    Like othello, the Clobber body is OpenSpiel observation_string() (= ToString():
    the "{rownum}{cells}" board rows + the " {column letters}" footer) verbatim —
    the validator's ClobberAgent uses observation_string for format_state (with a
    "You play {colour}." prefix the obs-transform adds) — so only the validator
    PvP envelope wrap is needed here.
    """
    return wrap_pvp_format(raw_obs, player_id=player_id)


# ---------------------------------------------------------------------------
# Goofspiel
# ---------------------------------------------------------------------------

def reformat_goofspiel_to_pvp(raw_obs: str, player_id: int = 0) -> str:
    """Reformat env-server Goofspiel observation into the PvP envelope.

    Unlike the other games, the env-server goofspiel obs carries the state
    fields (current point card, BOTH players' hands, remaining prizes, scores,
    win sequence) but emits NO "Legal Actions:" block — so we BUILD it from our
    hand: bidding card value c is action id c-1. We also DROP the opponent's
    hand to match the validator's imperfect-information view (you never see the
    opponent's cards). Field-reconstructing, like reformat_ld_to_pvp. Falls back
    to the raw obs (never crashes) if the point card / our hand can't be read.

    Byte-aligned to the validator's native OpenSpiel observation_string via
    probe_pvp_formats.py ground truth (2026-06-21): the body lines are
    "You are Player {pid} (P{pid}).", "Current player: {pid}",
    "Current point card: {prize}", "Remaining Point Cards: {future prize values,
    ascending, CONCATENATED no separators}", "Points: {p0} {p1}",
    "P{pid} hand: {space-separated}", "Win sequence: {winner player indices}".
    Action labels are "{id} -> [P{pid}]Bid: {card}". Earlier this emitted an
    env-server-derived body ("Current point card/Scores - you/Bid {c}") that did
    NOT match the eval -> the model rambled and forfeited 100/100 (PR #1217).
    """
    if not raw_obs:
        return raw_obs
    pm = re.search(r"You are Player (\d+)", raw_obs)
    pid = int(pm.group(1)) if pm else player_id

    pc = re.search(r"Current point card:\s*(\d+)", raw_obs)
    hand_m = re.search(rf"P{pid}\s+hand:\s*([\d ]+)", raw_obs)
    if not pc or not hand_m:
        return raw_obs  # can't build legal actions -> passthrough (never crash)
    try:
        hand = sorted(int(x) for x in hand_m.group(1).split())
    except ValueError:
        return raw_obs
    if not hand:
        return raw_obs

    # Remaining Point Cards: future prize values, ascending, CONCATENATED with no
    # separators (OpenSpiel native: {1,3,4,5} -> "1345", N=13 -> "10111213").
    # NB: anchor the value to the SAME line ([ \t]* not \s*, [^\n]* not [^\n]+).
    # On the final decision turn the remaining-cards value is empty; a \s* here
    # would swallow the newline and bleed the next "Player 0: N points" line into
    # the field (-> "Remaining Point Cards: 09"). The validator shows it empty.
    rem = re.search(r"Remaining [Pp]oint [Cc]ards:[ \t]*([^\n]*)", raw_obs)
    rem_vals = sorted(int(x) for x in re.findall(r"\d+", rem.group(1))) if rem else []
    # Points: absolute player order p0 p1 (NOT relative you/opponent).
    scores = dict(re.findall(r"Player (\d+):\s*(\d+) points", raw_obs))
    p0, p1 = scores.get("0", "0"), scores.get("1", "0")
    # Win sequence: space-separated winner player indices.
    seq = re.search(r"Win sequence:\s*([^\n(]*)", raw_obs)
    seq_txt = " ".join(re.findall(r"\d+", seq.group(1))) if seq else ""

    body = "\n".join([
        f"You are Player {pid} (P{pid}).",
        f"Current player: {pid}",
        f"Current point card: {pc.group(1)}",
        "Remaining Point Cards: " + "".join(str(v) for v in rem_vals),
        f"Points: {p0} {p1}",
        f"P{pid} hand: {' '.join(str(c) for c in hand)}",  # OUR hand only (imp-info)
        f"Win sequence: {seq_txt}",
    ])
    legal_block = "\n".join(f"{c - 1} -> [P{pid}]Bid: {c}" for c in hand)
    return (
        "Current State:\n" + body + "\n\n"
        f"You are Player {pid}.\n"
        f"Legal Actions:\n{legal_block}\n\n"
        "Your choice (ID only):"
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_REFORMATTERS = {
    "liars_dice":  reformat_ld_to_pvp,
    "leduc_poker": reformat_lp_to_pvp,
    "gin_rummy":   reformat_gr_to_pvp,
    "othello":     reformat_othello_to_pvp,
    "goofspiel":   reformat_goofspiel_to_pvp,
    "clobber":     reformat_clobber_to_pvp,
}


def reformat_to_pvp(raw_obs: str, game_name: str, player_id: int = 0) -> str:
    """Per-game dispatcher. Unknown game_name returns raw_obs unchanged."""
    fn = _REFORMATTERS.get(game_name)
    return fn(raw_obs, player_id=player_id) if fn else raw_obs
