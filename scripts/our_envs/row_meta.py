"""Per-row label metadata side-channel for Q-gap row weighting (research R3).

The expert/labeler that decides a turn's game_action sets ``(qgap, forced)``
immediately before returning it; the self-play emitter pops the slot and attaches
the raw metadata to the SFT row. The merge/train stage turns it into a row weight
(kept in ONE place so the weighting policy is centralised).

Thread-local so it is correct whether pvp_selfplay runs its workers as processes
(each process has its own module state) OR threads (each thread gets its own slot).
``pop_meta`` CLEARS the slot, so a labeler that forgets to ``set_meta`` yields
neutral metadata rather than a stale value carried over from the previous row.

Contract (do not change without updating all callers):
  - qgap: float >= 0. The value gap between the chosen action and the best
    alternative, in the teacher's OWN units (negamax score / CFR EV / regret).
    0.0 = a genuine tie (any legal action equally good). None = teacher did not
    report a gap (treated as neutral; weight 1.0).
  - forced: True when exactly one legal action existed (a pass/only-move turn
    carries no strategic signal; merge prunes these to a floor fraction).
"""
from __future__ import annotations

import threading

_local = threading.local()


def set_meta(qgap: "float | None" = None, forced: bool = False) -> None:
    """Record the current turn's label metadata. Call right before returning the
    chosen action id from a labeler/expert. Never raises."""
    _local.qgap = qgap
    _local.forced = bool(forced)


def pop_meta() -> dict:
    """Return {"qgap", "forced"} for the row just labeled and CLEAR the slot.
    Neutral ({None, False}) when nothing was set. Never raises."""
    q = getattr(_local, "qgap", None)
    f = getattr(_local, "forced", False)
    _local.qgap = None
    _local.forced = False
    return {"qgap": q, "forced": f}
