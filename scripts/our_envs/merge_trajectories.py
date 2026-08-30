"""Merge multiple per-env HF DatasetDicts into a single combined dataset.

Used by the multi-env SFT path: when validator assigns 2+ environments
per task (new R1+ structure), we generate each env's trajectories
separately, then merge into a single combined dataset and run one
tokenize + train pass. This avoids catastrophic forgetting that would
happen with sequential per-env training (train env A -> train env B
forgets A) while keeping a single SFT training session.

Per-env REBALANCE (optional): without it, an env whose example count is
inflated by the sliding-window (e.g. goofspiel 10k games -> ~119k windows)
drowns the others (e.g. intercode synth ~3k) in a joint run, so the model
barely learns the small env. With --target_per_env N each env is leveled to
~N examples: larger envs are randomly DOWNSAMPLED to N (no repetition, they
have plenty of unique windows); smaller envs are UPSAMPLED toward N but never
past --max_upsample x their size (a hard cap so a low-unique env like
intercode, ~1.6k unique tasks, is not repeated into memorization/overfit).
This generalizes to ANY env set (2/3/4/6 envs), not just one pair.

Usage:
  python -m our_envs.merge_trajectories \\
      --input_paths /path/to/sft_env_liars_dice /path/to/sft_env_gin_rummy \\
      --output_path /path/to/sft_env_combined \\
      --target_per_env 30000 --max_upsample 3 --shuffle_seed 42

Output: HF DatasetDict at output_path with train + validation splits,
each containing the (balanced) concatenation of all input splits (shuffled).
"""

import argparse
import json
import math
import os
import random

try:
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
except ImportError:  # pragma: no cover
    # The heavy `datasets` dep is absent in lightweight/test contexts. The pure
    # planning helpers below (_row_floor_plan, _qgap_plan, flag readers) need no
    # such dep and stay importable; production images always have `datasets`, so
    # main() / the Dataset wrappers run exactly as before there. Left as None so
    # any wrapper that needs a real Dataset fails loudly only when actually run.
    Dataset = DatasetDict = concatenate_datasets = load_from_disk = None


def _flag_on(name: str, default: bool = False) -> bool:
    """Env flag reader: treats unset -> ``default`` and "0"/"false"/"no"/"off"/""
    as OFF (everything else ON). Shared discipline so an un-flagged run is the
    current behaviour byte-for-byte."""
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def _float_env(name: str, default: float) -> float:
    """Parse a float env var; unset/blank/malformed -> ``default``. Never raises."""
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --- R3 Q-gap weighting + P2 row-floor: PURE planners (stdlib only) ----------
# These take/return plain Python so they are unit-testable without `datasets`.
# The thin Dataset wrappers underneath apply the plan via the HF API.

_META_COLS = ("_qgap", "_forced", "row_weight")


def _row_floor_plan(lengths: "list[int]", frac: float) -> "list[int]":
    """P2 row-floor target lengths. Any non-empty part shorter than
    ``frac * max(lengths)`` is RAISED to that floor (never reduced); every other
    part keeps its length. frac<=0 / empty -> unchanged. Guarantees target>=len
    for all parts (the oversampler must never drop data)."""
    lengths = list(lengths)
    if not lengths or frac <= 0:
        return lengths
    mx = max(lengths)
    floor = int(frac * mx)
    return [floor if (0 < n < floor) else n for n in lengths]


def _qgap_plan(qgaps: "list", forced: "list", beta: float, forced_keep: float,
               seed: int) -> "tuple[list[int], list[float]]":
    """R3 weighting plan. Returns (keep_indices, row_weights) where row_weights is
    index-aligned with keep_indices and renormalised to MEAN 1.0.

      * near-tie rows (qgap == 0 and NOT forced) are DROPPED (no strategic signal).
      * forced rows (exactly one legal action) are PRUNED to a floor fraction
        ``forced_keep`` of the forced rows (deterministic subsample by ``seed``).
      * weight = clamp(exp(qgap / beta), 0.25, 4.0); qgap None (unknown) -> 1.0.
    Pure; never raises (beta<=0 falls back to 1.0)."""
    n = len(qgaps)
    if beta is None or beta <= 0:
        beta = 1.0
    forced_idx = [i for i in range(n) if forced[i]]
    keep_forced = set()
    if forced_idx:
        k = int(forced_keep * len(forced_idx) + 0.5)  # round-half-up, deterministic
        k = max(0, min(len(forced_idx), k))
        shuffled = list(forced_idx)
        random.Random(seed).shuffle(shuffled)
        keep_forced = set(shuffled[:k])
    keep: "list[int]" = []
    for i in range(n):
        if forced[i]:
            if i in keep_forced:
                keep.append(i)
            continue
        q = qgaps[i]
        if q is not None and float(q) == 0.0:
            continue  # near-tie: dropped
        keep.append(i)
    raw: "list[float]" = []
    for i in keep:
        q = qgaps[i]
        if q is None:
            raw.append(1.0)
            continue
        try:
            w = math.exp(float(q) / beta)
        except OverflowError:
            w = 4.0
        raw.append(min(4.0, max(0.25, w)))
    if raw:
        mean = sum(raw) / len(raw)
        if mean > 0:
            raw = [w / mean for w in raw]
    return keep, raw


def _strip_meta_cols(ds: "Dataset") -> "Dataset":
    """Remove R3 side columns so a QGAP-OFF merge output is identical to today
    (game-env rows now always carry ``_qgap``/``_forced``; intercode rows do not,
    so stripping also re-aligns the schema before the cross-env concat)."""
    cols = [c for c in _META_COLS if c in ds.column_names]
    return ds.remove_columns(cols) if cols else ds


def _ensure_meta_cols(ds: "Dataset") -> "Dataset":
    """Add neutral ``_qgap``/``_forced`` columns to any part missing them (e.g.
    an intercode part) so the QGAP-ON cross-env concat has a uniform schema."""
    n = len(ds)
    if "_qgap" not in ds.column_names:
        ds = ds.add_column("_qgap", [None] * n)
    if "_forced" not in ds.column_names:
        ds = ds.add_column("_forced", [False] * n)
    return ds


def _apply_row_floor(parts: "list", labels: "list[str]", frac: float,
                     seed: int) -> "list":
    """Oversample (cyclic repeat, NEVER drop) any part below the P2 row-floor."""
    lengths = [len(p) for p in parts]
    targets = _row_floor_plan(lengths, frac)
    out = []
    for p, n, t, lbl in zip(parts, lengths, targets, labels):
        if t > n and n > 0:
            reps = (t + n - 1) // n
            p2 = concatenate_datasets([p] * reps).select(range(t))
            print(f"[merge_trajectories] row-floor {lbl}: {n} -> {t} "
                  f"(cyclic repeat, frac={frac})", flush=True)
            out.append(p2)
        else:
            out.append(p)
    return out


def _apply_qgap_weighting(ds: "Dataset", beta: float, forced_keep: float,
                          seed: int) -> "Dataset":
    """Apply the R3 plan to a (concatenated) train split: drop near-ties, prune
    forced rows, and attach the renormalised ``row_weight`` column (dropping the
    now-consumed ``_qgap``/``_forced`` columns)."""
    qgaps = list(ds["_qgap"]) if "_qgap" in ds.column_names else [None] * len(ds)
    forced = list(ds["_forced"]) if "_forced" in ds.column_names else [False] * len(ds)
    keep, weights = _qgap_plan(qgaps, forced, beta, forced_keep, seed)
    kept = _strip_meta_cols(ds.select(keep))
    kept = kept.add_column("row_weight", weights)
    print(f"[merge_trajectories] QGAP weighting: {len(ds)} -> {len(kept)} rows "
          f"(beta={beta} forced_keep={forced_keep}); row_weight mean="
          f"{(sum(weights) / len(weights)) if weights else 0.0:.4f}", flush=True)
    return kept


def _env_label(path: str) -> str:
    """Best-effort env name from a per-env dataset path for logging."""
    base = os.path.basename(path.rstrip("/"))
    return base.split("_")[-1] if "_" in base else base


def _parse_per_env_targets(raw: str) -> dict:
    """Parse the --per_env_targets JSON map (env-name -> target). Never raises:
    a malformed value degrades to {} so the uniform --target_per_env still applies."""
    if not raw or not raw.strip():
        return {}
    try:
        m = json.loads(raw)
        return {str(k): int(v) for k, v in m.items()} if isinstance(m, dict) else {}
    except (ValueError, TypeError):
        print(f"[merge_trajectories] WARN: bad --per_env_targets {raw!r}; ignoring", flush=True)
        return {}


def _resolve_target(path: str, per_env_map: dict, default: int) -> int:
    """Per-env target skew: if any per_env_map key is a substring of the dataset
    basename, use its target; else the uniform default. Substring-match (not
    _env_label) so multi-token names like 'gin_rummy' resolve, which the last-
    underscore-segment label would truncate to 'rummy'."""
    if per_env_map:
        base = os.path.basename(path.rstrip("/"))
        for k, v in per_env_map.items():
            if k and k in base:
                return v
    return default


def _ensure_tools_col(ds: "Dataset") -> "Dataset":
    """Give every part a ``tools`` column so parts from different envs concatenate.

    Generators that emit tool-call rows attach a per-row ``tools`` JSON string
    (train_sft_env json.loads it before apply_chat_template).  A generator that
    does not would produce a part whose Arrow features disagree, and
    ``concatenate_datasets`` refuses the whole merge rather than just that part
    -- one env without the column would take the entire task down.  An empty
    string means "no tools block", which is what ``example.get("tools") or None``
    already treats as absent.
    """
    try:
        if len(ds) == 0 or "tools" in ds.column_names:
            return ds
        return ds.add_column("tools", [""] * len(ds))
    except Exception as exc:
        print(f"[merge_trajectories] could not normalise tools column: {exc}", flush=True)
        return ds


def _balance_split(ds: "Dataset", target: int, max_upsample: float, seed: int,
                   label: str) -> "Dataset":
    """Level one env's train split toward `target` examples.

    - len >= target  -> random downsample to target (no repetition).
    - len <  target  -> upsample (repeat+shuffle) to min(target, max_upsample*len);
      the max_upsample cap prevents a low-unique env from being repeated into
      overfit (e.g. intercode's ~1.6k unique tasks must not be blown up to 30k).
    Never raises; returns the dataset unchanged when target is unset/<=0.
    """
    n = len(ds)
    if not target or target <= 0 or n == 0:
        return ds
    if n >= target:
        out = ds.shuffle(seed=seed).select(range(target))
        print(f"[merge_trajectories] balance {label}: {n} -> {target} (downsampled)", flush=True)
        return out
    goal = min(target, int(max_upsample * n))
    if goal <= n:
        print(f"[merge_trajectories] balance {label}: {n} kept (target {target}, "
              f"upsample capped at {max_upsample}x)", flush=True)
        return ds
    reps = (goal + n - 1) // n  # ceil
    out = concatenate_datasets([ds] * reps).shuffle(seed=seed).select(range(goal))
    print(f"[merge_trajectories] balance {label}: {n} -> {goal} "
          f"(upsampled {goal / n:.1f}x, cap {max_upsample}x, target {target})", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_paths", nargs="+", required=True,
                   help="One or more HF DatasetDict paths to merge.")
    p.add_argument("--output_path", required=True)
    p.add_argument("--shuffle_seed", type=int, default=42)
    p.add_argument("--target_per_env", type=int, default=0,
                   help="Per-env target example count (0=off). Larger envs are "
                        "downsampled to this; smaller envs upsampled toward it.")
    p.add_argument("--max_upsample", type=float, default=3.0,
                   help="Hard cap on the upsample factor for a small env "
                        "(guards low-unique envs like intercode from overfit).")
    p.add_argument("--replay_paths", nargs="*", default=[],
                   help="Optional THIN replay datasets (non-scored envs, R2+ "
                        "continuation anti-forgetting) balanced to "
                        "--replay_target_per_env (a small slice) and concatenated "
                        "alongside --input_paths so the model refreshes prior-round "
                        "skills without those envs crowding the scored ones.")
    p.add_argument("--replay_target_per_env", type=int, default=0,
                   help="Per-env target for --replay_paths (0 = passthrough at the "
                        "generated, already-thin size).")
    p.add_argument("--per_env_targets", type=str, default="",
                   help="Optional JSON map {env_name: target} that OVERRIDES "
                        "--target_per_env for matching envs (substring match on the "
                        "dataset path). Lets one env be weighted heavier than others "
                        "in a joint run, e.g. '{\"clobber\": 45000, \"gin_rummy\": 15000}'. "
                        "Unlisted envs fall back to --target_per_env.")
    args = p.parse_args()
    per_env_map = _parse_per_env_targets(args.per_env_targets)
    if per_env_map:
        print(f"[merge_trajectories] per-env target overrides: {per_env_map}", flush=True)

    if not args.input_paths:
        raise SystemExit("--input_paths must include at least one path")

    print(f"[merge_trajectories] merging {len(args.input_paths)} datasets "
          f"(target_per_env={args.target_per_env or 'off'} max_upsample={args.max_upsample}):",
          flush=True)
    for i, p_ in enumerate(args.input_paths):
        print(f"  [{i}] {p_}", flush=True)

    # Round-14: be tolerant of missing inputs. If one env's gen process
    # crashed silently (e.g., intercode hit by destructive gold killing
    # Python), we'd rather merge the surviving datasets than abort the
    # whole pipeline. Each missing input is warned, not raised.
    def _load_best_effort(paths: list, tag: str) -> "list[tuple[str, DatasetDict]]":
        out: list = []
        for p_ in paths:
            try:
                out.append((p_, load_from_disk(p_)))
            except FileNotFoundError as exc:
                print(f"[merge_trajectories] WARN: skipping missing {tag} {p_}: {exc}", flush=True)
            except Exception as exc:
                print(f"[merge_trajectories] WARN: skipping unreadable {tag} {p_}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
        return out

    kept = _load_best_effort(args.input_paths, "input")
    if not kept:
        raise SystemExit(
            f"All {len(args.input_paths)} inputs missing or unreadable; nothing to merge."
        )

    # ORPO/preference passthrough: a {prompt, chosen, rejected} dataset (from SWE_ORPO gen)
    # must NOT go through the SFT balancing + R3-meta (_qgap/_forced/row_weight) path — that
    # assumes the {messages} schema and would corrupt the preference columns. SWE-ORPO is a
    # single-env preference set, so just concatenate the inputs verbatim + shuffle + save.
    def _is_pref(d) -> bool:
        tr = d["train"] if "train" in d else None
        cols = tr.column_names if tr is not None else []
        return "chosen" in cols and "rejected" in cols

    if any(_is_pref(d) for _, d in kept):
        tr_parts = [d["train"] for _, d in kept if "train" in d]
        va_parts = [d["validation"] for _, d in kept if "validation" in d]
        merged: dict = {}
        if tr_parts:
            tr = concatenate_datasets(tr_parts) if len(tr_parts) > 1 else tr_parts[0]
            merged["train"] = tr.shuffle(seed=args.shuffle_seed)
        if va_parts:
            merged["validation"] = (
                concatenate_datasets(va_parts) if len(va_parts) > 1 else va_parts[0]
            )
        DatasetDict(merged).save_to_disk(args.output_path)
        print(
            f"[merge_trajectories] ORPO/preference passthrough -> {args.output_path} "
            f"train={len(merged.get('train', []))} validation={len(merged.get('validation', []))}",
            flush=True,
        )
        return

    replay_kept = _load_best_effort(args.replay_paths, "replay") if args.replay_paths else []
    if replay_kept:
        print(f"[merge_trajectories] + {len(replay_kept)} REPLAY datasets "
              f"(target_per_env={args.replay_target_per_env or 'passthrough'}):", flush=True)
        for p_, _ in replay_kept:
            print(f"  (replay) {p_}", flush=True)

    # Balance each env's TRAIN split toward its per-env target before concat.
    # Validation splits are left untouched (small; used only for eval-loss).
    train_parts = []
    train_labels: "list[str]" = []
    for p_, d in kept:
        if "train" in d:
            _tgt = _resolve_target(p_, per_env_map, args.target_per_env)
            train_parts.append(
                _balance_split(d["train"], _tgt, args.max_upsample,
                               args.shuffle_seed, _env_label(p_))
            )
            train_labels.append(_env_label(p_))
    # Thin replay slice (R2+ continuation anti-forgetting): a small per-env target
    # so prior-round envs are refreshed without crowding the scored envs.
    for p_, d in replay_kept:
        if "train" in d:
            train_parts.append(
                _balance_split(d["train"], args.replay_target_per_env, args.max_upsample,
                               args.shuffle_seed, "replay:" + _env_label(p_))
            )
            train_labels.append("replay:" + _env_label(p_))
    # Validation = scored envs only; eval-loss stays on the round's scored mix.
    # R3 side columns never carry weighting on the eval split -> always stripped
    # (also re-aligns schema when only some envs added _qgap/_forced).
    val_parts = [_strip_meta_cols(d["validation"]) for _, d in kept if "validation" in d]
    if not train_parts:
        raise SystemExit("No 'train' split found in any input dataset")

    # P2 row-floor (SFT_ROW_FLOOR_FRAC, default "0" = off): oversample small envs
    # up to a fraction of the largest env (cyclic repeat, never drops). Applied
    # AFTER per-env balance so it lifts any env the balance left below the floor.
    _row_floor_frac = _float_env("SFT_ROW_FLOOR_FRAC", 0.0)
    if _row_floor_frac > 0:
        print(f"[merge_trajectories] P2 row-floor frac={_row_floor_frac}", flush=True)
        train_parts = _apply_row_floor(train_parts, train_labels, _row_floor_frac,
                                       args.shuffle_seed)

    # R3 Q-gap weighting (QGAP_WEIGHTING, default OFF). OFF -> strip the side
    # columns so the concat + output are byte-identical to today. ON -> uniform
    # meta schema, concat, then drop near-ties / prune forced rows / attach the
    # renormalised row_weight column.
    if _flag_on("QGAP_WEIGHTING"):
        _beta = _float_env("QGAP_BETA", 1.0)
        _forced_keep = _float_env("QGAP_FORCED_KEEP", 0.10)
        train_parts = [_ensure_tools_col(_ensure_meta_cols(p)) for p in train_parts]
        merged_train = concatenate_datasets(train_parts).shuffle(seed=args.shuffle_seed)
        merged_train = _apply_qgap_weighting(merged_train, _beta, _forced_keep,
                                             args.shuffle_seed)
    else:
        train_parts = [_ensure_tools_col(_strip_meta_cols(p)) for p in train_parts]
        merged_train = concatenate_datasets(train_parts).shuffle(seed=args.shuffle_seed)
    val_parts = [_ensure_tools_col(p) for p in val_parts]
    merged_val = (
        concatenate_datasets(val_parts).shuffle(seed=args.shuffle_seed)
        if val_parts else None
    )

    dd_out = DatasetDict({"train": merged_train})
    if merged_val is not None:
        dd_out["validation"] = merged_val

    dd_out.save_to_disk(args.output_path)
    print(
        f"[merge_trajectories] saved to {args.output_path} "
        f"train={len(merged_train)} validation={len(merged_val) if merged_val else 0}",
        flush=True,
    )


if __name__ == "__main__":
    main()
