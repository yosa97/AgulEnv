"""Intercode (NL2Bash) SFT data generator.

Unlike the game envs (LD/LP/GR), intercode has no MCTS opponent and no cheap
heuristic expert: the GOLD bash command in the NL2Bash dataset is the expert.
This generator reads the validator-mounted intercode dataset (query, gold) and
formats each task into a single-turn ReAct SFT example whose prompt matches the
eval byte-for-byte (envs.intercode_format), then writes a {"messages": ...}
DatasetDict that the existing tokenize + train path consumes.

Data source: the whitelisted intercode dataset requested via requested_datasets
and mounted at MINER_DATASETS_DIR/<org--name>/. No synthetic env rollout and no
network access — the gold commands are the demonstrations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

from our_envs.intercode_format import (
    build_intercode_sft_example,
    build_intercode_multiturn_example,
    build_intercode_refine_example,
    derive_imperfect_first_attempt,
)
# PR #1201: intercode eval moved to native tool calling (execute_bash/submit,
# per-turn-fresh messages). In tool_call mode each task distills into N
# independent per-turn examples; the ReAct builders above remain the legacy
# (bare_id-era) surface.
from our_envs.pvp_tool_calling import tool_calling_enabled
from our_envs.intercode_tool_format import (
    build_tool_examples_single,
    build_tool_examples_multiturn,
    build_tool_examples_refine,
)


VALIDATION_RATIO = 0.01

# IC-2: fraction of pipeline golds turned into a refine-to-exact trajectory
# (execute[gold-minus-final-pipe] -> obs -> execute[gold] -> obs -> submit) so the
# model learns to refine rather than submit its first imperfect command. Selection
# is deterministic per gold (stable hash), so reruns are reproducible. Set to 0 to
# disable. Only applies where derive_imperfect_first_attempt(gold) is not None.
_REFINE_FRAC = float(os.environ.get("INTERCODE_REFINE_FRAC", "0.4"))


def _refine_selected(gold: str) -> bool:
    """Deterministic per-gold selection for the IC-2 refine fraction."""
    if _REFINE_FRAC <= 0.0:
        return False
    if _REFINE_FRAC >= 1.0:
        return True
    h = int(hashlib.md5(gold.encode("utf-8")).hexdigest()[:8], 16) % 1000
    return h < int(_REFINE_FRAC * 1000)

# Candidate column names across NL2Bash dataset variants. "gold"/"cmd"/"command"
# come first so a column holding the bash command wins over a stdout/"output"
# column when both are present.
# NOTE: the whitelisted dataset gradients-io-tournaments/intercode_bigcode_
# combined_12k stores each row as {"instruction": ..., "response": ...} (verified
# via the HF dataset card), so "instruction" + "response" MUST be in these tuples
# or every row is dropped. The eval's own baked nl2bash JSON uses query/gold;
# both naming schemes are covered here.
_QUERY_KEYS = ("query", "instruction", "nl", "question", "prompt", "input", "intent")
_GOLD_KEYS = ("gold", "cmd", "command", "bash", "gold_cmd", "response", "code", "answer", "output")


def _find_intercode_dir() -> "Path | None":
    """Locate the mounted intercode dataset dir (subdir name contains 'intercode')."""
    parent_raw = os.environ.get("MINER_DATASETS_DIR")
    names = os.environ.get("MINER_DATASETS", "")
    if not parent_raw or not names.strip():
        return None
    parent = Path(parent_raw)
    for entry in names.split(","):
        entry = entry.strip()
        if entry and "intercode" in entry.lower():
            d = parent / entry
            if d.exists():
                return d
    return None


def _load_rows(path: Path) -> list[dict]:
    """Best-effort load mounted dataset rows (save_to_disk / parquet / jsonl)."""
    try:
        if (path / "dataset_info.json").exists() or any(path.glob("*.arrow")):
            ds = load_from_disk(str(path))
            if isinstance(ds, DatasetDict):
                rows: list[dict] = []
                for split in ds:
                    rows.extend(ds[split])
                return rows
            return list(ds)
    except Exception:
        pass
    try:
        parquets = list(path.rglob("*.parquet"))
        if parquets:
            from datasets import load_dataset
            ds = load_dataset("parquet", data_files=[str(p) for p in parquets], split="train")
            return list(ds)
    except Exception:
        pass
    rows = []
    for jl in list(path.rglob("*.jsonl")) + list(path.rglob("*.json")):
        try:
            with open(jl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        f.seek(0)
                        data = json.load(f)
                        if isinstance(data, list):
                            rows.extend(d for d in data if isinstance(d, dict))
                        break
        except Exception:
            continue
    return rows


def _extract(row: dict, keys: tuple) -> str:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


# Heuristic detect non-bash code in a gold response. The whitelisted dataset is
# a 50/50 mix of NL2Bash (matches the eval) and bigcode self-oss-instruct
# (general code, mostly Python — wraps as Action 1: execute[<python>] in a bash
# ReAct prompt, which is off-distribution for the eval). Filtering drops the
# clearly-non-bash half. Toggle: INTERCODE_FILTER_BASH=0 to disable.
_NON_BASH_PREFIXES = (
    "def ", "import ", "from ", "class ", "@",
    "function ", "package ", "fn ", "func ", "fun ",
    "var ", "let ", "const ", "use ",
    "module ", "interface ", "trait ", "struct ", "enum ",
    "public ", "private ", "static ", "void ", "int ", "char ", "float ",
    "#include", "#!/usr/bin/env python", "#!/usr/bin/env node",
    "<?php", "<?xml", "<!DOCTYPE", "<html",
)


def _looks_like_non_bash(cmd: str) -> bool:
    """True if cmd is clearly non-bash (Python/JS/C/etc.) code. Conservative:
    only flags strong signals so legit bash is rarely dropped."""
    s = cmd.strip()
    if not s:
        return True
    # Prefix signals on the whole string.
    if s.startswith(_NON_BASH_PREFIXES):
        return True
    # Same signals on any of the first 5 lines.
    first5 = "\n".join(s.split("\n")[:5])
    for kw in ("\ndef ", "\nimport ", "\nfrom ", "\nclass ", "\nfunction "):
        if kw in first5:
            return True
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_path", required=True)
    p.add_argument("--dataset_dir", default=None,
                   help="Override mounted dataset dir; else auto-find from MINER_DATASETS.")
    p.add_argument("--max_examples", type=int, default=0, help="0 = no cap.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ds_dir = Path(args.dataset_dir) if args.dataset_dir else _find_intercode_dir()
    if ds_dir is None or not ds_dir.exists():
        raise RuntimeError(
            "intercode dataset not found. Expected a validator-mounted dir whose name "
            "contains 'intercode' under MINER_DATASETS_DIR, or pass --dataset_dir. "
            f"(MINER_DATASETS_DIR={os.environ.get('MINER_DATASETS_DIR')!r} "
            f"MINER_DATASETS={os.environ.get('MINER_DATASETS')!r})"
        )

    rows = _load_rows(ds_dir)
    print(f"[intercode_trajectories] loaded {len(rows)} raw rows from {ds_dir}", flush=True)

    filter_bash = os.environ.get("INTERCODE_FILTER_BASH", "1").strip() not in ("0", "false", "no", "")
    # Multi-turn ReAct: build (execute -> Observation -> submit) instead of
    # single-turn. zeus default is multi-turn ON: it teaches the eval's submit
    # pattern explicitly. Observation defaults to a generic placeholder; real
    # execution via intercode_local_bash_env requires baked fs snapshots
    # (image-build-time bake in standalone-text-trainer.dockerfile).
    #
    # REAL_EXEC default RE-ENABLED (1) after adding a pre-filter +
    # restricted-env step() in intercode_local_bash_env.py. The historic
    # incident (commit a13f8d5) was a gold command wiping /bin/sh + python.
    # Defense:
    #   1. is_safe_for_real_exec() pre-filter drops gold whose absolute paths
    #      escape /testbed//system//tmp scope, or matches destructive patterns
    #      (rm -rf /, find / -delete, fork bomb, mkfs, shutdown, package
    #      uninstall, etc.). Unsafe rows fall back to synthetic obs.
    #   2. step() runs with restricted env (PATH=/usr/bin:/bin, HOME=/tmp).
    #   3. reset() checks /bin/bash + /bin/tar exist before each row; if a
    #      prior row managed damage, raises -> circuit breaker disables
    #      real-exec for the rest of the run.
    # We can't use chroot or unprivileged userns: cap_drop=ALL on the trainer
    # container blocks CAP_SYS_CHROOT, and no-new-privileges can block the
    # userns escape. Filter-based defense is the realistic option.
    multi_turn = os.environ.get("INTERCODE_MULTI_TURN", "1").strip() not in ("0", "false", "no", "")
    real_exec = os.environ.get("INTERCODE_REAL_EXEC", "1").strip() not in ("0", "false", "no", "")
    bash_env_factory = None
    if multi_turn and real_exec:
        try:
            from our_envs.intercode_local_bash_env import LocalBashEnv, detect_fs_for_gold, is_safe_for_real_exec
            snapshot_root = Path(os.environ.get("INTERCODE_FS_ROOT", "/intercode_fs"))
            if not snapshot_root.exists():
                print(
                    f"[intercode_trajectories] WARN: INTERCODE_REAL_EXEC=1 but "
                    f"{snapshot_root} not present; falling back to synthetic obs.",
                    flush=True,
                )
                real_exec = False
            else:
                bash_env_factory = (LocalBashEnv, detect_fs_for_gold, is_safe_for_real_exec, snapshot_root)
        except ImportError as exc:
            print(
                f"[intercode_trajectories] WARN: real-exec module missing ({exc}); "
                f"falling back to synthetic obs.",
                flush=True,
            )
            real_exec = False

    # Aggregate safety: if real-exec fails this many times in a row, disable
    # it for the remaining rows (snapshot corruption, missing binaries, etc.
    # would otherwise burn budget on per-row timeouts for every remaining row).
    _MAX_CONSECUTIVE_EXEC_FAILS = 20

    # Round-9: walltime budget. After this many seconds, stop processing more
    # rows and save what we have (falling back to synthetic obs for remainder
    # would still take time). Round-8 found that real-exec with restored libs
    # actually runs gold commands -- 6700+ rows × ~1-5s each = hours, far past
    # the orchestrator's per-env budget (~2700s for 2-env split). Orchestrator
    # doesn't wait for intercode if GR hit its time cap, so partial dataset is
    # better than no dataset.
    # Round-10: default raised from 600s to 2700s to match GR's max_gen_seconds
    # (per-env cap in a 2-env split). Intercode runs in parallel with GR; total
    # wall-clock gen is max(intercode, GR) which is bounded by GR's 2700s cap
    # regardless. With ~8 row/s throughput, real-exec for all 6700+ qualifying
    # rows naturally finishes in ~1500s -- well under 2700s. The cap is a
    # safety net for unexpectedly slow runs.
    import time as _time
    _walltime_budget_sec = float(os.environ.get("INTERCODE_GEN_WALLTIME_SEC", "2700"))
    _start_ts = _time.monotonic()
    # Progress log every N rows so we can see gen actually running (round-8
    # had no per-row output -> impossible to tell hung from slow).
    _PROGRESS_EVERY = 500
    # Round-14: incremental save every N examples. If a destructive gold
    # kills Python interpreter mid-loop (round-13 silent crash at row ~8500),
    # the latest checkpoint stays on disk so merge can still proceed with
    # partial data. Atexit hook also tries final save on any clean exit.
    _SAVE_EVERY = 2000

    examples: list[dict] = []
    dropped = 0
    dropped_non_bash = 0
    real_exec_used = 0
    real_exec_skipped_fs3 = 0
    real_exec_skipped_unsafe = 0
    real_exec_failed = 0
    consecutive_fails = 0
    walltime_disabled_real_exec = False

    def _save_partial(label: str) -> None:
        """Save current examples to output_path. Tolerant of any failure
        (best-effort). Called periodically + atexit + at normal completion."""
        if not examples:
            return
        try:
            ds = Dataset.from_list(examples)
            splits = ds.train_test_split(test_size=VALIDATION_RATIO, seed=args.seed)
            dd = DatasetDict({"train": splits["train"], "validation": splits["test"]})
            dd.save_to_disk(args.output_path)
            print(
                f"[intercode_trajectories] {label} save: {len(examples)} examples "
                f"-> {args.output_path} train={len(dd['train'])} val={len(dd['validation'])}",
                flush=True,
            )
        except Exception as exc:
            print(f"[intercode_trajectories] {label} save FAILED: {exc}", flush=True)

    # atexit hook: best-effort save on any clean exit (does NOT fire on SIGKILL).
    import atexit
    atexit.register(_save_partial, "atexit-final")
    # Forensics: keep a rolling window of the last successful golds so when
    # damage is detected we can print the likely perpetrator and use it to
    # add a new filter pattern in the next iteration.
    recent_successful_golds: list[str] = []
    _RECENT_WINDOW = 5
    _printed_perpetrator = False
    for row_idx, row in enumerate(rows):
        # Round-9: walltime check at top of loop. If budget exceeded, switch
        # to synthetic-only for remaining rows so we finish processing them
        # quickly and the loop actually ends.
        if not walltime_disabled_real_exec and bash_env_factory is not None:
            elapsed = _time.monotonic() - _start_ts
            if elapsed > _walltime_budget_sec:
                print(
                    f"[intercode_trajectories] WALLTIME BUDGET {_walltime_budget_sec}s "
                    f"exceeded at row {row_idx}/{len(rows)} -- disabling real-exec "
                    f"for remaining rows (synthetic obs fallback) so gen finishes.",
                    flush=True,
                )
                bash_env_factory = None
                walltime_disabled_real_exec = True
        # Progress: print every N rows so we see live gen status.
        if row_idx > 0 and row_idx % _PROGRESS_EVERY == 0:
            elapsed = _time.monotonic() - _start_ts
            print(
                f"[intercode_trajectories] progress row={row_idx}/{len(rows)} "
                f"elapsed={elapsed:.0f}s built={len(examples)} "
                f"used={real_exec_used} skipped_unsafe={real_exec_skipped_unsafe}",
                flush=True,
            )
        if not isinstance(row, dict):
            dropped += 1
            continue
        gold = _extract(row, _GOLD_KEYS)
        if filter_bash and _looks_like_non_bash(gold):
            dropped_non_bash += 1
            continue
        query = _extract(row, _QUERY_KEYS)
        if multi_turn:
            observation = None
            obs_imperfect = None
            # IC-2: for a deterministic fraction of pipeline golds, build a
            # refine-to-exact trajectory. refine_head = gold minus its final pipe
            # stage (the imperfect first attempt); None disables refine for this row.
            refine_head = derive_imperfect_first_attempt(gold) if _refine_selected(gold) else None
            if bash_env_factory is not None:
                LBE, detect_fs, safe_check, snap_root = bash_env_factory
                fs = detect_fs(gold)
                if fs == -1:
                    # fs3 task: WOULD modify /workspace -> fall back to synthetic.
                    real_exec_skipped_fs3 += 1
                elif not safe_check(gold) or (refine_head is not None and not safe_check(refine_head)):
                    # Gold (or its refine head) escapes snapshot scope (touches
                    # /bin, /etc, etc.) or matches a destructive pattern -> synthetic
                    # obs to protect the trainer container.
                    real_exec_skipped_unsafe += 1
                else:
                    try:
                        env = LBE(fs_version=fs, snapshot_root=snap_root)
                        env.reset()
                        # IC-2: run the imperfect head first (same shell session) so
                        # Observation 1 is its real over-verbose output, then the gold.
                        if refine_head is not None:
                            obs_imperfect = env.step(refine_head)
                        observation = env.step(gold)
                        real_exec_used += 1
                        consecutive_fails = 0
                        # Track last K successful golds for post-mortem.
                        recent_successful_golds.append(gold)
                        if len(recent_successful_golds) > _RECENT_WINDOW:
                            recent_successful_golds.pop(0)
                    except Exception as exc:
                        print(f"[intercode_trajectories] real-exec fail (fs{fs}): {exc}", flush=True)
                        real_exec_failed += 1
                        consecutive_fails += 1
                        # On the FIRST fail of a run (damage just happened),
                        # print the last successful golds — one of these is the
                        # perpetrator that wiped the trainer fs. Use it to add
                        # a new filter pattern in the next code iteration.
                        if not _printed_perpetrator and "fs damaged" in str(exc):
                            print(
                                "[intercode_trajectories] LIKELY PERP — gold commands run "
                                f"just before damage (last {len(recent_successful_golds)}):",
                                flush=True,
                            )
                            for i, g in enumerate(recent_successful_golds, 1):
                                print(f"  [{i}] {g!r}", flush=True)
                            _printed_perpetrator = True
                        if consecutive_fails >= _MAX_CONSECUTIVE_EXEC_FAILS:
                            print(
                                f"[intercode_trajectories] {consecutive_fails} consecutive real-exec fails "
                                f"-> DISABLING real-exec for remaining rows (synthetic obs fallback).",
                                flush=True,
                            )
                            bash_env_factory = None
            if tool_calling_enabled():
                # #1201 tool-calling: N independent per-turn examples per task.
                if refine_head is not None:
                    tool_exs = build_tool_examples_refine(
                        query, gold, obs_imperfect, observation, refine_head
                    )
                else:
                    tool_exs = build_tool_examples_multiturn(query, gold, observation)
                if not tool_exs:
                    dropped += 1
                    continue
                examples.extend({"messages": m} for m in tool_exs)
                if len(examples) % _SAVE_EVERY < len(tool_exs):
                    _save_partial(f"checkpoint@{len(examples)}")
                if args.max_examples and len(examples) >= args.max_examples:
                    break
                continue
            if refine_head is not None:
                msgs = build_intercode_refine_example(
                    query, gold, obs_imperfect=obs_imperfect, obs_gold=observation
                )
                # If the refine split unexpectedly failed, fall back to 2-turn.
                if msgs is None:
                    msgs = build_intercode_multiturn_example(query, gold, observation=observation)
            else:
                msgs = build_intercode_multiturn_example(query, gold, observation=observation)
        else:
            if tool_calling_enabled():
                tool_exs = build_tool_examples_single(query, gold)
                if not tool_exs:
                    dropped += 1
                    continue
                examples.extend({"messages": m} for m in tool_exs)
                if args.max_examples and len(examples) >= args.max_examples:
                    break
                continue
            msgs = build_intercode_sft_example(query, gold)
        if msgs is None:
            dropped += 1
            continue
        examples.append({"messages": msgs})
        # Round-14: incremental checkpoint every _SAVE_EVERY examples. So if
        # Python dies later (silent crash from destructive gold), the latest
        # checkpoint remains on disk for merge_trajectories to pick up.
        if len(examples) > 0 and len(examples) % _SAVE_EVERY == 0:
            _save_partial(f"checkpoint@{len(examples)}")
        if args.max_examples and len(examples) >= args.max_examples:
            break

    print(
        f"[intercode_trajectories] built {len(examples)} SFT examples "
        f"(multi_turn={multi_turn}, real_exec={real_exec} -> used={real_exec_used} "
        f"failed={real_exec_failed} skipped_fs3={real_exec_skipped_fs3} "
        f"skipped_unsafe={real_exec_skipped_unsafe}), "
        f"dropped {dropped} (empty/invalid), "
        f"dropped_non_bash {dropped_non_bash} (filter_bash={filter_bash})",
        flush=True,
    )
    if not examples:
        raise RuntimeError(
            f"No intercode SFT examples built from {ds_dir}. Check dataset columns "
            f"(tried query keys {_QUERY_KEYS}, gold keys {_GOLD_KEYS})."
        )

    _save_partial("final")


if __name__ == "__main__":
    main()
