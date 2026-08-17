"""Load validator-mounted whitelisted datasets into the SFT messages shape.

Tournament contract (miner-dataset-whitelist): when the miner advertises
``requested_datasets`` in its /training_repo response, the validator downloads
the whitelisted datasets and mounts them read-only inside the training
container:

    MINER_DATASETS_DIR = /cache/miner_datasets        (parent dir)
    MINER_DATASETS     = "org--name,org2--name2"       (comma-separated subdirs,
                                                         "--" replaces the first "/")

This loader reads ONLY from that mount. It never reaches the internet: the
tournament trainer runs offline, and downloading data at train time both fails
and matches a pattern the validator's repo cheating-review flags. When the
mount is absent (local runs, or the validator did not mount anything) the
loader returns None and the caller falls back to pure synthetic trajectories.

Schema normalisation: each dataset has its own column names; we best-effort
convert every row to ``{"messages": [{"role": ..., "content": ...}, ...]}`` so
it concatenates directly with generate_trajectories.py output.

Contamination guard: STRICT game-filtering is the default. A row is kept only
when its game field positively matches one of the requested envs. Rows with no
game field (e.g. prose QA datasets whose ``Proof:/Answer:`` format clashes with
the action-ID game format) are DROPPED, so mixing them can never corrupt the
game SFT set.

This module follows the never-raise contract: any failure (no mount, unreadable
dataset, bad schema, empty after filter) yields None rather than aborting SFT.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, Value, concatenate_datasets, load_from_disk


# ---------------------------------------------------------------------------
# Schema normalisation
# ---------------------------------------------------------------------------

def _row_matches_games(row: dict[str, Any], env_names: list[str], strict: bool) -> bool:
    """Return True if the row's game field matches one of ``env_names``.

    strict=True (default): rows WITHOUT a recognisable game field are DROPPED.
    This is the contamination guard — only positively-game-tagged rows survive,
    so a format-clashing dataset (no game field) cannot poison the game SFT set.

    strict=False: rows without a game field are kept (legacy permissive mode).
    """
    # Per-row game tag is SINGULAR ("game"); env_training_gradients uses
    # exactly this column (cf. format_sft_env.py filter_column="game",
    # sft_pre_stage.py r.get("game")). One row = one trajectory = one game.
    # The plural "environment_names" is a TASK-level dataset_type field, NOT a
    # per-row field, so it must NOT be checked here.
    game = row.get("game") or row.get("env")
    if not isinstance(game, str) or not game.strip():
        return not strict
    if not env_names:
        return True
    wanted = {e.lower().strip() for e in env_names}
    return game.lower().strip() in wanted


def _render_tool_calls_to_text(msg: dict[str, Any]) -> str:
    """Render a STRUCTURED assistant message (content + tool_calls — the OpenAI
    messages+tools format of pvp-tool-calling-sft) into OUR canonical Qwen text
    surface: the reasoning content followed by one ``<tool_call>\\n{json}\\n
    </tool_call>`` block per call. tokenize_env_trajectories detects the
    <tool_call> markers and reconstructs the matching tools from the user prompt's
    "Legal actions:" block, so the row trains exactly like a self-play tool_call
    row and renders the same surface the eval scores on. No-op (returns the plain
    content) when there are no tool_calls — so non-tool datasets are unchanged."""
    content = str(msg.get("content") or "").strip()
    blocks = []
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass  # keep the raw string if it isn't valid JSON
        # ensure_ascii=True matches self-play's canonical _tool_call_content, so
        # the mixed train set renders non-ASCII (e.g. card suits) identically.
        blocks.append("<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>")
    if not blocks:
        return content
    return (content + "\n" if content else "") + "\n".join(blocks)


def _row_to_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    """Best-effort convert a raw row to a list of {role, content} messages."""
    # Schema 1: already "messages"-shaped. PRESERVE assistant tool_calls by
    # rendering them into the <tool_call> text surface (else a structured
    # tool_call row would drop its game_action and teach no move).
    if isinstance(row.get("messages"), list):
        out: list[dict[str, str]] = []
        for m in row["messages"]:
            if not isinstance(m, dict) or "role" not in m:
                continue
            if m.get("role") == "assistant" and m.get("tool_calls"):
                content = _render_tool_calls_to_text(m)
                if not content.strip():
                    continue  # all calls nameless / empty -> teaches no move, drop
            elif "content" in m:
                content = m["content"]
                if content is None:
                    continue
            else:
                continue
            out.append({"role": m["role"], "content": str(content)})
        return out or None
    # Schema 2: conversation lists with role/content or from/value.
    for key in ("conversations", "dialog", "chat", "turns"):
        seq = row.get(key)
        if isinstance(seq, list) and seq:
            converted: list[dict[str, str]] = []
            for m in seq:
                if not isinstance(m, dict):
                    continue
                role = m.get("role") or m.get("from") or m.get("speaker")
                content = m.get("content") or m.get("value") or m.get("text")
                if role and content is not None:
                    role_l = role.lower()
                    if role_l in ("gpt", "model", "ai", "assistant"):
                        role = "assistant"
                    elif role_l in ("human", "user"):
                        role = "user"
                    elif role_l == "system":
                        role = "system"
                    else:
                        role = role_l
                    converted.append({"role": role, "content": str(content)})
            if converted:
                return converted
    # Schema 3: single-turn instruction/output.
    instruction = row.get("instruction") or row.get("prompt") or row.get("question") or row.get("input")
    output = row.get("output") or row.get("response") or row.get("answer") or row.get("completion")
    if instruction and output:
        msgs = []
        system = row.get("system") or row.get("system_prompt")
        if system:
            msgs.append({"role": "system", "content": str(system)})
        msgs.append({"role": "user", "content": str(instruction)})
        msgs.append({"role": "assistant", "content": str(output)})
        return msgs
    return None


def _normalize_rows(rows: Iterable[dict[str, Any]], env_names: list[str], strict: bool) -> list[dict[str, Any]]:
    """Game-filter + convert each row to {"messages": [...]}; drop failures."""
    out: list[dict[str, Any]] = []
    for row in rows:
        if not _row_matches_games(row, env_names, strict):
            continue
        msgs = _row_to_messages(row)
        if msgs:
            out.append({"messages": msgs})
    return out


# ---------------------------------------------------------------------------
# On-disk loaders (HF save_to_disk -> parquet -> jsonl/json)
# ---------------------------------------------------------------------------

def _load_one_dataset(path: Path) -> Dataset | None:
    """Best-effort load a mounted dataset directory into a flat Dataset."""
    try:
        if (path / "dataset_info.json").exists() or any(path.glob("*.arrow")):
            ds = load_from_disk(str(path))
            if isinstance(ds, DatasetDict):
                parts: list[dict] = []
                for split in ds:
                    parts.extend(ds[split])
                return Dataset.from_list(list(parts))
            return ds
    except Exception:
        pass
    try:
        parquets = list(path.rglob("*.parquet"))
        if parquets:
            from datasets import load_dataset
            return load_dataset("parquet", data_files=[str(p) for p in parquets], split="train")
    except Exception:
        pass
    try:
        rows: list[dict[str, Any]] = []
        for jl in list(path.rglob("*.jsonl")) + list(path.rglob("*.json")):
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
        if rows:
            return Dataset.from_list(rows)
    except Exception:
        pass
    return None


def _get_miner_datasets_inventory() -> list[tuple[str, Path]]:
    """Return [(hf_repo_name, local_dir), ...] from the validator mount, or []."""
    parent_raw = os.environ.get("MINER_DATASETS_DIR")
    if not parent_raw:
        return []
    parent = Path(parent_raw)
    if not parent.exists():
        return []
    names = os.environ.get("MINER_DATASETS", "")
    if not names.strip():
        return []
    out: list[tuple[str, Path]] = []
    for entry in names.split(","):
        entry = entry.strip()
        if not entry:
            continue
        # The intercode dataset is consumed by envs.intercode_trajectories (it
        # formats the gold commands into the ReAct prompt), NOT by this mix.
        # Skip it here so its raw (query, gold) rows never get concatenated into
        # a game SFT set in the wrong (non-ReAct) format.
        if "intercode" in entry.lower():
            continue
        # The pvp-tool-calling-sft dataset is in the EXACT eval tool-calling format
        # (assistant messages carry structured `tool_calls`; per-row `tools`
        # column; per-row `env` game tag). It is now USED: _row_to_messages renders
        # the structured tool_calls into the <tool_call> text surface (preserving
        # the game_action + any memory edits), and tokenize_env_trajectories
        # reconstructs the matching tools from the user prompt's "Legal actions:"
        # block — exactly like a self-play tool_call row. Game-filtering keys off
        # the `env` column (see _row_matches_games), so only this task's games are
        # mixed in. (Previously skipped because the generic path dropped tool_calls
        # and taught no move; that conversion now lives in _render_tool_calls_to_text.)
        local = parent / entry
        if local.exists():
            out.append((entry.replace("--", "/", 1), local))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_miner_sft_dataset(
    env_names: list[str] | None = None,
    *,
    strict_game_match: bool = True,
    validation_ratio: float = 0.01,
    seed: int = 42,
    cap_rows: int | None = None,
) -> DatasetDict | None:
    """Build a DatasetDict from validator-mounted miner datasets, or None.

    Args:
        env_names: envs of the current task; rows are kept only when their game
            field matches one of these (see strict_game_match).
        strict_game_match: when True (default), rows with no game field are
            DROPPED — the contamination guard.
        validation_ratio: held-out fraction for the validation split.
        seed: split seed.
        cap_rows: optional cap on total rows (deterministic prefix).

    Returns:
        DatasetDict({"train", "validation"}) or None when nothing usable.
    """
    env_names = env_names or []
    inventory = _get_miner_datasets_inventory()
    if not inventory:
        return None

    all_rows: list[dict[str, Any]] = []
    print(
        f"[MINER_DATASETS] env_names={env_names} strict={strict_game_match} "
        f"loading {len(inventory)} mounted dataset(s)",
        flush=True,
    )
    for hf_name, local in inventory:
        try:
            ds = _load_one_dataset(local)
        except Exception as exc:
            print(f"[MINER_DATASETS] skipped {hf_name}: load error {exc}", flush=True)
            continue
        if ds is None:
            print(f"[MINER_DATASETS] skipped {hf_name}: cannot load from {local}", flush=True)
            continue
        normalised = _normalize_rows(ds, env_names, strict_game_match)
        if not normalised:
            print(
                f"[MINER_DATASETS] skipped {hf_name}: 0 rows match games={env_names} "
                f"(raw {len(ds)})",
                flush=True,
            )
            continue
        all_rows.extend(normalised)
        print(
            f"[MINER_DATASETS] loaded {hf_name}: kept {len(normalised)}/{len(ds)} rows "
            f"(game-filtered to {env_names})",
            flush=True,
        )

    if not all_rows:
        print("[MINER_DATASETS] no usable rows after game-filter + normalisation", flush=True)
        return None

    if cap_rows is not None and len(all_rows) > cap_rows:
        all_rows = all_rows[:cap_rows]

    base = Dataset.from_list(all_rows)
    if len(base) < 2:
        return DatasetDict({"train": base, "validation": base.select([])})
    splits = base.train_test_split(test_size=validation_ratio, seed=seed)
    return DatasetDict({"train": splits["train"], "validation": splits["test"]})


# Canonical STRUCTURED `messages` feature — matches generate_trajectories' assistant
# tool_calls so plain miner rows can concatenate with structured synthetic rows.
_MESSAGES_FEATURE = [{
    "role": Value("string"),
    "content": Value("string"),
    "tool_calls": [{"type": Value("string"),
                    "function": {"name": Value("string"), "arguments": Value("string")}}],
}]


def _align_messages_schema(ds: "Dataset") -> "Dataset":
    """Cast a {"messages":[...]} dataset to the canonical structured schema so a plain
    {role,content} dataset concatenates with a structured {role,content,tool_calls} one.
    Plain messages (no tool_calls) get an empty tool_calls list; other columns preserved.
    Never raises catastrophically — on a cast failure the caller keeps the original."""
    def _fix(ex):
        return {"messages": [
            {"role": m.get("role"), "content": m.get("content"),
             "tool_calls": (m.get("tool_calls") or [])}
            for m in ex["messages"]
        ]}
    ds = ds.map(_fix)
    feats = ds.features.copy()
    feats["messages"] = _MESSAGES_FEATURE
    return ds.cast(feats)


def merge_with_synthetic(miner_dd: DatasetDict | None, synthetic_path: str | None) -> DatasetDict | None:
    """Concatenate miner data with the synthetic trajectory DatasetDict.

    Both sides are {"messages": [...]} rows, so columns align directly. If only
    one side is present, return it; if neither, return None.
    """
    synth_dd: DatasetDict | None = None
    if synthetic_path and Path(synthetic_path).exists():
        try:
            loaded = load_from_disk(synthetic_path)
            if isinstance(loaded, DatasetDict):
                synth_dd = loaded
        except Exception:
            synth_dd = None

    if miner_dd is None and synth_dd is None:
        return None
    if miner_dd is None:
        return synth_dd
    if synth_dd is None:
        return miner_dd

    def _cat(split: str) -> Dataset | None:
        parts = [dd[split] for dd in (miner_dd, synth_dd) if split in dd and len(dd[split])]
        if not parts:
            return None
        # Align the messages schema before concatenate: our synthetic rows carry
        # STRUCTURED assistant tool_calls ({role,content,tool_calls}), while a mounted
        # whitelisted miner dataset (e.g. env_training_gradients ShareGPT) is plain
        # {role,content}. Arrow's concatenate can't unify those, so cast both to the
        # canonical structured schema (plain rows get an empty tool_calls list).
        parts = [_align_messages_schema(p) for p in parts]
        return concatenate_datasets(parts)

    out = DatasetDict()
    train = _cat("train")
    if train is not None:
        out["train"] = train.shuffle(seed=42)
    val = _cat("validation")
    if val is not None:
        out["validation"] = val
    return out or None
