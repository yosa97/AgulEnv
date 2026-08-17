"""SWE-Infinite SFT trajectory formatter (env_name = "swe_infinite").

SWE-Infinite is an INDIVIDUAL agentic env: the validator drives our served model
with the mini-SWE-agent ("miniswe") inside a per-task Docker checkout of a real
repo, up to 25 bash-action turns, scored on whether the repo's tests pass. There is
NO training sidecar (like intercode), so this is pure SFT — no GRPO, no self-play.

Legal training data = the whitelisted mounted dataset
``gradients-io-tournaments/SWE-ZERO-12M-trajectories-filtered`` (50k mini-swe-agent
trajectories, already decontaminated against the 74 vetted eval tasks). Its
``messages`` column is ALREADY in the exact mini-swe-agent turn format the eval
parses (system = the execution-free MiniSWE prompt; user = the GitHub issue, then
``Observation: ...`` results; assistant = ``THOUGHT: ...`` + exactly one ```bash```
block; finish by echoing ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``). So this generator
is a PASS-THROUGH (mirrors intercode_trajectories, not a from-scratch synth): load the
mounted rows, keep the high-quality ones, cap over-long tool observations so the
decisive final turn is not truncated at tokenize, and emit a
``DatasetDict({"messages": ...})`` for tokenize_env_trajectories (assistant-only loss,
NO tool-call schema — MiniSWE is plain-text bash, not OpenAI tool-calling).

TRAIN==EVAL is the load-bearing invariant (the clobber lesson): the messages carry the
exact MiniSWE system prompt + turn shape the eval uses, so we DO NOT reformat them.

Run: ``python -m our_envs.swe_trajectories --output_path /workspace/scripts/datasets/sft_env_<id>_swe_infinite``
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ``datasets`` is imported lazily inside the functions that need it so the pure
# filtering/format helpers stay importable (and unit-testable) without the heavy dep.

# Dataset dir substring that identifies the mounted SWE trajectories (validator mounts
# whitelisted datasets at MINER_DATASETS_DIR/<org--name>/; the SWE one contains "swe").
_SWE_DIR_HINT = "swe"

# exit_status values kept by default. SWE-ZERO is a MID-TRAINING corpus (mostly
# "incomplete" — the rollout ran out of turns); "Submitted" means the agent completed
# the explore->edit->submit loop (echoed COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT), which
# is the behaviour we want to distil.
# CURATED DEFAULT = "submitted" (2026-07-27): accept-all (~49k rows) was proven
# NON-VIABLE on the VM — ~10.75h for one epoch (grad-accum 64 x 20480-token transcripts
# on a 7B) >> the ~3h tournament budget, so it trains <30% and under-fits; and 96% of
# SWE-ZERO is 'incomplete' (explore-then-give-up) which distils "No patch generated" ->
# 0.0. Keep only completed submissions. Override SWE_EXIT_STATUS="all" for the A/B, or a
# comma list to keep specific statuses.
_DEFAULT_KEEP = "submitted"


def _keep_statuses() -> "set[str] | None":
    v = (os.environ.get("SWE_EXIT_STATUS") or _DEFAULT_KEEP).strip().lower()
    if v in ("all", "*", ""):
        return None  # keep everything
    return {s.strip() for s in v.split(",") if s.strip()}


def _obs_cap() -> int:
    """Max chars for a single tool OBSERVATION (user) turn. Long `cat`/`grep` dumps
    bloat the context and push the decisive final assistant turn past max_length at
    tokenize. The mini-swe-agent scaffold itself caps tool output, so capping here
    keeps train close to eval. 0 disables capping."""
    try:
        return max(0, int(os.environ.get("SWE_OBS_MAX_CHARS", "3000")))
    except ValueError:
        return 3000


def _char_budget() -> int:
    """Drop a trajectory whose total content chars exceed this (~chars/3.5 tokens, so
    ~28000 chars ~ 8000 tokens ~ the 8192 tokenize cap). Prevents silently truncating
    the submit turn. Default 28000 pairs with the curated _SWE_MAX_LENGTH=8192; env-tunable."""
    try:
        return max(2000, int(os.environ.get("SWE_MAX_CHARS", "28000")))
    except ValueError:
        return 28000


# A trajectory teaches EDITING (not just exploration) iff some assistant turn issues a
# file-mutating bash command. Diagnosed 2026-07-20: the model was learning to explore
# (cat/grep) then submit WITHOUT editing -> "No patch generated" -> 0.0. SWE-ZERO's
# 'incomplete' rows (96%) mostly explore-then-run-out; even 'Submitted' rows rarely (3%)
# submit without an edit. Keeping only edit+submit trajectories distils the fix behaviour.
_EDIT_RE = re.compile(
    r"sed -i|>\s*[^ >|&]|>>\s*\S|\btee\b|cat\s*<<|cat\s*>|\bpatch\b|apply_patch"
    r"|python[0-9]?\s+-c|printf[^|]*>",
    re.I,
)


def _require_edit() -> bool:
    """Default ON (2026-07-27) — keep only edit-bearing trajectories. Even 'Submitted'
    rows sometimes submit without editing; those (and all 'incomplete' rows) teach
    explore-then-give-up -> "No patch generated" -> 0.0. SWE_REQUIRE_EDIT=0 restores
    accept-all (A/B)."""
    return (os.environ.get("SWE_REQUIRE_EDIT", "1").strip().lower()
            not in ("0", "false", "no", "off"))


def _has_edit(messages) -> bool:
    for m in messages:
        if m.get("role") == "assistant" and _EDIT_RE.search(str(m.get("content", "") or "")):
            return True
    return False


# --- ORPO preference mode (SWE_ORPO=1) -------------------------------------------------
# Deep-dive verdict (2026-07-27): GRPO is a dead end offline (no SWE env-server, no
# reward oracle at train time), so the only preference lever is ORPO on the fixed corpus.
# We DON'T have natural same-task chosen/rejected pairs (SWE-ZERO has one trajectory per
# task), so we SYNTHESISE the rejected from each edit+submit trajectory: keep its explore
# turns, DROP the edits, keep the real submit turn = "explore-then-submit-without-editing"
# = the exact give-up that scores 0.0. Same prompt, real turns, contrasts edit vs give-up.
def _swe_orpo() -> bool:
    """SWE_ORPO=1 -> emit an ORPO preference dataset {prompt, chosen, rejected} instead of
    {messages} SFT rows. Default OFF: the SFT path stays byte-identical."""
    return (os.environ.get("SWE_ORPO", "0").strip().lower()
            not in ("0", "false", "no", "off", ""))


def _clean_turn(m: dict) -> dict:
    """Plain {role, content} copy. SWE mini-swe turns carry the bash command in the CONTENT
    (markdown), not in structured tool_calls, so ORPO's chat-template render needs no tool
    schema — strip everything else so apply_chat_template can't choke on Arrow tool fields."""
    return {"role": str(m.get("role", "") or ""), "content": str(m.get("content", "") or "")}


def _build_orpo_pairs(examples: "list[dict]") -> "list[dict]":
    """Edit+submit trajectories -> SAME-TASK ORPO triples. NEVER raises.

        prompt   = messages up to & incl. the first user turn   (shared conditioning)
        chosen   = the full explore->EDIT->submit body           (the fix behaviour)
        rejected = explore turns BEFORE the first edit + the real submit turn
                   (explore-then-submit-WITHOUT-editing = the "No patch generated" give-up)

    Skips any trajectory lacking both an edit turn and a strictly-later submit turn."""
    pairs: "list[dict]" = []
    for ex in examples:
        msgs = ex.get("messages") or []
        u = next((i for i, m in enumerate(msgs) if m.get("role") == "user"), None)
        if u is None:
            continue
        prompt = [_clean_turn(m) for m in msgs[: u + 1]]
        body = msgs[u + 1:]
        if not body:
            continue
        e = next((i for i, m in enumerate(body)
                  if m.get("role") == "assistant"
                  and _EDIT_RE.search(str(m.get("content", "") or ""))), None)
        s = next((i for i in range(len(body) - 1, -1, -1)
                  if body[i].get("role") == "assistant"), None)
        if e is None or s is None or s <= e:
            continue
        chosen = [_clean_turn(m) for m in body]
        rejected = [_clean_turn(m) for m in body[:e]] + [_clean_turn(body[s])]
        if chosen and rejected:
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return pairs


def _find_swe_dir() -> "Path | None":
    """Locate the mounted SWE dataset dir from MINER_DATASETS_DIR + MINER_DATASETS
    (mirrors intercode_trajectories/miner_dataset_loader)."""
    parent = os.environ.get("MINER_DATASETS_DIR")
    if not parent or not Path(parent).is_dir():
        return None
    names = [n.strip() for n in (os.environ.get("MINER_DATASETS") or "").split(",") if n.strip()]
    parent_p = Path(parent)
    # Prefer an explicitly-requested subdir whose name mentions swe; else any child dir
    # that looks like the SWE set.
    cands = [parent_p / n for n in names if _SWE_DIR_HINT in n.lower()]
    if not cands:
        cands = [d for d in parent_p.iterdir() if d.is_dir() and _SWE_DIR_HINT in d.name.lower()]
    for d in cands:
        if d.is_dir():
            return d
    return None


def _load_rows(path: Path) -> "list[dict]":
    """Best-effort load mounted rows: HF save_to_disk (arrow) / parquet / jsonl/json."""
    from datasets import load_from_disk
    try:
        if (path / "dataset_info.json").exists() or any(path.glob("*.arrow")):
            return list(load_from_disk(str(path)))
    except Exception as exc:  # noqa: BLE001
        print(f"[swe] load_from_disk failed ({exc}); trying parquet/jsonl", flush=True)
    parquets = list(path.rglob("*.parquet"))
    if parquets:
        try:
            from datasets import load_dataset
            return list(load_dataset("parquet", data_files=[str(p) for p in parquets], split="train"))
        except Exception as exc:  # noqa: BLE001
            print(f"[swe] parquet load failed: {exc}", flush=True)
    rows: "list[dict]" = []
    for jl in list(path.rglob("*.jsonl")) + list(path.rglob("*.json")):
        try:
            with open(jl, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[swe] jsonl read failed for {jl}: {exc}", flush=True)
    return rows


def _cap_observations(messages: "list[dict]", cap: int) -> "list[dict]":
    """Cap over-long NON-assistant (tool observation / user) turns; leave assistant
    (the loss targets: THOUGHT + bash) untouched. Keeps the head + tail of a long dump
    so the model still sees the shape of the output."""
    if cap <= 0:
        return messages
    out = []
    for m in messages:
        c = str(m.get("content", ""))
        if m.get("role") != "assistant" and len(c) > cap:
            head = c[: cap * 2 // 3]
            tail = c[-cap // 3:]
            c = f"{head}\n... [observation truncated] ...\n{tail}"
            m = {**m, "content": c}
        out.append(m)
    return out


def _valid_messages(messages) -> bool:
    """A usable trajectory: alternating-ish chat with >=1 assistant turn that emits a
    bash block, ending on a user/observation or assistant turn (not a bare system)."""
    if not isinstance(messages, list) or len(messages) < 3:
        return False
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    if "system" not in roles or "user" not in roles:
        return False
    return any(
        m.get("role") == "assistant" and "```bash" in str(m.get("content", ""))
        for m in messages
    )


def main() -> None:
    from datasets import Dataset, DatasetDict
    p = argparse.ArgumentParser()
    p.add_argument("--output_path", required=True)
    p.add_argument("--val_frac", type=float, default=0.01)
    args = p.parse_args()

    keep = _keep_statuses()
    cap = _obs_cap()
    budget = _char_budget()

    swe_dir = _find_swe_dir()
    if swe_dir is None:
        print("[swe] no mounted SWE dataset found (MINER_DATASETS_DIR/MINER_DATASETS). "
              "Request gradients-io-tournaments/SWE-ZERO-12M-trajectories-filtered in the "
              "miner training_repo. Writing EMPTY dataset.", flush=True)
        _save_empty(args.output_path)
        return

    rows = _load_rows(swe_dir)
    print(f"[swe] loaded {len(rows)} rows from {swe_dir} "
          f"(keep_status={keep or 'ALL'} obs_cap={cap} char_budget={budget})", flush=True)

    require_edit = _require_edit()
    examples: "list[dict]" = []
    n_status_drop = n_invalid = n_no_edit = n_toolong = 0
    for r in rows:
        status = str(r.get("exit_status", "")).strip().lower()
        if keep is not None and status not in keep:
            n_status_drop += 1
            continue
        messages = r.get("messages")
        if not _valid_messages(messages):
            n_invalid += 1
            continue
        if require_edit and not _has_edit(messages):
            n_no_edit += 1
            continue
        messages = _cap_observations(messages, cap)
        total = sum(len(str(m.get("content", ""))) for m in messages)
        if total > budget:
            n_toolong += 1
            continue
        examples.append({"messages": messages})

    # BUDGET CAP: accept-all keeps ~49k rows, but at ~100s/step (grad-accum 64 x 20480-token
    # transcripts on a 7B) one epoch is ~10.75h — far over the ~3h tournament budget, so it
    # only trains ~28% and under-fits. SWE_MAX_ROWS>0 truncates to a curated subset that
    # COMPLETES a full epoch in budget (shuffled with a fixed seed for a representative mix).
    _max_rows = int(os.environ.get("SWE_MAX_ROWS", "0") or "0")
    if _max_rows > 0 and len(examples) > _max_rows:
        import random as _rnd
        _rnd.Random(1234).shuffle(examples)
        examples = examples[:_max_rows]
        print(f"[swe] SWE_MAX_ROWS cap: truncated to {len(examples)} rows (shuffled seed 1234)", flush=True)

    print(f"[swe] kept={len(examples)} dropped: status={n_status_drop} invalid={n_invalid} "
          f"no_edit={n_no_edit} too_long={n_toolong}", flush=True)

    if not examples:
        print("[swe] no examples survived filtering; writing EMPTY dataset.", flush=True)
        _save_empty(args.output_path)
        return

    # ORPO preference mode: emit {prompt, chosen, rejected} instead of {messages}. The
    # trainer (train_sft_env) detects the chosen/rejected columns and switches to ORPO;
    # merge_trajectories passes the preference schema through untouched.
    if _swe_orpo():
        pairs = _build_orpo_pairs(examples)
        print(f"[swe] ORPO preference pairs: {len(pairs)} (from {len(examples)} kept rows)", flush=True)
        if not pairs:
            print("[swe] no ORPO pairs built (need edit+submit); writing EMPTY dataset.", flush=True)
            _save_empty(args.output_path)
            return
        n_val_orpo = max(1, int(len(pairs) * max(0.0, min(0.2, args.val_frac))))
        dd = DatasetDict({
            "train": Dataset.from_list(pairs[n_val_orpo:]),
            "validation": Dataset.from_list(pairs[:n_val_orpo]),
        })
        dd.save_to_disk(args.output_path)
        print(f"[swe] saved ORPO -> {args.output_path} "
              f"train={len(dd['train'])} val={len(dd['validation'])}", flush=True)
        return

    n_val = max(1, int(len(examples) * max(0.0, min(0.2, args.val_frac))))
    dd = DatasetDict({
        "train": Dataset.from_list(examples[n_val:]),
        "validation": Dataset.from_list(examples[:n_val]),
    })
    dd.save_to_disk(args.output_path)
    print(f"[swe] saved -> {args.output_path} train={len(dd['train'])} val={len(dd['validation'])}",
          flush=True)


def _save_empty(output_path: str) -> None:
    """Never crash the pipeline: an empty DatasetDict lets the mix/tokenize steps run
    (they no-op on empty) instead of failing the whole task."""
    from datasets import Dataset, DatasetDict
    empty = Dataset.from_list([{"messages": []}][:0])
    DatasetDict({"train": empty, "validation": empty}).save_to_disk(output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        # Best-effort empty output so the env-SFT pipeline degrades gracefully.
        try:
            args = sys.argv
            if "--output_path" in args:
                _save_empty(args[args.index("--output_path") + 1])
        except Exception:
            pass
