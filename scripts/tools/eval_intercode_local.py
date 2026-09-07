"""Local intercode scorer — the metric the validator actually computes.

G.O.D ships a local evaluator for swe_infinite only. There is none for
intercode or the PvP games, so every miner tunes those blind. In our own group
that showed up as four submissions landing on an identical 74.1% solve rate:
nobody was measuring, so nobody moved.

This reproduces the validator's intercode reward against the real NL2Bash task
set, using the LocalBashEnv and filesystem snapshots this repo already ships.

    reward = 0.01 + p1 + p2 + p3
      p1 = 0.33 * (1 - erf(|diff_miss| + |diff_extra|))   filesystem diff vs gold
      p2 = 0.33 * (common_changes_correct / common_total)  byte-identical changes
      p3 = 0.33 * tfidf_cosine(agent_obs, gold_obs)        stdout similarity

Two things to hold on to when reading the numbers:

  * p3 compares the agent's LAST observation to the gold's. An extra
    exploratory command AFTER the answer is already on screen overwrites it and
    collapses a third of the score. That is precisely what the 1:2 submit
    upweight in intercode_tool_format is defending against.
  * On a read-only task gold changes nothing, so p1 is free and p2 has nothing
    to compare. The differentiator is p3 almost alone.

Usage:
    python -m tools.eval_intercode_local --policy gold --num-seeds 20
    python -m tools.eval_intercode_local --policy hf --model <repo_or_path> --num-seeds 50

`--policy gold` is the self-test: a policy that runs the gold command and
submits must score ~1.0. If it does not, the harness is wrong, not the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_TASK_DIRS = ("/intercode_data", "/opt/intercode/data/nl2bash")
DEFAULT_SNAPSHOT_ROOT = "/intercode_fs"
MAX_TURNS = 10
MAX_TOKENS_PER_CALL = 512
OBS_TRUNCATE = 350


# ---------------------------------------------------------------- scoring ---
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+")


def _tokens(text: str) -> list:
    return _TOKEN_RE.findall((text or "").lower())


def tfidf_cosine(a: str, b: str) -> float:
    """TF-IDF cosine over the {agent, gold} pair, with an exact-match fallback.

    The validator fits its vectoriser on the pair too, so the shape matches;
    treat the absolute value as approximate and the ORDERING as the thing to
    trust when comparing two of your own runs.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    if a.strip() == b.strip():
        return 1.0

    ca, cb = Counter(ta), Counter(tb)
    vocab = set(ca) | set(cb)
    # idf over a 2-document corpus, smoothed as sklearn does by default.
    idf = {t: math.log((1 + 2) / (1 + ((t in ca) + (t in cb)))) + 1.0 for t in vocab}
    va = {t: ca[t] * idf[t] for t in vocab}
    vb = {t: cb[t] * idf[t] for t in vocab}
    dot = sum(va[t] * vb[t] for t in vocab)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    return (dot / (na * nb)) if na and nb else 0.0


def score_task(
    agent_obs: str,
    gold_obs: str,
    agent_fs: dict,
    gold_fs: dict,
    base_fs: dict,
    p2_when_empty: str = "full",
) -> dict:
    """The validator's continuous reward, component by component."""
    agent_changed = {p: h for p, h in agent_fs.items() if base_fs.get(p) != h}
    agent_changed.update({p: None for p in base_fs if p not in agent_fs})
    gold_changed = {p: h for p, h in gold_fs.items() if base_fs.get(p) != h}
    gold_changed.update({p: None for p in base_fs if p not in gold_fs})

    diff_miss = set(gold_changed) - set(agent_changed)     # gold changed, agent did not
    diff_extra = set(agent_changed) - set(gold_changed)    # agent changed, gold did not
    p1 = 0.33 * (1.0 - math.erf(len(diff_miss) + len(diff_extra)))

    common = set(agent_changed) & set(gold_changed)
    if not common:
        # Nothing was modified by both sides. Most NL2Bash tasks are read-only,
        # so this is the common case, and the validator's ratio is undefined
        # here -- exposed as a flag rather than silently assumed.
        p2 = 0.33 if p2_when_empty == "full" else 0.0
        p2_total = 0
        p2_correct = 0
    else:
        p2_total = len(common)
        p2_correct = sum(1 for p in common if agent_changed[p] == gold_changed[p])
        p2 = 0.33 * (p2_correct / p2_total)

    sim = tfidf_cosine(agent_obs, gold_obs)
    p3 = 0.33 * sim
    return {
        "reward": 0.01 + p1 + p2 + p3,
        "p1": p1, "p2": p2, "p3": p3, "similarity": sim,
        "diff_miss": len(diff_miss), "diff_extra": len(diff_extra),
        "common_total": p2_total, "common_correct": p2_correct,
    }


# ------------------------------------------------------------- filesystem ---
def snapshot_fs(managed_paths) -> dict:
    """sha256 of every regular file under the fs variant's managed paths."""
    out: dict = {}
    for root in managed_paths or ():
        rp = Path(root)
        if not rp.exists():
            continue
        for p in rp.rglob("*"):
            try:
                if not p.is_file() or p.is_symlink():
                    continue
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(65536), b""):
                        h.update(chunk)
                out[str(p)] = h.hexdigest()
            except OSError:
                continue
    return out


# ------------------------------------------------------------------ tasks ---
def load_tasks(explicit, num_seeds: int, seed: int) -> list:
    """NL2Bash tasks, from the same JSON the validator bakes.

    The trainer image does not carry them (the dockerfile deletes its intercode
    clone after building the fs tars), so point --tasks at a clone:
        git clone --depth 1 https://github.com/princeton-nlp/intercode /opt/intercode
    """
    paths = []
    if explicit:
        paths = [Path(p) for p in explicit]
    else:
        for d in DEFAULT_TASK_DIRS:
            found = sorted(Path(d).glob("nl2bash*.json")) if Path(d).is_dir() else []
            if found:
                paths = found
                break
    if not paths:
        raise SystemExit(
            "No NL2Bash task files found. Pass --tasks <file.json> ... or clone\n"
            "  git clone --depth 1 https://github.com/princeton-nlp/intercode /opt/intercode"
        )

    rows: list = []
    for p in paths:
        try:
            data = json.loads(Path(p).read_text())
        except Exception as exc:
            print(f"[eval] skipping {p}: {exc}", file=sys.stderr)
            continue
        items = data if isinstance(data, list) else data.get("data", data.get("tasks", []))
        fs_hint = 0
        m = re.search(r"fs_?(\d)", Path(p).name)
        if m:
            fs_hint = int(m.group(1))
        for it in items:
            if not isinstance(it, dict):
                continue
            query = it.get("query") or it.get("nl") or it.get("instruction")
            gold = it.get("gold") or it.get("cmd") or it.get("command")
            if query and gold:
                rows.append({"query": query, "gold": gold, "fs": it.get("fs", fs_hint)})

    if not rows:
        raise SystemExit(f"No usable query/gold rows in: {[str(p) for p in paths]}")
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:num_seeds] if num_seeds else rows


# --------------------------------------------------------------- policies ---
class GoldPolicy:
    """Self-test: run the gold command, then submit. Must score ~1.0."""

    name = "gold"

    def __init__(self, **_):
        self.done = set()

    def act(self, task, history):
        key = id(task)
        if key in self.done:
            return ("submit", None)
        self.done.add(key)
        return ("execute_bash", task["gold"])


class NoopPolicy:
    """Floor: submit immediately without looking. Shows what 'do nothing' scores."""

    name = "noop"

    def __init__(self, **_):
        pass

    def act(self, task, history):
        return ("submit", None)


class HFPolicy:
    """The real thing: the trained model, rendered exactly as training did.

    Same chat template and same `tools` block the SFT rows carry, because a
    prefix that differs from the one the model was trained on is the whole
    forfeit bug class this repo has fought before.
    """

    name = "hf"

    def __init__(self, model: str, max_tokens: int = MAX_TOKENS_PER_CALL, **_):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from our_envs.intercode_tool_format import INTERCODE_TOOLS

        self.tools = INTERCODE_TOOLS
        self.max_tokens = max_tokens
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
        self.model.eval()

    def act(self, task, history):
        import torch

        msgs = [{"role": "user", "content": task["query"]}]
        for turn in history:
            msgs.append({"role": "assistant", "content": turn["raw"]})
            msgs.append({"role": "user", "content": f"Observation: {turn['obs']}"})
        ids = self.tok.apply_chat_template(
            msgs, tools=self.tools, tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=self.max_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        text = self.tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        return parse_tool_call(text), text


_CALL_RE = re.compile(r'execute_bash\s*\(\s*command\s*=\s*(".*?"|\'.*?\')\s*\)', re.DOTALL)


def parse_tool_call(text: str):
    """Read one tool call out of a completion. Returns (kind, command)."""
    if not text:
        return ("none", None)
    # Structured tool_call JSON first, then the flat call form. Matched on the
    # "name" field rather than a brace-balanced blob: the arguments object nests
    # its own braces, which a [^{}] scan cannot cross.
    m = re.search(r'"name"\s*:\s*"(execute_bash|submit)"', text)
    if m:
        if m.group(1) == "submit":
            return ("submit", None)
        cm = re.search(r'"command"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if cm:
            try:
                return ("execute_bash", json.loads(f'"{cm.group(1)}"'))
            except Exception:
                return ("execute_bash", cm.group(1))
        return ("execute_bash", None)
    if re.search(r"\bsubmit\s*\(\s*\)", text):
        return ("submit", None)
    m = _CALL_RE.search(text)
    if m:
        raw = m.group(1)
        try:
            return ("execute_bash", json.loads(raw) if raw.startswith('"') else raw[1:-1])
        except Exception:
            return ("execute_bash", raw[1:-1])
    return ("none", None)


POLICIES = {"gold": GoldPolicy, "noop": NoopPolicy, "hf": HFPolicy}


# --------------------------------------------------------------- rollouts ---
def run_episode(policy, task, env, max_turns: int) -> str:
    """Play one task. Returns the FINAL observation -- what p3 is scored on."""
    env.reset(task["query"])
    history: list = []
    last_obs = ""
    for _ in range(max_turns):
        res = policy.act(task, history)
        raw = ""
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], tuple):
            (kind, cmd), raw = res
        else:
            kind, cmd = res
        if kind == "submit":
            break
        if kind != "execute_bash" or not cmd:
            history.append({"raw": raw or "", "obs": "Invalid tool call."})
            last_obs = "Invalid tool call."
            continue
        obs = env.step(cmd)
        last_obs = obs
        history.append({"raw": raw or f'execute_bash(command="{cmd}")', "obs": obs})
    return last_obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="gold", choices=sorted(POLICIES))
    ap.add_argument("--model", default=None, help="HF repo or local path (--policy hf)")
    ap.add_argument("--tasks", nargs="*", default=None, help="nl2bash_fs_*.json files")
    ap.add_argument("--snapshot-root", default=DEFAULT_SNAPSHOT_ROOT)
    ap.add_argument("--num-seeds", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS_PER_CALL)
    ap.add_argument("--p2-empty", default="full", choices=("full", "zero"))
    ap.add_argument("--json", default=None, help="write per-task detail here")
    args = ap.parse_args()

    from our_envs.intercode_local_bash_env import (
        LocalBashEnv,
        detect_fs_for_gold,
        is_safe_for_real_exec,
    )

    tasks = load_tasks(args.tasks, args.num_seeds, args.seed)
    policy = POLICIES[args.policy](model=args.model, max_tokens=args.max_tokens)
    print(f"[eval] policy={policy.name} tasks={len(tasks)} max_turns={args.max_turns}")

    rows: list = []
    skipped = 0
    for i, task in enumerate(tasks, 1):
        fs_v = task.get("fs") or detect_fs_for_gold(task["gold"])
        if fs_v not in (1, 2, 4) or not is_safe_for_real_exec(task["gold"]):
            skipped += 1
            continue
        try:
            env = LocalBashEnv(fs_v, Path(args.snapshot_root))
        except Exception as exc:
            print(f"[eval] fs{fs_v} unavailable ({exc}); skipping task {i}")
            skipped += 1
            continue

        env.reset(task["query"])
        base_fs = snapshot_fs(env.managed_paths)
        gold_obs = env.step(task["gold"])
        gold_fs = snapshot_fs(env.managed_paths)

        agent_obs = run_episode(policy, task, env, args.max_turns)
        agent_fs = snapshot_fs(env.managed_paths)

        s = score_task(agent_obs, gold_obs, agent_fs, gold_fs, base_fs, args.p2_empty)
        s.update(query=task["query"], gold=task["gold"], fs=fs_v)
        rows.append(s)
        if i % 10 == 0 or i == len(tasks):
            mean = sum(r["reward"] for r in rows) / len(rows)
            print(f"[eval] {i}/{len(tasks)}  running mean reward {mean:.4f}")

    if not rows:
        print("[eval] no scorable tasks")
        return 1

    n = len(rows)
    agg = {k: sum(r[k] for r in rows) / n for k in ("reward", "p1", "p2", "p3", "similarity")}
    print("\n=== intercode local score ===")
    print(f"  tasks scored : {n} (skipped {skipped})")
    print(f"  MEAN REWARD  : {agg['reward']:.4f}")
    print(f"    p1 filesystem diff : {agg['p1']:.4f} / 0.33")
    print(f"    p2 common changes  : {agg['p2']:.4f} / 0.33   (empty-case = {args.p2_empty})")
    print(f"    p3 answer match    : {agg['p3']:.4f} / 0.33   (mean similarity {agg['similarity']:.3f})")
    print(f"  perfect answers      : {sum(1 for r in rows if r['similarity'] > 0.999)}/{n}")
    print(f"  zero answers         : {sum(1 for r in rows if r['similarity'] < 0.001)}/{n}")
    print(f"  extra fs changes     : {sum(r['diff_extra'] for r in rows)} across all tasks")

    if args.json:
        Path(args.json).write_text(json.dumps({"aggregate": agg, "tasks": rows}, indent=2))
        print(f"  detail -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
