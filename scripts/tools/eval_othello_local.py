#!/usr/bin/env python3
"""Score a model on a PvP board game the way the validator does: vs MCTS.

Half of every task in this tournament is othello, and until now that half was
invisible -- every measurement we had was intercode.  This closes it.

The opponent is an in-process OpenSpiel MCTSBot with RandomRolloutEvaluator
(n_rollouts=1) at the game's EVAL simulation count, which is what
ENVIRONMENTS[<env>].eval_payload_extra pins validator-side (othello, clobber,
leduc, gin: 50; liars_dice: 225).  Training samples a BAND around that value
for state variety; a measurement must not, so the count is fixed here.

The prompt is built by the same functions the SFT rows are built from
(build_system_prompt + toolcall_user_prompt_from_reformatted + the per-turn
game_action schema), so what the model sees here is byte-aligned with what it
saw in training and with what it will see at eval.

Three policies:
  hf       the model under test
  random   the floor -- uniform over legal moves
  teacher  the canonical expert used to LABEL the SFT data.  This is the
           harness gate: if the teacher cannot beat MCTS at eval strength,
           the measurement is broken (or the teacher is), and a model score
           on top of it means nothing.

Invalid output is NOT silently repaired into a good move.  A turn with no
parseable game_action, or with an illegal id, is counted and reported on its
own line, because "plays legal but weak moves" and "cannot emit a tool call"
are different diseases with different cures.  --on-invalid decides whether
the game then continues on a random legal move (default, so move quality
stays measurable) or is scored as an immediate loss (what eval does).

    python -m tools.eval_othello_local --policy teacher --games 20
    python -m tools.eval_othello_local --policy hf --model <path> --games 40
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Validator eval opponent strength per game (mcts_opponent.py documents the
# source: G.O.D ENVIRONMENTS[<env>].eval_payload_extra). Training bands around
# these; a measurement pins them.
EVAL_SIMS = {
    "othello": 50, "clobber": 50, "leduc_poker": 50, "gin_rummy": 50,
    "liars_dice": 225,
}
# MCTSBot is sequential-only. goofspiel is a simultaneous-move game, so an MCTS
# opponent there is not "a weaker opponent", it is a wrong one.
UNSUPPORTED = {"goofspiel": "simultaneous-move; MCTSBot cannot drive that seat"}


# ------------------------------------------------------------------ prompt ---
def build_prompt(env_name: str, state, player_id: int, legal_ids: list):
    """(messages, tools) exactly as a training row for this turn would have."""
    from our_envs.pvp_selfplay import _build_obs
    from our_envs.pvp_prompts.pvp_prompt_loader import get_game_rules_prompt
    from our_envs.pvp_tool_calling import (
        MemoryState, build_system_prompt, pvp_tools_for_turn,
        toolcall_user_prompt_from_reformatted,
    )
    obs = _build_obs(env_name, state, player_id, legal_ids)
    msgs = [
        {"role": "system", "content": build_system_prompt(get_game_rules_prompt(env_name), MemoryState())},
        {"role": "user", "content": toolcall_user_prompt_from_reformatted(obs, player_id=player_id)},
    ]
    return msgs, pvp_tools_for_turn(legal_ids)


_TC_NAME_RE = re.compile(r'"name"\s*:\s*"game_action"')
_AID_RE = re.compile(r'"action_id"\s*:\s*"?(-?\d+)"?')
_BARE_RE = re.compile(r"^\s*(-?\d+)\s*$")


def parse_action(text: str) -> "tuple[int | None, str]":
    """Read a move out of a completion. Returns (action_id or None, how).

    `how` names the surface that matched, because the mix of surfaces is
    itself a finding: a model answering with a bare integer has learnt the
    game but not the tool protocol, and at eval that is a forfeit.
    """
    if not text:
        return None, "empty"
    if _TC_NAME_RE.search(text):
        m = _AID_RE.search(text, _TC_NAME_RE.search(text).end())
        if m:
            return int(m.group(1)), "tool_call"
        return None, "game_action tanpa action_id"
    m = _AID_RE.search(text)          # action_id present, name elsewhere/absent
    if m:
        return int(m.group(1)), "action_id longgar"
    m = _BARE_RE.match(text.strip().splitlines()[0] if text.strip() else "")
    if m:
        return int(m.group(1)), "bare id (forfeit di eval)"
    return None, "tak terparse"


# ---------------------------------------------------------------- policies ---
class RandomPolicy:
    name = "random"

    def __init__(self, **_):
        pass

    def act_batch(self, jobs):
        return [(random.choice(j["legal"]), "random") for j in jobs]


class TeacherPolicy:
    """The canonical labeler -- the move the SFT data teaches for this state."""
    name = "teacher"

    def __init__(self, env_name: str, **_):
        from our_envs.pvp_selfplay import _canonical_labeler, _expert
        self._label = _canonical_labeler(env_name) or None
        self._fallback = _expert(env_name, "a")
        self.env_name = env_name

    def act_batch(self, jobs):
        from our_envs.pvp_selfplay import _build_obs
        out = []
        for j in jobs:
            obs = _build_obs(self.env_name, j["state"], j["player"], j["legal"])
            try:
                aid = (self._label(obs, j["legal"]) if self._label
                       else self._fallback(obs, j["player"], j["legal"], rng=random))
                out.append((int(aid), "teacher"))
            except Exception as exc:
                out.append((None, f"teacher raised: {type(exc).__name__}"))
        return out


class HFPolicy:
    name = "hf"

    def __init__(self, env_name: str, model: str, base_model: str = None,
                 dtype: str = "auto", max_tokens: int = 128, batch: int = 16, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.env_name = env_name
        self.max_tokens = max_tokens
        self.batch = batch
        self.torch = torch

        adapter = self._adapter_base(model)
        is_adapter = adapter is not None or (Path(model) / "adapter_config.json").is_file()
        base = (base_model or adapter) if is_adapter else None
        td = getattr(torch, dtype) if dtype not in ("auto", None) else "auto"
        if base:
            print(f"[eval] LoRA adapter; base = {base}")
            from peft import PeftModel
            self.tok = AutoTokenizer.from_pretrained(base)
            inner = AutoModelForCausalLM.from_pretrained(base, dtype=td, device_map="auto")
            self.model = PeftModel.from_pretrained(inner, model)
        else:
            self.tok = AutoTokenizer.from_pretrained(model)
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=td, device_map="auto")
        self.model.eval()
        # Batched generation needs left padding: with right padding the newest
        # tokens of the shorter prompts sit behind pad, and the model continues
        # from padding instead of from the prompt.
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    @staticmethod
    def _adapter_base(model: str):
        local = Path(model) / "adapter_config.json"
        if local.is_file():
            return json.loads(local.read_text()).get("base_model_name_or_path")
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(model, "adapter_config.json")
            return json.loads(Path(p).read_text()).get("base_model_name_or_path")
        except Exception:
            return None

    def act_batch(self, jobs):
        texts = []
        for j in jobs:
            msgs, tools = build_prompt(self.env_name, j["state"], j["player"], j["legal"])
            texts.append(self.tok.apply_chat_template(
                msgs, tools=tools, tokenize=False, add_generation_prompt=True))
        out = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i:i + self.batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=self.max_tokens, do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )
            for k in range(len(chunk)):
                txt = self.tok.decode(gen[k][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
                out.append(parse_action(txt))
        return out


POLICIES = {"hf": HFPolicy, "random": RandomPolicy, "teacher": TeacherPolicy}


# ----------------------------------------------------------------- cohort ---
class Game:
    """One game in flight. Seat is fixed by the caller, not sampled, so every
    seed is played from both seats and seat advantage cancels out."""

    def __init__(self, env_name: str, seed: int, seat: int, sims: int, max_turn: int):
        from our_envs.pvp_selfplay import _config_id_for_seed, _load_game, _setup_initial_state
        from our_envs.mcts_opponent import make_mcts_bot
        self.env_name, self.seed, self.seat = env_name, seed, seat
        self.rng = random.Random(seed * 2 + seat)
        self.game = _load_game(env_name, _config_id_for_seed(seed, env_name))
        self.bot = make_mcts_bot(self.game, sims, seed)
        self.state = self.game.new_initial_state()
        _setup_initial_state(env_name, self.state, seed)
        self.max_turn = max_turn
        self.turns = 0
        self.my_turns = 0
        self.invalid = 0        # no parseable game_action
        self.illegal = 0        # parsed, but not a legal id
        self.hows: dict = {}
        self.done = False
        self.score = None       # 1 win / 0.5 draw / 0 loss, from our seat

    def advance_to_me(self):
        """Run chance nodes and the opponent until it is our move, or the game
        is over. Never raises out: a bad MCTS node degrades to a legal move."""
        from our_envs.mcts_opponent import mcts_step_or_none
        while not self.done:
            if self.state.is_terminal() or self.turns >= self.max_turn:
                self.finish()
                return
            if self.state.is_chance_node():
                acts, probs = zip(*self.state.chance_outcomes())
                self.state.apply_action(self.rng.choices(list(acts), weights=list(probs), k=1)[0])
                continue
            cp = self.state.current_player()
            if cp < 0:
                self.finish()
                return
            legal = self.state.legal_actions(cp)
            if not legal:
                self.finish()
                return
            if cp == self.seat:
                return
            a = mcts_step_or_none(self.bot, self.state)
            if a is None or a not in legal:
                a = self.rng.choice(legal)
            self.state.apply_action(a)
            self.turns += 1

    def apply(self, action, how: str, on_invalid: str):
        legal = self.state.legal_actions(self.seat)
        self.my_turns += 1
        self.hows[how] = self.hows.get(how, 0) + 1
        bad = action is None or action not in legal
        if bad:
            if action is None:
                self.invalid += 1
            else:
                self.illegal += 1
            if on_invalid == "lose":
                self.score, self.done = 0.0, True
                return
            action = self.rng.choice(legal)
        self.state.apply_action(action)
        self.turns += 1

    def finish(self):
        from our_envs.pvp_selfplay import _score
        if self.done:
            return
        self.done = True
        # An unfinished game is a draw: neither side earned the point, and
        # calling it a loss would punish long games rather than bad ones.
        self.score = _score(self.state.returns(), self.seat) if self.state.is_terminal() else 0.5


def run_cohort(policy, env_name, seeds, sims, max_turn, on_invalid, cohort, log_every):
    games: list = []
    t0 = time.time()
    pending = [(s, seat) for s in seeds for seat in (0, 1)]
    total = len(pending)
    finished: list = []
    while pending or games:
        while pending and len(games) < cohort:
            s, seat = pending.pop(0)
            g = Game(env_name, s, seat, sims, max_turn)
            g.advance_to_me()
            (finished if g.done else games).append(g)
        if not games:
            continue
        jobs = [{"state": g.state, "player": g.seat, "legal": g.state.legal_actions(g.seat)} for g in games]
        for g, (action, how) in zip(games, policy.act_batch(jobs)):
            g.apply(action, how, on_invalid)
            if not g.done:
                g.advance_to_me()
        still, just = [], []
        for g in games:
            (just if g.done else still).append(g)
        games, finished = still, finished + just
        if just and log_every and len(finished) % log_every < len(just):
            wr = sum(x.score for x in finished) / len(finished)
            el = time.time() - t0
            eta = el / max(len(finished), 1) * (total - len(finished))
            print(f"[eval] {len(finished)}/{total}  skor {wr:.3f}  "
                  f"elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)
    return finished


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="teacher", choices=sorted(POLICIES))
    ap.add_argument("--game", default="othello")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--games", type=int, default=20, help="seeds; each is played from BOTH seats")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--sims", type=int, default=None, help="override the eval simulation count")
    ap.add_argument("--max-turn", type=int, default=80)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--cohort", type=int, default=16, help="games advanced in lockstep")
    ap.add_argument("--batch", type=int, default=16, help="prompts per generate() call")
    ap.add_argument("--on-invalid", default="random", choices=("random", "lose"),
                    help="random: keep playing (move quality stays measurable); "
                         "lose: score the game 0 immediately (what eval does)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.game in UNSUPPORTED:
        print(f"[eval] {args.game}: {UNSUPPORTED[args.game]}")
        return 1
    sims = args.sims if args.sims is not None else EVAL_SIMS.get(args.game, 50)
    if args.policy == "hf" and not args.model:
        print("[eval] --policy hf butuh --model")
        return 1

    policy = POLICIES[args.policy](
        env_name=args.game, model=args.model, base_model=args.base_model,
        dtype=args.dtype, max_tokens=args.max_tokens, batch=args.batch,
    )
    seeds = list(range(args.seed0, args.seed0 + args.games))
    print(f"[eval] game={args.game} policy={policy.name} lawan=MCTS@{sims} "
          f"({len(seeds)} seed x 2 kursi = {2*len(seeds)} game) on-invalid={args.on_invalid}")

    t0 = time.time()
    rows = run_cohort(policy, args.game, seeds, sims, args.max_turn,
                      args.on_invalid, args.cohort, args.log_every)
    n = len(rows)
    if not n:
        print("[eval] tidak ada game")
        return 1

    wins = sum(1 for g in rows if g.score == 1.0)
    draws = sum(1 for g in rows if g.score == 0.5)
    losses = n - wins - draws
    my_turns = sum(g.my_turns for g in rows)
    invalid = sum(g.invalid for g in rows)
    illegal = sum(g.illegal for g in rows)
    hows: dict = {}
    for g in rows:
        for k, v in g.hows.items():
            hows[k] = hows.get(k, 0) + v

    print(f"\n=== {args.game} vs MCTS@{sims} ===")
    print(f"  policy         : {policy.name}")
    print(f"  game           : {n}")
    print(f"  MENANG         : {wins}  ({100*wins/n:.1f}%)")
    print(f"  seri           : {draws}  ({100*draws/n:.1f}%)")
    print(f"  kalah          : {losses}  ({100*losses/n:.1f}%)")
    print(f"  SKOR (w+.5d)   : {sum(g.score for g in rows)/n:.4f}")
    print(f"  giliran kita   : {my_turns}")
    print(f"    tanpa game_action : {invalid}  ({100*invalid/max(my_turns,1):.1f}%)  <- forfeit di eval")
    print(f"    id tidak legal    : {illegal}  ({100*illegal/max(my_turns,1):.1f}%)")
    for k, v in sorted(hows.items(), key=lambda kv: -kv[1]):
        print(f"    bentuk jawaban    : {k:<28} {v:>6} ({100*v/max(my_turns,1):.1f}%)")
    for seat in (0, 1):
        sub = [g for g in rows if g.seat == seat]
        if sub:
            print(f"  kursi P{seat}        : skor {sum(g.score for g in sub)/len(sub):.3f} "
                  f"atas {len(sub)} game")
    print(f"  waktu          : {(time.time()-t0)/60:.1f}m")
    print("\n  patokan: 0.500 = seimbang dengan MCTS eval | random biasanya jauh di bawah")
    print("           teacher harus jelas di atas 0.5, kalau tidak harness/teacher-nya rusak")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "game": args.game, "policy": policy.name, "sims": sims,
            "n": n, "wins": wins, "draws": draws, "losses": losses,
            "score": sum(g.score for g in rows) / n,
            "my_turns": my_turns, "invalid": invalid, "illegal": illegal,
            "hows": hows,
            "games": [{"seed": g.seed, "seat": g.seat, "score": g.score,
                       "my_turns": g.my_turns, "invalid": g.invalid,
                       "illegal": g.illegal} for g in rows],
        }, indent=2))
        print(f"  detail -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
