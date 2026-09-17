#!/usr/bin/env python3
"""Score a model on a PvP board game the way the validator does.

The validator's PvP metric is MODEL vs MODEL: it plays each pair of miners in
a group head to head (16 games a pair in the tournament we replayed) and ranks
them on that. So --opponent takes another model and that is the real thing;
--opponent mcts (an in-process OpenSpiel MCTSBot at the game's eval simulation
count) is the standing ladder to measure against when no rival model is at
hand -- useful, but a proxy, and worth saying so out loud.

Prompts are built by the same functions that build the SFT rows
(build_system_prompt + toolcall_user_prompt_from_reformatted + the per-turn
game_action schema), so what a model sees here matches training and eval.

Every seed is played from BOTH seats, so first-move advantage cancels exactly.

Invalid output is counted, never silently repaired into a good move: a turn
with no parseable game_action and a turn with an illegal id get their own
lines, because "cannot emit a tool call" and "plays legal but weak moves" are
different diseases. --on-invalid decides whether the game then continues on a
random legal move (default, so move quality stays measurable) or is scored an
immediate loss (what eval does).

    python -m tools.eval_othello_local --policy teacher --games 8
    python -m tools.eval_othello_local --policy <modelA> --opponent <modelB>
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


class MCTSPolicy:
    """The MCTS opponent, wrapped so both seats go through one interface. The
    bot is per-game (it carries search state), so it travels in the job."""
    name = "mcts"

    def __init__(self, **_):
        pass

    def act_batch(self, jobs):
        from our_envs.mcts_opponent import mcts_step_or_none
        out = []
        for j in jobs:
            a = mcts_step_or_none(j["bot"], j["state"])
            out.append((a, "mcts") if a is not None else (None, "mcts gagal"))
        return out


# ----------------------------------------------------------------- cohort ---
class Game:
    """One game in flight between side A and side B.

    Seats are assigned by the caller, not sampled, so every seed is played
    from both seats and first-move advantage cancels out exactly.
    """

    def __init__(self, env_name: str, seed: int, a_seat: int, sims: int, max_turn: int,
                 need_bot: bool):
        from our_envs.pvp_selfplay import _config_id_for_seed, _load_game, _setup_initial_state
        self.env_name, self.seed, self.a_seat = env_name, seed, a_seat
        self.rng = random.Random(seed * 2 + a_seat)
        self.game = _load_game(env_name, _config_id_for_seed(seed, env_name))
        self.bot = None
        if need_bot:
            from our_envs.mcts_opponent import make_mcts_bot
            self.bot = make_mcts_bot(self.game, sims, seed)
        self.state = self.game.new_initial_state()
        _setup_initial_state(env_name, self.state, seed)
        self.max_turn = max_turn
        self.turns = 0
        self.stats = {"a": {"turns": 0, "invalid": 0, "illegal": 0, "hows": {}},
                      "b": {"turns": 0, "invalid": 0, "illegal": 0, "hows": {}}}
        self.done = False
        self.score = None       # 1 / 0.5 / 0, from A's seat

    def side_to_move(self):
        """Run chance and simultaneous nodes, then say whose turn it is:
        "a", "b", or None when the game is over."""
        while not self.done:
            if self.state.is_terminal() or self.turns >= self.max_turn:
                self.finish()
                return None
            if self.state.is_chance_node():
                acts, probs = zip(*self.state.chance_outcomes())
                self.state.apply_action(self.rng.choices(list(acts), weights=list(probs), k=1)[0])
                continue
            if self.state.is_simultaneous_node():
                # Defensive only: the simultaneous env (goofspiel) is refused up
                # front. Never hang on an unexpected one.
                joint = []
                for pl in range(self.game.num_players()):
                    lg = self.state.legal_actions(pl)
                    joint.append(self.rng.choice(lg) if lg else 0)
                self.state.apply_actions(joint)
                self.turns += 1
                continue
            cp = self.state.current_player()
            if cp < 0 or not self.state.legal_actions(cp):
                self.finish()
                return None
            return "a" if cp == self.a_seat else "b"
        return None

    def apply(self, side: str, action, how: str, on_invalid: str):
        seat = self.a_seat if side == "a" else 1 - self.a_seat
        legal = self.state.legal_actions(seat)
        st = self.stats[side]
        st["turns"] += 1
        st["hows"][how] = st["hows"].get(how, 0) + 1
        if action is None or action not in legal:
            st["invalid" if action is None else "illegal"] += 1
            if on_invalid == "lose":
                self.score = 0.0 if side == "a" else 1.0
                self.done = True
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
        # scoring it a loss would punish long games rather than bad ones.
        self.score = _score(self.state.returns(), self.a_seat) if self.state.is_terminal() else 0.5


def run_cohort(pa, pb, env_name, seeds, sims, max_turn, on_invalid, cohort, log_every):
    """Play every seed from both seats, advancing a cohort in lockstep so each
    side's turns batch into one generate() call instead of one per game."""
    need_bot = isinstance(pa, MCTSPolicy) or isinstance(pb, MCTSPolicy)
    pending = [(s, seat) for s in seeds for seat in (0, 1)]
    total = len(pending)
    live: list = []
    finished: list = []
    t0 = time.time()
    while pending or live:
        while pending and len(live) < cohort:
            s, seat = pending.pop(0)
            live.append(Game(env_name, s, seat, sims, max_turn, need_bot))
        buckets = {"a": [], "b": []}
        for g in live:
            side = g.side_to_move()
            if side:
                buckets[side].append(g)
        for side, policy in (("a", pa), ("b", pb)):
            gs = buckets[side]
            if not gs:
                continue
            jobs = [{"state": g.state,
                     "player": g.a_seat if side == "a" else 1 - g.a_seat,
                     "legal": g.state.legal_actions(g.a_seat if side == "a" else 1 - g.a_seat),
                     "bot": g.bot} for g in gs]
            for g, (action, how) in zip(gs, policy.act_batch(jobs)):
                g.apply(side, action, how, on_invalid)
        still, just = [], []
        for g in live:
            (just if (g.done or g.side_to_move() is None) else still).append(g)
        live, finished = still, finished + just
        if just and log_every and len(finished) % log_every < len(just):
            sc = sum(x.score for x in finished) / len(finished)
            el = time.time() - t0
            print(f"[eval] {len(finished)}/{total}  skor A {sc:.3f}  "
                  f"elapsed {el/60:.1f}m  eta {el/max(len(finished),1)*(total-len(finished))/60:.1f}m",
                  flush=True)
    return finished


def make_policy(spec: str, env_name: str, args, is_opponent: bool):
    """A side is either a named policy (hf/random/teacher/mcts) or a model
    path -- anything that is not a known name is taken as a model."""
    if spec in POLICIES:
        return POLICIES[spec](env_name=env_name, model=args.model,
                              base_model=args.base_model, dtype=args.dtype,
                              max_tokens=args.max_tokens, batch=args.batch)
    if spec == "mcts":
        return MCTSPolicy()
    return HFPolicy(env_name=env_name, model=spec, base_model=args.base_model,
                    dtype=args.dtype, max_tokens=args.max_tokens, batch=args.batch)


def _side_report(rows, side: str, label: str, name: str):
    turns = sum(g.stats[side]["turns"] for g in rows)
    inv = sum(g.stats[side]["invalid"] for g in rows)
    ill = sum(g.stats[side]["illegal"] for g in rows)
    hows: dict = {}
    for g in rows:
        for k, v in g.stats[side]["hows"].items():
            hows[k] = hows.get(k, 0) + v
    print(f"  {label} ({name})")
    print(f"    giliran           : {turns}")
    print(f"    tanpa game_action : {inv} ({100*inv/max(turns,1):.1f}%)  <- forfeit di eval")
    print(f"    id tidak legal    : {ill} ({100*ill/max(turns,1):.1f}%)")
    for k, v in sorted(hows.items(), key=lambda kv: -kv[1])[:4]:
        print(f"    bentuk jawaban    : {k:<26} {v:>6} ({100*v/max(turns,1):.1f}%)")
    return {"turns": turns, "invalid": inv, "illegal": ill, "hows": hows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="teacher",
                    help="side A: hf | random | teacher | mcts | a model path/repo")
    ap.add_argument("--opponent", default="mcts",
                    help="side B: mcts (default) | random | teacher | a model path/repo")
    ap.add_argument("--game", default="othello")
    ap.add_argument("--model", default=None, help="model for --policy hf")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--games", type=int, default=8, help="seeds; each played from BOTH seats")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--max-turn", type=int, default=80)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--cohort", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--on-invalid", default="random", choices=("random", "lose"),
                    help="random: keep playing (move quality stays measurable); "
                         "lose: score the game 0 immediately (what eval does)")
    ap.add_argument("--log-every", type=int, default=8)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.game in UNSUPPORTED:
        print(f"[eval] {args.game}: {UNSUPPORTED[args.game]}")
        return 1
    sims = args.sims if args.sims is not None else EVAL_SIMS.get(args.game, 50)
    if args.policy == "hf" and not args.model:
        print("[eval] --policy hf butuh --model (atau beri path model langsung ke --policy)")
        return 1

    pa = make_policy(args.policy, args.game, args, False)
    pb = make_policy(args.opponent, args.game, args, True)
    name_a = args.model if args.policy == "hf" else args.policy
    name_b = args.opponent
    seeds = list(range(args.seed0, args.seed0 + args.games))
    print(f"[eval] {args.game}: A={name_a}  vs  B={name_b}"
          f"{f' @{sims} sims' if 'mcts' in (args.policy, args.opponent) else ''}  "
          f"({len(seeds)} seed x 2 kursi = {2*len(seeds)} game)  on-invalid={args.on_invalid}")

    t0 = time.time()
    rows = run_cohort(pa, pb, args.game, seeds, sims, args.max_turn,
                      args.on_invalid, args.cohort, args.log_every)
    n = len(rows)
    if not n:
        print("[eval] tidak ada game")
        return 1
    wins = sum(1 for g in rows if g.score == 1.0)
    draws = sum(1 for g in rows if g.score == 0.5)
    losses = n - wins - draws

    print(f"\n=== {args.game}: A vs B ===")
    print(f"  A  : {name_a}")
    print(f"  B  : {name_b}")
    print(f"  {n} game  ->  A menang {wins}, seri {draws}, kalah {losses}")
    print(f"  SKOR A (w+.5d) : {(wins + 0.5*draws)/n:.4f}")
    sa = _side_report(rows, "a", "sisi A", str(name_a))
    sb = _side_report(rows, "b", "sisi B", str(name_b))
    for seat in (0, 1):
        sub = [g for g in rows if g.a_seat == seat]
        if sub:
            print(f"  A di kursi P{seat} : {sum(g.score for g in sub)/len(sub):.3f} atas {len(sub)} game")
    print(f"  waktu : {(time.time()-t0)/60:.1f}m")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "game": args.game, "a": str(name_a), "b": str(name_b), "sims": sims,
            "n": n, "a_wins": wins, "draws": draws, "b_wins": losses,
            "a_score": (wins + 0.5 * draws) / n,
            "a_side": sa, "b_side": sb,
            "games": [{"seed": g.seed, "a_seat": g.a_seat, "score": g.score} for g in rows],
        }, indent=2))
        print(f"  detail -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
