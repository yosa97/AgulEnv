"""Step-synchronised, batch-generating rollout engine for GRPO game envs.

Why this exists
---------------
Every existing environment module runs episodes on a thread pool and then
serialises the only expensive part of them:

    thread_pool = ThreadPoolExecutor(max_workers=len(env_pool))   # shared_env.py
    generation_semaphore = Semaphore(1)                           # shared_env.py
    ...
    with generation_semaphore:                                    # every env module
        generate_rollout_completions(trainer, prompts=[messages], as_chat=True)

The thread pool parallelises the HTTP calls to the env-server, which are cheap,
while every LLM generation waits on a semaphore of size one and is issued as a
batch of exactly one prompt.  On a rollout of B episodes lasting T turns that is
B*T sequential single-prompt generations, and the GPU sits at a batch size of 1
for all of them.

This module inverts the loop.  Episodes advance in lockstep: every live episode
contributes its message list to a single ``generate_rollout_completions`` call
per turn, the completions are handed back, and the env steps run in parallel on
the thread pool.  That is T batched generations instead of B*T single ones, with
vLLM batching across episodes the way it is designed to.

Generation still happens on one thread under the same semaphore, so nothing
about thread-safety changes — only the shape of the work handed to the GPU.

The engine is game-agnostic.  A game supplies a ``GameSpec`` describing how to
turn a server observation into a prompt and how to read legal action ids back
out; everything else — reset, batching, token accounting, action masking,
illegal-move handling, terminal scoring — lives here.
"""

from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Semaphore
from typing import Callable

import requests
from trl.experimental.openenv import generate_rollout_completions

from our_envs.opponent_league import OpponentLeague
from our_envs.shared_env import GAMES_TO_TASK_ID_RANGE, init_env_pool, remove_reasoning_tags

_TIMEOUT = 2400

# Cap on prompts handed to one generation call.  Large enough to be a real
# batch, small enough that a long-context game does not blow the KV cache.
DEFAULT_GEN_CHUNK = 32

# Penalty per illegal action.  Deliberately small next to the +-1 terminal
# signal: the audit's own lesson is that shaping must never outweigh winning.
ILLEGAL_PENALTY = 0.05
# ...and a ceiling on the total, so an episode that spams illegal ids can
# never swamp the +-1 terminal signal it is meant to sit under.
MAX_ILLEGAL_PENALTY = 0.5

_LEGAL_LINE_RE = re.compile(r"^\s*(\d+)\s*->", re.MULTILINE)


def legal_ids_from_observation(obs: str) -> list[int]:
    """Action ids from the rendered ``"<id> -> <label>"`` legal-action block."""
    return [int(m) for m in _LEGAL_LINE_RE.findall(obs or "")]


def parse_action_id(completion_text: str) -> "int | None":
    """Read a bare action id out of a completion.

    Strips reasoning tags, an EOS marker and a ``Action:`` prefix, then takes
    the last integer in what remains — models that emit a short justification
    before the id still parse.
    """
    text = remove_reasoning_tags(completion_text or "")
    text = text.removesuffix("</s>").strip()
    if "Action:" in text:
        text = text.split("Action:")[-1]
    numbers = re.findall(r"-?\d+", text)
    if not numbers:
        return None
    try:
        return int(numbers[-1])
    except ValueError:
        return None


class GameHooks:
    """Per-game scoring, plugged into the generic cohort loop.

    The engine knows about terminal results and illegal actions; anything a
    specific game wants to observe or shape on lives here.  The default is
    outcome-only, which is what a pure win/loss board game wants.
    """

    def on_turn(self, ep: "_Episode", observation: str, action_id, legal_ids: list, illegal: bool) -> None:
        """Called once per model turn, before the env is stepped."""

    def episode_reward(self, ep: "_Episode", outcome: float) -> float:
        """Called once when the episode reaches a terminal state.

        ``outcome`` is the env result normalised to [-1, 1].
        """
        return outcome - min(ILLEGAL_PENALTY * float(ep.illegal_count), MAX_ILLEGAL_PENALTY)


@dataclass
class GameSpec:
    """Everything the engine needs to know about one game."""

    name: str
    # A string, or a zero-arg callable resolved per episode -- goofspiel's
    # curriculum decides per episode whether the strategy hint is attached.
    system_prompt: "str | Callable[[], str]"
    # Builds a FRESH raw-observation -> prompt-body transform per episode.
    # A factory, not a shared callable: othello's transform remembers which
    # colour the agent is playing so it can still label the terminal
    # observation, and sharing one instance across concurrent episodes would
    # let one game's colour leak into another's prompt.
    obs_transform_factory: Callable[[], Callable[[str], str]]
    # Base opponent payload; the league overrides ``mcts_max_simulations``.
    base_opponent: dict = field(default_factory=lambda: {"opponent": "mcts", "mcts_num_rollouts": 1})
    max_turn: int = 70
    # Longest prompt we will keep feeding back before abandoning the episode.
    max_prompt_len: int = 8192
    max_episode_tokens: int = 16384
    # Per-game scoring; the default is outcome minus illegal penalties.
    hooks: GameHooks = field(default_factory=lambda: GameHooks())


@dataclass
class _Episode:
    """Mutable per-episode state carried through the cohort loop."""

    index: int
    game_id: int
    endpoint: str
    opponent_key: str
    rng: random.Random

    episode_id: str = ""
    messages: list = field(default_factory=list)
    observation: str = ""

    prompt_ids: list = field(default_factory=list)
    completion_ids: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    action_mask: list = field(default_factory=list)
    prev_full_ids: "list | None" = None

    last_prompt_ids: list = field(default_factory=list)
    last_completion_ids: list = field(default_factory=list)
    last_logprobs: list = field(default_factory=list)
    opponent_payload: dict = field(default_factory=dict)
    obs_transform: "Callable[[str], str] | None" = None
    stats: dict = field(default_factory=dict)

    turn: int = 0
    done: bool = False
    failed: bool = False
    illegal_count: int = 0
    terminal_raw: "float | None" = None


def _normalize_terminal(raw) -> float:
    """Server terminal reward is [0, 1] with 0.5 = draw; map onto [-1, 1]."""
    try:
        r = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 <= r <= 1.0:
        return (r - 0.5) * 2.0
    return max(-1.0, min(1.0, r))


def _reset_episode(ep: _Episode, spec: GameSpec, opponent_payload: dict) -> None:
    """POST /reset and seed the message list. Marks ``failed`` on error."""
    try:
        res = requests.post(
            f"{ep.endpoint}/reset",
            json={"task_id": ep.game_id, "seed": ep.game_id, **opponent_payload},
            timeout=_TIMEOUT,
        )
        res.raise_for_status()
        block = res.json()["result"]
        ep.episode_id = block.get("episode_id", "")
        ep.observation = ep.obs_transform(block.get("observation", ""))
    except Exception as exc:
        print(f"[{spec.name}] reset failed (game {ep.game_id}): {exc}")
        ep.failed = True
        ep.done = True
        return

    system_prompt = spec.system_prompt() if callable(spec.system_prompt) else spec.system_prompt
    ep.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ep.observation},
    ]


def _step_episode(ep: _Episode, spec: GameSpec, action: str) -> None:
    """POST /step and fold the result back into the episode."""
    try:
        res = requests.post(
            f"{ep.endpoint}/step",
            json={"action": action, "episode_id": ep.episode_id},
            timeout=_TIMEOUT,
        )
        res.raise_for_status()
        block = res.json()["result"]
    except Exception as exc:
        print(f"[{spec.name}] step failed (game {ep.game_id}): {exc}")
        ep.failed = True
        ep.done = True
        return

    ep.done = bool(block.get("done", False))
    if ep.done:
        ep.terminal_raw = block.get("reward", 0)
    else:
        ep.observation = ep.obs_transform(block.get("observation", ""))
        ep.messages.append({"role": "user", "content": ep.observation})


def batched_generate(
    trainer,
    message_lists: list,
    semaphore: Semaphore,
    chunk_size: int = DEFAULT_GEN_CHUNK,
) -> list:
    """One generation call per chunk instead of one per episode.

    Falls back to per-prompt calls if the batched call raises or returns a
    result that is not aligned with its input, so a TRL version that does not
    batch degrades to the old behaviour rather than corrupting the rollout.
    """
    out: list = []
    for start in range(0, len(message_lists), max(1, chunk_size)):
        chunk = message_lists[start : start + max(1, chunk_size)]
        result = None
        try:
            with semaphore:
                result = generate_rollout_completions(trainer, prompts=chunk, as_chat=True)
        except Exception as exc:
            print(f"[batched_generate] batch of {len(chunk)} failed ({exc}); falling back")
            result = None

        if not isinstance(result, list) or len(result) != len(chunk):
            result = []
            for messages in chunk:
                with semaphore:
                    single = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)
                result.append(single[0])
        out.extend(result)
    return out


def _accumulate_full(ep: _Episode, prompt_ids, completion_ids, logprobs) -> None:
    """Prefix-delta token accounting with an action mask.

    mask=1 for tokens the model produced, mask=0 for text the environment
    inserted between turns.
    """
    if ep.turn == 0:
        ep.prompt_ids = list(prompt_ids)
        ep.prev_full_ids = list(prompt_ids)
    else:
        if ep.prev_full_ids is None:
            ep.prev_full_ids = list(prompt_ids)
        elif prompt_ids[: len(ep.prev_full_ids)] != ep.prev_full_ids:
            print(f"[cohort] token shift at turn {ep.turn}; skipping delta mask")
            ep.prev_full_ids = list(prompt_ids)
        else:
            delta = prompt_ids[len(ep.prev_full_ids) :]
            if delta:
                ep.completion_ids.extend(delta)
                ep.logprobs.extend([0.0] * len(delta))
                ep.action_mask.extend([0] * len(delta))
            ep.prev_full_ids = list(prompt_ids)

    if completion_ids:
        ep.completion_ids.extend(completion_ids)
        ep.logprobs.extend(logprobs)
        ep.action_mask.extend([1] * len(completion_ids))
        if ep.prev_full_ids is not None:
            ep.prev_full_ids = ep.prev_full_ids + list(completion_ids)


def run_cohort(
    prompts: list[str],
    trainer,
    spec: GameSpec,
    league: OpponentLeague,
    env_pool: list[dict],
    thread_pool: ThreadPoolExecutor,
    generation_semaphore: Semaphore,
    rank: int,
    use_full_prompt: bool,
    gen_chunk: int = DEFAULT_GEN_CHUNK,
) -> list:
    """Play a whole batch of episodes in lockstep. Returns one dict per prompt.

    Each returned dict carries ``prompt_ids`` / ``completion_ids`` /
    ``logprobs`` / ``reward`` (plus ``action_mask`` in full-prompt mode), or is
    ``None`` for an episode that never reached a terminal state — the caller
    drops those rather than substituting placeholder tokens.
    """
    tokenizer = trainer.processing_class
    num_servers = max(1, len(env_pool))

    episodes: list[_Episode] = []
    for i, prompt in enumerate(prompts):
        game_id = int(prompt)
        opponent = league.sample(random.Random(game_id))
        episodes.append(
            _Episode(
                index=i,
                game_id=game_id,
                endpoint=env_pool[(i + rank) % num_servers]["base_url"],
                opponent_key=opponent.key,
                rng=random.Random(game_id),
                obs_transform=spec.obs_transform_factory(),
            )
        )
        episodes[-1].opponent_payload = opponent.payload

    # --- Reset every episode in parallel (HTTP only) ---
    list(thread_pool.map(
        lambda ep: _reset_episode(ep, spec, ep.opponent_payload),
        episodes,
    ))

    # --- Lockstep turn loop ---
    for turn in range(spec.max_turn):
        live = [ep for ep in episodes if not ep.done and not ep.failed]
        if not live:
            break

        outputs = batched_generate(
            trainer, [ep.messages for ep in live], generation_semaphore, gen_chunk
        )

        actions: list = []
        for ep, out in zip(live, outputs):
            prompt_ids = out.get("prompt_ids", [])
            completion_ids = out.get("completion_ids", [])
            logprobs = out.get("logprobs", [])

            if len(prompt_ids) > spec.max_prompt_len:
                print(f"[{spec.name}] prompt over {spec.max_prompt_len} tokens at turn {turn}; ending episode")
                ep.done = True
                ep.failed = True
                actions.append(None)
                continue

            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            ep.messages.append({"role": "assistant", "content": completion_text})

            if use_full_prompt:
                _accumulate_full(ep, prompt_ids, completion_ids, logprobs)
            ep.last_prompt_ids = list(prompt_ids)
            ep.last_completion_ids = list(completion_ids)
            ep.last_logprobs = list(logprobs)

            legal = legal_ids_from_observation(ep.observation)
            action_id = parse_action_id(completion_text)
            illegal = action_id is None or (legal and action_id not in legal)
            spec.hooks.on_turn(ep, ep.observation, action_id, legal, illegal)
            if illegal:
                ep.illegal_count += 1
                # Keep the game alive on a legal substitute so the terminal
                # result still means something, but never substitute the expert
                # move — that would reward an illegal completion with good play.
                action_id = ep.rng.choice(legal) if legal else None
            actions.append(action_id)

        step_pairs = [
            (ep, str(a)) for ep, a in zip(live, actions) if a is not None and not ep.failed
        ]
        for ep, a in zip(live, actions):
            if a is None and not ep.failed:
                ep.failed = True
                ep.done = True
        list(thread_pool.map(lambda pair: _step_episode(pair[0], spec, pair[1]), step_pairs))

        for ep in live:
            ep.turn += 1

    # --- Score ---
    results: list = [None] * len(prompts)
    for ep in episodes:
        if ep.terminal_raw is None:
            continue  # never finished: no honest outcome to train on
        outcome = _normalize_terminal(ep.terminal_raw)
        league.record(ep.opponent_key, max(0.0, min(1.0, (outcome + 1.0) / 2.0)))
        reward = spec.hooks.episode_reward(ep, outcome)

        if use_full_prompt:
            completion_ids = ep.completion_ids[: spec.max_episode_tokens]
            logprobs = ep.logprobs[: spec.max_episode_tokens]
            action_mask = ep.action_mask[: spec.max_episode_tokens]
            if not completion_ids:
                continue
            results[ep.index] = {
                "prompt_ids": ep.prompt_ids,
                "completion_ids": completion_ids,
                "logprobs": logprobs,
                "action_mask": action_mask,
                "reward": reward,
                "outcome": outcome,
                "illegal": ep.illegal_count,
                "turns": ep.turn,
            }
        else:
            if not ep.last_completion_ids:
                continue
            results[ep.index] = {
                "prompt_ids": ep.last_prompt_ids,
                "completion_ids": ep.last_completion_ids,
                "logprobs": ep.last_logprobs,
                "reward": reward,
                "outcome": outcome,
                "illegal": ep.illegal_count,
                "turns": ep.turn,
            }
    return results



def pack_results(results: list, use_full_prompt: bool) -> dict:
    """Drop unfinished episodes and pack into the dict the trainer expects."""
    valid = [r for r in results if r is not None]
    if not valid:
        # An all-failed batch: return empty lists rather than synthetic tokens.
        empty: dict = {"prompt_ids": [], "completion_ids": [], "logprobs": [], "env_rewards": []}
        if use_full_prompt:
            empty["action_mask"] = []
        return empty

    packed = {
        "prompt_ids": [r["prompt_ids"] for r in valid],
        "completion_ids": [r["completion_ids"] for r in valid],
        "logprobs": [r["logprobs"] for r in valid],
        "env_rewards": [r["reward"] for r in valid],
    }
    if use_full_prompt:
        packed["action_mask"] = [r["action_mask"] for r in valid]
    return packed


def summarise(results: list, name: str, league: OpponentLeague) -> None:
    valid = [r for r in results if r is not None]
    if not valid:
        print(f"[{name}] no episode reached a terminal state this batch")
        return
    wins = sum(1 for r in valid if r["outcome"] > 0) / len(valid)
    print(
        f"[{name}] finished={len(valid)}/{len(results)} win={wins:.1%} "
        f"avg_reward={sum(r['reward'] for r in valid) / len(valid):.3f} "
        f"avg_turns={sum(r['turns'] for r in valid) / len(valid):.1f} "
        f"illegal/ep={sum(r['illegal'] for r in valid) / len(valid):.2f}"
    )
    print(league.format_status())



def run_forced_turn_cohort(
    prompts: list,
    trainer,
    spec: GameSpec,
    league: OpponentLeague,
    env_pool: list,
    thread_pool: ThreadPoolExecutor,
    generation_semaphore: Semaphore,
    rank: int,
    target_turn: int,
    forcing_action: Callable[[str], "int | None"],
    score_fn: Callable[["_Episode", str, "int | None", list, bool, "float | None"], float],
    gen_chunk: int = DEFAULT_GEN_CHUNK,
    playout: bool = True,
) -> list:
    """Strategy-forcing rollout, batched.

    The episode is driven by ``forcing_action`` for ``target_turn`` turns, the
    model plays exactly one turn, and the rest of the game is then played out by
    the forcing policy so the turn has a real terminal result attached to it.

    The forced phase and the playout are pure HTTP, so the whole batch needs
    exactly ONE generation call — against one per episode in the loop this
    replaces.
    """
    tokenizer = trainer.processing_class
    num_servers = max(1, len(env_pool))

    episodes: list[_Episode] = []
    for i, prompt in enumerate(prompts):
        game_id = int(prompt)
        opponent = league.sample(random.Random(game_id))
        ep = _Episode(
            index=i,
            game_id=game_id,
            endpoint=env_pool[(i + rank) % num_servers]["base_url"],
            opponent_key=opponent.key,
            rng=random.Random(game_id),
            obs_transform=spec.obs_transform_factory(),
        )
        ep.opponent_payload = opponent.payload
        episodes.append(ep)

    list(thread_pool.map(lambda e: _reset_episode(e, spec, e.opponent_payload), episodes))

    # --- Forced phase: no generation at all -------------------------------
    for _ in range(max(0, target_turn)):
        live = [e for e in episodes if not e.done and not e.failed]
        if not live:
            break
        pairs = []
        for ep in live:
            action = forcing_action(ep.observation)
            if action is None:
                ep.failed = True
                ep.done = True
                continue
            ep.messages.append({"role": "assistant", "content": str(action)})
            pairs.append((ep, str(action)))
        list(thread_pool.map(lambda pr: _step_episode(pr[0], spec, pr[1]), pairs))
        for ep, _ in pairs:
            ep.turn += 1

    # An episode that ended during forcing has no turn left to train on.
    live = [e for e in episodes if not e.done and not e.failed]
    if not live:
        return [None] * len(prompts)

    # --- The one generated turn -------------------------------------------
    outputs = batched_generate(
        trainer, [ep.messages for ep in live], generation_semaphore, gen_chunk
    )

    results: list = [None] * len(prompts)
    finishers = []
    for ep, out in zip(live, outputs):
        prompt_ids = out.get("prompt_ids", [])
        completion_ids = out.get("completion_ids", [])
        logprobs = out.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        ep.messages.append({"role": "assistant", "content": completion_text})

        legal = legal_ids_from_observation(ep.observation)
        action_id = parse_action_id(completion_text)
        illegal = action_id is None or (legal and action_id not in legal)
        if illegal:
            ep.illegal_count += 1

        ep.last_prompt_ids = list(prompt_ids)
        ep.last_completion_ids = list(completion_ids)
        ep.last_logprobs = list(logprobs)
        ep.stats["observation"] = ep.observation
        ep.stats["action_id"] = action_id
        ep.stats["legal"] = legal
        ep.stats["illegal"] = illegal
        if playout and not illegal:
            finishers.append(ep)

    # --- Play the rest out with the forcing policy (HTTP only) ------------
    def _finish(ep: _Episode) -> None:
        action = str(ep.stats["action_id"])
        for _ in range(spec.max_turn):
            _step_episode(ep, spec, action)
            if ep.done or ep.failed:
                return
            nxt = forcing_action(ep.observation)
            if nxt is None:
                ep.failed = True
                ep.done = True
                return
            action = str(nxt)

    if finishers:
        list(thread_pool.map(_finish, finishers))

    for ep in live:
        if not ep.last_completion_ids:
            continue
        outcome = None if ep.terminal_raw is None else _normalize_terminal(ep.terminal_raw)
        if outcome is not None:
            league.record(ep.opponent_key, max(0.0, min(1.0, (outcome + 1.0) / 2.0)))
        reward = score_fn(
            ep,
            ep.stats.get("observation", ""),
            ep.stats.get("action_id"),
            ep.stats.get("legal", []),
            bool(ep.stats.get("illegal")),
            outcome,
        )
        results[ep.index] = {
            "prompt_ids": ep.last_prompt_ids,
            "completion_ids": ep.last_completion_ids,
            "logprobs": ep.last_logprobs,
            "reward": reward,
            "outcome": 0.0 if outcome is None else outcome,
            "illegal": ep.illegal_count,
            "turns": ep.turn,
        }
    return results

class BoardGameEnv:
    """Ties a GameSpec to an env-server pool and an opponent league.

    One instance per game module.  Holds the pool lazily so importing the
    module (which ``env_configs`` does at import time) never touches the
    network, and exposes the two rollout entry points the registry expects.
    """

    def __init__(self, spec: GameSpec, task_key: str, gen_chunk: int = DEFAULT_GEN_CHUNK):
        self.spec = spec
        self.task_key = task_key
        self.gen_chunk = gen_chunk
        self._state: dict = {}

    # ------------------------------------------------------------- lifecycle
    def _ensure_initialized(self, trainer) -> None:
        if self._state.get("initialized"):
            return

        league = OpponentLeague.from_sims_ladder(dict(self.spec.base_opponent))
        warmup = dict(league.opponents[0].payload)
        warmup.update({"task_id": GAMES_TO_TASK_ID_RANGE[self.task_key][0], "seed": 42})
        rank, env_pool, num_servers, thread_pool, semaphore = init_env_pool(warmup)

        print(
            f"[{self.spec.name}] pool ready: {num_servers} server(s), "
            f"league={[o.key for o in league.opponents]}, gen_chunk={self.gen_chunk}"
        )
        self._state.update(
            initialized=True,
            rank=rank,
            env_pool=env_pool,
            thread_pool=thread_pool,
            semaphore=semaphore,
            league=league,
        )

    # --------------------------------------------------------------- rollout
    def _run(self, prompts: list, trainer, use_full_prompt: bool, max_turns=None) -> dict:
        self._ensure_initialized(trainer)
        spec = self.spec
        if max_turns:
            spec = GameSpec(**{**spec.__dict__, "max_turn": int(max_turns)})

        results = run_cohort(
            prompts=prompts,
            trainer=trainer,
            spec=spec,
            league=self._state["league"],
            env_pool=self._state["env_pool"],
            thread_pool=self._state["thread_pool"],
            generation_semaphore=self._state["semaphore"],
            rank=self._state["rank"],
            use_full_prompt=use_full_prompt,
            gen_chunk=self.gen_chunk,
        )
        summarise(results, spec.name, self._state["league"])
        return pack_results(results, use_full_prompt)

    def rollout_full(self, prompts: list, trainer, max_turns=None) -> dict:
        return self._run(prompts, trainer, use_full_prompt=True, max_turns=max_turns)

    def rollout_last(self, prompts: list, trainer, max_turns=None) -> dict:
        return self._run(prompts, trainer, use_full_prompt=False, max_turns=max_turns)
