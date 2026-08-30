import os
import random
import re
import requests
from trl.experimental.openenv import generate_rollout_completions

from our_envs.batched_rollout import (
    GameHooks,
    GameSpec,
    pack_results,
    run_cohort,
    run_forced_turn_cohort,
    summarise,
)
from our_envs.opponent_league import OpponentLeague
from our_envs.shared_env import (
    GAMES_TO_TASK_ID_RANGE,
    CurriculumScheduler,
    init_env_pool,
    rollout_reward_func,  # re-exported for callers
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SELECTED_GAME = "goofspiel"
_MAX_EPISODE_TOKENS = 16384
_MAX_PROMPT_LEN = 4225
_MAX_PLAYOUT_TURNS = 32  # hard stop for the post-training-turn forcing playout
_TIMEOUT = 2400

# Reward shaping parameters (full-prompt variant)
_STRATEGY_REWARD_WEIGHT = 0.5

# Reward parameters (last-prompt variant)
_STRATEGY_REWARD = 1.0
_INVALID_PENALTY = -0.1


def _env_float(name: str, default: float) -> float:
    """Env-var float that never raises at import time (audit P1 pattern)."""
    try:
        return float(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


# --- Outcome / shaping knobs, env-overridable so they can be A/B-ed ---------
# Weight on the TERMINAL game result.  The old last-prompt reward scored only
# "did the bid match the prize card" and multiplied it by prompt/completion
# length ratio, so the shortest completion won regardless of the game.  The
# game is now played out with the forcing policy after the training turn, so
# the terminal reward reflects the bid the model actually chose.
# 0.0 reproduces the pre-fix behaviour (strategy adherence only).
_OUTCOME_WEIGHT = max(0.0, min(1.0, _env_float("GOOFSPIEL_OUTCOME_WEIGHT", 0.3)))

# Partial credit for a near-miss bid, scaled by rank distance from the target
# card.  0.0 = all-or-nothing, which is the pre-fix behaviour.
_NEAR_MISS_REWARD = max(0.0, _env_float("GOOFSPIEL_NEAR_MISS", 0.0))

# Weight on strategy adherence vs terminal result in the full-prompt variant.
_STRATEGY_WEIGHT_FULL = max(0.0, min(1.0, _env_float("GOOFSPIEL_STRATEGY_WEIGHT", _STRATEGY_REWARD_WEIGHT)))

REASONING_TAG_PAIRS = [
    ("think", "think"), ("thinking", "thinking"), ("reasoning", "reasoning"),
    ("thought", "thought"), ("reflection", "reflection"),
]


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def extract_and_format_observation(obs_text: str) -> str:
    if 'Invalid action:' in obs_text and 'Legal Actions:' in obs_text:
        return obs_text
    state_match = re.search(r'Current State:\n(.*)', obs_text, re.DOTALL)
    if not state_match:
        return obs_text
    state_text = state_match.group(0)
    state_text = re.sub(r'\n\nWaiting for Player -2 to move\.\.\.$', '', state_text)
    player_match = re.search(r'You are Player (\d+)', obs_text)
    player_id = int(player_match.group(1)) if player_match else 0
    hand_match = re.search(rf'P{player_id} hand: ([\d\s]+)', state_text)
    if not hand_match:
        return state_text
    cards = [int(c) for c in hand_match.group(1).strip().split()]
    legal_actions = [f"{c - 1} -> [P{player_id}]Bid: {c}" for c in cards]
    return (
        state_text
        + "\n\nYou are Player " + str(player_id) + ".\nLegal Actions:\n"
        + "\n".join(legal_actions)
        + "\n\nYour choice (ID only):"
    )


def extract_prize_card(obs_text: str) -> "int | None":
    m = re.search(r'Current point card:\s*(\d+)', obs_text)
    return int(m.group(1)) if m else None




def get_hand_cards(observation_text: str, player_id: int = 0) -> list[int]:
    m = re.search(rf"P{player_id} hand:\s*([\d ]+)", observation_text)
    if not m:
        return []
    return [int(c) for c in m.group(1).strip().split()]


def target_card_for(prize_card: "int | None", hand_cards: list[int]) -> "int | None":
    """The card the forcing strategy would play this turn.

    While we still hold the matching card, that is the target: bid == prize is
    the proven-strong baseline this env distils, and with a full hand the rule
    below reduces to exactly that.

    Once the matching card has been spent, matching is *impossible* — and the
    old reward handed out 0.0 for every legal action on those turns, so the
    whole GRPO group scored identically and the advantage came entirely from
    the length term.  Fall back to the rank-relative bid the SFT expert uses
    (``goofspiel_trajectories._bid_weights``): map the prize's strength within
    the deck onto a rank within our remaining hand.  Unlike "cheapest card that
    still beats it", this does not burn a high card on a mid-value prize.
    """
    if prize_card is None or not hand_cards:
        return None
    cards = sorted(hand_cards)
    if prize_card in cards:
        return prize_card
    m = len(cards)
    if m == 1:
        return cards[0]
    deck_size = max(cards[-1], prize_card)
    prize_frac = (prize_card - 1) / max(deck_size - 1, 1)
    idx = int(round(prize_frac * (m - 1)))
    return cards[min(max(idx, 0), m - 1)]


def rank_credit(bid_card: "int | None", target: "int | None", hand_cards: list[int]) -> float:
    """1.0 for the target card, decaying linearly with rank distance in hand."""
    if bid_card is None or target is None:
        return 0.0
    cards = sorted(hand_cards)
    if len(cards) <= 1 or bid_card not in cards or target not in cards:
        return 0.0
    gap = abs(cards.index(bid_card) - cards.index(target))
    return max(0.0, 1.0 - gap / (len(cards) - 1))




def remove_reasoning_tags(text: str) -> str:
    cleaned = text
    for tag_name, close_name in REASONING_TAG_PAIRS:
        cleaned = re.sub(rf"<{tag_name}>.*?</{close_name}>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        close_tag = f"</{close_name}>"
        if close_tag in cleaned:
            cleaned = cleaned.split(close_tag)[-1]
        open_match = re.search(rf"<{tag_name}>", cleaned, flags=re.IGNORECASE)
        if open_match:
            cleaned = cleaned[: open_match.start()]
    cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Shared system prompt pieces
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = (
    "You are playing goofspiel.\n\n# Game Rules\n"
    "GOOFSPIEL RULES:\nSetup: Each player has bid cards numbered 1 to N. "
    "A prize deck with cards 1 to N is shuffled.\n"
    "Goal: Win the most points by bidding on prize cards.\n\n"
    "Each turn:\n1. Reveal top prize card (worth its face value in points)\n"
    "2. Players simultaneously play one bid card from their hand\n"
    "3. Highest bidder wins the prize card (adds its value to score)\n"
    "4. If bids tie, prize card is discarded (no one gets points)\n\n"
    "Winning: Player with most points after all rounds wins.\n\n\n"
    "# Output Format\nYou must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    'Examples:\n- For action "0 -> roll": respond "0"\n- For action "89 -> a3": respond "89"'
)

_HINT_PROMPT_LAST = (
    "\n\nDon't think for long. The best strategies is to bid the card with same value as the point card \n\n"
    "Example: \nIf the point card is 1, bid using card 1, likely action ID 0\n"
    "If the point card is 13, bid using card 13, likely action ID 12\n"
    "If the point card is 10, bid using card 10, likely action ID 9\n"
    "Always bid following this strategy to maximize your winning chance."
)

_HINT_PROMPT_FULL = (
    "\n\nThe best strategies is to bid the card with same value as the point card \n\n"
    "Example: \nIf the point card is 1, bid using card 1, likely action ID 0\n"
    "If the point card is 13, bid using card 13, likely action ID 12\n"
    "If the point card is 10, bid using card 10, likely action ID 9\n"
    "Always bid following this strategy to maximize your winning chance."
)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_state: dict = {}


def _curriculum_factory(args) -> CurriculumScheduler:
    """Construct this env's curriculum from training args. Referenced by env_configs registry."""
    return CurriculumScheduler(
        initial_max_turn=args.initial_max_turn,
        final_max_turn=13,
        rollouts_per_stage=args.rollouts_per_stage,
        initial_hint_prob=0.75,
        final_hint_prob=0.0,
        warmup_rollouts=args.rollouts_per_stage,
    )


def _ensure_initialized(trainer) -> None:
    if _state.get("initialized"):
        return

    reset_payload = {
        "task_id": GAMES_TO_TASK_ID_RANGE[_SELECTED_GAME][0],
        "seed": 42,
        "opponent": "mcts",
    }
    rank, env_pool, num_servers, thread_pool, generation_semaphore = init_env_pool(reset_payload)

    curriculum = _curriculum_factory(trainer.args)
    print(
        f"[CURRICULUM] Initialized: initial_max_turn={trainer.args.initial_max_turn}, "
        f"final_max_turn=13, rollouts_per_stage={trainer.args.rollouts_per_stage}, "
        "initial_hint_prob=0.75"
    )

    league = OpponentLeague.from_sims_ladder({"opponent": "mcts", "mcts_num_rollouts": 1})
    print(f"[goofspiel] league={[o.key for o in league.opponents]}")

    _state.update(
        initialized=True,
        rank=rank,
        env_pool=env_pool,
        num_servers=num_servers,
        thread_pool=thread_pool,
        generation_semaphore=generation_semaphore,
        curriculum=curriculum,
        league=league,
    )


# ---------------------------------------------------------------------------
# Public rollout functions
# ---------------------------------------------------------------------------

def rollout_first_prompt_and_completion(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """Non-parallelised legacy single-server rollout (first prompt only)."""
    if not getattr(rollout_first_prompt_and_completion, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_list = [u.strip() for u in raw_urls.split(",") if u.strip()]
        base_url = server_list[rank % len(server_list)] if server_list else ""
        rollout_first_prompt_and_completion.base_url = base_url
        try:
            payload = {"task_id": GAMES_TO_TASK_ID_RANGE[_SELECTED_GAME][0], "seed": 42, "opponent": "mcts"}
            requests.post(f"{base_url}/reset", json=payload, timeout=300).raise_for_status()
            rollout_first_prompt_and_completion.initialized = True
        except Exception as exc:
            raise RuntimeError(f"Failed to init: {exc}") from exc

    env_endpoint = rollout_first_prompt_and_completion.base_url
    tokenizer = trainer.processing_class
    TIMEOUT = _TIMEOUT

    all_prompt_ids, all_completion_ids, all_logprobs, all_rewards = [], [], [], []
    game_id = random.randint(*GAMES_TO_TASK_ID_RANGE[_SELECTED_GAME])

    for prompt in prompts:
        episode_prompt_ids:    list[int]   = []
        episode_completion_ids: list[int]  = []
        episode_logprobs:      list[float] = []
        done = False
        train_reward = 0.0
        turn_number  = 0

        try:
            reset_res = requests.post(
                f"{env_endpoint}/reset",
                json={"task_id": game_id, "seed": game_id, "opponent": "mcts"},
                timeout=TIMEOUT,
            )
            reset_res.raise_for_status()
            result_block = reset_res.json()["result"]
            episode_id = result_block.get("episode_id", "")
            current_observation = result_block.get("observation", "")
            current_observation += 'Your output must strictly follow this format: "Thought:\nyour thoughts ONLY in text.\n\nAction:\nONLY your action ID (a single number)."'
        except Exception as exc:
            print(f"Failed to reset environment (Game {game_id}): {exc}")
            continue

        messages = [{"role": "user", "content": current_observation}]

        while not done and turn_number < max_turns:
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]
            prompt_ids     = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs       = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            if turn_number == 0:
                episode_prompt_ids    = prompt_ids
                episode_completion_ids = completion_ids
                episode_logprobs       = logprobs

            messages.append({"role": "assistant", "content": completion_text})

            action_to_send = remove_reasoning_tags(completion_text).removesuffix("</s>").strip()
            if "Action:" in action_to_send:
                action_to_send = action_to_send.split("Action:")[-1].strip()

            try:
                step_res = requests.post(
                    f"{env_endpoint}/step",
                    json={"action": action_to_send, "episode_id": episode_id},
                    timeout=TIMEOUT,
                )
                step_res.raise_for_status()
                step_block = step_res.json()["result"]
                step_state  = step_block.get("observation", "")
                step_reward = step_block.get("reward", 0)
                done        = step_block.get("done", False)
                formatted_observation = step_state
            except Exception as exc:
                print(f"Step failed: {exc}")
                formatted_observation = "Invalid Action.\n\n"
                step_reward = -0.01
                done = False

            if done:
                train_reward = step_reward
            else:
                messages.append({"role": "user", "content": formatted_observation})
            turn_number += 1

        all_prompt_ids.append(episode_prompt_ids)
        all_completion_ids.append(episode_completion_ids)
        all_logprobs.append(episode_logprobs)
        all_rewards.append(train_reward)

    return {
        "prompt_ids":  all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs":    all_logprobs,
        "env_rewards": all_rewards,
    }




# ---------------------------------------------------------------------------
# Cohort wiring
#
# The per-episode loops this replaced ran one thread per episode and issued one
# single-prompt generation each, behind a size-1 semaphore.  Both variants now
# go through the shared cohort engine: episodes advance in lockstep and every
# live episode contributes to one batched generation per turn.  For the
# strategy-forcing variant the forced phase and the playout are pure HTTP, so
# the whole batch costs exactly ONE generation call.
# ---------------------------------------------------------------------------

def _forcing_action(observation: str) -> "int | None":
    """The bid the forcing policy plays: the target card, as an action id."""
    target = target_card_for(extract_prize_card(observation), get_hand_cards(observation))
    return None if target is None else target - 1


def _system_prompt_factory():
    """Resolve the hint decision per episode, off the live curriculum."""
    def _make() -> str:
        hint_prob = _state["curriculum"].get_hint_prob() if _state.get("curriculum") else 0.0
        return _BASE_SYSTEM_PROMPT + (_HINT_PROMPT_FULL if random.random() < hint_prob else "")
    return _make


class _GoofspielHooks(GameHooks):
    """Strategy adherence measured against the rank-relative target card.

    Counted per turn without the old ``all_steps_correct`` latch, so a turn
    played correctly still counts after an earlier miss.
    """

    def on_turn(self, ep, observation, action_id, legal_ids, illegal) -> None:
        target = target_card_for(extract_prize_card(observation), get_hand_cards(observation))
        bid = None if action_id is None else action_id + 1
        ep.stats["opportunities"] = ep.stats.get("opportunities", 0) + 1
        if bid is not None and target is not None and bid == target:
            ep.stats["hits"] = ep.stats.get("hits", 0) + 1

    def episode_reward(self, ep, outcome: float) -> float:
        opportunities = ep.stats.get("opportunities", 0)
        ratio = ep.stats.get("hits", 0) / opportunities if opportunities else 0.0
        reward = (
            _STRATEGY_WEIGHT_FULL * ratio
            + (1.0 - _STRATEGY_WEIGHT_FULL) * outcome
        )
        return reward - 0.05 * float(ep.illegal_count)


def _score_forced_turn(ep, observation, action_id, legal_ids, illegal, outcome) -> float:
    """Reward for the single generated turn of the strategy-forcing variant."""
    if illegal:
        return _INVALID_PENALTY

    hand = get_hand_cards(observation)
    target = target_card_for(extract_prize_card(observation), hand)
    bid = None if action_id is None else action_id + 1
    followed = bid is not None and target is not None and bid == target

    strategy_term = (
        _STRATEGY_REWARD if followed else _NEAR_MISS_REWARD * rank_credit(bid, target, hand)
    )
    if _OUTCOME_WEIGHT <= 0.0:
        return strategy_term
    # A playout that never reached a terminal state scores as a draw rather
    # than falling back to the unblended term -- otherwise samples inside one
    # GRPO group would sit on two different reward scales.
    settled = 0.0 if outcome is None else outcome
    return (1.0 - _OUTCOME_WEIGHT) * strategy_term + _OUTCOME_WEIGHT * settled


def _spec(max_turn: int) -> GameSpec:
    return GameSpec(
        name="goofspiel",
        system_prompt=_system_prompt_factory(),
        obs_transform_factory=lambda: extract_and_format_observation,
        max_turn=max_turn,
        max_prompt_len=_MAX_PROMPT_LEN,
        max_episode_tokens=_MAX_EPISODE_TOKENS,
        hooks=_GoofspielHooks(),
    )


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """Strategy-forcing rollout: one generated turn per episode, one batched call."""
    _ensure_initialized(trainer)
    curriculum = _state["curriculum"]
    current_max_turn = curriculum.get_max_turn()
    print(
        f"[CURRICULUM] Rollout {curriculum.total_rollouts}: max_turn={current_max_turn}, "
        f"hint_prob={curriculum.get_hint_prob():.2f}"
    )

    results = run_forced_turn_cohort(
        prompts=prompts,
        trainer=trainer,
        spec=_spec(current_max_turn),
        league=_state["league"],
        env_pool=_state["env_pool"],
        thread_pool=_state["thread_pool"],
        generation_semaphore=_state["generation_semaphore"],
        rank=_state["rank"],
        target_turn=max(0, current_max_turn - 1),
        forcing_action=_forcing_action,
        score_fn=_score_forced_turn,
        playout=_OUTCOME_WEIGHT > 0.0,
    )
    curriculum.step(len(prompts))
    summarise(results, "goofspiel/forced", _state["league"])
    return pack_results(results, use_full_prompt=False)


def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """Whole-game rollout with action masking, batched across the cohort."""
    _ensure_initialized(trainer)
    curriculum = _state["curriculum"]
    current_max_turn = curriculum.get_max_turn()
    print(
        f"[CURRICULUM] Rollout {curriculum.total_rollouts}: max_turn={current_max_turn}, "
        f"hint_prob={curriculum.get_hint_prob():.2f}"
    )

    results = run_cohort(
        prompts=prompts,
        trainer=trainer,
        spec=_spec(current_max_turn),
        league=_state["league"],
        env_pool=_state["env_pool"],
        thread_pool=_state["thread_pool"],
        generation_semaphore=_state["generation_semaphore"],
        rank=_state["rank"],
        use_full_prompt=True,
    )
    curriculum.step(len(prompts))
    summarise(results, "goofspiel/full", _state["league"])
    return pack_results(results, use_full_prompt=True)
