import functools
import os
import random
import re
from concurrent.futures import as_completed
from threading import Semaphore

import requests
from trl.experimental.openenv import generate_rollout_completions

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


def extract_bid_from_action(action_text: str, obs_text: str) -> "int | None":
    try:
        return int(action_text.strip()) + 1
    except Exception:
        return None


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


def normalize_outcome(env_reward) -> float:
    """Map the env-server terminal reward onto [-1, 1].

    The other envs in this repo treat the goofspiel/PvP terminal reward as
    [0, 1] with 0.5 = draw (see liar_dice's ``(r - 0.5) * 2``), so that is the
    convention assumed here; anything outside [0, 1] is passed through clipped
    rather than rescaled.  CONFIRM against the env-server you deploy.
    """
    try:
        r = float(env_reward)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 <= r <= 1.0:
        return (r - 0.5) * 2.0
    return max(-1.0, min(1.0, r))


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

    _state.update(
        initialized=True,
        rank=rank,
        env_pool=env_pool,
        num_servers=num_servers,
        thread_pool=thread_pool,
        generation_semaphore=generation_semaphore,
        curriculum=curriculum,
    )


# ---------------------------------------------------------------------------
# Episode runners
# The two rollout variants differ significantly in game logic:
#   - last:  "strategy forcing" — plays optimal moves until target_turn, then
#            trains on that single turn only (no full-prompt accumulation).
#   - full:  standard multi-turn loop with strategy reward shaping and full
#            token accumulation + action masking.
# They share _ensure_initialized and the outer dispatch boilerplate.
# ---------------------------------------------------------------------------

def _play_out_with_forcing(
    env_endpoint: str,
    episode_id: str,
    first_action_id: str,
    observation: str,
) -> "float | None":
    """Send ``first_action_id``, then finish the game with the forcing policy.

    Returns the env-server's terminal reward, or ``None`` if the game did not
    reach a terminal state (transport error, or the turn cap was hit).  No LLM
    generation happens here — only HTTP — so this costs env round-trips, not
    the generation semaphore that actually bottlenecks the rollout.
    """
    action = first_action_id
    obs = observation
    for _ in range(_MAX_PLAYOUT_TURNS):
        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block = step_res.json()["result"]
        except Exception as exc:
            print(f"Playout step failed: {exc}")
            return None

        if step_block.get("done", False):
            return step_block.get("reward", 0)

        obs = extract_and_format_observation(step_block.get("observation", ""))
        prize = extract_prize_card(obs)
        hand = get_hand_cards(obs)
        target = target_card_for(prize, hand)
        if target is None:
            return None
        action = str(target - 1)
    return None


def _run_episode_last(
    index: int,
    prompt: str,
    *,
    env_pool: list[dict],
    num_servers: int,
    rank: int,
    trainer,
    tokenizer,
    generation_semaphore: Semaphore,
    current_max_turn: int,
    current_hint_prob: float,
) -> tuple[int, "dict | None"]:
    """Strategy-forcing rollout: plays N-1 turns with the optimal strategy,
    then trains on the single target turn."""
    game_id = int(prompt)
    server_idx   = (index + rank) % num_servers
    env_endpoint = env_pool[server_idx]["base_url"]

    done              = False
    turn_number       = 0
    target_turn       = current_max_turn - 1
    use_hints         = random.random() < current_hint_prob

    # --- Reset environment ---
    try:
        reset_res = requests.post(
            f"{env_endpoint}/reset",
            json={"task_id": game_id, "seed": game_id, "opponent": "mcts"},
            timeout=_TIMEOUT,
        )
        reset_res.raise_for_status()
        result_block = reset_res.json()["result"]
        episode_id = result_block.get("episode_id", "")
        formatted_observation = extract_and_format_observation(result_block.get("observation", ""))
    except Exception as exc:
        print(f"Failed to reset environment (Game {game_id}): {exc}")
        return index, None

    system_prompt = _BASE_SYSTEM_PROMPT + (_HINT_PROMPT_LAST if use_hints else "")
    messages = [{"role": "system", "content": system_prompt}]

    # --- Strategy-forcing phase (turns 0 .. target_turn - 1) ---
    while not done and turn_number < target_turn:
        messages.append({"role": "user", "content": formatted_observation})
        hand_cards = get_hand_cards(formatted_observation)
        if len(hand_cards) <= 1:
            target_turn = turn_number
            break
        prize_card = extract_prize_card(formatted_observation)
        action_id  = prize_card - 1
        messages.append({"role": "assistant", "content": str(action_id)})
        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": str(action_id), "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block = step_res.json()["result"]
            formatted_observation = extract_and_format_observation(step_block.get("observation", ""))
            done = step_block.get("done", False)
        except Exception as exc:
            print(f"Step failed: {exc}")
            done = False
        turn_number += 1

    if done:
        print(f"[GT] Game {game_id} ended during strategy forcing at turn {turn_number}.")
        return index, None

    # --- Training turn ---
    messages.append({"role": "user", "content": formatted_observation})
    prize_card = extract_prize_card(formatted_observation)

    with generation_semaphore:
        rollout_out = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

    prompt_ids     = rollout_out.get("prompt_ids", [])
    completion_ids = rollout_out.get("completion_ids", [])
    logprobs       = rollout_out.get("logprobs", [])
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    messages.append({"role": "assistant", "content": completion_text})

    # removesuffix, not [:-5]: "</s>" is 4 chars, the old slice ate a digit.
    action_to_send = remove_reasoning_tags(completion_text).removesuffix("</s>").strip()
    if "Action:" in action_to_send:
        action_to_send = action_to_send.split("Action:")[-1].strip()

    hand_cards = get_hand_cards(formatted_observation)
    bid_card   = extract_bid_from_action(action_to_send, formatted_observation)
    target     = target_card_for(prize_card, hand_cards)

    # Legality is a property of the CARD, not the action id.  The observation
    # renders "id -> Bid: card" with id = card - 1, so the old check
    # (`action_id not in hand_cards`) compared an id against card values: it
    # flagged every bid of card 1 as invalid and mis-scored the rest.
    invalid_action = bid_card is None or bid_card not in hand_cards
    if invalid_action:
        print(f"Invalid action: {action_to_send!r} -> card {bid_card} not in hand {hand_cards}")

    strategy_followed = (not invalid_action) and target is not None and bid_card == target

    # Terminal result of the game this bid actually led to.  Only meaningful
    # when the bid was legal — an illegal bid never reaches the env.
    outcome = None
    if _OUTCOME_WEIGHT > 0.0 and not invalid_action:
        raw_outcome = _play_out_with_forcing(
            env_endpoint, episode_id, str(bid_card - 1), formatted_observation
        )
        # A failed playout scores as a draw rather than falling back to the
        # unblended strategy term — otherwise samples inside one GRPO group
        # would sit on two different reward scales.
        outcome = normalize_outcome(raw_outcome) if raw_outcome is not None else 0.0

    if invalid_action:
        reward = _INVALID_PENALTY
    else:
        strategy_term = (
            _STRATEGY_REWARD if strategy_followed
            else _NEAR_MISS_REWARD * rank_credit(bid_card, target, hand_cards)
        )
        if outcome is None:
            reward = strategy_term
        else:
            reward = (1.0 - _OUTCOME_WEIGHT) * strategy_term + _OUTCOME_WEIGHT * outcome

    print("--------------------------------")
    print(
        f"[GT] game={game_id} train_turn={target_turn} prize={prize_card} "
        f"bid={bid_card} target={target} strategy={strategy_followed} "
        f"outcome={'n/a' if outcome is None else f'{outcome:.2f}'} "
        f"reward={reward:.3f} hints={use_hints}"
    )
    print("--------------------------------")

    return index, {
        "prompt_ids":        prompt_ids,
        "completion_ids":    completion_ids,
        "logprobs":          logprobs,
        "reward":            reward,
        "strategy_followed": strategy_followed,
    }


def _run_episode_full(
    index: int,
    prompt: str,
    *,
    env_pool: list[dict],
    num_servers: int,
    rank: int,
    trainer,
    tokenizer,
    generation_semaphore: Semaphore,
    current_max_turn: int,
    current_hint_prob: float,
) -> tuple[int, "dict | None"]:
    """Full-prompt rollout with strategy reward shaping and action masking."""
    game_id = int(prompt)
    server_idx   = (index + rank) % num_servers
    env_endpoint = env_pool[server_idx]["base_url"]

    episode_prompt_ids:    list[int]   = []
    episode_completion_ids: list[int]  = []
    episode_logprobs:      list[float] = []
    episode_action_mask:   list[int]   = []
    prev_full_ids: "list[int] | None"  = None

    invalid_count              = 0
    done                       = False
    train_reward               = 0.0
    turn_number                = 0
    strategy_followed_count    = 0
    total_strategy_opportunities = 0
    use_hints = random.random() < current_hint_prob

    # --- Reset environment ---
    try:
        reset_res = requests.post(
            f"{env_endpoint}/reset",
            json={"task_id": game_id, "seed": game_id, "opponent": "mcts"},
            timeout=_TIMEOUT,
        )
        reset_res.raise_for_status()
        result_block = reset_res.json()["result"]
        episode_id = result_block.get("episode_id", "")
        formatted_observation = extract_and_format_observation(result_block.get("observation", ""))
    except Exception as exc:
        print(f"Failed to reset environment (Game {game_id}): {exc}")
        return index, None

    system_prompt = _BASE_SYSTEM_PROMPT + (_HINT_PROMPT_FULL if use_hints else "")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": formatted_observation},
    ]

    # --- Interaction loop ---
    while not done and turn_number < current_max_turn:
        prize_card = extract_prize_card(formatted_observation)

        with generation_semaphore:
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

        prompt_ids     = rollout_outputs.get("prompt_ids", [])
        completion_ids = rollout_outputs.get("completion_ids", [])
        logprobs       = rollout_outputs.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        if len(prompt_ids) > _MAX_PROMPT_LEN:
            print(f"Warning: Prompt exceeded {_MAX_PROMPT_LEN} tokens at turn {turn_number}, ending early")
            done = True
            break

        # --- Token accumulation ---
        if turn_number == 0:
            episode_prompt_ids = prompt_ids
            prev_full_ids = prompt_ids.copy()
        else:
            if prev_full_ids is None:
                prev_full_ids = prompt_ids.copy()
            elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                print(f"Warning: token shift at turn {turn_number}. Skipping delta mask.")
                prev_full_ids = prompt_ids.copy()
            else:
                delta = prompt_ids[len(prev_full_ids):]
                if delta:
                    episode_completion_ids.extend(delta)
                    episode_logprobs.extend([0.0] * len(delta))
                    episode_action_mask.extend([0] * len(delta))
                prev_full_ids = prompt_ids.copy()

        if completion_ids:
            episode_completion_ids.extend(completion_ids)
            episode_logprobs.extend(logprobs)
            episode_action_mask.extend([1] * len(completion_ids))
            if prev_full_ids is not None:
                prev_full_ids = prev_full_ids + completion_ids

        messages.append({"role": "assistant", "content": completion_text})

        # --- Parse action ---
        # removesuffix, not [:-5]: "</s>" is 4 chars and the old slice ate the
        # last digit of the action id.  remove_reasoning_tags matches the
        # last-prompt path, which already stripped <think> blocks.
        action_to_send = remove_reasoning_tags(completion_text).removesuffix("</s>").strip()
        if "Action:" in action_to_send:
            action_to_send = action_to_send.split("Action:")[-1].strip()

        # --- Strategy adherence check ---
        # No `all_steps_correct` latch: a turn played correctly counts even
        # after an earlier miss.  The latch zeroed every later correct turn, so
        # credit assignment collapsed at the first deviation.
        bid_card = extract_bid_from_action(action_to_send, formatted_observation)
        target   = target_card_for(prize_card, get_hand_cards(formatted_observation))
        total_strategy_opportunities += 1
        if bid_card is not None and target is not None and bid_card == target:
            strategy_followed_count += 1

        # --- Step environment ---
        try:
            formatted_observation = ""
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action_to_send, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block = step_res.json()["result"]
            formatted_observation = extract_and_format_observation(step_block.get("observation", ""))
            step_reward           = step_block.get("reward", 0)
            done                  = step_block.get("done", False)
        except Exception as exc:
            print(f"Step failed: {exc}")
            step_reward = -0.01
            done = False
            invalid_count += 1

        if "Nothing happens" in formatted_observation or "Invalid" in formatted_observation:
            invalid_count += 1

        if done:
            train_reward = step_reward
        else:
            messages.append({"role": "user", "content": formatted_observation})

        turn_number += 1

    if len(episode_completion_ids) > _MAX_EPISODE_TOKENS:
        episode_completion_ids = episode_completion_ids[:_MAX_EPISODE_TOKENS]
        episode_logprobs       = episode_logprobs[:_MAX_EPISODE_TOKENS]
        episode_action_mask    = episode_action_mask[:_MAX_EPISODE_TOKENS]

    strategy_ratio = strategy_followed_count / total_strategy_opportunities if total_strategy_opportunities else 0.0

    # Adherence and outcome now share one scale and sum to at most 1.0 before
    # penalties.  Previously `immediate_rewards` (0.1 per correct turn, up to
    # 1.3 over a 13-turn game) was added raw on top of a 0.5-weighted terminal
    # reward, so playing in-style outweighed winning.
    if done:
        shaped_reward = (
            _STRATEGY_WEIGHT_FULL * strategy_ratio
            + (1.0 - _STRATEGY_WEIGHT_FULL) * normalize_outcome(train_reward)
        )
    else:
        # No terminal result to score, so adherence only.
        shaped_reward = _STRATEGY_WEIGHT_FULL * strategy_ratio
    shaped_reward -= 0.05 * float(invalid_count)

    print(
        "[ID:{:<6} Done:{} T:{:>2d} | EnvR:{:>6.2f} | TrainR:{:>6.2f} | Inv:{:<2}]".format(
            str(game_id)[:6], int(done), turn_number, train_reward, shaped_reward, invalid_count,
        )
    )

    return index, {
        "prompt_ids":     episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "action_mask":    episode_action_mask,
        "logprobs":       episode_logprobs,
        "reward":         shaped_reward,
        "strategy_ratio": strategy_ratio,
        "final_score":    train_reward,
    }


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


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """Strategy-forcing rollout (trains on a single target turn per episode)."""
    _ensure_initialized(trainer)

    curriculum        = _state["curriculum"]
    current_max_turn  = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(f"[CURRICULUM] Rollout {curriculum.total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

    run = functools.partial(
        _run_episode_last,
        env_pool=_state["env_pool"],
        num_servers=_state["num_servers"],
        rank=_state["rank"],
        trainer=trainer,
        tokenizer=trainer.processing_class,
        generation_semaphore=_state["generation_semaphore"],
        current_max_turn=current_max_turn,
        current_hint_prob=current_hint_prob,
    )

    _fallback = {"prompt_ids": [1], "completion_ids": [1], "logprobs": [1.0], "reward": 0.0, "strategy_followed": False}

    results = [None] * len(prompts)
    futures = [_state["thread_pool"].submit(run, i, p) for i, p in enumerate(prompts)]
    for f in as_completed(futures):
        idx, res = f.result()
        results[idx] = res if res is not None else _fallback

    curriculum.step(len(prompts))

    valid = [r for r in results if r is not None]
    if valid:
        avg_strat  = sum(1 for r in valid if r["strategy_followed"]) / len(valid)
        avg_reward = sum(r["reward"] for r in valid) / len(valid)
        print(f"[GT-BATCH] Strategy: {avg_strat:.1%}, Avg Reward: {avg_reward:.3f}")

    return {
        "prompt_ids":     [r["prompt_ids"]     for r in results],
        "completion_ids": [r["completion_ids"] for r in results],
        "logprobs":       [r["logprobs"]       for r in results],
        "env_rewards":    [r["reward"]         for r in results],
    }


def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """Full-prompt rollout with strategy reward shaping and action masking."""
    _ensure_initialized(trainer)

    curriculum        = _state["curriculum"]
    current_max_turn  = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(f"[CURRICULUM] Rollout {curriculum.total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

    run = functools.partial(
        _run_episode_full,
        env_pool=_state["env_pool"],
        num_servers=_state["num_servers"],
        rank=_state["rank"],
        trainer=trainer,
        tokenizer=trainer.processing_class,
        generation_semaphore=_state["generation_semaphore"],
        current_max_turn=current_max_turn,
        current_hint_prob=current_hint_prob,
    )

    _fallback = {
        "prompt_ids": [1], "completion_ids": [1], "action_mask": [0],
        "logprobs": [1.0], "reward": 0.0, "strategy_ratio": 0.0, "final_score": 0.0,
    }

    results = [None] * len(prompts)
    futures = [_state["thread_pool"].submit(run, i, p) for i, p in enumerate(prompts)]
    for f in as_completed(futures):
        idx, res = f.result()
        results[idx] = res if res is not None else _fallback

    curriculum.step(len(prompts))

    list_results = [r for r in results if r is not None]
    avg_strategy = sum(r["strategy_ratio"] for r in list_results) / len(list_results) if list_results else 0
    avg_final    = sum(r["final_score"]    for r in list_results) / len(list_results) if list_results else 0
    print(f"[BATCH] Avg Strategy Adherence: {avg_strategy:.2%}, Avg Final Score: {avg_final:.3f}")

    return {
        "prompt_ids":     [r["prompt_ids"]     for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "action_mask":    [r["action_mask"]    for r in list_results],
        "logprobs":       [r["logprobs"]       for r in list_results],
        "env_rewards":    [r["reward"]         for r in list_results],
    }
