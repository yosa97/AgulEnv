import functools
import math
import os
import random
import re
from concurrent.futures import as_completed
from dataclasses import dataclass
from threading import Semaphore

import requests
from scipy.stats import binom
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

_SELECTED_GAME = "liars_dice"
_MAX_EPISODE_TOKENS = 16384
_MAX_PROMPT_LEN = 5000
_TIMEOUT = 2400

# Risky-move bonus parameters
BLUFF_PROB_THRESHOLD  = 0.35   # bid prob below this → bluff
RISKY_LIAR_PROB_MIN   = 0.35   # liar prob lower bound for "risky" range
RISKY_LIAR_PROB_MAX   = 0.60   # liar prob upper bound for "risky" range
BLUFF_WIN_BONUS       = 0.5    # bonus per bluff in a winning episode
RISKY_LIAR_WIN_BONUS  = 0.5    # bonus per risky liar call in a winning episode
RISKY_BONUS_MAX_COUNT = 2      # cap: at most 2 risky actions contribute bonus
SHUFFLE_PROB          = 0.5    # probability of shuffling the action list each observation
NORMALIZE_REWARDS     = False  # set True to separate intermediate/terminal rewards and normalize intermediate by T


# ---------------------------------------------------------------------------
# Data structures and probability helpers
# ---------------------------------------------------------------------------

@dataclass
class Bid:
    quantity: int
    face: int

    def __str__(self):
        return f"{self.quantity}-{self.face}"


@dataclass
class Action:
    SCORE_TEMPERATURE = 0.5  # sharpening temperature for the score formula

    action_id: int
    label: str
    bid: "Bid | None"
    prob: float = 0.0

    @property
    def is_liar(self) -> bool:
        return self.label.strip().lower() == "liar"

    @property
    def aggressiveness(self) -> float:
        return max(self.action_id / 59, 5 / 59) ** 0.5

    @property
    def score(self) -> float:
        a = self.SCORE_TEMPERATURE
        return (math.exp(self.prob / a) - 1.0) / (math.exp(1.0 / a) - 1.0)


@dataclass
class GameState:
    our_dice: list[int]
    total_dice: int
    current_bid: "Bid | None"
    actions: list[Action]

    @property
    def liar_action(self) -> "Action | None":
        return next((a for a in self.actions if a.is_liar), None)

    @property
    def bid_actions(self) -> list[Action]:
        return [a for a in self.actions if not a.is_liar]


def bid_probability(bid: "Bid | None", state: GameState) -> float:
    """P(bid is true) given the observable game state."""
    if bid is None:
        return 0.0

    our_dice   = state.our_dice   or []
    total_dice = state.total_dice or 0

    if bid.face == 6:
        our_count = sum(1 for d in our_dice if d == 6)
        p_hit = 1 / 6
    else:
        our_count = sum(1 for d in our_dice if d == bid.face or d == 6)
        p_hit = 2 / 6

    still_needed = bid.quantity - our_count
    if still_needed <= 0:
        return 1.0

    n_hidden = total_dice - len(our_dice)
    if n_hidden <= 0:
        return 0.0

    return 1.0 - binom.cdf(still_needed - 1, n=n_hidden, p=p_hit)


def _parse_bid_label(label: str) -> "Bid | None":
    m = re.fullmatch(r"(\d+)-(\d+)", label.strip())
    return Bid(int(m.group(1)), int(m.group(2))) if m else None


def parse_game_state(messages: "list[dict] | str") -> GameState:
    """Parse the last user message in a conversation into a GameState."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), None
    )
    if last_user_msg is None:
        raise ValueError("No user message found")

    dice_match = re.search(r"Your dice:\s*\[([^\]]+)\]", last_user_msg)
    if not dice_match:
        raise ValueError("Could not parse 'Your dice'")
    our_dice = [int(x.strip()) for x in dice_match.group(1).split(",")]

    total_match = re.search(r"Total dice in game:\s*(\d+)", last_user_msg)
    if not total_match:
        raise ValueError("Could not parse 'Total dice in game'")
    total_dice = int(total_match.group(1))

    bid_match = re.search(r'Current bid:\s*"(\d+)-(\d+)"', last_user_msg)
    current_bid = Bid(int(bid_match.group(1)), int(bid_match.group(2))) if bid_match else None

    raw_actions = re.findall(r"^\s*(\d+)\s*->\s*(.+)$", last_user_msg, re.MULTILINE)
    if not raw_actions:
        raise ValueError("Could not parse legal actions")

    tmp_state = GameState(our_dice=our_dice, total_dice=total_dice, current_bid=current_bid, actions=[])
    actions = []
    for aid, label in raw_actions:
        bid = _parse_bid_label(label)
        is_liar = label.strip().lower() == "liar"
        prob = (
            1.0 - bid_probability(current_bid, tmp_state) if is_liar and current_bid
            else (bid_probability(bid, tmp_state) if bid else 0.0)
        )
        actions.append(Action(action_id=int(aid), label=label.strip(), bid=bid, prob=prob))

    actions.sort(key=lambda a: a.action_id)
    return GameState(our_dice=our_dice, total_dice=total_dice, current_bid=current_bid, actions=actions)


class BayesianOpponentInference:
    """Track P(opponent_dice_roll | observed_bids) via sequential Bayesian update.

    For each enumerated possible opponent roll, we maintain a log-posterior
    updated after every opponent bid. The likelihood per roll is a mixture
    of:
      - bluff branch (probability `_adaptive_bluff_prob`): bid not backed
        by enough opp support;
      - honest branch: opp has at least the needed face-support.

    Used by the expert during LIAR-action selection to compute a second
    opinion on whether the standing bid is a bluff. Combined 50/50 with
    the analytic prob in get_expert_action so the demonstrator sees both
    statistical (analytic) and history-conditional (bayes) signals.

    Wild-six convention matches the env: face != 6 counts both the face
    AND any 6 as support; face == 6 counts only 6s.
    """

    def __init__(self, n_dice: int = 5, wild_six: bool = True) -> None:
        self.n_dice = n_dice
        self.wild_six = wild_six
        self._all_rolls: list[tuple[int, ...]] = self._enumerate_rolls(n_dice)
        n = len(self._all_rolls)
        self._log_probs: list[float] = [-math.log(n)] * n
        self._bids_observed: int = 0

    @staticmethod
    def _enumerate_rolls(n_dice: int) -> list[tuple[int, ...]]:
        if n_dice <= 0:
            return [()]
        result: list[tuple[int, ...]] = [()]
        for _ in range(n_dice):
            result = [r + (v,) for r in result for v in range(1, 7)]
        return result

    def _face_count(self, roll: tuple[int, ...], face: int) -> int:
        if self.wild_six and face != 6:
            return sum(1 for d in roll if d == face or d == 6)
        return sum(1 for d in roll if d == face)

    def _adaptive_bluff_prob(self, total_dice: int) -> float:
        """Adaptive bluff probability: higher in late game (fewer dice)."""
        phase_factor = max(0.0, 1.0 - (total_dice / 10.0))
        base = 0.10 + 0.18 * phase_factor
        decay = min(self._bids_observed / 5.0, 1.0)
        return max(0.08, base * (1.0 - 0.25 * decay))

    def update(self, bid: "Bid", own_support: int, total_dice: int = 10) -> None:
        """Update posterior given opponent's bid and our own support for that face."""
        self._bids_observed += 1
        need_from_opp = max(bid.quantity - own_support, 0)
        bp = self._adaptive_bluff_prob(total_dice)
        log_bluff = math.log(bp)
        log_support = math.log(1.0 - bp)

        try:
            import numpy as np
            rolls = np.array(self._all_rolls, dtype=np.int8)
            if self.wild_six and bid.face != 6:
                opp_support = np.sum((rolls == bid.face) | (rolls == 6), axis=1)
            else:
                opp_support = np.sum(rolls == bid.face, axis=1)
            log_ll = np.where(opp_support >= need_from_opp, log_support, log_bluff)
            log_probs = np.array(self._log_probs) + log_ll
            max_lp = log_probs.max()
            log_sum = max_lp + math.log(np.exp(log_probs - max_lp).sum())
            self._log_probs = (log_probs - log_sum).tolist()
        except ImportError:
            new_lp = []
            for i, roll in enumerate(self._all_rolls):
                support = self._face_count(roll, bid.face)
                ll = log_support if support >= need_from_opp else log_bluff
                new_lp.append(self._log_probs[i] + ll)
            max_lp = max(new_lp)
            log_sum = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in new_lp))
            self._log_probs = [lp - log_sum for lp in new_lp]

    def bid_posterior_prob(self, bid: "Bid", own_support: int) -> float:
        """P(bid is true | posterior). Inverse = P(bluff)."""
        need = max(bid.quantity - own_support, 0)
        return sum(
            math.exp(self._log_probs[i])
            for i, roll in enumerate(self._all_rolls)
            if self._face_count(roll, bid.face) >= need
        )

    def reset(self) -> None:
        n = len(self._all_rolls)
        self._log_probs = [-math.log(n)] * n
        self._bids_observed = 0


class RewardCalculator:
    """Shaped reward calculator for Liar's Dice training."""

    def __init__(self, gamma: float = 0.9):
        self.terminal_weight = 10.0
        self.gamma = gamma

    def calculate_step_reward(self, action: "Action | None", env_reward: float) -> float:
        reward = 0.0
        if action is not None:
            reward += action.score
        if env_reward != 0.0:
            reward += env_reward * self.terminal_weight
        return reward

    def calculate_discounted_return(
        self,
        rewards: list[float],
        step_scores: "list[float] | None" = None,
        terminal_reward: float = 0.0,
    ) -> float:
        """Compute the training return.

        When ``NORMALIZE_REWARDS`` is False (default):
            G = Σ γ^(T-1-i) * r_i  (legacy behaviour, length-biased)

        When ``NORMALIZE_REWARDS`` is True:
            Intermediate scores are discounted then divided by T so that the
            weighted average is length-invariant, and the terminal reward is
            added on top without discounting.
            G = (Σ γ^(T-1-i) * s_i) / T  +  terminal_reward
        """
        if not NORMALIZE_REWARDS:
            if not rewards:
                return 0.0
            T = len(rewards)
            return sum(self.gamma ** (T - 1 - i) * r for i, r in enumerate(rewards))

        # --- Normalized path ---
        scores = step_scores if step_scores is not None else []
        if not scores:
            return terminal_reward
        T = len(scores)
        discounted_sum = sum(self.gamma ** (T - 1 - i) * s for i, s in enumerate(scores))
        return discounted_sum / T + terminal_reward


# ---------------------------------------------------------------------------
# Module-level state (shared between full and last rollout functions)
# ---------------------------------------------------------------------------

_state: dict = {}


def _curriculum_factory(args) -> CurriculumScheduler:
    """Construct this env's curriculum from training args. Referenced by env_configs registry."""
    return CurriculumScheduler(
        initial_max_turn=args.initial_max_turn,
        final_max_turn=15,
        rollouts_per_stage=args.rollouts_per_stage,
        initial_hint_prob=0.5,
        final_hint_prob=0.0,
        warmup_rollouts=args.rollouts_per_stage,
    )


def _ensure_initialized(trainer) -> None:
    """Set up server pool and curriculum once per process (no-op afterwards)."""
    if _state.get("initialized"):
        return

    reset_payload = {
        "task_id": GAMES_TO_TASK_ID_RANGE[_SELECTED_GAME][0],
        "seed": 42,
        "opponent": "mcts",
        "mcts_max_simulations": 225,
        "mcts_num_rollouts": 1,
    }
    rank, env_pool, num_servers, thread_pool, generation_semaphore = init_env_pool(reset_payload)

    curriculum = _curriculum_factory(trainer.args)
    print(
        f"[CURRICULUM] Initialized: initial_max_turn={trainer.args.initial_max_turn}, "
        f"final_max_turn=30, rollouts_per_stage={trainer.args.rollouts_per_stage}"
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
# Observation reformatter
# ---------------------------------------------------------------------------

def _reformat_observation(obs: str, gs: GameState, use_hints: bool) -> str:
    """Randomly shuffle the displayed action order and optionally append a per-turn hint."""
    if random.random() < SHUFFLE_PROB:
        actions = gs.actions[:]
        random.shuffle(actions)
        action_block = "Legal Actions:\n" + "\n".join(f"{a.action_id} -> {a.label}" for a in actions)
        obs = re.sub(r"Legal Actions:\n(?:[ \t]*\d+[ \t]*->[ \t]*\S.*(?:\n|$))+", action_block + "\n", obs)
    if use_hints:
        scores = [a.score for a in gs.actions]
        best = random.choices(gs.actions, weights=scores, k=1)[0]
        obs += f"\n[Hint: action {best.action_id} ({best.label}) is recommended]"
    return obs


# ---------------------------------------------------------------------------
# Core episode runner (shared between full-prompt and last-prompt variants)
# ---------------------------------------------------------------------------

def _run_episode(
    index: int,
    prompt: str,
    *,
    use_full_prompt: bool,
    env_pool: list[dict],
    num_servers: int,
    rank: int,
    trainer,
    tokenizer,
    generation_semaphore: Semaphore,
    current_max_turn: int,
    current_hint_prob: float,
) -> tuple[int, "dict | None"]:
    """
    Run one episode.

    When ``use_full_prompt=True``, accumulates token IDs across all turns with
    action masking (mask=1 for LLM completions, mask=0 for environment turns).
    When ``use_full_prompt=False``, only the final turn's token IDs are kept.
    """
    current_max_turn = 5
    game_id = int(prompt)

    server_idx   = (index + rank) % num_servers
    env_endpoint = env_pool[server_idx]["base_url"]

    # --- Full-prompt accumulation state (only used when use_full_prompt=True) ---
    episode_prompt_ids:    list[int]   = []
    episode_completion_ids: list[int]  = []
    episode_logprobs:      list[float] = []
    episode_action_mask:   list[int]   = []
    prev_full_ids: "list[int] | None"  = None

    # Last-prompt fallback (overwritten every loop iteration in use_full_prompt=False mode)
    prompt_ids:     list[int]   = []
    completion_ids: list[int]   = []
    logprobs:       list[float] = []

    # --- Episode state ---
    invalid_count  = 0
    done           = False
    train_reward   = 0.0
    final_reward   = 0.0
    turn_number    = 0
    game_state_history: list[GameState] = []
    rewards:      list[float] = []   # legacy path
    step_scores:  list[float] = []   # normalized path: per-step action scores (incl. invalid penalty)
    terminal_reward: float    = 0.0  # normalized path: win/loss + bonuses
    calculator = RewardCalculator()
    bluff_count      = 0
    risky_liar_count = 0
    last_action_prob = 0.0

    # Per-episode hint flag
    use_hints = random.random() < current_hint_prob

    # --- Reset environment ---
    reset_payload = {
        "task_id": game_id,
        "seed": game_id,
        "opponent": "mcts",
        "mcts_max_simulations": 225,
        "mcts_num_rollouts": 1,
    }
    try:
        reset_res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        reset_res.raise_for_status()
        result_block = reset_res.json()["result"]
        episode_id = result_block.get("episode_id", "")
        raw_observation = result_block.get("observation", "")
        formatted_observation = raw_observation
        _init_gs = parse_game_state(formatted_observation)
        game_state_history.append(_init_gs)
        formatted_observation = _reformat_observation(formatted_observation, _init_gs, use_hints)
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"Failed to reset environment (Game {game_id}): {exc}")
        return index, None

    # --- Build system prompt ---
    system_prompt = (
        '"You are playing liars_dice.\n\n# Game Rules\nLIAR\'S DICE RULES:\n\n'
        'Setup: Each player has N dice (1-5 depending on variant). All players roll their dice secretly.\n\n'
        'Goal: Make bids about total dice across ALL players, or call "Liar" on opponent\'s bid.\n\n'
        'Actions:\n- Bid (quantity, face): Claim there are at least \'quantity\' dice showing \'face\' among all dice.\n'
        '- Call Liar: Challenge the previous bid.\n\n'
        'Bidding rules: Each bid must be higher than the previous bid. "Higher" means:\n'
        '  - Same face value but higher quantity (e.g., "2 fours" beats "1 four")\n'
        '  - Same quantity but higher face value (e.g., "2 fives" beats "2 fours")\n\n'
        'Wild dice: 6s are WILD and count as ANY face value.\n'
        '- When counting dice for a bid, include 6s in the count\n'
        '- Example: Bid "3 fours" means at least 3 dice showing EITHER 4 OR 6\n\n'
        'Winning: If you call Liar and previous bid was false, opponent loses. If bid was true or exact, you lose.\n\n\n\n'
        '# Output Format\nYou must respond with ONLY the action ID (a single number).\n'
        'Do NOT include descriptions or explanations.\n\n'
        'Examples:\n- For action "0 -> roll": respond "0"\n- For action "89 -> a3": respond "89"'
        '"'
    )
    if use_hints:
        system_prompt += (
            '\n# Strategy Tips\n'
            '- Count your dice that match the bid (including 6s as wild)\n'
            '- Call "Liar" when the bid is more likely false than any available bid is true.\n'
            '- Make conservative bids early, aggressive when opponent seems weak\n'
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": formatted_observation},
    ]

    # --- Interaction loop ---
    while not done and turn_number < current_max_turn:
        with generation_semaphore:
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

        prompt_ids     = rollout_outputs.get("prompt_ids", [])
        completion_ids = rollout_outputs.get("completion_ids", [])
        logprobs       = rollout_outputs.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # --- Token accumulation ---
        if use_full_prompt:
            if len(prompt_ids) > _MAX_PROMPT_LEN:
                print(
                    f"Warning: Prompt exceeded {_MAX_PROMPT_LEN} tokens "
                    f"({len(prompt_ids)}) at turn {turn_number}, ending episode early"
                )
                done = True
                break

            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                    # Re-tokenising the full conversation can shift earlier token IDs
                    # (BPE tokenisers are not context-free). Skip delta mask this turn.
                    print(
                        f"Warning: token shift at turn {turn_number} "
                        f"(expected prefix {len(prev_full_ids)}, got {len(prompt_ids)}). "
                        "Skipping delta mask for this turn."
                    )
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

        # --- Parse action (game-specific: strip EOS / "Action:" prefix) ---
        action_to_send = completion_text
        if action_to_send.endswith("</s>"):
            action_to_send = action_to_send[:-4]
        if "Action:" in action_to_send:
            action_to_send = action_to_send.split("Action:")[-1].strip()

        # --- Step environment ---
        is_invalid = False
        try:
            formatted_observation = ""
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action_to_send, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block = step_res.json()["result"]
            raw_observation       = step_block.get("observation", "")
            formatted_observation = raw_observation
            step_reward           = step_block.get("reward", 0)
            done                  = step_block.get("done", False)
        except Exception as exc:
            print(f"Step failed: {exc}")
            step_reward = -0.01
            done = False
            invalid_count += 1

        if "Nothing happens" in formatted_observation or "Invalid" in formatted_observation:
            invalid_count += 1
            is_invalid = True

        # Parse next game state early so we can reformat before the model sees it
        _next_gs = None
        if not done and not is_invalid and formatted_observation:
            try:
                _next_gs = parse_game_state(formatted_observation)
                formatted_observation = _reformat_observation(formatted_observation, _next_gs, use_hints)
            except Exception:
                pass

        if done:
            final_reward = step_reward
        messages.append({"role": "user", "content": formatted_observation})

        # --- Reward calculation ---
        last_action_prob = 0.0
        if not is_invalid:
            try:
                previous_game_state = game_state_history[-1]
                game_state = _next_gs if _next_gs is not None else (None if done else parse_game_state(formatted_observation))
                action_id = int(action_to_send.strip())
            except Exception as exc:
                print(f"Failed to parse game state or action id: {exc}")
                immediate_reward = -1.0
            else:
                taken_action = next(
                    (a for a in previous_game_state.actions if a.action_id == action_id), None
                )
                last_action_prob = taken_action.prob if taken_action else 0.0

                # Classify risky moves for end-of-episode bonus
                if taken_action is not None:
                    if taken_action.is_liar:
                        if RISKY_LIAR_PROB_MIN <= taken_action.prob <= RISKY_LIAR_PROB_MAX:
                            risky_liar_count += 1
                    else:
                        if taken_action.prob < BLUFF_PROB_THRESHOLD:
                            bluff_count += 1

                if not done:
                    game_state_history.append(game_state)
                    immediate_reward = calculator.calculate_step_reward(taken_action, 0.0)
                    step_scores.append(taken_action.score if taken_action else 0.0)
                else:
                    won = step_reward > 0.5
                    immediate_reward = (taken_action.score if taken_action else 0.0)
                    immediate_reward += (step_reward - 0.5) * 2.0
                    step_scores.append(taken_action.score if taken_action else 0.0)
                    terminal_reward = (step_reward - 0.5) * 2.0
                    if won:
                        immediate_reward += BLUFF_WIN_BONUS * min(bluff_count, RISKY_BONUS_MAX_COUNT)
                        immediate_reward += RISKY_LIAR_WIN_BONUS * min(risky_liar_count, RISKY_BONUS_MAX_COUNT)
                        terminal_reward  += BLUFF_WIN_BONUS * min(bluff_count, RISKY_BONUS_MAX_COUNT)
                        terminal_reward  += RISKY_LIAR_WIN_BONUS * min(risky_liar_count, RISKY_BONUS_MAX_COUNT)
        else:
            immediate_reward = -1.0
            step_scores.append(-1.0)

        rewards.append(immediate_reward)
        turn_number += 1

    # --- Final reward ---
    train_reward = calculator.calculate_discounted_return(
        rewards,
        step_scores=step_scores,
        terminal_reward=terminal_reward,
    )

    print(
        "[ID:{:<6} Hints:{} Done:{} T:{:>2d} | Reward:{:>8.2f} | LastProb:{:>7.3f} | "
        "EnvR:{:>6.1f} | Bluffs:{:<2} RiskyLiar:{:<2} Inv:{:<2}]".format(
            str(game_id)[:6], int(use_hints), int(done), turn_number,
            train_reward, last_action_prob, final_reward,
            bluff_count, risky_liar_count, invalid_count,
        )
    )

    # --- Build result ---
    if use_full_prompt:
        if len(episode_completion_ids) > _MAX_EPISODE_TOKENS:
            episode_completion_ids = episode_completion_ids[:_MAX_EPISODE_TOKENS]
            episode_logprobs       = episode_logprobs[:_MAX_EPISODE_TOKENS]
            episode_action_mask    = episode_action_mask[:_MAX_EPISODE_TOKENS]
        return index, {
            "prompt_ids":     episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "action_mask":    episode_action_mask,
            "logprobs":       episode_logprobs,
            "reward":         train_reward,
            "final_score":    final_reward,
        }
    else:
        return index, {
            "prompt_ids":     prompt_ids,
            "completion_ids": completion_ids,
            "logprobs":       logprobs,
            "reward":         train_reward,
            "final_score":    final_reward,
        }


# ---------------------------------------------------------------------------
# Public rollout functions
# ---------------------------------------------------------------------------

def _dispatch(prompts, trainer, *, use_full_prompt: bool) -> dict[str, list]:
    """Common dispatch + aggregation logic for both rollout variants."""
    _ensure_initialized(trainer)

    curriculum        = _state["curriculum"]
    current_max_turn  = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(f"[CURRICULUM] Rollout {curriculum.total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

    run = functools.partial(
        _run_episode,
        use_full_prompt=use_full_prompt,
        env_pool=_state["env_pool"],
        num_servers=_state["num_servers"],
        rank=_state["rank"],
        trainer=trainer,
        tokenizer=trainer.processing_class,
        generation_semaphore=_state["generation_semaphore"],
        current_max_turn=current_max_turn,
        current_hint_prob=current_hint_prob,
    )

    _fallback = (
        {"prompt_ids": [1], "completion_ids": [1], "action_mask": [0], "logprobs": [1.0], "reward": 0.0, "final_score": 0.0}
        if use_full_prompt else
        {"prompt_ids": [1], "completion_ids": [1], "logprobs": [1.0], "reward": 0.0, "final_score": 0.0}
    )

    results = [None] * len(prompts)
    futures = [_state["thread_pool"].submit(run, i, p) for i, p in enumerate(prompts)]
    for f in as_completed(futures):
        idx, res = f.result()
        results[idx] = res if res is not None else _fallback

    curriculum.step(len(prompts))

    list_results = [r for r in results if r is not None]
    finished  = sum(1 for r in list_results if r["final_score"] != 0)
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0
    print(f"[BATCH] Finished: {finished}/{len(list_results)}, AvgReturn: {avg_return:.2f}")

    out = {
        "prompt_ids":     [r["prompt_ids"]     for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "logprobs":       [r["logprobs"]       for r in list_results],
        "env_rewards":    [r["reward"]         for r in list_results],
    }
    if use_full_prompt:
        out["action_mask"] = [r["action_mask"] for r in list_results]
    return out


def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """Parallelised rollout — accumulates all turns with action masking."""
    return _dispatch(prompts, trainer, use_full_prompt=True)


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """Parallelised rollout — returns only the last turn's token IDs."""
    return _dispatch(prompts, trainer, use_full_prompt=False)
