import json
import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock, Semaphore

import requests
from scipy.stats import binom
from trl.experimental.openenv import generate_rollout_completions

GAME_TO_TASK_ID_RANGE = {
    "goofspiel": (0, 99999999),
    "liars_dice": (100000000, 199999999),
    "leduc_poker": (200000000, 299999999),
    "gin_rummy": (300000000, 399999999),
    "othello": (400000000, 499999999),
    "backgammon": (500000000, 599999999),
    "hex": (600000000, 699999999),
    "clobber": (700000000, 799999999),
}

SELECTED_GAME = "liars_dice"
REQUEST_TIMEOUT_SECONDS = 2400
INIT_TIMEOUT_SECONDS = 300
MAX_EPISODE_TOKENS = 16384
MAX_PROMPT_LEN = 16384 - 512

MCTS_CONFIG = {
    "opponent": "mcts",
    "mcts_max_simulations": 225,
    "mcts_num_rollouts": 1,
}

# Reward settings
INVALID_ACTION_PENALTY = 0.10
PASS_MISSED_CHALLENGE_PENALTY = 0.04
CORRECT_CHALLENGE_BONUS = 0.04
OVERCONFIDENT_CHALLENGE_PENALTY = 0.02
BID_PLAUSIBILITY_BONUS = 0.02
BID_PLAUSIBILITY_PENALTY = 0.02
PER_STEP_SHAPING_CLIP = 0.05
SHAPING_REWARD_CLIP = 0.25
TERMINAL_REWARD_CLIP = 1.00

STRATEGY_TIPS = """
STRATEGY TIPS:
- Keep bids minimally stronger than current bid when uncertain.
- Use your own dice + wild 6s to estimate plausible total counts.
- Prefer calling Liar when the required quantity is implausibly high.
- Avoid large overbids unless your private dice strongly support it.
"""

REASONING_TAG_PAIRS = [
    ("think", "think"),
    ("thinking", "thinking"),
    ("reasoning", "reasoning"),
    ("thought", "thought"),
    ("reflection", "reflection"),
]

_ROLLOUT_STATE: dict = {}


def _is_truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def extract_and_format_observation(obs_text: str) -> str:
    # Liar's Dice observations already contain structured legal-action blocks.
    return obs_text or ""


# ---------------------------------------------------------------------------
# Game-state parser: data structures + probability helpers used by
# liar_dice_trajectories.get_expert_action. Hosted here so the SFT pipeline
# depends only on *_opponent_modeling / *_trajectories, not the legacy *_env.
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


class EpisodeTraceLogger:
    """Thread-safe JSONL episode tracer."""

    def __init__(self, trace_dir: str, rank: int):
        self.trace_dir = trace_dir
        self.rank = rank
        self._lock = Lock()
        self.log_path = os.path.join(self.trace_dir, f"liars_dice_episode_traces_rank{rank}.jsonl")
        self.max_text_chars = int(os.environ.get("EPISODE_TRACE_MAX_TEXT_CHARS", "4000"))
        self.sample_rate = float(os.environ.get("EPISODE_TRACE_SAMPLE_RATE", "1.0"))

        os.makedirs(self.trace_dir, exist_ok=True)
        print(f"[EPISODE_TRACE] Writing traces to {self.log_path}")

    def should_log(self) -> bool:
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        return random.random() <= self.sample_rate

    def clip_text(self, text: str) -> str:
        if not text:
            return ""
        if len(text) <= self.max_text_chars:
            return text
        return text[: self.max_text_chars] + f"... [truncated {len(text) - self.max_text_chars} chars]"

    def log_episode(self, payload: dict) -> None:
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")


class CurriculumScheduler:
    """Progressive turn-limit curriculum."""

    def __init__(
        self,
        initial_max_turn: int = 2,
        final_max_turn: int = 20,
        rollouts_per_stage: int = 1280,
        initial_hint_prob: float = 0.0,
        final_hint_prob: float = 0.0,
        warmup_rollouts: int = 128,
    ):
        self.initial_max_turn = initial_max_turn
        self.final_max_turn = final_max_turn
        self.rollouts_per_stage = rollouts_per_stage
        self.initial_hint_prob = initial_hint_prob
        self.final_hint_prob = final_hint_prob
        self.warmup_rollouts = warmup_rollouts
        self.total_rollouts = 0

    def get_max_turn(self) -> int:
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_max_turn
        adjusted_rollouts = self.total_rollouts - self.warmup_rollouts
        stage = adjusted_rollouts // self.rollouts_per_stage
        return min(self.initial_max_turn + stage, self.final_max_turn)

    def get_hint_prob(self) -> float:
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_hint_prob
        total_stages = max(self.final_max_turn - self.initial_max_turn, 1)
        total_decay_rollouts = total_stages * self.rollouts_per_stage
        adjusted_rollouts = self.total_rollouts - self.warmup_rollouts
        progress = min(adjusted_rollouts / total_decay_rollouts, 1.0)
        current_prob = self.initial_hint_prob - progress * (self.initial_hint_prob - self.final_hint_prob)
        return max(current_prob, self.final_hint_prob)

    def step(self, num_rollouts: int = 1) -> None:
        self.total_rollouts += num_rollouts


def remove_reasoning_tags(text: str) -> str:
    cleaned = text
    for tag_name, close_name in REASONING_TAG_PAIRS:
        cleaned = re.sub(
            rf"<{tag_name}>.*?</{close_name}>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        close_tag = f"</{close_name}>"
        if close_tag in cleaned:
            cleaned = cleaned.split(close_tag)[-1]
        open_match = re.search(rf"<{tag_name}>", cleaned, flags=re.IGNORECASE)
        if open_match:
            cleaned = cleaned[: open_match.start()]
    cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()


def _extract_legal_action_map(observation: str) -> dict[str, str]:
    if not observation:
        return {}
    match = re.search(
        r"Legal Actions:\s*\n(.*?)(?:\n\nYour choice|\nYour choice|\Z)",
        observation,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return {}

    block = match.group(1)
    mapping: dict[str, str] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "->" in line:
            left, right = line.split("->", 1)
            action_id = left.strip()
            label = right.strip()
        else:
            action_id = line.strip()
            label = action_id
        if re.fullmatch(r"-?\d+", action_id):
            mapping[action_id] = label
    return mapping


def _extract_bid_tuple(label_or_text: str) -> tuple[int, int] | None:
    if not label_or_text:
        return None
    match = re.search(r"(\d+)\s*-\s*(\d+)", label_or_text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _extract_state_features(observation: str) -> dict:
    dice: list[int] = []
    dice_match = re.search(r"Your dice:\s*\[([^\]]*)\]", observation)
    if dice_match:
        dice_str = dice_match.group(1).strip()
        if dice_str:
            dice = [int(x.strip()) for x in dice_str.split(",") if x.strip().isdigit()]

    total_dice_match = re.search(r"Total dice in game:\s*(\d+)", observation)
    total_dice = int(total_dice_match.group(1)) if total_dice_match else 0

    current_bid_match = re.search(r'Current bid:\s*"([^"]+)"', observation)
    current_bid = _extract_bid_tuple(current_bid_match.group(1)) if current_bid_match else None

    # 6s are wild in this repo's Liar's Dice variant. The environment
    # observation does not restate the wild rule (only the system prompt
    # does), so detecting by string would always be False and silently
    # bias every bid's truth probability. Hardcode True; override at the
    # observation level if a future variant disables wilds.
    return {
        "own_dice": dice,
        "total_dice": total_dice,
        "current_bid": current_bid,
        "wild_six_enabled": True,
    }


def _is_liar_label(label: str) -> bool:
    return "liar" in (label or "").strip().lower()


def _bid_rank(bid: tuple[int, int]) -> int:
    quantity, face = bid
    return quantity * 6 + face


def _count_face_support(own_dice: list[int], target_face: int, wild_six_enabled: bool) -> int:
    if wild_six_enabled and target_face != 6:
        return sum(1 for value in own_dice if value == target_face or value == 6)
    return sum(1 for value in own_dice if value == target_face)


def _binomial_tail_probability(num_trials: int, success_prob: float, min_successes: int) -> float:
    if min_successes <= 0:
        return 1.0
    if num_trials <= 0:
        return 0.0

    success_prob = _clamp(success_prob, 0.0, 1.0)
    tail_probability = 0.0
    for successes in range(min_successes, num_trials + 1):
        tail_probability += math.comb(num_trials, successes) * (success_prob ** successes) * (
            (1.0 - success_prob) ** (num_trials - successes)
        )
    return _clamp(tail_probability, 0.0, 1.0)


def _estimate_bid_statistics(state_features: dict, bid: tuple[int, int]) -> dict:
    own_dice = state_features.get("own_dice") or []
    total_dice = int(state_features.get("total_dice") or 0)
    wild_six_enabled = bool(state_features.get("wild_six_enabled"))
    quantity, face = bid

    if total_dice <= 0 or not own_dice:
        return {
            "known_support": 0,
            "unknown_dice": 0,
            "expected_total": 0.0,
            "std_dev": 0.0,
            "z_score": 0.0,
            "truth_probability": 0.0,
        }

    known_support = _count_face_support(own_dice, face, wild_six_enabled)
    unknown_dice = max(total_dice - len(own_dice), 0)
    per_die_success_prob = 2.0 / 6.0 if (wild_six_enabled and face != 6) else 1.0 / 6.0

    if unknown_dice == 0:
        expected_total = float(known_support)
        std_dev = 0.0
    else:
        expected_total = known_support + unknown_dice * per_die_success_prob
        std_dev = math.sqrt(unknown_dice * per_die_success_prob * (1.0 - per_die_success_prob))

    additional_needed = max(quantity - known_support, 0)
    truth_probability = _binomial_tail_probability(
        num_trials=unknown_dice,
        success_prob=per_die_success_prob,
        min_successes=additional_needed,
    )

    if std_dev > 0:
        z_score = (quantity - expected_total) / std_dev
    elif quantity <= expected_total:
        z_score = -1.0
    else:
        z_score = 3.0

    return {
        "known_support": known_support,
        "unknown_dice": unknown_dice,
        "expected_total": expected_total,
        "std_dev": std_dev,
        "z_score": z_score,
        "truth_probability": truth_probability,
    }


def _score_bid_plausibility(state_features: dict, bid: tuple[int, int]) -> float:
    own_dice = state_features.get("own_dice") or []
    total_dice = int(state_features.get("total_dice") or 0)
    current_bid = state_features.get("current_bid")

    if total_dice <= 0 or not own_dice:
        return 0.0

    bid_stats = _estimate_bid_statistics(state_features, bid)
    truth_probability = float(bid_stats["truth_probability"])

    # Continuous shaping: reward proportional to how much more likely the bid
    # is to be true than a coin flip. p=1.0 -> +BID_PLAUSIBILITY_BONUS,
    # p=0.0 -> -BID_PLAUSIBILITY_PENALTY, p=0.5 -> 0. No dead zone.
    signed = 2.0 * truth_probability - 1.0
    if signed >= 0:
        reward = signed * BID_PLAUSIBILITY_BONUS
    else:
        reward = signed * BID_PLAUSIBILITY_PENALTY

    if current_bid is not None:
        jump = _bid_rank(bid) - _bid_rank(current_bid)
        if jump <= 2:
            reward += 0.01
        elif jump >= 7:
            if truth_probability < 0.30:
                reward -= 0.03
            else:
                reward += 0.01

    return reward

def _parse_action_id(completion_text: str, legal_action_map: dict[str, str]) -> str:
    if not legal_action_map:
        return ""

    cleaned = remove_reasoning_tags(completion_text or "")
    cleaned = cleaned.removesuffix("</s>")
    if "Action:" in cleaned:
        cleaned = cleaned.split("Action:")[-1].strip()

    normalized = cleaned.strip().lower()

    # Exact label match ("Liar", "7-4") — preferred over numeric regex so
    # stray numbers in reasoning text can't hijack the parse.
    for action_id, label in legal_action_map.items():
        if normalized == label.strip().lower():
            return action_id

    # Full "id -> label" form.
    for action_id, label in legal_action_map.items():
        compound = f"{action_id} -> {label}".strip().lower()
        if normalized == compound:
            return action_id

    # "Liar" keyword match before numeric: guards against a reply that
    # mentions "10 dice" (number) but is really calling Liar.
    if "liar" in normalized:
        for action_id, label in legal_action_map.items():
            if _is_liar_label(label):
                return action_id

    # Bid tuple ("7-4", "7 - 4"): check before raw numeric match so the
    # embedded '-' isn't parsed as a sign on the second number.
    bid_tuple = _extract_bid_tuple(cleaned)
    if bid_tuple is not None:
        for action_id, label in legal_action_map.items():
            if _extract_bid_tuple(label) == bid_tuple:
                return action_id

    # Numeric id fallback.
    for num in re.findall(r"\d+", cleaned):
        if num in legal_action_map:
            return num

    return ""


def _score_challenge_decision(
    state_features: dict,
    chose_liar: bool,
    proposed_bid: tuple[int, int] | None,
) -> tuple[float, dict]:
    current_bid = state_features.get("current_bid")
    if current_bid is None:
        return 0.0, {"current_bid_z": 0.0, "current_bid_truth_probability": 0.0}

    current_bid_stats = _estimate_bid_statistics(state_features, current_bid)
    current_bid_z = float(current_bid_stats["z_score"])
    current_bid_truth_probability = float(current_bid_stats["truth_probability"])
    reward = 0.0

    if chose_liar:
        # Symmetric with the pass-through branch so the expected reward for a
        # well-timed challenge is not zero.
        if current_bid_truth_probability <= 0.20:
            reward += CORRECT_CHALLENGE_BONUS * (
                1.0 + _clamp((0.20 - current_bid_truth_probability) / 0.20, 0.0, 1.0)
            )
        elif current_bid_truth_probability >= 0.60:
            reward -= OVERCONFIDENT_CHALLENGE_PENALTY
    elif proposed_bid is not None:
        if current_bid_truth_probability <= 0.20:
            reward -= PASS_MISSED_CHALLENGE_PENALTY * (
                1.0 + _clamp((0.20 - current_bid_truth_probability) / 0.20, 0.0, 1.0)
            )
        elif current_bid_truth_probability >= 0.55:
            reward += 0.01

    return reward, {
        "current_bid_z": current_bid_z,
        "current_bid_truth_probability": current_bid_truth_probability,
    }


def _select_fallback_action(legal_action_map: dict[str, str], state_features: dict) -> str:
    if not legal_action_map:
        return ""

    liar_actions = [
        action_id
        for action_id, label in legal_action_map.items()
        if _is_liar_label(label)
    ]
    current_bid = state_features.get("current_bid")

    # Call Liar if the current bid is clearly implausible.
    if liar_actions and current_bid is not None:
        current_bid_truth_probability = float(
            _estimate_bid_statistics(state_features, current_bid)["truth_probability"]
        )
        if current_bid_truth_probability <= 0.15:
            return liar_actions[0]

    # Among bid actions, pick a random one with truth_probability >= 0.5.
    # This both (a) avoids the guaranteed-loss "always open with 1-1" bias
    # of sorting-by-id and (b) keeps the fallback from being predictable
    # enough that the model can exploit it.
    bid_candidates: list[tuple[str, float]] = []
    for action_id, label in legal_action_map.items():
        if _is_liar_label(label):
            continue
        bid = _extract_bid_tuple(label)
        if bid is None:
            continue
        p_true = float(
            _estimate_bid_statistics(state_features, bid)["truth_probability"]
        )
        bid_candidates.append((action_id, p_true))

    plausible = [aid for aid, p in bid_candidates if p >= 0.5]
    if plausible:
        return random.choice(plausible)
    if bid_candidates:
        # Least-implausible bid beats a random/arbitrary pick.
        bid_candidates.sort(key=lambda item: item[1], reverse=True)
        return bid_candidates[0][0]
    # Only Liar is legal, or no bid labels parsed — random legal id.
    return random.choice(list(legal_action_map.keys()))


def _extract_terminal_reward(step_block: dict, observation_text: str) -> float:
    info = step_block.get("info", {}) if isinstance(step_block, dict) else {}

    cumulative_reward = info.get("cumulative_reward")
    if isinstance(cumulative_reward, (int, float)):
        return _clamp(float(cumulative_reward), -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)

    your_return_match = re.search(r"Your Return:\s*([+-]?\d+(?:\.\d+)?)", observation_text or "")
    if your_return_match:
        return _clamp(float(your_return_match.group(1)), -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)

    normalized_match = re.search(r"Normalized Score:\s*([+-]?\d+(?:\.\d+)?)", observation_text or "")
    result_match = re.search(r"Result:\s*(WIN|LOSS|DRAW)", observation_text or "", flags=re.IGNORECASE)
    if normalized_match:
        normalized_value = float(normalized_match.group(1))
        if result_match:
            result = result_match.group(1).upper()
            if result == "LOSS":
                normalized_value = -abs(normalized_value) if normalized_value != 0 else -1.0
            elif result == "WIN":
                normalized_value = abs(normalized_value) if normalized_value != 0 else 1.0
            else:
                normalized_value = 0.0
        return _clamp(normalized_value, -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)

    step_reward = _safe_float(step_block.get("reward", 0.0), default=0.0)
    return _clamp(step_reward, -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)


def _build_env_pool(server_urls: list[str]) -> list[dict[str, str]]:
    env_pool = []
    init_task_id = GAME_TO_TASK_ID_RANGE[SELECTED_GAME][0]

    for idx, base_url in enumerate(server_urls):
        try:
            print(f"[INIT] Initializing env on server {idx}: {base_url}")
            payload = {"task_id": init_task_id, "seed": 42, **MCTS_CONFIG}
            res = requests.post(f"{base_url}/reset", json=payload, timeout=INIT_TIMEOUT_SECONDS)
            res.raise_for_status()
            env_pool.append({"base_url": base_url})
            print(f"[INIT] Server {idx} ready")
        except Exception as e:
            raise RuntimeError(f"Failed to init server {base_url}: {e}") from e

    return env_pool


def _initialize_rollout_state(trainer) -> None:
    if _ROLLOUT_STATE.get("initialized", False):
        return

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
    server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
    if not server_urls:
        raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")

    env_pool = _build_env_pool(server_urls)
    rollout_per_stage = int(getattr(trainer.args, "rollouts_per_stage", 1280))
    initial_max_turn = int(getattr(trainer.args, "initial_max_turn", 2))
    final_max_turn = int(os.environ.get("LIARS_DICE_FINAL_MAX_TURN", "20"))
    initial_hint_prob = float(os.environ.get("LIARS_DICE_INITIAL_HINT_PROB", "0.0"))
    final_hint_prob = float(os.environ.get("LIARS_DICE_FINAL_HINT_PROB", "0.0"))

    _ROLLOUT_STATE["rank"] = rank
    _ROLLOUT_STATE["env_pool"] = env_pool
    _ROLLOUT_STATE["num_servers"] = len(env_pool)
    _ROLLOUT_STATE["thread_pool"] = ThreadPoolExecutor(max_workers=len(env_pool))
    _ROLLOUT_STATE["generation_semaphore"] = Semaphore(1)
    _ROLLOUT_STATE["curriculum"] = CurriculumScheduler(
        initial_max_turn=initial_max_turn,
        final_max_turn=final_max_turn,
        rollouts_per_stage=rollout_per_stage,
        initial_hint_prob=initial_hint_prob,
        final_hint_prob=final_hint_prob,
        warmup_rollouts=128,
    )
    _ROLLOUT_STATE["initialized"] = True

    trace_enabled = _is_truthy_env(os.environ.get("EPISODE_TRACE_ENABLED", "1"))
    trace_dir = os.environ.get("EPISODE_TRACE_DIR", "").strip()
    _ROLLOUT_STATE["trace_logger"] = None
    if trace_enabled and trace_dir:
        try:
            _ROLLOUT_STATE["trace_logger"] = EpisodeTraceLogger(trace_dir=trace_dir, rank=rank)
        except Exception as e:
            print(f"[EPISODE_TRACE] Failed to initialize logger: {e}")
    elif rank == 0:
        print("[EPISODE_TRACE] Disabled (set EPISODE_TRACE_ENABLED=1 and EPISODE_TRACE_DIR)")


def _reset_environment(env_endpoint: str, game_id: int, timeout: int) -> tuple[str, str]:
    payload = {"task_id": game_id, "seed": random.randint(0, 2**31 - 1), **MCTS_CONFIG}
    reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=timeout)
    reset_res.raise_for_status()
    reset_data = reset_res.json()
    result_block = reset_data["result"]
    episode_id = result_block.get("episode_id", "")
    raw_observation = result_block.get("observation", "")
    return episode_id, extract_and_format_observation(raw_observation)


def _step_environment(
    env_endpoint: str,
    episode_id: str,
    action_to_send: str,
    timeout: int,
) -> tuple[str, float, bool, dict]:
    step_payload = {"action": action_to_send, "episode_id": episode_id}
    step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=timeout)
    step_res.raise_for_status()
    step_data = step_res.json()
    step_block = step_data["result"]
    raw_observation = step_block.get("observation", "")
    formatted_observation = extract_and_format_observation(raw_observation)
    step_reward = _safe_float(step_block.get("reward", 0.0), default=0.0)
    done = bool(step_block.get("done", False))
    return formatted_observation, step_reward, done, step_block


def _last_prompt_fallback_result() -> dict:
    return {
        "prompt_ids": [1],
        "completion_ids": [1],
        "logprobs": [1.0],
        "reward": 0.0,
        "final_score": 0.0,
    }


def _full_prompt_fallback_result() -> dict:
    return {
        "prompt_ids": [1],
        "completion_ids": [1],
        "action_mask": [0],
        "logprobs": [1.0],
        "reward": 0.0,
        "final_score": 0.0,
    }


def _execute_parallel_rollouts(prompts, executor, run_single_prompt, fallback_builder):
    results = [None] * len(prompts)
    futures = [executor.submit(run_single_prompt, i, p) for i, p in enumerate(prompts)]

    for future in as_completed(futures):
        try:
            idx, res = future.result()
        except Exception as exc:
            # audit P1: one crashed game thread must not kill the GRPO step.
            print(f"[ERROR] Game thread threw unhandled exception: {type(exc).__name__}: {exc}")
            continue
        results[idx] = res if res is not None else fallback_builder()

    return [r for r in results if r is not None]


def _log_batch_statistics(list_results: list[dict]) -> None:
    finished = sum(1 for r in list_results if r["final_score"] != 0)
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0.0
    print(f"[BATCH] Finished: {finished}/{len(list_results)}, AvgReturn: {avg_return:.3f}")


def _get_system_prompt(use_hints: bool) -> str:
    system_prompt = """You are playing liars_dice.

# Game Rules
LIAR'S DICE RULES:

Setup: Each player has N dice (1-5 depending on variant). All players roll their dice secretly.

Goal: Make bids about total dice across ALL players, or call "Liar" on opponent's bid.

Actions:
- Bid (quantity, face): Claim there are at least 'quantity' dice showing 'face' among all dice.
- Call Liar: Challenge the previous bid.

Bidding rules: Each bid must be higher than the previous bid. "Higher" means:
  - Same face value but higher quantity (e.g., "2 fours" beats "1 four")
  - Same quantity but higher face value (e.g., "2 fives" beats "2 fours")

Wild dice: 6s are WILD and count as ANY face value.
- When counting dice for a bid, include 6s in the count
- Example: Bid "3 fours" means at least 3 dice showing EITHER 4 OR 6

Winning: If you call Liar and previous bid was false, opponent loses. If bid was true or exact, you lose.

# Output Format
You must respond with ONLY the action ID (a single number). 
Do NOT include descriptions or explanations. 
Examples:
- For action "59 -> 10-6": respond "59"
- For action "60 -> Liar": respond "60"
"""
    if use_hints:
        system_prompt += "\n" + STRATEGY_TIPS
    return system_prompt


def _rollout_parallelized_curriculum(
    prompts: list[str],
    trainer,
    include_action_mask: bool,
) -> dict[str, list]:
    _initialize_rollout_state(trainer)

    rank = _ROLLOUT_STATE["rank"]
    env_pool = _ROLLOUT_STATE["env_pool"]
    num_servers = _ROLLOUT_STATE["num_servers"]
    curriculum: CurriculumScheduler = _ROLLOUT_STATE["curriculum"]
    trace_logger = _ROLLOUT_STATE["trace_logger"]

    tokenizer = trainer.processing_class
    timeout = REQUEST_TIMEOUT_SECONDS
    current_max_turn = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(
        f"[CURRICULUM] Rollout {curriculum.total_rollouts}: "
        f"max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}"
    )

    def run_single_prompt(index: int, prompt: str):
        game_id = int(prompt)
        server_idx = (index + rank) % num_servers
        server = env_pool[server_idx]
        env_endpoint = server["base_url"]

        invalid_count = 0
        done = False
        final_reward = 0.0
        turn_number = 0
        accumulated_shaping_reward = 0.0
        step_records = []
        termination_reason = "unknown"
        last_step_block: dict = {}

        if include_action_mask:
            episode_prompt_ids: list[int] = []
            episode_completion_ids: list[int] = []
            episode_logprobs: list[float] = []
            episode_action_mask: list[int] = []
            prev_full_ids: list[int] | None = None
        else:
            prompt_ids_last: list[int] = []
            completion_ids_last: list[int] = []
            logprobs_last: list[float] = []

        try:
            episode_id, formatted_observation = _reset_environment(
                env_endpoint=env_endpoint,
                game_id=game_id,
                timeout=timeout,
            )
        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            if trace_logger and trace_logger.should_log():
                trace_logger.log_episode(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "game_id": game_id,
                        "status": "reset_failed",
                        "error": str(e),
                    }
                )
            return index, None

        use_hints = random.random() < current_hint_prob
        messages = [
            {"role": "system", "content": _get_system_prompt(use_hints=use_hints)},
            {"role": "user", "content": formatted_observation},
        ]

        while not done and turn_number < current_max_turn:
            observation_before_action = formatted_observation
            legal_action_map = _extract_legal_action_map(observation_before_action)
            state_features = _extract_state_features(observation_before_action)

            if not legal_action_map:
                accumulated_shaping_reward -= INVALID_ACTION_PENALTY
                termination_reason = "no_legal_actions"
                break

            with _ROLLOUT_STATE["generation_semaphore"]:
                try:
                    rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]
                except Exception as exc:
                    # audit P1: one vLLM error must end THIS episode, not the batch.
                    print(
                        f"Warning: vLLM error at turn {turn_number} "
                        f"(game {game_id}): {type(exc).__name__}: {exc}"
                    )
                    done = True
                    break

            prompt_ids = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            if include_action_mask:
                if len(prompt_ids) > MAX_PROMPT_LEN:
                    print(
                        f"Warning: Prompt exceeded {MAX_PROMPT_LEN} tokens ({len(prompt_ids)}) at turn {turn_number}"
                    )
                    termination_reason = "max_prompt_len_exceeded"
                    break

                if turn_number == 0:
                    episode_prompt_ids = prompt_ids
                    prev_full_ids = prompt_ids.copy()
                else:
                    if prev_full_ids is None:
                        prev_full_ids = prompt_ids.copy()
                    elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                        prev_full_ids = prompt_ids.copy()
                    else:
                        delta_prompt_ids = prompt_ids[len(prev_full_ids) :]
                        if delta_prompt_ids:
                            episode_completion_ids.extend(delta_prompt_ids)
                            episode_logprobs.extend([0.0] * len(delta_prompt_ids))
                            episode_action_mask.extend([0] * len(delta_prompt_ids))
                        prev_full_ids = prompt_ids.copy()

                if completion_ids:
                    episode_completion_ids.extend(completion_ids)
                    episode_logprobs.extend(logprobs)
                    episode_action_mask.extend([1] * len(completion_ids))
                    if prev_full_ids is not None:
                        prev_full_ids = prev_full_ids + completion_ids
            else:
                prompt_ids_last = prompt_ids
                completion_ids_last = completion_ids
                logprobs_last = logprobs

            messages.append({"role": "assistant", "content": completion_text})

            action_to_send = _parse_action_id(completion_text, legal_action_map)
            parse_failed = not action_to_send
            if parse_failed or action_to_send not in legal_action_map:
                invalid_count += 1
                accumulated_shaping_reward -= INVALID_ACTION_PENALTY
                action_to_send = _select_fallback_action(legal_action_map, state_features)

            action_label = legal_action_map.get(action_to_send, "")
            liar_action = _is_liar_label(action_label)
            parsed_bid = _extract_bid_tuple(action_label)

            bid_shaping = 0.0
            decision_shaping = 0.0
            current_bid_z = 0.0
            current_bid_truth_probability = 0.0
            if parsed_bid is not None:
                bid_shaping = _score_bid_plausibility(state_features, parsed_bid)
            decision_shaping, decision_meta = _score_challenge_decision(
                state_features=state_features,
                chose_liar=liar_action,
                proposed_bid=parsed_bid,
            )
            current_bid_z = float(decision_meta.get("current_bid_z", 0.0))
            current_bid_truth_probability = float(decision_meta.get("current_bid_truth_probability", 0.0))
            step_shaping = _clamp(
                bid_shaping + decision_shaping,
                -PER_STEP_SHAPING_CLIP,
                PER_STEP_SHAPING_CLIP,
            )
            accumulated_shaping_reward += step_shaping

            try:
                formatted_observation, step_reward, done, last_step_block = _step_environment(
                    env_endpoint=env_endpoint,
                    episode_id=episode_id,
                    action_to_send=action_to_send,
                    timeout=timeout,
                )
            except Exception as e:
                print(f"Step failed: {e}")
                formatted_observation = ""
                step_reward = -0.01
                done = False
                invalid_count += 1
                accumulated_shaping_reward -= INVALID_ACTION_PENALTY
                last_step_block = {"reward": step_reward, "done": False}

            observation_lower = formatted_observation.lower()
            invalid_or_noop = (
                "invalid" in observation_lower
                or "nothing happens" in observation_lower
                or "nothing happened" in observation_lower
            )
            if invalid_or_noop:
                invalid_count += 1
                accumulated_shaping_reward -= INVALID_ACTION_PENALTY

            if done:
                final_reward = _extract_terminal_reward(last_step_block, formatted_observation)
                termination_reason = "done"
            else:
                messages.append({"role": "user", "content": formatted_observation})

            step_records.append(
                {
                    "turn": turn_number,
                    "assistant_text": trace_logger.clip_text(completion_text) if trace_logger else completion_text,
                    "parsed_action": action_to_send,
                    "action_label": action_label,
                    "observation_before_action": (
                        trace_logger.clip_text(observation_before_action)
                        if trace_logger
                        else observation_before_action
                    ),
                    "observation_after_action": (
                        trace_logger.clip_text(formatted_observation) if trace_logger else formatted_observation
                    ),
                    "step_reward": float(step_reward),
                    "bid_shaping": float(bid_shaping),
                    "decision_shaping": float(decision_shaping),
                    "current_bid_z": float(current_bid_z),
                    "current_bid_truth_probability": float(current_bid_truth_probability),
                    "done": bool(done),
                    "invalid_or_noop": invalid_or_noop,
                    "parse_failed": bool(parse_failed),
                }
            )

            turn_number += 1

        if not done:
            if termination_reason == "unknown":
                termination_reason = "max_turn_reached"
            final_reward = 0.0

        clipped_shaping = _clamp(accumulated_shaping_reward, -SHAPING_REWARD_CLIP, SHAPING_REWARD_CLIP)
        train_reward = final_reward + clipped_shaping

        print(
            f"[ID:{game_id} Done:{int(done)} T:{turn_number:2d} "
            f"Env:{final_reward:+.3f} Shape:{accumulated_shaping_reward:+.3f} "
            f"ClipShape:{clipped_shaping:+.3f} Inv:{invalid_count}"
        )

        if trace_logger and trace_logger.should_log():
            trace_logger.log_episode(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "game_id": game_id,
                    "episode_id": episode_id,
                    "environment": "liars_dice",
                    "status": "completed" if done else "truncated",
                    "termination_reason": termination_reason,
                    "turns": turn_number,
                    "final_reward": float(final_reward),
                    "raw_shaping_reward": float(accumulated_shaping_reward),
                    "clipped_shaping_reward": float(clipped_shaping),
                    "train_reward": float(train_reward),
                    "invalid_count": invalid_count,
                    "steps": step_records,
                }
            )

        if include_action_mask:
            if len(episode_completion_ids) > MAX_EPISODE_TOKENS:
                episode_completion_ids = episode_completion_ids[:MAX_EPISODE_TOKENS]
                episode_logprobs = episode_logprobs[:MAX_EPISODE_TOKENS]
                episode_action_mask = episode_action_mask[:MAX_EPISODE_TOKENS]

            return index, {
                "prompt_ids": episode_prompt_ids,
                "completion_ids": episode_completion_ids,
                "action_mask": episode_action_mask,
                "logprobs": episode_logprobs,
                "reward": train_reward,
                "final_score": final_reward,
            }

        return index, {
            "prompt_ids": prompt_ids_last,
            "completion_ids": completion_ids_last,
            "logprobs": logprobs_last,
            "reward": train_reward,
            "final_score": final_reward,
        }

    executor = _ROLLOUT_STATE["thread_pool"]
    fallback_builder = _full_prompt_fallback_result if include_action_mask else _last_prompt_fallback_result
    list_results = _execute_parallel_rollouts(
        prompts=prompts,
        executor=executor,
        run_single_prompt=run_single_prompt,
        fallback_builder=fallback_builder,
    )

    curriculum.step(len(prompts))
    _log_batch_statistics(list_results)

    if include_action_mask:
        return {
            "prompt_ids": [r["prompt_ids"] for r in list_results],
            "completion_ids": [r["completion_ids"] for r in list_results],
            "action_mask": [r["action_mask"] for r in list_results],
            "logprobs": [r["logprobs"] for r in list_results],
            "env_rewards": [r["reward"] for r in list_results],
        }

    return {
        "prompt_ids": [r["prompt_ids"] for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "logprobs": [r["logprobs"] for r in list_results],
        "env_rewards": [r["reward"] for r in list_results],
    }


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    del max_turns  # Curriculum controls effective horizon.
    return _rollout_parallelized_curriculum(prompts=prompts, trainer=trainer, include_action_mask=False)


def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    del max_turns  # Curriculum controls effective horizon.
    return _rollout_parallelized_curriculum(prompts=prompts, trainer=trainer, include_action_mask=True)


def rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)


def _curriculum_factory(args) -> CurriculumScheduler:
    """Construct the LD curriculum from training args. Required by env_configs registry."""
    return CurriculumScheduler(
        initial_max_turn=args.initial_max_turn,
        final_max_turn=20,
        rollouts_per_stage=args.rollouts_per_stage,
        initial_hint_prob=0.0,
        final_hint_prob=0.0,
        warmup_rollouts=args.rollouts_per_stage,
    )