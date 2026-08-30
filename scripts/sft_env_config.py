"""
Orchestrator config for EnvTask SFT training.
Counterpart of instruct_config.py for the SFT environment mode.
No GRPO content; no vLLM/num_generations/beta.
"""

import os
from copy import deepcopy


def _env_int(name: str, default: int) -> int:
    """Env-var int that never raises at import time."""
    try:
        return int(float(os.environ.get(name) or default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


def _log(*args, **kwargs):
    """Flush-by-default print for the SFT orchestrator. Each concurrent per-env gen
    process buffers its stdout independently, so flush keeps the interleaved logs live."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)
from lrs_lookup import get_lr_from_ar_instruct
from model_utility import (
    disable_flash_attention,
    get_gpu_count,
    get_model_architecture,
    get_model_num_params,
    get_use_liger,
)

# Envs whose rows are long enough to need their own micro-batch (see get_training_json).
_SWE_ENV = "swe_infinite"
_SWE_MAX_MICRO_BATCH = int(os.environ.get("SWE_MAX_MICRO_BATCH", "2"))  # per-device rows; 2 fits the curated 8192-token SWE transcripts
# SWE submit config — CURATED (2026-07-27, replaces the 2026-07-20 accept-all). accept-all
# (~49k rows @ 20480/1-epoch) was proven NON-VIABLE on the VM: ~10.75h/epoch >> the ~3h
# tournament budget (trains <30%, under-fits) AND 96% 'incomplete' data distils give-up ->
# "No patch generated" -> 0.0. Curated = submitted+edit rows (swe_trajectories defaults),
# max_length 8192 (median ~22k chars ~5.5k tok fits), micro-batch 2, 3 epochs -> completes
# a full epoch in budget on ~1683 rows. Non-SWE envs keep 4096 / 1 epoch (byte-identical).
# Accept-all A/B still available via SWE_EXIT_STATUS=all / SWE_REQUIRE_EDIT=0 / SWE_MAX_LENGTH
# / SWE_EPOCHS. ORPO preference stage: SWE_ORPO=1 (gated OFF by default).
_DEFAULT_MAX_LENGTH = 4096
_SWE_MAX_LENGTH = int(os.environ.get("SWE_MAX_LENGTH", "8192"))
_SWE_EPOCHS = int(os.environ.get("SWE_EPOCHS", "3"))
# The GPU count the env recipe was validated at (env tasks used to get a fixed 4xH100).
# grad-accum is scaled to keep the global batch = batch x this, regardless of the actual
# allocation, so the reduced 1x/2x R1 allocation doesn't halve the effective batch.
_VALIDATED_ENV_GPU_COUNT = 4

SFT_ENV_SIZE_CONFIG: dict[str, dict] = {
    "0_1_b":  {"lr": 5e-5,   "distributed": "ddp", "gpu_count": 1, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": False},
    "1_2_b":  {"lr": 5e-5,   "distributed": "ddp", "gpu_count": 1, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": False},
    "2_4_b":  {"lr": 5e-5,   "distributed": "ddp", "gpu_count": 1, "batch_size": 24,  "gradient_accumulation_steps": 1, "use_lora": False},
    "4_5_b":  {"lr": 7e-5,   "distributed": "ddp", "gpu_count": 2, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": True},
    "5_9_b":  {"lr": 3.5e-5, "distributed": "ddp", "gpu_count": 2, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": True},
    "9_12_b": {"lr": 1e-4,   "distributed": "ddp", "gpu_count": 2, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": True},
    "12_15_b":{"lr": 1e-4,   "distributed": "ds",  "gpu_count": 4, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": True},
    "15_40_b":{"lr": 8e-5,   "distributed": "ds",  "gpu_count": 4, "batch_size": 16,  "gradient_accumulation_steps": 2, "use_lora": True},
    "40_80_b":{"lr": 8e-5,   "distributed": "ds",  "gpu_count": 8, "batch_size": 16,  "gradient_accumulation_steps": 2, "use_lora": True},
}

for _key in SFT_ENV_SIZE_CONFIG:
    SFT_ENV_SIZE_CONFIG[_key]["label"] = _key


def get_sft_env_config(param_nums: int) -> dict:
    if param_nums is None:
        raise ValueError("Cannot determine model size: weight counting failed")
    result = {"lr": 4e-5, "distributed": "ds", "gpu_count": 8, "batch_size": 6, "use_lora": True}
    if param_nums < 1_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["0_1_b"]
    elif param_nums < 2_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["1_2_b"]
    elif param_nums < 4_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["2_4_b"]
    elif param_nums < 5_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["4_5_b"]
    elif param_nums < 9_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["5_9_b"]
    elif param_nums < 12_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["9_12_b"]
    elif param_nums < 15_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["12_15_b"]
    elif param_nums < 40_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["15_40_b"]
    elif param_nums < 80_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["40_80_b"]
    else:
        _log(f"Model size {param_nums} is not supported, using 40_80_b")
    return deepcopy(result)


def get_generation_time_ratio(param_nums: int) -> float:
    """Fraction of total time budget spent on data generation, by model size.

    Smaller models train faster, so they can afford to spend more of the
    fixed wall-clock budget generating data.
    """
    if param_nums < 2_000_000_000:
        return 0.25
    elif param_nums < 4_000_000_000:
        return 0.2
    elif param_nums < 6_000_000_000:
        return 0.15
    else:
        return 0.12


def get_run_cmd(config: dict, gpu_nums: int) -> str:
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger_kernel",
        "optimizer",
        "use_lora",
        "packing",
        "disable_fa",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

    gpu_nums = get_gpu_count()
    run_type = config["distributed"]
    if gpu_nums > 1 and run_type == "ddp":
        start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    elif run_type == "ds":
        start_cmd = "deepspeed"
    else:
        start_cmd = "python"

    template = (
        start_cmd
        + """ train_sft_env.py \
    --request_path {request_path} \
    --bf16 True \
    --report_to wandb \
    --output_dir {output_dir} \
    --num_train_epochs {epoch_num} \
    --per_device_train_batch_size {batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --eval_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy epoch \
    --logging_steps 5 \
    --learning_rate {learning_rate} \
    --weight_decay 0. \
    --warmup_steps 35 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger_kernel {use_liger_kernel} \
    --packing {packing} \
    --disable_fa {disable_fa} \
    --max_length {max_length}"""
    )

    if run_type == "ds":
        template += " --deepspeed ds_config/zero3.json"

    if config.get("use_lora", False):
        template += " --use_lora True"

    if not config.get("disable_fa", False):
        template += " --padding_free True"

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))

    return template


def get_training_json(train_info: dict) -> dict:
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_path)
    config = get_sft_env_config(param_nums)

    task_id = train_info["task_id"]
    env_names = train_info.get("dataset_type", {}).get("environment_names") or ["liars_dice"]
    dataset_path = f"/workspace/scripts/datasets/sft_env_{task_id}"

    run_config = {
        "epoch_num": _SWE_EPOCHS if _SWE_ENV in env_names else 1,
        "max_length": _SWE_MAX_LENGTH if _SWE_ENV in env_names else _DEFAULT_MAX_LENGTH,
        "batch_size": config["batch_size"],
        "learning_rate": config["lr"],
        "min_lr_rate": 0.25,
        "use_liger_kernel": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": config.get("use_lora", False),
        "disable_fa": disable_flash_attention(model_architecture),
        "packing": "False",  # pre-tokenised dataset; TRL packing not used
        "gpu_nums": config["gpu_count"],
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": config.get("distributed", "ddp"),
        "gradient_checkpointing": "True",
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
    }

    # Hold the EFFECTIVE (global) batch invariant to the GPU allocation. The whole
    # recipe — LR, warmup, epochs — was validated when environment tasks always got a
    # FIXED 4xH100 (validator gpu_requirements returned H100_4X for env), i.e.
    # batch 32 x 4 GPUs = 128. Our forfeit diagnosis pinned this exactly: eff-64
    # under-trained game_action emission and FORFEITED, eff-128 fixed it. PR #1321
    # drops env R1 to 1xH100 (<=4B) / 2xH100 (>4B), so a flat per-device batch would
    # SILENTLY HALVE the effective batch to 64 — back in the forfeit regime — and the
    # frozen boss (batch 32, grad-accum 1) is now stuck exactly there. We instead scale
    # grad-accum by the ACTUAL gpu count (what torchrun will use — get_run_cmd reads the
    # same get_gpu_count()) so the global batch stays at the validated target on ANY
    # allocation. grad-accum only accumulates gradients; it does NOT raise per-step
    # memory. At the old 4xH100 this yields grad-accum 1 (byte-identical to before).
    #
    # SWE additionally needs a smaller per-DEVICE micro-batch: its agentic transcripts
    # pack out near max_length and OOM'd every rank at batch 32 on 80GB. Cap the
    # micro-batch for memory FIRST, then let grad-accum rebuild the same global batch.
    _target_effective = run_config["batch_size"] * _VALIDATED_ENV_GPU_COUNT
    _actual_gpu = max(1, get_gpu_count())
    if _SWE_ENV in env_names:
        run_config["batch_size"] = max(1, min(run_config["batch_size"], _SWE_MAX_MICRO_BATCH))
    run_config["gradient_accumulation_steps"] = max(
        1, round(_target_effective / (run_config["batch_size"] * _actual_gpu))
    )
    _log(
        f"[sft-env-config] effective-batch invariance: {_actual_gpu}gpu x "
        f"micro-batch {run_config['batch_size']} x grad-accum {run_config['gradient_accumulation_steps']} "
        f"= {run_config['batch_size'] * _actual_gpu * run_config['gradient_accumulation_steps']} "
        f"(target {_target_effective}"
        f"{'; SWE micro-batched' if _SWE_ENV in env_names else ''})",
        flush=True,
    )

    if train_info.get("find_lk_lr"):
        lr = get_lr_from_ar_instruct(model_architecture, param_nums)
        if lr is not None:
            _log(f"Using lr from architecture config: {lr}", flush=True)
            run_config["learning_rate"] = lr
        else:
            _log(f"Using lr from config: {run_config['learning_rate']}", flush=True)

    run_config["learning_rate"] *= train_info["reg_ratio"]

    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])

    train_request = deepcopy(train_info)
    train_request["dataset_path"] = dataset_path
    train_request["save_before_remaining_time"] = 5
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 200
    train_request["checking_step"] = 70
    train_request["min_steps"] = max(
        int(train_info["hours_to_complete"] * 70),
        train_info.get("min_steps", 100),
    )

    gen_seconds = int(train_info["hours_to_complete"] * 3600 * get_generation_time_ratio(param_nums))
    # Per-env trajectory generation via OUR env subsystem (our_envs): game envs roll out our
    # per-game teacher vs an in-process OpenSpiel MCTS opponent (teacher-vs-MCTS, the
    # eval-aligned state distribution that fixed the clobber forfeit); intercode uses our
    # eval-grounded synthetic ReAct generator; swe_infinite passes the mounted SWE-ZERO
    # mini-swe-agent trajectories through. Each step writes a {"messages": ...} DatasetDict to a
    # per-env path; merge_trajectories combines them into dataset_path (what train_sft_env loads
    # + masks).
    #
    # PARALLEL: every env generates concurrently (backgrounded subshells + `wait`), so the whole
    # env set fits inside ONE gen-time window (--max-gen-seconds) instead of the sum over envs —
    # essential for the multi-env tournament budget. When >1 game env runs at once, the in-process
    # worker pools are split across cores so the N pools don't oversubscribe (each game worker
    # holds a live pyspiel state); the offline envs (intercode/swe) are light and left uncapped.
    _GAME_ARGS = {
        "clobber":     "--num_games 8500 --max_turn 70 --window_turns 12 --window_step 6",
        "othello":     "--num_games 6000 --max_turn 64 --window_turns 12 --window_step 6",
        "gin_rummy":   "--num_games 6000 --max_turn 40 --window_turns 10 --window_step 5",
        "leduc_poker": "--num_games 8000 --max_turn 12 --window_turns 8  --window_step 4",
        "goofspiel":   "--num_games 6000 --max_turn 26 --window_turns 10 --window_step 5",
        "liars_dice":  "--num_games 8000 --max_turn 16 --window_turns 8  --window_step 4",
    }
    _game_envs = [e for e in env_names if e not in ("intercode", "swe_infinite")]
    _game_workers = max(1, (os.cpu_count() or 8) // len(_game_envs)) if len(_game_envs) > 1 else 0
    _per_env_paths: list[str] = []
    _steps: list[str] = []
    for _env in env_names:
        _out = f"{dataset_path}_{_env}"
        _per_env_paths.append(_out)
        if _env == "intercode":
            _steps.append(
                f"python -m our_envs.intercode_synth_gen --output_path {_out}"
                f" --per_fs 4000 --max_per_fs 600"
            )
        elif _env == "swe_infinite":
            _steps.append(f"python -m our_envs.swe_trajectories --output_path {_out}")
        else:
            _game_args = _GAME_ARGS.get(
                _env, "--num_games 5000 --max_turn 30 --window_turns 10 --window_step 5"
            )
            _wc = f" --num_workers {_game_workers}" if _game_workers else ""
            _steps.append(
                f"python -m our_envs.generate_trajectories"
                f" --environment_name {_env}"
                f" --output_path {_out}"
                f" {_game_args}{_wc}"
                f" --max-gen-seconds {gen_seconds}"
                f" --wins-only"
            )
    # Parallel: each env's gen runs as a backgrounded subshell; `wait` blocks for all; merge after.
    _parallel = " & ".join(f"({s})" for s in _steps) + " & wait"
    # Per-env balancing. merge_trajectories implements it, but the command built
    # here never passed --target_per_env, so _balance_split returned immediately
    # and the merge was a plain concatenation. On a joint task that is not
    # neutral: goofspiel generates 6000 games through a sliding window while
    # intercode is capped at --max_per_fs 600 per filesystem, so the scored
    # intercode half of the task was outnumbered by more than an order of
    # magnitude. --max_upsample still guards a low-unique env from being
    # repeated into overfit.
    _target = _env_int("SFT_TARGET_PER_ENV", 20000)
    _max_up = _env_float("SFT_MAX_UPSAMPLE", 3.0)
    _merge = (
        f"python -m our_envs.merge_trajectories"
        f" --input_paths {' '.join(_per_env_paths)}"
        f" --output_path {dataset_path}"
        f" --target_per_env {_target}"
        f" --max_upsample {_max_up}"
    )
    generate_cmd = f"{_parallel} && {_merge}"

    print("Run command:", run_cmd)

    return {
        "train_request": train_request,
        "run_cmd": run_cmd,
        "generate_cmd": generate_cmd,
    }
