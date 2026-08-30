"""Preflight checks for an ENVIRONMENT task, to run inside the trainer image.

Not part of training.  This exists because the failures that cost a tournament
run are the quiet ones: an env name that resolves to nothing, a rollout that
returns zero usable episodes, a generation helper that silently ignores
batching.  Each check below prints PASS/FAIL and the reason.

    python -m tools.preflight                 # static checks only
    python -m tools.preflight --probe-env     # also reset/step every game server
    python -m tools.preflight --probe-env --game goofspiel
"""

import argparse
import inspect
import os
import sys

# The six PvP games an environment tournament can assign
# (G.O.D core/constants/environments.py).
PVP_GAMES = ["gin_rummy", "liars_dice", "leduc_poker", "othello", "clobber", "goofspiel"]

# How each game turns a raw sidecar observation into the prompt body the model
# sees.  This matters for the probe: several games do not receive a rendered
# "<id> -> <label>" block at all -- goofspiel SYNTHESISES one from "P0 hand:",
# othello and clobber re-wrap the state and inject a colour prefix.  Parsing
# legal ids straight off the raw observation therefore proves nothing; the
# transform is exactly the layer that has to be right.
# (module, attribute, needs_instantiation)
OBS_TRANSFORMS = {
    "goofspiel":   ("our_envs.goof_spiel_env", "extract_and_format_observation", False),
    "gin_rummy":   ("our_envs.gin_rummy_opponent_modeling", "extract_and_format_observation", False),
    "liars_dice":  ("our_envs.liar_dice_opponent_modeling", "extract_and_format_observation", False),
    "leduc_poker": ("our_envs.leduc_poker_opponent_modeling", "_format_observation", False),
    "othello":     ("our_envs.othello_trajectories", "_OthelloObsTransform", True),
    "clobber":     ("our_envs.clobber_trajectories", "_ClobberObsTransform", True),
}


def load_obs_transform(game: str):
    """Return the game's raw-observation -> prompt-body callable, or None."""
    entry = OBS_TRANSFORMS.get(game)
    if entry is None:
        return None
    module_name, attr, instantiate = entry
    try:
        module = __import__(module_name, fromlist=[attr])
        obj = getattr(module, attr)
        return obj() if instantiate else obj
    except Exception as exc:
        check(f"{game} obs_transform import", False, f"{type(exc).__name__}: {exc}")
        return None

_ok = True
_MAX_OBS_LINES = 40


def check(label: str, passed: bool, detail: str = "") -> None:
    global _ok
    if not passed:
        _ok = False
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_versions() -> None:
    section("Runtime")
    print(f"        python: {sys.executable}")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    check(
        "running inside the trainer venv",
        in_venv,
        "packages live in /workspace/.grpo_env -- "
        "source /workspace/.grpo_env/bin/activate first",
    )
    try:
        import torch

        cuda = torch.cuda.is_available()
        check("CUDA visible", cuda)
        if cuda:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(
                    f"        gpu{i}: {props.name} "
                    f"{props.total_memory / 1024**3:.0f}GB sm_{props.major}{props.minor}"
                )
        print(f"        torch {torch.__version__}")
    except Exception as exc:
        check("import torch", False, str(exc))

    for mod in ("trl", "vllm", "transformers", "peft", "bitsandbytes", "flash_attn"):
        try:
            m = __import__(mod)
            print(f"        {mod} {getattr(m, '__version__', '?')}")
        except Exception as exc:
            check(f"import {mod}", False, str(exc))


def check_generation_api() -> None:
    section("Batched generation")
    try:
        from trl.experimental.openenv import generate_rollout_completions
    except Exception as exc:
        check("import generate_rollout_completions", False, str(exc))
        return
    check("import generate_rollout_completions", True)
    try:
        sig = inspect.signature(generate_rollout_completions)
        print(f"        signature: {sig}")
        check(
            "takes a 'prompts' list (batching possible)",
            "prompts" in sig.parameters,
            "batched_rollout falls back to one call per prompt if not",
        )
    except (TypeError, ValueError) as exc:
        check("inspect signature", False, str(exc))


def check_registry() -> None:
    section("Environment registry")
    try:
        from our_envs.env_configs import get_env_config
        from our_envs.shared_env import GAMES_TO_TASK_ID_RANGE
    except Exception as exc:
        check("import env_configs", False, str(exc))
        return

    for game in PVP_GAMES:
        has_range = game in GAMES_TO_TASK_ID_RANGE
        try:
            cfg = get_env_config(game)
            resolved = True
            detail = f"num_generations={cfg.num_generations}, ctx={cfg.vllm_max_model_length}"
        except Exception as exc:
            resolved = False
            detail = str(exc)
        check(f"{game:12s} task-id range", has_range)
        check(f"{game:12s} registry entry", resolved, detail)


def probe_env(game: str) -> None:
    """One reset + one step against a live sidecar."""
    try:
        import requests

        from our_envs.batched_rollout import legal_ids_from_observation
        from our_envs.shared_env import GAMES_TO_TASK_ID_RANGE
    except Exception as exc:
        check(f"{game} imports", False, f"{type(exc).__name__}: {exc}")
        return

    urls = [u.strip() for u in os.environ.get("ENVIRONMENT_SERVER_URLS", "").split(",") if u.strip()]
    if not urls:
        check("ENVIRONMENT_SERVER_URLS set", False, "no sidecar to probe")
        return
    print(f"        servers: {urls}")

    task_id = GAMES_TO_TASK_ID_RANGE[game][0] + 1
    payload = {
        "task_id": task_id,
        "seed": task_id,
        "opponent": "mcts",
        "mcts_max_simulations": 50,
        "mcts_num_rollouts": 1,
    }
    try:
        res = requests.post(f"{urls[0]}/reset", json=payload, timeout=120)
        res.raise_for_status()
        block = res.json()["result"]
    except Exception as exc:
        check(f"{game} reset", False, str(exc))
        return

    obs = block.get("observation", "")
    episode_id = block.get("episode_id", "")
    check(f"{game} reset", bool(obs), f"episode_id={episode_id!r}")

    print(f"        --- RAW observation ({len(obs.splitlines())} lines) ---")
    for line in obs.splitlines()[:_MAX_OBS_LINES]:
        print(f"        {line}")
    if len(obs.splitlines()) > _MAX_OBS_LINES:
        print(f"        ... ({len(obs.splitlines()) - _MAX_OBS_LINES} more)")

    transform = load_obs_transform(game)
    if transform is None:
        check(f"{game} obs_transform known", False, "no entry in OBS_TRANSFORMS")
        return
    try:
        body = transform(obs)
    except Exception as exc:
        check(f"{game} obs_transform runs", False, f"{type(exc).__name__}: {exc}")
        return
    check(f"{game} obs_transform runs", True)

    changed = body != obs
    check(
        f"{game} obs_transform changed the observation",
        changed,
        "returned the raw text unchanged -- its state regex did not match this "
        "sidecar's format, so the model would get no legal-action block",
    )

    print(f"        --- TRANSFORMED prompt body ({len(body.splitlines())} lines) ---")
    for line in body.splitlines()[:_MAX_OBS_LINES]:
        print(f"        {line}")
    if len(body.splitlines()) > _MAX_OBS_LINES:
        print(f"        ... ({len(body.splitlines()) - _MAX_OBS_LINES} more)")

    legal = legal_ids_from_observation(body)
    check(
        f"{game} legal ids parse after transform",
        bool(legal),
        f"{legal[:10]}{'...' if len(legal) > 10 else ''}" if legal else "none found",
    )

    if legal:
        try:
            res = requests.post(
                f"{urls[0]}/step",
                json={"action": str(legal[0]), "episode_id": episode_id},
                timeout=120,
            )
            res.raise_for_status()
            step = res.json()["result"]
            check(
                f"{game} step",
                "observation" in step or "done" in step,
                f"done={step.get('done')}, reward={step.get('reward')}",
            )
        except Exception as exc:
            check(f"{game} step", False, str(exc))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-env", action="store_true", help="hit the live env sidecar")
    ap.add_argument("--game", default=None, help="probe only this game")
    ap.add_argument("--obs-lines", type=int, default=40, help="observation lines to print")
    args = ap.parse_args()

    global _MAX_OBS_LINES
    _MAX_OBS_LINES = args.obs_lines

    print("PREFLIGHT -- environment task")
    print(f"  GEN_CHUNK={os.environ.get('GEN_CHUNK', '16 (default)')}  "
          f"STEP_WORKERS={os.environ.get('STEP_WORKERS', 'auto')}")

    check_versions()
    check_generation_api()
    check_registry()

    if args.probe_env:
        for game in [args.game] if args.game else PVP_GAMES:
            section(f"Live probe: {game}")
            probe_env(game)

    print("\n" + ("ALL CHECKS PASSED" if _ok else "SOME CHECKS FAILED -- see FAIL lines above"))
    return 0 if _ok else 1


if __name__ == "__main__":
    sys.exit(main())
