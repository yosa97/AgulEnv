"""Per-env baseline_stats helpers — both env-file readers and HP overrides.

Validator drops an EnvBaselineStats JSON at ``BASELINE_STATS_PATH`` when
``MODEL_PREP_ENABLED_ENV`` is True. This module is the single source of
truth for:
  * loading + caching the payload (``_load_once``, ``load_baseline_stats``)
  * classifying severity (``classify_augmentation_severity``)
  * computing the HP recovery override (``apply_augmentation_recovery_override``)
  * thin per-env score accessors used by rollout files

Importable from any layer: ``grpo_env_config.py`` (config builder),
``text_trainer.py`` (orchestrator), ``train_grpo_env.py`` (subprocess),
and any ``scripts/envs/<game>_opponent_modeling.py`` rollout file.

Kept dependency-free (stdlib only) so it loads without torch/transformers.
"""

from __future__ import annotations

import json
import os
from threading import Lock
from typing import Any


_CACHE: dict[str, Any] = {"loaded": False, "payload": None}
_CACHE_LOCK = Lock()


# --- Severity classification + HP override --------------------------------

_AUGMENTATION_RECOVERY_BUMP = {
    # severity: (lr_multiplier, warmup_steps, rollouts_per_stage_multiplier)
    "severe":   (2.0, 100, 2.0),
    "moderate": (1.5, 70,  1.0),
    "mild":     (1.0, 35,  1.0),  # telemetry only — no functional bump
    "none":     (1.0, 35,  1.0),
}


def _load_once() -> dict | None:
    if _CACHE["loaded"]:
        return _CACHE["payload"]
    with _CACHE_LOCK:
        if _CACHE["loaded"]:
            return _CACHE["payload"]
        payload: dict | None = None
        # Production path: validator embeds baseline_stats as a JSON field on
        # TrainRequest; the trainer service serialises it into BASELINE_STATS_JSON
        # env var on the training container so we can read it without writing
        # a fixture file to the shared cache volume.
        raw_json = os.environ.get("BASELINE_STATS_JSON", "").strip()
        if raw_json:
            try:
                payload = json.loads(raw_json)
            except Exception:
                payload = None
        # Test/legacy path: BASELINE_STATS_PATH points to a fixture file we
        # mount into the container (used by test_augment_detection.sh etc).
        if payload is None:
            path = os.environ.get("BASELINE_STATS_PATH", "").strip()
            if path and os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        payload = json.load(f)
                except Exception:
                    payload = None
        _CACHE["loaded"] = True
        _CACHE["payload"] = payload
        return payload


# Public alias for callers that prefer explicit naming
def load_baseline_stats() -> dict | None:
    return _load_once()


def classify_augmentation_severity(baseline_stats: dict | None) -> str:
    """Map per-env baseline mean_score to a severity bucket.

      mean >= 0.50 → none      (no augment / trivial)
      mean >= 0.30 → mild      (gaussian small std, scaling near-1)
      mean >= 0.15 → moderate  (pruning ~5-10%, scaling farther)
      mean <  0.15 → severe    (layer reinit / pruning all-layers)
    """
    if not baseline_stats or baseline_stats.get("task_type") != "env":
        return "none"
    env_stats = baseline_stats.get("env_stats") or {}
    means = [(s or {}).get("mean_score", 0.0) for s in env_stats.values()]
    means = [float(m) for m in means if m is not None]
    if not means:
        return "none"
    avg = sum(means) / len(means)
    if avg >= 0.50:
        return "none"
    if avg >= 0.30:
        return "mild"
    if avg >= 0.15:
        return "moderate"
    return "severe"


def apply_augmentation_recovery_override(config: dict, baseline_stats: dict | None) -> dict:
    """Return new config dict with severity-driven HP bumps applied.

    Only ``severe`` and ``moderate`` actually change config. ``mild`` and
    ``none`` return a label-tagged copy so downstream logs can see the tier
    without changing training behavior.

    Caller should invoke this AFTER bucket-level overrides (e.g. LD bucket
    override) so this is the final layer.
    """
    severity = classify_augmentation_severity(baseline_stats)
    lr_mult, warmup, rps_mult = _AUGMENTATION_RECOVERY_BUMP[severity]
    merged = dict(config)
    merged["augmentation_severity"] = severity  # for downstream logging only
    if severity in ("severe", "moderate"):
        merged["lr"] = float(config.get("lr", 1e-5)) * lr_mult
        merged["warmup_steps"] = warmup
        if severity == "severe":
            base_rps = int(config.get("rollouts_per_stage", 1280))
            merged["rollouts_per_stage"] = int(base_rps * rps_mult)
        print(
            f"[augmentation] severity={severity} -> "
            f"lr={merged['lr']:.2e} warmup_steps={warmup} "
            f"rollouts_per_stage={merged.get('rollouts_per_stage', config.get('rollouts_per_stage', 1280))}",
            flush=True,
        )
    else:
        print(f"[augmentation] severity={severity} -> no HP override", flush=True)
    return merged


# --- Per-env score accessors (used by rollout files) ----------------------


def get_env_baseline_mean(env_name: str, default: float = 0.5) -> float:
    """Return the per-env baseline mean_score, or ``default`` if absent.

    ``env_name`` is the bare env name (``liars_dice`` / ``leduc_poker`` /
    ``gin_rummy``) — match what validator stores in env_stats keys.
    """
    payload = _load_once()
    if not payload or payload.get("task_type") != "env":
        return default
    env_stats = (payload.get("env_stats") or {}).get(env_name)
    if not env_stats:
        return default
    try:
        return float(env_stats.get("mean_score", default))
    except (TypeError, ValueError):
        return default


def get_env_baseline_std(env_name: str, default: float = 0.0) -> float:
    """Per-env baseline std_score, or ``default`` if absent."""
    payload = _load_once()
    if not payload or payload.get("task_type") != "env":
        return default
    env_stats = (payload.get("env_stats") or {}).get(env_name)
    if not env_stats:
        return default
    try:
        return float(env_stats.get("std_score", default))
    except (TypeError, ValueError):
        return default


def has_baseline_stats() -> bool:
    """True iff BASELINE_STATS_PATH resolves to a parseable env JSON."""
    payload = _load_once()
    return bool(payload) and payload.get("task_type") == "env"


def get_weight_group_stats(group_name: str) -> dict | None:
    """Per-layer-group weight stats (rms / norm / max_abs), or None.

    Useful for detecting which layer family was augmented — e.g. a group
    whose ``weight_rms`` is far from the typical range for the model
    architecture probably got WEIGHT_SCALING or LAYER_REINIT.
    """
    payload = _load_once()
    if not payload:
        return None
    by_group = (payload.get("weights") or {}).get("by_group") or {}
    return by_group.get(group_name)


# For tests + smoke runners: reset the cache so a re-read picks up a fresh file.
def _reset_cache_for_testing() -> None:
    with _CACHE_LOCK:
        _CACHE["loaded"] = False
        _CACHE["payload"] = None
