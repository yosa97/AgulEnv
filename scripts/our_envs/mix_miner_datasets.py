"""Mix validator-mounted whitelisted datasets into the synthetic SFT set.

Runs as the last generation step, after generate_trajectories.py (+ optional
merge_trajectories.py) has produced the synthetic combined DatasetDict at
--synthetic_path. It game-filters the mounted miner datasets to the task's
envs, concatenates them with the synthetic data, and writes the result to
--output_path, which tokenize_env_trajectories.py then consumes.

Never raises and never needs the network:
  - No validator mount (local runs, or nothing requested) -> the synthetic set
    is passed through to --output_path unchanged.
  - Any loader error -> same passthrough.

Usage:
  python -m our_envs.mix_miner_datasets \\
      --synthetic_path /workspace/scripts/datasets/sft_env_<task>_combined \\
      --output_path    /workspace/scripts/datasets/sft_env_<task>_mixed \\
      --env_names liars_dice leduc_poker gin_rummy [--cap_rows N]
"""

import argparse

from datasets import load_from_disk

from our_envs.miner_dataset_loader import build_miner_sft_dataset, merge_with_synthetic


def _passthrough(synthetic_path: str, output_path: str, reason: str) -> None:
    """Copy the synthetic DatasetDict to output_path unchanged."""
    dd = load_from_disk(synthetic_path)
    dd.save_to_disk(output_path)
    n_train = len(dd["train"]) if "train" in dd else 0
    print(
        f"[mix_miner_datasets] passthrough ({reason}); "
        f"synthetic-only train={n_train} -> {output_path}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic_path", required=True,
                   help="HF DatasetDict path produced by generate/merge trajectories.")
    p.add_argument("--output_path", required=True,
                   help="Where to write the mixed (or passed-through) DatasetDict.")
    p.add_argument("--env_names", nargs="+", default=[],
                   help="Task envs; miner rows are kept only when their game matches one.")
    p.add_argument("--cap_rows", type=int, default=None,
                   help="Optional cap on total miner rows mixed in.")
    args = p.parse_args()

    try:
        miner_dd = build_miner_sft_dataset(env_names=args.env_names, cap_rows=args.cap_rows)
    except Exception as exc:
        print(f"[mix_miner_datasets] loader error ({exc}); passing synthetic through", flush=True)
        _passthrough(args.synthetic_path, args.output_path, "loader-error")
        return

    if miner_dd is None:
        _passthrough(args.synthetic_path, args.output_path, "no-miner-data")
        return

    mixed = merge_with_synthetic(miner_dd, args.synthetic_path)
    if mixed is None or "train" not in mixed or len(mixed["train"]) == 0:
        _passthrough(args.synthetic_path, args.output_path, "empty-merge")
        return

    mixed.save_to_disk(args.output_path)
    miner_train = len(miner_dd["train"]) if "train" in miner_dd else 0
    print(
        f"[mix_miner_datasets] mixed miner+synthetic -> {args.output_path} "
        f"train={len(mixed['train'])} "
        f"validation={len(mixed['validation']) if 'validation' in mixed else 0} "
        f"(miner_train={miner_train}, games={args.env_names})",
        flush=True,
    )


if __name__ == "__main__":
    main()
