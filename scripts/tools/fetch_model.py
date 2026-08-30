"""Fetch a base model into the trainer's cache layout.

The trainer sizes the model by counting parameters in the weight files ON DISK
(model_utility.get_model_size_from_local_path -> count_params_from_safetensors),
so the model must already be in /cache/models before training starts.  The
official example does this with a separate trainer-downloader image; this does
the same job from inside the trainer image, which already has huggingface_hub.

Layout must match trainer_downloader.download_axolotl_base_model:
    /cache/models/<repo_id with "/" replaced by "--">

    python -m tools.fetch_model unsloth/Meta-Llama-3.1-8B-Instruct
"""

import os
import sys

CACHE_MODELS_DIR = os.environ.get("CACHE_MODELS_DIR", "/cache/models")
_WEIGHT_SUFFIXES = (".safetensors", ".bin")


def has_weights(path: str) -> bool:
    try:
        return any(f.endswith(_WEIGHT_SUFFIXES) for f in os.listdir(path))
    except OSError:
        return False


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m tools.fetch_model <hf_repo_id>", file=sys.stderr)
        return 2
    repo_id = sys.argv[1]
    dest = os.path.join(CACHE_MODELS_DIR, repo_id.replace("/", "--"))

    if has_weights(dest):
        print(f"[fetch_model] already present: {dest}")
        return 0

    os.makedirs(CACHE_MODELS_DIR, exist_ok=True)
    print(f"[fetch_model] downloading {repo_id} -> {dest}", flush=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=dest,
        local_dir_use_symlinks=False,
    )

    if not has_weights(dest):
        print(
            f"[fetch_model] FAILED: no .safetensors/.bin under {dest} after download. "
            "The trainer counts parameters from these files and will abort with "
            "'Cannot determine model size'.",
            file=sys.stderr,
        )
        return 1
    print(f"[fetch_model] ok: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
