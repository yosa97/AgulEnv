#!/usr/bin/env bash
#
# Local ENVIRONMENT-task run for a PvP game, against THIS repo's trainer image.
#
# Adapted from G.O.D ops/examples/training/run_environment_task.sh, which
# differs in two ways that matter:
#
#   1. It builds the trainer from G.O.D's own reference dockerfile.  We build
#      from this repo's dockerfiles/standalone-text-trainer.dockerfile, because
#      the point is to exercise OUR rollout code.
#   2. Its default game is alfworld, whose servers this repo starts itself
#      (alfworld_run.sh), so it never needs a sidecar.  The six PvP games get a
#      sidecar from the validator instead -- image gradientsio/mcts-api:latest
#      (G.O.D core/constants: MCTS_API_DOCKER_IMAGE).  We start one here and
#      hand its URL to the trainer via ENVIRONMENT_SERVER_URLS.
#
# Usage:
#   GAMES=goofspiel,intercode MODEL=unsloth/Meta-Llama-3.1-8B-Instruct HOURS=1.5 \
#   HF_TOKEN=hf_xxx HF_USER=yourname \
#   bash scripts/tools/run_env_task_local.sh
#
# Games: gin_rummy liars_dice leduc_poker othello clobber goofspiel intercode
#
# IMPORTANT -- which code path this exercises.  text_trainer.py dispatches on
# the environment_names LIST:
#
#     env_names = dataset_type_dict.get("environment_names", [])
#     if env_names and all(supports_sft(n) for n in env_names):  -> SFT pipeline
#     else:                                                       -> GRPO pipeline
#
# All six PvP games sit in sft_env_configs._SFT_REGISTRY, and intercode /
# swe_infinite sit in _SFT_NON_ROLLOUT_ENVS, so ANY real task list takes the
# SFT pipeline (generate trajectories -> merge -> train_sft_env).  The GRPO
# pipeline only runs for a name in NEITHER set, e.g. alfworld.  Set FORCE_GRPO=1
# to append such a name and exercise the GRPO path instead.
set -euo pipefail

GAMES="${GAMES:-${GAME:-goofspiel}}"          # comma-separated; GAME still works
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
HOURS="${HOURS:-1.0}"
FORCE_GRPO="${FORCE_GRPO:-0}"
# Numeric, matching the official example -- it becomes a checkpoint dir name.
TASK_ID="${TASK_ID:-1}"
PRIMARY_GAME="${GAMES%%,*}"
REPO_NAME="${REPO_NAME:-env-smoke-$PRIMARY_GAME}"
HF_TOKEN="${HF_TOKEN:-}"
HF_USER="${HF_USER:-}"
WANDB_TOKEN="${WANDB_TOKEN:-}"

# Rollout knobs. Small chunk + more step workers is the right shape for a
# one-GPU box: the validator starts ~one sidecar per GPU, so stepping is the
# thing that needs concurrency, not generation.
GEN_CHUNK="${GEN_CHUNK:-8}"
STEP_WORKERS="${STEP_WORKERS:-16}"

# The trainer wants a dataset argument even for EnvTask; game data comes from
# the sidecar, so any small reachable file satisfies the downloader.
# Real EnvTask records carry the literal string "env_task_dummy_dataset" as
# training_data -- game data comes from the sidecar, not from a file.
DATASET="${DATASET:-env_task_dummy_dataset}"
FILE_FORMAT="${FILE_FORMAT:-s3}"

NET="${NET:-god-local}"
SIDECAR="${SIDECAR:-mcts-api-$PRIMARY_GAME}"
MCTS_IMAGE="${MCTS_IMAGE:-gradientsio/mcts-api:latest}"
TRAINER_IMAGE="${TRAINER_IMAGE:-agulenv-trainer}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINTS_DIR="$REPO_ROOT/secure_checkpoints"
OUTPUTS_DIR="$REPO_ROOT/outputs"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"
chmod 777 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"

echo "=== RESOLVED SETTINGS ==="
echo "    games : $GAMES"
echo "    model : $MODEL"
echo "    hours : $HOURS"
echo "    force_grpo=$FORCE_GRPO  gen_chunk=$GEN_CHUNK  step_workers=$STEP_WORKERS"
# A backslash-continued command line loses its inline assignments if any line
# has trailing whitespace after the backslash -- the assignments then run as
# their own statement and the script sees only defaults.  Say so rather than
# silently training the wrong model.
if [ "$GAMES" = "goofspiel" ] && [ "$MODEL" = "Qwen/Qwen2.5-3B-Instruct" ]; then
    echo "    NOTE: both GAMES and MODEL are at their defaults. If you meant to"
    echo "          override them, check for a space after a trailing backslash,"
    echo "          or export the variables before calling this script."
fi


cleanup() {
    echo "--- sidecar logs (tail) ---"
    docker logs --tail 40 "$SIDECAR" 2>&1 || true
    docker rm -f "$SIDECAR" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker network create "$NET" >/dev/null 2>&1 || true

# Build the environment_names JSON list the trainer actually reads.
NAMES_JSON=""
IFS=',' read -ra _GAME_ARR <<< "$GAMES"
for g in "${_GAME_ARR[@]}"; do
    g="$(echo "$g" | xargs)"
    [ -z "$g" ] && continue
    NAMES_JSON="${NAMES_JSON:+$NAMES_JSON, }\"$g\""
done
if [ "$FORCE_GRPO" = "1" ]; then
    NAMES_JSON="$NAMES_JSON, \"alfworld\""
    echo "    FORCE_GRPO=1 -> appending alfworld so dispatch falls to the GRPO path"
fi
DATASET_TYPE="{\"environment_names\": [$NAMES_JSON]}"
echo "    dataset-type: $DATASET_TYPE"

echo "=== 1/5 starting MCTS sidecar ==="
docker rm -f "$SIDECAR" >/dev/null 2>&1 || true
docker pull "$MCTS_IMAGE"
docker run -d --name "$SIDECAR" --network "$NET" "$MCTS_IMAGE"

# Discover the port the image actually exposes rather than assuming one.
PORT="$(docker inspect -f '{{range $p, $_ := .Config.ExposedPorts}}{{$p}}{{end}}' "$SIDECAR" \
        | head -1 | cut -d/ -f1)"
PORT="${PORT:-8000}"
ENV_URL="http://$SIDECAR:$PORT"
echo "    sidecar at $ENV_URL"

echo "    waiting for it to answer..."
for i in $(seq 1 60); do
    if docker run --rm --network "$NET" curlimages/curl:8.11.1 \
         -sf -o /dev/null "$ENV_URL/docs" 2>/dev/null \
       || docker run --rm --network "$NET" curlimages/curl:8.11.1 \
            -s -o /dev/null -w '%{http_code}' "$ENV_URL/" 2>/dev/null | grep -qv '^000$'; then
        echo "    up after ${i}s"; break
    fi
    [ "$i" = 60 ] && { echo "    SIDECAR NEVER ANSWERED -- see logs above"; exit 1; }
    sleep 1
done

echo "=== 2/5 building trainer image from THIS repo ==="
DOCKER_BUILDKIT=1 docker build -t "$TRAINER_IMAGE" \
    -f "$REPO_ROOT/dockerfiles/standalone-text-trainer.dockerfile" "$REPO_ROOT"

echo "=== 3/5 fetching base model into /cache ==="
# The trainer counts parameters from the weight files on disk to pick its size
# bucket, so the model must be in /cache/models BEFORE training -- otherwise it
# aborts with "Cannot determine model size: weight counting failed".  The
# official example does this with a separate trainer-downloader image; the
# trainer image already has huggingface_hub, so we use it directly.
docker run --rm --network "$NET" \
    --volume "$CHECKPOINTS_DIR:/cache:rw" \
    -e HF_TOKEN="$HF_TOKEN" -e HUGGINGFACE_TOKEN="$HF_TOKEN" \
    -e HF_HUB_ENABLE_HF_TRANSFER=0 \
    --entrypoint bash "$TRAINER_IMAGE" \
    -lc "source /workspace/.grpo_env/bin/activate \
         && cd /workspace/scripts \
         && python -m tools.fetch_model '$MODEL'"

echo "=== 4/5 preflight inside the trainer image ==="
# The image installs trl/vllm/etc into a venv, not the system python -- see
# run_text_trainer.sh, which sources it before doing anything.  Without this
# activation preflight runs against the base interpreter and reports every
# package missing.  --gpus all so the CUDA check is meaningful.
docker run --rm --gpus all --network "$NET" \
    -e ENVIRONMENT_SERVER_URLS="$ENV_URL" \
    -e GEN_CHUNK="$GEN_CHUNK" -e STEP_WORKERS="$STEP_WORKERS" \
    --entrypoint bash "$TRAINER_IMAGE" \
    -lc "source /workspace/.grpo_env/bin/activate \
         && cd /workspace/scripts \
         && python -m tools.preflight --probe-env --game '$GAMES'"

echo "=== 5/5 training ==="
docker run --rm --gpus all --network "$NET" \
    --security-opt=no-new-privileges --cap-drop=ALL \
    --memory=64g --cpus=8 \
    --volume "$CHECKPOINTS_DIR:/cache:rw" \
    --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
    -e ENVIRONMENT_SERVER_URLS="$ENV_URL" \
    -e GEN_CHUNK="$GEN_CHUNK" -e STEP_WORKERS="$STEP_WORKERS" \
    -e HUGGINGFACE_TOKEN="$HF_TOKEN" -e HUGGINGFACE_USERNAME="$HF_USER" \
    -e WANDB_TOKEN="$WANDB_TOKEN" \
    --name "env-task-$PRIMARY_GAME" \
    "$TRAINER_IMAGE" \
    --task-id "$TASK_ID" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --dataset-type "$DATASET_TYPE" \
    --task-type "EnvTask" \
    --file-format "$FILE_FORMAT" \
    --hours-to-complete "$HOURS" \
    --expected-repo-name "$REPO_NAME"
