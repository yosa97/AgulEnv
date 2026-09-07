#!/usr/bin/env bash
# One command: measure a model on the OTHELLO half of the task.
#
#   bash scripts/tools/run_othello_eval.sh
#
# Runs three policies against the same opponent (in-process OpenSpiel MCTS at
# the validator's eval strength) so the model's number has a floor and a
# ceiling around it:
#
#   random   the floor      -- uniform legal moves
#   teacher  the gate       -- the expert that LABELS the SFT data. If it
#                             cannot beat the opponent it was built to beat,
#                             the model cannot learn to either, and a model
#                             score measured here would be meaningless.
#   hf       the model
#
# Override: MODEL=<path or repo>  GAMES=40  GAME=clobber  SKIP_GATE=1
set -euo pipefail

MODEL="${MODEL:-}"
GAME="${GAME:-othello}"
GAMES="${GAMES:-30}"          # seeds; each played from BOTH seats
GATE_GAMES="${GATE_GAMES:-15}"
IMAGE="${IMAGE:-agulenv-trainer}"
HF_TOKEN="${HF_TOKEN:-}"
BASE_MODEL="${BASE_MODEL:-}"
SKIP_GATE="${SKIP_GATE:-0}"
LABEL="${LABEL:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO_ROOT/eval_out"
HF_CACHE="${HF_CACHE:-$REPO_ROOT/secure_checkpoints/hf_cache}"
mkdir -p "$OUT" "$HF_CACHE"; chmod 777 "$OUT" "$HF_CACHE"

MODEL_MOUNT=()
if [ -n "$MODEL" ] && [ -d "$MODEL" ]; then
    MODEL="$(cd "$MODEL" && pwd)"
    MODEL_MOUNT=(-v "$MODEL:$MODEL:ro")
fi
BASE_FLAG=()
if [ -n "$MODEL" ] && [ -d "$MODEL" ] && [ -f "$MODEL/adapter_config.json" ] && [ -n "$BASE_MODEL" ]; then
    BASE_FLAG=(--base-model "$BASE_MODEL")
fi

run() {   # run <python args...>
    docker run --rm --gpus all \
        -v "$OUT:/out" -v "$REPO_ROOT/scripts:/workspace/scripts:ro" \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -e HF_TOKEN="$HF_TOKEN" -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
        -e HF_HUB_ENABLE_HF_TRANSFER=0 -e TRL_EXPERIMENTAL_SILENCE=1 \
        "${MODEL_MOUNT[@]}" --entrypoint bash "$IMAGE" -lc \
        "source /workspace/.grpo_env/bin/activate && cd /workspace/scripts && python -m tools.eval_othello_local $*"
}

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "=== build image sekali ==="
    DOCKER_BUILDKIT=1 docker build -t "$IMAGE" \
        -f "$REPO_ROOT/dockerfiles/standalone-text-trainer.dockerfile" "$REPO_ROOT"
fi

echo "=== 1/3 lantai: random ==="
run --policy random --game "$GAME" --games "$GATE_GAMES" \
    --json "/out/othello_random.json" 2>&1 | tee "$OUT/othello_random.log"
FLOOR=$(grep -Eo 'SKOR \(w\+\.5d\) *: *[0-9.]+' "$OUT/othello_random.log" | grep -Eo '[0-9.]+$' || echo 0)

if [ "$SKIP_GATE" = "1" ]; then
    echo "=== 2/3 gerbang teacher DILEWATI (SKIP_GATE=1) ==="
else
    echo "=== 2/3 gerbang: teacher harus > 0.50 ==="
    run --policy teacher --game "$GAME" --games "$GATE_GAMES" \
        --json "/out/othello_teacher.json" 2>&1 | tee "$OUT/othello_teacher.log"
    T=$(grep -Eo 'SKOR \(w\+\.5d\) *: *[0-9.]+' "$OUT/othello_teacher.log" | grep -Eo '[0-9.]+$' || echo 0)
    echo "    teacher = $T   (random = $FLOOR)"
    if awk "BEGIN{exit !($T < 0.50)}"; then
        echo
        echo "    GERBANG GAGAL. Guru yang MELABELI data SFT tidak bisa menang"
        echo "    lawan MCTS@eval. Model dilatih meniru label yang kalah, jadi"
        echo "    skor model di atas ini tidak berarti apa-apa. Perbaiki guru"
        echo "    atau harness-nya dulu; kirim othello_teacher.log."
        exit 1
    fi
    echo "    OK."
fi

if [ -z "$MODEL" ]; then
    echo "=== 3/3 dilewati: MODEL tidak diset ==="
    exit 0
fi
echo "=== 3/3 model ==="
echo "    model: $MODEL${LABEL:+   [$LABEL]}"
run --policy hf --model "$MODEL" "${BASE_FLAG[@]}" --game "$GAME" --games "$GAMES" \
    --json "/out/othello_model.json" 2>&1 | tee "$OUT/othello_model.log"

echo
echo "patokan: random = $FLOOR | 0.500 = seimbang dengan lawan eval"
echo "hasil di: $OUT/othello_*.json"
