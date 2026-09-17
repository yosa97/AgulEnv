#!/usr/bin/env bash
# Replay a tournament group locally, with the real opponents.
#
#   bash scripts/tools/run_replay_group.sh
#
# Every model in round 1 group 3 of tourn_4f605297e6d85428_20260907 is public
# on HuggingFace, so the group can be played again here: intercode scored per
# model, othello played head to head between each pair (which is what the
# validator actually does -- model vs model, not model vs MCTS), then the
# confirmed ranking formula applied.
#
# The result is checked against the numbers the validator published. That
# check is the whole point: a replay that cannot reproduce a known outcome
# has no business being trusted on an unknown one.
#
# Then, to ask "what would my NEW model have scored in that group":
#   SWAP_ME=/root/AgulEnv/outputs/9002/ab-intercode-capbaru \
#   bash scripts/tools/run_replay_group.sh
#
# Budget an hour or two on one H100. Everything is resumable: a stage whose
# JSON already exists is skipped, so an interrupted run continues where it
# stopped.
set -euo pipefail

PREFIX="gradients-io-tournaments/tournament-tourn_4f605297e6d85428_20260907-ce540755-5a5d-4877-b8d4-35ddb2829843"
TAGS=(5CcwdezW 5E6p1x3e 5EeEWvKf 5GU4Xkd3)
ME="${ME:-5CcwdezW}"
SWAP_ME="${SWAP_ME:-}"          # local checkpoint to stand in for ME
GAMES="${GAMES:-8}"             # seeds per pair; x2 seats = 16 games, as the validator played
IC_SEEDS="${IC_SEEDS:-200}"
IMAGE="${IMAGE:-agulenv-trainer}"
HF_TOKEN="${HF_TOKEN:-}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
INTERCODE_DIR="${INTERCODE_DIR:-/opt/intercode}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO_ROOT/eval_out/replay"
HF_CACHE="${HF_CACHE:-$REPO_ROOT/secure_checkpoints/hf_cache}"
mkdir -p "$OUT" "$HF_CACHE"; chmod -R 777 "$REPO_ROOT/eval_out" "$HF_CACHE" 2>/dev/null || true

model_for() {
    if [ "$1" = "$ME" ] && [ -n "$SWAP_ME" ]; then echo "$SWAP_ME"; else echo "$PREFIX-$1"; fi
}

MOUNTS=()
if [ -n "$SWAP_ME" ] && [ -d "$SWAP_ME" ]; then
    SWAP_ME="$(cd "$SWAP_ME" && pwd)"
    MOUNTS=(-v "$SWAP_ME:$SWAP_ME:ro")
    echo "!! $ME diganti dengan checkpoint lokal: $SWAP_ME"
    echo "   Ini SKENARIO ANDAI, bukan hasil turnamen."
fi

run() {   # run <module> <args...>
    local mod="$1"; shift
    docker run --rm --gpus all \
        -v "$OUT:/out" -v "$REPO_ROOT/scripts:/workspace/scripts:ro" \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -v "$INTERCODE_DIR:$INTERCODE_DIR:ro" \
        -e HF_TOKEN="$HF_TOKEN" -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
        -e HF_HUB_ENABLE_HF_TRANSFER=0 -e TRL_EXPERIMENTAL_SILENCE=1 \
        "${MOUNTS[@]}" --entrypoint bash "$IMAGE" -lc \
        "source /workspace/.grpo_env/bin/activate && cd /workspace/scripts && python -m tools.$mod $*"
}

if [ ! -d "$INTERCODE_DIR/data/nl2bash" ]; then
    git clone --depth 1 -q https://github.com/princeton-nlp/intercode "$INTERCODE_DIR"
fi
TASKS="$INTERCODE_DIR/data/nl2bash/nl2bash_fs_1.json \
       $INTERCODE_DIR/data/nl2bash/nl2bash_fs_2.json \
       $INTERCODE_DIR/data/nl2bash/nl2bash_fs_4.json"

echo "############ 1/3  intercode per model ############"
for t in "${TAGS[@]}"; do
    f="$OUT/intercode_$t.json"
    if [ -s "$f" ]; then echo "  $t sudah ada, lewati"; continue; fi
    echo "  --- $t ---"
    run eval_intercode_local --policy hf --model "$(model_for "$t")" \
        --base-model "$BASE_MODEL" --tasks $TASKS --num-seeds "$IC_SEEDS" \
        --json "/out/intercode_$t.json" 2>&1 | tail -18
done

echo "############ 2/3  othello head-to-head ############"
for ((i=0; i<${#TAGS[@]}; i++)); do
  for ((j=i+1; j<${#TAGS[@]}; j++)); do
    a="${TAGS[$i]}"; b="${TAGS[$j]}"
    f="$OUT/othello_${a}__${b}.json"
    if [ -s "$f" ]; then echo "  $a vs $b sudah ada, lewati"; continue; fi
    echo "  --- $a vs $b ---"
    run eval_othello_local --game othello \
        --policy "$(model_for "$a")" --opponent "$(model_for "$b")" \
        --base-model "$BASE_MODEL" --games "$GAMES" \
        --json "/out/othello_${a}__${b}.json" 2>&1 | tail -16
  done
done

echo "############ 3/3  hitung ulang skor grup ############"
python3 "$REPO_ROOT/scripts/tools/replay_group.py" --dir "$OUT" --me "$ME" --expect group3
