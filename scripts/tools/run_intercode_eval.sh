#!/usr/bin/env bash
# One command: measure this repo's model on the validator's intercode metric.
#
#   bash scripts/tools/run_intercode_eval.sh
#
# Clones the NL2Bash task set, builds the trainer image if needed, runs the
# gold self-test as a hard gate, then scores the model over all 200 tasks.
# Everything lands in ./eval_out.
#
# Override with env vars if needed:
#   MODEL=<hf repo or path>   HF_TOKEN=<token>   NUM_SEEDS=200   SKIP_BUILD=1
set -euo pipefail

MODEL="${MODEL:-gradients-io-tournaments/tournament-tourn_31f2e0fe36783f71_20260831-19ec162a-39a4-4c85-9f29-a858099773e3-5EFLCMFD}"
NUM_SEEDS="${NUM_SEEDS:-200}"
SELFTEST_SEEDS="${SELFTEST_SEEDS:-40}"
IMAGE="${IMAGE:-agulenv-trainer}"
INTERCODE_DIR="${INTERCODE_DIR:-/opt/intercode}"
HF_TOKEN="${HF_TOKEN:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO_ROOT/eval_out"
mkdir -p "$OUT"; chmod 777 "$OUT"

# The image sets HF_HUB_ENABLE_HF_TRANSFER=1 but does not ship hf_transfer, so
# any download aborts with a ValueError before it starts; turn it off.
# The cache is mounted from the host so a 3B model is fetched once, not once
# per run.
HF_CACHE="${HF_CACHE:-$REPO_ROOT/secure_checkpoints/hf_cache}"
mkdir -p "$HF_CACHE"; chmod 777 "$HF_CACHE"

TASKS="$INTERCODE_DIR/data/nl2bash/nl2bash_fs_1.json \
       $INTERCODE_DIR/data/nl2bash/nl2bash_fs_2.json \
       $INTERCODE_DIR/data/nl2bash/nl2bash_fs_4.json"

run_in_image() {  # run_in_image <extra docker args> -- <python args...>
    local dockerargs=() ; while [ "$1" != "--" ]; do dockerargs+=("$1"); shift; done; shift
    docker run --rm --gpus all \
        -v "$INTERCODE_DIR:$INTERCODE_DIR:ro" -v "$OUT:/out" \
        -v "$REPO_ROOT/scripts:/workspace/scripts:ro" \
        -e HF_TOKEN="$HF_TOKEN" -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
        -e HF_HUB_ENABLE_HF_TRANSFER=0 \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        "${dockerargs[@]}" --entrypoint bash "$IMAGE" -lc \
        "source /workspace/.grpo_env/bin/activate && cd /workspace/scripts && python -m tools.eval_intercode_local $*"
}

echo "=== 1/4 NL2Bash task set ==="
if [ -d "$INTERCODE_DIR/data/nl2bash" ]; then
    echo "    sudah ada di $INTERCODE_DIR"
else
    git clone --depth 1 -q https://github.com/princeton-nlp/intercode "$INTERCODE_DIR"
    echo "    cloned -> $INTERCODE_DIR"
fi

echo "=== 2/4 trainer image ==="
# The image bakes `COPY scripts /workspace/scripts` at build time, so a
# prebuilt image carries whatever the repo looked like THEN -- which is how a
# freshly added tool ends up as "No module named tools.eval_intercode_local".
# Mounting the working tree over that path instead means the container always
# runs current code and the image never needs rebuilding for a script change.
# Only the venv (/workspace/.grpo_env, a sibling path) comes from the image.
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "    image '$IMAGE' ada; scripts/ di-mount dari working tree (tanpa rebuild)"
else
    echo "    image '$IMAGE' belum ada -- build sekali (beberapa menit)"
    DOCKER_BUILDKIT=1 docker build -t "$IMAGE" \
        -f "$REPO_ROOT/dockerfiles/standalone-text-trainer.dockerfile" "$REPO_ROOT"
fi

echo "=== 3/4 swa-uji: policy gold harus ~1.0000 ==="
# Gold runs the reference command and submits. Anything below ~0.99 means the
# harness is wrong on THIS machine -- almost always fs_1/fs_2 snapshot restore
# -- and a model score measured on top of it would be meaningless. Hard gate.
run_in_image -- --policy gold --tasks $TASKS \
    --num-seeds "$SELFTEST_SEEDS" --json /out/selftest.json 2>&1 | tee "$OUT/selftest.log"

GOLD=$(grep -Eo 'MEAN REWARD *: *[0-9.]+' "$OUT/selftest.log" | grep -Eo '[0-9.]+$' || echo 0)
echo "    gold = $GOLD"
if awk "BEGIN{exit !($GOLD < 0.99)}"; then
    echo
    echo "    SWA-UJI GAGAL (gold=$GOLD, seharusnya ~1.0)."
    echo "    Harness tidak jujur di mesin ini -- kemungkinan besar restore"
    echo "    snapshot /intercode_fs/fs1.tar atau fs2.tar. Skor model tidak"
    echo "    akan berarti apa-apa di atas ini. Berhenti; kirim selftest.log."
    exit 1
fi
echo "    OK, harness jujur."

echo "=== 4/4 skor model ==="
echo "    model: $MODEL"
run_in_image -- --policy hf --model "$MODEL" --tasks $TASKS --print-prompt \
    2>&1 | tee "$OUT/prompt.log" | tail -40

run_in_image -- --policy hf --model "$MODEL" --tasks $TASKS \
    --num-seeds "$NUM_SEEDS" --json /out/hasil_intercode.json 2>&1 | tee "$OUT/eval.log"

echo
echo "=== lima task terburuk ==="
python3 - "$OUT/hasil_intercode.json" <<'PY' 2>/dev/null || echo "    (lewati: butuh python3 di host)"
import json, sys
d = json.load(open(sys.argv[1]))
rows = sorted(d.get("tasks", []), key=lambda r: r["reward"])[:5]
for r in rows:
    print(f"\n  reward {r['reward']:.4f}  sim {r['similarity']:.3f}")
    print(f"    query     : {r['query'][:120]}")
    print(f"    gold      : {r['gold'][:120]}")
    print(f"    gold_obs  : {r.get('gold_obs','')[:120]!r}")
    print(f"    agent_obs : {r.get('agent_obs','')[:120]!r}")
PY

echo
echo "patokan: 0.719 = tidak melakukan apa-apa | 0.741 = skor turnamen Anda"
echo "         0.760 = pemenang grup           | 1.000 = sempurna"
echo "hasil di: $OUT/"
