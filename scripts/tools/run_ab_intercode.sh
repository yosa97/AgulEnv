#!/usr/bin/env bash
#
# A/B: does raising the intercode diversity caps move the validator score?
#
#   bash scripts/tools/run_ab_intercode.sh
#
# What it does, in one command:
#   1. trains a model with the NEW caps (INTERCODE_MAX_PER_FS=2500 -> ~22k
#      unique examples instead of 5,276 upsampled 3.8x),
#   2. scores it on the exact same 140-task NL2Bash set with the exact same
#      local reimplementation of the validator's intercode metric,
#   3. prints the delta against the baselines already measured on this box.
#
# Established baselines (112 scored tasks, fs1+fs2+fs4, identical set):
#     0.6921  noop      -- floor: submit nothing, every turn
#     0.7163  turnamen  -- the model actually submitted (old caps)
#     0.9903  gold      -- ceiling: run the reference command
#
# ARM=both also trains an OLD-caps arm first, so both numbers come from the
# same base model, same hours, same box. That is the honest comparison and it
# costs 2x the GPU time. ARM=b (default) trains only the new-caps arm and
# compares to 0.7163 -- which came from a different task list and a different
# time budget, so a small move there is NOT evidence on its own.
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
GAMES="${GAMES:-intercode}"
HOURS="${HOURS:-4.0}"
ARM="${ARM:-b}"
NUM_SEEDS="${NUM_SEEDS:-200}"
HF_TOKEN="${HF_TOKEN:-}"
HF_USER="${HF_USER:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO_ROOT/eval_out"
mkdir -p "$OUT"

echo "================ A/B INTERCODE ================"
echo "  base model : $BASE_MODEL"
echo "  games      : $GAMES"
echo "  hours/arm  : $HOURS"
echo "  arm        : $ARM   (b = cap baru saja | both = cap lama + cap baru)"
echo "==============================================="
if [ -z "$HF_TOKEN" ]; then
    echo "  HF_TOKEN kosong. export HF_TOKEN=... dulu, lalu jalankan lagi."; exit 1
fi

train_arm() {  # train_arm <tag> <task_id> <per_fs> <max_per_fs>
    local tag="$1" tid="$2" per="$3" maxper="$4"
    local repo="ab-intercode-$tag"
    echo
    echo "########## LATIH ARM '$tag'  (PER_FS=$per MAX_PER_FS=$maxper) ##########"
    if [ -d "$REPO_ROOT/outputs/$tid/$repo" ] && [ "$(ls -A "$REPO_ROOT/outputs/$tid/$repo" | wc -l)" -gt 2 ]; then
        echo "    sudah ada di outputs/$tid/$repo -- lewati latihan"
        return 0
    fi
    INTERCODE_PER_FS="$per" INTERCODE_MAX_PER_FS="$maxper" \
    GAMES="$GAMES" MODEL="$BASE_MODEL" HOURS="$HOURS" \
    TASK_ID="$tid" REPO_NAME="$repo" \
    HF_TOKEN="$HF_TOKEN" HF_USER="$HF_USER" \
    bash "$REPO_ROOT/scripts/tools/run_env_task_local.sh"
}

score_arm() {  # score_arm <tag> <task_id> <skip_selftest>
    local tag="$1" tid="$2" skip="$3"
    local repo="ab-intercode-$tag"
    local path="$REPO_ROOT/outputs/$tid/$repo"
    echo
    echo "########## SKOR ARM '$tag' ##########"
    if [ ! -d "$path" ]; then
        echo "    TIDAK ADA $path -- latihan arm ini gagal, lewati."; return 1
    fi
    MODEL="$path" BASE_MODEL="$BASE_MODEL" HF_TOKEN="$HF_TOKEN" \
    NUM_SEEDS="$NUM_SEEDS" OUT_JSON="ab_$tag.json" LABEL="$tag" \
    SKIP_SELFTEST="$skip" \
    bash "$REPO_ROOT/scripts/tools/run_intercode_eval.sh"
}

SKIP=0
if [ "$ARM" = "both" ]; then
    train_arm caplama 9001 4000 600
    score_arm caplama 9001 "$SKIP" || true
    SKIP=1          # gold gate sudah lewat di run pertama
fi
train_arm capbaru 9002 30000 2500
score_arm capbaru 9002 "$SKIP" || true

echo
echo "================ RINGKASAN ================"
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
rows = [("noop (lantai)", 0.6921, None), ("turnamen (cap lama)", 0.7163, None)]
for tag, label in (("caplama", "arm cap lama"), ("capbaru", "arm cap baru")):
    p = os.path.join(out, f"ab_{tag}.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    t = d.get("tasks", [])
    if not t:
        continue
    mean = sum(r["reward"] for r in t) / len(t)
    p3 = sum(r.get("similarity", 0.0) for r in t) / len(t)
    perfect = sum(1 for r in t if r["reward"] > 0.99)
    rows.append((f"{label}  (n={len(t)})", mean, (p3, perfect, len(t))))
rows.append(("gold (langit-langit)", 0.9903, None))

print(f"  {'policy':<28} {'reward':>8} {'vs lantai':>10}  detail")
for name, v, extra in rows:
    d = f"{v-0.6921:+.4f}"
    det = ""
    if extra:
        p3, perfect, n = extra
        det = f"sim {p3:.3f}   sempurna {perfect}/{n}"
    print(f"  {name:<28} {v:>8.4f} {d:>10}  {det}")

have = {t: json.load(open(os.path.join(out, f"ab_{t}.json")))
        for t in ("caplama", "capbaru") if os.path.exists(os.path.join(out, f"ab_{t}.json"))}
if "capbaru" in have:
    nb = have["capbaru"]["tasks"]
    mb = sum(r["reward"] for r in nb) / len(nb)
    if "caplama" in have:
        na = have["caplama"]["tasks"]
        ma = sum(r["reward"] for r in na) / len(na)
        print(f"\n  DELTA cap baru - cap lama : {mb-ma:+.4f}  (perbandingan bersih)")
    else:
        print(f"\n  DELTA cap baru - turnamen : {mb-0.7163:+.4f}")
        print("  CATATAN: dua model ini beda task-list dan beda jam latih, jadi")
        print("           delta kecil di sini bukan bukti. ARM=both untuk bukti.")
PY
echo "==========================================="
echo "hasil mentah: $OUT/ab_*.json"
