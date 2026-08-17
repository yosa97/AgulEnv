#!/usr/bin/env bash
# Build the InterCode NL2Bash filesystem snapshots for SFT real-execution gen.
#
# Adapted from G.O.D/dockerfiles/intercode_build_fs.sh
# (validator/evaluation/eval_intercode.py uses the same snapshots at eval).
#
# ONE SAFETY DEVIATION from upstream: fs3 is SKIPPED. fs3 manages /workspace +
# /backup, and our trainer container's working dir IS /workspace/scripts/.
# Restoring fs3 at gen time would `rm -rf /workspace` and destroy the running
# training. fs3 rows fall back to synthetic-observation multi-turn in
# scripts/our_envs/intercode_trajectories.py (no fs touch -> safe).
#
# Inputs (set by the dockerfile):
#   INTERCODE_REPO   path to a checkout of princeton-nlp/intercode
# Outputs:
#   /intercode_fs/fs1.tar  (managed paths: /testbed)
#   /intercode_fs/fs2.tar  (managed paths: /system)
#   /intercode_fs/fs4.tar  (empty; filesystem-agnostic)
#   fs3.tar is intentionally not produced.

set -uo pipefail

INTERCODE_REPO="${INTERCODE_REPO:-/opt/intercode}"
OUT_DIR="${OUT_DIR:-/intercode_fs}"
mkdir -p "$OUT_DIR"

declare -A FS_PATHS=(
    [1]="/testbed"
    [2]="/system"
)
# fs3 deliberately omitted (see header).
# fs4 has no managed filesystem (tasks are filesystem-agnostic).

for fs in 1 2; do
    echo "=== building intercode fs_${fs} ==="
    script="$INTERCODE_REPO/docker/bash_scripts/setup_nl2b_fs_${fs}.sh"
    if [[ ! -f "$script" ]]; then
        echo "missing setup script: $script" >&2
        exit 1
    fi
    chmod +x "$script"
    # Upstream scripts don't use `set -e`; tolerate non-fatal exits as upstream does.
    bash "$script" || echo "warning: setup_nl2b_fs_${fs}.sh exited non-zero (matches upstream)"

    # Snapshot only this variant's managed paths.
    paths="${FS_PATHS[$fs]}"
    tar_args=()
    for p in $paths; do
        if [[ -e "$p" ]]; then
            tar_args+=("$(realpath --relative-to=/ "$p")")
        else
            echo "warning: expected path $p not created by fs_${fs} setup" >&2
        fi
    done
    if (( ${#tar_args[@]} == 0 )); then
        echo "no paths to snapshot for fs_${fs}, skipping" >&2
        continue
    fi
    (cd / && tar --acls --xattrs -cpf "$OUT_DIR/fs${fs}.tar" "${tar_args[@]}")
    echo "wrote $OUT_DIR/fs${fs}.tar"

    # Clean up before the next variant. We ONLY touch our own managed paths
    # (/testbed, /system) -- NEVER /workspace or /backup (those are the
    # trainer's). This is also why fs3 is skipped entirely.
    for p in /testbed /system; do
        rm -rf "$p"
    done
done

# fs4 = empty tar for symmetry (intercode_local_bash_env.LocalBashEnv won't try
# to extract when managed_paths is empty, but the file simplifies error checks).
tar -cf "$OUT_DIR/fs4.tar" -T /dev/null
echo "wrote $OUT_DIR/fs4.tar (empty, fs_4 is filesystem-agnostic)"
echo "=== done; fs3 SKIPPED on purpose (conflict with trainer /workspace) ==="
