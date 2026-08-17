FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
COPY --from=ghcr.io/astral-sh/uv:0.9.14 /uv /uvx /bin/

# Round-12: Set CUDA_HOME explicitly + skip deepspeed CUDA op precompilation.
# Without these, SFTTrainer init crashes at `from deepspeed import` because
# deepspeed/ops/op_builder/builder.py raises MissingCUDAException when
# CUDA_HOME is unset (some deepspeed versions are strict about this even
# when CUDA libs are reachable via LD_LIBRARY_PATH). DS_BUILD_OPS=0 tells
# deepspeed to skip the at-import-time CUDA op compatibility check;
# CUDA_HOME points to the standard runpod/pytorch base location.
ENV CUDA_HOME=/usr/local/cuda
ENV DS_BUILD_OPS=0

# Round-20: HTTPS clients (wandb-core Go binary, Python ssl/requests, curl) need
# a valid CA bundle to verify TLS certs. Destructive gold commands during
# intercode REAL_EXEC can wipe /etc/ssl/certs/ before training starts; then
# wandb.init() crashes with "x509: certificate signed by unknown authority" and
# kills the training subprocess at on_train_begin. Defense in depth:
#   1) Bake a copy of the system CA bundle to /.intercode_backup/etc_ssl_certs/
#      (hidden top-dir, survives `find * ...` style wipes per R3 rationale).
#   2) Point all standard HTTPS clients at the backup path via env vars, so they
#      bypass /etc/ssl entirely. wandb-core (Go) reads SSL_CERT_FILE; Python
#      requests reads REQUESTS_CA_BUNDLE; curl reads CURL_CA_BUNDLE.
ENV SSL_CERT_FILE=/.intercode_backup/etc_ssl_certs/ca-certificates.crt
ENV SSL_CERT_DIR=/.intercode_backup/etc_ssl_certs
ENV REQUESTS_CA_BUNDLE=/.intercode_backup/etc_ssl_certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/.intercode_backup/etc_ssl_certs/ca-certificates.crt

# Round-13: nvcc shim. runpod/pytorch base has CUDA RUNTIME (libcudart etc.)
# but NO toolkit (no nvcc compiler). deepspeed's installed_cuda_version()
# tries `subprocess.check_output(['/usr/local/cuda/bin/nvcc', '-V'])` at
# module-import time -> FileNotFoundError. The function only parses the
# version string from nvcc output, never compiles anything. Provide a
# tiny shim that emits realistic CUDA 12.8 version banner. DS_BUILD_OPS=0
# already skips actual op compilation, so we only need to fake the version
# probe.
RUN mkdir -p /usr/local/cuda/bin && \
    printf '%s\n' \
        '#!/bin/bash' \
        'echo "nvcc: NVIDIA (R) Cuda compiler driver"' \
        'echo "Copyright (c) 2005-2024 NVIDIA Corporation"' \
        'echo "Built on Tue_Mar_19_15:00:00_PDT_2024"' \
        'echo "Cuda compilation tools, release 12.8, V12.8.0"' \
        'echo "Build cuda_12.8.r12.8/compiler.34058467_0"' \
        > /usr/local/cuda/bin/nvcc && \
    chmod +x /usr/local/cuda/bin/nvcc && \
    /usr/local/cuda/bin/nvcc -V

# Round-20: CA bundle backup MUST happen BEFORE any RUN that uses HTTPS
# (apt-get, pip install, git clone). The SSL_CERT_FILE / REQUESTS_CA_BUNDLE
# env vars set above point HTTPS clients at /.intercode_backup/etc_ssl_certs/
# so that path needs to exist before clients try to read it. Base image
# already ships /etc/ssl/certs/ca-certificates.crt, so we just snapshot it.
RUN mkdir -p /.intercode_backup/etc_ssl_certs && \
    cp -L /etc/ssl/certs/ca-certificates.crt /.intercode_backup/etc_ssl_certs/ca-certificates.crt && \
    ls -lh /.intercode_backup/etc_ssl_certs/

# System dependencies. `tar` is needed by intercode_local_bash_env.LocalBashEnv
# to restore fs1/fs2 snapshots at gen time (runpod/pytorch base does NOT ship
# tar by default — verified via "[Errno 2] No such file or directory: 'tar'"
# from subprocess.run(["tar", ...]) at runtime).
RUN apt-get update && apt-get install -y \
    vim \
    zip \
    tmux \
    iotop \
    nvtop \
    bmon \
    wget \
    nano \
    zsh \
    htop \
    redis-server \
    tar \
    python-is-python3 \
    jq \
    psmisc \
    bsdmainutils \
    file \
    dnsutils \
    tree \
    net-tools \
    iputils-ping \
    cpio \
    acl \
    attr \
    imagemagick \
    libnuma1 \
    && rm -rf /var/lib/apt/lists/*
# Default dir
RUN mkdir -p /workspace
RUN mkdir -p /cache
RUN mkdir -p /workspace/scripts/datasets
RUN mkdir -p /app/checkpoints
WORKDIR /workspace/scripts

# # Setup AlfWorld server env
# COPY scripts/alfworld_setup.sh /workspace/scripts/alfworld_setup.sh
# COPY scripts/alfworld_run.sh /workspace/scripts/alfworld_run.sh
# RUN chmod +x /workspace/scripts/alfworld_setup.sh
# RUN /workspace/scripts/alfworld_setup.sh

# Install main dependencies
COPY scripts/grpo_requirements.txt /workspace/scripts/grpo_requirements.txt
RUN python -m venv /workspace/.grpo_env
RUN bash -c "source /workspace/.grpo_env/bin/activate && \
    pip install uv && \
    uv pip install --no-build-isolation -r /workspace/scripts/grpo_requirements.txt && \
    uv pip install --no-build-isolation flash-attn==2.8.3 && \
    git clone --depth 1 https://github.com/WooooDyy/AgentGym && \
    uv pip install --no-build-isolation AgentGym/agentenv && \
    deactivate"

# OpenSpiel (pyspiel) — powers local in-process teacher-vs-teacher self-play data
# generation (scripts/our_envs/pvp_selfplay.py), now the DEFAULT SFT path for all 6
# PvP game envs. Both seats are teacher-quality, so it yields ~2-4x rows/game at
# far higher throughput than the env-server+MCTS HTTP loop, with prompts built
# from the same local pyspiel engine the PvP eval scores on. Installed at build
# time (internet available) so it runs offline at tournament time. The install is
# best-effort + verified: a failure must NOT break the image, because
# sft_env_configs falls back to the env-server + MCTS generators at runtime when
# pyspiel is not importable (or when SELFPLAY_DISABLE=1). The build-time import
# check just makes the self-play status loud in the build log.
RUN bash -c "source /workspace/.grpo_env/bin/activate && \
    (uv pip install open_spiel || echo '[warn] open_spiel install failed') && \
    (python -c 'import pyspiel; print(\"[ok] pyspiel importable — self-play DEFAULT enabled\")' \
       || echo '[warn] pyspiel NOT importable — self-play disabled, runtime falls back to env-server+MCTS') && \
    deactivate"

# Backup critical binaries AND shared libraries — intercode REAL_EXEC sometimes
# runs gold commands that wipe /bin/bash + /lib/libc.so.6 + the dynamic linker
# etc. _heal_critical_bins() in LocalBashEnv.reset() restores from this backup
# so the trainer container survives (training phase needs /bin/sh +
# /usr/bin/python for accelerate/tokenize/merge subprocess launches).
#
# CRITICAL: backup MUST live at a HIDDEN top-level path. A gold like
# `find * -maxdepth 0 -exec rm -rf '{}' ';'` (perpetrator in commit 4c88cdf
# round-2 test) expands `*` to EVERY visible top-level entry and rm-rf's
# them all -- /opt, /bin, /usr, EVERYTHING. Hidden entries (starting with `.`)
# are NOT matched by `*` glob (bash default `dotglob` is off), so a backup
# under /.intercode_backup/ survives `find * ...` style attacks.
#
# Round-8: also backup shared libs. Heal-restored /bin/bash is unloadable
# without /lib64/ld-linux-x86-64.so.2 (dynamic linker) + /lib/.../libc.so.6.
# Linux returns ENOENT for "interpreter missing" during execve, so missing
# libc looks the same as missing binary.
RUN mkdir -p /.intercode_backup/lib /.intercode_backup/lib64 && \
    for b in /bin/bash /bin/tar /bin/sh /bin/cp /bin/rm /bin/mv /bin/ls \
             /bin/mkdir /bin/cat /bin/chmod /bin/ln /bin/touch /bin/grep \
             /usr/bin/find /usr/bin/python3 /usr/bin/python /usr/bin/tar \
             /usr/bin/sed /usr/bin/awk /usr/bin/sort /usr/bin/head \
             /usr/bin/tail /usr/bin/wc /usr/bin/cut /usr/bin/tr \
             /usr/bin/uniq /usr/bin/xargs; do \
        if [ -e "$b" ]; then cp -L "$b" /.intercode_backup/; fi; \
    done && \
    cp -L /lib64/ld-linux-x86-64.so.2 /.intercode_backup/lib64/ 2>/dev/null || true && \
    for lib in libc.so.6 libm.so.6 libpthread.so.0 libdl.so.2 librt.so.1 \
               libtinfo.so.6 libreadline.so.8 libgcc_s.so.1 libstdc++.so.6 \
               libresolv.so.2 libnss_dns.so.2 libnss_files.so.2 \
               libutil.so.1 libcrypt.so.1 libz.so.1; do \
        src="/lib/x86_64-linux-gnu/$lib"; \
        if [ -e "$src" ]; then cp -L "$src" /.intercode_backup/lib/; fi; \
    done && \
    if [ -e /usr/local/cuda/bin/nvcc ]; then \
        cp -L /usr/local/cuda/bin/nvcc /.intercode_backup/nvcc; \
    fi && \
    echo "=== /.intercode_backup/ root ===" && ls -lh /.intercode_backup/ | head -10 && \
    echo "=== /.intercode_backup/lib/ ===" && ls -lh /.intercode_backup/lib/ | head -10 && \
    echo "=== /.intercode_backup/lib64/ ===" && ls -lh /.intercode_backup/lib64/ && \
    echo "=== /.intercode_backup/etc_ssl_certs/ ===" && ls -lh /.intercode_backup/etc_ssl_certs/

# InterCode NL2Bash filesystem snapshots for SFT real-execution gen
# (scripts/our_envs/intercode_local_bash_env.py:LocalBashEnv). Bakes:
#   /intercode_fs/fs1.tar  (managed paths: /testbed)
#   /intercode_fs/fs2.tar  (managed paths: /system)
#   /intercode_fs/fs4.tar  (empty; filesystem-agnostic tasks)
# fs3 is deliberately SKIPPED — it manages /workspace + /backup, which
# conflict with this trainer container's working dir; restoring fs3 would
# `rm -rf /workspace` and destroy the running training. See
# dockerfiles/intercode_build_fs.sh header for the safety rationale.
# Real execution is gated by INTERCODE_REAL_EXEC=1 at run time (default off
# in scripts/our_envs/intercode_trajectories.py); these snapshots stay dormant
# until that flag is flipped. Image-build-time only; no internet at run.
RUN git clone --depth 1 https://github.com/princeton-nlp/intercode /opt/intercode
COPY dockerfiles/intercode_build_fs.sh /tmp/intercode_build_fs.sh
RUN chmod +x /tmp/intercode_build_fs.sh && \
    INTERCODE_REPO=/opt/intercode OUT_DIR=/intercode_fs /tmp/intercode_build_fs.sh && \
    rm -rf /opt/intercode /tmp/intercode_build_fs.sh

# Copy current folder to /workspace/auto_ml
COPY scripts /workspace/scripts
# # Make entrypoint script executable
# RUN chmod +x /workspace/scripts/alfworld_run.sh

RUN chmod +x /workspace/scripts/run_text_trainer.sh
# RUN chmod +x /workspace/scripts/entrypoint.sh

ENTRYPOINT ["./run_text_trainer.sh"]