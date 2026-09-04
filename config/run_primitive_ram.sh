#!/usr/bin/env bash
# Serve primitive-ai/Qwen3.8-Flash-Next-NVFP4 on one RTX PRO 6000 Blackwell
# (WSL2 Ubuntu, native docker - not Docker Desktop).
#
# PLE (n-gram) table: sglang's NATIVE --ple-offload-embedding (Qwen4ExpPinnedHostEmbedding
# in sglang/srt/models/qwen4_exp.py) pins it in HOST RAM. No sglang_ssd_stream plugin,
# no ssd-stream.json, no NVMe streaming - this is the whole point of this profile vs.
# the rest of the repo (which streams the PLE table from SSD instead).
# The checkpoint's own config.json already sets text_config.ple_embedding_dtype=
# "float8_e4m3fn", so the ~95GiB BF16-on-disk table is stored at fp8 once pinned
# (~48GiB resident) - well inside a 100GB+ RAM budget.
#
# No --security-opt seccomp=unconfined needed here: that flag exists elsewhere in this
# repo only for the SSD Stream plugin's io_uring syscalls, which this profile doesn't use.
#
# IMAGE defaults to sglang-flash-ram:qsa-fp8-fix: the official day-1 image
# (lmsysorg/sglang:qwen38flashnext, commit 593134d17) plus a one-file backport of
# sgl-project/sglang@d6ff2d881e78d0746e0393e9860ce3de5d84de8b (docker-target/Dockerfile.qsa-fp8-fix).
# Without it, --kv-cache-dtype fp8_e4m3 crashes ("Unsupported rhs dtype fp8e4nv") the
# first time QSA's forward_extend() hits a cached prefix under fp8 KV (2nd+ chunked-
# prefill chunk, or any multi-turn/radix-cache-hit request) - see docs/09-PRIMITIVE-RAM-OFFLOAD.md.
# Build: cd docker-target && docker build -t sglang-flash-ram:qsa-fp8-fix -f Dockerfile.qsa-fp8-fix .
set -euo pipefail

IMAGE="${IMAGE:-sglang-flash-ram:qsa-fp8-fix}"
NAME="${NAME:-sglang-flash-ram}"
PORT="${PORT:-8888}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-NVFP4}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

[ -d "$MODEL_DIR" ] || { echo "MODEL_DIR not found: $MODEL_DIR" >&2; exit 1; }
[ -d "$MODEL_DIR/mtp" ] || echo ">> warning: $MODEL_DIR/mtp missing - run scripts/primitive/mtp_synth.py first (NEXTN speculative decoding needs it)"

CTX="${CTX:-262144}"
MAXREQ="${MAXREQ:-4}"
FRACTION="${FRACTION:-0.96}"
CHUNKED="${CHUNKED:-2048}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
SPEC="${SPEC:-on}"
DRAFT8="${DRAFT8:-on}"
# The checkpoint's own config.json sets text_config.ple_embedding_dtype=
# "float8_e4m3fn", which silently downcasts the PLE/n-gram table to an
# uncalibrated fp8 (weight_scale hardcoded to 1.0) even though this checkpoint
# ships real BF16 PLE shards and the model card documents ~100GB BF16 host RAM
# residency for exactly this checkpoint. PLE_DTYPE_OVERRIDE=1 (default) undoes
# that via --json-model-override-args, restoring the BF16 table the checkpoint
# author intended (~95GB resident instead of ~48GB). Historical note:
# combining this override with --ple-offload-embedding used to OOM at boot
# ("Not enough GPU memory for hybrid (mamba/linear-attention) state cache")
# because of a transient ~95GB GPU allocation during model construction --
# fixed in docker-target/patches-official/qwen4_exp.py (see
# docs/09-PRIMITIVE-RAM-OFFLOAD.md, "PLE offload + BF16 table" section); IMAGE
# below already includes that fix. Set PLE_DTYPE_OVERRIDE=0 to keep the
# smaller uncalibrated fp8 downcast instead, if RAM budget is tighter than
# ~100GB.
PLE_DTYPE_OVERRIDE="${PLE_DTYPE_OVERRIDE:-1}"
OVERRIDE_FLAGS=()
if [ "$PLE_DTYPE_OVERRIDE" = "1" ]; then
  OVERRIDE_FLAGS=(--json-model-override-args '{"text_config": {"ple_embedding_dtype": null}}')
fi

SPEC_FLAGS=()
if [ "$SPEC" = "on" ]; then
  SPEC_FLAGS=(--speculative-algorithm NEXTN
    --speculative-draft-model-path /model/mtp
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
  if [ "$DRAFT8" = "on" ]; then
    SPEC_FLAGS+=(--speculative-draft-model-quantization fp8)
  fi
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" --gpus all \
  -p "${PORT}:8000" \
  --ipc=host \
  -v "$MODEL_DIR:/model:ro" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path /model \
    --served-model-name current \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --quantization modelopt_fp4 \
    --kv-cache-dtype "$KV_DTYPE" \
    --page-size 64 \
    --mamba-radix-cache-strategy extra_buffer_lazy --mamba-track-interval 64 \
    --chunked-prefill-size "$CHUNKED" \
    --max-running-requests "$MAXREQ" \
    --context-length "$CTX" \
    --mem-fraction-static "$FRACTION" \
    --ple-offload-embedding \
    "${OVERRIDE_FLAGS[@]}" \
    "${SPEC_FLAGS[@]}" \
    --allow-auto-truncate \
    --reasoning-parser auto --tool-call-parser auto \
    --enable-metrics

echo "waiting for the server (PLE table load into host RAM adds to cold start)..."
for _ in $(seq 1 90); do
  if docker logs "$NAME" 2>&1 | grep -q "The server is fired up"; then
    echo "ready on http://127.0.0.1:${PORT}"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container exited:"; docker logs --tail 60 "$NAME"; exit 1
  fi
  sleep 10
done
echo "timed out; last log lines:"; docker logs --tail 60 "$NAME"; exit 1
