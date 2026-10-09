#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B-AWQ exactly as run_qwen3_8b.sh, but with thinking disabled for every
# request via --default-chat-template-kwargs (the chat template's enable_thinking
# switch), so it can be compared with non-reasoning models. Served under a
# different name so run files tell the two modes apart.
#
# Shared flags (WSL2 env vars, fp8 KV cache + TRITON_ATTN, --max-num-batched-tokens
# 4096, 0.88 GPU memory) are explained in run_qwen3_8b.sh. Written for the
# model comparison in brazilian-open-data-agent specs/010-model-comparison.

cd "$(dirname "${BASH_SOURCE[0]}")"

VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
./venv/bin/vllm serve Qwen/Qwen3-8B-AWQ \
  --served-model-name qwen3-8b-nothink \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --max-num-batched-tokens 4096 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 8000 \
  "$@"
