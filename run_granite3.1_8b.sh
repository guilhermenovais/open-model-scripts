#!/usr/bin/env bash
set -euo pipefail

# Serves Granite 3.1 8B Instruct as RedHatAI's W4A16 (GPTQ-style, group 128,
# compressed-tensors) quant, since bf16 (~16GB) doesn't fit in 12GB. Granite 3.1's
# own chat template supports tools; the granite parser reads its
# <|tool_call|>[...] output.
#
# Shared flags (WSL2 env vars, fp8 KV cache + TRITON_ATTN, --max-num-batched-tokens
# 4096, 0.88 GPU memory) are explained in run_qwen3_8b.sh. Written for the
# model comparison in brazilian-open-data-agent specs/010-model-comparison.

cd "$(dirname "${BASH_SOURCE[0]}")"

VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
./venv/bin/vllm serve RedHatAI/granite-3.1-8b-instruct-quantized.w4a16 \
  --served-model-name granite-3.1-8b \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser granite \
  --port 8000 \
  "$@"
