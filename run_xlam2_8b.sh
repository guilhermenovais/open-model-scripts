#!/usr/bin/env bash
set -euo pipefail

# Serves Salesforce/Llama-xLAM-2-8b-fc-r as a local W4A16 quant (int4 weights, group
# 128, symmetric, round-to-nearest; compressed-tensors) made by quantize_xlam_w4a16.py:
# bf16 doesn't fit in 12GB, vLLM 0.29 dropped bitsandbytes, and no GPTQ/AWQ checkpoint
# exists (the only int4 one is AMD Quark format). RTN without calibration, so slightly
# lossier than the GPTQ/AWQ quants of the other 8B models. The
# model's own chat template supports tools; vLLM's built-in xlam parser reads its
# JSON-list tool calls.
#
# Shared flags (WSL2 env vars, fp8 KV cache + TRITON_ATTN, --max-num-batched-tokens
# 4096, 0.88 GPU memory) are explained in run_qwen3_8b.sh. Written for the
# model comparison in brazilian-open-data-agent specs/010-model-comparison.

cd "$(dirname "${BASH_SOURCE[0]}")"

VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
./venv/bin/vllm serve models/Llama-xLAM-2-8b-fc-r-W4A16 \
  --served-model-name xlam-2-8b \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser xlam \
  --port 8000 \
  "$@"
