#!/usr/bin/env bash
set -euo pipefail

# Serves microsoft/Phi-4-mini-instruct (3.8B) with vLLM's online FP8 weight
# quantization (--quantization fp8). In bf16 the weights alone took 7.17GiB and left
# 0.31GiB of KV cache (~5k tokens), too little for the agent's prompts. Tool calls: the phi4_mini_json parser expects
# functools[{"name":...,"arguments":...}], which is what vLLM's
# examples/tool_chat_template_phi4_mini.jinja (v0.29.0, copied here) asks for;
# the HF template has no tool support.
#
# Shared flags (WSL2 env vars, fp8 KV cache + TRITON_ATTN, --max-num-batched-tokens
# 4096, 0.88 GPU memory) are explained in run_qwen3_8b.sh. Written for the
# model comparison in brazilian-open-data-agent specs/010-model-comparison.

cd "$(dirname "${BASH_SOURCE[0]}")"

VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
./venv/bin/vllm serve microsoft/Phi-4-mini-instruct \
  --served-model-name phi-4-mini \
  --quantization fp8 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser phi4_mini_json \
  --chat-template tool_chat_template_phi4_mini.jinja \
  --port 8000 \
  "$@"
