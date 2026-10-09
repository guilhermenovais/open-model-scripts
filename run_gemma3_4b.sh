#!/usr/bin/env bash
set -euo pipefail

# Serves Gemma 3 4B IT with vLLM's online FP8 weight quantization (--quantization
# fp8; bf16 weights, ~8GB, leave no room for KV cache, as with Phi-4-mini) from
# unsloth/gemma-3-4b-it, an ungated mirror of
# google/gemma-3-4b-it (same weights; Google's repo needs a license click-through).
# Gemma 3 has no tool-call tokens: vLLM's examples/tool_chat_template_gemma3_pythonic.jinja
# (v0.29.0, copied here) describes the tools and asks for [func(arg=...)] calls,
# which the pythonic parser reads. --limit-mm-per-prompt image=0 skips the vision
# encoder's memory profiling (text-only workload).
#
# Shared flags (WSL2 env vars, fp8 KV cache + TRITON_ATTN, --max-num-batched-tokens
# 4096, 0.88 GPU memory) are explained in run_qwen3_8b.sh. Written for the
# model comparison in brazilian-open-data-agent specs/010-model-comparison.

cd "$(dirname "${BASH_SOURCE[0]}")"

VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
./venv/bin/vllm serve unsloth/gemma-3-4b-it \
  --served-model-name gemma-3-4b \
  --limit-mm-per-prompt '{"image": 0}' \
  --quantization fp8 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser pythonic \
  --chat-template tool_chat_template_gemma3_pythonic.jinja \
  --port 8000 \
  "$@"
