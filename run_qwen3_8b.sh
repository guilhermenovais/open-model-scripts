#!/usr/bin/env bash
set -euo pipefail

# Serves Qwen3-8B (AWQ 4-bit quantized) with vLLM, sized for a large-context
# agent workload on this machine (RTX 5070, 12GB VRAM, WSL2).
#
# --max-model-len 40960 is Qwen3-8B's native context limit (no RoPE scaling
# needed to reach it). To fit that much KV cache in 12GB VRAM alongside the
# ~5.5GB AWQ weights:
#   - --kv-cache-dtype fp8 halves KV cache memory per token vs fp16.
#   - --attention-backend TRITON_ATTN is required with fp8 KV cache here:
#     FlashAttention doesn't support fp8 KV, so vLLM would otherwise pick
#     FlashInfer, which JIT-compiles with nvcc — not installed on this
#     machine (no full CUDA toolkit, only the driver). Triton's kernels
#     compile fine with the gcc/libpython3.14-dev already installed.
#   - --max-num-batched-tokens 4096 caps the chunked-prefill activation
#     buffer. Without it, vLLM sizes that buffer close to max-model-len,
#     which was eating ~1GB that should go to KV cache instead (measured:
#     1.89GiB available KV cache without this cap vs 2.99GiB with it,
#     the difference between NOT fitting 40960 tokens and fitting them
#     with a bit of headroom: 43,552 KV tokens for a 40,960-token request).
#
# Other flags:
# - VLLM_WSL2_ENABLE_PIN_MEMORY=1 works around vLLM's default of disabling
#   pinned memory under WSL2, which otherwise makes UVA buffer allocation
#   fail at startup ("RuntimeError: UVA is not available").
# - VLLM_USE_FLASHINFER_SAMPLER=0 avoids flashinfer JIT-compiling a sampling
#   kernel with nvcc (same missing-toolchain issue as above).
# - --reasoning-parser qwen3 splits the model's <think> trace into a
#   separate `reasoning` field in the API response instead of inlining it.
# - --enable-auto-tool-choice + --tool-call-parser hermes are required for
#   tool_choice: "auto" (and function calling in general). Qwen3's chat
#   template emits tool calls as <tool_call>{"name":...,"arguments":...}
#   </tool_call>, which is the Hermes format, not the qwen3_xml/qwen3_coder
#   one (that's for the separate Qwen3-Coder models).
#
# If you see "Free memory on device ... is less than desired GPU memory
# utilization", something else is using VRAM (e.g. a desktop compositor) —
# lower --gpu-memory-utilization slightly, or close the other GPU consumer.

cd "$(dirname "${BASH_SOURCE[0]}")"

VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
./venv/bin/vllm serve Qwen/Qwen3-8B-AWQ \
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
