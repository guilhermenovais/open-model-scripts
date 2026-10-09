#!/usr/bin/env bash
set -euo pipefail

# Serves Qwen3.5-4B (dense, native bf16 weights) with vLLM on this machine
# (RTX 5070, 12GB VRAM, WSL2).
#
# Unlike run_qwen3_8b.sh, this model does NOT need AWQ quantization, fp8 KV
# cache, a forced TRITON_ATTN backend, or a chunked-prefill batch cap:
#   - It's a dense 4.66B-param model (not the larger Qwen3.5 MoE variants),
#     so full bf16 weights are only ~8.7GiB, no quantized checkpoint needed.
#   - Its architecture is hybrid: 24 of its 32 layers are linear attention
#     (Gated DeltaNet, constant-size recurrent state) and only 8 are full
#     attention. KV cache only grows with context length on those 8 layers,
#     so it stays cheap even in plain bf16 (no need for fp8 KV to fit).
#
# --max-model-len 40960 is NOT this model's native limit (that's 262144 —
# see max_position_embeddings in its config). It's capped here because bf16
# weights (~8.7GiB) already take most of the 12GB card, leaving only ~1.5GiB
# for KV cache + activations at --gpu-memory-utilization 0.88 — enough for
# roughly 40-45k tokens of KV cache at bf16, not the full 262k context.
# Raise it (and/or --gpu-memory-utilization) if you free up VRAM elsewhere,
# or lower it if startup fails with a "not enough KV cache" style error.
#
# Other flags:
# - VLLM_WSL2_ENABLE_PIN_MEMORY=1 and VLLM_USE_FLASHINFER_SAMPLER=0 are the
#   same WSL2/no-CUDA-toolkit workarounds as in run_qwen3_8b.sh — unrelated
#   to model choice, needed for any vLLM serve on this machine.
# - --reasoning-parser qwen3 splits <think> traces into a separate
#   `reasoning` API field, same as Qwen3.
# - --enable-auto-tool-choice + --tool-call-parser qwen3_coder are what
#   vLLM's own Qwen3.5-4B deployment recipe uses for tool_choice: "auto"
#   support (Qwen3.5's tool-call format differs from Qwen3's Hermes-style
#   one, so the qwen3 script's --tool-call-parser hermes does NOT apply
#   here).
# - --trust-remote-code: Qwen3.5's hybrid linear/full-attention layers ship
#   custom modeling code in the repo.
#
# If you see "Free memory on device ... is less than desired GPU memory
# utilization", something else is using VRAM (e.g. a desktop compositor) —
# lower --gpu-memory-utilization slightly, or close the other GPU consumer.

cd "$(dirname "${BASH_SOURCE[0]}")"

VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
./venv/bin/vllm serve Qwen/Qwen3.5-4B \
  --trust-remote-code \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.88 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --port 8000 \
  "$@"
