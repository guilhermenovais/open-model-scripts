#!/usr/bin/env bash
set -euo pipefail

# Serves Llama-3.1-8B-Instruct (AWQ 4-bit quantized) with vLLM, sized for a
# large-context agent workload on this machine (RTX 5070, 12GB VRAM, WSL2).
#
# Why Llama 3.1 rather than Llama 3: Llama 3.0 has no tool-call format and
# only an 8192-token context, so agent clients sending `tools` /
# tool_choice: "auto" or large prompts got HTTP 400s. Llama 3.1 is the same
# size and architecture but supports tool calling and 131072 tokens of
# context (via its llama3 RoPE scaling, already in config.json).
#
# Why AWQ: full bf16 weights for an 8B model are ~16GB, which doesn't fit on
# the 12GB card. hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 is a
# 4-bit AWQ (GEMM) quant, ~5.7GB of weights, and unlike Meta's own repo it is
# NOT gated (no HF token / license click-through needed).
#
# --max-model-len 40960 matches run_qwen3_8b.sh, NOT this model's native
# limit (131072): the full 128k context would need ~8GiB of KV cache even in
# fp8, far more than is left after the weights. Fitting 40960 uses the same
# three memory tricks as run_qwen3_8b.sh (see its comments for details):
#   - --kv-cache-dtype fp8 halves KV cache per token vs bf16 (~64KiB/token
#     here: 32 layers x 8 KV heads x 128 dims x K+V).
#   - --attention-backend TRITON_ATTN, because FlashAttention doesn't support
#     fp8 KV and the FlashInfer fallback needs nvcc, which isn't installed.
#   - --max-num-batched-tokens 4096 caps the chunked-prefill activation
#     buffer so that memory goes to KV cache instead.
#
# Other flags:
# - VLLM_WSL2_ENABLE_PIN_MEMORY=1 and VLLM_USE_FLASHINFER_SAMPLER=0 are the
#   same WSL2/no-CUDA-toolkit workarounds as in run_qwen3_8b.sh — unrelated
#   to model choice, needed for any vLLM serve on this machine.
# - --enable-auto-tool-choice + --tool-call-parser llama3_json are required
#   for tool_choice: "auto". Llama 3.1's chat template emits tool calls as a
#   bare JSON object {"name": ..., "parameters": ...}, which is what the
#   llama3_json parser expects.
# - --chat-template tool_chat_template_llama3.1_json.jinja (vLLM's
#   examples/tool_chat_template_llama3.1_json.jinja, v0.29.0, kept next to
#   this script) replaces Meta's built-in chat template. With Meta's, the
#   model called a tool for EVERY prompt (even "say hello") and never used
#   tool results in its answer (0/5 on both in testing). vLLM's adds a
#   default system prompt telling it to answer directly when no tool fits
#   and to use tool output, and wraps tool results as {"output": ...}.
#   The local copy has one change: tools go in the system prompt instead of
#   the first user message (see the comment inside the template for why).
#
#   Known limitation: that default system prompt is replaced when a client
#   sends its own. In testing, with a custom system prompt the model still
#   called tools correctly (5/5) and used their results (5/5), but it also
#   called made-up tools (e.g. "eval" for "What is 2+2?") instead of
#   answering directly (0/10). Adding that rule to the tool instructions
#   didn't fix it (3/10) and made the no-system-prompt case worse, so it
#   was left out. If your client sends a system prompt, include something
#   like "Only call the listed functions; otherwise answer directly", and
#   ignore tool calls whose name isn't in your tool list.
#   Limitation: one tool call per turn (no parallel tool calls).
# - --served-model-name llama-3.1-8b: clients must send this as `model`.
# - No --reasoning-parser: Llama 3.1 has no <think> reasoning traces.
#
# If you see "Free memory on device ... is less than desired GPU memory
# utilization", something else is using VRAM (e.g. a desktop compositor) —
# lower --gpu-memory-utilization slightly, or close the other GPU consumer.

cd "$(dirname "${BASH_SOURCE[0]}")"

VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
./venv/bin/vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --served-model-name llama-3.1-8b \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  --chat-template tool_chat_template_llama3.1_json.jinja \
  --port 8000 \
  "$@"
