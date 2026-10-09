#!/usr/bin/env bash
set -euo pipefail

# Serves Ministral-8B-Instruct-2410 as warshanks' AWQ (W4A16 asym, group 128,
# compressed-tensors) quant; bf16 doesn't fit in 12GB. The tokenizer is the HF one
# (models/ministral-8b-hf-tokenizer: the repo's tokenizer.json, tokenizer_config.json,
# special_tokens_map.json and chat_template.jinja, which renders [AVAILABLE_TOOLS] /
# [TOOL_CALLS]), not Mistral's tekken.json: with the Mistral tokenizer, vLLM 0.29
# builds a tool-call grammar that can't resolve JSON-schema $defs refs, and every
# agent request failed with "Grammar error: Pointer '/$defs/EqualsCondition' does
# not exist". A separate directory is needed because transformers 5 loads
# MistralCommonBackend whenever tekken.json sits next to the weights, even with
# --tokenizer-mode hf. With the HF tokenizer the same $defs bug remained in the
# Mistral parser's tool_choice="required" schema (which pydantic-ai sends), hence
# the mistral_defs_fix parser plugin (see mistral_defs_fix_tool_parser.py).
# --max-model-len 32768: the checkpoint's
# max_position_embeddings / sliding_window.
#
# Shared flags (WSL2 env vars, fp8 KV cache + TRITON_ATTN, --max-num-batched-tokens
# 4096, 0.88 GPU memory) are explained in run_qwen3_8b.sh. Written for the
# model comparison in brazilian-open-data-agent specs/010-model-comparison.

cd "$(dirname "${BASH_SOURCE[0]}")"

VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
./venv/bin/vllm serve warshanks/Ministral-8B-Instruct-2410-AWQ \
  --served-model-name ministral-8b \
  --tokenizer models/ministral-8b-hf-tokenizer \
  --tokenizer-mode hf \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-parser-plugin mistral_defs_fix_tool_parser.py \
  --tool-call-parser mistral_defs_fix \
  --port 8000 \
  "$@"
