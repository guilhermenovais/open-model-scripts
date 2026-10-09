# open-model-scripts

Scripts for serving small open-weight models with [vLLM](https://github.com/vllm-project/vllm)
on one 12 GB GPU (RTX 5070, WSL2, no CUDA toolkit), and for benchmarking them with the
[brazilian-open-data-agent](https://github.com/guilhermenovais/brazilian-open-data-agent)
testset. The results of the 2026-10-09 comparison are in that repo, in
`specs/010-model-comparison/results.md`.

Each script's header comment explains its flags. Shared flags (fp8 KV cache, `TRITON_ATTN`,
`--max-num-batched-tokens 4096`, the WSL2 env vars) are explained in `run_qwen3_8b.sh`.

## Serve scripts

All serve on port 8000 and pass any extra arguments to `vllm serve`.

| Script | Model | Weights | Served name |
|---|---|---|---|
| `run_qwen3_8b.sh` | `Qwen/Qwen3-8B-AWQ`, thinking on | AWQ int4 | `Qwen/Qwen3-8B-AWQ` |
| `run_qwen3_8b_nothink.sh` | same, thinking off | AWQ int4 | `qwen3-8b-nothink` |
| `run_qwen3.5_4b.sh` | `Qwen/Qwen3.5-4B` | bf16 | `Qwen/Qwen3.5-4B` |
| `run_llama3.1_8b.sh` | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | AWQ int4 | `llama-3.1-8b` |
| `run_granite3.1_8b.sh` | `RedHatAI/granite-3.1-8b-instruct-quantized.w4a16` | W4A16 | `granite-3.1-8b` |
| `run_ministral_8b.sh` | `warshanks/Ministral-8B-Instruct-2410-AWQ` | AWQ W4A16 | `ministral-8b` |
| `run_xlam2_8b.sh` | `Salesforce/Llama-xLAM-2-8b-fc-r`, local quant | RTN W4A16 | `xlam-2-8b` |
| `run_phi4_mini.sh` | `microsoft/Phi-4-mini-instruct` | online FP8 | `phi-4-mini` |
| `run_gemma3_4b.sh` | `unsloth/gemma-3-4b-it` | online FP8 | `gemma-3-4b` |

Files these scripts depend on:

- `tool_chat_template_*.jinja` are vLLM's example tool templates (v0.29.0). The Llama one has
  a local change (tools go in the system prompt). Not all of them are used; see each script.
- `mistral_defs_fix_tool_parser.py` is a vLLM tool-parser plugin for Ministral. vLLM's Mistral
  parser breaks on tool schemas that use `$defs` when `tool_choice="required"`.
- `quantize_xlam_w4a16.py` builds `models/Llama-xLAM-2-8b-fc-r-W4A16` (CPU, about 2 minutes;
  needs `quant-venv`, see Setup).
- `make_ministral_hf_tokenizer.py` builds `models/ministral-8b-hf-tokenizer`.

## Benchmarking

- `bench_model.sh <serve-script> <served-name>` runs one model end to end:
  1. starts the server, with the comparison's uniform settings (xgrammar guided decoding without
     free whitespace, an 8192-token output cap);
  2. runs `smoke_tool_call.py`;
  3. runs the 50-question testset with the agent's `testset_runner`;
  4. stops the server.

  Logs go to `logs/`. A failed smoke test skips the testset run unless `IGNORE_SMOKE=1`. The smoke
  test uses `tool_choice="auto"`, but the agent sends `"required"`, so treat it as a check of the
  model's native tool format only.
- `smoke_tool_call.py <served-name>` checks that a tool question gets a parsed tool call and a
  plain question gets text.
- `run_queue*.sh` are the queues that ran the comparison, kept as a record. `run_queue_round2.sh`
  is the final one.

## Setup

```bash
python3.14 -m venv venv && ./venv/bin/pip install vllm==0.29.0       # serving
uv venv quant-venv --python 3.12 && VIRTUAL_ENV=quant-venv uv pip install llmcompressor  # quantization only
CUDA_VISIBLE_DEVICES="" ./quant-venv/bin/python quantize_xlam_w4a16.py
./venv/bin/python make_ministral_hf_tokenizer.py
```

`bench_model.sh` expects the agent repo at `/home/guilherme/brazilian-open-data-agent`.
