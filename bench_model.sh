#!/usr/bin/env bash
# Serves one model, smoke-tests its tool calling, then runs the 50-question testset
# against it, and stops the server. Usage: bench_model.sh <serve-script> <served-model-name>
# Logs go to ~/llm/logs/<served-name>.{serve,smoke,run}.log.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
script="$1"; name="$2"; tag="$(echo "$name" | tr / _)"
repo=/home/guilherme/brazilian-open-data-agent

# Same server-side settings for every model in the comparison:
# - xgrammar without free whitespace: pydantic-ai sends tool_choice="required", which
#   vLLM enforces with guided JSON; with free whitespace allowed, weaker models filled
#   their whole output budget with whitespace (round 1 of the comparison).
# - an 8192-token output cap, so a runaway generation fails in ~1.5 min instead of
#   running to the context limit, while leaving room for Qwen3's thinking.
setsid ./"$script" \
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
  --override-generation-config '{"max_new_tokens": 8192}' \
  > "logs/$tag.serve.log" 2>&1 &
pgid=$!
trap 'kill -- -$pgid 2>/dev/null; sleep 5; kill -9 -- -$pgid 2>/dev/null' EXIT

for i in $(seq 1 180); do
  curl -sf http://localhost:8000/v1/models >/dev/null && break
  kill -0 $pgid 2>/dev/null || { echo "SERVER DIED (see logs/$tag.serve.log)"; exit 2; }
  sleep 5
done
curl -sf http://localhost:8000/v1/models >/dev/null || { echo "SERVER TIMEOUT"; exit 2; }
grep -E "Maximum concurrency|KV cache size|max_model_len" "logs/$tag.serve.log" | tail -3

python3 smoke_tool_call.py "$name" 2>&1 | tee "logs/$tag.smoke.log"
smoke=${PIPESTATUS[0]}
[ "$smoke" != 0 ] && [ "${IGNORE_SMOKE:-0}" != 1 ] && { echo "SMOKE FAILED, skipping testset run"; exit 3; }
[ "${SMOKE_ONLY:-0}" = 1 ] && exit 0

cd "$repo"
uv run python -m testset_runner.cli run --testset data/testsets/orcamentos-aeb-csv.json \
  --model "$name" --base-url http://localhost:8000/v1 --api-key x --max-attempts 3 \
  2>&1 | tee "$HOME/llm/logs/$tag.run.log" | tail -40
