#!/usr/bin/env bash
# Round 2 of the model comparison: all 8 configurations with the uniform server
# settings in bench_model.sh. Smoke test is informational only (it uses
# tool_choice "auto"; the agent sends "required").
cd "$(dirname "${BASH_SOURCE[0]}")"
export IGNORE_SMOKE=1
for spec in "run_qwen3_8b_nothink.sh qwen3-8b-nothink" "run_granite3.1_8b.sh granite-3.1-8b" \
            "run_phi4_mini.sh phi-4-mini" "run_xlam2_8b.sh xlam-2-8b" "run_ministral_8b.sh ministral-8b" \
            "run_gemma3_4b.sh gemma-3-4b" "run_llama3.1_8b.sh llama-3.1-8b" "run_qwen3_8b.sh Qwen/Qwen3-8B-AWQ"; do
  set -- $spec
  echo "=== $(date +%T) $2"
  ./bench_model.sh "$1" "$2" 2>&1 | grep -v "^(EngineCore\|^(APIServer" | tail -25
  echo "=== $(date +%T) $2 exit=${PIPESTATUS[0]}"
  sleep 10
done
echo "=== ROUND 2 DONE"
