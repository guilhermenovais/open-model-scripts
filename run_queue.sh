#!/usr/bin/env bash
# Runs the remaining comparison models one after another (only one fits in VRAM).
cd "$(dirname "${BASH_SOURCE[0]}")"
while pgrep -f "bench_model.sh run_qwen3_8b.sh" >/dev/null; do sleep 20; done
for spec in "run_qwen3_8b_nothink.sh qwen3-8b-nothink" "run_llama3.1_8b.sh llama-3.1-8b" \
            "run_phi4_mini.sh phi-4-mini" "run_granite3.1_8b.sh granite-3.1-8b" \
            "run_gemma3_4b.sh gemma-3-4b" "run_xlam2_8b.sh xlam-2-8b" "run_ministral_8b.sh ministral-8b"; do
  set -- $spec
  echo "=== $(date +%T) $2"
  ./bench_model.sh "$1" "$2" 2>&1 | tail -25
  echo "=== $(date +%T) $2 exit=${PIPESTATUS[0]}"
  sleep 10
done
