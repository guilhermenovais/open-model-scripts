#!/usr/bin/env bash
# Reruns models whose first attempt failed, after run_queue.sh finishes.
cd "$(dirname "${BASH_SOURCE[0]}")"
while pgrep -f "bash ./run_queue.sh" >/dev/null; do sleep 20; done
for spec in "run_phi4_mini.sh phi-4-mini" "run_gemma3_4b.sh gemma-3-4b"; do
  set -- $spec
  echo "=== $(date +%T) $2"
  ./bench_model.sh "$1" "$2" 2>&1 | tail -25
  echo "=== $(date +%T) $2 exit=${PIPESTATUS[0]}"
done
