#!/usr/bin/env bash
# Third batch: models whose serve config had to change (xLAM quant, Ministral tokenizer).
cd "$(dirname "${BASH_SOURCE[0]}")"
while pgrep -f "bash ./run_queue_retry.sh" >/dev/null; do sleep 20; done
for spec in "run_xlam2_8b.sh xlam-2-8b" "run_ministral_8b.sh ministral-8b"; do
  set -- $spec
  echo "=== $(date +%T) $2"
  ./bench_model.sh "$1" "$2" 2>&1 | tail -25
  echo "=== $(date +%T) $2 exit=${PIPESTATUS[0]}"
  sleep 10
done
