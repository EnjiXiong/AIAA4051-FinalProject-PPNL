#!/bin/bash
# Evaluate the two new CoT SFT seed runs on all 5 standard PPNL test sets.
# Run from grid-path-planning/. Assumes conda env `dl` activated.

set -e

echo "===== eval seed 1234 starting $(date) ====="
python run_eval.py \
    --model_dir models/t5-base_cot_ep15_lr0.0003_seed1234/best \
    --input_format cot \
    --save_predictions

echo "===== eval seed 7890 starting $(date) ====="
python run_eval.py \
    --model_dir models/t5-base_cot_ep15_lr0.0003_seed7890/best \
    --input_format cot \
    --save_predictions

echo "===== ALL EVAL DONE $(date) ====="
