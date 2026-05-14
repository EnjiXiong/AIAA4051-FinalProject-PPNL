#!/bin/bash
# Follow-up eval: run only the novel-OOD test set (sizes 8/9/11/14/20)
# on both new CoT seed checkpoints. Skips the 5 built-in PPNL test sets
# (already evaluated by run_seed_eval.sh).
# Run from grid-path-planning/. Assumes conda env `dl` activated.

set -e

echo "===== OOD_novel eval seed 1234 starting $(date) ====="
python run_eval.py \
    --model_dir models/t5-base_cot_ep15_lr0.0003_seed1234/best \
    --input_format cot \
    --extra_test OOD_novel=data/ood_novel_sizes.json \
    --only_extra \
    --save_predictions

echo "===== OOD_novel eval seed 7890 starting $(date) ====="
python run_eval.py \
    --model_dir models/t5-base_cot_ep15_lr0.0003_seed7890/best \
    --input_format cot \
    --extra_test OOD_novel=data/ood_novel_sizes.json \
    --only_extra \
    --save_predictions

echo "===== ALL OOD_novel EVAL DONE $(date) ====="
