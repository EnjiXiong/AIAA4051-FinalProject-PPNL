#!/bin/bash
# run_all.sh - Run all experiments for PPNL Grid Path Planning project
# Adjust epochs/batch_size based on your GPU memory

set -e

echo "============================================"
echo "  PPNL Grid Path Planning - Full Pipeline"
echo "============================================"

# ─── Phase 1: Baseline Fine-tuning (Part 1: 30%) ────────────────────────────

echo ""
echo ">>> Phase 1: Baseline fine-tuning"

# T5-small (vanilla)
echo "--- T5-small vanilla ---"
python train.py \
    --model t5-small \
    --input_format vanilla \
    --epochs 20 \
    --batch_size 32 \
    --lr 3e-4 \
    --bf16

# T5-base (vanilla)
echo "--- T5-base vanilla ---"
python train.py \
    --model t5-base \
    --input_format vanilla \
    --epochs 15 \
    --batch_size 16 \
    --lr 3e-4 \
    --bf16

# ─── Phase 2: Improved Fine-tuning (Part 2: 50%) ────────────────────────────

echo ""
echo ">>> Phase 2: Improved fine-tuning"

# T5-base with structured input
echo "--- T5-base structured ---"
python train.py \
    --model t5-base \
    --input_format structured \
    --epochs 15 \
    --batch_size 16 \
    --lr 3e-4 \
    --bf16

# T5-base with CoT supervision
echo "--- T5-base CoT ---"
python train.py \
    --model t5-base \
    --input_format cot \
    --epochs 15 \
    --batch_size 16 \
    --lr 3e-4 \
    --max_target_len 256 \
    --bf16

# ─── Phase 3: Evaluation on all test sets ────────────────────────────────────

echo ""
echo ">>> Phase 3: Evaluation"

for model_dir in models/*/best; do
    model_name=$(basename $(dirname $model_dir))
    
    # Determine input format from model name
    format="vanilla"
    if [[ $model_name == *"structured"* ]]; then
        format="structured"
    elif [[ $model_name == *"cot"* ]]; then
        format="cot"
    fi
    
    echo "--- Evaluating $model_name ($format) ---"
    python run_eval.py \
        --model_dir $model_dir \
        --input_format $format \
        --save_predictions
done

# ─── Phase 4: Prompting experiments ──────────────────────────────────────────

echo ""
echo ">>> Phase 4: Prompting experiments"

for strategy in zero_shot few_shot cot_coordinate cot_plan_then_act cot_grid; do
    for test_file in data/1_goals_test_seen_6x6_samples.json data/1_goals_test_unseen_7x7_samples.json; do
        test_name=$(basename $test_file | sed 's/_samples.json//')
        echo "--- Flan-T5-base / $strategy / $test_name ---"
        python prompt_eval.py \
            --model google/flan-t5-base \
            --strategy $strategy \
            --test_file $test_file \
            --test_name $test_name \
            --save_predictions
    done
done

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "============================================"