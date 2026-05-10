"""
Evaluate a trained model on all test sets (ID and OOD).

Usage:
    python run_eval.py --model_dir models/t5-base_vanilla_ep15_lr0.0003/best
    python run_eval.py --model_dir models/t5-base_cot_ep15_lr0.0003/best --input_format cot

    # Custom OOD test set
    python run_eval.py --model_dir models/sft_multiscale_40ep/best \
        --extra_test OOD_novel=data/ood_novel_sizes.json --save_predictions

    # Only run custom test sets
    python run_eval.py --model_dir models/sft_multiscale_40ep/best \
        --extra_test OOD_novel=data/ood_novel_sizes.json --only_extra
"""
import argparse
import json
import os
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from data_utils import (
    PPNLDataset, load_ppnl_data, extract_actions_from_cot,
    parse_nl_description
)
from evaluate_utils import evaluate_batch


TEST_SETS = {
    'ID_seen_6x6': 'data/1_goals_test_seen_6x6_samples.json',
    'ID_unseen_6x6': 'data/1goals_unseen_6x6_samples.json',
    'OOD_5x5': 'data/1_goals_test_unseen_5x5_samples.json',
    'OOD_7x7': 'data/1_goals_test_unseen_7x7_samples.json',
    'OOD_6x6_dense': 'data/1_goals_test_unseen_6x6more_obstacles_samples.json',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to saved model directory')
    parser.add_argument('--input_format', type=str, default='vanilla',
                        choices=['vanilla', 'structured', 'cot'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_source_len', type=int, default=512)
    parser.add_argument('--max_target_len', type=int, default=256)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--test_sets', nargs='+', default=None,
                        help='Specific built-in test sets to run (default: all)')
    parser.add_argument('--extra_test', nargs='+', default=[],
                        help='Extra test sets as NAME=PATH pairs')
    parser.add_argument('--only_extra', action='store_true',
                        help='Skip built-in test sets, only run --extra_test')
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save per-sample predictions')
    return parser.parse_args()


def compute_per_size_breakdown(filtered, predictions, result):
    """Compute per-grid-size metrics for datasets with mixed sizes."""
    size_groups = {}
    for sample, pred in zip(filtered, predictions):
        if 'grid_size' in sample:
            gs = sample['grid_size']
            size_key = f"{gs[0]}x{gs[1]}" if isinstance(gs, list) else str(gs)
        else:
            try:
                info = parse_nl_description(sample['nl_description'])
                r, c = info['grid_size']
                size_key = f"{r}x{c}"
            except Exception:
                continue

        if size_key not in size_groups:
            size_groups[size_key] = ([], [])
        size_groups[size_key][0].append(sample)
        size_groups[size_key][1].append(pred)

    if len(size_groups) <= 1:
        return {}

    per_size = {}
    for size_key in sorted(size_groups.keys(),
                           key=lambda s: int(s.split('x')[0])):
        s_data, s_preds = size_groups[size_key]
        s_result = evaluate_batch(s_data, s_preds)
        per_size[size_key] = s_result['metrics']
    return per_size


def run_eval(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    # Build test set dict
    test_sets = {}
    if not args.only_extra:
        test_sets.update(TEST_SETS)
        if args.test_sets:
            test_sets = {k: v for k, v in test_sets.items()
                         if k in args.test_sets}
    for extra in args.extra_test:
        name, path = extra.split('=', 1)
        test_sets[name] = path

    # Output directory
    model_name = Path(args.model_dir).parent.name
    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for name, path in test_sets.items():
        if not os.path.exists(path):
            print(f"  Skipping {name}: file not found")
            continue

        print(f"\nEvaluating on {name}...")
        test_data = load_ppnl_data(path)

        dataset = PPNLDataset(
            test_data, tokenizer,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            input_format=args.input_format
        )
        loader = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=2, pin_memory=True)

        # Generate predictions
        predictions = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=args.max_target_len,
                    num_beams=args.num_beams,
                    early_stopping=True if args.num_beams > 1 else False,
                )
                decoded = tokenizer.batch_decode(outputs,
                                                 skip_special_tokens=True)
                predictions.extend(decoded)

        # Extract actions from CoT if needed
        if args.input_format == 'cot':
            predictions = [extract_actions_from_cot(p) for p in predictions]

        # Filter unreachable samples
        filtered = [s for s in test_data
                    if 'Goal not reachable' not in s.get('agent_as_a_point', '')]

        result = evaluate_batch(filtered, predictions)
        metrics = result['metrics']

        print(f"  Total: {metrics['total']}")
        print(f"  Success Rate:  {metrics['success_rate']:.4f}")
        print(f"  Feasibility:   {metrics['feasibility']:.4f}")
        print(f"  Optimality:    {metrics['optimality']:.4f}")
        print(f"  Exact Match:   {metrics['exact_match']:.4f}")
        if metrics.get('error_distribution'):
            print(f"  Errors: {metrics['error_distribution']}")

        # Per-size breakdown
        per_size = compute_per_size_breakdown(filtered, predictions, result)
        if per_size:
            print(f"  Per-size breakdown:")
            for size_key, sm in per_size.items():
                print(f"    {size_key}: success={sm['success_rate']:.4f}  "
                      f"feasible={sm['feasibility']:.4f}")

        summary = dict(metrics)
        if per_size:
            summary['per_size'] = per_size
        all_results[name] = summary

        # Save predictions if requested
        if args.save_predictions:
            pred_file = out_dir / f'{name}_predictions.json'
            pred_data = []
            for i, (sample, pred) in enumerate(zip(filtered, predictions)):
                pred_data.append({
                    'nl_description': sample['nl_description'],
                    'ground_truth': sample['agent_as_a_point'].strip(),
                    'predicted': pred,
                    'success': result['per_sample'][i]['success'],
                    'error_type': result['per_sample'][i]['error_type'],
                })
            with open(pred_file, 'w') as f:
                json.dump(pred_data, f, indent=2)
            print(f"  Predictions saved to {pred_file}")

    # Save summary
    summary_file = out_dir / 'eval_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Summary saved to {summary_file}")

    # Print comparison table
    print(f"\n{'Dataset':<20} {'Success':>10} {'Feasible':>10} {'Optimal':>10}")
    print('-' * 52)
    for name, m in all_results.items():
        print(f"{name:<20} {m['success_rate']:>10.4f} "
              f"{m['feasibility']:>10.4f} {m['optimality']:>10.4f}")

    return all_results


if __name__ == '__main__':
    args = parse_args()
    run_eval(args)