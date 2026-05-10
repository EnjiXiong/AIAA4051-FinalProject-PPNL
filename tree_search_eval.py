"""
Tree Search inference for grid path planning.
Model provides learned policy prior; executor enforces hard constraints.

Algorithm: At each step, the model scores 4 candidate actions (up/down/left/right).
The executor prunes illegal moves (obstacles, out-of-bounds). Top-K beams survive.
Position deduplication prevents wasted beam capacity. Cycle detection avoids loops.

Supports two decoder modes:
  - vanilla: each step = 1 action token
  - cot: each step = action token + forced coordinate continuation,
         preserving the CoT model's spatial state tracking

Usage:
    # Multi-scale model + tree search (beam_width=4)
    python tree_search_eval.py --model_dir models/sft_multiscale_40ep/best \
        --input_format vanilla --beam_width 4

    # CoT model + tree search (beam_width=8)
    python tree_search_eval.py --model_dir models/t5-base_cot_ep15_lr0.0003/best \
        --input_format cot --beam_width 8

    # With custom OOD test set
    python tree_search_eval.py --model_dir models/sft_multiscale_40ep/best \
        --input_format vanilla --beam_width 4 \
        --extra_test OOD_novel=data/ood_novel_sizes.json

    # Only run custom test sets (skip built-in)
    python tree_search_eval.py --model_dir models/sft_multiscale_40ep/best \
        --input_format vanilla --beam_width 4 \
        --extra_test OOD_novel=data/ood_novel_sizes.json --only_extra
"""
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from data_utils import (
    load_ppnl_data, format_vanilla, format_structured,
    parse_nl_description, extract_actions_from_cot
)
from evaluate_utils import evaluate_batch, find_position


# ─── Constants ──────────────────────────────────────────────────────────────

TEST_SETS = {
    'ID_seen_6x6': 'data/1_goals_test_seen_6x6_samples.json',
    'ID_unseen_6x6': 'data/1goals_unseen_6x6_samples.json',
    'OOD_5x5': 'data/1_goals_test_unseen_5x5_samples.json',
    'OOD_7x7': 'data/1_goals_test_unseen_7x7_samples.json',
    'OOD_6x6_dense': 'data/1_goals_test_unseen_6x6more_obstacles_samples.json',
}

ACTION_MAP = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1),
}

ACTION_NAMES = ['up', 'down', 'left', 'right']


# ─── Beam state ─────────────────────────────────────────────────────────────

@dataclass
class Beam:
    position: Tuple[int, int]
    actions: List[str]
    log_prob: float
    decoder_ids: torch.Tensor      # [1, seq_len]
    visited: frozenset = field(default_factory=frozenset)


# ─── Token ID helpers ───────────────────────────────────────────────────────

def get_action_token_ids(tokenizer) -> Dict[str, int]:
    """
    Map each action name to its SentencePiece token ID.
    T5 tokenizer: 'up' -> [▁up] (single token).
    Validates that each action is indeed a single token.
    """
    ids = {}
    for action in ACTION_NAMES:
        encoded = tokenizer.encode(action, add_special_tokens=False)
        if len(encoded) != 1:
            # Fallback: try with space prefix
            encoded = tokenizer.encode(f" {action}", add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(
                    f"Action '{action}' tokenizes to {len(encoded)} tokens "
                    f"({encoded}), expected 1. Cannot use tree search."
                )
        ids[action] = encoded[0]
    return ids


def tokenize_cot_prefix(tokenizer, start: Tuple[int, int]) -> List[int]:
    """Tokenize the CoT prefix: 'Start at (r,c) |'."""
    text = f"Start at ({start[0]},{start[1]}) |"
    return tokenizer.encode(text, add_special_tokens=False)


def tokenize_cot_continuation(
    tokenizer, action: str, new_pos: Tuple[int, int], is_goal: bool
) -> List[int]:
    """
    Tokenize the CoT step continuation AFTER the action token.
    E.g., ' -> (1,3) |' or ' -> (2,1) | Done' if goal reached.
    """
    text = f" -> ({new_pos[0]},{new_pos[1]}) |"
    if is_goal:
        text += " Done"
    return tokenizer.encode(text, add_special_tokens=False)


# ─── Core tree search ───────────────────────────────────────────────────────

def tree_search_single(
    model,
    tokenizer,
    input_text: str,
    grid: List[List[int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    action_token_ids: Dict[str, int],
    device: torch.device,
    input_format: str = 'vanilla',
    beam_width: int = 4,
    max_steps: int = 60,
    max_source_len: int = 512,
) -> str:
    """
    Run tree search for a single sample.

    Args:
        model:            T5/BART seq2seq model
        tokenizer:        corresponding tokenizer
        input_text:       formatted input string
        grid:             2D grid array (0=empty, 1=obstacle, 2=start, 3=goal)
        start, goal:      (row, col) tuples
        action_token_ids: {action_name: token_id}
        device:           torch device
        input_format:     'vanilla' or 'cot'
        beam_width:       number of beams to keep
        max_steps:        max action steps before giving up
        max_source_len:   max encoder input length

    Returns:
        Action string, e.g., 'left left down right'
    """
    nx, ny = len(grid), len(grid[0])

    # ── Encode input once ────────────────────────────────────────────────
    enc = tokenizer(
        input_text, max_length=max_source_len,
        padding='max_length', truncation=True, return_tensors='pt'
    )
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    with torch.no_grad():
        encoder_out = model.get_encoder()(
            input_ids=input_ids, attention_mask=attention_mask
        )
    encoder_hidden = encoder_out.last_hidden_state  # [1, src_len, hidden]

    # ── Decoder start token ──────────────────────────────────────────────
    dec_start = model.config.decoder_start_token_id
    if dec_start is None:
        dec_start = tokenizer.pad_token_id

    # ── Build initial decoder prefix ─────────────────────────────────────
    if input_format == 'cot':
        prefix_ids = [dec_start] + tokenize_cot_prefix(tokenizer, start)
    else:
        prefix_ids = [dec_start]

    init_dec = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    # ── Initialize beam set ──────────────────────────────────────────────
    beams = [Beam(
        position=start,
        actions=[],
        log_prob=0.0,
        decoder_ids=init_dec,
        visited=frozenset([start]),
    )]

    best_solution = None        # (actions_list, log_prob)

    # ── Main search loop ─────────────────────────────────────────────────
    for step in range(max_steps):
        if not beams:
            break

        # Batch all beams into a single forward pass
        num_beams = len(beams)
        max_dec_len = max(b.decoder_ids.shape[1] for b in beams)

        batch_dec = torch.full(
            (num_beams, max_dec_len), tokenizer.pad_token_id,
            dtype=torch.long, device=device
        )
        batch_dec_mask = torch.zeros(
            (num_beams, max_dec_len), dtype=torch.long, device=device
        )
        last_positions = []

        for i, beam in enumerate(beams):
            L = beam.decoder_ids.shape[1]
            batch_dec[i, :L] = beam.decoder_ids[0]
            batch_dec_mask[i, :L] = 1
            last_positions.append(L - 1)

        # Expand encoder output to beam batch
        exp_hidden = encoder_hidden.expand(num_beams, -1, -1)
        exp_mask = attention_mask.expand(num_beams, -1)

        with torch.no_grad():
            outputs = model(
                encoder_outputs=(exp_hidden,),
                attention_mask=exp_mask,
                decoder_input_ids=batch_dec,
                decoder_attention_mask=batch_dec_mask,
            )

        # ── Branch each beam ─────────────────────────────────────────────
        candidates = []

        for i, beam in enumerate(beams):
            logits = outputs.logits[i, last_positions[i], :]   # [vocab_size]
            log_probs = torch.log_softmax(logits, dim=-1)

            for action_name in ACTION_NAMES:
                dr, dc = ACTION_MAP[action_name]
                new_pos = (beam.position[0] + dr, beam.position[1] + dc)

                # Prune: out of bounds
                if not (0 <= new_pos[0] < nx and 0 <= new_pos[1] < ny):
                    continue

                # Prune: obstacle
                if grid[new_pos[0]][new_pos[1]] == 1:
                    continue

                # Prune: cycle (revisiting)
                if new_pos in beam.visited:
                    continue

                token_id = action_token_ids[action_name]
                a_log_prob = log_probs[token_id].item()
                new_log_prob = beam.log_prob + a_log_prob
                new_actions = beam.actions + [action_name]
                new_visited = beam.visited | {new_pos}
                is_goal = (new_pos == goal)

                # Build new decoder_input_ids
                if input_format == 'cot':
                    cont_ids = tokenize_cot_continuation(
                        tokenizer, action_name, new_pos, is_goal
                    )
                    append_ids = [token_id] + cont_ids
                else:
                    append_ids = [token_id]

                new_dec = torch.cat([
                    beam.decoder_ids,
                    torch.tensor([append_ids], dtype=torch.long, device=device)
                ], dim=1)

                if is_goal:
                    # Record solution; don't expand further
                    if (best_solution is None
                            or new_log_prob > best_solution[1]):
                        best_solution = (new_actions, new_log_prob)
                else:
                    candidates.append(Beam(
                        position=new_pos,
                        actions=new_actions,
                        log_prob=new_log_prob,
                        decoder_ids=new_dec,
                        visited=new_visited,
                    ))

        # ── Position deduplication: keep best beam per position ──────────
        pos_best: Dict[Tuple[int, int], Beam] = {}
        for c in candidates:
            key = c.position
            if key not in pos_best or c.log_prob > pos_best[key].log_prob:
                pos_best[key] = c

        # ── Top-K selection ──────────────────────────────────────────────
        survivors = sorted(pos_best.values(), key=lambda b: b.log_prob,
                           reverse=True)
        beams = survivors[:beam_width]

        # ── Early termination ────────────────────────────────────────────
        if best_solution is not None:
            # Solution found; stop if all surviving beams are worse
            if not beams or beams[0].log_prob < best_solution[1]:
                break

    # ── Return best result ───────────────────────────────────────────────
    if best_solution is not None:
        return ' '.join(best_solution[0])

    # No solution: return the beam closest to goal
    if beams:
        closest = min(
            beams,
            key=lambda b: abs(b.position[0] - goal[0])
                        + abs(b.position[1] - goal[1])
        )
        return ' '.join(closest.actions)

    return ''


# ─── Evaluation driver ──────────────────────────────────────────────────────

def format_input_text(sample: Dict, input_format: str) -> str:
    """Format a sample's input text according to the specified format."""
    if input_format == 'vanilla':
        return format_vanilla(sample)
    elif input_format in ('structured', 'cot'):
        # CoT model uses structured encoder input
        return format_structured(sample)
    else:
        raise ValueError(f"Unknown input_format: {input_format}")


def run_eval_on_dataset(
    model, tokenizer, data: List[Dict], action_token_ids: Dict[str, int],
    device: torch.device, args, dataset_name: str
) -> Dict:
    """Run tree search evaluation on a single dataset."""
    # Filter unreachable samples
    filtered = [s for s in data
                if 'Goal not reachable' not in s.get('agent_as_a_point', '')]

    predictions = []
    t0 = time.time()

    for idx, sample in enumerate(filtered):
        grid = sample['world']
        start = find_position(grid, 2)
        goal = find_position(grid, 3)

        if start is None or goal is None:
            predictions.append('')
            continue

        input_text = format_input_text(sample, args.input_format)

        # Adaptive max_steps: proportional to grid size
        nx, ny = len(grid), len(grid[0])
        max_steps = min(args.max_steps, 3 * (nx + ny))

        pred = tree_search_single(
            model=model,
            tokenizer=tokenizer,
            input_text=input_text,
            grid=grid,
            start=start,
            goal=goal,
            action_token_ids=action_token_ids,
            device=device,
            input_format=args.input_format,
            beam_width=args.beam_width,
            max_steps=max_steps,
            max_source_len=args.max_source_len,
        )
        predictions.append(pred)

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(filtered) - idx - 1) / rate
            print(f"    [{dataset_name}] {idx+1}/{len(filtered)} "
                  f"({rate:.1f} samples/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"    [{dataset_name}] Done: {len(filtered)} samples in {elapsed:.1f}s "
          f"({len(filtered)/elapsed:.1f} samples/s)")

    # Evaluate with standard metrics
    result = evaluate_batch(filtered, predictions)
    metrics = result['metrics']

    # Per-size breakdown if data has grid_size info
    per_size = {}
    if any('grid_size' in s or 'nl_description' in s for s in filtered):
        size_groups: Dict[str, Tuple[List, List]] = {}
        for sample, pred in zip(filtered, predictions):
            # Determine grid size
            if 'grid_size' in sample:
                gs = sample['grid_size']
                if isinstance(gs, list):
                    size_key = f"{gs[0]}x{gs[1]}"
                else:
                    size_key = str(gs)
            else:
                info = parse_nl_description(sample['nl_description'])
                r, c = info['grid_size']
                size_key = f"{r}x{c}"

            if size_key not in size_groups:
                size_groups[size_key] = ([], [])
            size_groups[size_key][0].append(sample)
            size_groups[size_key][1].append(pred)

        if len(size_groups) > 1:
            for size_key in sorted(size_groups.keys(),
                                   key=lambda s: int(s.split('x')[0])):
                s_data, s_preds = size_groups[size_key]
                s_result = evaluate_batch(s_data, s_preds)
                per_size[size_key] = s_result['metrics']

    return {
        'metrics': metrics,
        'per_size': per_size,
        'predictions': list(zip(
            [s.get('nl_description', '') for s in filtered],
            predictions
        )) if args.save_predictions else None,
        'per_sample': result['per_sample'],
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Tree Search evaluation for path planning')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to saved model directory')
    parser.add_argument('--input_format', type=str, default='vanilla',
                        choices=['vanilla', 'structured', 'cot'])
    parser.add_argument('--beam_width', type=int, default=4,
                        help='Number of beams to keep (default: 4)')
    parser.add_argument('--max_steps', type=int, default=60,
                        help='Global max action steps (capped per grid size)')
    parser.add_argument('--max_source_len', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--extra_test', nargs='+', default=[],
                        help='Extra test sets as NAME=PATH pairs')
    parser.add_argument('--only_extra', action='store_true',
                        help='Skip built-in test sets, only run --extra_test')
    parser.add_argument('--test_sets', nargs='+', default=None,
                        help='Subset of built-in test sets to run')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    # Get action token IDs
    action_token_ids = get_action_token_ids(tokenizer)
    print(f"Action token IDs: {action_token_ids}")

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
    tag = f"tree_bw{args.beam_width}"
    if args.input_format != 'vanilla':
        tag += f"_{args.input_format}"
    out_dir = Path(args.output_dir) / model_name / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Tree Search Evaluation")
    print(f"  Model:        {args.model_dir}")
    print(f"  Input format: {args.input_format}")
    print(f"  Beam width:   {args.beam_width}")
    print(f"  Max steps:    {args.max_steps}")
    print(f"  Test sets:    {list(test_sets.keys())}")
    print(f"  Output dir:   {out_dir}")
    print(f"{'='*60}\n")

    all_results = {}

    for name, path in test_sets.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: file not found ({path})")
            continue

        print(f"\n── {name} ──")
        data = load_ppnl_data(path)
        result = run_eval_on_dataset(
            model, tokenizer, data, action_token_ids, device, args, name
        )

        metrics = result['metrics']
        print(f"  Total:        {metrics['total']}")
        print(f"  Success Rate: {metrics['success_rate']:.4f}")
        print(f"  Feasibility:  {metrics['feasibility']:.4f}")
        print(f"  Optimality:   {metrics['optimality']:.4f}")
        if metrics.get('error_distribution'):
            print(f"  Errors:       {metrics['error_distribution']}")

        if result['per_size']:
            print(f"  Per-size breakdown:")
            for size_key, sm in result['per_size'].items():
                print(f"    {size_key}: success={sm['success_rate']:.4f}  "
                      f"feasible={sm['feasibility']:.4f}  "
                      f"errors={sm.get('error_distribution', {})}")

        # Store summary (without per_sample detail)
        summary = dict(metrics)
        if result['per_size']:
            summary['per_size'] = result['per_size']
        all_results[name] = summary

        # Save predictions if requested
        if args.save_predictions and result['predictions']:
            pred_file = out_dir / f'{name}_predictions.json'
            pred_data = []
            for (nl, pred), ps in zip(result['predictions'],
                                       result['per_sample']):
                pred_data.append({
                    'nl_description': nl,
                    'predicted': pred,
                    'success': ps['success'],
                    'error_type': ps['error_type'],
                })
            with open(pred_file, 'w') as f:
                json.dump(pred_data, f, indent=2)

    # Save summary
    summary_file = out_dir / 'eval_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

    # Print comparison table
    print(f"\n{'Dataset':<20} {'Success':>10} {'Feasible':>10} {'Optimal':>10}")
    print('-' * 52)
    for name, m in all_results.items():
        print(f"{name:<20} {m['success_rate']:>10.4f} "
              f"{m['feasibility']:>10.4f} {m['optimality']:>10.4f}")


if __name__ == '__main__':
    main()