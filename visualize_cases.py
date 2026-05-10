"""
Generate cross-method path comparison figures for the report.
Finds matching samples across prediction files and plots side-by-side.

Usage:
    python visualize_cases.py \
        --case a \
        --pred1 results/exp_A/OOD_7x7_predictions.json \
        --pred2 results/exp_C/OOD_7x7_predictions.json \
        --data data/1_goals_test_unseen_7x7_samples.json \
        --label1 "Vanilla SFT (greedy)" \
        --label2 "Multi-scale SFT (greedy)" \
        --output visualizations/case_a.pdf

    python visualize_cases.py \
        --case b \
        --pred1 results/exp_A/OOD_6x6_dense_predictions.json \
        --pred2 results/exp_cot/OOD_6x6_dense_predictions.json \
        --data data/1_goals_test_unseen_6x6more_obstacles_samples.json \
        --label1 "Vanilla SFT (greedy)" \
        --label2 "CoT SFT (greedy)" \
        --output visualizations/case_b.pdf

    python visualize_cases.py \
        --case c \
        --pred1 results/exp_C/OOD_novel_predictions.json \
        --pred2 results/exp_G/OOD_novel_predictions.json \
        --data data/ood_novel_sizes.json \
        --label1 "Multi-scale (greedy)" \
        --label2 "Multi-scale + Tree Search" \
        --target_size 14 \
        --output visualizations/case_c.pdf
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from evaluate_utils import simulate_path, find_position, parse_actions
from data_utils import parse_nl_description


# NeurIPS style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
})

# Colors
C_EMPTY    = '#F5F5F5'
C_OBSTACLE = '#2c3e50'
C_START    = '#228833'
C_GOAL     = '#CC3311'
C_GT_PATH  = '#4477AA'
C_PRED_OK  = '#228833'
C_PRED_FAIL = '#EE7733'
C_COLLISION = '#CC3311'


def draw_grid(ax, grid, predicted_path, gt_path, status, title,
              show_legend=False):
    """Draw a single grid with paths."""
    rows, cols = len(grid), len(grid[0])

    # Draw cells
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                color = C_OBSTACLE
            elif grid[r][c] == 2:
                color = C_START
            elif grid[r][c] == 3:
                color = C_GOAL
            else:
                color = C_EMPTY

            rect = plt.Rectangle((c, rows - 1 - r), 1, 1,
                                 linewidth=0.4, edgecolor='#BBBBBB',
                                 facecolor=color)
            ax.add_patch(rect)

            if grid[r][c] == 2:
                ax.text(c + 0.5, rows - 0.5 - r, 'S', ha='center',
                        va='center', fontsize=8, fontweight='bold',
                        color='white')
            elif grid[r][c] == 3:
                ax.text(c + 0.5, rows - 0.5 - r, 'G', ha='center',
                        va='center', fontsize=8, fontweight='bold',
                        color='white')

    # Ground truth path
    if gt_path and len(gt_path) > 1:
        xs = [c + 0.5 for r, c in gt_path]
        ys = [rows - 0.5 - r for r, c in gt_path]
        ax.plot(xs, ys, 'o-', color=C_GT_PATH, linewidth=1.5,
                markersize=3.5, alpha=0.4, label='Ground truth',
                zorder=2)

    # Predicted path
    if predicted_path and len(predicted_path) > 1:
        pred_color = C_PRED_OK if status == 'success' else C_PRED_FAIL
        xs = [c + 0.5 for r, c in predicted_path]
        ys = [rows - 0.5 - r for r, c in predicted_path]
        ax.plot(xs, ys, 's--', color=pred_color, linewidth=1.5,
                markersize=3.5, alpha=0.85, label='Predicted',
                zorder=3)

        # Mark collision point
        if status == 'obstacle' and len(predicted_path) > 0:
            last = predicted_path[-1]
            ax.plot(last[1] + 0.5, rows - 0.5 - last[0], 'x',
                    color=C_COLLISION, markersize=10, markeredgewidth=2.5,
                    zorder=4)

        # Mark end point for wrong_end
        if status == 'wrong_end' and len(predicted_path) > 0:
            last = predicted_path[-1]
            ax.plot(last[1] + 0.5, rows - 0.5 - last[0], 'x',
                    color=C_PRED_FAIL, markersize=8, markeredgewidth=2,
                    zorder=4)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')

    # Status indicator in title
    status_str = {
        'success': '(Success)',
        'obstacle': '(Obstacle collision)',
        'wrong_end': '(Wrong end)',
        'out_of_bounds': '(Out of bounds)',
    }.get(status, f'({status})')
    color = C_PRED_OK if status == 'success' else C_COLLISION
    ax.set_title(f'{title}\n{status_str}', fontsize=8, color=color,
                 fontweight='bold')

    ax.set_xticks([])
    ax.set_yticks([])

    if show_legend:
        ax.legend(fontsize=6, loc='upper right', framealpha=0.8)


def find_matching_case(pred1_list, pred2_list, data_list,
                       error_type_1, error_type_2=None,
                       target_size=None, max_search=500):
    """
    Find a sample where pred1 has error_type_1 and pred2 has error_type_2.
    Returns (sample, pred1_entry, pred2_entry) or None.
    """
    if error_type_2 is None:
        error_type_2 = 'none'  # success

    # Build lookup by nl_description
    pred2_lookup = {p['nl_description']: p for p in pred2_list}

    # Filter data for matching
    data_lookup = {}
    for s in data_list:
        if 'Goal not reachable' in s.get('agent_as_a_point', ''):
            continue
        data_lookup[s['nl_description']] = s

    count = 0
    for p1 in pred1_list:
        if count >= max_search:
            break
        count += 1

        nl = p1['nl_description']
        if nl not in pred2_lookup or nl not in data_lookup:
            continue

        p2 = pred2_lookup[nl]
        sample = data_lookup[nl]

        # Check target size
        if target_size is not None:
            info = parse_nl_description(nl)
            if info['grid_size'][0] != target_size:
                continue

        # Check error types
        if p1.get('error_type') == error_type_1 and \
           p2.get('error_type') == error_type_2:
            return sample, p1, p2

    return None


def generate_comparison(sample, p1, p2, label1, label2, save_path,
                        case_label=""):
    """Generate a side-by-side comparison figure."""
    grid = sample['world']
    start = find_position(grid, 2)
    rows, cols = len(grid), len(grid[0])

    # Simulate both paths
    sim1 = simulate_path(grid, start, p1['predicted'])
    sim2 = simulate_path(grid, start, p2['predicted'])

    # Ground truth path
    gt_path = [tuple(c) for c in sample['solution_coordinates']]

    # Figure size scales with grid
    cell_size = max(0.4, min(0.7, 5.0 / cols))
    fig_w = cols * cell_size * 2 + 1.5
    fig_h = rows * cell_size + 0.8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    draw_grid(ax1, grid, sim1['positions'], gt_path,
              sim1['status'], label1, show_legend=True)
    draw_grid(ax2, grid, sim2['positions'], gt_path,
              sim2['status'], label2)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    print(f"    {label1}: {sim1['status']} "
          f"(path len {len(sim1['positions'])-1})")
    print(f"    {label2}: {sim2['status']} "
          f"(path len {len(sim2['positions'])-1})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, required=True,
                        choices=['a', 'b', 'c'],
                        help='Which case to generate')
    parser.add_argument('--pred1', type=str, required=True)
    parser.add_argument('--pred2', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--label1', type=str, default='Method 1')
    parser.add_argument('--label2', type=str, default='Method 2')
    parser.add_argument('--target_size', type=int, default=None)
    parser.add_argument('--output', type=str, default='visualizations/case.pdf')
    parser.add_argument('--n_cases', type=int, default=3,
                        help='Number of cases to generate')
    args = parser.parse_args()

    with open(args.pred1) as f:
        pred1 = json.load(f)
    with open(args.pred2) as f:
        pred2 = json.load(f)
    with open(args.data) as f:
        data = json.load(f)

    # Define what to look for per case
    case_specs = {
        'a': ('wrong_end', 'none',     'Path length memorization'),
        'b': ('obstacle',  'none',     'Dense obstacle failure'),
        'c': ('obstacle',  'none',     'Collision scaling vs tree search'),
    }

    err1, err2, case_title = case_specs[args.case]
    print(f"\nCase ({args.case}): {case_title}")
    print(f"  Looking for: pred1={err1}, pred2={err2}")
    if args.target_size:
        print(f"  Target grid size: {args.target_size}x{args.target_size}")

    found = 0
    # Try multiple times with shuffled order
    import random
    random.seed(42)
    shuffled_pred1 = list(pred1)
    random.shuffle(shuffled_pred1)

    pred2_lookup = {p['nl_description']: p for p in pred2}
    data_lookup = {}
    for s in data:
        if 'Goal not reachable' not in s.get('agent_as_a_point', ''):
            data_lookup[s['nl_description']] = s

    for p1 in shuffled_pred1:
        if found >= args.n_cases:
            break

        nl = p1['nl_description']
        if nl not in pred2_lookup or nl not in data_lookup:
            continue

        p2 = pred2_lookup[nl]
        sample = data_lookup[nl]

        if args.target_size:
            info = parse_nl_description(nl)
            if info['grid_size'][0] != args.target_size:
                continue

        if p1.get('error_type') == err1 and p2.get('error_type') == err2:
            out_path = args.output.replace('.pdf', f'_{found}.pdf')
            out_path_png = out_path.replace('.pdf', '.png')
            generate_comparison(sample, p1, p2, args.label1, args.label2,
                                out_path, case_title)
            # Also save PNG
            generate_comparison(sample, p1, p2, args.label1, args.label2,
                                out_path_png, case_title)
            found += 1

    if found == 0:
        print(f"  No matching cases found! Try different error type combo.")
        # Show available error types
        from collections import Counter
        e1 = Counter(p.get('error_type') for p in pred1)
        e2 = Counter(p.get('error_type') for p in pred2)
        print(f"  pred1 errors: {dict(e1)}")
        print(f"  pred2 errors: {dict(e2)}")
    else:
        print(f"\n  Generated {found} comparison figures.")


if __name__ == '__main__':
    main()