"""
Visualization utilities for PPNL path planning experiments.
Generates grid visualizations, error analysis plots, and result comparisons.

Usage:
    python visualize.py --predictions results/t5-small_vanilla/ID_seen_6x6_predictions.json
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from evaluate_utils import simulate_path, find_position, parse_actions


# ─── Grid Visualization ─────────────────────────────────────────────────────

def plot_grid_path(grid: List[List[int]], predicted_path: List[tuple],
                   gt_path: Optional[List[tuple]] = None,
                   title: str = "", save_path: Optional[str] = None):
    """
    Visualize a grid with predicted and ground truth paths.
    
    Args:
        grid: 2D grid (0=empty, 1=obstacle, 2=start, 3=goal)
        predicted_path: list of (row, col) positions
        gt_path: optional ground truth path
        title: plot title
        save_path: path to save figure
    """
    rows, cols = len(grid), len(grid[0])
    fig, ax = plt.subplots(1, 1, figsize=(cols + 1, rows + 1))
    
    # Draw grid
    for r in range(rows):
        for c in range(cols):
            color = 'white'
            if grid[r][c] == 1:
                color = '#2c3e50'  # obstacle
            elif grid[r][c] == 2:
                color = '#27ae60'  # start
            elif grid[r][c] == 3:
                color = '#e74c3c'  # goal
            
            rect = plt.Rectangle((c, rows - 1 - r), 1, 1,
                                linewidth=1, edgecolor='gray',
                                facecolor=color)
            ax.add_patch(rect)
            
            # Cell labels
            if grid[r][c] == 2:
                ax.text(c + 0.5, rows - 0.5 - r, 'S', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white')
            elif grid[r][c] == 3:
                ax.text(c + 0.5, rows - 0.5 - r, 'G', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white')
            elif grid[r][c] == 1:
                ax.text(c + 0.5, rows - 0.5 - r, 'X', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white')
    
    # Draw ground truth path
    if gt_path and len(gt_path) > 1:
        gt_xs = [c + 0.5 for r, c in gt_path]
        gt_ys = [rows - 0.5 - r for r, c in gt_path]
        ax.plot(gt_xs, gt_ys, 'o-', color='#3498db', linewidth=2,
               markersize=6, alpha=0.5, label='Ground Truth')
    
    # Draw predicted path
    if predicted_path and len(predicted_path) > 1:
        pred_xs = [c + 0.5 for r, c in predicted_path]
        pred_ys = [rows - 0.5 - r for r, c in predicted_path]
        ax.plot(pred_xs, pred_ys, 's--', color='#e67e22', linewidth=2,
               markersize=6, alpha=0.8, label='Predicted')
    
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    
    # Grid coordinates
    ax.set_xticks(range(cols + 1))
    ax.set_yticks(range(rows + 1))
    ax.set_xticklabels(range(cols + 1), fontsize=8)
    ax.set_yticklabels(list(range(rows, -1, -1)), fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_failure_cases(predictions_file: str, data_file: str,
                       output_dir: str, max_cases: int = 3):
    """
    Visualize representative failure cases by error type.
    """
    with open(predictions_file) as f:
        preds = json.load(f)
    with open(data_file) as f:
        data = json.load(f)
    
    data_filtered = [s for s in data
                     if 'Goal not reachable' not in s.get('agent_as_a_point', '')]
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by error type
    errors_by_type = {}
    for i, p in enumerate(preds):
        if p.get('error_type', 'none') != 'none':
            et = p['error_type']
            if et not in errors_by_type:
                errors_by_type[et] = []
            errors_by_type[et].append(i)
    
    for error_type, indices in errors_by_type.items():
        for case_num, idx in enumerate(indices[:max_cases]):
            sample = data_filtered[idx]
            pred_text = preds[idx]['predicted']
            gt_text = sample['agent_as_a_point']
            
            grid = sample['world']
            start = find_position(grid, 2)
            
            # Simulate predicted path
            sim = simulate_path(grid, start, pred_text)
            pred_positions = sim['positions']
            
            # Ground truth path
            gt_positions = [tuple(c) for c in sample['solution_coordinates']]
            
            title = (f"{error_type} | Pred: \"{pred_text.strip()[:40]}\" | "
                    f"GT: \"{gt_text.strip()[:40]}\"")
            
            save_path = out_dir / f"failure_{error_type}_{case_num}.png"
            plot_grid_path(grid, pred_positions, gt_positions,
                          title=title, save_path=str(save_path))
    
    print(f"Failure visualizations saved to {out_dir}")


# ─── Result Comparison Plots ────────────────────────────────────────────────

def plot_results_comparison(results_dict: Dict[str, Dict[str, Dict]],
                            save_path: str = 'visualizations/results_comparison.png'):
    """
    Create grouped bar chart comparing models across test sets.
    
    Args:
        results_dict: {model_name: {test_set: {metric: value}}}
    """
    metrics = ['success_rate', 'feasibility', 'optimality']
    metric_labels = ['Success Rate', 'Feasibility', 'Optimality']
    
    # Collect all test sets
    all_test_sets = set()
    for model_results in results_dict.values():
        all_test_sets.update(model_results.keys())
    test_sets = sorted(all_test_sets)
    
    models = list(results_dict.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        x = np.arange(len(test_sets))
        width = 0.8 / n_models
        
        for i, model in enumerate(models):
            values = []
            for ts in test_sets:
                if ts in results_dict[model]:
                    values.append(results_dict[model][ts].get(metric, 0))
                else:
                    values.append(0)
            
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model,
                         color=colors[i], edgecolor='white')
            
            # Value labels on bars
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=7)
        
        ax.set_xlabel('Test Set')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(test_sets, rotation=30, ha='right', fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {save_path}")


def plot_error_distribution(predictions_file: str,
                            save_path: str = 'visualizations/error_dist.png'):
    """Plot error type distribution as a bar chart."""
    with open(predictions_file) as f:
        preds = json.load(f)
    
    error_counts = {}
    for p in preds:
        et = p.get('error_type', 'none')
        error_counts[et] = error_counts.get(et, 0) + 1
    
    fig, ax = plt.subplots(figsize=(8, 4))
    types = list(error_counts.keys())
    counts = [error_counts[t] for t in types]
    colors_map = {
        'none': '#27ae60', 'out_of_bounds': '#e74c3c',
        'obstacle': '#f39c12', 'wrong_end': '#3498db',
        'format_error': '#9b59b6'
    }
    bar_colors = [colors_map.get(t, '#95a5a6') for t in types]
    
    ax.bar(types, counts, color=bar_colors, edgecolor='white')
    ax.set_ylabel('Count')
    ax.set_title('Error Type Distribution')
    
    for i, (t, c) in enumerate(zip(types, counts)):
        pct = c / sum(counts) * 100
        ax.text(i, c + max(counts) * 0.02, f'{pct:.1f}%',
               ha='center', fontsize=9)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error distribution plot saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, default=None,
                        help='Path to predictions JSON for failure analysis')
    parser.add_argument('--data', type=str,
                        default='data/1_goals_test_seen_6x6_samples.json')
    parser.add_argument('--output_dir', type=str, default='visualizations/')
    args = parser.parse_args()
    
    # Demo: visualize a grid with GT path
    data = json.loads(open(args.data).read())
    sample = data[5]
    grid = sample['world']
    gt_path = [tuple(c) for c in sample['solution_coordinates']]
    
    plot_grid_path(
        grid, gt_path, title="Example: Ground Truth Path",
        save_path=f'{args.output_dir}/example_gt_path.png'
    )
    print("Demo visualization saved.")
