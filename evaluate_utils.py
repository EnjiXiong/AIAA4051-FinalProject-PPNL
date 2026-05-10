"""
Evaluation utilities for single-goal path planning.
Extracted and cleaned from PPNL benchmark executor-point-sg.py
"""
import json
import heapq
from typing import List, Tuple, Dict, Optional


def parse_actions(actions: str) -> List[str]:
    """Parse action string into list of individual actions."""
    acts = ['up', 'down', 'left', 'right']
    # Fix concatenated actions (e.g., "upleft" -> "up left")
    for i in range(len(acts)):
        for j in range(len(acts)):
            actions = actions.replace(acts[i] + acts[j], acts[i] + ' ' + acts[j])
    return [a for a in actions.strip().split(' ') if a]


def simulate_path(grid: List[List[int]], start: Tuple[int, int],
                  actions: str) -> Dict:
    """
    Simulate an action sequence on the grid.
    
    Returns dict with:
        - 'positions': list of (row, col) visited
        - 'status': 'success' | 'out_of_bounds' | 'obstacle' | 'wrong_end' | 'format_error'
        - 'final_pos': final position
        - 'error_step': step index where error occurred (if any)
    """
    nx, ny = len(grid), len(grid[0])
    pos = start
    positions = [pos]
    sequence = parse_actions(actions)
    
    action_map = {
        'left': (0, -1), 'right': (0, 1),
        'up': (-1, 0), 'down': (1, 0)
    }
    
    for step_idx, action in enumerate(sequence):
        if action not in action_map:
            return {
                'positions': positions, 'status': 'format_error',
                'final_pos': pos, 'error_step': step_idx
            }
        
        dx, dy = action_map[action]
        new_pos = (pos[0] + dx, pos[1] + dy)
        
        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= nx or new_pos[1] >= ny:
            return {
                'positions': positions, 'status': 'out_of_bounds',
                'final_pos': pos, 'error_step': step_idx
            }
        
        if grid[new_pos[0]][new_pos[1]] == 1:
            return {
                'positions': positions + [new_pos], 'status': 'obstacle',
                'final_pos': new_pos, 'error_step': step_idx
            }
        
        pos = new_pos
        positions.append(pos)
    
    if grid[pos[0]][pos[1]] == 3:
        return {
            'positions': positions, 'status': 'success',
            'final_pos': pos, 'error_step': None
        }
    else:
        return {
            'positions': positions, 'status': 'wrong_end',
            'final_pos': pos, 'error_step': None
        }


def a_star_distance(grid: List[List[int]], start: Tuple[int, int],
                    goal: Tuple[int, int]) -> int:
    """Compute shortest path length from start to goal using A*."""
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    visited = set()
    heap = [(0, 0, start)]  # (f_score, g_score, position)
    counter = 1
    
    while heap:
        f, g, current = heapq.heappop(heap)
        
        if current == goal:
            return g
        
        if current in visited:
            continue
        visited.add(current)
        
        for dx, dy in actions:
            nb = (current[0] + dx, current[1] + dy)
            if (0 <= nb[0] < len(grid) and 0 <= nb[1] < len(grid[0])
                    and grid[nb[0]][nb[1]] != 1 and nb not in visited):
                new_g = g + 1
                heapq.heappush(heap, (new_g + heuristic(nb), new_g, nb))
                counter += 1
    
    return -1  # unreachable


def find_position(grid: List[List[int]], value: int) -> Optional[Tuple[int, int]]:
    """Find position of a value in the grid."""
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == value:
                return (i, j)
    return None


def evaluate_single(grid: List[List[int]], predicted: str,
                    ground_truth: str) -> Dict:
    """
    Evaluate a single prediction.
    
    Returns:
        Dict with keys: success, feasible, optimal, exact_match,
                        distance_to_goal, error_type, path_length
    """
    start = find_position(grid, 2)
    goal = find_position(grid, 3)
    
    if start is None or goal is None:
        return {'success': False, 'feasible': False, 'optimal': False,
                'exact_match': False, 'distance_to_goal': -1,
                'error_type': 'invalid_grid', 'path_length': 0}
    
    sim = simulate_path(grid, start, predicted)
    
    success = sim['status'] == 'success'
    feasible = sim['status'] in ('success', 'wrong_end')  # stayed in bounds, no obstacles
    exact_match = predicted.strip().replace(' ', '') == ground_truth.strip().replace(' ', '')
    
    # Check optimality
    gt_actions = parse_actions(ground_truth)
    pred_actions = parse_actions(predicted)
    optimal = success and len(pred_actions) <= len(gt_actions)
    
    # Distance to goal
    if feasible:
        dist = a_star_distance(grid, sim['final_pos'], goal)
    else:
        dist = -1
    
    return {
        'success': success,
        'feasible': feasible,
        'optimal': optimal,
        'exact_match': exact_match,
        'distance_to_goal': dist,
        'error_type': sim['status'] if not success else 'none',
        'path_length': len(pred_actions),
        'predicted_path': sim['positions'],
    }


def evaluate_batch(data: List[Dict], predictions: List[str]) -> Dict:
    """
    Evaluate a batch of predictions.
    
    Args:
        data: list of sample dicts (with 'world', 'agent_as_a_point' keys)
        predictions: list of predicted action strings
    
    Returns:
        Dict with aggregated metrics and per-sample results
    """
    assert len(data) == len(predictions), \
        f"Data ({len(data)}) and predictions ({len(predictions)}) length mismatch"
    
    results = []
    for sample, pred in zip(data, predictions):
        gt = sample['agent_as_a_point']
        if 'Goal not reachable' in gt:
            continue
        r = evaluate_single(sample['world'], pred, gt)
        results.append(r)
    
    n = len(results)
    if n == 0:
        return {'metrics': {}, 'per_sample': []}
    
    n_feasible_with_dist = sum(1 for r in results if r['feasible'] and r['distance_to_goal'] >= 0)
    
    metrics = {
        'total': n,
        'success_rate': sum(r['success'] for r in results) / n,
        'feasibility': sum(r['feasible'] for r in results) / n,
        'optimality': sum(r['optimal'] for r in results) / n,
        'exact_match': sum(r['exact_match'] for r in results) / n,
        'avg_distance_to_goal': (
            sum(r['distance_to_goal'] for r in results if r['feasible'] and r['distance_to_goal'] >= 0)
            / max(n_feasible_with_dist, 1)
        ),
    }
    
    # Error type distribution
    error_counts = {}
    for r in results:
        et = r['error_type']
        error_counts[et] = error_counts.get(et, 0) + 1
    metrics['error_distribution'] = error_counts
    
    return {'metrics': metrics, 'per_sample': results}


if __name__ == '__main__':
    # Quick self-test
    import sys
    
    with open('data/1_train_set_6x6_samples.json') as f:
        train = json.load(f)
    
    # Test with ground truth (should be 100% success)
    subset = train[:100]
    preds = [s['agent_as_a_point'] for s in subset]
    result = evaluate_batch(subset, preds)
    
    print("Self-test with ground truth predictions:")
    for k, v in result['metrics'].items():
        if k != 'error_distribution':
            print(f"  {k}: {v}")
    print(f"  errors: {result['metrics']['error_distribution']}")
