"""
Reward function for GRPO training on PPNL path planning.

Reward design principles:
- Format errors are strongly penalized (-1.0)
- Invalid paths (out of bounds, obstacle) get moderate penalty (-0.5)
- Legal but non-goal paths get distance-based shaping (0.0 ~ 0.2)
- Successful paths get high reward (0.8 for non-optimal, 1.0 for optimal)
"""
from typing import List, Tuple
from evaluate_utils import (
    simulate_path, find_position, a_star_distance, parse_actions
)


def compute_reward(grid: List[List[int]],
                   predicted_actions: str,
                   ground_truth_actions: str = None) -> dict:
    """
    Compute reward for a predicted action sequence.

    Args:
        grid: 2D grid (0=empty, 1=obstacle, 2=start, 3=goal)
        predicted_actions: action sequence string
        ground_truth_actions: optional GT sequence, used for optimality bonus

    Returns:
        dict with keys:
            - 'reward': scalar reward value in [-1.0, 1.0]
            - 'status': executor status
            - 'info': diagnostic info (distance, path_length, etc.)
    """
    start = find_position(grid, 2)
    goal = find_position(grid, 3)

    if start is None or goal is None:
        return {'reward': -1.0, 'status': 'invalid_grid', 'info': {}}

    sim = simulate_path(grid, start, predicted_actions)
    status = sim['status']
    pred_len = len(parse_actions(predicted_actions))

    info = {'path_length': pred_len, 'status': status}

    if status == 'format_error':
        return {'reward': -1.0, 'status': status, 'info': info}

    if status == 'out_of_bounds':
        return {'reward': -0.5, 'status': status, 'info': info}

    if status == 'obstacle':
        return {'reward': -0.5, 'status': status, 'info': info}

    if status == 'wrong_end':
        # Legal path but did not reach goal.
        # Shape reward by remaining distance: closer to goal → higher reward.
        dist = a_star_distance(grid, sim['final_pos'], goal)
        if dist < 0:
            # goal unreachable from final pos (shouldn't happen in valid grids)
            return {'reward': 0.0, 'status': status, 'info': info}
        max_dist = len(grid) + len(grid[0])  # rough normalization
        shaped = 0.2 * max(0.0, 1.0 - dist / max_dist)
        info['distance_to_goal'] = dist
        return {'reward': shaped, 'status': status, 'info': info}

    if status == 'success':
        # Reached the goal. Check optimality.
        if ground_truth_actions is not None:
            gt_len = len(parse_actions(ground_truth_actions))
            info['gt_length'] = gt_len
            if pred_len <= gt_len:
                return {'reward': 1.0, 'status': 'success_optimal', 'info': info}
            else:
                return {'reward': 0.8, 'status': 'success_nonoptimal', 'info': info}
        else:
            # No GT available; can't check optimality. Give moderate reward.
            return {'reward': 0.9, 'status': 'success', 'info': info}

    # Unknown status
    return {'reward': 0.0, 'status': status, 'info': info}


if __name__ == '__main__':
    # Self-test
    import json
    with open('data/1_train_set_6x6_samples.json') as f:
        data = json.load(f)

    sample = data[0]
    grid = sample['world']
    gt = sample['agent_as_a_point']

    # Test 1: GT should give reward 1.0
    r = compute_reward(grid, gt, gt)
    print(f"GT reward: {r}")
    assert r['reward'] == 1.0

    # Test 2: Empty actions should give shaped reward based on start-to-goal distance
    r = compute_reward(grid, "", gt)
    print(f"Empty actions reward: {r}")

    # Test 3: Out-of-bounds should give -0.5
    r = compute_reward(grid, "up up up up up up", gt)
    print(f"OOB reward: {r}")
    assert r['reward'] == -0.5

    # Test 4: Non-optimal successful path (GT + extra moves that revisit)
    # Can't easily construct this without knowing grid; skip

    # Test 5: Format error
    r = compute_reward(grid, "blah foo bar", gt)
    print(f"Format error reward: {r}")
    assert r['reward'] == -1.0

    print("\n✓ All reward tests passed")