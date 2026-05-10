"""
Generate diverse grid environments for RL training.

Creates environments across multiple grid sizes with varying obstacle counts.
For each environment, computes the A* optimal solution so the reward function
can check optimality.

Usage:
    python generate_rl_envs.py --output data/rl_envs_diverse.json

    # Custom sizes
    python generate_rl_envs.py --sizes 5 6 7 8 10 12 --envs_per_size 500
"""
import argparse
import heapq
import json
import random
from typing import List, Tuple, Optional, Dict


def generate_grid(dim: int, num_obstacles: int, seed: int = None) -> Optional[Dict]:
    """
    Generate a random grid with obstacles, start, and goal positions.
    Ensures a valid path exists from start to goal.

    Returns None if no valid configuration found after retries.
    """
    if seed is not None:
        random.seed(seed)

    max_retries = 50
    for _ in range(max_retries):
        # Place obstacles
        all_cells = [(r, c) for r in range(dim) for c in range(dim)]
        if num_obstacles >= len(all_cells) - 2:
            num_obstacles = max(0, len(all_cells) - 10)  # leave room

        obstacle_cells = set()
        candidates = all_cells.copy()
        random.shuffle(candidates)
        for cell in candidates:
            if len(obstacle_cells) >= num_obstacles:
                break
            obstacle_cells.add(cell)

        # Place start and goal on non-obstacle cells
        free_cells = [c for c in all_cells if c not in obstacle_cells]
        if len(free_cells) < 2:
            continue

        random.shuffle(free_cells)
        start = free_cells[0]
        goal = free_cells[1]

        # Build grid
        grid = [[0] * dim for _ in range(dim)]
        for (r, c) in obstacle_cells:
            grid[r][c] = 1
        grid[start[0]][start[1]] = 2
        grid[goal[0]][goal[1]] = 3

        # Check path exists
        path = a_star_path(grid, start, goal)
        if path is not None:
            # Convert path to actions
            actions = path_to_actions(path)
            # Build NL description
            obs_list = sorted(obstacle_cells)
            nl = build_nl_description(dim, obs_list, start, goal)

            return {
                'world': grid,
                'nl_description': nl,
                'solution_coordinates': [list(p) for p in path],
                'agent_as_a_point': ' '.join(actions) + ' ',
                'grid_size': dim,
                'num_obstacles': num_obstacles,
            }

    return None  # failed to generate valid env


def a_star_path(grid: List[List[int]], start: Tuple[int, int],
                goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """Find shortest path using A*. Returns list of positions or None."""
    dim = len(grid)
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    visited = set()
    heap = [(0, 0, start, [start])]  # (f, g, pos, path)
    counter = 1

    while heap:
        f, g, current, path = heapq.heappop(heap)

        if current == goal:
            return path

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in actions:
            nb = (current[0] + dx, current[1] + dy)
            if (0 <= nb[0] < dim and 0 <= nb[1] < dim
                    and grid[nb[0]][nb[1]] != 1 and nb not in visited):
                new_g = g + 1
                heapq.heappush(
                    heap,
                    (new_g + heuristic(nb), new_g, nb, path + [nb])
                )
                counter += 1

    return None  # no path


def path_to_actions(path: List[Tuple[int, int]]) -> List[str]:
    """Convert coordinate path to action sequence."""
    actions = []
    action_map = {
        (-1, 0): 'up', (1, 0): 'down',
        (0, -1): 'left', (0, 1): 'right'
    }
    for i in range(len(path) - 1):
        dr = path[i + 1][0] - path[i][0]
        dc = path[i + 1][1] - path[i][1]
        actions.append(action_map[(dr, dc)])
    return actions


def build_nl_description(dim: int, obstacles: List[Tuple[int, int]],
                         start: Tuple[int, int],
                         goal: Tuple[int, int]) -> str:
    """Build natural language description matching PPNL format."""
    desc = f"You are in a {dim} by {dim} world."
    if obstacles:
        obs_str = ", ".join(f"({r},{c})" for r, c in obstacles)
        desc += f" There are obstacles that you have to avoid at: {obs_str}."
    desc += f" Go from ({start[0]},{start[1]}) to ({goal[0]},{goal[1]})"
    return desc


def get_obstacle_range(dim: int) -> Tuple[int, int]:
    """
    Determine reasonable obstacle count range for a grid size.
    Rule: 5-25% of total cells, with at least 1.
    """
    total = dim * dim
    min_obs = max(1, total // 20)     # 5%
    max_obs = max(2, total // 4)      # 25%
    return min_obs, max_obs


def main():
    parser = argparse.ArgumentParser(
        description='Generate diverse RL training environments')
    parser.add_argument('--sizes', nargs='+', type=int,
                        default=[5, 6, 7, 8, 10, 12],
                        help='Grid sizes to generate')
    parser.add_argument('--envs_per_size', type=int, default=500,
                        help='Environments per grid size')
    parser.add_argument('--output', type=str,
                        default='data/rl_envs_diverse.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    all_envs = []

    for dim in args.sizes:
        min_obs, max_obs = get_obstacle_range(dim)
        generated = 0
        attempts = 0
        max_attempts = args.envs_per_size * 5

        print(f"Generating {args.envs_per_size} environments for "
              f"{dim}x{dim} (obstacles: {min_obs}-{max_obs})...")

        while generated < args.envs_per_size and attempts < max_attempts:
            num_obs = random.randint(min_obs, max_obs)
            env = generate_grid(dim, num_obs)
            if env is not None:
                all_envs.append(env)
                generated += 1
            attempts += 1

        if generated < args.envs_per_size:
            print(f"  Warning: only generated {generated}/{args.envs_per_size}")

        # Stats
        subset = [e for e in all_envs if e['grid_size'] == dim]
        avg_path = sum(len(e['agent_as_a_point'].strip().split())
                       for e in subset) / max(len(subset), 1)
        print(f"  Generated: {len(subset)}, avg path length: {avg_path:.1f}")

    # Shuffle so different sizes are interleaved during training
    random.shuffle(all_envs)

    with open(args.output, 'w') as f:
        json.dump(all_envs, f)

    print(f"\nTotal: {len(all_envs)} environments saved to {args.output}")

    # Summary
    print("\nSummary:")
    print(f"{'Size':<8} {'Count':<8} {'Avg Path':<10} {'Obs Range'}")
    print('-' * 40)
    for dim in args.sizes:
        subset = [e for e in all_envs if e['grid_size'] == dim]
        if subset:
            avg_path = sum(len(e['agent_as_a_point'].strip().split())
                           for e in subset) / len(subset)
            obs_counts = [e['num_obstacles'] for e in subset]
            print(f"{dim}x{dim:<5} {len(subset):<8} {avg_path:<10.1f} "
                  f"{min(obs_counts)}-{max(obs_counts)}")


if __name__ == '__main__':
    main()