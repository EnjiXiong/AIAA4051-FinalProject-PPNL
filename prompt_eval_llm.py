"""
LLM Prompting evaluation for grid path planning using DeepSeek API.
Tests whether a frontier LLM can solve path planning via prompting alone.

Prerequisites:
    pip install openai

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    python prompt_eval_llm.py --sample_size 200
    python prompt_eval_llm.py --sample_size 50 --strategies zero_shot  # quick test

Output:
    results/llm_prompting/eval_summary.json
    results/llm_prompting/{strategy}_{dataset}_predictions.json
"""
import argparse
import json
import os
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple

from openai import OpenAI

from data_utils import load_ppnl_data, parse_nl_description
from evaluate_utils import evaluate_batch

# ─── Prompt templates ────────────────────────────────────────────────────────

ZERO_SHOT_PROMPT = """You are a path planning agent in a grid world.
Given a grid world description, output ONLY the shortest sequence of actions (up, down, left, right) to navigate from the start to the goal while avoiding obstacles.

Rules:
- Actions move the agent one cell: up (row-1), down (row+1), left (col-1), right (col+1)
- You cannot move through obstacles or outside the grid boundaries
- Output ONLY the action sequence, separated by spaces. No explanation.

Task: {nl_description}

Actions:"""

FEW_SHOT_PROMPT = """You are a path planning agent in a grid world.
Given a grid world description, output the shortest sequence of actions (up, down, left, right) to navigate from the start to the goal while avoiding obstacles.

Example 1:
Task: You are in a 4 by 4 world. There are no obstacles. Go from (0,0) to (2,3)
Actions: right right right down down

Example 2:
Task: You are in a 5 by 5 world. There are obstacles at: (1,1), (2,2). Go from (0,0) to (3,3)
Actions: right right right down down down

Example 3:
Task: You are in a 4 by 4 world. There are obstacles at: (1,0), (1,1). Go from (0,0) to (2,0)
Actions: right right down down left left

Now solve this task. Output ONLY the action sequence, no explanation.

Task: {nl_description}

Actions:"""

COT_PROMPT = """You are a path planning agent in a grid world.
Plan a path step by step, tracking your position after each move. Avoid obstacles and stay within bounds.

Example:
Task: You are in a 4 by 4 world. There are no obstacles. Go from (0,0) to (1,2)
Reasoning: I am at (0,0). Goal is (1,2). I need to go right 2 and down 1.
Step 1: right -> (0,1)
Step 2: right -> (0,2)
Step 3: down -> (1,2). Reached goal!
Actions: right right down

Now solve this task. First reason step by step tracking your coordinates, then output the final action sequence on a line starting with "Actions:".

Task: {nl_description}

Reasoning:"""


STRATEGIES = {
    'zero_shot': ZERO_SHOT_PROMPT,
    'few_shot': FEW_SHOT_PROMPT,
    'cot': COT_PROMPT,
}

# ─── Test sets ───────────────────────────────────────────────────────────────

TEST_SETS = {
    'ID_seen_6x6': 'data/1_goals_test_seen_6x6_samples.json',
    'OOD_7x7': 'data/1_goals_test_unseen_7x7_samples.json',
}

# Add OOD_novel if available
OPTIONAL_TESTS = {
    'OOD_novel': 'data/ood_novel_sizes.json',
}

# ─── API call ────────────────────────────────────────────────────────────────

def call_deepseek(client: OpenAI, prompt: str, model: str,
                  max_retries: int = 3) -> str:
    """Call DeepSeek API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    API failed after {max_retries} attempts: {e}")
                return ""


def extract_actions(response: str, strategy: str) -> str:
    """Extract the action sequence from LLM response."""
    valid_actions = {'up', 'down', 'left', 'right'}

    if strategy == 'cot':
        # Look for "Actions:" line
        for line in response.split('\n'):
            if line.strip().lower().startswith('actions:'):
                action_part = line.split(':', 1)[1].strip()
                tokens = action_part.split()
                actions = [t for t in tokens if t.lower() in valid_actions]
                if actions:
                    return ' '.join(a.lower() for a in actions)

    # Fallback: extract all valid action tokens from response
    tokens = response.replace(',', ' ').replace('.', ' ').split()
    actions = [t.lower() for t in tokens if t.lower() in valid_actions]
    return ' '.join(actions)


# ─── Main evaluation ────────────────────────────────────────────────────────

def run_eval(args):
    # Setup API client
    api_key = args.api_key or os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError(
            "Provide API key via --api_key or DEEPSEEK_API_KEY env var")

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
    )
    model = args.model
    print(f"Using model: {model}")
    print(f"API base: {args.base_url}")

    # Build test sets
    test_sets = {}
    for name, path in TEST_SETS.items():
        if os.path.exists(path):
            test_sets[name] = path
    for name, path in OPTIONAL_TESTS.items():
        if os.path.exists(path):
            test_sets[name] = path

    if args.test_sets:
        test_sets = {k: v for k, v in test_sets.items() if k in args.test_sets}

    strategies = args.strategies if args.strategies else list(STRATEGIES.keys())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for strategy in strategies:
        template = STRATEGIES[strategy]
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")

        for ds_name, ds_path in test_sets.items():
            print(f"\n  Dataset: {ds_name}")
            data = load_ppnl_data(ds_path)

            # Filter unreachable
            filtered = [s for s in data
                        if 'Goal not reachable' not in
                        s.get('agent_as_a_point', '')]

            # Sample
            n = min(args.sample_size, len(filtered))
            random.seed(42)
            sampled_indices = random.sample(range(len(filtered)), n)
            sampled = [filtered[i] for i in sampled_indices]
            print(f"  Sampled {n} / {len(filtered)} examples")

            predictions = []
            pred_details = []
            t0 = time.time()

            for idx, sample in enumerate(sampled):
                nl = sample['nl_description']
                prompt = template.format(nl_description=nl)
                raw_response = call_deepseek(client, prompt, model)
                actions = extract_actions(raw_response, strategy)
                predictions.append(actions)

                pred_details.append({
                    'nl_description': nl,
                    'ground_truth': sample['agent_as_a_point'].strip(),
                    'raw_response': raw_response[:500],
                    'extracted_actions': actions,
                })

                if (idx + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    rate = (idx + 1) / elapsed
                    print(f"    {idx+1}/{n} ({rate:.1f}/s)")

                # Rate limiting
                time.sleep(args.delay)

            elapsed = time.time() - t0
            print(f"  Done: {n} samples in {elapsed:.1f}s")

            # Evaluate
            result = evaluate_batch(sampled, predictions)
            metrics = result['metrics']

            print(f"  Success:     {metrics['success_rate']:.4f}")
            print(f"  Feasibility: {metrics['feasibility']:.4f}")
            print(f"  Optimality:  {metrics['optimality']:.4f}")
            if metrics.get('error_distribution'):
                print(f"  Errors:      {metrics['error_distribution']}")

            key = f"{strategy}_{ds_name}"
            all_results[key] = metrics

            # Per-size breakdown for OOD_novel
            if ds_name == 'OOD_novel':
                size_groups = {}
                for sample, pred in zip(sampled, predictions):
                    info = parse_nl_description(sample['nl_description'])
                    r, c = info['grid_size']
                    size_key = f"{r}x{c}"
                    if size_key not in size_groups:
                        size_groups[size_key] = ([], [])
                    size_groups[size_key][0].append(sample)
                    size_groups[size_key][1].append(pred)

                if len(size_groups) > 1:
                    print(f"  Per-size:")
                    per_size = {}
                    for sk in sorted(size_groups.keys(),
                                     key=lambda s: int(s.split('x')[0])):
                        sd, sp = size_groups[sk]
                        sr = evaluate_batch(sd, sp)
                        sm = sr['metrics']
                        per_size[sk] = sm
                        print(f"    {sk}: success={sm['success_rate']:.4f}  "
                              f"n={sm['total']}")
                    all_results[key]['per_size'] = per_size

            # Save per-sample predictions
            for i, ps in enumerate(result['per_sample']):
                pred_details[i]['success'] = ps['success']
                pred_details[i]['error_type'] = ps['error_type']

            pred_file = out_dir / f'{key}_predictions.json'
            with open(pred_file, 'w') as f:
                json.dump(pred_details, f, indent=2)

    # Save summary
    summary_file = out_dir / 'eval_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print final table
    print(f"\n{'='*60}")
    print(f"Summary (model: {model})")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'Success':>10} {'Feasible':>10} {'Optimal':>10}")
    print('-' * 62)
    for key, m in all_results.items():
        sr = m.get('success_rate', 0)
        fe = m.get('feasibility', 0)
        op = m.get('optimality', 0)
        print(f"{key:<30} {sr:>10.4f} {fe:>10.4f} {op:>10.4f}")

    print(f"\nResults saved to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='LLM Prompting evaluation for path planning')
    parser.add_argument('--api_key', type=str, default=None,
                        help='DeepSeek API key (or set DEEPSEEK_API_KEY)')
    parser.add_argument('--base_url', type=str,
                        default='https://api.deepseek.com',
                        help='API base URL')
    parser.add_argument('--model', type=str, default='deepseek-v4-flash',
                        help='Model name (default: deepseek-v4-flash)')
    parser.add_argument('--sample_size', type=int, default=200,
                        help='Samples per test set (default: 200)')
    parser.add_argument('--strategies', nargs='+', default=None,
                        choices=['zero_shot', 'few_shot', 'cot'],
                        help='Which strategies to run (default: all)')
    parser.add_argument('--test_sets', nargs='+', default=None,
                        help='Which test sets to run (default: all)')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between API calls in seconds')
    parser.add_argument('--output_dir', type=str,
                        default='results/llm_prompting/')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_eval(args)