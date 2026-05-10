"""
Create a smaller training subset for SFT warm-start before RL.

Usage:
    python make_sft_subset.py --n 2000 --seed 42
"""
import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='data/1_train_set_6x6_samples.json')
    parser.add_argument('--output', type=str,
                        default='data/1_train_set_6x6_samples_small2k.json')
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    with open(args.input) as f:
        data = json.load(f)

    # Filter out unreachable samples
    data = [s for s in data
            if 'Goal not reachable' not in s.get('agent_as_a_point', '')]

    if args.n >= len(data):
        print(f"Requested n={args.n} >= available {len(data)}, using all")
        subset = data
    else:
        subset = random.sample(data, args.n)

    # Also create a dev split from the remaining data for SFT monitoring
    # (not strictly necessary, but helpful)

    with open(args.output, 'w') as f:
        json.dump(subset, f)

    print(f"Wrote {len(subset)} samples to {args.output}")
    # Print path length distribution
    lengths = [len(s['agent_as_a_point'].strip().split()) for s in subset]
    print(f"Path length: min={min(lengths)}, max={max(lengths)}, "
          f"avg={sum(lengths)/len(lengths):.1f}")


if __name__ == '__main__':
    main()