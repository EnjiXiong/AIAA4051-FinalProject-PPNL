"""
Prompting experiments with Flan-T5 for PPNL path planning.
Tests zero-shot, few-shot, and Chain-of-Thought prompting strategies.

Usage:
    python prompt_eval.py --model google/flan-t5-base --strategy zero_shot
    python prompt_eval.py --model google/flan-t5-base --strategy few_shot
    python prompt_eval.py --model google/flan-t5-base --strategy cot_coordinate
    python prompt_eval.py --model google/flan-t5-large --strategy cot_plan_then_act
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from data_utils import load_ppnl_data, parse_nl_description
from evaluate_utils import evaluate_batch


# ─── Prompt Templates ───────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "nl": "You are in a 4 by 4 world. There are obstacles that you have to avoid at: (1,2). Go from (0,0) to (2,3)",
        "answer": "right right right down down"
    },
    {
        "nl": "You are in a 4 by 4 world. There are obstacles that you have to avoid at: (0,1), (2,2). Go from (0,0) to (3,3)",
        "answer": "down down down right right right"
    },
    {
        "nl": "You are in a 5 by 5 world. There are no obstacles. Go from (2,0) to (0,4)",
        "answer": "up up right right right right"
    },
]


def make_zero_shot_prompt(nl_description: str) -> str:
    return (
        f"You are a path planning agent in a grid world. "
        f"Output the shortest sequence of actions (up, down, left, right) "
        f"to reach the goal while avoiding obstacles.\n\n"
        f"Task: {nl_description}\n\n"
        f"Actions:"
    )


def make_few_shot_prompt(nl_description: str) -> str:
    examples = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples += f"Task: {ex['nl']}\nActions: {ex['answer']}\n\n"
    
    return (
        f"You are a path planning agent in a grid world. "
        f"Output the shortest sequence of actions (up, down, left, right) "
        f"to reach the goal while avoiding obstacles.\n\n"
        f"{examples}"
        f"Task: {nl_description}\n"
        f"Actions:"
    )


def make_cot_coordinate_prompt(nl_description: str) -> str:
    """CoT prompt: track coordinates at each step."""
    info = parse_nl_description(nl_description)
    
    example = (
        "Example:\n"
        "Task: You are in a 4 by 4 world. No obstacles. Go from (0,0) to (1,2)\n"
        "Reasoning: I am at (0,0). Goal is (1,2). "
        "I need to go right 2 and down 1. "
        "Step 1: right -> now at (0,1). "
        "Step 2: right -> now at (0,2). "
        "Step 3: down -> now at (1,2). Reached goal!\n"
        "Actions: right right down\n\n"
    )
    
    return (
        f"You are a path planning agent. Plan a path step by step, "
        f"tracking your position after each move. Avoid obstacles and stay in bounds.\n\n"
        f"{example}"
        f"Task: {nl_description}\n"
        f"Reasoning: I am at ({info['start'][0]},{info['start'][1]}). "
        f"Goal is ({info['goal'][0]},{info['goal'][1]}). "
    )


def make_cot_plan_then_act_prompt(nl_description: str) -> str:
    """CoT prompt: first describe high-level plan, then generate actions."""
    info = parse_nl_description(nl_description)
    obs_str = ", ".join(f"({r},{c})" for r, c in info['obstacles']) if info['obstacles'] else "none"
    
    return (
        f"You are a path planning agent in a {info['grid_size'][0]}x{info['grid_size'][1]} grid.\n"
        f"Start: ({info['start'][0]},{info['start'][1]})\n"
        f"Goal: ({info['goal'][0]},{info['goal'][1]})\n"
        f"Obstacles: {obs_str}\n\n"
        f"First, describe your plan in words. "
        f"Then output the action sequence.\n\n"
        f"Plan:"
    )


def make_cot_grid_reconstruction_prompt(nl_description: str) -> str:
    """CoT prompt: reconstruct ASCII grid, then plan."""
    info = parse_nl_description(nl_description)
    rows, cols = info['grid_size']
    
    # Build ASCII grid
    grid_lines = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if (r, c) == info['start']:
                row.append('S')
            elif (r, c) == info['goal']:
                row.append('G')
            elif (r, c) in info['obstacles']:
                row.append('X')
            else:
                row.append('.')
        grid_lines.append(' '.join(row))
    ascii_grid = '\n'.join(grid_lines)
    
    return (
        f"Here is a grid where S=start, G=goal, X=obstacle, .=empty:\n"
        f"{ascii_grid}\n\n"
        f"Navigate from S to G avoiding X. "
        f"Output actions (up/down/left/right):\n"
        f"Actions:"
    )


STRATEGY_MAP = {
    'zero_shot': make_zero_shot_prompt,
    'few_shot': make_few_shot_prompt,
    'cot_coordinate': make_cot_coordinate_prompt,
    'cot_plan_then_act': make_cot_plan_then_act_prompt,
    'cot_grid': make_cot_grid_reconstruction_prompt,
}


# ─── Action extraction from model output ────────────────────────────────────

def extract_actions(text: str) -> str:
    """Extract valid action words from model output."""
    valid = {'up', 'down', 'left', 'right'}
    actions = []
    for word in text.lower().replace(',', ' ').replace('.', ' ').split():
        if word in valid:
            actions.append(word)
    return ' '.join(actions)


# ─── Main evaluation ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Prompting experiments')
    parser.add_argument('--model', type=str, default='google/flan-t5-base')
    parser.add_argument('--strategy', type=str, default='zero_shot',
                        choices=list(STRATEGY_MAP.keys()))
    parser.add_argument('--test_file', type=str,
                        default='data/1_goals_test_seen_6x6_samples.json')
    parser.add_argument('--test_name', type=str, default='ID_seen_6x6')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of samples (for quick testing)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--save_predictions', action='store_true')
    return parser.parse_args()


def run_prompt_eval(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(device)
    model.eval()
    
    # Load data
    data = load_ppnl_data(args.test_file)
    data = [s for s in data if 'Goal not reachable' not in s.get('agent_as_a_point', '')]
    
    if args.max_samples:
        data = data[:args.max_samples]
    
    print(f"Test set: {args.test_name} ({len(data)} samples)")
    print(f"Strategy: {args.strategy}")
    
    prompt_fn = STRATEGY_MAP[args.strategy]
    
    # Generate prompts
    prompts = [prompt_fn(s['nl_description']) for s in data]
    
    # Print example
    print(f"\n--- Example prompt ---")
    print(prompts[0])
    print(f"--- End example ---\n")
    
    # Batch inference
    predictions = []
    t0 = time.time()
    
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i:i + args.batch_size]
        
        inputs = tokenizer(
            batch_prompts, return_tensors='pt', padding=True,
            truncation=True, max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                early_stopping=True,
            )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
        
        if (i // args.batch_size) % 10 == 0:
            print(f"  Processed {min(i + args.batch_size, len(prompts))}/{len(prompts)}")
    
    elapsed = time.time() - t0
    print(f"Generation time: {elapsed:.1f}s ({elapsed/len(data):.3f}s/sample)")
    
    # Extract actions from raw outputs
    action_preds = [extract_actions(p) for p in predictions]
    
    # Show some predictions
    print(f"\n--- Sample predictions ---")
    for i in range(min(5, len(data))):
        print(f"  Input: {data[i]['nl_description']}")
        print(f"  Raw output: {predictions[i]}")
        print(f"  Actions: {action_preds[i]}")
        print(f"  Ground truth: {data[i]['agent_as_a_point'].strip()}")
        print()
    
    # Evaluate
    result = evaluate_batch(data, action_preds)
    metrics = result['metrics']
    
    print(f"\n{'='*50}")
    print(f"Results: {args.model} / {args.strategy} / {args.test_name}")
    print(f"{'='*50}")
    print(f"  Success Rate:  {metrics['success_rate']:.4f}")
    print(f"  Feasibility:   {metrics['feasibility']:.4f}")
    print(f"  Optimality:    {metrics['optimality']:.4f}")
    print(f"  Errors: {metrics.get('error_distribution', {})}")
    
    # Save results
    model_short = args.model.split('/')[-1]
    out_dir = Path(args.output_dir) / f"prompt_{model_short}_{args.strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'model': args.model,
        'strategy': args.strategy,
        'test_set': args.test_name,
        'num_samples': len(data),
        'metrics': metrics,
        'generation_time_s': elapsed,
    }
    
    with open(out_dir / f'{args.test_name}_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    if args.save_predictions:
        preds_out = []
        for i, (s, raw, act) in enumerate(zip(data, predictions, action_preds)):
            preds_out.append({
                'nl_description': s['nl_description'],
                'ground_truth': s['agent_as_a_point'].strip(),
                'raw_output': raw,
                'extracted_actions': act,
                'success': result['per_sample'][i]['success'],
                'error_type': result['per_sample'][i]['error_type'],
            })
        with open(out_dir / f'{args.test_name}_predictions.json', 'w') as f:
            json.dump(preds_out, f, indent=2)
    
    return metrics


if __name__ == '__main__':
    args = parse_args()
    run_prompt_eval(args)
