"""
Data utilities for loading PPNL data and preparing it for T5/BART fine-tuning.
Supports vanilla, structured input, and CoT-augmented formats.
"""
import json
import re
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset


def load_ppnl_data(path: str) -> List[Dict]:
    """Load PPNL JSON data."""
    with open(path) as f:
        return json.load(f)


def parse_nl_description(nl: str) -> Dict:
    """
    Parse the natural language description into structured components.
    
    Example input: "You are in a 6 by 6 world. There are obstacles that you 
                    have to avoid at: (5,3). Go from (1,4) to (2,1)"
    """
    # Extract grid size
    size_match = re.search(r'(\d+) by (\d+)', nl)
    rows, cols = int(size_match.group(1)), int(size_match.group(2))
    
    # Extract obstacles
    obstacle_matches = re.findall(r'\((\d+),(\d+)\)', nl.split('Go from')[0])
    obstacles = [(int(r), int(c)) for r, c in obstacle_matches]
    
    # Extract start and goal
    nav_part = nl.split('Go from')[1]
    nav_matches = re.findall(r'\((\d+),(\d+)\)', nav_part)
    start = (int(nav_matches[0][0]), int(nav_matches[0][1]))
    goal = (int(nav_matches[1][0]), int(nav_matches[1][1]))
    
    return {
        'grid_size': (rows, cols),
        'obstacles': obstacles,
        'start': start,
        'goal': goal,
    }


# ─── Input format functions ─────────────────────────────────────────────────

def format_vanilla(sample: Dict) -> str:
    """Use the original NL description as-is."""
    return sample['nl_description']


def format_structured(sample: Dict) -> str:
    """
    Reformat input into a more structured representation.
    Hypothesis: structured format helps model parse spatial info more reliably.
    """
    info = parse_nl_description(sample['nl_description'])
    rows, cols = info['grid_size']
    
    parts = [
        f"Grid: {rows}x{cols}",
        f"Start: ({info['start'][0]},{info['start'][1]})",
        f"Goal: ({info['goal'][0]},{info['goal'][1]})",
    ]
    
    if info['obstacles']:
        obs_str = " ".join(f"({r},{c})" for r, c in info['obstacles'])
        parts.append(f"Obstacles: {obs_str}")
    else:
        parts.append("Obstacles: none")
    
    parts.append("Output the shortest path as a sequence of: up down left right")
    
    return " | ".join(parts)


def format_coordinate_tracking(sample: Dict) -> Tuple[str, str]:
    """
    Create CoT-augmented input/output pairs with coordinate tracking.
    The target includes intermediate coordinates at each step.
    
    Returns: (input_text, cot_target)
    """
    input_text = format_structured(sample)
    
    # Build CoT target from solution_coordinates
    coords = sample['solution_coordinates']
    actions = sample['agent_as_a_point'].strip().split()
    
    if len(coords) < 2 or len(actions) == 0:
        # Fallback to vanilla target
        return input_text, sample['agent_as_a_point'].strip()
    
    cot_parts = [f"Start at ({coords[0][0]},{coords[0][1]})"]
    for i, action in enumerate(actions):
        if i + 1 < len(coords):
            cot_parts.append(
                f"{action} -> ({coords[i+1][0]},{coords[i+1][1]})"
            )
        else:
            cot_parts.append(action)
    cot_parts.append("Done")
    
    cot_target = " | ".join(cot_parts)
    return input_text, cot_target


# ─── Dataset class ──────────────────────────────────────────────────────────

class PPNLDataset(Dataset):
    """PyTorch Dataset for PPNL path planning."""
    
    def __init__(self, data: List[Dict], tokenizer, max_source_len: int = 256,
                 max_target_len: int = 128, input_format: str = 'vanilla'):
        """
        Args:
            data: list of PPNL sample dicts
            tokenizer: HuggingFace tokenizer
            max_source_len: max input token length
            max_target_len: max target token length
            input_format: 'vanilla' | 'structured' | 'cot'
        """
        self.data = [s for s in data if 'Goal not reachable' not in s.get('agent_as_a_point', '')]
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.input_format = input_format
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.input_format == 'vanilla':
            source = format_vanilla(sample)
            target = sample['agent_as_a_point'].strip()
        elif self.input_format == 'structured':
            source = format_structured(sample)
            target = sample['agent_as_a_point'].strip()
        elif self.input_format == 'cot':
            source, target = format_coordinate_tracking(sample)
        else:
            raise ValueError(f"Unknown input_format: {self.input_format}")
        
        source_enc = self.tokenizer(
            source, max_length=self.max_source_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        target_enc = self.tokenizer(
            target, max_length=self.max_target_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        
        labels = target_enc['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_enc['input_ids'].squeeze(),
            'attention_mask': source_enc['attention_mask'].squeeze(),
            'labels': labels,
        }


def extract_actions_from_cot(cot_output: str) -> str:
    """
    Extract plain action sequence from CoT output.
    E.g., "Start at (1,4) | left -> (1,3) | down -> (2,3) | Done"
          -> "left down"
    """
    actions = []
    valid_actions = {'up', 'down', 'left', 'right'}
    for part in cot_output.split('|'):
        part = part.strip()
        for token in part.split():
            if token in valid_actions:
                actions.append(token)
    return ' '.join(actions)


if __name__ == '__main__':
    data = load_ppnl_data('data/1_train_set_6x6_samples.json')
    sample = data[0]
    
    print("=== Vanilla format ===")
    print(format_vanilla(sample))
    print(f"Target: {sample['agent_as_a_point'].strip()}")
    
    print("\n=== Structured format ===")
    print(format_structured(sample))
    
    print("\n=== CoT format ===")
    inp, tgt = format_coordinate_tracking(sample)
    print(f"Input: {inp}")
    print(f"Target: {tgt}")
    
    print("\n=== Parsed description ===")
    info = parse_nl_description(sample['nl_description'])
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    # Test CoT extraction
    print(f"\n=== CoT -> actions ===")
    print(f"Extracted: {extract_actions_from_cot(tgt)}")
