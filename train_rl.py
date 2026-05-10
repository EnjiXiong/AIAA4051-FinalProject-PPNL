"""
GRPO (Group Relative Policy Optimization) training for PPNL path planning.

Pipeline:
    1. Load a SFT-initialized model (trained by train.py)
    2. Sample K action sequences per prompt
    3. Compute rewards using the executor
    4. Update policy with group-relative advantages + KL penalty to reference

Usage:
    # First, get an SFT warm-start (small data, a few epochs):
    python train.py --model t5-base --input_format structured \\
        --epochs 5 --batch_size 16 --lr 3e-4 --bf16 \\
        --output_dir models/sft_warmstart/ \\
        --train_data data/1_train_set_6x6_samples_small2k.json

    # Then run GRPO:
    python train_rl.py \\
        --sft_model models/sft_warmstart/t5-base_structured_ep5_lr0.0003/best \\
        --input_format structured \\
        --epochs 3 --k 8 --lr 1e-5 --bf16
"""
import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from data_utils import (
    PPNLDataset, load_ppnl_data, format_vanilla, format_structured
)
from reward import compute_reward
from evaluate_utils import evaluate_batch


# ─── Argument parsing ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='GRPO RL training')

    # Model
    parser.add_argument('--sft_model', type=str, required=True,
                        help='Path to SFT-initialized model (from train.py)')

    # Data
    parser.add_argument('--train_data', type=str,
                        default='data/1_train_set_6x6_samples.json')
    parser.add_argument('--val_data', type=str,
                        default='data/1dev_set_6x6_samples.json')
    parser.add_argument('--input_format', type=str, default='structured',
                        choices=['vanilla', 'structured'],
                        help='Input format (cot not supported for RL)')

    # RL hyperparams
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of RL epochs over training set')
    parser.add_argument('--k', type=int, default=8,
                        help='Group size K: samples per prompt')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Num prompts per RL step (effective batch = k * batch_size)')
    parser.add_argument('--lr', type=float, default=5e-7,
                        help='RL lr (must be much smaller than SFT lr)')
    parser.add_argument('--kl_coef', type=float, default=0.2,
                        help='KL penalty coefficient β')
    parser.add_argument('--clip_eps', type=float, default=0.2,
                        help='PPO clipping ε')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_new_tokens', type=int, default=64,
                        help='Max tokens to generate (increase for large grids)')

    # General
    parser.add_argument('--max_source_len', type=int, default=512,
                        help='Max source length (increase for large grids)')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_every', type=int, default=50,
                        help='Evaluate every N RL steps')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Stop after N steps (None for full epochs)')
    parser.add_argument('--output_dir', type=str, default='models/rl/')
    parser.add_argument('--grad_clip', type=float, default=1.0)

    return parser.parse_args()


# ─── Input formatting for RL (encoder input only) ───────────────────────────

def format_input(sample: dict, fmt: str) -> str:
    if fmt == 'vanilla':
        return format_vanilla(sample)
    elif fmt == 'structured':
        return format_structured(sample)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


# ─── Core: compute log-probs of a generated sequence under a model ──────────

def compute_sequence_logprob(model, input_ids, attention_mask,
                             gen_output, amp_dtype=None):
    """
    Compute per-token log-probabilities of a generated sequence.

    Args:
        model: Seq2Seq model (T5 or BART)
        input_ids: encoder input [B, L_src]
        attention_mask: [B, L_src]
        gen_output: model.generate() output, shape [B, L_gen]. Note this
                    includes the decoder_start_token_id at position 0.
        amp_dtype: optional autocast dtype

    Returns:
        target_logprobs: [B, L_gen-1] per-token log-probabilities.
            target_logprobs[:, t] = log p(gen_output[:, t+1] | context, gen_output[:, :t+1])
    """
    # gen_output[:, :-1] is the decoder input (includes decoder_start)
    # gen_output[:, 1:]  is the target (the tokens we want to score)
    decoder_input_ids = gen_output[:, :-1].contiguous()
    target_ids = gen_output[:, 1:].contiguous()

    if amp_dtype is not None:
        autocast_ctx = torch.amp.autocast('cuda', dtype=amp_dtype)
    else:
        from contextlib import nullcontext
        autocast_ctx = nullcontext()

    with autocast_ctx:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        logits = outputs.logits  # [B, L_gen-1, V]

    log_probs_all = F.log_softmax(logits.float(), dim=-1)
    target_logprobs = log_probs_all.gather(
        2, target_ids.unsqueeze(-1)
    ).squeeze(-1)  # [B, L_gen-1]

    return target_logprobs


# ─── Main GRPO loop ─────────────────────────────────────────────────────────

def train_rl(args):
    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    amp_dtype = torch.bfloat16 if args.bf16 and device.type == 'cuda' else None
    if amp_dtype is not None:
        print("Using bf16 mixed precision")

    # ─── Load tokenizer & models ────────────────────────────────────────────
    print(f"Loading SFT model from {args.sft_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    policy = AutoModelForSeq2SeqLM.from_pretrained(args.sft_model).to(device)

    # Reference model is a frozen copy of the SFT model
    ref_model = AutoModelForSeq2SeqLM.from_pretrained(args.sft_model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    param_count = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"Policy parameters: {param_count:.1f}M")

    # ─── Output dir ─────────────────────────────────────────────────────────
    sft_name = Path(args.sft_model).parent.name
    run_name = f"grpo_{sft_name}_k{args.k}_lr{args.lr}_kl{args.kl_coef}"
    save_dir = Path(args.output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {save_dir}")

    # ─── Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    train_data = load_ppnl_data(args.train_data)
    train_data = [s for s in train_data
                  if 'Goal not reachable' not in s.get('agent_as_a_point', '')]
    val_data = load_ppnl_data(args.val_data)
    val_data = [s for s in val_data
                if 'Goal not reachable' not in s.get('agent_as_a_point', '')]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # ─── Optimizer ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr,
                                  weight_decay=0.0)

    # ─── Training loop ──────────────────────────────────────────────────────
    history = []
    best_val_success = 0.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Shuffle training data
        random.shuffle(train_data)

        # Iterate in batches of prompts
        t0 = time.time()
        running_reward = 0.0
        running_steps = 0
        running_success = 0
        running_format_err = 0
        running_oob = 0
        running_obstacle = 0

        for batch_start in range(0, len(train_data), args.batch_size):
            batch = train_data[batch_start:batch_start + args.batch_size]
            if len(batch) == 0:
                continue

            # ─── Step 1: Build prompts ──────────────────────────────────────
            prompts = [format_input(s, args.input_format) for s in batch]
            enc = tokenizer(prompts, return_tensors='pt',
                            padding=True, truncation=True,
                            max_length=args.max_source_len).to(device)

            # Expand each prompt K times for group sampling
            #   input_ids: [B * K, L_src]
            expanded_input_ids = enc['input_ids'].repeat_interleave(
                args.k, dim=0)
            expanded_attn_mask = enc['attention_mask'].repeat_interleave(
                args.k, dim=0)

            # ─── Step 2: Sample K responses per prompt ──────────────────────
            policy.eval()
            with torch.no_grad():
                gen_kwargs = dict(
                    input_ids=expanded_input_ids,
                    attention_mask=expanded_attn_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
                # generate returns including decoder_start_token_id prepended
                gen_output = policy.generate(**gen_kwargs)
                # gen_output: [B*K, L_gen], starts with decoder_start

            # Decode to text
            generated_texts = tokenizer.batch_decode(
                gen_output, skip_special_tokens=True)

            # ─── Step 3: Compute rewards ────────────────────────────────────
            rewards = []
            statuses = []
            for i, sample in enumerate(batch):
                for j in range(args.k):
                    idx = i * args.k + j
                    r = compute_reward(
                        sample['world'],
                        generated_texts[idx],
                        sample['agent_as_a_point']
                    )
                    rewards.append(r['reward'])
                    statuses.append(r['status'])
            rewards_t = torch.tensor(rewards, dtype=torch.float32,
                                     device=device)  # [B*K]

            # Track stats
            for s in statuses:
                if 'success' in s:
                    running_success += 1
                elif s == 'format_error':
                    running_format_err += 1
                elif s == 'out_of_bounds':
                    running_oob += 1
                elif s == 'obstacle':
                    running_obstacle += 1
            running_reward += rewards_t.mean().item()
            running_steps += 1

            # ─── Step 4: Group-relative advantage ───────────────────────────
            # Reshape to [B, K] to compute per-group stats
            B = len(batch)
            rewards_grouped = rewards_t.view(B, args.k)
            group_mean = rewards_grouped.mean(dim=1, keepdim=True)
            group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
            advantages = (rewards_grouped - group_mean) / group_std  # [B, K]
            advantages = advantages.view(-1)  # [B*K]

            # If all rewards in a group are equal, advantages will be 0
            # (std = eps, numerator = 0 → 0). This is fine; no gradient signal.

            # ─── Step 5: Compute log-probs under policy and reference ───────
            # compute_sequence_logprob returns logprobs for gen_output[:, 1:]
            # (scoring the tokens that were actually generated, excluding
            # the decoder_start_token prepended by generate()).

            # Mask: target positions are gen_output[:, 1:].
            # A target token is valid if it is not pad_token_id.
            target_ids_for_mask = gen_output[:, 1:]
            target_mask = (target_ids_for_mask != tokenizer.pad_token_id).float()
            # Also, since T5's pad_token_id == decoder_start_token_id (= 0)
            # for BART this may also be the case for initial tokens. But we've
            # already skipped position 0 by taking [:, 1:].
            # The mask correctly excludes trailing pad tokens.

            policy.train()
            # Policy log-probs (with gradient)
            policy_logprobs = compute_sequence_logprob(
                policy, expanded_input_ids, expanded_attn_mask,
                gen_output, amp_dtype=amp_dtype
            )  # [B*K, L_gen-1]

            # Reference log-probs (no gradient)
            with torch.no_grad():
                ref_logprobs = compute_sequence_logprob(
                    ref_model, expanded_input_ids, expanded_attn_mask,
                    gen_output, amp_dtype=amp_dtype
                )  # [B*K, L_gen-1]

            # ─── Step 6: Compute GRPO loss ──────────────────────────────────
            # KL penalty per-token (masked), averaged over valid tokens
            kl_per_token = (policy_logprobs - ref_logprobs) * target_mask
            # sum over tokens; divide by num valid tokens for per-sequence mean
            num_valid = target_mask.sum(dim=1).clamp(min=1.0)  # [B*K]
            kl_per_seq = kl_per_token.sum(dim=1) / num_valid  # [B*K]

            # Policy gradient term with advantage.
            # Since we sample fresh each step, π_old == π_policy at sample
            # time, so ratio ≈ 1 on the first (and only) gradient step.
            # We use the REINFORCE form: L_pg = -E[A · Σ_t log π(a_t|s)].
            # Normalize sum by sequence length to keep losses comparable
            # across variable-length outputs.
            policy_logprob_seq = (
                policy_logprobs * target_mask).sum(dim=1) / num_valid

            # Policy gradient loss (negative because we maximize advantage)
            pg_loss = -(advantages.detach() * policy_logprob_seq).mean()

            # KL penalty (per-sequence, averaged across batch)
            kl_loss = kl_per_seq.mean()

            total_loss = pg_loss + args.kl_coef * kl_loss

            # ─── Step 7: Backprop & update ──────────────────────────────────
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1

            # ─── Logging ────────────────────────────────────────────────────
            if global_step % 10 == 0:
                avg_r = running_reward / running_steps
                total_samples = running_steps * args.batch_size * args.k
                success_pct = running_success / total_samples * 100
                fmt_err_pct = running_format_err / total_samples * 100
                oob_pct = running_oob / total_samples * 100
                obs_pct = running_obstacle / total_samples * 100
                print(f"  step {global_step:5d} | "
                      f"pg_loss={pg_loss.item():.4f} | "
                      f"kl={kl_loss.item():.4f} | "
                      f"avg_r={avg_r:.3f} | "
                      f"succ={success_pct:.1f}% | "
                      f"fmt_err={fmt_err_pct:.1f}% | "
                      f"oob={oob_pct:.1f}% | "
                      f"obs={obs_pct:.1f}%")
                # Reset running stats every 10 steps
                running_reward = 0.0
                running_steps = 0
                running_success = 0
                running_format_err = 0
                running_oob = 0
                running_obstacle = 0

            # ─── Periodic eval ──────────────────────────────────────────────
            if global_step % args.eval_every == 0:
                val_metrics = run_eval(policy, tokenizer, val_data,
                                       args, device)
                elapsed = time.time() - t0
                log = {
                    'step': global_step, 'epoch': epoch,
                    'val_success_rate': val_metrics['success_rate'],
                    'val_feasibility': val_metrics['feasibility'],
                    'val_optimality': val_metrics['optimality'],
                    'time': elapsed,
                }
                # Include per-size metrics if available
                if 'per_size' in val_metrics:
                    log['per_size'] = {
                        k: {'success': v['success_rate'],
                            'feasible': v['feasibility']}
                        for k, v in val_metrics['per_size'].items()
                    }
                history.append(log)
                print(f"  [eval] step={global_step} | "
                      f"success={val_metrics['success_rate']:.4f} | "
                      f"feasible={val_metrics['feasibility']:.4f} | "
                      f"optimal={val_metrics['optimality']:.4f}")
                # Print per-size breakdown
                if 'per_size' in val_metrics:
                    for k, v in val_metrics['per_size'].items():
                        print(f"         {k}: succ={v['success_rate']:.3f} "
                              f"feas={v['feasibility']:.3f}")

                if val_metrics['success_rate'] > best_val_success:
                    best_val_success = val_metrics['success_rate']
                    policy.save_pretrained(save_dir / 'best')
                    tokenizer.save_pretrained(save_dir / 'best')
                    print(f"  ★ New best saved (success={best_val_success:.4f})")

            if args.max_steps is not None and global_step >= args.max_steps:
                print(f"Reached max_steps={args.max_steps}, stopping.")
                break

        if args.max_steps is not None and global_step >= args.max_steps:
            break

        # End-of-epoch eval
        val_metrics = run_eval(policy, tokenizer, val_data, args, device)
        print(f"\nEnd of epoch {epoch}: "
              f"success={val_metrics['success_rate']:.4f} | "
              f"feasible={val_metrics['feasibility']:.4f} | "
              f"optimal={val_metrics['optimality']:.4f}")
        if 'per_size' in val_metrics:
            for k, v in val_metrics['per_size'].items():
                print(f"  {k}: succ={v['success_rate']:.3f} "
                      f"feas={v['feasibility']:.3f}")
        log_entry = {
            'step': global_step, 'epoch': epoch, 'end_of_epoch': True,
            'val_success_rate': val_metrics['success_rate'],
            'val_feasibility': val_metrics['feasibility'],
            'val_optimality': val_metrics['optimality'],
        }
        if 'per_size' in val_metrics:
            log_entry['per_size'] = {
                k: {'success': v['success_rate'],
                    'feasible': v['feasibility']}
                for k, v in val_metrics['per_size'].items()
            }
        history.append(log_entry)

        if val_metrics['success_rate'] > best_val_success:
            best_val_success = val_metrics['success_rate']
            policy.save_pretrained(save_dir / 'best')
            tokenizer.save_pretrained(save_dir / 'best')

    # Save final model
    policy.save_pretrained(save_dir / 'final')
    tokenizer.save_pretrained(save_dir / 'final')

    # Save history + config
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    config = vars(args)
    config['best_val_success'] = best_val_success
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nRL training complete. Best val success: {best_val_success:.4f}")
    print(f"Models saved to: {save_dir}")


# ─── Evaluation helper (greedy decoding on val) ─────────────────────────────

def run_eval(model, tokenizer, val_data, args, device, batch_size=32):
    """Run greedy eval on validation set. Reports per-size metrics if mixed."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i:i + batch_size]
            prompts = [format_input(s, args.input_format) for s in batch]
            enc = tokenizer(prompts, return_tensors='pt', padding=True,
                            truncation=True,
                            max_length=args.max_source_len).to(device)
            out = model.generate(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask'],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            predictions.extend(decoded)

    result = evaluate_batch(val_data, predictions)
    metrics = result['metrics']

    # If data contains 'grid_size' field, also report per-size breakdown
    if any('grid_size' in s for s in val_data):
        sizes = sorted(set(s.get('grid_size', 0) for s in val_data))
        if len(sizes) > 1:
            per_size = {}
            for size in sizes:
                idx = [i for i, s in enumerate(val_data)
                       if s.get('grid_size', 0) == size]
                if not idx:
                    continue
                size_data = [val_data[i] for i in idx]
                size_preds = [predictions[i] for i in idx]
                size_result = evaluate_batch(size_data, size_preds)
                per_size[f'{size}x{size}'] = size_result['metrics']
            metrics['per_size'] = per_size

    return metrics


if __name__ == '__main__':
    args = parse_args()
    train_rl(args)