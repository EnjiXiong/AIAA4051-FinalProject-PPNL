"""
Training script for fine-tuning T5/BART on PPNL single-goal path planning.

Usage:
    python train.py --model t5-small --input_format vanilla --epochs 20
    python train.py --model t5-base --input_format structured --epochs 15
    python train.py --model facebook/bart-base --input_format vanilla --epochs 20
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
    set_seed,
)

from data_utils import PPNLDataset, load_ppnl_data, extract_actions_from_cot
from evaluate_utils import evaluate_batch


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune T5/BART on PPNL')
    
    # Model
    parser.add_argument('--model', type=str, default='t5-small',
                        help='Model name (e.g. t5-small) or local path (e.g. ./pretrained/bart-base)')
    
    # Data
    parser.add_argument('--train_data', type=str,
                        default='data/1_train_set_6x6_samples.json')
    parser.add_argument('--val_data', type=str,
                        default='data/1dev_set_6x6_samples.json')
    parser.add_argument('--input_format', type=str, default='vanilla',
                        choices=['vanilla', 'structured', 'cot'],
                        help='Input formatting strategy')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--max_source_len', type=int, default=256)
    parser.add_argument('--max_target_len', type=int, default=128)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--fp16', action='store_true',
                        help='Use fp16 mixed precision (may cause NaN with T5)')
    parser.add_argument('--bf16', action='store_true',
                        help='Use bf16 mixed precision (recommended for T5 on Ampere+ GPUs)')
    
    # Generation
    parser.add_argument('--num_beams', type=int, default=1,
                        help='Beam size for generation during eval')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='models/')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_best', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for torch / numpy / random / dataloader. '
                             'If set, the run_name is suffixed with _seedN.')

    return parser.parse_args()


def generate_predictions(model, tokenizer, dataloader, device,
                         max_length=128, num_beams=1):
    """Generate predictions for a dataset."""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True if num_beams > 1 else False,
            )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_preds.extend(decoded)
    
    return all_preds


def train(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Seeding (optional; preserves unseeded behavior when --seed not passed)
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
        print(f"Seed: {args.seed}")

    # Model name for saving (suffix with _seedN when seed is set)
    model_short = args.model.split('/')[-1]
    run_name = f"{model_short}_{args.input_format}_ep{args.epochs}_lr{args.lr}"
    if args.seed is not None:
        run_name = f"{run_name}_seed{args.seed}"
    save_dir = Path(args.output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"Model: {args.model}")
    print(f"Input format: {args.input_format}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"{'='*60}\n")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.1f}M")
    
    # Load data
    print("Loading data...")
    train_data = load_ppnl_data(args.train_data)
    val_data = load_ppnl_data(args.val_data)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    train_dataset = PPNLDataset(
        train_data, tokenizer,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        input_format=args.input_format
    )
    val_dataset = PPNLDataset(
        val_data, tokenizer,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        input_format=args.input_format
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=2, pin_memory=True)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Mixed precision
    use_amp = False
    amp_dtype = torch.float32
    scaler = None
    if device.type == 'cuda':
        if args.bf16:
            use_amp = True
            amp_dtype = torch.bfloat16
            print("Using bf16 mixed precision")
        elif args.fp16:
            use_amp = True
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler('cuda')
            print("Using fp16 mixed precision (warning: may cause NaN with T5)")
    
    # Training loop
    best_success_rate = 0.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / args.grad_accum
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / args.grad_accum
                loss.backward()
            
            epoch_loss += outputs.loss.item()
            num_batches += 1
            
            if (step + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - t0
        
        log = {'epoch': epoch, 'train_loss': avg_loss, 'time': elapsed}
        
        # Evaluation
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | "
                  f"time={elapsed:.0f}s | Evaluating...")
            
            predictions = generate_predictions(
                model, tokenizer, val_loader, device,
                max_length=args.max_target_len,
                num_beams=args.num_beams
            )
            
            # If using CoT format, extract plain actions
            if args.input_format == 'cot':
                predictions = [extract_actions_from_cot(p) for p in predictions]
            
            # Filter unreachable samples (same as dataset does)
            val_filtered = [s for s in val_data
                           if 'Goal not reachable' not in s.get('agent_as_a_point', '')]
            
            result = evaluate_batch(val_filtered, predictions)
            metrics = result['metrics']
            
            log.update({
                'val_success_rate': metrics['success_rate'],
                'val_feasibility': metrics['feasibility'],
                'val_optimality': metrics['optimality'],
            })
            
            print(f"  → Success: {metrics['success_rate']:.4f} | "
                  f"Feasible: {metrics['feasibility']:.4f} | "
                  f"Optimal: {metrics['optimality']:.4f}")
            
            # Save best
            if args.save_best and metrics['success_rate'] > best_success_rate:
                best_success_rate = metrics['success_rate']
                model.save_pretrained(save_dir / 'best')
                tokenizer.save_pretrained(save_dir / 'best')
                print(f"  ★ New best model saved (success={best_success_rate:.4f})")
        else:
            print(f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | time={elapsed:.0f}s")
        
        history.append(log)
    
    # Save final model and training history
    model.save_pretrained(save_dir / 'final')
    tokenizer.save_pretrained(save_dir / 'final')
    
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config = vars(args)
    config['best_success_rate'] = best_success_rate
    config['total_params_M'] = param_count
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining complete. Best success rate: {best_success_rate:.4f}")
    print(f"Models saved to: {save_dir}")
    
    return save_dir


if __name__ == '__main__':
    args = parse_args()
    train(args)