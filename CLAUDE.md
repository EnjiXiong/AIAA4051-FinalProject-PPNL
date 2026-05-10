# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

AIAA 4051 final project: prompting and fine-tuning seq2seq LMs (T5, BART, Flan-T5) on the PPNL single-goal grid path planning benchmark (Aghzal et al., ICLR 2024 Workshop). Input is a natural-language description of a grid world; target is a sequence of `up`/`down`/`left`/`right` actions.

This directory is the GitHub repo root (`EnjiXiong/AIAA4051-FinalProject-PPNL`). Trained checkpoints (`models/`) are gitignored and live on the Hugging Face mirror at <https://huggingface.co/EnjiXiong/AIAA4051-FinalProject-PPNL> — pull one with `hf download EnjiXiong/AIAA4051-FinalProject-PPNL --include "grid-path-planning/models/<run>/**/best/**" --local-dir .`. The upstream PPNL reference code (also in the HF mirror) is at <https://github.com/MohamedAghzal/llms-as-path-planners>. For pretrained backbones, pass `--model facebook/bart-base` (or `t5-base`, etc.) and let HF auto-fetch — no local `pretrained/` dir is shipped.

## Common commands

Setup:
```bash
pip install torch transformers sentencepiece accelerate datasets matplotlib pandas
```

End-to-end pipeline (SFT × 4 configs → eval all checkpoints → prompting):
```bash
bash run_all.sh
```

Single-config SFT (writes to `models/<model>_<format>_ep<E>_lr<LR>/{best,final}`):
```bash
python train.py --model t5-base --input_format vanilla   --epochs 15 --batch_size 16 --lr 3e-4 --bf16
python train.py --model t5-base --input_format structured --epochs 15 --batch_size 16 --lr 3e-4 --bf16
python train.py --model t5-base --input_format cot       --epochs 15 --batch_size 16 --lr 3e-4 --max_target_len 256 --bf16
```

Evaluate one checkpoint on all 5 built-in test sets (writes JSON to `results/<run_name>/`):
```bash
python run_eval.py --model_dir models/t5-base_vanilla_ep15_lr0.0003/best --save_predictions
# Match --input_format to how the model was trained:
python run_eval.py --model_dir models/t5-base_cot_ep15_lr0.0003/best --input_format cot --save_predictions
# Add a custom OOD test:
python run_eval.py --model_dir models/sft_multiscale_40ep/best \
    --extra_test OOD_novel=data/ood_novel_sizes.json --save_predictions
```

Constrained decoding via tree search (model = policy prior, executor = hard constraint):
```bash
python tree_search_eval.py --model_dir models/sft_multiscale_40ep/best --input_format vanilla --beam_width 4
python tree_search_eval.py --model_dir models/t5-base_cot_ep15_lr0.0003/best --input_format cot --beam_width 8
```

Prompting baselines (Flan-T5 local, or DeepSeek API):
```bash
python prompt_eval.py --model google/flan-t5-base --strategy zero_shot
python prompt_eval_llm.py --sample_size 200          # needs DEEPSEEK_API_KEY
```

GRPO RL fine-tuning (requires SFT warm-start + diverse RL envs):
```bash
python make_sft_subset.py --n 2000                    # write data/1_train_set_6x6_samples_small2k.json
python generate_rl_envs.py --output data/rl_envs_diverse.json
python train.py --model t5-base --input_format structured --epochs 5 --bf16 \
    --train_data data/1_train_set_6x6_samples_small2k.json --output_dir models/sft_warmstart/
python train_rl.py --sft_model models/sft_warmstart/.../best --input_format structured \
    --epochs 3 --k 8 --lr 1e-5 --bf16
```

Self-tests on the executor and reward (run from `grid-path-planning/`):
```bash
python evaluate_utils.py    # GT predictions → 100% success
python reward.py            # asserts reward shape on canned cases
python data_utils.py        # dumps the three input format variants
```

## Code architecture

Two layers — a shared core and several training / inference / eval entrypoints that compose it. Run all commands from the `grid-path-planning/` directory; data and model paths are relative.

**Core (shared by every entrypoint):**
- `data_utils.py` — loads PPNL JSON, parses NL into `{grid_size, start, goal, obstacles}`, exposes three input-formatting functions and the `PPNLDataset` torch wrapper. Three formats:
  - `vanilla`: original NL string, target is plain action sequence
  - `structured`: `Grid: 6x6 | Start: (r,c) | Goal: (r,c) | Obstacles: ...` — parsed/normalized
  - `cot`: structured input + coordinate-tracking target `Start at (r,c) | left -> (r,c-1) | ... | Done`
  Samples whose ground truth contains `'Goal not reachable'` are filtered everywhere (dataset, evaluator, RL). When generating with `cot`, callers must run `extract_actions_from_cot` to recover the plain action string before evaluation.
- `evaluate_utils.py` — the executor (`simulate_path` returns one of `success`/`out_of_bounds`/`obstacle`/`wrong_end`/`format_error`), `a_star_distance` for shortest-path checks, and `evaluate_batch` which returns Success / Feasibility / Optimality / Exact-Match / error distribution. **Optimality is `success AND pred_len <= gt_len`**, not strict equality. Grid encoding: `0=empty, 1=obstacle, 2=start, 3=goal`.
- `reward.py` — the GRPO reward (calls the same executor): `+1.0` optimal, `+0.8` non-optimal success, `+0.2 * (1 - dist/max_dist)` shaped for `wrong_end`, `-0.5` OOB/obstacle, `-1.0` format_error.

**Entrypoints (each is a thin script around the core):**
- `train.py` — supervised fine-tuning loop for any HF seq2seq model. Prefer `--bf16` over `--fp16` (T5 frequently NaNs in fp16). Saves `best/` (highest val success rate) and `final/` plus `training_history.json` and `config.json`.
- `train_rl.py` — GRPO with K samples per prompt, group-relative advantages, KL penalty to a frozen reference. Requires an SFT-initialized model (`--sft_model`); `cot` format is **not** supported. Use a much smaller LR (1e-5 to 5e-7).
- `run_eval.py` — runs a checkpoint over the five canonical test sets and emits per-sample predictions plus per-grid-size breakdown when sizes vary. `--input_format` must match training.
- `tree_search_eval.py` — beam search where the model only scores the four action tokens at each step and the executor prunes illegal moves. Two decoder modes: `vanilla` (one action token per step) and `cot` (action token + forced ` -> (r,c) |` coordinate continuation that keeps the CoT model's spatial state).
- `prompt_eval.py` (Flan-T5 local) and `prompt_eval_llm.py` (DeepSeek API) — five and three prompting strategies respectively (zero/few-shot, several CoT variants).
- `generate_rl_envs.py`, `make_sft_subset.py` — data prep utilities.
- `visualize.py`, `visualize_cases.py` — plot grids and per-experiment result comparisons.

**Test set naming (used by `run_eval.py` and `tree_search_eval.py`):**
- `ID_seen_6x6` / `ID_unseen_6x6`: in-distribution
- `OOD_5x5` / `OOD_7x7`: size generalization
- `OOD_6x6_dense`: obstacle-density generalization
- Pass any extra set with `--extra_test NAME=path.json`; combine with `--only_extra` to skip the built-ins.

**Model naming convention:** `models/<model>_<format>_ep<E>_lr<LR>/{best,final}`. `run_all.sh`'s eval phase infers `--input_format` by substring-matching the directory name (`structured`, `cot`, else `vanilla`); custom run names that drop those keywords will silently default to `vanilla` and produce wrong scores.

## Gotchas

- `cot` mode requires `--max_target_len 256` (paths plus coordinate annotations exceed the 128 default).
- The `cot` predictions must go through `extract_actions_from_cot` before being passed to `evaluate_batch` — this happens automatically in `train.py`/`run_eval.py` only when `--input_format cot` is set.
- `tree_search_eval.py` assumes each action name (`up`/`down`/`left`/`right`) is a single tokenizer token — it asserts this at startup. Holds for T5/BART; verify before swapping in another tokenizer.
- `prompt_eval_llm.py` defaults to model `deepseek-v4-flash` and reads `DEEPSEEK_API_KEY` from env.
- For BART, pass `--model facebook/bart-base` (HF auto-fetches). The repo no longer bundles local pretrained weights.
