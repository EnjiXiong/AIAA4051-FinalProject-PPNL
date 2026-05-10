# AIAA 4051 Final Project — Grid Path Planning with Language Models

**Languages:** **English** · [中文](README.zh-CN.md)

Prompting and fine-tuning seq2seq language models (T5, BART, Flan-T5) on the
**PPNL single-goal grid path planning benchmark** (Aghzal et al., ICLR 2024
Workshop). Given a natural-language description of a grid world, the model must
output a sequence of `up`/`down`/`left`/`right` actions that reaches the goal
without colliding with obstacles or leaving the grid.

This GitHub repo contains all code and small artifacts (data, results, plots).
The trained checkpoints (~25 GB across 13 runs) live in a companion
**Hugging Face mirror** of the full project tree:

> 📦 **Hugging Face mirror:**
> [`EnjiXiong/AIAA4051-FinalProject-PPNL`](https://huggingface.co/EnjiXiong/AIAA4051-FinalProject-PPNL)
> — full directory snapshot including every `best/` checkpoint and the upstream
> PPNL reference code.

---

## Headline results

Five-test-set evaluation (success rate). `OOD_novel` is a custom 1500-sample
test set spanning grid sizes 4×4–10×10 not seen during training; the others are
the canonical PPNL test sets.

| Method | ID 6×6 (seen) | ID 6×6 (unseen) | OOD 5×5 | OOD 7×7 | OOD 6×6 dense | OOD novel |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4 zero-shot prompting | 0.255 | — | — | 0.320 | — | 0.035 |
| Flan-T5 prompting (best) | low | — | — | low | — | — |
| **SFT** T5-base, vanilla, 6×6 only | 0.983 | 0.978 | 0.975 | 0.543 | 0.872 | — |
| **SFT** T5-base, structured input | 0.977 | 0.977 | 0.974 | 0.543 | 0.860 | — |
| **SFT** T5-base, **CoT** target | **0.987** | **0.987** | 0.982 | 0.548 | 0.896 | 0.117 |
| **Multi-scale SFT** T5-base, 5×5–7×7, 40 ep | 0.897 | 0.907 | 0.928 | 0.925 | 0.730 | 0.505 |
| **GRPO RL** on top of vanilla SFT | 0.609 | 0.613 | 0.693 | 0.534 | 0.447 | 0.201 |
| **Tree search (bw=4)** + multi-scale SFT | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **0.943** |
| **Tree search (bw=4)** + CoT SFT | **1.000** | **1.000** | **1.000** | 0.999 | **1.000** | 0.793 |

Take-aways: vanilla 6×6-only SFT collapses on size shift (54% on 7×7); training
on multiple grid sizes recovers most of that. The cleanest gain comes from
**inference-time tree search** that uses the model as a learned action prior
and the executor as a hard constraint — saturating ID/OOD success and bringing
truly novel grid sizes (e.g. 10×10) to **94%**. GRPO on top of SFT in our
configuration *hurt* greedy success despite training rewards going up — likely
exploration collapse on the small action vocabulary; flagged in the report.

Per-config raw numbers and per-sample predictions are in `results/`; see
[Results layout](#results-layout) below.

---

## Quick start

```bash
# 1. Environment (CUDA 12.8 build of torch — adjust index URL for your CUDA)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. Reproduce one of the headline numbers
python train.py --model t5-base --input_format cot --epochs 15 \
    --batch_size 16 --lr 3e-4 --max_target_len 256 --bf16
python tree_search_eval.py \
    --model_dir models/t5-base_cot_ep15_lr0.0003/best \
    --input_format cot --beam_width 4

# 3. Or skip training and pull a pretrained checkpoint from the HF mirror:
hf download EnjiXiong/AIAA4051-FinalProject-PPNL \
    --include "grid-path-planning/models/sft_multiscale_40ep/**/best/**" \
    --local-dir .
python tree_search_eval.py \
    --model_dir grid-path-planning/models/sft_multiscale_40ep/t5-base_vanilla_ep40_lr0.0003/best \
    --input_format vanilla --beam_width 4
```

---

## File-by-file reference

### Core library (imported by every entrypoint)

**`data_utils.py`** — Data loading + three input-format adapters.
- `load_ppnl_data(path)` reads a PPNL JSON list.
- `parse_nl_description(nl)` regex-parses a sentence like
  `"You are in a 6 by 6 world. There are obstacles ... at: (5,3). Go from (1,4) to (2,1)"`
  into `{grid_size, start, goal, obstacles}`.
- `format_vanilla` returns the NL string verbatim.
- `format_structured` reformats it as
  `Grid: 6x6 | Start: (1,4) | Goal: (2,1) | Obstacles: (5,3) | Output the shortest path ...`.
  Hypothesis tested in the report: a structured front matter cuts NL parsing
  noise for small encoder-decoders.
- `format_coordinate_tracking` produces a CoT target like
  `Start at (1,4) | left -> (1,3) | left -> (1,2) | ... | Done`. The model is
  forced to emit positions as it moves, so wrong-direction errors become
  visible to the loss.
- `PPNLDataset` is the torch `Dataset`. It silently filters samples whose
  ground truth is `Goal not reachable` so train/eval never sees them.
- `extract_actions_from_cot(cot)` recovers the bare action sequence from a CoT
  decoder output (used during eval of CoT-trained models).

**`evaluate_utils.py`** — The deterministic executor and metric aggregator.
- `simulate_path(grid, start, actions)` walks the action sequence and returns
  one of five statuses: `success`, `out_of_bounds`, `obstacle`, `wrong_end`,
  `format_error`. Concatenated tokens like `"upleft"` are auto-split first.
- `a_star_distance(grid, start, goal)` is the Manhattan-heuristic A\* used to
  measure (a) shortest-path length for the optimality metric and (b) residual
  distance to goal for the shaped RL reward.
- `evaluate_batch(data, predictions)` returns `success_rate`, `feasibility`
  (stayed in bounds and didn't hit obstacles, regardless of whether it reached
  the goal), `optimality` (`success AND len(pred) <= len(gt)`), `exact_match`,
  `avg_distance_to_goal`, and an error-type histogram.
- Grid encoding: `0=empty, 1=obstacle, 2=start, 3=goal`.

**`reward.py`** — GRPO scalar reward built on the same executor.

| Status | Reward |
|---|---:|
| `format_error` | −1.0 |
| `out_of_bounds`, `obstacle` | −0.5 |
| `wrong_end` (legal but didn't reach goal) | `0.2 · (1 − dist/(rows+cols))`, shaped |
| `success`, longer than GT | +0.8 |
| `success`, ≤ GT length (optimal) | +1.0 |

The shaped middle term is what lets GRPO improve from random — without it the
gradient is mostly zero on a hard task.

### Training entrypoints

**`train.py`** — Standard SFT loop for any HF seq2seq model.
- Selects bf16 (recommended on Ampere+) or fp16 mixed precision; T5 frequently
  produces NaN losses in fp16, so bf16 is the safe default.
- After each epoch, runs greedy validation, computes success rate via
  `evaluate_batch`, and saves the best-so-far checkpoint to
  `models/<run_name>/best/`. Final-epoch weights go to `final/`.
- Run-name format: `<model>_<input_format>_ep<E>_lr<LR>` (used by every other
  script and by `run_all.sh` to infer `--input_format`).
- Writes `training_history.json` (per-epoch loss + val metrics) and
  `config.json` (full args + best score) alongside the checkpoint.

**`train_rl.py`** — GRPO (Group Relative Policy Optimization) RL fine-tuning.
- Loads a frozen *reference* copy of the SFT-initialized policy for KL
  regularization, samples K trajectories per prompt at temperature τ, computes
  group-relative advantages `(rᵢ − mean) / std`, and applies PPO-style clipped
  policy-gradient loss with a `kl_coef · KL(π‖π_ref)` penalty.
- Requires `--sft_model PATH` pointing to a `train.py` output. CoT format is
  *not* supported (would make sampling decode unstable on intermediate
  coordinates).
- LR must be much smaller than SFT LR — defaults to 5e-7. Sample K=8 by
  default.

### Inference / evaluation entrypoints

**`run_eval.py`** — Greedy/beam-search evaluation of a checkpoint on the five
canonical PPNL test sets:
`ID_seen_6x6`, `ID_unseen_6x6`, `OOD_5x5`, `OOD_7x7`, `OOD_6x6_dense`.
- Pass `--extra_test NAME=path.json` to add custom sets (e.g.
  `OOD_novel=data/ood_novel_sizes.json`); add `--only_extra` to skip the
  built-ins. When a test set mixes grid sizes, prints a per-size breakdown.
- `--input_format` *must* match the format the checkpoint was trained with;
  for CoT models, predictions are auto-passed through
  `extract_actions_from_cot` before metric computation.
- Writes `results/<run_name>/eval_summary.json` (metric table) and, with
  `--save_predictions`, per-sample dumps `<set_name>_predictions.json`.

**`tree_search_eval.py`** — Constrained inference where the model only scores
the four action tokens at each step and the executor prunes illegal moves.
- Two decoder modes:
  - `vanilla`: each step extends the decoder prefix by exactly one action
    token.
  - `cot`: each step appends the action token *plus* the forced ` -> (r,c) |`
    coordinate continuation, so the CoT model's next-action distribution stays
    conditioned on its (correctly tracked) position.
- Beams are deduplicated by visited-position set to avoid burning beam capacity
  on cycles. Beam width is the only knob users typically tune; bw=4 saturates
  most test sets.
- This is what produces the headline `1.00` success rates above. The model
  acts as a *learned action prior*, the executor as a *hard constraint*.

**`prompt_eval.py`** — Prompting baselines using local Flan-T5 (no API).
Strategies: `zero_shot`, `few_shot` (3 worked examples), `cot_coordinate`
(track coordinates step by step), `cot_plan_then_act` (high-level plan first,
then actions), `cot_grid` (reconstruct ASCII grid first, then plan).

**`prompt_eval_llm.py`** — Prompting baselines using a frontier API model
(DeepSeek-V4 by default, OpenAI-compatible client). Three strategies:
`zero_shot`, `few_shot`, `cot`. Reads `DEEPSEEK_API_KEY` from env. Sub-samples
each test set (default 200) to keep API spend bounded; results are in
`results/llm_prompting/`.

### Data preparation utilities

**`generate_rl_envs.py`** — Synthesises diverse training environments across
configurable grid sizes / obstacle counts. Uses A\* to verify a valid path
exists and to record the optimal solution alongside each env (so the GRPO
reward can score optimality during sampling). Output:
`data/rl_envs_diverse.json`.

**`make_sft_subset.py`** — Random sub-samples the 16k training set down to
*N* (default 2 000) for the brief SFT warm-start that precedes GRPO. Filters
out `Goal not reachable` examples.

### Visualisation

**`visualize.py`** — Single-prediction grid plots and bulk error analysis from
a `*_predictions.json` file.

**`visualize_cases.py`** — Three-up case-study figures used in the report
(e.g. *Vanilla SFT vs. Multi-scale SFT* on the same OOD\_7×7 sample, *Vanilla
greedy vs. Tree search* on novel sizes). Output PDFs/PNGs are pre-rendered in
`visualizations/`.

### Convenience

**`run_all.sh`** — Phase 1 fine-tunes the four headline SFT configs, Phase 2
evaluates every `models/*/best/` on every canonical test set, Phase 3 runs the
five Flan-T5 prompting strategies on `ID_seen_6x6` and `OOD_7x7`. Set
`--input_format` is inferred by substring-matching the run-name (`structured`,
`cot`, else `vanilla`).

---

## Data layout (`data/`)

| File | Source | Purpose |
|---|---|---|
| `1_train_set_6x6_samples.json` (16 032) | PPNL upstream | Main SFT training data |
| `1dev_set_6x6_samples.json` (2 004) | PPNL upstream | Validation (used by `train.py` for best-checkpoint selection) |
| `1_goals_test_seen_6x6_samples.json` | PPNL | ID test, environments seen at training |
| `1goals_unseen_6x6_samples.json` | PPNL | ID test, unseen environments same size |
| `1_goals_test_unseen_5x5_samples.json` | PPNL | OOD: smaller grid |
| `1_goals_test_unseen_7x7_samples.json` | PPNL | OOD: larger grid |
| `1_goals_test_unseen_6x6more_obstacles_samples.json` | PPNL | OOD: denser obstacles |
| `1_train_set_6x6_samples_small2k.json` | from `make_sft_subset.py` | 2 000-sample SFT warm-start for GRPO |
| `rl_envs_diverse.json` | from `generate_rl_envs.py` | Multi-size env pool for GRPO |
| `ood_novel_sizes.json` | this project | Custom OOD: 1500 envs at sizes not in PPNL (4×4, 8×8, 9×9, 10×10) |

Each PPNL sample has the schema:
```json
{
  "world": [[0, ...], ...],          // 2D grid, 0=empty 1=obstacle 2=start 3=goal
  "nl_description": "You are in a 6 by 6 world. ...",
  "solution_coordinates": [[r, c], ...],  // shortest-path waypoints
  "agent_as_a_point": "left left left down ",  // ground-truth action sequence
  ...
}
```

---

## Results layout (`results/`)

```
results/
├── exp_A/             — Vanilla SFT, 6×6 only, 15 ep                (baseline)
├── exp_C/             — Multi-scale SFT, 5×5–7×7 mix, 40 ep
├── exp_cot/           — CoT SFT, 6×6 only, 15 ep
├── exp_cot_tree/      — CoT SFT + tree search                       (decoder-mode A/B)
├── exp_D/             — Multi-scale SFT, warm-start, 5 ep
├── exp_E/             — GRPO RL on top of vanilla SFT
├── exp_G/             — Multi-scale SFT + tree search bw4 / bw8
├── llm_prompting/     — DeepSeek-V4 prompting (zero/few/CoT)
├── t5-base_vanilla_ep15_lr0.0003/    — direct run_eval.py output
└── t5-base_structured_ep15_lr0.0003/ — direct run_eval.py output
```

Each subdir contains an `eval_summary*.json` (the metric tables aggregated in
the headline table above) and per-test-set `*_predictions.json` files that
include the natural-language description, ground truth, model prediction,
success flag, and error type for every sample. These are what the failure-case
visualizations in `visualizations/` are drawn from.

---

## Pretrained models

The trained checkpoints (~25 GB total) live on the Hugging Face mirror.
Pull a single one with:

```bash
hf download EnjiXiong/AIAA4051-FinalProject-PPNL \
    --include "grid-path-planning/models/<run_name>/**/best/**" \
    --local-dir .
```

…then point `run_eval.py` / `tree_search_eval.py` at
`grid-path-planning/models/<run_name>/.../best`.

We do *not* ship `facebook/bart-base` weights in either repo — pass
`--model facebook/bart-base` to `train.py` and HF will fetch them.

---

## Citation / credit

This project builds on the PPNL benchmark and uses its data and upstream
executor verbatim:

> Aghzal, M., Plaku, E., & Yao, Z. (2024). *Can Large Language Models be Good
> Path Planners? A Benchmark and Investigation on Spatial-temporal Reasoning.*
> ICLR 2024 Workshop on LLM Agents.
> [GitHub: MohamedAghzal/llms-as-path-planners](https://github.com/MohamedAghzal/llms-as-path-planners) ·
> [Paper](https://arxiv.org/abs/2310.03249)

The reference upstream tree (`llms-as-path-planners/`) is included in the
Hugging Face mirror for completeness; only this project's code is in the
GitHub repo. The upstream `evaluate/` scripts are kept under `grid-path-planning/evaluate/`
for traceability against the canonical scoring convention.
