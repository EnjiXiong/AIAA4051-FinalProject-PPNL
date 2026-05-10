# AIAA 4051 期末项目 — 基于语言模型的网格路径规划

**语言：** [English](README.md) · **中文**

本项目研究在 **PPNL 单目标网格路径规划基准**（Aghzal 等，ICLR 2024 Workshop）上对
seq2seq 语言模型（T5、BART、Flan-T5）进行提示与微调。模型接收对一个网格世界的自然
语言描述，需要输出一串 `up`/`down`/`left`/`right` 动作序列，从起点到达目标位置，
且不能撞上障碍物或越界。

本 GitHub 仓库包含全部代码与小体积构件（数据、评估结果、可视化）。13 个训练好的
模型权重（共约 25 GB）放在配套的 **Hugging Face 镜像仓库**中：

> 📦 **Hugging Face 镜像：**
> [`EnjiXiong/AIAA4051-FinalProject-PPNL`](https://huggingface.co/EnjiXiong/AIAA4051-FinalProject-PPNL)
> — 完整目录快照，包含每个 `best/` 检查点以及 PPNL 上游参考代码。

---

## 关键结果

五个测试集的成功率（Success Rate）。其中 `OOD_novel` 是我们自建的 1500 样本测试
集，覆盖训练时未见过的网格尺寸（4×4–10×10）；其余为 PPNL 标准测试集。

| 方法 | ID 6×6（已见） | ID 6×6（未见） | OOD 5×5 | OOD 7×7 | OOD 6×6 密集 | OOD novel |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4 零样本提示 | 0.255 | — | — | 0.320 | — | 0.035 |
| Flan-T5 提示（最佳） | 较低 | — | — | 较低 | — | — |
| **SFT** T5-base，原始输入，仅 6×6 | 0.983 | 0.978 | 0.975 | 0.543 | 0.872 | — |
| **SFT** T5-base，结构化输入 | 0.977 | 0.977 | 0.974 | 0.543 | 0.860 | — |
| **SFT** T5-base，**CoT** 目标 | **0.987** | **0.987** | 0.982 | 0.548 | 0.896 | 0.117 |
| **多尺度 SFT** T5-base，5×5–7×7，40 ep | 0.897 | 0.907 | 0.928 | 0.925 | 0.730 | 0.505 |
| **GRPO 强化学习** 在原始 SFT 基础上 | 0.609 | 0.613 | 0.693 | 0.534 | 0.447 | 0.201 |
| **树搜索 (bw=4)** + 多尺度 SFT | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **0.943** |
| **树搜索 (bw=4)** + CoT SFT | **1.000** | **1.000** | **1.000** | 0.999 | **1.000** | 0.793 |

主要发现：仅在 6×6 数据上做原始 SFT 在尺寸偏移下急剧崩溃（7×7 上仅 54%）；在
多种网格尺寸上联合训练能挽回大部分性能。最显著的提升来自 **推理阶段的树搜索** ——
模型作为已学习的动作先验，执行器作为硬约束，把所有 ID/OOD 测试集打满，并把全新
网格尺寸（如 10×10）也提到 **94%**。在我们这个配置下，GRPO 反而拉低了贪心解码下
的成功率，尽管训练奖励一直在上升 —— 推测是在小动作词表上的探索坍缩；这一点已在
报告中说明。

每个配置的原始数值与逐样本预测都在 `results/` 下，详见 [结果组织](#结果组织results)。

---

## 快速开始

```bash
# 1. 环境（这里是 CUDA 12.8 的 torch 版本，请按你的 CUDA 调整 index URL）
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. 复现表中某一行
python train.py --model t5-base --input_format cot --epochs 15 \
    --batch_size 16 --lr 3e-4 --max_target_len 256 --bf16
python tree_search_eval.py \
    --model_dir models/t5-base_cot_ep15_lr0.0003/best \
    --input_format cot --beam_width 4

# 3. 或者直接从 HF 镜像下载训练好的权重，跳过训练：
hf download EnjiXiong/AIAA4051-FinalProject-PPNL \
    --include "grid-path-planning/models/sft_multiscale_40ep/**/best/**" \
    --local-dir .
python tree_search_eval.py \
    --model_dir grid-path-planning/models/sft_multiscale_40ep/t5-base_vanilla_ep40_lr0.0003/best \
    --input_format vanilla --beam_width 4
```

---

## 文件逐一说明

### 核心库（被所有入口脚本导入）

**`data_utils.py`** — 数据加载与三种输入格式适配器。
- `load_ppnl_data(path)` 读取 PPNL JSON 列表。
- `parse_nl_description(nl)` 用正则把
  `"You are in a 6 by 6 world. There are obstacles ... at: (5,3). Go from (1,4) to (2,1)"`
  这样的句子解析成 `{grid_size, start, goal, obstacles}`。
- `format_vanilla` 直接返回原始自然语言字符串。
- `format_structured` 把它重排为
  `Grid: 6x6 | Start: (1,4) | Goal: (2,1) | Obstacles: (5,3) | Output the shortest path ...`。
  这是报告中要验证的一个假设：结构化前置信息能减少小型 encoder-decoder
  在自然语言解析上的噪声。
- `format_coordinate_tracking` 生成 CoT 目标序列，例如
  `Start at (1,4) | left -> (1,3) | left -> (1,2) | ... | Done`。
  模型被强制在每一步同时输出位置，从而把"走错方向"这类错误暴露给损失函数。
- `PPNLDataset` 是 PyTorch 的 `Dataset`，会自动过滤真值是 `Goal not reachable`
  的样本，因此训练 / 评估阶段都看不到这种样本。
- `extract_actions_from_cot(cot)` 从 CoT 解码输出里抽出纯动作序列（用于 CoT
  模型的评估阶段）。

**`evaluate_utils.py`** — 确定性执行器与指标聚合器。
- `simulate_path(grid, start, actions)` 顺着动作序列走一遍，返回五种状态之一：
  `success`、`out_of_bounds`、`obstacle`、`wrong_end`、`format_error`。
  `"upleft"` 这类被串到一起的 token 会先自动拆开。
- `a_star_distance(grid, start, goal)` 是用曼哈顿启发式的 A\*；用来计算
  (a) 最优性指标里的最短路径长度，(b) RL 形状化奖励里的"剩余距离"。
- `evaluate_batch(data, predictions)` 返回 `success_rate`、`feasibility`
  （是否始终在边界内、不撞障碍物，不管是否到达目标）、`optimality`
  （`success 且 len(pred) <= len(gt)`）、`exact_match`、
  `avg_distance_to_goal`，以及一张错误类型直方图。
- 网格编码：`0=空`、`1=障碍`、`2=起点`、`3=目标`。

**`reward.py`** — 基于上面同一个执行器的 GRPO 标量奖励。

| 状态 | 奖励 |
|---|---:|
| `format_error` | −1.0 |
| `out_of_bounds`、`obstacle` | −0.5 |
| `wrong_end`（合法路径但没到目标） | `0.2 · (1 − dist/(rows+cols))`，形状化 |
| `success`，比真值长 | +0.8 |
| `success`，长度 ≤ 真值（即最优） | +1.0 |

中间这一档形状化奖励是 GRPO 能从随机起点学起来的关键 —— 没有它的话在这个任务
上梯度大部分时候是零。

### 训练入口

**`train.py`** — 任意 HF seq2seq 模型的标准 SFT 训练循环。
- 支持 bf16（Ampere+ 推荐）和 fp16 混合精度；T5 在 fp16 下经常 NaN，因此
  默认且推荐使用 bf16。
- 每个 epoch 结束后跑一次贪心验证，用 `evaluate_batch` 计算成功率，把当前最优
  保存到 `models/<run_name>/best/`，最终 epoch 的权重保存到 `final/`。
- 运行命名格式：`<model>_<input_format>_ep<E>_lr<LR>`（其他脚本以及
  `run_all.sh` 都依赖这个名字反推 `--input_format`）。
- 同时会写出 `training_history.json`（每 epoch loss 与验证指标）和
  `config.json`（完整 args 与最佳分数）。

**`train_rl.py`** — GRPO（Group Relative Policy Optimization）强化学习微调。
- 加载一份冻结的 *参考模型*（参与 KL 正则项），在温度 τ 下对每个 prompt 采
  K 条轨迹，计算群组相对优势 `(rᵢ − mean) / std`，用 PPO 风格的 clipped
  策略梯度损失 + `kl_coef · KL(π‖π_ref)` 惩罚项更新策略。
- 必须通过 `--sft_model PATH` 指向 `train.py` 的输出。CoT 格式 *不支持*
  （否则采样解码在中间坐标位置会非常不稳定）。
- 学习率必须远小于 SFT 学习率 —— 默认 5e-7。默认每个 prompt 采 K=8 条。

### 推理 / 评估入口

**`run_eval.py`** — 在五个 PPNL 标准测试集上对一个检查点做贪心 / 束搜索评估：
`ID_seen_6x6`、`ID_unseen_6x6`、`OOD_5x5`、`OOD_7x7`、`OOD_6x6_dense`。
- 通过 `--extra_test NAME=path.json` 加自定义测试集（例如
  `OOD_novel=data/ood_novel_sizes.json`）；加 `--only_extra` 跳过内置测试集。
  测试集网格尺寸混合时会自动按尺寸打印分组结果。
- `--input_format` 必须 *和训练时一致*；CoT 模型的预测会自动经过
  `extract_actions_from_cot` 转换后再算指标。
- 输出 `results/<run_name>/eval_summary.json`（指标表）；加
  `--save_predictions` 还会保存逐样本结果 `<set_name>_predictions.json`。

**`tree_search_eval.py`** — 受约束推理：模型每一步只对四个动作 token 打分，
执行器实时剪掉所有非法动作。
- 两种解码模式：
  - `vanilla`：每一步只在 decoder 前缀后扩展一个动作 token。
  - `cot`：每一步在动作 token 之后强制追加 ` -> (r,c) |` 这段坐标延续，
    使 CoT 模型的下一动作分布始终建立在（被正确追踪的）当前位置之上。
- 各 beam 按"已访问位置集合"去重，避免束容量浪费在循环路径上。一般只需要调
  beam 宽度；bw=4 已能打满大部分测试集。
- 上面表格里的 `1.00` 成功率正是这种推理方式产出的：模型作为 *学习到的动作
  先验*，执行器作为 *硬约束*。

**`prompt_eval.py`** — 用本地 Flan-T5（无需 API）的提示基线。策略：`zero_shot`、
`few_shot`（3 个已解决示例）、`cot_coordinate`（逐步追踪坐标）、`cot_plan_then_act`
（先给高层计划再给动作）、`cot_grid`（先重建 ASCII 网格再规划）。

**`prompt_eval_llm.py`** — 用前沿 API 模型（默认 DeepSeek-V4，OpenAI 兼容
客户端）的提示基线。三种策略：`zero_shot`、`few_shot`、`cot`。
从环境变量 `DEEPSEEK_API_KEY` 读取 key。每个测试集会下采样（默认 200）
来控制 API 花销；结果在 `results/llm_prompting/`。

### 数据准备工具

**`generate_rl_envs.py`** — 在多种网格尺寸 / 障碍数下合成多样化的训练环境。
通过 A\* 验证存在合法路径，并在每个 env 旁记录最优解（这样 GRPO 奖励在采样
时就能判断最优性）。输出：`data/rl_envs_diverse.json`。

**`make_sft_subset.py`** — 把 16k 训练集随机下采样到 *N* 条
（默认 2000），用于 GRPO 之前那次简短的 SFT 预热。会过滤
`Goal not reachable` 样本。

### 可视化

**`visualize.py`** — 从某个 `*_predictions.json` 出发，画单条预测的网格图，
也支持批量错误分析。

**`visualize_cases.py`** — 报告中用的三联对比图（例如同一个 OOD\_7×7 样本下
*原始 SFT vs. 多尺度 SFT*，以及在 novel 尺寸上的 *原始贪心 vs. 树搜索*）。
PDF / PNG 已预先渲染在 `visualizations/` 下。

### 便利脚本

**`run_all.sh`** — 阶段 1 训练四个主打 SFT 配置，阶段 2 在每个标准测试集上
评估 `models/*/best/` 下的所有检查点，阶段 3 跑五种 Flan-T5 提示策略
（`ID_seen_6x6` 与 `OOD_7x7`）。`--input_format` 通过对 run-name
做子串匹配来推断（`structured`、`cot`，否则默认 `vanilla`）。

---

## 数据组织（`data/`）

| 文件 | 来源 | 用途 |
|---|---|---|
| `1_train_set_6x6_samples.json`（16 032） | PPNL 上游 | 主 SFT 训练集 |
| `1dev_set_6x6_samples.json`（2 004） | PPNL 上游 | 验证集（`train.py` 用来选最佳检查点） |
| `1_goals_test_seen_6x6_samples.json` | PPNL | ID 测试，环境训练时见过 |
| `1goals_unseen_6x6_samples.json` | PPNL | ID 测试，未见环境同尺寸 |
| `1_goals_test_unseen_5x5_samples.json` | PPNL | OOD：更小的网格 |
| `1_goals_test_unseen_7x7_samples.json` | PPNL | OOD：更大的网格 |
| `1_goals_test_unseen_6x6more_obstacles_samples.json` | PPNL | OOD：更密集的障碍物 |
| `1_train_set_6x6_samples_small2k.json` | 由 `make_sft_subset.py` 生成 | 给 GRPO 用的 2 000 样本 SFT 预热集 |
| `rl_envs_diverse.json` | 由 `generate_rl_envs.py` 生成 | GRPO 用的多尺寸环境池 |
| `ood_novel_sizes.json` | 本项目 | 自建 OOD：1500 个 PPNL 中没有的尺寸（4×4、8×8、9×9、10×10） |

每条 PPNL 样本的字段：

```json
{
  "world": [[0, ...], ...],          // 二维网格，0=空 1=障碍 2=起点 3=目标
  "nl_description": "You are in a 6 by 6 world. ...",
  "solution_coordinates": [[r, c], ...],  // 最短路径上的坐标点
  "agent_as_a_point": "left left left down ",  // 真值动作序列
  ...
}
```

---

## 结果组织（`results/`）

```
results/
├── exp_A/             — 原始 SFT，仅 6×6，15 ep                   （baseline）
├── exp_C/             — 多尺度 SFT，5×5–7×7 混合，40 ep
├── exp_cot/           — CoT SFT，仅 6×6，15 ep
├── exp_cot_tree/      — CoT SFT + 树搜索                          （A/B 解码模式）
├── exp_D/             — 多尺度 SFT，预热，5 ep
├── exp_E/             — 在原始 SFT 上叠 GRPO 强化学习
├── exp_G/             — 多尺度 SFT + 树搜索 bw4 / bw8
├── llm_prompting/     — DeepSeek-V4 提示（zero/few/CoT）
├── t5-base_vanilla_ep15_lr0.0003/    — run_eval.py 直接产物
└── t5-base_structured_ep15_lr0.0003/ — run_eval.py 直接产物
```

每个子目录里都有一个 `eval_summary*.json`（指标表，对应上面那张总表里的某些行），
以及每个测试集的 `*_predictions.json` —— 后者保存了每条样本的自然语言描述、
真值、模型预测、是否成功、错误类型。`visualizations/` 里的失败案例图就是从
这些预测文件里挑出来画的。

---

## 预训练模型

训练好的检查点（合计约 25 GB）在 Hugging Face 镜像仓库里。下载某一个：

```bash
hf download EnjiXiong/AIAA4051-FinalProject-PPNL \
    --include "grid-path-planning/models/<run_name>/**/best/**" \
    --local-dir .
```

下载完成后把 `run_eval.py` / `tree_search_eval.py` 的 `--model_dir`
指向 `grid-path-planning/models/<run_name>/.../best` 即可。

我们 *不* 把 `facebook/bart-base` 的权重打进任一仓库 —— 直接给 `train.py`
传 `--model facebook/bart-base`，HF 会自动下载。

---

## 引用与致谢

本项目基于 PPNL 基准，并直接使用了它的数据与上游执行器：

> Aghzal, M., Plaku, E., & Yao, Z. (2024). *Can Large Language Models be Good
> Path Planners? A Benchmark and Investigation on Spatial-temporal Reasoning.*
> ICLR 2024 Workshop on LLM Agents.
> [GitHub: MohamedAghzal/llms-as-path-planners](https://github.com/MohamedAghzal/llms-as-path-planners) ·
> [论文](https://arxiv.org/abs/2310.03249)

PPNL 上游代码树（`llms-as-path-planners/`）作为完整存档的一部分被放在了
Hugging Face 镜像仓库；GitHub 这边只保留本项目自己的代码。
`grid-path-planning/evaluate/` 下保留了上游的评估脚本，便于追溯与官方评分
约定的一致性。
