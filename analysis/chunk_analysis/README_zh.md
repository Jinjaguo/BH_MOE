# Chunk-wise Hidden-state Failure Analysis 中文说明

这份文档说明 `/home/jinjaguo/BH_MOE/analysis/chunk_analysis` 下这一组脚本的整体目的、输入输出路径、运行方式、每个实验在回答什么问题，以及如何把 soft success 和 causal intervention 的结果纳入同一套分析流程。

当前项目的核心任务是：

```text
put_the_cream_cheese_on_the_plate
```

对应 BDDL 文件：

```text
/home/jinjaguo/BH_MOE/custom_bddl/libero_goal/dif_start_end_loc/put_the_cream_cheese_on_the_plate.bddl
```

推荐 Python 环境：

```bash
/home/jinjaguo/anaconda3/envs/libero/bin/python
```

工作目录默认使用：

```bash
cd /home/jinjaguo/BH_MOE
```

---

## 1. 项目总目标

这个项目不是单纯统计 rollout 成功率，而是分析 OpenPI / LIBERO policy 在每个 action chunk 内部 hidden representation 的变化，回答几个研究问题：

1. 成功和失败轨迹的 hidden state 是否从早期就开始分叉？
2. 这种分叉发生在哪一个 normalized task progress 位置？
3. 哪一个内部 state 对 failure 最敏感？
4. failure 是语义 grounding 早期偏移，还是后期控制 / placement 误差？
5. 失败轨迹变长、重复尝试、policy collapse 是否能从 representation drift 里预测？
6. 如果把 failure rollout 早期 hidden state 替换成 success hidden state，能否把 trajectory 拉回成功 manifold？
7. LIBERO strict `On` 判定过严时，如何保留 benchmark 可复现性，同时用 soft success 得到更符合视频观察的分析标签？

因此整个工程分成四层：

```text
原始 rollout + hidden-state chunk 记录
        ↓
数据体检与 normalized-time 表示构建
        ↓
chunk-wise divergence / AUC / onset / transition / phase / stretch 分析
        ↓
soft success 与 causal intervention rollout 复跑，再进入同一套分析
```

---

## 2. 关键目录总览

### 2.1 脚本目录

```text
/home/jinjaguo/BH_MOE/analysis/chunk_analysis
```

这里保存所有分析脚本：

```text
chunk_analysis_common.py
01_data_health_check.py
02_divergence_heatmap.py
03_linear_probe_auc.py
04_failure_onset.py
05_transition_analysis.py
06_phase_structure_discovery.py
07_time_stretch_analysis.py
08_start_causal_intervention_server.py
09_run_causal_intervention_rollouts.py
10_run_soft_success_rollouts.py
README_zh.md
```

### 2.2 原始 hidden-state chunk 数据目录

```text
/home/jinjaguo/BH_MOE/OOD_exp/outputs/chunk_wise
```

一个 task 对应一个子目录。例如原始 strict LIBERO 判定数据：

```text
/home/jinjaguo/BH_MOE/OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate
```

soft success 或 causal intervention 复跑结果也会保存成新的 task 目录，例如：

```text
/home/jinjaguo/BH_MOE/OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate_soft_success
/home/jinjaguo/BH_MOE/OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate__causal_intervention
```

每个 task 目录下面是 trial：

```text
trial_0/
trial_1/
trial_2/
...
```

每个 trial 目录中最重要的文件：

```text
chunk_00000.pt
chunk_00001.pt
...
rollouts_finalize.jsonl
rollouts_state_record.jsonl
rollouts_soft_success.jsonl   # 只有 soft-success runner 会生成
```

### 2.3 分析结果目录

所有图、CSV、NPY、metadata 默认保存到：

```text
/home/jinjaguo/BH_MOE/analysis/chunk_analysis/results
```

建议每个 task 单独一个结果目录。例如：

```text
/home/jinjaguo/BH_MOE/analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate
/home/jinjaguo/BH_MOE/analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate_soft_success
```

### 2.4 视频结果目录

原始或 OOD rollout 视频通常在：

```text
/home/jinjaguo/BH_MOE/OOD_exp/outputs/videos
```

soft success stopper 视频：

```text
/home/jinjaguo/BH_MOE/OOD_exp/outputs/videos/soft_success/<task_name>
```

causal intervention 视频：

```text
/home/jinjaguo/BH_MOE/OOD_exp/outputs/videos/causal_intervention/<task_name>
```

---

## 3. Hidden-state chunk 文件格式

每个 `chunk_XXXXX.pt` 是一个 PyTorch 保存的 dict，分析脚本默认读取以下 5 个 state：

```text
text_encoder_output
cross_attention_action_x_after
action_head_input
chunk_vector_mean
action_chunk_output
```

这些字段的语义大致是：

| state | 含义 | 分析价值 |
|---|---|---|
| `text_encoder_output` | 文本 / prompt 编码后的 token-level 表示 | 检查 instruction grounding 是否稳定 |
| `cross_attention_action_x_after` | action token 经过 expert cross-attention 后的表示 | 检查视觉-语言到动作 token 的融合状态 |
| `action_head_input` | action head 前的 hidden state | 通常对后续 action 生成最直接 |
| `chunk_vector_mean` | `action_head_input` 在 token / horizon 维度上的平均摘要 | 低维摘要，常用于快速比较 chunk 表示 |
| `action_chunk_output` | policy 输出的 action chunk | 更接近控制输出，可检查 action-level failure |

不同 state 可能是不同 shape，例如：

```text
(1, 10, 1024)
(1, 50, 1024)
(1, 7)
```

分析时会统一做 vector 化：

1. squeeze batch 维度。
2. 如果是 token-level / sequence-level tensor，例如 `(1, 10, 1024)`，对 token 维度取 mean，得到 `(1024,)`。
3. 每个 chunk、每个 state 最终变成一个固定向量。

---

## 4. 为什么必须做时间归一化

原始成功轨迹通常接近 20 个 chunk，但失败轨迹长度可能差异很大，例如：

```text
20, 21, 22, 23, 26, 54, 60
```

这说明 failure 里可能存在：

```text
行为停滞
重复尝试
policy collapse
长时间没有达到成功判定
```

因此不能直接按原始 chunk index 对齐。否则 `chunk 10` 对于短轨迹可能已经接近结尾，对于 60 chunk 失败轨迹可能还在早期重复阶段。

本项目统一使用 normalized time axis：

```text
tau ∈ [0, 1]
target_length = 20
```

对任意长度为 `L` 的 trajectory，映射到 20 个 normalized chunk：

```text
tau_i = i / (T - 1), i = 0 ... T-1
source_index = tau_i * (L - 1)
```

然后使用线性插值：

```text
h_norm[i] = linear_interpolate(h_raw[floor(source_index)], h_raw[ceil(source_index)])
```

这样所有 trajectory 都被映射为：

```text
20 normalized chunks
```

之后再做 cosine divergence、linear probe AUC、failure onset、transition 和 phase structure 才是有效的。

---

## 5. 核心分析脚本

下面所有命令都假设你在：

```bash
cd /home/jinjaguo/BH_MOE
```

并使用：

```bash
/home/jinjaguo/anaconda3/envs/libero/bin/python
```

### 5.1 公共工具：`chunk_analysis_common.py`

路径：

```text
analysis/chunk_analysis/chunk_analysis_common.py
```

这是共享工具模块，不需要直接运行。它负责：

```text
发现 trial_* 目录
读取 rollouts_finalize.jsonl / rollouts_state_record.jsonl 中的 success 标签
加载 chunk_*.pt
把每个 state 转成固定向量
执行 normalized-time 线性插值
构建 fused representation
计算 cosine distance
保存 heatmap / CSV / JSON
```

重要函数：

| 函数 | 作用 |
|---|---|
| `discover_trials(root)` | 扫描 task 目录，得到 trial id、success label、chunk 文件列表 |
| `select_balanced_trials(...)` | 选择前 20 个 success 和前 20 个 failure |
| `state_to_vector(...)` | 把任意 hidden tensor 压成一个向量 |
| `linear_interpolate_sequence(...)` | 按 `tau ∈ [0,1]` 做线性插值 |
| `load_group_normalized_state_tensors(...)` | 加载一组 trial，并统一到 fixed length |
| `make_fused_from_states(...)` | 每个 state 先 L2 normalize，再 concat 成 fused representation |

---

### 5.2 第一阶段：数据体检 `01_data_health_check.py`

路径：

```text
analysis/chunk_analysis/01_data_health_check.py
```

目的：

在做任何对齐和统计分析前，先检查原始数据是否健康。这个脚本不会做时间归一化对齐，而是直接检查原始 trajectory。

它会回答：

```text
成功和失败 trial 的 chunk 数是否一致？
每个 hidden state 的 shape 是否一致？
是否存在 NaN？
是否存在 Inf？
是否存在全零 tensor？
每个 state 的 L2 norm 分布是什么样？
success 和 failure 的 activation magnitude 是否天然尺度不同？
是否有 norm 爆炸或塌缩？
```

运行：

```bash
/home/jinjaguo/anaconda3/envs/libero/bin/python \
  analysis/chunk_analysis/01_data_health_check.py \
  --root OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate \
  --out_dir analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate \
  --n_success 20 \
  --n_failure 20
```

输出目录：

```text
analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate/data_health
```

输出文件：

| 文件 | 含义 |
|---|---|
| `trial_summary.csv` | 每个 trial 的 success label、json 记录 chunk 数、实际 chunk 文件数 |
| `state_health_summary.csv` | 每个 state 的 shape 统计、NaN/Inf/zero 统计、norm mean/std、异常 norm 数量 |
| `norm_distribution_<state>.png` | 每个 state 的 success/failure L2 norm 分布图 |
| `health_metadata.json` | 本次选择的 success/failure trial id、共同 chunk ids、原始 chunk counts |

解读重点：

1. 如果 failure 的 chunk 数跨度很大，必须使用后续脚本的 normalized-time 插值。
2. 如果某个 state 的 norm 在 failure 中显著爆炸或塌缩，这本身就是强现象。
3. 如果 success/failure norm 尺度差别巨大，后续 cosine distance 比 L2 distance 更可靠。

---

### 5.3 第二阶段：cosine divergence heatmap `02_divergence_heatmap.py`

路径：

```text
analysis/chunk_analysis/02_divergence_heatmap.py
```

目的：

构建每个 normalized chunk、每个 state 的 success/failure 均值表示，然后计算二者的 cosine distance。

数学定义：

```text
S[c, s] = success trajectories 在 normalized chunk c、state s 的 embedding
F[c, s] = failure trajectories 在 normalized chunk c、state s 的 embedding
D[c, s] = cosine_distance(mean(S[c, s]), mean(F[c, s]))
```

输出核心矩阵：

```text
D ∈ R^{20 × 5}
```

同时构建 fused representation：

```text
每个 state 先 L2 normalize
然后 concat 5 个 state
Z_fused ∈ R^{20 trials × 20 chunks × 5d}
```

运行：

```bash
/home/jinjaguo/anaconda3/envs/libero/bin/python \
  analysis/chunk_analysis/02_divergence_heatmap.py \
  --root OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate \
  --out_dir analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate \
  --n_success 20 \
  --n_failure 20 \
  --target_length 20
```

输出目录：

```text
analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate/divergence
```

输出文件：

| 文件 | 含义 |
|---|---|
| `cosine_divergence_statewise.csv` | `20 × 5` 的 state-wise cosine divergence |
| `cosine_divergence_statewise.png` | failure divergence heatmap |
| `cosine_divergence_fused.csv` | fused concat representation 的 divergence |
| `cosine_divergence_fused.png` | fused divergence heatmap |
| `Z_success_<state>.npy` | success normalized-time state tensor |
| `Z_failure_<state>.npy` | failure normalized-time state tensor |
| `Z_fused_success.npy` | success fused tensor |
| `Z_fused_failure.npy` | failure fused tensor |
| `divergence_metadata.json` | trial ids、原始 chunk counts、normalized tau、shape 信息 |

解读重点：

1. 如果某一列 state 在早期 chunk 变亮，说明该 state 很早就区分 success/failure。
2. 如果所有 state 都在后期才变亮，失败可能更多来自后期执行或 placement。
3. 如果某个 state 始终不亮，说明它对该 failure mode 不敏感。

---

### 5.4 第三阶段：linear probe AUC `03_linear_probe_auc.py`

路径：

```text
analysis/chunk_analysis/03_linear_probe_auc.py
```

目的：

只看 mean divergence 不够。这个脚本对每个 `(chunk, state)` 单独训练一个 logistic regression linear probe，判断该 hidden embedding 是否能预测 success/failure。

输入：

```text
x = chunk/state embedding
y = success or failure
```

输出：

```text
AUC[c, s] ∈ R^{20 × 5}
```

默认使用 5-fold stratified cross validation。如果设置 `--cv 1` 或 `--cv >= 样本数`，会退化为 leave-one-out。

运行：

```bash
/home/jinjaguo/anaconda3/envs/libero/bin/python \
  analysis/chunk_analysis/03_linear_probe_auc.py \
  --root OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate \
  --out_dir analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate \
  --n_success 20 \
  --n_failure 20 \
  --target_length 20 \
  --cv 5
```

输出目录：

```text
analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate/linear_probe_auc
```

输出文件：

| 文件 | 含义 |
|---|---|
| `auc_statewise.csv` | `20 × 5` 的 state-wise AUC |
| `auc_statewise.png` | AUC heatmap |
| `auc_fused.csv` | fused concat representation 的 AUC |
| `auc_fused.png` | fused AUC heatmap |
| `probe_metadata.json` | CV 参数、trial ids、chunk counts、shape 信息 |

解读重点：

1. `AUC ≈ 0.5`：该位置 hidden state 基本不能区分 success/failure。
2. `AUC` 明显高于 0.5：该位置具有 failure-diagnostic 信息。
3. 最强证据来自 divergence 和 AUC 的交集：某个 chunk/state 同时高 divergence、高 AUC，说明它不仅均值不同，而且可预测成败。

---

### 5.5 第四阶段：failure onset `04_failure_onset.py`

路径：

```text
analysis/chunk_analysis/04_failure_onset.py
```

目的：

找到 success/failure representation 最早稳定分叉的位置，也就是 failure onset。

它读取前两个脚本生成的：

```text
divergence/cosine_divergence_statewise.csv
linear_probe_auc/auc_statewise.csv
```

然后定义：

```text
auc_effect = abs(AUC - 0.5) * 2
onset_score = minmax(cosine_divergence) + minmax(auc_effect)
```

为了避免单点噪声，要求连续若干个 chunk 超过阈值：

```text
first c where onset_score[c:c+consecutive] >= threshold
```

运行：

```bash
/home/jinjaguo/anaconda3/envs/libero/bin/python \
  analysis/chunk_analysis/04_failure_onset.py \
  --out_dir analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate \
  --threshold 1.2 \
  --consecutive 2
```

输出目录：

```text
analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate/failure_onset
```

输出文件：

| 文件 | 含义 |
|---|---|
| `onset_score_statewise.csv` | 每个 normalized chunk/state 的 onset score |
| `onset_score_statewise.png` | onset score heatmap |
| `onset_summary.csv` | 每个 state 的最早 onset chunk 和 onset tau |
| `onset_metadata.json` | 阈值、连续 chunk 数、来源文件、normalized tau |

解读重点：

1. onset 在 `tau ≈ 0.0 - 0.15`：可能是 instruction grounding / early semantic error。
2. onset 在中段：可能是 grasp → move 或 move → place 的 phase transition 出错。
3. onset 在末端：可能是 representation 前面基本正确，最后 placement / release 失败。

---

### 5.6 第五阶段：intra-trajectory transition `05_transition_analysis.py`

路径：

```text
analysis/chunk_analysis/05_transition_analysis.py
```

目的：

分析每条 trajectory 内部相邻 chunk 的 hidden representation 是否出现 regime shift。

定义：

```text
sim[c] = cosine(h[c], h[c+1])
delta[c] = 1 - sim[c]
```

先对每条 trajectory 做 normalized-time 插值，再计算：

```text
delta_success[c, s]
delta_failure[c, s]
```

运行：

```bash
/home/jinjaguo/anaconda3/envs/libero/bin/python \
  analysis/chunk_analysis/05_transition_analysis.py \
  --root OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate \
  --out_dir analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate \
  --n_success 20 \
  --n_failure 20 \
  --target_length 20
```

输出目录：

```text
analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate/transition
```

输出文件：

| 文件 | 含义 |
|---|---|
| `transition_delta_mean.csv` | success/failure 平均 transition delta |
| `transition_delta_std.csv` | success/failure transition delta 标准差 |
| `transition_delta_<state>.png` | 每个 state 的 success/failure transition 曲线 |
| `transition_delta_success.npy` | success 每条 trial 的 transition delta |
| `transition_delta_failure.npy` | failure 每条 trial 的 transition delta |
| `transition_metadata.json` | trial ids、chunk counts、normalized tau |

解读重点：

1. 成功轨迹可能有稳定的 phase transition peak。
2. 失败轨迹的 peak 可能提前、延后、消失，或者出现多个异常 peak。
3. 这说明 failure 不只是 hidden state 不一样，而是 temporal organization 被破坏。

---

### 5.7 第六阶段：phase structure discovery `06_phase_structure_discovery.py`

路径：

```text
analysis/chunk_analysis/06_phase_structure_discovery.py
```

目的：

把所有 normalized chunk 的 fused representation 放在一起做 PCA 或 UMAP，观察 latent phase structure。

每个点是：

```text
one trial × one normalized chunk
```

颜色有三种：

```text
chunk index
success/failure label
phase label: approach / grasp / move / place / unknown
```

phase label 来自 `rollouts_finalize.jsonl` 中保存的：

```text
chunk_ranges
phase_boundaries
```

运行 PCA：

```bash
/home/jinjaguo/anaconda3/envs/libero/bin/python \
  analysis/chunk_analysis/06_phase_structure_discovery.py \
  --root OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate \
  --out_dir analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate \
  --n_success 20 \
  --n_failure 20 \
  --target_length 20 \
  --method pca
```

运行 UMAP：

```bash
/home/jinjaguo/anaconda3/envs/libero/bin/python \
  analysis/chunk_analysis/06_phase_structure_discovery.py \
  --root OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate \
  --out_dir analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate \
  --method umap
```

注意：`--method umap` 需要环境里安装 `umap-learn`。

输出目录：

```text
analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate/phase_structure
```

输出文件：

| 文件 | 含义 |
|---|---|
| `embedding_points.csv` | 每个点的 2D 坐标、trial id、label、chunk、tau、phase |
| `phase_structure_by_chunk.png` | 按 normalized chunk index 上色 |
| `phase_structure_by_outcome.png` | 按 success/failure 上色 |
| `phase_structure_by_phase.png` | 按 phase 上色 |
| `phase_structure_metadata.json` | method、trial ids、chunk counts、normalized tau |

解读重点：

1. 如果 success 形成稳定流形而 failure 分散，说明 success 有稳定 latent dynamics。
2. 如果 failure 点只在某些 chunk 区域分离，说明 failure-specific region 是局部的。
3. 如果 phase label 和 embedding cluster 对齐，说明 phase parsing 有 representation basis。

---

### 5.8 第七阶段：time stretch analysis `07_time_stretch_analysis.py`

路径：

```text
analysis/chunk_analysis/07_time_stretch_analysis.py
```

目的：

分析 failure trajectory 变长这件事本身。定义：

```text
stretch_ratio = L_failure / L_success_reference
```

默认 `L_success_reference` 是所选 success trajectories 的 median chunk count，也可以通过 `--reference_success_length` 手动指定。

然后检查：

```text
stretch 越大，是否 early representation drift 越大？
stretch 越大，是否某些 state 的 divergence 越早？
```

运行：

```bash
/home/jinjaguo/anaconda3/envs/libero/bin/python \
  analysis/chunk_analysis/07_time_stretch_analysis.py \
  --root OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate \
  --out_dir analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate \
  --n_success 20 \
  --n_failure 20 \
  --target_length 20 \
  --early_fraction 0.25
```

输出目录：

```text
analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate/time_stretch
```

输出文件：

| 文件 | 含义 |
|---|---|
| `stretch_ratios.csv` | 每条 failure 的长度、reference success length、stretch ratio |
| `stretch_drift_correlation_pearson.csv/png` | stretch 与 drift 的 Pearson 相关热图 |
| `stretch_drift_correlation_spearman.csv/png` | stretch 与 drift 的 Spearman 相关热图 |
| `early_drift_vs_stretch.csv` | 每条 failure 的 early drift 汇总 |
| `early_drift_vs_stretch_<state>.png` | 每个 state 的 early drift vs stretch 散点图 |
| `time_stretch_metadata.json` | trial ids、chunk counts、reference length、early fraction |

解读重点：

如果看到：

```text
early drift ↑  →  stretch_ratio ↑
```

可以形成更强的解释链：

```text
early representation drift
    → execution repeatedly tries or stalls
    → trajectory becomes longer
    → final failure
```

---

## 6. 一键顺序跑完整分析

对原始 strict LIBERO 结果跑完整分析：

```bash
ROOT=OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate
OUT=analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate
PY=/home/jinjaguo/anaconda3/envs/libero/bin/python

$PY analysis/chunk_analysis/01_data_health_check.py --root $ROOT --out_dir $OUT --n_success 20 --n_failure 20
$PY analysis/chunk_analysis/02_divergence_heatmap.py --root $ROOT --out_dir $OUT --n_success 20 --n_failure 20 --target_length 20
$PY analysis/chunk_analysis/03_linear_probe_auc.py --root $ROOT --out_dir $OUT --n_success 20 --n_failure 20 --target_length 20 --cv 5
$PY analysis/chunk_analysis/04_failure_onset.py --out_dir $OUT --threshold 1.2 --consecutive 2
$PY analysis/chunk_analysis/05_transition_analysis.py --root $ROOT --out_dir $OUT --n_success 20 --n_failure 20 --target_length 20
$PY analysis/chunk_analysis/06_phase_structure_discovery.py --root $ROOT --out_dir $OUT --n_success 20 --n_failure 20 --target_length 20 --method pca
$PY analysis/chunk_analysis/07_time_stretch_analysis.py --root $ROOT --out_dir $OUT --n_success 20 --n_failure 20 --target_length 20
```

如果要分析 soft-success 复跑结果，只需要换 `ROOT` 和 `OUT`：

```bash
ROOT=OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate_soft_success
OUT=analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate_soft_success
```

---

## 7. LIBERO strict success 为什么过严

当前 BDDL goal：

```lisp
(And (On cream_cheese_1 plate_1))
```

LIBERO 的 `On` predicate 内部会调用 object state 的 `check_ontop`，它通常要求：

```text
cream_cheese z 高于 plate
cream_cheese 和 plate 有 MuJoCo contact
xy center distance < 0.03
```

对 plate 来说，`xy_distance < 0.03` 非常严格。实际视频中 cream cheese 已经放到 plate 上，但只要中心点稍微偏离，就可能被 strict `On` 判成失败。

因此本项目采用保守策略：

```text
保留原始 BDDL 和 LIBERO strict done
额外在 rollout runner 里加 soft success stopper
```

这样不会污染 benchmark，同时可以把视频上行为成功但 strict predicate 失败的轨迹单独标记出来。

---

## 8. Soft success stopper

脚本：

```text
analysis/chunk_analysis/10_run_soft_success_rollouts.py
```

核心逻辑：

```python
strict_done = env_done
soft_done = soft_success_stable(env)

if strict_done or soft_done:
    break
```

每个 trial 保存三个概念：

```text
strict_success       原始 LIBERO / BDDL 判定
soft_success         soft evaluator 判定，strict success 自动算 soft success
stop_reason          strict_success / soft_success / timeout
```

进一步可以分成：

```text
strict_success       LIBERO 原始成功
soft_success_only    LIBERO 判失败，但视频和 soft evaluator 认为成功
true_failure         strict 和 soft 都失败
```

当前 soft success 默认条件：

```text
xy_distance(cream_cheese, plate) < 0.06
0 < cream_cheese_z - plate_z < 0.12
cream_cheese 和 plate 有 MuJoCo contact
连续 stable_steps=5 帧满足
默认不要求 gripper_open，因为这个环境里 gripper qpos 可能对视觉成功样本不可靠
```

运行前必须先启动一个会保存 hidden-state chunks 的 policy server。普通 `start_serve.py` 不会保存 chunk hidden states。

### 8.1 启动 baseline hidden-state tracing server

使用已有的记录 server：

```bash
/home/jinjaguo/anaconda3/envs/libero/bin/python \
  start_server_record.py \
  --env pi05_libero \
  --port 8000 \
  --trace_root OOD_exp/outputs/chunk_wise
```

### 8.2 运行 soft-success rollout client

```bash
/home/jinjaguo/anaconda3/envs/libero/bin/python \
  analysis/chunk_analysis/10_run_soft_success_rollouts.py \
  --bddl_path custom_bddl/libero_goal/dif_start_end_loc/put_the_cream_cheese_on_the_plate.bddl \
  --task_name put_the_cream_cheese_on_the_plate_soft_success \
  --host localhost \
  --port 8000 \
  --trials 20 \
  --xy_threshold 0.06 \
  --height_gap_max 0.12 \
  --stable_steps 5
```

输出 hidden-state chunks：

```text
OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate_soft_success/trial_<id>/chunk_*.pt
```

输出 trial metadata：

```text
OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate_soft_success/trial_<id>/rollouts_finalize.jsonl
OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate_soft_success/trial_<id>/rollouts_soft_success.jsonl
OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate_soft_success/trial_<id>/rollouts_state_record.jsonl
```

注意：`rollouts_finalize.jsonl` 里的顶层 `success` 是 soft-success label，方便后续 chunk analysis 直接把它当成功标签。原始 strict label 被保存到：

```text
final_info.strict_success
final_info.libero_strict_success
rollouts_soft_success.jsonl 中的 strict_success
```

输出视频和 summary：

```text
OOD_exp/outputs/videos/soft_success/put_the_cream_cheese_on_the_plate_soft_success/trial<id>_<suffix>.mp4
OOD_exp/outputs/videos/soft_success/put_the_cream_cheese_on_the_plate_soft_success/soft_success_summary.csv
```

其中 suffix 是：

```text
strict_success
soft_success
failure
```

### 8.3 为什么 soft-success runner 会检查 chunk 文件

`10_run_soft_success_rollouts.py` 本身是 rollout client，它不直接写 hidden state tensor。hidden state tensor 必须由 websocket policy server 写。

因此脚本会在每个 trial 结束后检查：

```text
期望 num_chunks 个 chunk_*.pt
实际 trial 目录中也必须有 num_chunks 个 chunk_*.pt
```

如果报错：

```text
RuntimeError: Expected 22 hidden-state chunk files ..., found 0.
```

说明：

```text
你连接的 server 不是 tracing server
或者 server 保存的 output_task_name 和 client 的 --task_name 不一致
```

解决方法：

1. baseline soft-success：使用 `start_server_record.py`。
2. causal-intervention soft-success：使用 `08_start_causal_intervention_server.py`，并且 server 的 `--output_task_name` 必须等于 client 的 `--task_name`。

---

## 9. Causal intervention 实验

causal intervention 的核心问题是：

```text
如果 failure rollout 在早期 chunk 的 hidden state 被替换成 success hidden state，
它是否能从 failure 变成 success？
```

典型实验：

```text
h_fail[c=2] → replace with h_success[c=2]
然后继续 rollout
```

如果替换后成功率上升，或者 trajectory 被拉回 success manifold，可以支持：

```text
early latent state causally determines success or failure
```

这个实验需要两个进程：

```text
server: 08_start_causal_intervention_server.py
client: 09_run_causal_intervention_rollouts.py 或 10_run_soft_success_rollouts.py
```

### 9.1 启动 causal intervention server

脚本：

```text
analysis/chunk_analysis/08_start_causal_intervention_server.py
```

作用：

1. 启动 OpenPI websocket policy server。
2. 加载原始 success traces 作为 reference。
3. 在指定 chunk 和 state 上替换 hidden state。
4. 把 intervention rollout 的 hidden-state chunks 写入新的 task 目录。

支持替换的 state：

```text
cross_attention_action_x_after
action_head_input
chunk_vector_mean
action_chunk_output
```

推荐先用：

```text
intervention_chunk = 2
state = action_head_input
reference_mode = mean_success
```

启动示例：

```bash
/home/jinjaguo/anaconda3/envs/libero/bin/python \
  analysis/chunk_analysis/08_start_causal_intervention_server.py \
  --env pi05_libero \
  --port 8001 \
  --trace_root OOD_exp/outputs/chunk_wise \
  --task_name put_the_cream_cheese_on_the_plate \
  --output_task_name put_the_cream_cheese_on_the_plate__causal_intervention \
  --intervention_chunk 2 \
  --state action_head_input \
  --reference_mode mean_success
```

重要参数：

| 参数 | 含义 |
|---|---|
| `--trace_root` | 读取 baseline traces、保存 intervention traces 的根目录 |
| `--task_name` | baseline task 名，用于读取 success reference |
| `--output_task_name` | intervention 结果保存到哪个 task 目录 |
| `--intervention_chunk` | 替换哪一个 chunk |
| `--state` | 替换哪一个 hidden state |
| `--reference_mode mean_success` | 用多个 success trial 在该 chunk/state 的均值作为 reference |
| `--reference_mode trial --reference_trial <id>` | 使用指定 success trial 的一个 state 作为 reference |

server 输出：

```text
OOD_exp/outputs/chunk_wise/<output_task_name>/intervention_manifest.json
OOD_exp/outputs/chunk_wise/<output_task_name>/trial_<id>/chunk_*.pt
```

每个被保存的 `chunk_*.pt` 里会包含 intervention metadata，例如：

```text
intervention_applied
intervention_state
reference_source
```

### 9.2 使用 strict LIBERO done 的 causal rollout client

脚本：

```text
analysis/chunk_analysis/09_run_causal_intervention_rollouts.py
```

作用：

1. 从 baseline chunk root 里读取失败 trial ids。
2. 用相同 seed 复跑这些 failure rollout。
3. 连接 causal intervention server。
4. 保存 rollout metadata 和视频。
5. hidden-state chunks 由 server 写入同一个 output task 目录。

运行：

```bash
/home/jinjaguo/anaconda3/envs/libero/bin/python \
  analysis/chunk_analysis/09_run_causal_intervention_rollouts.py \
  --bddl_path custom_bddl/libero_goal/dif_start_end_loc/put_the_cream_cheese_on_the_plate.bddl \
  --source_chunk_root OOD_exp/outputs/chunk_wise \
  --source_task_name put_the_cream_cheese_on_the_plate \
  --output_task_name put_the_cream_cheese_on_the_plate__causal_intervention \
  --host localhost \
  --port 8001 \
  --num_trials 20
```

输出 metadata：

```text
OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate__causal_intervention/trial_<id>/rollouts_finalize.jsonl
```

输出视频：

```text
OOD_exp/outputs/videos/causal_intervention/put_the_cream_cheese_on_the_plate__causal_intervention/source_trial<id>_intervention_<success_or_failure>.mp4
OOD_exp/outputs/videos/causal_intervention/put_the_cream_cheese_on_the_plate__causal_intervention/intervention_summary.csv
```

### 9.3 使用 soft-success stopper 的 causal rollout

由于 LIBERO strict success 对 plate task 过严，当前更推荐把 soft-success 结果作为 causal intervention 的主要分析标签。

启动 server 时，把 `--output_task_name` 设成 soft-success task 名：

```bash
/home/jinjaguo/anaconda3/envs/libero/bin/python \
  analysis/chunk_analysis/08_start_causal_intervention_server.py \
  --env pi05_libero \
  --port 8001 \
  --trace_root OOD_exp/outputs/chunk_wise \
  --task_name put_the_cream_cheese_on_the_plate \
  --output_task_name put_the_cream_cheese_on_the_plate_soft_success \
  --intervention_chunk 2 \
  --state action_head_input \
  --reference_mode mean_success
```

然后用 soft-success runner 连接这个 causal server：

```bash
/home/jinjaguo/anaconda3/envs/libero/bin/python \
  analysis/chunk_analysis/10_run_soft_success_rollouts.py \
  --bddl_path custom_bddl/libero_goal/dif_start_end_loc/put_the_cream_cheese_on_the_plate.bddl \
  --task_name put_the_cream_cheese_on_the_plate_soft_success \
  --host localhost \
  --port 8001 \
  --trials 20 \
  --xy_threshold 0.06 \
  --height_gap_max 0.12 \
  --stable_steps 5
```

关键要求：

```text
server --output_task_name == client --task_name
```

否则 client 会在自己的 task 目录里找不到 server 写的 `chunk_*.pt`。

---

## 10. 对 causal / soft-success 结果继续做 chunk analysis

soft-success 或 causal-intervention rollout 完成后，它们和 baseline 一样，都可以作为新的 `--root` 进入 `01-07` 分析脚本。

例如分析 soft-success causal intervention 结果：

```bash
ROOT=OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate_soft_success
OUT=analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate_soft_success
PY=/home/jinjaguo/anaconda3/envs/libero/bin/python

$PY analysis/chunk_analysis/01_data_health_check.py --root $ROOT --out_dir $OUT --n_success 20 --n_failure 20
$PY analysis/chunk_analysis/02_divergence_heatmap.py --root $ROOT --out_dir $OUT --n_success 20 --n_failure 20 --target_length 20
$PY analysis/chunk_analysis/03_linear_probe_auc.py --root $ROOT --out_dir $OUT --n_success 20 --n_failure 20 --target_length 20 --cv 5
$PY analysis/chunk_analysis/04_failure_onset.py --out_dir $OUT --threshold 1.2 --consecutive 2
$PY analysis/chunk_analysis/05_transition_analysis.py --root $ROOT --out_dir $OUT --n_success 20 --n_failure 20 --target_length 20
$PY analysis/chunk_analysis/06_phase_structure_discovery.py --root $ROOT --out_dir $OUT --n_success 20 --n_failure 20 --target_length 20 --method pca
$PY analysis/chunk_analysis/07_time_stretch_analysis.py --root $ROOT --out_dir $OUT --n_success 20 --n_failure 20 --target_length 20
```

注意：

如果 soft-success 之后失败样本少于 20，`--n_failure 20` 会报错。此时要根据实际数量调小：

```bash
--n_failure <实际 true_failure 数量>
```

同理，如果 success 少于 20，也要调小 `--n_success`。

---

## 11. 当前已有结果说明

当前已有 baseline 结果目录：

```text
analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate
```

里面包括：

```text
divergence/
linear_probe_auc/
failure_onset/
transition/
phase_structure/
time_stretch/
```

重点文件：

```text
divergence/cosine_divergence_statewise.png
linear_probe_auc/auc_statewise.png
failure_onset/onset_summary.csv
transition/transition_delta_action_head_input.png
phase_structure/phase_structure_by_outcome.png
time_stretch/early_drift_vs_stretch_action_head_input.png
```

已有 causal intervention 视频 summary：

```text
OOD_exp/outputs/videos/causal_intervention/put_the_cream_cheese_on_the_plate__causal_intervention/intervention_summary.csv
```

已有 soft-success summary 位置：

```text
OOD_exp/outputs/videos/soft_success/put_the_cream_cheese_on_the_plate_soft_success/soft_success_summary.csv
```

如果重新跑 soft-success causal intervention，建议使用新的 task name 或确保旧 trial 目录会被清理。`10_run_soft_success_rollouts.py` 默认会删除对应 trial 目录再跑，防止旧 chunk 文件污染。

---

## 12. 推荐实验报告结构

最终写实验结论时，可以按下面逻辑组织。

### 12.1 数据体检

报告：

```text
success/failure trial 数量
每个 trial chunk count
hidden state shape
NaN/Inf/zero tensor 检查
norm distribution
```

关键判断：

```text
是否必须 normalized-time
是否 cosine distance 比 L2 distance 更合适
是否存在 activation magnitude instability
```

### 12.2 Failure divergence

报告：

```text
D[c,s] heatmap
哪个 state 最早变亮
哪个 normalized tau 开始分叉
```

### 12.3 Linear probe

报告：

```text
AUC[c,s] heatmap
哪些 chunk/state 能稳定预测 success/failure
```

核心结论来自：

```text
high cosine divergence ∩ high AUC
```

### 12.4 Failure onset

报告：

```text
onset_summary.csv 中每个 state 的 onset chunk / onset tau
```

解释：

```text
早期 onset → instruction grounding / semantic drift
中段 onset → phase transition failure
末端 onset → final placement / release failure
```

### 12.5 Temporal organization

报告：

```text
transition delta curves
success/failure phase transition peak 是否对齐
failure 是否多峰、提前、延后或消失
```

### 12.6 Latent phase structure

报告：

```text
PCA/UMAP by chunk
PCA/UMAP by outcome
PCA/UMAP by phase
```

判断 success 是否形成稳定 latent dynamics。

### 12.7 Time stretch

报告：

```text
stretch_ratio
early drift vs stretch
stretch/drift correlation heatmap
```

构造链条：

```text
early drift → prolonged execution → failure
```

### 12.8 Causal intervention

报告：

```text
intervention chunk
intervention state
reference source
strict_success
soft_success
soft_success_only
true_failure
```

核心比较：

```text
baseline failure seeds
vs
same seeds with h_fail[c=2] replaced by h_success[c=2]
```

如果 intervention 后 success 或 soft_success 明显增加，说明该早期 latent state 不只是相关，而可能具有因果作用。

---

## 13. 常见问题与排查

### 13.1 `Expected X hidden-state chunk files ..., found 0`

原因：

```text
client 只跑环境，不写 hidden-state chunk
hidden-state chunk 必须由 tracing policy server 写
```

解决：

```text
baseline soft-success 用 start_server_record.py
causal soft-success 用 08_start_causal_intervention_server.py
确保 server --output_task_name 和 client --task_name 完全一致
```

### 13.2 `Need 20 successes, found N`

原因：

```text
当前 root 下 success trial 少于 --n_success
或者 soft-success / causal-intervention 后 true_failure 数量变少
```

解决：

```bash
--n_success N
--n_failure M
```

根据实际数量调小。

### 13.3 strict failure 但视频看起来成功

原因：

LIBERO `On` predicate 对 plate 的中心距离阈值过紧，可能需要 `xy_distance < 0.03` 且 contact 成立。

解决：

使用 `10_run_soft_success_rollouts.py`：

```text
保留 strict_success
额外记录 soft_success
后续分析使用 soft_success label
```

### 13.4 gripper 明明释放了但 soft success 失败

当前脚本默认不要求 gripper released：

```text
require_gripper_released = False
```

如果手动打开：

```bash
--require_gripper_released
```

可能因为环境里的 `robot0_gripper_qpos` 接近 0 而错判。plate task 当前不推荐开启。

### 13.5 UMAP 报 ImportError

安装或改用 PCA：

```bash
--method pca
```

PCA 不依赖 `umap-learn`。

### 13.6 causal server 使用了 JAX checkpoint

hidden-state intervention 需要 PyTorch OpenPI policy。脚本会优先查找：

```text
~/.cache/openpi/pytorch_checkpoints/pi05_libero/model.safetensors
```

如果不存在，可能退回默认 checkpoint，而默认可能是 JAX，无法做 hidden-state hook。

解决：

确保本地 PyTorch checkpoint 存在，或显式传入：

```text
policy:checkpoint --policy.config ... --policy.dir ...
```

---

## 14. 路径速查表

| 内容 | 路径 |
|---|---|
| 项目根目录 | `/home/jinjaguo/BH_MOE` |
| Python 环境 | `/home/jinjaguo/anaconda3/envs/libero/bin/python` |
| chunk analysis 脚本 | `/home/jinjaguo/BH_MOE/analysis/chunk_analysis` |
| baseline hidden chunks | `/home/jinjaguo/BH_MOE/OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate` |
| soft-success hidden chunks | `/home/jinjaguo/BH_MOE/OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate_soft_success` |
| baseline analysis results | `/home/jinjaguo/BH_MOE/analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate` |
| soft-success analysis results | `/home/jinjaguo/BH_MOE/analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate_soft_success` |
| causal videos | `/home/jinjaguo/BH_MOE/OOD_exp/outputs/videos/causal_intervention` |
| soft-success videos | `/home/jinjaguo/BH_MOE/OOD_exp/outputs/videos/soft_success` |
| BDDL 文件 | `/home/jinjaguo/BH_MOE/custom_bddl/libero_goal/dif_start_end_loc/put_the_cream_cheese_on_the_plate.bddl` |

---

## 15. 最推荐的下一步实验

当前最合理的主线是：

1. 使用 `08_start_causal_intervention_server.py` 在 `chunk=2, state=action_head_input` 做 intervention。
2. 使用 `10_run_soft_success_rollouts.py` 连接 causal server，按 soft success stopper 复跑。
3. 确保输出目录是：

```text
OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate_soft_success
```

4. 用 `01-07` 对 soft-success causal 结果重新跑分析。
5. 对比 baseline strict、baseline soft、causal soft 三组：

```text
success rate / soft_success rate
failure onset tau
transition peak structure
phase PCA/UMAP separation
time stretch ratio
```

最关键的结论应该来自：

```text
早期 chunk/state 的 high divergence + high AUC
        ↓
该位置 causal replacement 后 soft_success 上升
        ↓
说明 early latent drift 不只是相关，而是可能造成 failure 的原因
```

