# Template Centroid 分析说明

这个目录用于把 chunk-wise latent 结果组织成一套可复现、可画图、可报告的 template dynamics 分析。核心目标是回答：

1. early latent 是否已经偏向某个任务模板。
2. failure 是否在早期激活了错误模板。
3. early latent replacement 是否能把轨迹重新拉回 success template。
4. recovered 和 unrecovered intervention 能否与 natural success / natural failure 放进同一套指标体系比较。

当前默认任务是：

```bash
put_the_cream_cheese_on_the_plate
```

默认分析的 latent site 是：

```bash
action_head_input
chunk_vector_mean
```

其中 `chunk_vector_mean` 基本是 `action_head_input` action-token 表示的 mean pooling，所以两者的趋势通常会非常接近。

## 整体思路

我们把每个 early chunk 的 latent 向量投影到两个 template centroid 上：

- `success centroid`：来自 natural success rollout 的 early chunks。
- `bowl compositional centroid`：来自 natural failure 中 `wrong_receptacle` failure mode 的 early chunks。

template margin 定义为：

```text
margin = cos(z, success_centroid) - cos(z, bowl_compositional_centroid)
```

解释方式：

- `margin > 0`：latent 更接近 success template。
- `margin < 0`：latent 更接近 bowl compositional template。
- margin 越大，success template alignment 越强。
- margin 越小或越负，bowl/wrong-template activation 越强。

默认使用 `chunk_aligned` centroid，也就是 chunk 0 用 chunk 0 的 success/bowl centroid 比，chunk 1 用 chunk 1 的 centroid 比。这样可以减少 early trajectory 自身时间漂移造成的误导。

## 四组结果如何纳入同一套分析

当前脚本把所有样本统一放进同一套 success/bowl template 坐标系里比较：

| 组名 | 含义 | 来源 |
|---|---|---|
| `natural_success` | 原始 rollout 成功 | chunk manifest 中 `success=1` |
| `natural_failure` | 原始 rollout 失败，且 failure mode 是 `wrong_receptacle` | chunk manifest 中 `success=0` 且 `dominant_failure_mode=wrong_receptacle` |
| `intervention_recovered_success` | intervention 后成功 | soft success summary 中 `analysis_success=True` |
| `intervention_unrecovered_failure` | intervention 后仍失败 | soft success summary 中 `analysis_success=False` |

注意：我们当前延续的设定是 intervention success 使用 `analysis_success=True`，所以包括 strict success 和 soft success，当前是 13 条成功 intervention。不是只用 `soft_success_only=True` 的 4 条。

intervention 的 latent 来自：

```text
OOD_exp/dif_start_end_loc/outputs/chunk_wise/put_the_cream_cheese_on_the_plate_soft_success
```

intervention summary 来自：

```text
OOD_exp/dif_start_end_loc/outputs/videos/soft_success/put_the_cream_cheese_on_the_plate_soft_success/soft_success_summary.csv
```

intervention manifest 在：

```text
OOD_exp/dif_start_end_loc/outputs/chunk_wise/put_the_cream_cheese_on_the_plate_soft_success/intervention_manifest.json
```

当前默认 replacement chunk 是：

```text
chunk 2
```

对应 manifest 里：

```json
"intervention_chunk": 2,
"intervention_state": "action_head_input"
```

因此 paper metrics 里会特别报告：

- pre-patch window：chunk 0 到 chunk 1。
- post-patch window：chunk 2 到 chunk 4。

如果 post-patch 的 success-template ratio 或 margin 明显高于 pre-patch，就说明 replacement 的 causal timing 证据更强。

## 输入路径

核心输入包括：

```text
OOD_exp/dif_start_end_loc/manifests/pi05_put_the_cream_cheese_on_the_plate_rollout_manifest.csv
OOD_exp/dif_start_end_loc/manifests/pi05_put_the_cream_cheese_on_the_plate_chunk_manifest.csv
OOD_exp/dif_start_end_loc/annotations/put_the_cream_cheese_on_the_plate_label_review.csv
OOD_exp/dif_start_end_loc/outputs/chunk_wise/put_the_cream_cheese_on_the_plate
OOD_exp/dif_start_end_loc/outputs/chunk_wise/put_the_cream_cheese_on_the_plate_soft_success
OOD_exp/dif_start_end_loc/outputs/videos/soft_success/put_the_cream_cheese_on_the_plate_soft_success/soft_success_summary.csv
```

`chunk_manifest.csv` 是最重要的入口。每一行对应一个 latent sample，里面有：

- `latent_pt_path`：chunk `.pt` 文件路径。
- `success`：natural rollout 是否成功。
- `dominant_failure_mode`：failure mode。
- `valid_for_probe`：是否纳入 probe/analysis。
- `is_early_chunk`：是否 early chunk。

每个 `.pt` 里保存多个 latent key。当前主要读取：

```text
action_head_input
chunk_vector_mean
```

## 输出路径

默认输出目录是：

```text
analysis/template_centroid/results/put_the_cream_cheese_on_the_plate
```

主要输出分为四类：

1. 图：用于直观看 template activation 和 dynamics。
2. chunk/rollout 级 CSV：用于进一步统计或 debug。
3. paper metrics：可直接写入论文表格。
4. task summary：更高层的 task-level 汇总。

## 运行方式

### 方式一：一键任务级流程

推荐用这个脚本，它会自动定位 manifest、intervention summary，并依次运行 template centroid 分析和 paper metrics。

```bash
python analysis/template_centroid/03_task_level_template_summary.py \
    --task-name put_the_cream_cheese_on_the_plate
```

常用显式参数：

```bash
python analysis/template_centroid/03_task_level_template_summary.py \
    --task-name put_the_cream_cheese_on_the_plate \
    --fields action_head_input chunk_vector_mean \
    --failure-mode wrong_receptacle \
    --intervention-chunk 2 \
    --output-dir analysis/template_centroid/results/put_the_cream_cheese_on_the_plate
```

### 方式二：分步运行

第一步，生成 template centroid 图和基础 CSV：

```bash
python analysis/template_centroid/01_template_centroid_analysis.py \
    --chunk-manifest OOD_exp/dif_start_end_loc/manifests/pi05_put_the_cream_cheese_on_the_plate_chunk_manifest.csv \
    --output-dir analysis/template_centroid/results/put_the_cream_cheese_on_the_plate \
    --fields action_head_input chunk_vector_mean \
    --failure-mode wrong_receptacle \
    --max-early-chunk 4 \
    --full-max-chunk 30 \
    --normalized-bins 20
```

第二步，把视觉结果转成论文指标：

```bash
python analysis/template_centroid/02_paper_report_metrics.py \
    --results-dir analysis/template_centroid/results/put_the_cream_cheese_on_the_plate \
    --intervention-chunk 2 \
    --late-fraction 0.2
```

## 每个脚本回答什么问题

### `01_template_centroid_analysis.py`

回答的问题：

- early chunks 是否已经能区分 success template 和 bowl compositional template。
- intervention success 是否在 replacement 后更靠近 success template。
- failure dynamics 是稳定负 margin，还是在两个模板之间频繁切换。
- recovered 和 unrecovered intervention 在同一 template 坐标系下表现如何。

主要输出：

```text
template_margin_curve.png
nearest_template_distribution.png
full_rollout_margin_curve.png
normalized_full_rollout_margin_curve.png
rollout_margin_heatmap.png
pca_scatter.png
template_margin_by_chunk.csv
nearest_template_distribution.csv
full_rollout_sample_margins.csv
full_rollout_margin_by_chunk.csv
normalized_full_rollout_margin_by_bin.csv
rollout_dynamics_summary.csv
summary.json
```

### `02_paper_report_metrics.py`

回答的问题：

- 哪些数值可以直接报告到论文里。
- early margin、post-patch margin、nearest success ratio、switch count、late alignment 是否支持 causal/template dynamics claim。

主要输出：

```text
paper_metrics_rollout_level.csv
paper_metrics_summary.csv
paper_metrics_summary.json
```

### `03_task_level_template_summary.py`

回答的问题：

- 一个 task 的核心 template dynamics 是否能被压缩成一张 summary 表。
- 多个图和多个 CSV 如何整合成统一指标。
- 如果未来换任务，能否用同一命令跑出同样结构的结果。

主要输出：

```text
task_summary.csv
task_summary_metadata.json
normalized_cosine_drift_curve.png
template_switch_rate_boxplot.png
template_entropy_boxplot.png
margin_variance_boxplot.png
```

## 图片如何阅读

### `template_margin_curve.png`

这是 early template margin curve。

横轴是 early chunk id，默认 chunk 0 到 chunk 4。纵轴是：

```text
cos(z, success_centroid) - cos(z, bowl_centroid)
```

读法：

- natural success 应该整体在 0 上方。
- natural failure 应该整体在 0 下方。
- intervention recovered success 如果在 chunk 2 后上升，说明 replacement 后 latent 被拉向 success template。
- intervention unrecovered failure 如果仍然低或不稳定，说明 replacement 没有产生足够稳定的成功模板激活。

### `nearest_template_distribution.png`

标题是：

```text
Template Assigned by Early Latent Nearest Centroid
```

x 轴是 rollout group。每组有两个柱子：

- nearest 到 success template。
- nearest 到 bowl compositional template。

柱子上的数字是比例。

读法：

- natural success 的 success-template ratio 越高越好。
- natural failure 的 bowl-template ratio 越高，越说明 early wrong-template misactivation。
- intervention recovered success 的 success-template ratio 如果接近 natural success，说明 replacement 后 template assignment 被纠正。
- intervention unrecovered failure 可以帮助判断失败是否来自 template 没有恢复，或者恢复了但后续执行仍失败。

### `full_rollout_margin_curve.png`

这是按真实 chunk id 的 full rollout margin curve。默认只画到前 30 个 chunk，避免不同组长度差异太大导致后半段只剩少数组。

读法：

- natural success 如果后期趋正，说明最终成功时 template alignment 回到 success side。
- natural failure 如果频繁跨 0 或长期不稳定，说明不是简单的 early error，而是 template dynamics 不稳定。
- intervention recovered success 如果在 chunk 2 后抬升，说明 replacement 有后续动态影响。

### `normalized_full_rollout_margin_curve.png`

这是 normalized time 版本。每条 rollout 被插值到固定 20 个 bins。

读法：

- 更适合比较不同长度 rollout 的完整趋势。
- 如果 intervention recovered success 在后 20% 明显高于 natural failure，就支持最终回到 success template。

### `rollout_margin_heatmap.png`

每个子图的横轴是 chunk，纵轴是 rollout，颜色是 margin。

读法：

- 红/正值表示更靠近 success template。
- 蓝/负值表示更靠近 bowl compositional template。
- 一条 rollout 上颜色频繁红蓝切换，说明 template switching 或 unstable competition。
- natural failure 如果有大量切换，比“稳定负 margin”更适合描述为 unstable competing template dynamics。

### `pca_scatter.png`

PCA 只是辅助可视化，不作为主证据。

读法：

- 只能辅助说明 early latent 是否有粗略分离。
- 论文主证据应优先使用 margin、nearest ratio、switch count、late alignment。

### 任务级补充图

`03_task_level_template_summary.py` 会额外输出：

```text
normalized_cosine_drift_curve.png
template_switch_rate_boxplot.png
template_entropy_boxplot.png
margin_variance_boxplot.png
```

读法：

- cosine drift 越大，表示相邻 chunk latent 变化越大。
- switch rate 越高，表示 success/bowl template assignment 切换越频繁。
- entropy 越高，表示 template assignment 更混乱。
- margin variance 越高，表示 template alignment 更不稳定。

这些图用于支持 dynamics / instability claim，但不要替代 early margin 和 nearest ratio 这两个主证据。

## CSV 指标如何判断

### `paper_metrics_summary.csv`

这是最适合直接写进论文表格的文件。每行是：

```text
latent_key, group, metric, mean, ci_low, ci_high, n_rollouts
```

常用指标：

#### `early_mean_margin_chunk_0_4`

每条 rollout 先对 chunk 0 到 chunk 4 求平均 margin，然后按组统计均值和 95% CI。

判断：

- natural success 应为正。
- natural failure 应为负。
- intervention recovered success 如果为正，说明 intervention 成功组 early latent 更偏 success template。

#### `post_intervention_mean_margin_chunk_2_4`

因为 intervention 默认发生在 chunk 2，所以这个指标只看 chunk 2 到 chunk 4。

判断：

- 如果 intervention recovered success 在 post window 明显高于 pre window，就说明 replacement 后发生了模板纠正。
- 这个指标比 chunk 0 到 4 更公平，因为 chunk 0 到 1 还没被 intervention 修正。

#### `nearest_success_ratio_chunk_0_4`

early chunks 中 nearest 到 success template 的比例。

判断：

- natural success 高，natural failure 低，是 early separability 证据。
- intervention recovered success 越接近 natural success，说明 replacement 后 template assignment 更像成功轨迹。

#### `nearest_success_ratio_pre_patch_chunk_0_1`

patch 前窗口。

判断：

- 对 intervention recovered success，这个值可以作为 intervention 前状态。

#### `nearest_success_ratio_post_patch_chunk_2_4`

patch 后窗口。

判断：

- 如果 intervention recovered success 的 post ratio 显著高于 pre ratio，这是很强的 causal timing evidence。

#### `template_switch_count`

把每个 chunk 的 margin 正负转成 nearest success/bowl template，然后统计整条 trajectory 的切换次数。

判断：

- natural failure 如果明显高于 natural success，说明 failure dynamics 更不稳定。
- intervention recovered success 如果接近 natural success、低于 natural failure，说明 replacement 降低了 template switching。

#### `late_success_alignment_final_20pct`

每条 rollout 最后 20% chunk 的平均 margin。

判断：

- natural success 和 intervention recovered success 如果为正，说明最终成功时 latent 回到 success template。
- natural failure 如果接近 0 或负，说明最终没有稳定回到 success template。

### `task_summary.csv`

这是更高层的 summary，适合跨任务比较。它把 paper metrics 里最常用的指标重命名成更短的列，例如：

- `early_margin_gap_mean`
- `nearest_success_template_ratio_mean`
- `post_patch_redirection_ratio_mean`
- `final_margin_mean`
- `late_final_20pct_margin_mean`
- `switch_count_mean`
- `switch_rate_mean`
- `template_entropy_mean`
- `mean_cosine_drift_mean`

判断方式与 `paper_metrics_summary.csv` 相同，只是列更宽，更适合直接做 task-level 表格。

### `full_rollout_sample_margins.csv`

每行是一个 chunk sample 的 margin。

适合：

- debug 某条 rollout。
- 画自定义 heatmap。
- 检查某个 chunk 是否异常。

### `rollout_dynamics_summary.csv`

每行是一条 rollout 的 dynamics summary。

适合：

- 看单条 rollout 的 switch count。
- 筛选 persistent negative / persistent positive。
- 找失败案例或 intervention 未恢复案例。

## 当前任务的关键结论应该怎么表述

以 `action_head_input` 为主，当前结果支持下面这条链：

1. natural success 和 natural failure 在 early template margin 上可分。
2. natural failure 的 early chunks 更常被分配到 bowl compositional template。
3. intervention success 在 chunk 2 后 success-template ratio 明显提升。
4. natural failure 的 template switch count 明显高于 natural success。
5. intervention recovered success 的 late alignment 为正，说明最终成功轨迹回到 success template side。

更稳妥的表述是：

```text
Early task-template misactivation is measurable before action decoding, and
successful intervention redirects the latent trajectory toward the success
template after the replacement chunk.
```

如果只看 failure dynamics，不要强行说 persistent negative。当前数据更适合说：

```text
Failure trajectories exhibit elevated template switching / unstable competition
between success and bowl-compositional templates.
```

## 添加新任务时需要做什么

如果新任务已经有同样结构的 manifest、label csv、chunk-wise latent 和 intervention summary，只需要运行：

```bash
python analysis/template_centroid/03_task_level_template_summary.py \
    --task-name <new_task_name>
```

要求：

- `OOD_exp/dif_start_end_loc/manifests/` 下有对应 rollout/chunk manifest。
- `OOD_exp/dif_start_end_loc/annotations/` 下最好有 `<task_name>_label_review.csv`。
- natural chunk latent 在 `OOD_exp/dif_start_end_loc/outputs/chunk_wise/<task_name>`。
- intervention latent 在 `OOD_exp/dif_start_end_loc/outputs/chunk_wise/<task_name>_soft_success`。
- intervention summary 在 `OOD_exp/dif_start_end_loc/outputs/videos/soft_success/<task_name>_soft_success/soft_success_summary.csv`。

如果 intervention chunk 不是 2，需要显式指定：

```bash
python analysis/template_centroid/03_task_level_template_summary.py \
    --task-name <new_task_name> \
    --intervention-chunk <chunk_id>
```

