# LIBERO / pi0.5 Attention Trace 使用说明

这个目录里的代码接入方式和仓库根目录 `README.md` 一样：仍然使用两个终端。

区别是 Terminal A 不再启动普通 `start_server_record.py`，而是启动 attention tracing server：

```bash
cd ~/openpi
source .venv/bin/activate
python /home/jinjaguo/BH_MOE/attention_analysis/scripts/start_attention_server.py \
  --attention-mode baseline \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero
```

Terminal B 使用 attention_analysis 里的独立 rollout 脚本。这个脚本从
`ood_libero_rollouts.py` 移植而来，但默认输出位置已经改到
`attention_analysis/outputs/libero_rollouts`，不会覆盖之前 `OOD_exp/...`
下的实验结果：

```bash
conda activate libero
cd /home/jinjaguo/BH_MOE
python attention_analysis/attention_rollouts.py \
  --input_dir /home/jinjaguo/BH_MOE/custom_bddl/libero_goal \
  --tasks_info /home/jinjaguo/BH_MOE/custom_bddl/libero_goal/dif_start_end_loc/tasks_info.txt \
  --libero_root /home/jinjaguo/LIBERO \
  --host localhost \
  --port 8000 \
  --max_trials 1
```

如果你只想跑某一个任务子目录，也可以把 `--input_dir` 指到子目录，此时可以省略 `--tasks_info`，脚本会自动使用该目录下的 `tasks_info.txt`：

```bash
python attention_analysis/attention_rollouts.py \
  --input_dir /home/jinjaguo/BH_MOE/custom_bddl/libero_goal/dif_start_end_loc \
  --libero_root /home/jinjaguo/LIBERO \
  --host localhost \
  --port 8000 \
  --max_trials 1
```

## `attention_rollouts.py` 参数说明

这个脚本是给 attention 诊断用的，不是用来像 `ood_libero_rollouts.py` 那样收集大量成功/失败 trial。它不再提供 `target_successes/target_failures` 停止条件，只按 `--max_trials` 控制每个任务运行几次。

```text
--max_trials 1
```

常用参数：

```text
--input_dir
  BDDL 根目录。可以是 custom_bddl/libero_goal，也可以直接指向某个子目录，
  例如 custom_bddl/libero_goal/dif_start_end_loc。

--tasks_info
  可选任务列表。如果省略，脚本会优先使用 --input_dir/tasks_info.txt；
  如果该文件不存在，就递归扫描 --input_dir 下的 .bddl 文件。
  tasks_info 里支持 libero/bddl_files/<suite>/<task>.bddl 这种 LIBERO 原始格式，
  会自动映射到 --input_dir/<suite>/<task>.bddl。

--max_trials
  每个任务最多跑几个 trial。attention 分析建议先用 1。

--seed
  LIBERO 环境初始 seed。不同 trial 会使用 seed + trial_id。

--output_root
  只保存 rollout video 的目录，默认是
  attention_analysis/outputs/libero_rollouts/videos。

--skip_existing / --no-skip-existing
  默认会跳过已有视频的任务。如果你想复跑同一任务并重新请求 attention server，
  使用 --no-skip-existing，或者换一个 --output_root。
```

最小 attention 诊断建议：

```bash
python attention_analysis/attention_rollouts.py \
  --input_dir /home/jinjaguo/BH_MOE/custom_bddl/libero_goal \
  --tasks_info /home/jinjaguo/BH_MOE/custom_bddl/libero_goal/libero_goal/tasks_info.txt \
  --libero_root /home/jinjaguo/LIBERO \
  --host localhost \
  --port 8000 \
  --max_trials 1 \
  --no-skip-existing
```

## 输出位置

默认 attention 输出会保存到：

```text
/home/jinjaguo/BH_MOE/attention_analysis/outputs/attention_trace/
  <task_name>/
    trial_<trial_id>/
      chunk_<chunk_id>/
        baseline/
          token_map.json
          config.json
          attention_summary.parquet
          attention_topk.jsonl
          hidden_state_norms.parquet
          sink_tokens.json
          action_outputs.npz
```

`task_name/trial_id/chunk_id` 来自 `attention_rollouts.py` 的 websocket request，所以 attention server 仍然可以按 chunk 保存 attention trace。

`attention_summary.parquet` 和 `attention_topk.jsonl` 里会记录 action denoise 阶段：

```text
action_denoise   连续 action token 每个 denoise step 推理时，每个 action token 在每一层看哪些 prefix/action token
```

索引含义：

```text
query_position      当前 attention 矩阵里的 query 行号
query_token_index   该 query 对应完整 token_map.json 里的 token index
key_position        当前 attention 矩阵里的 key 列号
key_token_index     该 key 对应完整 token_map.json 里的 token index
```

默认保存 top-k 和聚合统计。如果要保存每层完整 attention 矩阵，启动 server 时加：

```bash
--save_full_attention True \
--layers_to_save 0 1 2 4 8 12 15 16
```

完整矩阵会保存到：

```text
optional_full_attention/attention_phase=<phase>_step=<step>_layer=<layer>.npz
```

独立 rollout 侧只保存视频，默认保存到：

```text
/home/jinjaguo/BH_MOE/attention_analysis/outputs/libero_rollouts/
  videos/
```

如果你想显式指定，也可以加：

```bash
--output_root /home/jinjaguo/BH_MOE/attention_analysis/outputs/libero_rollouts/videos
```

## 跑重标定实验

先跑一遍 baseline。结束后重新启动 Terminal A，把 `--attention-mode baseline` 改成：

```bash
--attention-mode recalibrated
```

再用 Terminal B 跑同一批 LIBERO 任务。这样同一个 rollout 推理方式不变，但 server 端会在 Gemma eager attention 内部把 text sink 的 attention mass 释放到 non-sink text tokens，然后用修改后的 attention 继续计算：

```text
attn_probs = recalibrate(attn_probs)
context = attn_probs @ value_states
```

也就是说 `recalibrated` 不是只改保存下来的 attention，而是真的进入动作 chunk 推理。

## 控制实验

`start_attention_server.py` 支持四种模式：

```text
baseline
recalibrated
random_text
visual_uniform
```

例如：

```bash
python /home/jinjaguo/BH_MOE/attention_analysis/scripts/start_attention_server.py \
  --attention-mode random_text \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero
```

## 画图

rollout 跑完后：

```bash
cd /home/jinjaguo/BH_MOE
python attention_analysis/scripts/plot_attention_trace.py \
  --trace_root attention_analysis/outputs/attention_trace \
  --output_dir attention_analysis/outputs/attention_trace_plots
```

图会保存到：

```text
attention_analysis/outputs/attention_trace_plots/
```

## 当前实现连接点

1. `start_attention_server.py` 复用 `start_server_record.py` 的 OpenPI policy 创建逻辑。
2. `AttentionTracingPolicy` 包装 PyTorch OpenPI policy，不改变 websocket client。
3. `ood_libero_rollouts.py` 每个 chunk request 里的 `task_name/trial_id/chunk_id` 会决定 attention 输出目录。
4. wrapper patch `transformers.models.gemma.modeling_gemma.eager_attention_forward`，因此保存的是实际用于 `attn @ V` 的 attention。
5. `recalibrated` 模式会在 `attn @ V` 之前修改 attention，因此可以测试动作级因果效应。

## 注意

必须使用包含 `model.safetensors` 的 PyTorch checkpoint，例如：

```text
/home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero
```

如果使用只有 JAX `params/` 的 checkpoint，attention tracing server 会直接报错。
