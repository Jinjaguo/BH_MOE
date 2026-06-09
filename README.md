# BH_MOE

This repository is currently used to:
1. Start an OpenPI websocket policy server.
2. Run LIBERO / OOD BDDL rollouts.
3. Record server-side chunk hidden states.
4. Record rollout-side trial metadata, event timestamps, and phase boundaries.

Below is the recommended end-to-end workflow.

## 1. Clone and set up `openpi`

```bash
cd ~
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source .venv/bin/activate
```

Install `openpi-client`:

```bash
cd ~/openpi/packages/openpi-client
pip install -e .
```

Apply the required transformers patch:

```bash
cd ~/openpi
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

Download the default checkpoint:

```bash
cd ~/openpi
source .venv/bin/activate
python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))"
```

If you want to use `pi0_libero`, download that checkpoint instead:

```bash
python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi0_libero'))"
```

At the end of this section, you also need to download the PyTorch checkpoint files from:

```text
https://huggingface.co/lerobot/pi05_libero_base/tree/main
```

Download all files from that page and place them under:

```text
~/.cache/openpi/pytorch_checkpoints/pi05_libero/
```

After downloading, this directory should contain files such as:

```text
~/.cache/openpi/pytorch_checkpoints/pi05_libero/model.safetensors
~/.cache/openpi/pytorch_checkpoints/pi05_libero/config.json
~/.cache/openpi/pytorch_checkpoints/pi05_libero/policy_preprocessor.json
~/.cache/openpi/pytorch_checkpoints/pi05_libero/policy_postprocessor.json
```

Reason:

`start_server_record.py` records hidden states through the PyTorch OpenPI inference
path. The default OpenPI checkpoint downloaded from `gs://openpi-assets/...` only
contains JAX `params/` weights, which are sufficient for normal serving but not
for this hidden-state tracing workflow. The Hugging Face `pi05_libero_base`
checkpoint provides the required `model.safetensors` PyTorch weights.

## 2. Clone and set up `LIBERO`

It is recommended to use a separate `conda` environment for LIBERO:

```bash
cd ~
conda create -n libero python=3.8.13 -y
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
```

To run rollouts, `openpi-client` must also be available in the same environment:

```bash
conda activate libero
cd ~/openpi/packages/openpi-client
pip install -e .
```

## 3. Clone `BH_MOE`

```bash
cd ~
git clone https://github.com/jinjaguo/BH_MOE.git
cd BH_MOE
```

Important entry-point scripts:

1. [start_serve.py](start_serve.py)  
   Standard OpenPI websocket server.
2. [start_server_record.py](start_server_record.py)  
   Websocket server with hidden-state tracing enabled.
3. [ood_libero_rollouts.py](ood_libero_rollouts.py)  
   Main rollout entry point for OOD BDDL evaluation.
4. [batch_libero_rollout.py](batch_libero_rollout.py)  
   Older batch rollout entry point. The recommended script is `ood_libero_rollouts.py`.

All helper utilities are located under:

```text
scripts/
```

## 4. How to run

It is recommended to use two terminals.

### Terminal A: start the policy server

```bash
cd ~/openpi
source .venv/bin/activate
python /home/jinjaguo/BH_MOE/start_server_record.py \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero
```

If you want to run `pi0_libero` instead:

```bash
cd ~/openpi
source .venv/bin/activate
python /home/jinjaguo/BH_MOE/start_server_record.py --env pi0_libero
```

The server saves hidden states to the `trace_root` sent by the rollout client.
For OOD rollouts this normally becomes:

```text
/home/jinjaguo/BH_MOE/OOD_exp/<experiment_name>/outputs/chunk_wise/<task_name>/trial_<trial_id>/chunk_<chunk_id>.pt
```

### Terminal B: run OOD rollouts

```bash
conda activate libero
cd /home/jinjaguo/BH_MOE
python ood_libero_rollouts.py \
  --input_dir /home/jinjaguo/BH_MOE/custom_bddl/libero_goal \
  --tasks_info /home/jinjaguo/BH_MOE/custom_bddl/libero_goal/dif_start_end_loc/tasks_info.txt \
  --libero_root /home/jinjaguo/LIBERO \
  --host localhost \
  --port 8000
```

The experiment output folder is inferred from the parent directory of
`--tasks_info`. For example, `.../dif_start_end_loc/tasks_info.txt` writes under
`OOD_exp/dif_start_end_loc/`. You can override this name explicitly:

```bash
python ood_libero_rollouts.py \
  --input_dir /home/jinjaguo/BH_MOE/custom_bddl/libero_goal \
  --tasks_info /home/jinjaguo/BH_MOE/custom_bddl/libero_goal/<exp_name>/tasks_info.txt \
  --libero_root /home/jinjaguo/LIBERO \
  --host localhost \
  --port 8000
```
Or
```bash
python ood_libero_rollouts.py \
  --input_dir /home/jinjaguo/BH_MOE/custom_bddl/libero_goal \
  --tasks_info /home/jinjaguo/BH_MOE/custom_bddl/libero_goal/change_pos/tasks_info.txt \
  --libero_root /home/jinjaguo/LIBERO \
  --experiment_name <experiment_name> \
  --host localhost \
  --port 8000
```

Current default stopping rule in `ood_libero_rollouts.py`:

1. Collect at least `20` successful trials per task.
2. Collect at least `20` failed trials per task.
3. Stop after `100` trials if the two targets above are not both reached.

These correspond to:

```bash
--target_successes 20
--target_failures 20
--max_trials 100
```

You can override them explicitly:

```bash
python ood_libero_rollouts.py \
  --input_dir /home/jinjaguo/BH_MOE/custom_bddl/libero_goal \
  --libero_root /home/jinjaguo/LIBERO \
  --host localhost \
  --port 8000 \
  --target_successes 20 \
  --target_failures 20 \
  --max_trials 100
```

## Output directories

Server-side chunk hidden states:

```text
OOD_exp/<experiment_name>/outputs/chunk_wise/<task_name>/trial_<trial_id>/chunk_<chunk_id>.pt
```

Rollout-side trial metadata:

```text
OOD_exp/<experiment_name>/outputs/chunk_wise/<task_name>/trial_<trial_id>/rollouts_state_record.jsonl
OOD_exp/<experiment_name>/outputs/chunk_wise/<task_name>/trial_<trial_id>/rollouts_finalize.jsonl
```

Rollout videos:

```text
OOD_exp/<experiment_name>/outputs/videos/<task_name>/
```

## Minimal run example

If everything is already installed, the shortest workflow is:

### Terminal A

```bash
cd ~/openpi
source .venv/bin/activate
python /home/jinjaguo/BH_MOE/start_server_record.py \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero
```

### Terminal B

```bash
conda activate libero
cd /home/jinjaguo/BH_MOE
python ood_libero_rollouts.py \
  --input_dir /home/jinjaguo/BH_MOE/custom_bddl/libero_goal \
  --libero_root /home/jinjaguo/LIBERO \
  --host localhost \
  --port 8000
```
