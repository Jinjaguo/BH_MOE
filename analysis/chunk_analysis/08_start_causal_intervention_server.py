"""
Causal latent-intervention OpenPI websocket server.

Purpose
-------
Start an OpenPI websocket policy server that performs a causal intervention on
one selected inference chunk. During a failure rollout, the server can replace
the model's hidden state at chunk c with a hidden state taken from successful
rollouts, then continue the rollout normally. This tests whether an early
latent representation causally determines success/failure.

Parameters
----------
--env: default OpenPI environment/policy preset to serve.
--port: websocket port.
--source_trace_root: directory containing existing success traces. Defaults to
                     --trace_root for backward compatibility.
--trace_root: directory receiving new intervention traces.
--task_name: task folder name under trace_root.
--intervention_chunk: chunk index to intervene on, e.g. 2.
--state: intervention target. Supported values are
         cross_attention_action_x_after, action_head_input, chunk_vector_mean,
         and action_chunk_output.
--reference_mode: mean_success or trial. mean_success averages successful
                  traces at the intervention chunk; trial loads one success
                  trial specified by --reference_trial.
--reference_trial: success trial id used when --reference_mode trial.
--max_reference_successes: cap for mean_success reference averaging.
--output_task_suffix: suffix appended to task_name when saving intervention
                      traces, so baseline traces are not overwritten.
--output_task_name: exact output task folder for intervention traces. If set,
                    this overrides --output_task_suffix. Use this to align the
                    server's chunk_*.pt files with a rollout client's task_name.
policy:checkpoint --policy.config ... --policy.dir ... can be used as in
start_server_record.py to load an explicit checkpoint.

Usage
-----
python analysis/chunk_analysis/08_start_causal_intervention_server.py \
  --source_trace_root OOD_exp/dif_start_end_loc/outputs/chunk_wise \ 原来的chunk位置
  --trace_root OOD_exp/dif_start_end_loc/outputs/intervetion/put_the_cream_cheese_on_the_plate \ 输出位置
  --task_name put_the_cream_cheese_on_the_plate \ 任务名
  --output_task_name chunk0 \ 修改这里
  --intervention_chunk 0 \ 修改这里
  --state action_head_input \
  --port 8001
运行的时候删掉所有的中文，反斜杠后面必须立刻换行(空格也得删掉)

If no explicit policy:checkpoint is provided and the local PyTorch checkpoint
exists at ~/.cache/openpi/pytorch_checkpoints/pi05_libero/model.safetensors,
the script automatically uses it. Hidden-state intervention cannot run on the
default JAX `params/` checkpoint.

Outputs
-------
New intervention traces are saved under:
  <trace_root>/<task_name><output_task_suffix>/trial_<trial_id>/chunk_*.pt

Each intervened chunk includes metadata describing the reference source and
whether the hidden-state replacement was applied.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import socket
import sys
import types
from typing import Any, Literal

import numpy as np
import torch
import tyro


def find_repo_root(start: pathlib.Path | None = None) -> pathlib.Path:
    current = (start or pathlib.Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for path in [current, *current.parents]:
        if (path / "OOD_exp").is_dir() and ((path / "analysis").is_dir() or (path / "custom_bddl").is_dir()):
            return path
    raise RuntimeError(f"Could not find repository root from {current}")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.action_chunk_record_helper import _safe_prompt, _sanitize_name, _to_numpy
from scripts.rollouts_state_record_helper import sanitize_task_name
from start_server_record import (
    Checkpoint,
    Default,
    EnvMode,
    create_policy,
)

from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server


SUPPORTED_STATES = {
    "cross_attention_action_x_after",
    "action_head_input",
    "chunk_vector_mean",
    "action_chunk_output",
}

DEFAULT_PYTORCH_CHECKPOINTS = {
    EnvMode.pi05_libero: pathlib.Path.home() / ".cache" / "openpi" / "pytorch_checkpoints" / "pi05_libero",
}

DEFAULT_PYTORCH_CONFIGS = {
    EnvMode.pi05_libero: "pi05_libero",
}


@dataclasses.dataclass
class Args:
    """Arguments for the causal intervention websocket server."""

    env: EnvMode = EnvMode.pi05_libero
    default_prompt: str | None = None
    port: int = 8001
    source_trace_root: pathlib.Path | None = None
    trace_root: pathlib.Path = REPO_ROOT / "OOD_exp" / "dif_start_end_loc" / "outputs" / "chunk_wise"
    task_name: str = "put_the_cream_cheese_on_the_plate"
    intervention_chunk: int = 2
    state: str = "action_head_input"
    reference_mode: Literal["mean_success", "trial"] = "mean_success"
    reference_trial: int | None = None
    max_reference_successes: int = 20
    output_task_suffix: str = "__causal_intervention"
    output_task_name: str | None = None
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


@dataclasses.dataclass
class _InterventionState:
    text_encoder_output: np.ndarray | None = None
    text_token_mask: np.ndarray | None = None
    cross_attention_action_x_after: np.ndarray | None = None
    action_head_input: np.ndarray | None = None
    chunk_vector_mean: np.ndarray | None = None
    action_chunk_output: np.ndarray | None = None
    final_timestep: np.ndarray | None = None
    intervention_applied: bool = False
    intervention_state: str | None = None
    reference_source: str | None = None


def read_first_jsonl(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def success_trial_ids(task_dir: pathlib.Path) -> list[int]:
    ids: list[int] = []
    for trial_dir in sorted(task_dir.glob("trial_*"), key=lambda p: int(p.name.split("_", 1)[1])):
        finalize = trial_dir / "rollouts_finalize.jsonl"
        if not finalize.exists():
            continue
        record = read_first_jsonl(finalize)
        if bool(record.get("success")):
            ids.append(int(trial_dir.name.split("_", 1)[1]))
    return ids


def load_reference_tensor(
    *,
    trace_root: pathlib.Path,
    task_name: str,
    chunk_id: int,
    state: str,
    mode: str,
    reference_trial: int | None,
    max_reference_successes: int,
) -> tuple[np.ndarray, str]:
    load_kwargs = {"map_location": "cpu", "weights_only": False}
    task_dir = trace_root / sanitize_task_name(task_name)
    if mode == "trial":
        if reference_trial is None:
            raise ValueError("--reference_trial is required when --reference_mode trial")
        path = task_dir / f"trial_{reference_trial}" / f"chunk_{chunk_id:05d}.pt"
        payload = torch.load(path, **load_kwargs)
        return _to_numpy(payload[state]), f"trial_{reference_trial}/chunk_{chunk_id:05d}.pt:{state}"

    ids = success_trial_ids(task_dir)[:max_reference_successes]
    if not ids:
        raise ValueError(f"No successful reference trials found under {task_dir}")
    values = []
    used = []
    for trial_id in ids:
        path = task_dir / f"trial_{trial_id}" / f"chunk_{chunk_id:05d}.pt"
        if not path.exists():
            continue
        payload = torch.load(path, **load_kwargs)
        if state not in payload or payload[state] is None:
            continue
        values.append(_to_numpy(payload[state]).astype(np.float32, copy=False))
        used.append(trial_id)
    if not values:
        raise ValueError(f"No reference tensors found for state={state} chunk={chunk_id} in {task_dir}")
    ref = np.mean(np.stack(values, axis=0), axis=0)
    return ref, f"mean_success_trials={used}:chunk_{chunk_id:05d}.pt:{state}"


def adapt_reference(reference: np.ndarray, target: torch.Tensor, *, state: str) -> torch.Tensor:
    ref = torch.as_tensor(reference, dtype=target.dtype, device=target.device)
    if tuple(ref.shape) == tuple(target.shape):
        return ref
    if ref.ndim == target.ndim - 1 and tuple(ref.shape) == tuple(target.shape[1:]):
        return ref.unsqueeze(0).expand_as(target)
    if ref.ndim == 1 and target.ndim == 3 and ref.shape[0] == target.shape[-1]:
        return ref.view(1, 1, -1).expand_as(target)
    if state == "chunk_vector_mean" and ref.ndim in (1, 2) and target.ndim == 3:
        vec = ref.reshape(-1)
        if vec.shape[0] == target.shape[-1]:
            return vec.to(dtype=target.dtype, device=target.device).view(1, 1, -1).expand_as(target)
    raise ValueError(f"Cannot adapt reference shape {tuple(ref.shape)} to target shape {tuple(target.shape)}")


class _CausalInterventionCollector:
    """Collect traces and optionally replace one hidden state at one chunk."""

    def __init__(
        self,
        pi_model: Any,
        *,
        state: str,
        intervention_chunk: int,
        reference: np.ndarray,
        reference_source: str,
    ):
        self._pi_model = pi_model
        self._pg_model = pi_model.paligemma_with_expert
        self._state_name = state
        self._intervention_chunk = intervention_chunk
        self._reference = reference
        self._reference_source = reference_source
        self._state = _InterventionState()
        self._active = False
        self._current_chunk_id: int | None = None
        self._last_lang_seq_len: int | None = None
        self._last_lang_mask: torch.Tensor | None = None
        self._handles: list[Any] = []
        self._install()

    def _install(self) -> None:
        self._restore_eager_sample_actions()
        self._patch_embed_prefix()
        self._patch_denoise_step()
        self._patch_paligemma_forward()
        self._register_hooks()

    def _restore_eager_sample_actions(self) -> None:
        self._pi_model.sample_actions = types.MethodType(type(self._pi_model).sample_actions, self._pi_model)

    def _should_intervene(self) -> bool:
        return self._active and self._current_chunk_id == self._intervention_chunk

    def _mark_applied(self) -> None:
        self._state.intervention_applied = True
        self._state.intervention_state = self._state_name
        self._state.reference_source = self._reference_source

    def _patch_embed_prefix(self) -> None:
        original = self._pi_model.embed_prefix

        def wrapped(images, img_masks, lang_tokens, lang_masks):
            if self._active:
                self._last_lang_seq_len = int(lang_masks.shape[1])
                self._last_lang_mask = lang_masks.detach().cpu()
            return original(images, img_masks, lang_tokens, lang_masks)

        self._pi_model.embed_prefix = wrapped

    def _patch_denoise_step(self) -> None:
        original = self._pi_model.denoise_step

        def wrapped(state, prefix_pad_masks, past_key_values, x_t, timestep):
            if self._active:
                self._state.final_timestep = _to_numpy(timestep)
            return original(state, prefix_pad_masks, past_key_values, x_t, timestep)

        self._pi_model.denoise_step = wrapped

    def _patch_paligemma_forward(self) -> None:
        original = self._pg_model.forward

        def wrapped(*args, **kwargs):
            outputs = original(*args, **kwargs)
            if not self._active:
                return outputs

            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is None and len(args) >= 4:
                inputs_embeds = args[3]
            if not inputs_embeds or len(inputs_embeds) != 2:
                return outputs

            prefix_embeds, suffix_embeds = inputs_embeds
            if prefix_embeds is not None and suffix_embeds is None:
                prefix_output = outputs[0][0]
                if prefix_output is not None and self._last_lang_seq_len is not None:
                    text_hidden = prefix_output[:, -self._last_lang_seq_len :, :]
                    self._state.text_encoder_output = _to_numpy(text_hidden)
                    if self._last_lang_mask is not None:
                        self._state.text_token_mask = _to_numpy(self._last_lang_mask)
            return outputs

        self._pg_model.forward = wrapped

    def _register_hooks(self) -> None:
        last_expert_layer = self._pg_model.gemma_expert.model.layers[-1]

        def maybe_replace_cross_attention(_module, inputs):
            if not self._active or not inputs:
                return None
            hidden = inputs[0]
            horizon = self._pi_model.config.action_horizon
            suffix = hidden[:, -horizon:, :]
            self._state.cross_attention_action_x_after = _to_numpy(suffix)
            if self._state_name != "cross_attention_action_x_after" or not self._should_intervene():
                return None
            new_hidden = hidden.clone()
            new_hidden[:, -horizon:, :] = adapt_reference(self._reference, suffix, state=self._state_name)
            self._state.cross_attention_action_x_after = _to_numpy(new_hidden[:, -horizon:, :])
            self._mark_applied()
            return (new_hidden, *inputs[1:])

        def maybe_replace_action_head_input(_module, inputs):
            if not self._active or not inputs:
                return None
            hidden = inputs[0]
            self._state.action_head_input = _to_numpy(hidden)
            self._state.chunk_vector_mean = _to_numpy(hidden).mean(axis=1)
            if self._state_name not in {"action_head_input", "chunk_vector_mean"} or not self._should_intervene():
                return None
            new_hidden = adapt_reference(self._reference, hidden, state=self._state_name)
            self._state.action_head_input = _to_numpy(new_hidden)
            self._state.chunk_vector_mean = _to_numpy(new_hidden).mean(axis=1)
            self._mark_applied()
            return (new_hidden, *inputs[1:])

        self._handles.append(last_expert_layer.post_attention_layernorm.register_forward_pre_hook(maybe_replace_cross_attention))
        self._handles.append(self._pi_model.action_out_proj.register_forward_pre_hook(maybe_replace_action_head_input))

    def start(self, chunk_id: int) -> None:
        self._active = True
        self._current_chunk_id = chunk_id
        self._state = _InterventionState()
        self._last_lang_seq_len = None
        self._last_lang_mask = None

    def maybe_replace_action_chunk(self, actions: Any) -> Any:
        if self._state_name != "action_chunk_output" or not self._should_intervene():
            return actions
        target = torch.as_tensor(actions) if not isinstance(actions, torch.Tensor) else actions
        replacement = adapt_reference(self._reference, target, state=self._state_name)
        self._mark_applied()
        if isinstance(actions, np.ndarray):
            return replacement.detach().cpu().numpy().astype(actions.dtype, copy=False)
        return replacement

    def finish(self, action_chunk: Any) -> dict[str, Any]:
        self._state.action_chunk_output = _to_numpy(action_chunk)
        self._active = False
        self._current_chunk_id = None
        return dataclasses.asdict(self._state)

    def reset(self) -> None:
        self._active = False
        self._current_chunk_id = None
        self._state = _InterventionState()


class CausalInterventionPolicy:
    """Policy wrapper that replaces one hidden state and records intervention traces."""

    def __init__(
        self,
        policy: Any,
        *,
        record_root: pathlib.Path,
        source_task_name: str,
        output_task_name: str,
        intervention_chunk: int,
        state: str,
        reference: np.ndarray,
        reference_source: str,
    ):
        if state not in SUPPORTED_STATES:
            raise ValueError(f"Unsupported state: {state}. Supported: {sorted(SUPPORTED_STATES)}")
        if not getattr(policy, "_is_pytorch_model", False):
            raise TypeError("CausalInterventionPolicy requires a PyTorch OpenPI policy.")
        pi_model = getattr(policy, "_model", None)
        if pi_model is None or not hasattr(pi_model, "paligemma_with_expert"):
            raise TypeError("Wrapped policy does not expose the expected OpenPI PyTorch model structure.")

        self._policy = policy
        self._record_root = pathlib.Path(record_root).expanduser().resolve()
        self._source_task_name = sanitize_task_name(source_task_name)
        self._output_task_name = sanitize_task_name(output_task_name)
        self._intervention_chunk = intervention_chunk
        self._state_name = state
        self._reference_source = reference_source
        self._collector = _CausalInterventionCollector(
            pi_model,
            state=state,
            intervention_chunk=intervention_chunk,
            reference=reference,
            reference_source=reference_source,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        missing = [key for key in ("trial_id", "chunk_id") if key not in obs]
        if missing:
            raise KeyError(f"Policy request is missing intervention ids: {missing}")
        trial_id = str(obs["trial_id"])
        chunk_id = int(obs["chunk_id"])

        self._collector.start(chunk_id)
        try:
            result = self._policy.infer(obs, noise=noise)
            actions = self._collector.maybe_replace_action_chunk(result["actions"])
            result = dict(result)
            result["actions"] = actions
            trace = self._collector.finish(actions)
            self._save_trace(obs, trial_id, chunk_id, trace)
            return result
        except Exception:
            self._collector.reset()
            raise

    def _save_trace(self, obs: dict[str, Any], trial_id: str, chunk_id: int, trace: dict[str, Any]) -> None:
        task_dir = self._record_root / self._output_task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = task_dir / "intervention_manifest.json"
        if not manifest_path.exists():
            manifest_path.write_text(
                json.dumps(
                    {
                        "source_task_name": self._source_task_name,
                        "output_task_name": self._output_task_name,
                        "intervention_chunk": self._intervention_chunk,
                        "intervention_state": self._state_name,
                        "reference_source": self._reference_source,
                        "fields": [
                            "text_encoder_output",
                            "cross_attention_action_x_after",
                            "action_head_input",
                            "chunk_vector_mean",
                            "action_chunk_output",
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

        trial_dir = task_dir / f"trial_{trial_id}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "task_name": self._output_task_name,
                "source_task_name": self._source_task_name,
                "trial_id": trial_id,
                "chunk_id": chunk_id,
                "prompt": _safe_prompt(obs),
                "observation_done_flag": bool(obs.get("done", False)),
            },
            **trace,
        }
        torch.save(payload, trial_dir / f"chunk_{chunk_id:05d}.pt")


def resolve_policy_args(args: Args) -> Args:
    """Use a local PyTorch checkpoint by default when one is available."""
    if not isinstance(args.policy, Default):
        return args

    checkpoint_dir = DEFAULT_PYTORCH_CHECKPOINTS.get(args.env)
    config_name = DEFAULT_PYTORCH_CONFIGS.get(args.env)
    if checkpoint_dir is None or config_name is None:
        return args

    weight_path = checkpoint_dir.expanduser() / "model.safetensors"
    if not weight_path.exists():
        logging.warning(
            "No local PyTorch checkpoint found at %s. Falling back to the default %s policy, "
            "which may be JAX and cannot support hidden-state intervention.",
            weight_path,
            args.env.value,
        )
        return args

    logging.info(
        "Using local PyTorch checkpoint for intervention: config=%s dir=%s",
        config_name,
        checkpoint_dir.expanduser(),
    )
    return dataclasses.replace(
        args,
        policy=Checkpoint(config=config_name, dir=str(checkpoint_dir.expanduser())),
    )


def main(args: Args) -> None:
    if args.state not in SUPPORTED_STATES:
        raise ValueError(f"Unsupported --state {args.state}; choose from {sorted(SUPPORTED_STATES)}")

    trace_root = args.trace_root.expanduser().resolve()
    source_trace_root = (
        args.source_trace_root.expanduser().resolve()
        if args.source_trace_root is not None
        else trace_root
    )
    source_task_name = sanitize_task_name(args.task_name)
    output_task_name = sanitize_task_name(args.output_task_name or f"{source_task_name}{args.output_task_suffix}")
    reference, reference_source = load_reference_tensor(
        trace_root=source_trace_root,
        task_name=source_task_name,
        chunk_id=args.intervention_chunk,
        state=args.state,
        mode=args.reference_mode,
        reference_trial=args.reference_trial,
        max_reference_successes=args.max_reference_successes,
    )

    policy_args = resolve_policy_args(args)
    policy = create_policy(policy_args)
    policy_metadata = policy.metadata
    policy = CausalInterventionPolicy(
        policy,
        record_root=trace_root,
        source_task_name=source_task_name,
        output_task_name=output_task_name,
        intervention_chunk=args.intervention_chunk,
        state=args.state,
        reference=reference,
        reference_source=reference_source,
    )

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating causal intervention server (host: %s, ip: %s)", hostname, local_ip)
    logging.info("Intervention: task=%s chunk=%s state=%s ref=%s", source_task_name, args.intervention_chunk, args.state, reference_source)
    logging.info("Loading intervention references from: %s/%s/", source_trace_root, source_task_name)
    logging.info("Saving intervention traces under: %s/%s/", trace_root, output_task_name)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
