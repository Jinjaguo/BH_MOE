"""
Purpose
-------
Record chunk-level hidden states for OpenPI PyTorch policy inference with minimal
intrusion to the websocket server stack.

This helper is designed for the server-side policy path:
1. Clear per-inference trace cache.
2. Reuse pre-registered hooks / patches.
3. Run the wrapped policy normally.
4. Save the current chunk's hidden states to disk.
5. Return the action chunk unchanged.

Recorded Features
-----------------
For each emitted action chunk, this helper records:
1. `text_encoder_output`
   Contextualized language token hidden states from the prefix/text encoder pass.
2. `cross_attention_action_x_after`
   Action-token representation right after attention residual addition and right
   before the post-attention layernorm / MLP path.
3. `action_head_input`
   Action-token representation actually sent into the action head.
4. `chunk_vector_mean`
   Mean pooling over the action-token sequence.
5. `action_chunk_output`
   Final action chunk returned by the policy.

Arguments
---------
`HiddenStateTracingPolicy(policy, record_root=None, default_task_name=None, disable_torch_compile=True)`
  `policy`:
    The already-created OpenPI policy instance to wrap.
  `record_root`:
    Optional fallback root directory where per-task trace folders are saved.
    If the websocket request contains `trace_root`, that request value is used
    instead so rollout-side metadata and server-side hidden states can share one
    chunk_wise directory.
  `default_task_name`:
    Deprecated fallback task name. Request payloads are expected to provide
    `task_name`, `trial_id`, and `chunk_id` explicitly.
  `disable_torch_compile`:
    If True, restore the eager `sample_actions` method so hooks stay reliable.

Usage
-----
from action_chunk_record_helper import HiddenStateTracingPolicy

wrapped = HiddenStateTracingPolicy(
    policy=policy,
    default_task_name="put_the_bowl_on_the_rack",
)

obs["trace_root"] = "OOD_exp/change_pos/outputs/chunk_wise"
result = wrapped.infer(obs)

Outputs
-------
Trace files are saved under:
`<record_root>/<task_name>/trial_<trial_id>/chunk_<chunk_id>.pt`

Each `.pt` file stores a dictionary containing metadata and the five recorded
features listed above.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import types
from typing import Any

import numpy as np
import torch


def _to_numpy(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32).cpu().numpy()
    return np.asarray(value)


def _safe_prompt(obs: dict[str, Any]) -> str | None:
    prompt = obs.get("prompt")
    if prompt is None:
        return None
    return str(prompt)


def _sanitize_name(value: str | None, fallback: str) -> str:
    raw = (value or fallback).strip()
    safe = raw.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return safe or fallback


@dataclasses.dataclass
class _TraceState:
    text_encoder_output: np.ndarray | None = None
    text_token_mask: np.ndarray | None = None
    cross_attention_action_x_after: np.ndarray | None = None
    action_head_input: np.ndarray | None = None
    chunk_vector_mean: np.ndarray | None = None
    action_chunk_output: np.ndarray | None = None
    final_timestep: np.ndarray | None = None


class _ActionChunkTraceCollector:
    """Collect trace tensors from an eager OpenPI PyTorch model."""

    def __init__(self, pi_model: Any, *, disable_torch_compile: bool = True):
        self._pi_model = pi_model
        self._pg_model = pi_model.paligemma_with_expert
        self._handles: list[Any] = []
        self._active = False
        self._state = _TraceState()
        self._last_lang_seq_len: int | None = None
        self._last_lang_mask: torch.Tensor | None = None
        self._disable_torch_compile = disable_torch_compile
        self._install()

    def _install(self) -> None:
        if self._disable_torch_compile:
            self._restore_eager_sample_actions()
        self._patch_embed_prefix()
        self._patch_denoise_step()
        self._patch_paligemma_forward()
        self._register_module_hooks()

    def _restore_eager_sample_actions(self) -> None:
        eager_fn = types.MethodType(type(self._pi_model).sample_actions, self._pi_model)
        self._pi_model.sample_actions = eager_fn

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

    def _register_module_hooks(self) -> None:
        last_expert_layer = self._pg_model.gemma_expert.model.layers[-1]

        def capture_x_after(_module, inputs):
            if not self._active or not inputs:
                return
            hidden = inputs[0]
            self._state.cross_attention_action_x_after = _to_numpy(
                hidden[:, -self._pi_model.config.action_horizon :, :]
            )

        def capture_action_head_input(_module, inputs):
            if not self._active or not inputs:
                return
            hidden = inputs[0]
            hidden_np = _to_numpy(hidden)
            self._state.action_head_input = hidden_np
            self._state.chunk_vector_mean = hidden_np.mean(axis=1)

        self._handles.append(last_expert_layer.post_attention_layernorm.register_forward_pre_hook(capture_x_after))
        self._handles.append(self._pi_model.action_out_proj.register_forward_pre_hook(capture_action_head_input))

    def start(self) -> None:
        self._active = True
        self._state = _TraceState()
        self._last_lang_seq_len = None
        self._last_lang_mask = None

    def finish(self, action_chunk: Any) -> dict[str, Any]:
        self._state.action_chunk_output = _to_numpy(action_chunk)
        self._active = False
        return dataclasses.asdict(self._state)

    def reset(self) -> None:
        self._active = False
        self._state = _TraceState()
        self._last_lang_seq_len = None
        self._last_lang_mask = None


class HiddenStateTracingPolicy:
    """Transparent policy wrapper that records one trace file per emitted action chunk."""

    def __init__(
        self,
        policy: Any,
        record_root: str | pathlib.Path | None = None,
        *,
        default_task_name: str | None = None,
        disable_torch_compile: bool = True,
    ):
        self._policy = policy
        self._record_root = pathlib.Path(record_root).expanduser().resolve() if record_root is not None else None
        if self._record_root is not None:
            self._record_root.mkdir(parents=True, exist_ok=True)
        self._default_task_name = _sanitize_name(default_task_name, "default_task")

        if not getattr(policy, "_is_pytorch_model", False):
            raise TypeError(
                "HiddenStateTracingPolicy currently supports only PyTorch OpenPI policies. "
                "The loaded policy is not a PyTorch policy, which usually means the checkpoint "
                "contains JAX `params/` weights instead of a `model.safetensors` file."
            )

        pi_model = getattr(policy, "_model", None)
        if pi_model is None or not hasattr(pi_model, "paligemma_with_expert"):
            raise TypeError("Wrapped policy does not expose the expected OpenPI PyTorch model structure.")

        self._collector = _ActionChunkTraceCollector(
            pi_model,
            disable_torch_compile=disable_torch_compile,
        )
        self._chunk_counter = 0

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        self._validate_request_ids(obs)
        task_name = _sanitize_name(obs.get("task_name"), self._default_task_name)
        trial_id, chunk_id = self._next_ids(obs)
        record_root = self._resolve_record_root(obs)
        self._collector.start()
        try:
            result = self._policy.infer(obs, noise=noise)
            trace = self._collector.finish(result["actions"])
            self._save_trace(obs, record_root, task_name, trial_id, chunk_id, trace)
            return result
        except Exception:
            self._collector.reset()
            raise

    def _next_ids(self, obs: dict[str, Any]) -> tuple[str, int]:
        explicit_trial = obs.get("trial_id")
        explicit_chunk = obs.get("chunk_id")

        trial_id = str(explicit_trial)
        chunk_id = int(explicit_chunk)
        self._chunk_counter = chunk_id + 1
        return trial_id, chunk_id

    def _validate_request_ids(self, obs: dict[str, Any]) -> None:
        missing = [key for key in ("task_name", "trial_id", "chunk_id") if key not in obs]
        if missing:
            raise KeyError(f"Policy request is missing required tracing ids: {missing}")

    def _resolve_record_root(self, obs: dict[str, Any]) -> pathlib.Path:
        request_trace_root = obs.get("trace_root")
        if request_trace_root:
            record_root = pathlib.Path(str(request_trace_root)).expanduser().resolve()
        elif self._record_root is not None:
            record_root = self._record_root
        else:
            raise KeyError(
                "Policy request is missing `trace_root` and the server was started without --trace_root. "
                "Pass trace_root from the rollout client or start the server with --trace_root."
            )
        record_root.mkdir(parents=True, exist_ok=True)
        return record_root

    def _save_trace(
        self,
        obs: dict[str, Any],
        record_root: pathlib.Path,
        task_name: str,
        trial_id: str,
        chunk_id: int,
        trace: dict[str, Any],
    ) -> None:
        task_dir = record_root / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = task_dir / "manifest.json"
        if not manifest_path.exists():
            manifest_path.write_text(
                json.dumps(
                    {
                        "task_name": task_name,
                        "record_dir": str(task_dir),
                        "format": "torch_save_dict",
                        "fields": [
                            "text_encoder_output",
                            "cross_attention_action_x_after",
                            "action_head_input",
                            "chunk_vector_mean",
                            "action_chunk_output",
                        ],
                    },
                    indent=2,
                )
            )

        trial_dir = task_dir / f"trial_{trial_id}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        output_path = trial_dir / f"chunk_{chunk_id:05d}.pt"

        payload = {
            "meta": {
                "task_name": task_name,
                "trial_id": trial_id,
                "chunk_id": chunk_id,
                "trace_root": str(record_root),
                "prompt": _safe_prompt(obs),
                "observation_done_flag": bool(obs.get("done", False)),
            },
            **trace,
        }
        torch.save(payload, output_path)
