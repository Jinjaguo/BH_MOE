"""
Purpose
-------
Connect attention tracing utilities to the existing OpenPI websocket inference
path used by LIBERO rollouts.

Parameters
----------
AttentionTracingPolicy(policy, record_root, mode, ...)
  policy: PyTorch OpenPI policy created by start_server_record.create_policy.
  record_root: root directory for attention trace outputs.
  mode: baseline, recalibrated, random_text, or visual_uniform.
  save_full_attention: whether to save compressed full attention tensors.
  layers_to_save: optional layer ids for full attention dumps.
  topk: number of top attended keys saved per action query.
  sink_strategy: value_projection detects sink keys from value-state norms.

Usage
-----
from attention_trace.openpi_policy import AttentionTracingPolicy

policy = AttentionTracingPolicy(
    policy=policy,
    record_root="attention_analysis/outputs/attention_trace",
    mode="baseline",
)
result = policy.infer(obs)

Outputs
-------
For each websocket action request, files are saved under:
<record_root>/<task_name>/trial_<trial_id>/chunk_<chunk_id>/<mode>/

The run directory contains token_map.json, config.json,
attention_summary.parquet, attention_topk.jsonl, hidden_state_norms.parquet,
sink_tokens.json, and action_outputs.npz.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import random
import types
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from .hooks import AttentionTracer
from .metrics import flatten_indices
from .recalibration import recalibrate_attention_probs
from .sinks import detect_sink_tokens


Mode = Literal["baseline", "recalibrated", "random_text", "visual_uniform"]
SinkStrategy = Literal["value_projection", "none"]


def _to_numpy(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32).cpu().numpy()
    return np.asarray(value)


def _sanitize_name(value: str | None, fallback: str) -> str:
    raw = (value or fallback).strip()
    safe = raw.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return safe or fallback


def _infer_target_object(prompt: str | None) -> str | None:
    if not prompt:
        return None
    words = [w.lower() for w in prompt.replace("_", " ").split()]
    stops = {"the", "a", "an", "up", "pick", "move", "put", "to", "on", "in", "front", "of", "and"}
    content = [w for w in words if w not in stops]
    return content[-1] if content else None


def _flatten_for_sink(value_states: torch.Tensor) -> torch.Tensor:
    """Convert [B, H, K, D] value states into [B, K, H*D] token features."""

    return value_states.transpose(1, 2).contiguous().flatten(start_dim=2)


@dataclasses.dataclass
class _PrefixInfo:
    image_token_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    text_token_ids: list[int] = dataclasses.field(default_factory=list)
    text_mask: list[bool] = dataclasses.field(default_factory=list)
    prefix_len: int = 0


class _OpenPIAttentionCollector:
    """Patch eager OpenPI attention so captured probabilities are the ones used for V multiplication."""

    def __init__(
        self,
        pi_model: Any,
        *,
        record_root: pathlib.Path,
        mode: Mode,
        save_full_attention: bool,
        layers_to_save: list[int] | None,
        topk: int,
        sink_strategy: SinkStrategy,
        p: float,
        camera_names: tuple[str, ...],
        disable_torch_compile: bool,
    ) -> None:
        self._pi_model = pi_model
        self._record_root = record_root
        self._mode = mode
        self._save_full_attention = save_full_attention
        self._layers_to_save = layers_to_save
        self._topk = topk
        self._sink_strategy = sink_strategy
        self._p = p
        self._camera_names = camera_names
        self._disable_torch_compile = disable_torch_compile

        self._active = False
        self._phase: str | None = None
        self._denoise_step = -1
        self._tracer: AttentionTracer | None = None
        self._run_context: dict[str, Any] = {}
        self._prefix_info = _PrefixInfo()
        self._original_embed_prefix = None
        self._original_denoise_step = None
        self._original_eager_attention_forward = None
        self._modeling_gemma = None
        self._install()

    def _install(self) -> None:
        if self._disable_torch_compile:
            self._pi_model.sample_actions = types.MethodType(type(self._pi_model).sample_actions, self._pi_model)
        self._patch_embed_prefix()
        self._patch_denoise_step()
        self._patch_eager_attention()

    def _patch_embed_prefix(self) -> None:
        self._original_embed_prefix = self._pi_model.embed_prefix

        def wrapped(images, img_masks, lang_tokens, lang_masks):
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._original_embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
            )
            if self._active:
                self._record_prefix_info(images, lang_tokens, lang_masks, prefix_embs)
                self._ensure_tracer(prefix_embs.device)
            return prefix_embs, prefix_pad_masks, prefix_att_masks

        self._pi_model.embed_prefix = wrapped

    def _record_prefix_info(self, images, lang_tokens, lang_masks, prefix_embs) -> None:
        text_len = int(lang_tokens.shape[1])
        prefix_len = int(prefix_embs.shape[1])
        num_images = max(1, len(images))
        image_total = max(0, prefix_len - text_len)
        base_count = image_total // num_images
        remainder = image_total % num_images
        counts = {}
        for idx in range(num_images):
            name = self._camera_names[idx] if idx < len(self._camera_names) else f"camera_{idx}"
            counts[name] = base_count + (1 if idx < remainder else 0)

        self._prefix_info = _PrefixInfo(
            image_token_counts=counts,
            text_token_ids=[int(v) for v in lang_tokens[0].detach().cpu().tolist()],
            text_mask=[bool(v) for v in lang_masks[0].detach().cpu().tolist()],
            prefix_len=prefix_len,
        )

    def _patch_denoise_step(self) -> None:
        self._original_denoise_step = self._pi_model.denoise_step

        def wrapped(state, prefix_pad_masks, past_key_values, x_t, timestep):
            if not self._active:
                return self._original_denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)
            self._phase = "denoise"
            self._denoise_step += 1
            try:
                return self._original_denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)
            finally:
                self._phase = None

        self._pi_model.denoise_step = wrapped

    def _patch_eager_attention(self) -> None:
        from transformers.models.gemma import modeling_gemma

        self._modeling_gemma = modeling_gemma
        self._original_eager_attention_forward = modeling_gemma.eager_attention_forward

        def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            if not self._active or self._tracer is None or self._phase != "denoise":
                return self._original_eager_attention_forward(
                    module,
                    query,
                    key,
                    value,
                    attention_mask,
                    scaling,
                    dropout=dropout,
                    **kwargs,
                )

            key_states = modeling_gemma.repeat_kv(key, module.num_key_value_groups)
            value_states = modeling_gemma.repeat_kv(value, module.num_key_value_groups)
            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
            layer_idx = int(getattr(module, "layer_idx", -1))
            phase, query_map, key_map, query_group = self._attention_layout(
                query_len=int(attn_weights.shape[-2]),
                key_len=int(attn_weights.shape[-1]),
            )

            value_proxy = _flatten_for_sink(value_states)
            sink_tokens = self._detect_sinks(value_proxy)
            attn_for_model = self._apply_mode(attn_weights, sink_tokens)

            self._tracer.capture(
                denoise_step=self._denoise_step if phase == "action_denoise" else -1,
                layer_idx=layer_idx,
                module_name=f"{phase}.layers.{layer_idx}.self_attn",
                attn_probs=attn_for_model,
                hidden_states=value_proxy,
                inference_phase=phase,
                query_indices=list(range(attn_for_model.shape[-2])),
                query_group=query_group,
                query_token_map=query_map,
                key_token_map=key_map,
            )

            attn_output = torch.matmul(attn_for_model, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, attn_for_model

        modeling_gemma.eager_attention_forward = wrapped

    def _attention_layout(
        self,
        *,
        query_len: int,
        key_len: int,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], str]:
        token_map = self._tracer.token_map if self._tracer is not None else []
        prefix_len = int(self._prefix_info.prefix_len)
        action_indices = sorted(flatten_indices(self._tracer.token_spans.get("continuous_action"))) if self._tracer else []

        if query_len == prefix_len and key_len == prefix_len:
            return (
                "prefix_prefill",
                token_map[:query_len],
                token_map[:key_len],
                "prefix_token",
            )

        action_query_map = [token_map[idx] for idx in action_indices[:query_len] if idx < len(token_map)]
        if len(action_query_map) < query_len:
            action_query_map.extend({"index": prefix_len + i, "type": "continuous_action"} for i in range(len(action_query_map), query_len))
        return (
            "action_denoise",
            action_query_map,
            token_map[:key_len],
            "continuous_action",
        )

    def _detect_sinks(self, value_proxy: torch.Tensor) -> dict[str, list[int]]:
        if self._sink_strategy == "none":
            return {"all": [], "visual": [], "text": [], "proprio": [], "action": [], "spike_dims": []}
        if self._tracer is None:
            return {"all": [], "visual": [], "text": [], "proprio": [], "action": [], "spike_dims": []}
        try:
            return detect_sink_tokens(value_proxy, self._tracer.token_spans)
        except ValueError:
            return {"all": [], "visual": [], "text": [], "proprio": [], "action": [], "spike_dims": []}

    def _apply_mode(self, attn_weights: torch.Tensor, sink_tokens: dict[str, list[int]]) -> torch.Tensor:
        if self._tracer is None:
            return attn_weights

        text_indices = flatten_indices(self._tracer.token_spans.get("text"))
        text_sink_indices = sink_tokens.get("text", [])
        non_sink_text_indices = sorted(text_indices - set(text_sink_indices))
        query_indices = list(range(attn_weights.shape[-2]))

        if self._mode == "baseline":
            return attn_weights
        if self._mode == "recalibrated":
            return recalibrate_attention_probs(
                attn_weights,
                text_sink_indices,
                non_sink_text_indices,
                query_indices=query_indices,
                p=self._p,
            )
        if self._mode == "random_text":
            out = attn_weights.clone()
            valid_text = sorted(i for i in non_sink_text_indices if i < out.shape[-1])
            if not valid_text:
                return out
            shuffled = valid_text[:]
            random.shuffle(shuffled)
            out[..., valid_text] = out[..., shuffled]
            return out
        if self._mode == "visual_uniform":
            out = attn_weights.clone()
            visual = sorted(i for i in flatten_indices(self._tracer.token_spans.get("image")) if i < out.shape[-1])
            if not visual:
                return out
            visual_mass = out[..., visual].sum(dim=-1, keepdim=True)
            out[..., visual] = visual_mass / max(1, len(visual))
            return out
        return attn_weights

    def start(self, obs: dict[str, Any], run_dir: pathlib.Path) -> None:
        self._active = True
        self._phase = None
        self._denoise_step = -1
        self._tracer = None
        self._prefix_info = _PrefixInfo()
        self._run_context = {
            "prompt": str(obs.get("prompt") or ""),
            "task_name": _sanitize_name(obs.get("task_name"), "default_task"),
            "trial_id": str(obs.get("trial_id")),
            "chunk_id": int(obs.get("chunk_id")),
            "run_dir": run_dir,
        }

    def _ensure_tracer(self, device: torch.device) -> None:
        if self._tracer is not None:
            return
        token_map, token_spans = self._build_token_map()
        query_indices = list(range(int(self._pi_model.config.action_horizon)))
        target_object = _infer_target_object(self._run_context.get("prompt"))
        self._tracer = AttentionTracer(
            token_map,
            token_spans,
            self._run_context["run_dir"],
            save_full=self._save_full_attention,
            topk=self._topk,
            layers_to_save=self._layers_to_save,
            target_object=target_object,
            distractor_objects=[],
            query_indices=query_indices,
        )
        self._tracer.start_run(
            prompt=self._run_context["prompt"],
            seed=int(self._run_context["trial_id"]) if str(self._run_context["trial_id"]).isdigit() else 0,
            mode=self._mode,
        )
        config_path = pathlib.Path(self._run_context["run_dir"]) / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(
            {
                "task_name": self._run_context["task_name"],
                "trial_id": self._run_context["trial_id"],
                "chunk_id": self._run_context["chunk_id"],
                "query_layout": {
                    "action_denoise": "query_position is local action token order; query_token_index maps to full token order",
                },
                "key_layout": {
                    "action_denoise": "key_position follows full prefix image/text plus suffix action token order",
                },
                "captured_phases": ["action_denoise"],
                "mode": self._mode,
                "sink_strategy": self._sink_strategy,
                "device": str(device),
            }
        )
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    def _build_token_map(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        token_map: list[dict[str, Any]] = []
        token_spans: dict[str, Any] = {"image": {}, "text": [], "proprio": [], "continuous_action": [], "fast_action": []}
        index = 0
        patch_size = 16

        for camera, count in self._prefix_info.image_token_counts.items():
            cols = int(math.sqrt(count)) if int(math.sqrt(count)) ** 2 == count else int(math.ceil(math.sqrt(count)))
            rows = int(math.ceil(count / max(cols, 1)))
            span = []
            for patch_id in range(count):
                patch_row = patch_id // cols
                patch_col = patch_id % cols
                token_map.append(
                    {
                        "index": index,
                        "type": "image",
                        "camera": camera,
                        "patch_row": patch_row,
                        "patch_col": patch_col,
                        "patch_id": patch_id,
                        "patch_grid_rows": rows,
                        "patch_grid_cols": cols,
                        "center_x": float((patch_col + 0.5) * patch_size),
                        "center_y": float((patch_row + 0.5) * patch_size),
                        "object_label": None,
                        "raw_span": None,
                    }
                )
                span.append(index)
                index += 1
            token_spans["image"][camera] = span

        for text_pos, token_id in enumerate(self._prefix_info.text_token_ids):
            is_valid_text = text_pos >= len(self._prefix_info.text_mask) or self._prefix_info.text_mask[text_pos]
            token_map.append(
                {
                    "index": index,
                    "type": "text" if is_valid_text else "text_padding",
                    "token_position": text_pos,
                    "token_id": int(token_id),
                    "token_str": None,
                    "char_start": None,
                    "char_end": None,
                    "valid": is_valid_text,
                }
            )
            if is_valid_text:
                token_spans["text"].append(index)
            index += 1

        if not getattr(self._pi_model, "pi05", False):
            token_map.append({"index": index, "type": "proprio", "name": "state_embedding", "value": None, "bin": None})
            token_spans["proprio"].append(index)
            index += 1

        for timestep in range(int(self._pi_model.config.action_horizon)):
            token_map.append({"index": index, "type": "continuous_action", "timestep": timestep})
            token_spans["continuous_action"].append(index)
            index += 1

        return token_map, token_spans

    def finish(self, action_chunk: Any) -> None:
        if self._tracer is not None:
            self._tracer.finish_run({"actions": _to_numpy(action_chunk)})
        self._active = False
        self._phase = None

    def reset(self) -> None:
        self._active = False
        self._phase = None
        self._tracer = None


class AttentionTracingPolicy:
    """Transparent websocket policy wrapper that records attention traces per action chunk."""

    def __init__(
        self,
        policy: Any,
        record_root: str | pathlib.Path,
        *,
        mode: Mode = "baseline",
        save_full_attention: bool = False,
        layers_to_save: list[int] | None = None,
        topk: int = 20,
        sink_strategy: SinkStrategy = "value_projection",
        p: float = 0.6,
        default_task_name: str | None = None,
        camera_names: tuple[str, ...] = ("front", "wrist"),
        disable_torch_compile: bool = True,
    ) -> None:
        self._policy = policy
        self._record_root = pathlib.Path(record_root).expanduser().resolve()
        self._record_root.mkdir(parents=True, exist_ok=True)
        self._mode = mode
        self._default_task_name = _sanitize_name(default_task_name, "default_task")

        if not getattr(policy, "_is_pytorch_model", False):
            raise TypeError("AttentionTracingPolicy requires a PyTorch OpenPI policy with model.safetensors.")

        pi_model = getattr(policy, "_model", None)
        if pi_model is None or not hasattr(pi_model, "paligemma_with_expert"):
            raise TypeError("Wrapped policy does not expose the expected PI0 PyTorch model structure.")

        self._collector = _OpenPIAttentionCollector(
            pi_model,
            record_root=self._record_root,
            mode=mode,
            save_full_attention=save_full_attention,
            layers_to_save=layers_to_save,
            topk=topk,
            sink_strategy=sink_strategy,
            p=p,
            camera_names=camera_names,
            disable_torch_compile=disable_torch_compile,
        )
        if disable_torch_compile and hasattr(policy, "_sample_actions"):
            policy._sample_actions = pi_model.sample_actions

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, Any]:
        self._validate_request_ids(obs)
        task_name = _sanitize_name(obs.get("task_name"), self._default_task_name)
        trial_id = str(obs.get("trial_id"))
        chunk_id = int(obs.get("chunk_id"))
        run_dir = self._record_root / task_name / f"trial_{trial_id}" / f"chunk_{chunk_id:05d}" / self._mode

        self._collector.start(obs, run_dir)
        try:
            result = self._policy.infer(obs, noise=noise)
            self._collector.finish(result.get("actions"))
            return result
        except Exception:
            self._collector.reset()
            raise

    @staticmethod
    def _validate_request_ids(obs: dict[str, Any]) -> None:
        missing = [key for key in ("task_name", "trial_id", "chunk_id") if key not in obs]
        if missing:
            raise KeyError(f"Policy request is missing required attention tracing ids: {missing}")
