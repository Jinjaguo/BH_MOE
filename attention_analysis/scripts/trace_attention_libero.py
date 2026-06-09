"""
LIBERO / pi0.5 attention tracing entry point.

Purpose
-------
Run a fixed LIBERO scene through a pi0.5/OpenPI policy with attention capture
enabled, comparing baseline inference against text-attention recalibration for
multiple prompts and random seeds.

Parameters
----------
--checkpoint: OpenPI checkpoint directory or file.
--libero_task: LIBERO task suite/task identifier.
--scene_id: fixed scene index to reset and reuse across prompts.
--prompts: one or more language prompts, for example "pick up the cup".
--seeds: random seeds for initial continuous action noise.
--num_denoise_steps: denoising / flow integration steps to trace.
--save_dir: root output directory for trace artifacts.
--save_full_attention: whether to save compressed full attention tensors.
--attn_backend: attention backend; use eager/debug mode when attention
                probabilities are hidden by memory-efficient attention.
--layers_to_save: optional selected layers for full attention dumps.
--target_objects: optional object labels aligned with prompts.
--distractor_objects: optional object labels used for visual/text contrast.
--dry_run_synthetic: generate synthetic attention tensors to validate the
                     output schema without OpenPI/LIBERO.

Usage
-----
python attention_analysis/scripts/trace_attention_libero.py \
  --checkpoint /home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero \
  --libero_task libero_object \
  --scene_id 0 \
  --prompts "pick up the cup" "pick up the bowl" \
  --seeds 0 1 2 3 4 \
  --num_denoise_steps 10 \
  --save_dir attention_analysis/outputs/attention_trace \
  --save_full_attention false \
  --attn_backend eager

Outputs
-------
Each run saves files under:
<save_dir>/scene_id=<scene_id>/seed=<seed>/prompt=<safe_prompt>/<mode>/

The run directory contains token_map.json, config.json,
attention_summary.parquet, attention_topk.jsonl, hidden_state_norms.parquet,
sink_tokens.json, action_outputs.npz, and optional_full_attention/ when enabled.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ATTENTION_ROOT = SCRIPT_DIR.parent
if str(ATTENTION_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTENTION_ROOT))

from attention_trace.hooks import AttentionTracer
from attention_trace.recalibration import recalibrate_attention_probs
from attention_trace.token_map import TokenSpanBuilder


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def safe_prompt(prompt: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.strip().lower()).strip("_")
    return safe or "prompt"


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def infer_target_object(prompt: str) -> str | None:
    words = re.findall(r"[a-zA-Z0-9_]+", prompt.lower())
    for stop in ("the", "a", "an", "up", "pick", "move", "put", "to", "on", "in", "front", "of"):
        while stop in words:
            words.remove(stop)
    return words[-1] if words else None


@dataclass
class PreparedRun:
    processor_outputs: dict[str, Any]
    tokenizer: Any
    images: dict[str, Any]
    proprio: np.ndarray
    continuous_action_tokens: np.ndarray
    action_outputs: dict[str, Any]


class SyntheticExperiment:
    """Small local generator for validating trace files before real model hooks are wired."""

    def __init__(self, num_denoise_steps: int) -> None:
        self.num_denoise_steps = num_denoise_steps
        self.vocab = {"pick": 10, "up": 11, "the": 12, "cup": 13, "bowl": 14}

    def prepare(self, prompt: str, seed: int) -> PreparedRun:
        tokens = re.findall(r"[a-zA-Z0-9_]+", prompt.lower())
        input_ids = np.asarray([[self.vocab.get(token, 99) for token in tokens]], dtype=np.int64)
        offsets = []
        cursor = 0
        for token in tokens:
            start = prompt.lower().find(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            cursor = end

        class Tokenizer:
            def __init__(self, inverse: dict[int, str]) -> None:
                self.inverse = inverse

            def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
                return [self.inverse.get(int(i), f"<unk_{int(i)}>") for i in ids]

        rng = np.random.default_rng(seed)
        return PreparedRun(
            processor_outputs={
                "input_ids": input_ids,
                "offset_mapping": np.asarray([offsets], dtype=np.int64),
                "image_token_counts": {"front": 64, "wrist": 64},
            },
            tokenizer=Tokenizer({value: key for key, value in self.vocab.items()}),
            images={"front": np.zeros((128, 128, 3), dtype=np.uint8), "wrist": np.zeros((128, 128, 3), dtype=np.uint8)},
            proprio=rng.normal(size=8),
            continuous_action_tokens=rng.normal(size=(10, 7)),
            action_outputs={"actions": rng.normal(size=(10, 7))},
        )

    def emit_attention(self, tracer: AttentionTracer, mode: str) -> None:
        key_len = len(tracer.token_map)
        query_len = max(max(tracer.query_indices) + 1, key_len)
        for step in range(self.num_denoise_steps):
            for layer in range(4):
                logits = torch.randn(1, 2, query_len, key_len)
                attn = torch.softmax(logits, dim=-1)
                if mode == "recalibrated":
                    text = set(tracer.token_spans.get("text", []))
                    sink = sorted(list(text)[:1])
                    non_sink = sorted(text - set(sink))
                    attn = recalibrate_attention_probs(attn, sink, non_sink, query_indices=tracer.query_indices)
                hidden = torch.randn(1, key_len, 64)
                tracer.capture(step, layer, f"synthetic.layers.{layer}.self_attn", attn, hidden)


class OpenPIExperiment:
    """Adapter boundary for the real OpenPI/LIBERO runtime."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._ensure_importable()
        raise RuntimeError(
            "OpenPI/LIBERO runtime integration still needs a model-specific attention forward patch. "
            "Run with --dry_run_synthetic true to validate outputs, then patch the attention module so "
            "it calls AttentionTracer.capture(attn_probs) and uses recalibrate_attention_probs before "
            "`context = attn_probs @ value_states`."
        )

    @staticmethod
    def _ensure_importable() -> None:
        try:
            import openpi  # noqa: F401

            return
        except ModuleNotFoundError as exc:
            candidate = pathlib.Path.home() / "openpi" / "src"
            if candidate.exists():
                sys.path.insert(0, str(candidate))
                return
            raise RuntimeError(
                "Could not import openpi. Activate the OpenPI environment or ensure ~/openpi/src exists."
            ) from exc


def write_run(
    prepared: PreparedRun,
    prompt: str,
    seed: int,
    mode: str,
    args: argparse.Namespace,
    experiment: SyntheticExperiment,
    target_object: str | None,
) -> None:
    token_result = TokenSpanBuilder().build(
        prepared.processor_outputs,
        prepared.tokenizer,
        prepared.images,
        prepared.proprio,
        continuous_action_tokens=prepared.continuous_action_tokens,
    )
    run_dir = (
        pathlib.Path(args.save_dir)
        / f"scene_id={args.scene_id}"
        / f"seed={seed}"
        / f"prompt={safe_prompt(prompt)}"
        / mode
    )
    tracer = AttentionTracer(
        token_result["token_map"],
        token_result["token_spans"],
        run_dir,
        save_full=args.save_full_attention,
        topk=args.topk,
        layers_to_save=args.layers_to_save,
        target_object=target_object,
        distractor_objects=args.distractor_objects,
    )
    tracer.start_run(prompt, seed, mode)
    experiment.emit_attention(tracer, mode)
    tracer.finish_run(prepared.action_outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace LIBERO / pi0.5 attention during action chunk inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--libero_task", required=True)
    parser.add_argument("--scene_id", type=int, default=0)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--num_denoise_steps", type=int, default=10)
    parser.add_argument("--save_dir", type=pathlib.Path, default=ATTENTION_ROOT / "outputs" / "attention_trace")
    parser.add_argument("--save_full_attention", type=str_to_bool, default=False)
    parser.add_argument("--attn_backend", default="eager")
    parser.add_argument("--layers_to_save", nargs="*", type=int, default=None)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--target_objects", nargs="*", default=None)
    parser.add_argument("--distractor_objects", nargs="*", default=[])
    parser.add_argument("--dry_run_synthetic", type=str_to_bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": args.checkpoint,
        "libero_task": args.libero_task,
        "scene_id": args.scene_id,
        "prompts": args.prompts,
        "seeds": args.seeds,
        "num_denoise_steps": args.num_denoise_steps,
        "attn_backend": args.attn_backend,
        "dry_run_synthetic": args.dry_run_synthetic,
    }
    (args.save_dir / "experiment_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.dry_run_synthetic:
        OpenPIExperiment(args)
        return

    experiment = SyntheticExperiment(args.num_denoise_steps)
    modes = ["baseline", "recalibrated"]
    for seed in args.seeds:
        set_all_seeds(seed)
        for prompt_idx, prompt in enumerate(args.prompts):
            target_object = (
                args.target_objects[prompt_idx]
                if args.target_objects and prompt_idx < len(args.target_objects)
                else infer_target_object(prompt)
            )
            prepared = experiment.prepare(prompt, seed)
            for mode in modes:
                write_run(prepared, prompt, seed, mode, args, experiment, target_object)


if __name__ == "__main__":
    main()
