"""
Start an OpenPI websocket server with LIBERO attention tracing enabled.

Purpose
-------
Run the same server-side action inference workflow described in README.md, but
wrap the PyTorch OpenPI policy with AttentionTracingPolicy so every LIBERO
action chunk also saves attention summaries.

Parameters
----------
--env: default OpenPI environment/policy preset.
--port: websocket port.
--trace_root: root directory receiving attention trace outputs.
--attention_mode: baseline, recalibrated, random_text, or visual_uniform.
--save_full_attention: whether to save compressed full attention tensors.
--layers_to_save: optional layer ids for full attention dumps.
--topk: number of top attended keys saved per selected action query.
--sink_strategy: value_projection or none.
--p: sink retention factor for recalibrated mode.
policy:checkpoint --policy.config ... --policy.dir ...
  Optional explicit checkpoint override, same as start_server_record.py.

Usage
-----
cd ~/openpi
source .venv/bin/activate
python /home/jinjaguo/BH_MOE/attention_analysis/scripts/start_attention_server.py \
  --attention-mode baseline \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero

Outputs
-------
Attention traces are saved under:
<trace_root>/<task_name>/trial_<trial_id>/chunk_<chunk_id>/<attention_mode>/

Default trace_root:
/home/jinjaguo/BH_MOE/attention_analysis/outputs/attention_trace
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import socket
import sys
from typing import Literal

import tyro


def find_repo_root(start: pathlib.Path | None = None) -> pathlib.Path:
    current = (start or pathlib.Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for path in [current, *current.parents]:
        if (path / "start_server_record.py").exists() and (path / "attention_analysis").is_dir():
            return path
    raise RuntimeError(f"Could not find BH_MOE repo root from {current}")


REPO_ROOT = find_repo_root()
ATTENTION_ROOT = REPO_ROOT / "attention_analysis"
for candidate in (REPO_ROOT, ATTENTION_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from attention_trace.openpi_policy import AttentionTracingPolicy
from start_server_record import Checkpoint, Default, EnvMode, create_policy
from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server


@dataclasses.dataclass
class Args:
    """Arguments for the attention-tracing websocket server."""

    env: EnvMode = EnvMode.pi05_libero
    default_prompt: str | None = None
    port: int = 8000
    trace_root: pathlib.Path = ATTENTION_ROOT / "outputs" / "attention_trace"
    attention_mode: Literal["baseline", "recalibrated", "random_text", "visual_uniform"] = "baseline"
    save_full_attention: bool = False
    layers_to_save: list[int] | None = None
    topk: int = 20
    sink_strategy: Literal["value_projection", "none"] = "value_projection"
    p: float = 0.6
    record_policy_io: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if not getattr(policy, "_is_pytorch_model", False):
        raise RuntimeError(
            "Attention tracing requires the PyTorch OpenPI checkpoint path containing model.safetensors. "
            "Use policy:checkpoint --policy.config pi05_libero --policy.dir "
            "/home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero."
        )

    policy = AttentionTracingPolicy(
        policy=policy,
        record_root=args.trace_root,
        mode=args.attention_mode,
        save_full_attention=args.save_full_attention,
        layers_to_save=args.layers_to_save,
        topk=args.topk,
        sink_strategy=args.sink_strategy,
        p=args.p,
        default_task_name="default_task",
    )

    if args.record_policy_io:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating attention-tracing server (host: %s, ip: %s)", hostname, local_ip)
    logging.info("Saving attention traces under: %s", args.trace_root)

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
