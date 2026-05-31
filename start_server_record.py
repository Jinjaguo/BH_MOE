"""
Purpose
-------
Start an OpenPI websocket policy server that records one hidden-state trace file
 for every emitted action chunk.

This script mirrors `start_serve.py`, but wraps the policy with
`HiddenStateTracingPolicy` so each server-side inference saves the five requested
representations:
1. text encoder result
2. action-token representation after cross attention residual (`x_after`)
3. action head input
4. mean-pooled chunk vector
5. action chunk output

Arguments
---------
`--env`
  Which default OpenPI environment/policy preset to serve.
`--default_prompt`
  Fallback prompt if the request does not provide one.
`--port`
  Websocket port.
`--record`
  Whether to also enable the original OpenPI `PolicyRecorder`.
`--trace_root`
  Root directory used to save hidden-state trace files.
`--default_task_name`
  Fallback task name when the websocket request does not provide `task_name`.
`policy:checkpoint --policy.config ... --policy.dir ...`
  Optional explicit checkpoint override, same as `start_serve.py`.

Examples
--------
cd ~/openpi
source .venv/bin/activate
python /home/jinjaguo/BH_MOE/start_server_record.py --env pi05_libero

python /home/jinjaguo/BH_MOE/start_server_record.py \
  --env pi05_libero \
  --trace_root /home/jinjaguo/BH_MOE/OOD_exp/dif_start_end_loc/outputs/chunk_wise \
  --default_task_name libero_debug_run

Outputs
-------
Hidden-state trace files are saved under:
`<trace_root>/<task_name>/trial_<trial_id>/chunk_<chunk_id>.pt`

If `--record` is enabled, the original OpenPI policy recorder still writes to:
`policy_records/`
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import os
import pathlib
import socket
import sys

import tyro

from scripts.action_chunk_record_helper import HiddenStateTracingPolicy


def _ensure_openpi_importable() -> None:
    try:
        import openpi  # noqa: F401

        return
    except ModuleNotFoundError:
        candidate = pathlib.Path(__file__).resolve().parent.parent / "openpi" / "src"
        if candidate.exists():
            sys.path.insert(0, str(candidate))


_ensure_openpi_importable()

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


#TODO : change your root here !!
DEFAULT_TRACE_ROOT = pathlib.Path("/home/jinjaguo/BH_MOE/OOD_exp/dif_start_end_loc/outputs/chunk_wise")


class EnvMode(enum.Enum):
    """Supported environments."""

    pi0_base = "pi0_base"
    pi0_libero = "pi0_libero"
    pi05_base = "pi05_base"
    pi05_libero = "pi05_libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the trace-enabled websocket policy server."""

    env: EnvMode = EnvMode.pi05_libero
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    trace_root: pathlib.Path = DEFAULT_TRACE_ROOT
    default_task_name: str = "default_task"
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.pi0_base: Checkpoint(
        config="pi0_aloha",
        dir="gs://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.pi0_libero: Checkpoint(
        config="pi0_libero",
        dir="gs://openpi-assets/checkpoints/pi0_libero",
    ),
    EnvMode.pi05_base: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.pi05_libero: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def _maybe_load_fallback_norm_stats(
    train_config: _config.TrainConfig,
    checkpoint_dir: str | pathlib.Path,
    *,
    config_name: str,
):
    checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser()
    weight_path = checkpoint_dir / "model.safetensors"
    assets_dir = checkpoint_dir / "assets"

    if not weight_path.exists():
        return None

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        return None

    local_norm_stats_path = assets_dir / data_config.asset_id / "norm_stats.json"
    if local_norm_stats_path.exists():
        return None

    fallback_assets_dir = pathlib.Path.home() / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / config_name / "assets"
    fallback_norm_stats_path = fallback_assets_dir / data_config.asset_id / "norm_stats.json"
    if not fallback_norm_stats_path.exists():
        return None

    logging.info(
        "Checkpoint assets are missing at %s. Falling back to norm stats from %s",
        local_norm_stats_path,
        fallback_norm_stats_path,
    )
    return _checkpoints.load_norm_stats(fallback_assets_dir, data_config.asset_id)


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        train_config = _config.get_config(checkpoint.config)
        return _policy_config.create_trained_policy(
            train_config,
            checkpoint.dir,
            default_prompt=default_prompt,
            norm_stats=_maybe_load_fallback_norm_stats(
                train_config,
                checkpoint.dir,
                config_name=checkpoint.config,
            ),
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            return _policy_config.create_trained_policy(
                train_config,
                args.policy.dir,
                default_prompt=args.default_prompt,
                norm_stats=_maybe_load_fallback_norm_stats(
                    train_config,
                    args.policy.dir,
                    config_name=args.policy.config,
                ),
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if not getattr(policy, "_is_pytorch_model", False):
        raise RuntimeError(
            "start_server_record.py requires a PyTorch OpenPI checkpoint because hidden-state tracing "
            "is implemented on the PyTorch inference path only.\n"
            "The current policy was loaded as a non-PyTorch policy.\n"
            "Most likely cause: the checkpoint directory contains `params/` but not `model.safetensors`.\n"
            "If you want to serve without tracing, use `start_serve.py` instead."
        )

    policy = HiddenStateTracingPolicy(
        policy=policy,
        record_root=args.trace_root,
        default_task_name=args.default_task_name,
    )

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating trace-enabled server (host: %s, ip: %s)", hostname, local_ip)
    logging.info("Saving hidden-state traces under: %s/<task_name>/trial_<trial_id>/", args.trace_root)

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
