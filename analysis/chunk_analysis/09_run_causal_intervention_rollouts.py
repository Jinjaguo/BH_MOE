"""
Run LIBERO causal-intervention rollouts.

Purpose
-------
Replay failure seeds against a causal-intervention websocket server. The server
performs the actual hidden-state replacement, while this client runs the LIBERO
environment, sends chunk ids, saves videos, and writes rollout metadata. This
tests whether replacing an early failure hidden state with a success hidden
state can turn failure rollouts into successes or pull trajectories back onto
the successful manifold.

Parameters
----------
--bddl_path: LIBERO BDDL file for the task.
--libero_root: path to the LIBERO checkout.
--source_chunk_root: existing baseline chunk-wise output root.
--source_task_name: baseline task folder used to find failed trial ids.
--intervention_chunk: chunk index replaced by the server. Used for output
                      folder names; the server must be started with the same
                      value.
--output_task_name: leaf folder name for new metadata/traces. If omitted, this
                    is chunk<intervention_chunk>.
--host/--port: causal-intervention websocket server address.
--trial_ids: optional list of original failure trial ids to replay. If omitted,
             the script reads failed trial ids from source_chunk_root.
--num_trials: optional cap on failed trial ids to replay when --trial_ids is
              omitted. By default all failed source trials are replayed.
--seed: original rollout base seed. Each replay uses seed + original_trial_id.
--xy_threshold/--height_gap_max/--stable_steps: soft-success stopping and
                                                labeling settings.
--wait_steps/--max_steps/--replan_steps: rollout timing settings.
--resolution/--resize: LIBERO render and policy image sizes.
--output_root: video output root.
--chunk_root: rollout metadata root. The default writes to
              OOD_exp/dif_start_end_loc/outputs/intervetion/<source_task_name>/.

Usage
-----
python analysis/chunk_analysis/09_run_causal_intervention_rollouts.py \
    --bddl_path custom_bddl/libero_goal/put_the_cream_cheese_on_the_plate.bddl \
    --source_chunk_root OOD_exp/dif_start_end_loc/outputs/chunk_wise \
    --source_task_name put_the_cream_cheese_on_the_plate \
    --intervention_chunk 2 \ 修改这里
    --host localhost \
    --port 8001 \
    --num_trials 20
运行的时候删掉所有的中文，反斜杠后面必须立刻换行(空格也得删掉)

Outputs
-------
Videos are saved under:
  OOD_exp/dif_start_end_loc/outputs/videos/causal_intervention/<source_task_name>/chunk<id>/

Rollout metadata is saved under:
  <chunk_root>/<source_task_name>/chunk<id>/trial_<trial_id>/rollouts_finalize.jsonl

Server-side intervention traces are saved by
08_start_causal_intervention_server.py under the same task folder.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pathlib
import shlex
import subprocess
import sys
from typing import Any

import imageio
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy


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

from ood_libero_rollouts import (  # noqa: E402
    DEFAULT_CHUNK_ROOT,
    DEFAULT_OUTPUT_ROOT,
    build_env,
    import_libero_modules,
    read_prompt_for_bddl,
)
from scripts.rollouts_state_record_helper import sanitize_task_name  # noqa: E402


DEFAULT_INTERVENTION_ROOT = REPO_ROOT / "OOD_exp" / "dif_start_end_loc" / "outputs" / "intervetion"


def load_soft_success_module() -> Any:
    path = pathlib.Path(__file__).with_name("10_run_soft_success_rollouts.py")
    spec = importlib.util.spec_from_file_location("run_soft_success_rollouts", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load soft-success helper from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_soft_success_rollouts", module)
    spec.loader.exec_module(module)
    return module


def read_first_jsonl(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def failed_trial_ids(source_chunk_root: pathlib.Path, source_task_name: str) -> list[int]:
    task_dir = source_chunk_root / sanitize_task_name(source_task_name)
    ids: list[int] = []
    for trial_dir in sorted(task_dir.glob("trial_*"), key=lambda p: int(p.name.split("_", 1)[1])):
        finalize = trial_dir / "rollouts_finalize.jsonl"
        if not finalize.exists():
            continue
        record = read_first_jsonl(finalize)
        final_info = record.get("final_info") if isinstance(record.get("final_info"), dict) else {}
        success = record.get("analysis_success", final_info.get("analysis_success", record.get("success")))
        if not bool(success):
            ids.append(int(trial_dir.name.split("_", 1)[1]))
    if not ids:
        raise ValueError(f"No failed trials found under {task_dir}")
    return ids


def write_summary(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_trial_id",
                "intervention_trial_id",
                "intervention_chunk",
                "seed",
                "success",
                "strict_success",
                "soft_success",
                "analysis_success",
                "soft_success_only",
                "true_failure",
                "stop_reason",
                "steps",
                "num_chunks",
                "video_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def is_local_host(host: str) -> bool:
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def check_local_server_args(
    *,
    host: str,
    port: int,
    expected_trace_root: pathlib.Path,
    expected_output_task_name: str,
) -> None:
    if not is_local_host(host):
        return

    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,args"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return

    candidates = []
    for line in result.stdout.splitlines():
        if "08_start_causal_intervention_server.py" not in line:
            continue
        if f"--port {port}" not in line and f"--port={port}" not in line:
            continue
        candidates.append(line.strip())

    if not candidates:
        return

    expected_root_path = expected_trace_root.expanduser().resolve()
    expected_root_text = str(expected_root_path)
    mismatches = []
    for cmdline in candidates:
        try:
            parts = shlex.split(cmdline)
        except ValueError:
            parts = cmdline.split()

        trace_root_arg = None
        output_task_name_arg = None
        for idx, part in enumerate(parts):
            if part in {"--trace_root", "--trace-root"} and idx + 1 < len(parts):
                trace_root_arg = parts[idx + 1]
            elif part.startswith("--trace_root=") or part.startswith("--trace-root="):
                trace_root_arg = part.split("=", 1)[1]
            elif part in {"--output_task_name", "--output-task-name"} and idx + 1 < len(parts):
                output_task_name_arg = parts[idx + 1]
            elif part.startswith("--output_task_name=") or part.startswith("--output-task-name="):
                output_task_name_arg = part.split("=", 1)[1]

        has_root = False
        if trace_root_arg is not None:
            has_root = pathlib.Path(trace_root_arg).expanduser().resolve() == expected_root_path
        has_output = output_task_name_arg == expected_output_task_name
        if not (has_root and has_output):
            mismatches.append(cmdline)

    if mismatches:
        raise RuntimeError(
            "A local causal-intervention server is already running on the requested port, "
            "but its output args do not match this client. Stop that server and restart it "
            "with the new OOD_exp/dif_start_end_loc paths.\n"
            f"Expected trace_root: {expected_root_text}\n"
            f"Expected output_task_name: {expected_output_task_name}\n"
            "Running server command(s):\n  "
            + "\n  ".join(mismatches)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl_path", type=pathlib.Path, required=True)
    parser.add_argument("--libero_root", type=pathlib.Path, default=pathlib.Path.home() / "LIBERO")
    parser.add_argument("--source_chunk_root", type=pathlib.Path, default=DEFAULT_CHUNK_ROOT)
    parser.add_argument("--source_task_name", type=str, default="put_the_cream_cheese_on_the_plate")
    parser.add_argument("--intervention_chunk", type=int, default=2)
    parser.add_argument("--output_task_name", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--trial_ids", type=int, nargs="*", default=None)
    parser.add_argument("--num_trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--wait_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--save_wrist", action="store_true")
    parser.add_argument("--output_root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT / "causal_intervention")
    parser.add_argument("--chunk_root", type=pathlib.Path, default=DEFAULT_INTERVENTION_ROOT)
    parser.add_argument("--ood_type", type=str, default="causal_intervention")
    parser.add_argument("--xy_threshold", type=float, default=0.06)
    parser.add_argument("--height_gap_max", type=float, default=0.12)
    parser.add_argument("--gripper_open_threshold", type=float, default=0.02)
    parser.add_argument("--stable_steps", type=int, default=5)
    parser.add_argument("--require_contact", dest="require_contact", action="store_true", default=True)
    parser.add_argument("--no-require_contact", dest="require_contact", action="store_false")
    parser.add_argument("--require_gripper_released", action="store_true")
    parser.add_argument(
        "--keep_existing_trial_dir",
        action="store_true",
        help="Do not delete an existing output trial directory before rerun. By default old chunks are removed.",
    )
    args = parser.parse_args()

    bddl_path = args.bddl_path.expanduser().resolve()
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file does not exist: {bddl_path}")

    source_chunk_root = args.source_chunk_root.expanduser().resolve()
    source_task_name = sanitize_task_name(args.source_task_name)
    output_task_name = sanitize_task_name(args.output_task_name or f"chunk{args.intervention_chunk}")
    if args.trial_ids:
        replay_ids = list(args.trial_ids)
    else:
        replay_ids = failed_trial_ids(source_chunk_root, source_task_name)
        if args.num_trials is not None and args.num_trials > 0:
            replay_ids = replay_ids[: args.num_trials]

    soft_module = load_soft_success_module()
    soft_config = soft_module.SoftSuccessConfig(
        xy_threshold=args.xy_threshold,
        height_gap_max=args.height_gap_max,
        gripper_open_threshold=args.gripper_open_threshold,
        stable_steps=args.stable_steps,
        require_contact=args.require_contact,
        require_gripper_released=args.require_gripper_released,
    )
    offscreen_env_cls = import_libero_modules(args.libero_root)
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    prompt_text = read_prompt_for_bddl(bddl_path)

    output_dir = args.output_root.expanduser().resolve() / source_task_name / output_task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_root = args.chunk_root.expanduser().resolve() / source_task_name
    check_local_server_args(
        host=args.host,
        port=args.port,
        expected_trace_root=chunk_root,
        expected_output_task_name=output_task_name,
    )
    summary_rows: list[dict] = []

    for source_trial_id in replay_ids:
        intervention_trial_id = source_trial_id
        trial_seed = args.seed + source_trial_id
        print(
            f"[intervention] replay source_trial={source_trial_id} "
            f"as trial={intervention_trial_id} seed={trial_seed}",
            flush=True,
        )
        if not args.keep_existing_trial_dir:
            soft_module.reset_trial_dir(
                chunk_root=chunk_root,
                task_name=output_task_name,
                trial_id=intervention_trial_id,
            )
        env = build_env(
            bddl_path,
            resolution=args.resolution,
            seed=trial_seed,
            offscreen_env_cls=offscreen_env_cls,
        )
        try:
            soft_record, frames, wrist_frames = soft_module.run_single_trial_soft_stop(
                env=env,
                prompt_text=prompt_text,
                task_name=output_task_name,
                trial_id=intervention_trial_id,
                ood_type=args.ood_type,
                policy=policy,
                resize=args.resize,
                wait_steps=args.wait_steps,
                max_steps=args.max_steps,
                replan_steps=args.replan_steps,
                save_wrist=args.save_wrist,
                chunk_root=chunk_root,
                soft_config=soft_config,
            )
        finally:
            env.close()

        if soft_record["strict_success"]:
            suffix = "strict_success"
        elif soft_record["soft_success"]:
            suffix = "soft_success"
        else:
            suffix = "failure"
        video_path = output_dir / f"source_trial{source_trial_id}_intervention_{suffix}.mp4"
        imageio.mimwrite(str(video_path), [np.asarray(x) for x in frames], fps=30)
        if args.save_wrist:
            wrist_path = output_dir / f"source_trial{source_trial_id}_intervention_{suffix}_wrist.mp4"
            imageio.mimwrite(str(wrist_path), [np.asarray(x) for x in wrist_frames], fps=30)

        summary_rows.append(
            {
                "source_trial_id": source_trial_id,
                "intervention_trial_id": intervention_trial_id,
                "intervention_chunk": args.intervention_chunk,
                "seed": trial_seed,
                "success": soft_record["analysis_success"],
                "strict_success": soft_record["strict_success"],
                "soft_success": soft_record["soft_success"],
                "analysis_success": soft_record["analysis_success"],
                "soft_success_only": soft_record["soft_success_only"],
                "true_failure": soft_record["true_failure"],
                "stop_reason": soft_record["stop_reason"],
                "steps": int(soft_record["episode_length"]),
                "num_chunks": int(soft_record["num_chunks"]),
                "video_path": str(video_path),
            }
        )
        write_summary(output_dir / "intervention_summary.csv", summary_rows)
        print(
            f"[intervention] trial={source_trial_id} soft={soft_record['soft_success']} "
            f"strict={soft_record['strict_success']} reason={soft_record['stop_reason']} "
            f"steps={soft_record['episode_length']} video={video_path}",
            flush=True,
        )

    successes = sum(1 for row in summary_rows if row["analysis_success"])
    print(f"Finished intervention rollouts: success={successes}/{len(summary_rows)} summary={output_dir / 'intervention_summary.csv'}")


if __name__ == "__main__":
    main()
