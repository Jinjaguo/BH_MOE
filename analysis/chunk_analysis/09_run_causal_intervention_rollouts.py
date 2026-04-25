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
--output_task_name: intervention task folder name for new metadata/traces.
--host/--port: causal-intervention websocket server address.
--trial_ids: optional list of original failure trial ids to replay. If omitted,
             the script reads failed trial ids from source_chunk_root.
--num_trials: number of failed trial ids to replay when --trial_ids is omitted.
--seed: original rollout base seed. Each replay uses seed + original_trial_id.
--wait_steps/--max_steps/--replan_steps: rollout timing settings.
--resolution/--resize: LIBERO render and policy image sizes.
--output_root: video output root.
--chunk_root: rollout metadata root. Should usually match the server trace_root.

Usage
-----
python analysis/chunk_analysis/09_run_causal_intervention_rollouts.py \
    --bddl_path custom_bddl/libero_goal/put_the_cream_cheese_on_the_plate.bddl \
    --source_chunk_root OOD_exp/outputs/chunk_wise \
    --source_task_name put_the_cream_cheese_on_the_plate \
    --output_task_name put_the_cream_cheese_on_the_plate__causal_intervention \
    --host localhost \
    --port 8001 \
    --num_trials 20

Outputs
-------
Videos are saved under:
  OOD_exp/outputs/videos/causal_intervention/<output_task_name>/

Rollout metadata is saved under:
  <chunk_root>/<output_task_name>/trial_<trial_id>/rollouts_finalize.jsonl

Server-side intervention traces are saved by
08_start_causal_intervention_server.py under the same task folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

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
    run_single_trial,
)
from scripts.rollouts_state_record_helper import sanitize_task_name  # noqa: E402


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
        if not bool(record.get("success")):
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
                "seed",
                "success",
                "steps",
                "video_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl_path", type=pathlib.Path, required=True)
    parser.add_argument("--libero_root", type=pathlib.Path, default=pathlib.Path.home() / "LIBERO")
    parser.add_argument("--source_chunk_root", type=pathlib.Path, default=DEFAULT_CHUNK_ROOT)
    parser.add_argument("--source_task_name", type=str, default="put_the_cream_cheese_on_the_plate")
    parser.add_argument("--output_task_name", type=str, default="put_the_cream_cheese_on_the_plate__causal_intervention")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--trial_ids", type=int, nargs="*", default=None)
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--wait_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--save_wrist", action="store_true")
    parser.add_argument("--output_root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT / "causal_intervention")
    parser.add_argument("--chunk_root", type=pathlib.Path, default=DEFAULT_CHUNK_ROOT)
    parser.add_argument("--ood_type", type=str, default="causal_intervention")
    args = parser.parse_args()

    bddl_path = args.bddl_path.expanduser().resolve()
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file does not exist: {bddl_path}")

    source_chunk_root = args.source_chunk_root.expanduser().resolve()
    output_task_name = sanitize_task_name(args.output_task_name)
    if args.trial_ids:
        replay_ids = list(args.trial_ids)
    else:
        replay_ids = failed_trial_ids(source_chunk_root, args.source_task_name)[: args.num_trials]

    offscreen_env_cls = import_libero_modules(args.libero_root)
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    prompt_text = read_prompt_for_bddl(bddl_path)

    output_dir = args.output_root.expanduser().resolve() / output_task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for source_trial_id in replay_ids:
        intervention_trial_id = source_trial_id
        trial_seed = args.seed + source_trial_id
        print(
            f"[intervention] replay source_trial={source_trial_id} "
            f"as trial={intervention_trial_id} seed={trial_seed}",
            flush=True,
        )
        env = build_env(
            bddl_path,
            resolution=args.resolution,
            seed=trial_seed,
            offscreen_env_cls=offscreen_env_cls,
        )
        try:
            done, steps, frames, wrist_frames = run_single_trial(
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
                chunk_root=args.chunk_root.expanduser().resolve(),
            )
        finally:
            env.close()

        suffix = "success" if done else "failure"
        video_path = output_dir / f"source_trial{source_trial_id}_intervention_{suffix}.mp4"
        imageio.mimwrite(str(video_path), [np.asarray(x) for x in frames], fps=30)
        if args.save_wrist:
            wrist_path = output_dir / f"source_trial{source_trial_id}_intervention_{suffix}_wrist.mp4"
            imageio.mimwrite(str(wrist_path), [np.asarray(x) for x in wrist_frames], fps=30)

        summary_rows.append(
            {
                "source_trial_id": source_trial_id,
                "intervention_trial_id": intervention_trial_id,
                "seed": trial_seed,
                "success": bool(done),
                "steps": int(steps),
                "video_path": str(video_path),
            }
        )
        write_summary(output_dir / "intervention_summary.csv", summary_rows)
        print(f"[intervention] trial={source_trial_id} success={done} steps={steps} video={video_path}", flush=True)

    successes = sum(1 for row in summary_rows if row["success"])
    print(f"Finished intervention rollouts: success={successes}/{len(summary_rows)} summary={output_dir / 'intervention_summary.csv'}")


if __name__ == "__main__":
    main()
