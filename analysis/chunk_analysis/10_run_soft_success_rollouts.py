"""
Run LIBERO rollouts with a post-hoc soft success stopper.

Purpose
-------
Run the plate task while preserving the original LIBERO/BDDL strict success
predicate, but add a rollout-side soft success stopper. This prevents trials
that have already placed the cream cheese on the plate from continuing until
timeout only because LIBERO's strict object-object `On` predicate is too tight.

Parameters
----------
--bddl_path: LIBERO BDDL file for the task.
--libero_root: path to the LIBERO checkout.
--task_name: output task folder name.
--host/--port: websocket policy server address. This can be the normal server
               or the causal-intervention server.
--trials: number of trials to run.
--seed: base seed. Trial i uses seed + i unless --trial_ids is provided.
--trial_ids: optional explicit trial ids to run. Seeds are seed + trial_id.
--xy_threshold: maximum xy center distance between cream cheese and plate.
--height_gap_max: maximum z gap between cream cheese and plate center.
--gripper_open_threshold: minimum gripper qpos mean to count as released when
                          --require_gripper_released is enabled.
--stable_steps: number of consecutive soft-success frames required to stop.
--require_contact/--no-require_contact: whether to require MuJoCo contact
                                        between cream cheese and plate.
--require_gripper_released: if set, require gripper_open > threshold. Disabled
                            by default because this LIBERO setup can report
                            near-zero gripper qpos for visually successful
                            placements.
--wait_steps/--max_steps/--replan_steps: rollout timing settings.
--resolution/--resize: LIBERO render and policy image sizes.
--output_root: video output root.
--chunk_root: rollout metadata and server trace root.

Usage
-----
python analysis/chunk_analysis/10_run_soft_success_rollouts.py \
    --bddl_path custom_bddl/libero_goal/dif_start_end_loc/put_the_cream_cheese_on_the_plate.bddl \
    --task_name put_the_cream_cheese_on_the_plate__soft_stop \
    --host localhost \
    --port 8000 \
    --trials 20 \
    --xy_threshold 0.06 \
    --stable_steps 5

Outputs
-------
Videos are saved under:
  OOD_exp/dif_start_end_loc/outputs/videos/soft_success/<task_name>/

Analysis metadata is saved under:
  <chunk_root>/<task_name>/trial_<trial_id>/rollouts_finalize.jsonl
where the `success` field is the soft-success label. The original LIBERO/BDDL
strict label is preserved as `strict_success` in `final_info` and in
`rollouts_soft_success.jsonl`.

Soft-stop labels are saved under:
  <chunk_root>/<task_name>/trial_<trial_id>/rollouts_soft_success.jsonl
and summarized at:
  OOD_exp/dif_start_end_loc/outputs/videos/soft_success/<task_name>/soft_success_summary.csv

Per-chunk hidden-state files are expected to be saved by the websocket policy
server under:
  <chunk_root>/<task_name>/trial_<trial_id>/chunk_*.pt

Use `start_server_record.py` or `08_start_causal_intervention_server.py`; the
plain `start_serve.py` server does not emit hidden-state chunks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import shutil
import sys
from dataclasses import asdict, dataclass
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
    LIBERO_DUMMY_ACTION,
    build_env,
    import_libero_modules,
    make_policy_observation,
    read_prompt_for_bddl,
)
from scripts.rollouts_state_record_helper import TrialLogger, sanitize_task_name  # noqa: E402


@dataclass
class SoftSuccessConfig:
    xy_threshold: float = 0.06
    height_gap_max: float = 0.12
    gripper_open_threshold: float = 0.02
    stable_steps: int = 5
    require_contact: bool = True
    require_gripper_released: bool = False


@dataclass
class SoftSuccessMetrics:
    xy_distance: float
    height_gap: float
    gripper_open: float
    contact: bool
    z_order_ok: bool
    xy_ok: bool
    height_ok: bool
    gripper_released: bool
    gripper_ok: bool
    soft_success_frame: bool


def unwrap_libero_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while not hasattr(current, "obj_body_id") and hasattr(current, "env"):
        if id(current) in seen:
            break
        seen.add(id(current))
        current = current.env
    return current


def object_position(env: Any, object_name: str) -> np.ndarray:
    base_env = unwrap_libero_env(env)
    return np.asarray(base_env.sim.data.body_xpos[base_env.obj_body_id[object_name]], dtype=np.float64)


def check_object_contact(env: Any, object_a: str, object_b: str) -> bool:
    base_env = unwrap_libero_env(env)
    return bool(base_env.check_contact(base_env.get_object(object_a), base_env.get_object(object_b)))


def gripper_open_from_obs(obs: dict[str, Any]) -> float:
    qpos = np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float64)
    if qpos.size == 0:
        return float("nan")
    return float(np.mean(qpos))


def soft_success_metrics(
    *,
    env: Any,
    obs: dict[str, Any],
    config: SoftSuccessConfig,
    cream_name: str = "cream_cheese_1",
    plate_name: str = "plate_1",
) -> SoftSuccessMetrics:
    cream_pos = object_position(env, cream_name)
    plate_pos = object_position(env, plate_name)
    xy_distance = float(np.linalg.norm(cream_pos[:2] - plate_pos[:2]))
    height_gap = float(cream_pos[2] - plate_pos[2])
    gripper_open = gripper_open_from_obs(obs)
    contact = check_object_contact(env, cream_name, plate_name)

    z_order_ok = bool(cream_pos[2] > plate_pos[2])
    xy_ok = bool(xy_distance < config.xy_threshold)
    height_ok = bool(0.0 < height_gap < config.height_gap_max)
    gripper_released = bool(np.isfinite(gripper_open) and gripper_open > config.gripper_open_threshold)
    gripper_ok = gripper_released or not config.require_gripper_released
    contact_ok = contact or not config.require_contact
    soft_success_frame = bool(xy_ok and z_order_ok and height_ok and gripper_ok and contact_ok)

    return SoftSuccessMetrics(
        xy_distance=xy_distance,
        height_gap=height_gap,
        gripper_open=gripper_open,
        contact=contact,
        z_order_ok=z_order_ok,
        xy_ok=xy_ok,
        height_ok=height_ok,
        gripper_released=gripper_released,
        gripper_ok=gripper_ok,
        soft_success_frame=soft_success_frame,
    )


def append_jsonl(path: pathlib.Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def reset_trial_dir(*, chunk_root: pathlib.Path, task_name: str, trial_id: int) -> None:
    trial_dir = chunk_root.expanduser().resolve() / sanitize_task_name(task_name) / f"trial_{int(trial_id)}"
    if trial_dir.exists():
        shutil.rmtree(trial_dir)


def prune_extra_chunk_files(*, trial_dir: pathlib.Path, num_chunks: int) -> None:
    for chunk_path in trial_dir.glob("chunk_*.pt"):
        try:
            chunk_id = int(chunk_path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if chunk_id >= num_chunks:
            chunk_path.unlink()


def verify_hidden_state_chunks(*, trial_dir: pathlib.Path, num_chunks: int) -> None:
    chunk_files = sorted(trial_dir.glob("chunk_*.pt"))
    if len(chunk_files) != num_chunks:
        raise RuntimeError(
            f"Expected {num_chunks} hidden-state chunk files in {trial_dir}, "
            f"found {len(chunk_files)}. The rollout client does not create these files; "
            "they must be written by a tracing policy server. Start the policy with "
            "start_server_record.py, or start the causal intervention server with an "
            "--output_task_name matching this --task_name."
        )


def write_summary(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id",
        "seed",
        "strict_success",
        "soft_success",
        "analysis_success",
        "soft_success_only",
        "true_failure",
        "stop_reason",
        "steps",
        "num_chunks",
        "stable_steps_required",
        "xy_threshold",
        "height_gap_max",
        "gripper_open_threshold",
        "require_contact",
        "require_gripper_released",
        "final_xy_distance",
        "final_height_gap",
        "final_gripper_open",
        "final_contact",
        "video_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_single_trial_soft_stop(
    *,
    env: Any,
    prompt_text: str,
    task_name: str,
    trial_id: int,
    ood_type: str,
    policy: WebsocketClientPolicy,
    resize: int,
    wait_steps: int,
    max_steps: int,
    replan_steps: int,
    save_wrist: bool,
    chunk_root: pathlib.Path,
    soft_config: SoftSuccessConfig,
) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray]]:
    logger = TrialLogger(
        task_name=task_name,
        trial_id=trial_id,
        ood_type=ood_type,
        chunk_root=chunk_root,
    )

    obs = env.reset()
    logger.observe_event_source(obs=obs, info=None)

    strict_done = False
    soft_success = False
    stop_reason = "timeout"
    stable_counter = 0
    t = 0
    frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    action_plan: list[list[float]] = []
    plan_i = 0
    reset_server = True
    info = None
    final_metrics: SoftSuccessMetrics | None = None

    while t < max_steps + wait_steps:
        if t < wait_steps:
            obs, _, strict_done, info = env.step(LIBERO_DUMMY_ACTION)
            logger.advance_steps(1)
            t += 1
            logger.observe_event_source(obs=obs, info=info)
            if strict_done:
                stop_reason = "strict_success"
                break
            continue

        if not isinstance(obs, dict):
            raise RuntimeError(f"Expected dict obs, got {type(obs)}")

        agentview, wrist, img, wrist_img, state = make_policy_observation(obs, resize)
        frames.append(agentview)
        if save_wrist:
            wrist_frames.append(wrist)

        if plan_i >= len(action_plan):
            base_payload = {
                "done": reset_server,
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": state,
                "prompt": str(prompt_text),
            }
            req = logger.build_policy_payload(base_payload=base_payload)
            received = policy.infer(req)
            action_chunk = np.asarray(received["actions"], dtype=np.float32)
            if action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
                raise ValueError(f"Expected actions shape (T, 7), got {action_chunk.shape}")
            action_plan = action_chunk[:replan_steps].tolist()
            logger.register_chunk(chunk_horizon=len(action_plan))
            plan_i = 0
            reset_server = False

        action = action_plan[plan_i]
        plan_i += 1
        obs, _, strict_done, info = env.step(action)
        logger.advance_steps(1)
        t += 1
        logger.observe_event_source(obs=obs, info=info)

        final_metrics = soft_success_metrics(env=env, obs=obs, config=soft_config)
        if final_metrics.soft_success_frame:
            stable_counter += 1
        else:
            stable_counter = 0

        if strict_done:
            stop_reason = "strict_success"
            break
        if stable_counter >= soft_config.stable_steps:
            soft_success = True
            stop_reason = "soft_success"
            break

    if final_metrics is None and isinstance(obs, dict):
        final_metrics = soft_success_metrics(env=env, obs=obs, config=soft_config)
        soft_success = bool(final_metrics.soft_success_frame and stable_counter >= soft_config.stable_steps)

    strict_success = bool(strict_done)
    soft_success = bool(soft_success or strict_success)
    soft_success_only = bool(soft_success and not strict_success)
    true_failure = bool((not strict_success) and (not soft_success))

    final_info = dict(info or {})
    final_info.update(
        {
            "strict_success": strict_success,
            "soft_success": soft_success,
            "soft_success_only": soft_success_only,
            "true_failure": true_failure,
            "stop_reason": stop_reason,
            "soft_success_config": asdict(soft_config),
            "soft_success_stable_counter": stable_counter,
            "final_soft_metrics": asdict(final_metrics) if final_metrics is not None else None,
        }
    )
    analysis_success = soft_success
    final_info["analysis_success_label"] = "soft_success"
    final_info["libero_strict_success"] = strict_success
    strict_record = logger.finalize(success=analysis_success, final_info=final_info)

    soft_record = {
        "task_name": sanitize_task_name(task_name),
        "trial_id": int(trial_id),
        "ood_type": ood_type,
        "strict_success": strict_success,
        "soft_success": soft_success,
        "analysis_success": analysis_success,
        "analysis_success_label": "soft_success",
        "soft_success_only": soft_success_only,
        "true_failure": true_failure,
        "stop_reason": stop_reason,
        "episode_length": int(t),
        "num_chunks": strict_record.get("num_chunks"),
        "chunk_ranges": strict_record.get("chunk_ranges"),
        "phase_boundaries": strict_record.get("phase_boundaries"),
        "soft_success_config": asdict(soft_config),
        "soft_success_stable_counter": stable_counter,
        "final_soft_metrics": asdict(final_metrics) if final_metrics is not None else None,
    }
    append_jsonl(logger.trial_dir / "rollouts_soft_success.jsonl", soft_record)
    prune_extra_chunk_files(trial_dir=logger.trial_dir, num_chunks=int(soft_record["num_chunks"]))
    verify_hidden_state_chunks(trial_dir=logger.trial_dir, num_chunks=int(soft_record["num_chunks"]))
    return soft_record, frames, wrist_frames


def parse_trial_ids(raw: list[int] | None, trials: int) -> list[int]:
    if raw:
        return list(raw)
    return list(range(trials))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl_path", type=pathlib.Path, required=True)
    parser.add_argument("--libero_root", type=pathlib.Path, default=pathlib.Path.home() / "LIBERO")
    parser.add_argument("--task_name", type=str, default="put_the_cream_cheese_on_the_plate__soft_stop")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--trial_ids", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--wait_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--save_wrist", action="store_true")
    parser.add_argument("--output_root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT / "soft_success")
    parser.add_argument("--chunk_root", type=pathlib.Path, default=DEFAULT_CHUNK_ROOT)
    parser.add_argument("--ood_type", type=str, default="soft_success")
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

    task_name = sanitize_task_name(args.task_name)
    soft_config = SoftSuccessConfig(
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
    output_dir = args.output_root.expanduser().resolve() / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for trial_id in parse_trial_ids(args.trial_ids, args.trials):
        trial_seed = args.seed + trial_id
        print(f"[soft-stop] trial={trial_id} seed={trial_seed}", flush=True)
        if not args.keep_existing_trial_dir:
            reset_trial_dir(
                chunk_root=args.chunk_root,
                task_name=task_name,
                trial_id=trial_id,
            )
        env = build_env(
            bddl_path,
            resolution=args.resolution,
            seed=trial_seed,
            offscreen_env_cls=offscreen_env_cls,
        )
        try:
            soft_record, frames, wrist_frames = run_single_trial_soft_stop(
                env=env,
                prompt_text=prompt_text,
                task_name=task_name,
                trial_id=trial_id,
                ood_type=args.ood_type,
                policy=policy,
                resize=args.resize,
                wait_steps=args.wait_steps,
                max_steps=args.max_steps,
                replan_steps=args.replan_steps,
                save_wrist=args.save_wrist,
                chunk_root=args.chunk_root.expanduser().resolve(),
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
        video_path = output_dir / f"trial{trial_id}_{suffix}.mp4"
        imageio.mimwrite(str(video_path), [np.asarray(x) for x in frames], fps=30)
        if args.save_wrist:
            wrist_path = output_dir / f"trial{trial_id}_{suffix}_wrist.mp4"
            imageio.mimwrite(str(wrist_path), [np.asarray(x) for x in wrist_frames], fps=30)

        metrics = soft_record.get("final_soft_metrics") or {}
        row = {
            "trial_id": trial_id,
            "seed": trial_seed,
            "strict_success": soft_record["strict_success"],
            "soft_success": soft_record["soft_success"],
            "analysis_success": soft_record["analysis_success"],
            "soft_success_only": soft_record["soft_success_only"],
            "true_failure": soft_record["true_failure"],
            "stop_reason": soft_record["stop_reason"],
            "steps": soft_record["episode_length"],
            "num_chunks": soft_record["num_chunks"],
            "stable_steps_required": soft_config.stable_steps,
            "xy_threshold": soft_config.xy_threshold,
            "height_gap_max": soft_config.height_gap_max,
            "gripper_open_threshold": soft_config.gripper_open_threshold,
            "require_contact": soft_config.require_contact,
            "require_gripper_released": soft_config.require_gripper_released,
            "final_xy_distance": metrics.get("xy_distance", math.nan),
            "final_height_gap": metrics.get("height_gap", math.nan),
            "final_gripper_open": metrics.get("gripper_open", math.nan),
            "final_contact": metrics.get("contact", False),
            "video_path": str(video_path),
        }
        summary_rows.append(row)
        write_summary(output_dir / "soft_success_summary.csv", summary_rows)
        print(
            f"[soft-stop] trial={trial_id} strict={row['strict_success']} "
            f"soft={row['soft_success']} reason={row['stop_reason']} steps={row['steps']}",
            flush=True,
        )

    strict_count = sum(1 for row in summary_rows if row["strict_success"])
    soft_only_count = sum(1 for row in summary_rows if row["soft_success_only"])
    true_failure_count = sum(1 for row in summary_rows if row["true_failure"])
    print(
        "Finished soft-stop rollouts: "
        f"strict_success={strict_count}, soft_success_only={soft_only_count}, "
        f"true_failure={true_failure_count}, total={len(summary_rows)}"
    )


if __name__ == "__main__":
    main()
