"""
Purpose
-------
Run OOD LIBERO BDDL tasks with the OpenPI websocket policy and save rollout
videos only.

This is an attention-analysis copy of the repository-level
`ood_libero_rollouts.py`. It keeps the same LIBERO action-inference behavior,
but its default outputs live under `attention_analysis/outputs/libero_rollouts`
so attention experiments do not overwrite earlier OOD_exp rollout results.
This version intentionally does not write rollout-side chunk_wise metadata. It
still sends `task_name`, `trial_id`, and `chunk_id` in each websocket request so
the attention-tracing server can save its own chunk-aligned attention files.

For each BDDL task, it:
1. Repeatedly rolls out the task.
2. Sends explicit `task_name`, `trial_id`, and `chunk_id` with every websocket
   request.
3. Saves rollout videos.
4. Stops after a small diagnostic number of trials by default. Attention
   tracing is intended to inspect action-token attention, not to collect large
   success/failure rollout datasets.

Arguments
---------
`--input_dir`
  Root directory containing custom BDDL tasks.
`--tasks_info`
  Optional task list file containing relative BDDL paths.
`--libero_root`
  Path to the LIBERO checkout.
`--max_trials`
  Maximum number of trials to run per task. Defaults to 1.
`--host/--port`
  OpenPI websocket server address.
`--output_root`
  Root folder for rollout videos.

Examples
--------
python /home/jinjaguo/BH_MOE/attention_analysis/attention_rollouts.py \
  --input_dir /home/jinjaguo/BH_MOE/custom_bddl/libero_goal \
  --tasks_info /home/jinjaguo/BH_MOE/custom_bddl/libero_goal/tasks_info.txt \
  --libero_root /home/jinjaguo/LIBERO \
  --host localhost \
  --port 8000 \
  --max_trials 1

Outputs
-------
Videos are saved under:
`attention_analysis/outputs/libero_rollouts/videos/<suite_name>/<relative_task_dir>/<task_name>/`
"""

import argparse
import faulthandler
import gc
import math
import os
import pathlib
import sys
from typing import Optional, Tuple

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from robosuite.utils.errors import RandomizationError


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
DEFAULT_LIBERO_ROOT = pathlib.Path.home() / "LIBERO"
DEFAULT_ATTENTION_ROLLOUT_ROOT = pathlib.Path(__file__).resolve().parent / "outputs" / "libero_rollouts"
DEFAULT_OUTPUT_ROOT = DEFAULT_ATTENTION_ROLLOUT_ROOT / "videos"


def sanitize_task_name(task_name: str) -> str:
    safe = str(task_name).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    return safe or "default_task"


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quat shape (..., 4), got {quat.shape}")

    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)

    aa = (quat[:3] * 2.0 * math.acos(quat[3])) / den
    return aa.astype(np.float32)


def configure_libero_import(libero_root: pathlib.Path) -> pathlib.Path:
    libero_root = libero_root.expanduser().resolve()
    if not libero_root.exists():
        raise FileNotFoundError(f"LIBERO root does not exist: {libero_root}")
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))
    return libero_root


def import_libero_modules(libero_root: pathlib.Path):
    configure_libero_import(libero_root)
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv


def preprocess_rgb(rgb: np.ndarray, resize_size: int) -> np.ndarray:
    rgb = np.ascontiguousarray(rgb)
    rgb = image_tools.resize_with_pad(rgb, resize_size, resize_size)
    rgb = image_tools.convert_to_uint8(rgb)
    return rgb


def read_prompt_for_bddl(bddl_path: pathlib.Path) -> str:
    sibling_prompt = bddl_path.with_suffix(".txt")
    if sibling_prompt.exists():
        return sibling_prompt.read_text().strip()

    prompt_txt = bddl_path.parent / "prompt.txt"
    if prompt_txt.exists():
        return prompt_txt.read_text().strip()

    notes_txt = bddl_path.parent / "notes.txt"
    if notes_txt.exists():
        lines = notes_txt.read_text().strip().splitlines()
        if lines:
            return lines[0].strip()

    text = bddl_path.read_text()
    language_marker = "(:language"
    if language_marker in text:
        start = text.index(language_marker) + len(language_marker)
        end = text.find(")", start)
        if end != -1:
            return text[start:end].strip()

    return bddl_path.stem.replace("_", " ")


def build_env(task_bddl_file: pathlib.Path, resolution: int, seed: int, offscreen_env_cls):
    env = offscreen_env_cls(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def build_env_with_placement_retries(
    task_bddl_file: pathlib.Path,
    resolution: int,
    seed: int,
    offscreen_env_cls,
    placement_retries: int,
):
    last_error: Optional[RandomizationError] = None
    for retry_i in range(max(1, placement_retries)):
        retry_seed = seed + retry_i
        try:
            return build_env(
                task_bddl_file=task_bddl_file,
                resolution=resolution,
                seed=retry_seed,
                offscreen_env_cls=offscreen_env_cls,
            )
        except RandomizationError as exc:
            last_error = exc
            gc.collect()
            print(
                f"    placement sample failed for {task_bddl_file.name} "
                f"with seed={retry_seed} ({retry_i + 1}/{placement_retries}): {exc}",
                flush=True,
            )

    raise RuntimeError(
        f"Could not place all objects for {task_bddl_file} after {placement_retries} retries. "
        "Check BDDL regions for overlap, too-small ranges, object sizes, and yaw_rotation."
    ) from last_error


def make_policy_observation(obs: dict, resize: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = preprocess_rgb(agentview, resize)
    wrist_img = preprocess_rgb(wrist, resize)
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return agentview, wrist, img, wrist_img, state


def run_single_trial(
    *,
    env,
    prompt_text: str,
    task_name: str,
    trial_id: int,
    policy: WebsocketClientPolicy,
    resize: int,
    wait_steps: int,
    max_steps: int,
    replan_steps: int,
    save_wrist: bool,
):
    print(f"  [trial {trial_id}] calling env.reset()", flush=True)
    obs = env.reset()
    print(f"  [trial {trial_id}] env.reset() done", flush=True)

    done = False
    t = 0
    frames = []
    wrist_frames = []
    action_plan = []
    plan_i = 0
    chunk_id = 0
    reset_server = True
    info = None

    while t < max_steps + wait_steps and not done:
        if t < wait_steps:
            print(f"  [trial {trial_id}] warmup step {t + 1}/{wait_steps}", flush=True)
            obs, _, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        if not isinstance(obs, dict):
            raise RuntimeError(f"Expected dict obs, got {type(obs)}")

        agentview, wrist, img, wrist_img, state = make_policy_observation(obs, resize)
        frames.append(agentview)
        if save_wrist:
            wrist_frames.append(wrist)

        if plan_i >= len(action_plan):
            print(f"  [trial {trial_id}] requesting chunk at env step {t}", flush=True)
            base_payload = {
                "done": reset_server,
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": state,
                "prompt": str(prompt_text),
                "task_name": task_name,
                "trial_id": int(trial_id),
                "chunk_id": int(chunk_id),
            }
            received = policy.infer(base_payload)
            action_chunk = np.asarray(received["actions"], dtype=np.float32)
            if action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
                raise ValueError(f"Expected actions shape (T, 7), got {action_chunk.shape}")
            action_plan = action_chunk[:replan_steps].tolist()
            chunk_id += 1
            plan_i = 0
            reset_server = False
            print(
                f"  [trial {trial_id}] received chunk with horizon={len(action_plan)} at env step {t}",
                flush=True,
            )

        action = action_plan[plan_i]
        plan_i += 1
        obs, _, done, info = env.step(action)
        t += 1

    print(f"  [trial {trial_id}] rollout done, total_steps={t}, success={done}", flush=True)
    return done, t, frames, wrist_frames


def collect_task_trials(
    *,
    bddl_path: pathlib.Path,
    prompt_text: str,
    task_name: str,
    output_dir: pathlib.Path,
    policy: WebsocketClientPolicy,
    offscreen_env_cls,
    resolution: int,
    resize: int,
    wait_steps: int,
    max_steps: int,
    replan_steps: int,
    max_trials: int,
    seed: int,
    placement_retries: int,
    save_wrist: bool,
) -> Tuple[int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running task: {bddl_path}")
    print(f"Prompt text: {prompt_text}")
    print(f"Output dir: {output_dir}")
    print(f"Max trials: {max_trials}")

    successes = 0
    failures = 0
    trials_run = 0

    for trial_id in range(max_trials):
        print(f"  [trial {trial_id}] building env for {bddl_path.name}", flush=True)
        env = build_env_with_placement_retries(
            task_bddl_file=bddl_path,
            resolution=resolution,
            seed=seed + trial_id,
            offscreen_env_cls=offscreen_env_cls,
            placement_retries=placement_retries,
        )
        print(f"  [trial {trial_id}] env created", flush=True)
        try:
            done, t, frames, wrist_frames = run_single_trial(
                env=env,
                prompt_text=prompt_text,
                task_name=task_name,
                trial_id=trial_id,
                policy=policy,
                resize=resize,
                wait_steps=wait_steps,
                max_steps=max_steps,
                replan_steps=replan_steps,
                save_wrist=save_wrist,
            )
        finally:
            print(f"  [trial {trial_id}] closing env", flush=True)
            env.close()

        suffix = "success" if done else "failure"
        video_path = output_dir / f"trial{trial_id}_{suffix}.mp4"
        print(f"  [trial {trial_id}] writing video to {video_path}", flush=True)
        imageio.mimwrite(str(video_path), [np.asarray(x) for x in frames], fps=30)

        if save_wrist:
            wrist_path = output_dir / f"trial{trial_id}_{suffix}_wrist.mp4"
            imageio.mimwrite(str(wrist_path), [np.asarray(x) for x in wrist_frames], fps=30)

        trials_run += 1
        if done:
            successes += 1
        else:
            failures += 1

        print(
            f"  trial={trial_id} done={done} steps={t} video={video_path} "
            f"progress: success={successes}, failure={failures}, "
            f"trials={trials_run}/{max_trials}"
        )

    return successes, failures, trials_run


def iter_bddl_files(input_dir: pathlib.Path):
    yield from sorted(input_dir.rglob("*.bddl"))


def iter_bddl_files_from_tasks_info(tasks_info: pathlib.Path, input_dir: pathlib.Path):
    for raw_line in tasks_info.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rel_path = pathlib.Path(line)
        if rel_path.is_absolute():
            yield rel_path.resolve()
            continue

        candidates = []
        rel_variants = [rel_path]
        parts = rel_path.parts
        if parts and parts[0] == input_dir.name:
            rel_variants.append(pathlib.Path(*parts[1:]))
        if len(parts) >= 3 and parts[0] == "libero" and parts[1] == "bddl_files":
            rel_variants.append(pathlib.Path(*parts[2:]))

        for variant in rel_variants:
            candidates.append((input_dir / variant).resolve())
            candidates.append((input_dir / variant.name).resolve())

        for candidate in candidates:
            if candidate.exists():
                yield candidate
                break
        else:
            candidate_lines = "\n".join(f"  - {path}" for path in candidates)
            raise FileNotFoundError(
                f"Could not resolve BDDL path from tasks_info line: {line}\n"
                f"tasks_info: {tasks_info}\n"
                f"input_dir: {input_dir}\n"
                f"Tried:\n{candidate_lines}"
            )


def resolve_tasks_info_path(tasks_info: Optional[pathlib.Path], input_dir: pathlib.Path) -> Optional[pathlib.Path]:
    if tasks_info is None:
        default_path = input_dir / "tasks_info.txt"
        return default_path if default_path.exists() else None

    requested = tasks_info.expanduser().resolve()
    if requested.exists():
        return requested

    if tasks_info.name != "tasks_info.txt":
        raise FileNotFoundError(f"tasks_info file does not exist: {requested}")

    candidates = sorted(input_dir.rglob("tasks_info.txt"))
    if len(candidates) == 1:
        print(f"tasks_info file not found at {requested}; using {candidates[0].resolve()}", flush=True)
        return candidates[0].resolve()

    if not candidates:
        raise FileNotFoundError(
            f"tasks_info file does not exist: {requested}\n"
            f"No tasks_info.txt files were found under input_dir: {input_dir}"
        )

    candidate_lines = "\n".join(f"  - {path.resolve()}" for path in candidates)
    raise FileNotFoundError(
        f"tasks_info file does not exist: {requested}\n"
        "Multiple tasks_info.txt files were found under input_dir. Please pass the exact one:\n"
        f"{candidate_lines}"
    )


def task_has_existing_results(*, output_dir: pathlib.Path) -> bool:
    if output_dir.exists():
        if any(output_dir.glob("trial*_success.mp4")) or any(output_dir.glob("trial*_failure.mp4")):
            return True

    return False


def main():
    faulthandler.enable(all_threads=True)
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=pathlib.Path, required=True)
    parser.add_argument("--libero_root", type=pathlib.Path, default=DEFAULT_LIBERO_ROOT)
    parser.add_argument("--tasks_info", type=pathlib.Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--wait_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--max_trials", type=int, default=5)
    parser.add_argument(
        "--placement_retries",
        type=int,
        default=20,
        help="Retry environment construction with different seeds when BDDL placement sampling fails.",
    )
    parser.add_argument("--output_root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--skip_existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip tasks that already have rollout outputs. Enabled by default.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Force rerunning tasks even if outputs already exist.",
    )
    parser.add_argument("--save_wrist", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    print(f"input dir path: {input_dir}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    offscreen_env_cls = import_libero_modules(args.libero_root)
    policy = WebsocketClientPolicy(host=args.host, port=args.port)

    tasks_info = resolve_tasks_info_path(args.tasks_info, input_dir)
    if tasks_info is not None:
        print(f"Using tasks_info: {tasks_info}", flush=True)
        bddl_files = list(iter_bddl_files_from_tasks_info(tasks_info, input_dir))
    else:
        bddl_files = list(iter_bddl_files(input_dir))

    if not bddl_files:
        raise FileNotFoundError(f"No .bddl files found under: {input_dir}")

    suite_name = input_dir.name
    print(f"Suit name: {suite_name}")
    print(f"LIBERO source: {args.libero_root.expanduser().resolve()}")
    print(f"Input dir: {input_dir}")
    print(f"Detected suite root: {suite_name}")
    print(f"Found {len(bddl_files)} BDDL files")

    for bddl_path in bddl_files:
        if not bddl_path.exists():
            raise FileNotFoundError(f"BDDL file does not exist: {bddl_path}")

        prompt_text = read_prompt_for_bddl(bddl_path)
        task_name = sanitize_task_name(bddl_path.stem)
        rel_parent = bddl_path.parent.relative_to(input_dir)
        output_dir = args.output_root / suite_name / rel_parent / task_name
        print(f"Output path: {output_dir}")

        if args.skip_existing and task_has_existing_results(output_dir=output_dir):
            print(f"Skipping task {task_name}: existing rollout outputs detected")
            continue

        successes, failures, trials_run = collect_task_trials(
            bddl_path=bddl_path,
            prompt_text=prompt_text,
            task_name=task_name,
            output_dir=output_dir,
            policy=policy,
            offscreen_env_cls=offscreen_env_cls,
            resolution=args.resolution,
            resize=args.resize,
            wait_steps=args.wait_steps,
            max_steps=args.max_steps,
            replan_steps=args.replan_steps,
            max_trials=args.max_trials,
            seed=args.seed,
            placement_retries=args.placement_retries,
            save_wrist=args.save_wrist,
        )
        print(
            f"Finished task {task_name}: success={successes}, "
            f"failure={failures}, trials_run={trials_run}/{args.max_trials}"
        )


if __name__ == "__main__":
    main()
