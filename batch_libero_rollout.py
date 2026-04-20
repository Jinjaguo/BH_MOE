"""
Purpose
-------
Batch rollout custom LIBERO BDDL tasks, save videos, and write rollout-side
chunk/trial metadata aligned with server-side hidden-state traces.

This script:
1. Discovers custom `.bddl` tasks.
2. Sends explicit `task_name`, `trial_id`, and `chunk_id` with every websocket
   policy request.
3. Saves rollout videos.
4. Writes per-trial JSONL metadata under `OOD_exp/outputs/chunk_wise/`.

Arguments
---------
`--input_dir`         Root directory containing custom BDDL tasks.
`--tasks_info`        Optional task list file.
`--libero_root`       Path to the LIBERO checkout.
`--trials`            Maximum rollout attempts per task.
`--target_successes`  Stop the task once this many successful trials are reached.
`--replan_steps`      Number of actions consumed from each chunk.
`--chunk_root`        Root folder for chunk-wise rollout metadata.

Examples
--------
python batch_libero_rollout.py \
  --input_dir /home/jinjaguo/BH_MOE/custom_bddl/libero_goal \
  --tasks_info /home/jinjaguo/BH_MOE/custom_bddl/libero_goal/tasks_info.txt \
  --libero_root /home/jinjaguo/LIBERO \
  --trials 100 \
  --target_successes 5 \
  --host localhost \
  --port 8000

Outputs
-------
Videos are saved under `OOD_exp/outputs/videos/`.

Trial metadata is saved under:
`OOD_exp/outputs/chunk_wise/<task_name>/trial_<trial_id>/`
"""

import argparse
import math
import pathlib
import sys
from typing import Tuple

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from scripts.rollouts_state_record_helper import TrialLogger, sanitize_task_name


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
DEFAULT_LIBERO_ROOT = pathlib.Path.home() / "LIBERO"
DEFAULT_OUTPUT_ROOT = pathlib.Path(__file__).resolve().parent / "OOD_exp" / "outputs" / "videos"
DEFAULT_CHUNK_ROOT = pathlib.Path(__file__).resolve().parent / "OOD_exp" / "outputs" / "chunk_wise"


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


def make_policy_observation(obs: dict, resize: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    chunk_root: pathlib.Path,
    ood_type: str,
):
    logger = TrialLogger(
        task_name=task_name,
        trial_id=trial_id,
        ood_type=ood_type,
        chunk_root=chunk_root,
    )

    obs = env.reset()
    logger.observe_event_source(obs=obs, info=None)

    done = False
    t = 0
    frames = []
    wrist_frames = []
    action_plan = []
    plan_i = 0
    reset_server = True
    chunk_id = 0
    info = None

    while t < max_steps + wait_steps and not done:
        if t < wait_steps:
            obs, _, done, info = env.step(LIBERO_DUMMY_ACTION)
            logger.advance_steps(1)
            t += 1
            logger.observe_event_source(obs=obs, info=info)
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
            chunk_id += 1
            plan_i = 0
            reset_server = False

        action = action_plan[plan_i]
        plan_i += 1
        obs, _, done, info = env.step(action)
        logger.advance_steps(1)
        t += 1
        logger.observe_event_source(obs=obs, info=info)

    logger.finalize(success=done, final_info=info)
    return done, t, frames, wrist_frames


def rollout_task(
    *,
    bddl_path: pathlib.Path,
    prompt_text: str,
    task_name: str,
    output_dir: pathlib.Path,
    chunk_root: pathlib.Path,
    policy: WebsocketClientPolicy,
    offscreen_env_cls,
    resolution: int,
    resize: int,
    wait_steps: int,
    max_steps: int,
    replan_steps: int,
    trials: int,
    target_successes: int,
    seed: int,
    save_wrist: bool,
    ood_type: str,
) -> Tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running task: {bddl_path}")
    print(f"Prompt text: {prompt_text}")
    print(f"Output dir: {output_dir}")
    print(f"Target successes: {target_successes}")
    print(f"Max trials: {trials}")

    successes = 0
    trials_run = 0

    for trial_id in range(trials):
        env = build_env(
            bddl_path,
            resolution=resolution,
            seed=seed + trial_id,
            offscreen_env_cls=offscreen_env_cls,
        )
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
                chunk_root=chunk_root,
                ood_type=ood_type,
            )
        finally:
            env.close()

        suffix = "success" if done else "failure"
        video_path = output_dir / f"trial{trial_id}_{suffix}.mp4"
        imageio.mimwrite(str(video_path), [np.asarray(x) for x in frames], fps=30)

        if save_wrist:
            wrist_path = output_dir / f"trial{trial_id}_{suffix}_wrist.mp4"
            imageio.mimwrite(str(wrist_path), [np.asarray(x) for x in wrist_frames], fps=30)

        print(f"  trial={trial_id} done={done} steps={t} video={video_path}")
        trials_run += 1
        if done:
            successes += 1
        print(f"  progress: successes={successes}/{target_successes}, trials={trials_run}/{trials}")

        if successes >= target_successes:
            print(f"  reached target successes for {bddl_path.name}; moving to next task")
            break

    return successes, trials_run


def iter_bddl_files(input_dir: pathlib.Path):
    yield from sorted(input_dir.rglob("*.bddl"))


def iter_bddl_files_from_tasks_info(tasks_info: pathlib.Path, input_dir: pathlib.Path):
    for raw_line in tasks_info.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rel_path = pathlib.Path(line)
        if rel_path.parts and rel_path.parts[0] == input_dir.name:
            rel_path = pathlib.Path(*rel_path.parts[1:])
        yield (input_dir / rel_path).resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=pathlib.Path, required=True)
    parser.add_argument("--libero_root", type=pathlib.Path, default=DEFAULT_LIBERO_ROOT)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--target_successes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--wait_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--output_root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_subdir", type=pathlib.Path, default=None)
    parser.add_argument("--tasks_info", type=pathlib.Path, default=None)
    parser.add_argument("--chunk_root", type=pathlib.Path, default=DEFAULT_CHUNK_ROOT)
    parser.add_argument("--save_wrist", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    offscreen_env_cls = import_libero_modules(args.libero_root)
    policy = WebsocketClientPolicy(host=args.host, port=args.port)

    if args.tasks_info is not None:
        tasks_info = args.tasks_info.expanduser().resolve()
        if not tasks_info.exists():
            raise FileNotFoundError(f"tasks_info file does not exist: {tasks_info}")
        bddl_files = list(iter_bddl_files_from_tasks_info(tasks_info, input_dir))
    else:
        bddl_files = list(iter_bddl_files(input_dir))
    if not bddl_files:
        raise FileNotFoundError(f"No .bddl files found under: {input_dir}")

    suite_name = input_dir.name
    print(f"LIBERO source: {args.libero_root.expanduser().resolve()}")
    print(f"Input dir: {input_dir}")
    print(f"Detected suite root: {suite_name}")
    print(f"Found {len(bddl_files)} BDDL files")

    for bddl_path in bddl_files:
        if not bddl_path.exists():
            raise FileNotFoundError(f"BDDL file does not exist: {bddl_path}")
        prompt_text = read_prompt_for_bddl(bddl_path)
        task_name = sanitize_task_name(bddl_path.stem)
        if args.output_subdir is not None:
            output_dir = args.output_root / args.output_subdir / task_name
        else:
            rel_parent = bddl_path.parent.relative_to(input_dir)
            output_dir = args.output_root / suite_name / rel_parent / task_name

        successes, trials_run = rollout_task(
            bddl_path=bddl_path,
            prompt_text=prompt_text,
            task_name=task_name,
            output_dir=output_dir,
            chunk_root=args.chunk_root,
            policy=policy,
            offscreen_env_cls=offscreen_env_cls,
            resolution=args.resolution,
            resize=args.resize,
            wait_steps=args.wait_steps,
            max_steps=args.max_steps,
            replan_steps=args.replan_steps,
            trials=args.trials,
            target_successes=args.target_successes,
            seed=args.seed,
            save_wrist=args.save_wrist,
            ood_type=suite_name,
        )
        print(
            f"Finished task {task_name}: successes={successes}/{args.target_successes}, "
            f"trials_run={trials_run}/{args.trials}"
        )


if __name__ == "__main__":
    main()
