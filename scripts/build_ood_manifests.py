"""
Build rollout-level and chunk-level manifests for OOD chunk-wise rollout data.

Purpose
-------
Create two CSV files that connect every latent chunk sample to a stable
`rollout_id`:
1. Rollout manifest: one row per trial / rollout.
2. Chunk manifest: one row per saved chunk `.pt` sample.

Arguments
---------
  --task-dir:
      Directory containing `trial_<id>` folders with chunk `.pt` files and
      rollout metadata jsonl files.
  --task-name:
      LIBERO task name used in stable ids and manifest columns.
  --model:
      Policy/model identifier for manifest columns.
  --dataset:
      Dataset identifier for manifest columns.
  --ood-type:
      OOD split/type identifier for manifest columns.
  --output-dir:
      Directory where the two manifest CSV files are saved.
  --early-chunk-threshold:
      Mark chunks with `chunk_id < threshold` as early chunks.

Usage
-----
python scripts/build_ood_manifests.py \
    --task-dir OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate \
    --task-name put_the_cream_cheese_on_the_plate \
    --model pi05 \
    --dataset libero_ood \
    --ood-type diff_start_end_loc \
    --output-dir OOD_exp/manifests

Outputs
-------
  OOD_exp/manifests/pi05_put_the_cream_cheese_on_the_plate_rollout_manifest.csv
  OOD_exp/manifests/pi05_put_the_cream_cheese_on_the_plate_chunk_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


RECORDED_LATENT_KEYS = [
    "text_encoder_output",
    "cross_attention_action_x_after",
    "action_head_input",
    "chunk_vector_mean",
    "action_chunk_output",
]


ROLLOUT_COLUMNS = [
    "rollout_id",
    "model",
    "dataset",
    "ood_type",
    "task_name",
    "trial_id",
    "success",
    "num_chunks",
    "has_intervention",
    "video_path",
    "latent_dir",
    "event_log_path",
    "finalize_path",
    "rollout_result_path",
    "dominant_failure_mode",
    "valid_for_probe",
    "annotation_status",
    "notes",
]


CHUNK_COLUMNS = [
    "chunk_uid",
    "rollout_id",
    "model",
    "dataset",
    "ood_type",
    "task_name",
    "trial_id",
    "chunk_id",
    "norm_chunk_id",
    "latent_pt_path",
    "success",
    "failure_mode",
    "dominant_failure_mode",
    "valid_for_probe",
    "is_early_chunk",
    "latent_keys",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate"),
    )
    parser.add_argument("--task-name", default="put_the_cream_cheese_on_the_plate")
    parser.add_argument("--model", default="pi05")
    parser.add_argument("--dataset", default="libero_ood")
    parser.add_argument("--ood-type", default="diff_start_end_loc")
    parser.add_argument("--output-dir", type=Path, default=Path("OOD_exp/manifests"))
    parser.add_argument("--early-chunk-threshold", type=int, default=5)
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    return path.as_posix()


def read_first_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def trial_id_from_path(path: Path) -> int:
    match = re.fullmatch(r"trial_(\d+)", path.name)
    if not match:
        raise ValueError(f"Unexpected trial directory name: {path}")
    return int(match.group(1))


def chunk_id_from_path(path: Path) -> int:
    match = re.fullmatch(r"chunk_(\d+)\.pt", path.name)
    if not match:
        raise ValueError(f"Unexpected chunk filename: {path}")
    return int(match.group(1))


def stable_rollout_id(
    *, model: str, dataset: str, ood_type: str, task_name: str, trial_id: int
) -> str:
    return f"{model}_{dataset}_{ood_type}_{task_name}_trial_{trial_id:03d}"


def stable_chunk_uid(rollout_id: str, chunk_id: int) -> str:
    return f"{rollout_id}_chunk_{chunk_id:03d}"


def find_video_path(task_name: str, trial_id: int, success: int) -> str:
    status = "success" if success else "failure"
    candidates = [
        Path("OOD_exp/outputs/videos/libero_goal/dif_start_end_loc")
        / task_name
        / f"trial{trial_id}_{status}.mp4",
        Path("OOD_exp/outputs/videos/libero_goal/diff_start_end_loc")
        / task_name
        / f"trial{trial_id}_{status}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return repo_relative(candidate)
    return repo_relative(candidates[0])


def int_bool(value: Any, default: bool = True) -> int:
    if value is None:
        return int(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null", "nan"}:
            return int(default)
        if text in {"1", "true", "yes", "y"}:
            return 1
        if text in {"0", "false", "no", "n"}:
            return 0
    return int(bool(value))


def default_dominant_failure_mode(result_data: dict[str, Any], success: int) -> str:
    if success:
        return "success"
    value = result_data.get("dominant_failure_mode", "unlabeled")
    return str(value) if value else "unlabeled"


def annotation_note(result_path: Path, success: int, annotation_status: str) -> str:
    if not result_path.exists():
        return "label_auto_success" if success else "failure_label_pending"
    if success:
        return "label_auto_success"
    if annotation_status:
        return "failure_label_verified"
    return "failure_label_pending"


def build_manifests(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_dir = args.task_dir
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory does not exist: {task_dir}")

    rollout_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    trial_dirs = sorted(
        [path for path in task_dir.iterdir() if path.is_dir() and path.name.startswith("trial_")],
        key=trial_id_from_path,
    )

    for trial_dir in trial_dirs:
        trial_id = trial_id_from_path(trial_dir)
        finalize_path = trial_dir / "rollouts_finalize.jsonl"
        event_log_path = trial_dir / "rollouts_state_record.jsonl"
        finalize = read_first_jsonl(finalize_path)
        state_record = read_first_jsonl(event_log_path)
        record = finalize or state_record
        chunk_paths = sorted(trial_dir.glob("chunk_*.pt"), key=chunk_id_from_path)
        num_chunks = int(record.get("num_chunks") or len(chunk_paths))
        result_path = trial_dir / "rollout_result.json"
        result_data = read_json_if_exists(result_path)
        success = int_bool(result_data.get("success"), default=bool(record.get("success", False)))
        rollout_id = stable_rollout_id(
            model=args.model,
            dataset=args.dataset,
            ood_type=args.ood_type,
            task_name=args.task_name,
            trial_id=trial_id,
        )
        dominant_failure_mode = default_dominant_failure_mode(result_data, success)
        valid_for_probe = int_bool(result_data.get("valid_for_probe"), default=True)
        annotation_status = str(result_data.get("annotation_status") or "")
        notes = annotation_note(result_path, success, annotation_status)
        if num_chunks != len(chunk_paths):
            notes = f"{notes};" if notes else ""
            notes += f"num_chunks_metadata={num_chunks},chunk_files={len(chunk_paths)}"

        rollout_rows.append(
            {
                "rollout_id": rollout_id,
                "model": args.model,
                "dataset": args.dataset,
                "ood_type": args.ood_type,
                "task_name": args.task_name,
                "trial_id": trial_id,
                "success": success,
                "num_chunks": len(chunk_paths),
                "has_intervention": 0,
                "video_path": find_video_path(args.task_name, trial_id, success),
                "latent_dir": repo_relative(trial_dir),
                "event_log_path": repo_relative(event_log_path),
                "finalize_path": repo_relative(finalize_path),
                "rollout_result_path": repo_relative(result_path),
                "dominant_failure_mode": dominant_failure_mode,
                "valid_for_probe": valid_for_probe,
                "annotation_status": annotation_status,
                "notes": notes,
            }
        )

        for chunk_path in chunk_paths:
            chunk_id = chunk_id_from_path(chunk_path)
            chunk_rows.append(
                {
                    "chunk_uid": stable_chunk_uid(rollout_id, chunk_id),
                    "rollout_id": rollout_id,
                    "model": args.model,
                    "dataset": args.dataset,
                    "ood_type": args.ood_type,
                    "task_name": args.task_name,
                    "trial_id": trial_id,
                    "chunk_id": chunk_id,
                    "norm_chunk_id": chunk_id,
                    "latent_pt_path": repo_relative(chunk_path),
                    "success": success,
                    "failure_mode": dominant_failure_mode,
                    "dominant_failure_mode": dominant_failure_mode,
                    "valid_for_probe": valid_for_probe,
                    "is_early_chunk": int(chunk_id < args.early_chunk_threshold),
                    "latent_keys": "|".join(RECORDED_LATENT_KEYS),
                }
            )

    return rollout_rows, chunk_rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rollout_rows, chunk_rows = build_manifests(args)
    prefix = f"{args.model}_{args.task_name}"
    rollout_path = args.output_dir / f"{prefix}_rollout_manifest.csv"
    chunk_path = args.output_dir / f"{prefix}_chunk_manifest.csv"
    write_csv(rollout_path, ROLLOUT_COLUMNS, rollout_rows)
    write_csv(chunk_path, CHUNK_COLUMNS, chunk_rows)
    print(f"Wrote {len(rollout_rows)} rollout rows to {rollout_path}")
    print(f"Wrote {len(chunk_rows)} chunk rows to {chunk_path}")


if __name__ == "__main__":
    main()
