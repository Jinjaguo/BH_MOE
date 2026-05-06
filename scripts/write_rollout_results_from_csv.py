#!/usr/bin/env python3
"""
Convert a rollout-level annotation CSV into per-trial rollout_result.json files.

Purpose
-------
Read a human-reviewed rollout label CSV and write one structured
`rollout_result.json` file into each trial directory. These files become the
single source for manifest columns such as `dominant_failure_mode`,
`annotation_status`, and `valid_for_probe`.

Arguments
---------
  --csv:
      Annotation CSV path. Expected columns include `rollout_id`, `trial_id`,
      `success`, `dominant_failure_mode`, `secondary_failure_modes`,
      `failure_stage`, `target_object`, `target_receptacle`, `video_checked`,
      `event_log_checked`, `annotation_status`, and `notes`.
  --task-root:
      Directory containing trial folders such as `trial_0` or `trial_000`.
  --task-name:
      Task name to save in each result JSON.
  --out-name:
      Output filename inside each trial directory. Defaults to
      `rollout_result.json`.

Usage
-----
python scripts/write_rollout_results_from_csv.py \
    --csv OOD_exp/annotations/put_the_cream_cheese_on_the_plate_label_review.csv \
    --task-root OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate \
    --task-name put_the_cream_cheese_on_the_plate \
    --out-name rollout_result.json

Outputs
-------
Writes one file per CSV row under:
  OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate/trial_<id>/rollout_result.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_bool(value: Any) -> bool | None:
    """Parse common CSV boolean encodings into Python bool or None."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    if text in {"", "none", "null", "nan"}:
        return None
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def clean_optional_str(value: Any) -> str | None:
    """Convert empty CSV cells into None and strip surrounding quotes/spaces."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    text = text.strip('"').strip("“").strip("”").strip()
    if text.lower() in {"none", "null", "nan"}:
        return None
    return text


def parse_secondary_modes(value: Any) -> list[str]:
    """
    Parse secondary failure modes from a CSV cell.

    Supported examples:
        ""
        "template_oscillation"
        "template_oscillation|place_miss"
        "template_oscillation,place_miss"
    """
    text = clean_optional_str(value)
    if text is None:
        return []
    if "|" in text:
        parts = text.split("|")
    else:
        parts = text.split(",")
    return [part.strip() for part in parts if part.strip()]


def find_trial_dir(task_root: Path, trial_id: int) -> Path:
    """
    Find the real trial directory.

    This supports common folder names:
        trial_0
        trial_000
        trial-0
        trial-000
    """
    candidates = [
        task_root / f"trial_{trial_id}",
        task_root / f"trial_{trial_id:03d}",
        task_root / f"trial-{trial_id}",
        task_root / f"trial-{trial_id:03d}",
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    raise FileNotFoundError(
        f"Could not find trial directory for trial_id={trial_id} under {task_root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--out-name", type=str, default="rollout_result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_written = 0

    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            trial_id = int(row["trial_id"])
            trial_dir = find_trial_dir(args.task_root, trial_id)

            result = {
                "rollout_id": clean_optional_str(row.get("rollout_id")),
                "task_name": args.task_name,
                "trial_id": trial_id,
                "success": parse_bool(row.get("success")),
                "bddl_success": parse_bool(row.get("bddl_success")),
                "manual_success": parse_bool(row.get("manual_success")),
                "dominant_failure_mode": clean_optional_str(
                    row.get("dominant_failure_mode")
                ),
                "secondary_failure_modes": parse_secondary_modes(
                    row.get("secondary_failure_modes")
                ),
                "failure_stage": clean_optional_str(row.get("failure_stage")),
                "target_object": clean_optional_str(row.get("target_object")),
                "target_receptacle": clean_optional_str(row.get("target_receptacle")),
                "wrong_object_name": clean_optional_str(row.get("wrong_object_name")),
                "wrong_receptacle_name": clean_optional_str(
                    row.get("wrong_receptacle_name")
                ),
                "video_checked": parse_bool(row.get("video_checked")),
                "event_log_checked": parse_bool(row.get("event_log_checked")),
                "annotation_status": clean_optional_str(row.get("annotation_status")),
                "valid_for_probe": parse_bool(row.get("valid_for_probe")),
                "notes": clean_optional_str(row.get("notes")),
            }

            if result["bddl_success"] is None:
                result["bddl_success"] = result["success"]
            if result["manual_success"] is None:
                result["manual_success"] = result["success"]
            if result["valid_for_probe"] is None:
                result["valid_for_probe"] = True

            out_path = trial_dir / args.out_name
            with out_path.open("w", encoding="utf-8") as out_handle:
                json.dump(result, out_handle, indent=2, ensure_ascii=False)
                out_handle.write("\n")

            num_written += 1

    print(f"Wrote {num_written} result files under {args.task_root}")


if __name__ == "__main__":
    main()
