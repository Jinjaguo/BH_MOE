"""
Purpose
-------
Record rollout-side trial metadata, chunk coverage, raw event summaries, and
finalized phase boundaries for chunk-wise server traces.

This helper provides a single `TrialLogger` interface for rollout code:
1. Build websocket payloads that always include `task_name`, `trial_id`,
   and `chunk_id`.
2. Track rollout state such as `ood_type`, current low-level step count, chunk
   coverage ranges, success/failure, and episode length.
3. Finalize the rollout by inferring phases from raw events and writing JSONL
   records for later offline analysis.

Arguments
---------
`TrialLogger(task_name, trial_id, ood_type, chunk_root)`
  `task_name`:
    Stable task identifier used in the output path.
  `trial_id`:
    Rollout index inside the task.
  `ood_type`:
    Benchmark suite or OOD category associated with the rollout.
  `chunk_root`:
    Root directory, usually `OOD_exp/outputs/chunk_wise`.

Examples
--------
from rollouts_state_record_helper import TrialLogger

logger = TrialLogger(
    task_name="put_the_bowl_on_the_rack",
    trial_id=3,
    ood_type="libero_goal",
    chunk_root="OOD_exp/outputs/chunk_wise",
)

req = logger.build_policy_payload(base_payload=req)
logger.register_chunk(chunk_horizon=5)
logger.advance_steps(1)
logger.observe_event_source(obs=obs, info=info)
logger.finalize(success=True)

Outputs
-------
Per-trial files are written to:
`OOD_exp/outputs/chunk_wise/<task_name>/trial_<trial_id>/`

The helper appends one JSON object per trial to:
1. `rollouts_state_record.jsonl`
2. `rollouts_finalize.jsonl`
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from scripts.phase_boundary_helper import PhaseBoundaryRecorder
from scripts.phase_infer_helper import infer_phase_boundaries


def sanitize_task_name(task_name: str) -> str:
    safe = str(task_name).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    return safe or "default_task"


class TrialLogger:
    def __init__(
        self,
        *,
        task_name: str,
        trial_id: int,
        ood_type: str,
        chunk_root: str | pathlib.Path,
    ):
        self.task_name = sanitize_task_name(task_name)
        self.trial_id = int(trial_id)
        self.ood_type = str(ood_type)
        self.chunk_root = pathlib.Path(chunk_root).expanduser().resolve()
        self.task_dir = self.chunk_root / self.task_name
        self.trial_dir = self.task_dir / f"trial_{self.trial_id}"
        self.trial_dir.mkdir(parents=True, exist_ok=True)

        self._event_recorder = PhaseBoundaryRecorder()
        self._chunk_id_counter = 0
        self._step_counter = 0
        self._chunk_ranges: list[dict[str, Any]] = []
        self._debug_messages: list[dict[str, Any]] = []
        self._success: bool | None = None

    def build_policy_payload(self, *, base_payload: dict[str, Any]) -> dict[str, Any]:
        request = dict(base_payload)
        request["task_name"] = self.task_name
        request["trial_id"] = self.trial_id
        request["chunk_id"] = self._chunk_id_counter
        return request

    def register_chunk(self, *, chunk_horizon: int) -> None:
        chunk_horizon = int(max(0, chunk_horizon))
        self._chunk_ranges.append(
            {
                "chunk_id": self._chunk_id_counter,
                "start_step": self._step_counter,
                "end_step_exclusive": self._step_counter + chunk_horizon,
            }
        )
        self._chunk_id_counter += 1

    def advance_steps(self, num_steps: int = 1) -> None:
        self._step_counter += int(num_steps)

    def observe_event_source(self, *, obs: dict[str, Any], info: dict[str, Any] | None = None) -> None:
        self._event_recorder.observe(step_idx=self._step_counter, obs=obs, info=info)

    def add_debug_message(self, message: str, **payload: Any) -> None:
        self._debug_messages.append({"message": message, **payload})

    def finalize(self, *, success: bool, final_info: dict[str, Any] | None = None) -> dict[str, Any]:
        self._success = bool(success)
        total_steps = self._step_counter
        event_summary = self._event_recorder.summary()
        phase_info = infer_phase_boundaries(total_steps=total_steps, event_summary=event_summary)

        state_record = {
            "task_name": self.task_name,
            "trial_id": self.trial_id,
            "ood_type": self.ood_type,
            "success": self._success,
            "episode_length": total_steps,
            "num_chunks": len(self._chunk_ranges),
            "chunk_ranges": self._chunk_ranges,
            "phase_boundaries": phase_info["phases"],
        }

        finalize_record = {
            "task_name": self.task_name,
            "trial_id": self.trial_id,
            "ood_type": self.ood_type,
            "success": self._success,
            "total_steps": total_steps,
            "episode_length": total_steps,
            "num_chunks": len(self._chunk_ranges),
            "chunk_ranges": self._chunk_ranges,
            "phase_boundaries": phase_info["phases"],
            "phase_events": event_summary,
            "chosen_events": phase_info["chosen_events"],
            "final_info": final_info or {},
            "debug_messages": self._debug_messages,
        }

        self._append_jsonl(self.trial_dir / "rollouts_state_record.jsonl", state_record)
        self._append_jsonl(self.trial_dir / "rollouts_finalize.jsonl", finalize_record)
        return finalize_record

    def _append_jsonl(self, path: pathlib.Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")


TrialStateRecorder = TrialLogger
