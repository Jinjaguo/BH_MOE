"""
Purpose
-------
Infer four rollout phases from recorded raw events after a rollout finishes or
in an offline analysis pass.

The four phases are:
1. approach
2. grasp
3. move
4. place

Inference prefers raw event timestamps over online hard-coded boundaries and
supports partial / failed rollouts through fallback rules.

Arguments
---------
`infer_phase_boundaries(total_steps, event_summary)`
  `total_steps`:
    Number of executed low-level environment steps in the rollout.
  `event_summary`:
    Output of `PhaseBoundaryRecorder.summary()`.

Examples
--------
from phase_infer_helper import infer_phase_boundaries

phase_info = infer_phase_boundaries(
    total_steps=143,
    event_summary=event_summary,
)

Outputs
-------
This helper returns a JSON-serializable dictionary and does not save files by
itself. The result is intended to be written under:
`OOD_exp/outputs/chunk_wise/<task_name>/trial_<trial_id>/`
"""

from __future__ import annotations

from typing import Any


def _find_first_step(events: list[dict[str, Any]], event_name: str, *, min_step: int = 0) -> int | None:
    for event in events:
        if event.get("event") != event_name:
            continue
        step_idx = int(event["step_idx"])
        if step_idx >= min_step:
            return step_idx
    return None


def _segment(start: int, end: int, *, empty_when_equal: bool = True) -> dict[str, Any]:
    start = int(max(0, start))
    end = int(max(start, end))
    is_empty = start == end if empty_when_equal else False
    return {
        "start_step": start,
        "end_step_exclusive": end,
        "is_empty": is_empty,
    }


def infer_phase_boundaries(total_steps: int, event_summary: dict[str, Any]) -> dict[str, Any]:
    total_steps = int(max(0, total_steps))
    event_log = list(event_summary.get("event_log", []))
    first_events = dict(event_summary.get("first_events", {}))

    near_object = first_events.get("near_object")
    grasp = first_events.get("grasp_established")

    approach_end = (
        int(near_object["step_idx"])
        if near_object is not None
        else int(grasp["step_idx"]) if grasp is not None else total_steps
    )

    grasp_start = approach_end
    grasp_step = int(grasp["step_idx"]) if grasp is not None else None

    if grasp_step is None:
        return {
            "phases": {
                "approach": _segment(0, approach_end),
                "grasp": _segment(grasp_start, total_steps),
                "move": _segment(total_steps, total_steps),
                "place": _segment(total_steps, total_steps),
            },
            "chosen_events": {
                "approach_end": approach_end,
                "grasp_established": None,
                "place_region_entered": None,
                "release_detected": None,
            },
        }

    near_goal_step = _find_first_step(event_log, "place_region_entered", min_step=grasp_step)
    release_after_goal_step = None
    if near_goal_step is not None:
        release_after_goal_step = _find_first_step(event_log, "release_detected", min_step=near_goal_step)

    if near_goal_step is None:
        return {
            "phases": {
                "approach": _segment(0, approach_end),
                "grasp": _segment(grasp_start, grasp_step),
                "move": _segment(grasp_step, total_steps),
                "place": _segment(total_steps, total_steps),
            },
            "chosen_events": {
                "approach_end": approach_end,
                "grasp_established": grasp_step,
                "place_region_entered": None,
                "release_detected": None,
            },
        }

    place_end = release_after_goal_step if release_after_goal_step is not None else total_steps
    return {
        "phases": {
            "approach": _segment(0, approach_end),
            "grasp": _segment(grasp_start, grasp_step),
            "move": _segment(grasp_step, near_goal_step),
            "place": _segment(near_goal_step, place_end),
        },
        "chosen_events": {
            "approach_end": approach_end,
            "grasp_established": grasp_step,
            "place_region_entered": near_goal_step,
            "release_detected": release_after_goal_step,
        },
    }
