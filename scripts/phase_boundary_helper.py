"""
Purpose
-------
Record raw event timestamps and supporting diagnostics for pick-and-place style
rollouts without hard-coding phase boundaries during control.

This helper focuses on more primitive and reliable information than phase labels.
It records first-occurrence events and all detected event occurrences for later
phase inference:
1. near_object
2. grasp_established
3. object_following
4. place_region_entered
5. release_detected

Arguments
---------
`PhaseBoundaryRecorder(...)`
  `near_object_dist_thresh`:
    Distance threshold for entering the object neighborhood.
  `near_goal_dist_thresh`:
    Distance threshold for entering the goal / receptacle neighborhood.
  `follow_dist_thresh`:
    Maximum end-effector-to-object distance for considering the object as
    following the gripper.
  `follow_rel_change_thresh`:
    Maximum change in relative pose used for follow consistency.
  `motion_thresh`:
    Minimum motion to count as meaningful movement.
  `sustain_steps`:
    Number of consecutive steps required for sustained conditions like following
    and release.

Examples
--------
from phase_boundary_helper import PhaseBoundaryRecorder

recorder = PhaseBoundaryRecorder()
recorder.observe(step_idx=0, obs=obs, info=info)
summary = recorder.summary()

Outputs
-------
This helper does not save files by itself.
Its JSON-serializable summary is intended to be written by
`rollouts_state_record_helper.py` into:
`OOD_exp/outputs/chunk_wise/<task_name>/trial_<trial_id>/`
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


def _as_float_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return arr


def _to_scalar(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    return float(arr[0])


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"true", "1", "yes", "y"}:
            return True
        if norm in {"false", "0", "no", "n"}:
            return False
    return None


def _first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


@dataclasses.dataclass
class _CandidateDistances:
    name: str | None
    distance: float | None


class PhaseBoundaryRecorder:
    def __init__(
        self,
        *,
        near_object_dist_thresh: float = 0.08,
        near_goal_dist_thresh: float = 0.10,
        follow_dist_thresh: float = 0.08,
        follow_rel_change_thresh: float = 0.03,
        motion_thresh: float = 0.005,
        sustain_steps: int = 3,
    ):
        self._near_object_dist_thresh = near_object_dist_thresh
        self._near_goal_dist_thresh = near_goal_dist_thresh
        self._follow_dist_thresh = follow_dist_thresh
        self._follow_rel_change_thresh = follow_rel_change_thresh
        self._motion_thresh = motion_thresh
        self._sustain_steps = sustain_steps

        self._first_events: dict[str, dict[str, Any] | None] = {
            "near_object": None,
            "grasp_established": None,
            "object_following": None,
            "place_region_entered": None,
            "release_detected": None,
        }
        self._event_log: list[dict[str, Any]] = []
        self._debug_steps: list[dict[str, Any]] = []

        self._prev_eef_pos: np.ndarray | None = None
        self._prev_gripper_open: float | None = None
        self._prev_object_positions: dict[str, np.ndarray] = {}
        self._tracked_object_name: str | None = None
        self._follow_streak = 0
        self._release_streak = 0

    def observe(self, *, step_idx: int, obs: dict[str, Any], info: dict[str, Any] | None = None) -> None:
        info = info or {}
        eef_pos = _as_float_array(obs.get("robot0_eef_pos"))
        gripper_open = _to_scalar(obs.get("robot0_gripper_qpos"))
        object_positions = self._extract_object_positions(obs, info)
        goal_positions = self._extract_goal_positions(obs, info)
        nearest_object = self._nearest_distance(eef_pos, object_positions)
        nearest_goal = self._nearest_distance(eef_pos, goal_positions)

        explicit_grasp = self._extract_explicit_grasp(info)
        explicit_release = self._extract_explicit_release(info)
        explicit_place = self._extract_explicit_place(info)

        follow_candidate = self._detect_object_following(
            eef_pos=eef_pos,
            object_positions=object_positions,
        )

        if nearest_object.distance is not None and nearest_object.distance <= self._near_object_dist_thresh:
            self._record_event("near_object", step_idx, "distance", {"object_name": nearest_object.name})

        if explicit_grasp is True or follow_candidate:
            source = "explicit" if explicit_grasp is True else "heuristic_follow"
            self._record_event("grasp_established", step_idx, source, {"object_name": self._tracked_object_name})

        if follow_candidate:
            self._record_event("object_following", step_idx, "heuristic_follow", {"object_name": self._tracked_object_name})

        if explicit_place is True or (
            nearest_goal.distance is not None and nearest_goal.distance <= self._near_goal_dist_thresh
        ):
            source = "explicit" if explicit_place is True else "distance"
            self._record_event("place_region_entered", step_idx, source, {"goal_name": nearest_goal.name})

        if self._first_events["grasp_established"] is not None:
            if explicit_release is True or self._detect_release(follow_candidate, gripper_open):
                source = "explicit" if explicit_release is True else "heuristic_release"
                self._record_event("release_detected", step_idx, source, {"object_name": self._tracked_object_name})

        self._debug_steps.append(
            {
                "step_idx": int(step_idx),
                "nearest_object_name": nearest_object.name,
                "nearest_object_distance": nearest_object.distance,
                "nearest_goal_name": nearest_goal.name,
                "nearest_goal_distance": nearest_goal.distance,
                "tracked_object_name": self._tracked_object_name,
                "follow_streak": int(self._follow_streak),
                "release_streak": int(self._release_streak),
                "gripper_open": gripper_open,
                "explicit_grasp": explicit_grasp,
                "explicit_place": explicit_place,
                "explicit_release": explicit_release,
            }
        )

        self._prev_eef_pos = None if eef_pos is None else eef_pos.copy()
        self._prev_gripper_open = gripper_open
        self._prev_object_positions = {key: value.copy() for key, value in object_positions.items()}

    def summary(self) -> dict[str, Any]:
        return {
            "first_events": self._first_events,
            "event_log": self._event_log,
            "debug_last": self._debug_steps[-1] if self._debug_steps else None,
            "debug_steps_sample": self._debug_steps[-20:],
        }

    def _extract_object_positions(self, obs: dict[str, Any], info: dict[str, Any]) -> dict[str, np.ndarray]:
        candidates: dict[str, np.ndarray] = {}
        for source in (obs, info):
            for key, value in source.items():
                if not key.endswith("_pos"):
                    continue
                if key.startswith("robot0_"):
                    continue
                low = key.lower()
                if any(token in low for token in ["goal", "target", "region", "receptacle"]):
                    continue
                arr = _as_float_array(value)
                if arr is not None and arr.size >= 3:
                    candidates[key] = arr[:3]
        return candidates

    def _extract_goal_positions(self, obs: dict[str, Any], info: dict[str, Any]) -> dict[str, np.ndarray]:
        candidates: dict[str, np.ndarray] = {}
        for source in (obs, info):
            for key, value in source.items():
                if not key.endswith("_pos"):
                    continue
                low = key.lower()
                if not any(token in low for token in ["goal", "target", "region", "receptacle"]):
                    continue
                arr = _as_float_array(value)
                if arr is not None and arr.size >= 3:
                    candidates[key] = arr[:3]
        return candidates

    def _nearest_distance(self, eef_pos: np.ndarray | None, candidates: dict[str, np.ndarray]) -> _CandidateDistances:
        if eef_pos is None or not candidates:
            return _CandidateDistances(name=None, distance=None)
        best_name = None
        best_distance = None
        for name, position in candidates.items():
            distance = float(np.linalg.norm(position[:3] - eef_pos[:3]))
            if best_distance is None or distance < best_distance:
                best_name = name
                best_distance = distance
        return _CandidateDistances(name=best_name, distance=best_distance)

    def _extract_explicit_grasp(self, info: dict[str, Any]) -> bool | None:
        return _to_bool(
            _first_present(
                info,
                [
                    "grasp_established",
                    "grasped",
                    "is_grasped",
                    "object_grasped",
                    "pick_success",
                ],
            )
        )

    def _extract_explicit_place(self, info: dict[str, Any]) -> bool | None:
        return _to_bool(
            _first_present(
                info,
                [
                    "place_region_entered",
                    "near_goal",
                    "inside_goal_region",
                    "inside_target_region",
                    "in_receptacle",
                ],
            )
        )

    def _extract_explicit_release(self, info: dict[str, Any]) -> bool | None:
        return _to_bool(
            _first_present(
                info,
                [
                    "release_detected",
                    "released",
                    "object_released",
                    "is_released",
                ],
            )
        )

    def _detect_object_following(self, *, eef_pos: np.ndarray | None, object_positions: dict[str, np.ndarray]) -> bool:
        if eef_pos is None or self._prev_eef_pos is None or not object_positions:
            self._follow_streak = 0
            return False

        best_name = None
        best_score = None
        for name, curr_obj_pos in object_positions.items():
            prev_obj_pos = self._prev_object_positions.get(name)
            if prev_obj_pos is None:
                continue
            rel_curr = curr_obj_pos[:3] - eef_pos[:3]
            rel_prev = prev_obj_pos[:3] - self._prev_eef_pos[:3]
            rel_change = float(np.linalg.norm(rel_curr - rel_prev))
            grip_dist = float(np.linalg.norm(rel_curr))
            obj_motion = float(np.linalg.norm(curr_obj_pos[:3] - prev_obj_pos[:3]))
            eef_motion = float(np.linalg.norm(eef_pos[:3] - self._prev_eef_pos[:3]))
            is_following = (
                grip_dist <= self._follow_dist_thresh
                and rel_change <= self._follow_rel_change_thresh
                and obj_motion >= self._motion_thresh
                and eef_motion >= self._motion_thresh
            )
            if not is_following:
                continue
            score = grip_dist + rel_change
            if best_score is None or score < best_score:
                best_score = score
                best_name = name

        if best_name is None:
            self._follow_streak = 0
            return False

        self._tracked_object_name = best_name
        self._follow_streak += 1
        return self._follow_streak >= self._sustain_steps

    def _detect_release(self, following_now: bool, gripper_open: float | None) -> bool:
        reopened = False
        if gripper_open is not None and self._prev_gripper_open is not None:
            reopened = gripper_open > self._prev_gripper_open + 0.002

        if not following_now and (self._tracked_object_name is not None or reopened):
            self._release_streak += 1
        else:
            self._release_streak = 0

        return self._release_streak >= self._sustain_steps

    def _record_event(self, event_name: str, step_idx: int, source: str, payload: dict[str, Any]) -> None:
        event = {
            "event": event_name,
            "step_idx": int(step_idx),
            "source": source,
            **payload,
        }
        self._event_log.append(event)
        if self._first_events[event_name] is None:
            self._first_events[event_name] = event
