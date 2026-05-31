"""
Shared utilities for chunk-wise hidden-state analysis.

Purpose
-------
This module is imported by the standalone analysis scripts in this folder. It
discovers trials, reads success/failure labels from jsonl files, loads chunk
hidden states from torch `.pt` files, converts each state to a fixed vector,
and provides small numeric helpers.

Parameters
----------
This file is not intended to be called directly. See the sibling scripts for
their command-line arguments.

Usage
-----
Import from a script, for example:
    from chunk_analysis_common import discover_trials, load_trial_vectors

Outputs
-------
This module does not write files on its own. Output paths are controlled by the
calling script.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def find_repo_root(start: Path | None = None) -> Path:
    """Find the BH_MOE repository root regardless of where this script lives."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for path in [current, *current.parents]:
        if (path / "OOD_exp").is_dir() and ((path / "analysis").is_dir() or (path / "custom_bddl").is_dir()):
            return path
    raise RuntimeError(f"Could not find repository root from {current}")


REPO_ROOT = find_repo_root()

DEFAULT_ROOT = (
    REPO_ROOT
    / "OOD_exp"
    / "dif_start_end_loc"
    / "outputs"
    / "chunk_wise"
    / "put_the_cream_cheese_on_the_plate"
)

DEFAULT_FIELDS = [
    "text_encoder_output",
    "cross_attention_action_x_after",
    "action_head_input",
    "chunk_vector_mean",
    "action_chunk_output",
]

CHUNK_RE = re.compile(r"chunk_(\d+)\.pt$")


@dataclass(frozen=True)
class TrialInfo:
    trial_id: int
    path: Path
    success: bool
    num_chunks_json: Optional[int]
    chunk_ids: Tuple[int, ...]
    finalize_record: Dict[str, Any]

    @property
    def num_chunk_files(self) -> int:
        return len(self.chunk_ids)


def read_first_jsonl(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def load_manifest_fields(root: Path) -> List[str]:
    manifest = root / "manifest.json"
    if not manifest.exists():
        return list(DEFAULT_FIELDS)
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return list(DEFAULT_FIELDS)
    fields = data.get("fields")
    if isinstance(fields, list) and fields:
        return [str(x) for x in fields]
    return list(DEFAULT_FIELDS)


def discover_trials(root: Path) -> List[TrialInfo]:
    trials: List[TrialInfo] = []
    for trial_dir in sorted(root.glob("trial_*"), key=trial_sort_key):
        if not trial_dir.is_dir():
            continue
        finalize = trial_dir / "rollouts_finalize.jsonl"
        state_record = trial_dir / "rollouts_state_record.jsonl"
        record_path = finalize if finalize.exists() else state_record
        if not record_path.exists():
            continue
        record = read_first_jsonl(record_path)
        if "success" not in record:
            continue
        chunk_ids = []
        for p in trial_dir.glob("chunk_*.pt"):
            m = CHUNK_RE.match(p.name)
            if m:
                chunk_ids.append(int(m.group(1)))
        trial_id = parse_trial_id(trial_dir)
        trials.append(
            TrialInfo(
                trial_id=trial_id,
                path=trial_dir,
                success=bool(record["success"]),
                num_chunks_json=record.get("num_chunks"),
                chunk_ids=tuple(sorted(chunk_ids)),
                finalize_record=record,
            )
        )
    return sorted(trials, key=lambda t: t.trial_id)


def parse_trial_id(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except Exception:
        return -1


def trial_sort_key(path: Path) -> Tuple[int, str]:
    return (parse_trial_id(path), path.name)


def select_balanced_trials(
    trials: Sequence[TrialInfo],
    n_success: int,
    n_failure: int,
    min_chunks: int = 1,
) -> Tuple[List[TrialInfo], List[TrialInfo]]:
    successes = [t for t in trials if t.success and t.num_chunk_files >= min_chunks]
    failures = [t for t in trials if (not t.success) and t.num_chunk_files >= min_chunks]
    if len(successes) < n_success:
        raise ValueError(f"Need {n_success} successes, found {len(successes)}")
    if len(failures) < n_failure:
        raise ValueError(f"Need {n_failure} failures, found {len(failures)}")
    return successes[:n_success], failures[:n_failure]


def common_chunk_ids(trials: Sequence[TrialInfo]) -> List[int]:
    if not trials:
        return []
    common = set(trials[0].chunk_ids)
    for trial in trials[1:]:
        common &= set(trial.chunk_ids)
    return sorted(common)


def load_torch_object(path: Path) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required to load chunk .pt files. Run this script in the "
            "same environment that produced the hidden-state outputs."
        ) from exc
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj).__name__}")
    return obj


def tensor_to_numpy(value: Any) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for tensor conversion") from exc
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def state_to_vector(value: Any) -> np.ndarray:
    arr = tensor_to_numpy(value).astype(np.float64, copy=False)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1, arr.shape[-1]).mean(axis=0)


def load_chunk_vectors(chunk_path: Path, fields: Sequence[str]) -> Tuple[List[np.ndarray], List[str]]:
    obj = load_torch_object(chunk_path)
    vectors: List[np.ndarray] = []
    names: List[str] = []
    for field in fields:
        if field not in obj:
            continue
        vectors.append(state_to_vector(obj[field]))
        names.append(field)
    if not vectors:
        available = ", ".join(sorted(obj.keys()))
        raise KeyError(f"No requested fields found in {chunk_path}; available: {available}")
    return vectors, names


def load_trial_vectors(
    trial: TrialInfo,
    chunk_ids: Sequence[int],
    fields: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    chunks: List[np.ndarray] = []
    state_names: Optional[List[str]] = None
    expected_dims: Optional[Tuple[int, ...]] = None
    for chunk_id in chunk_ids:
        chunk_path = trial.path / f"chunk_{chunk_id:05d}.pt"
        vectors, names = load_chunk_vectors(chunk_path, fields)
        dims = tuple(v.shape[0] for v in vectors)
        if expected_dims is None:
            expected_dims = dims
            state_names = names
        elif dims != expected_dims:
            raise ValueError(
                f"State dims changed in trial {trial.trial_id}, chunk {chunk_id}: "
                f"{dims} vs expected {expected_dims}"
            )
        chunks.append(np.stack(vectors, axis=0))
    if state_names is None:
        state_names = list(fields)
    return np.stack(chunks, axis=0), state_names


def load_trial_state_vectors(
    trial: TrialInfo,
    chunk_ids: Sequence[int],
    fields: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    per_state: Dict[str, List[np.ndarray]] = {}
    state_names: Optional[List[str]] = None
    expected_dims: Dict[str, int] = {}
    for chunk_id in chunk_ids:
        chunk_path = trial.path / f"chunk_{chunk_id:05d}.pt"
        vectors, names = load_chunk_vectors(chunk_path, fields)
        if state_names is None:
            state_names = names
        elif names != state_names:
            raise ValueError(f"State field mismatch in trial {trial.trial_id}, chunk {chunk_id}: {names} vs {state_names}")
        for name, vec in zip(names, vectors):
            dim = int(vec.shape[0])
            if name not in expected_dims:
                expected_dims[name] = dim
            elif expected_dims[name] != dim:
                raise ValueError(
                    f"State dim changed for {name} in trial {trial.trial_id}, chunk {chunk_id}: "
                    f"{dim} vs expected {expected_dims[name]}"
                )
            per_state.setdefault(name, []).append(vec)
    if state_names is None:
        state_names = list(fields)
    return {name: np.stack(vals, axis=0) for name, vals in per_state.items()}, state_names


def load_group_state_tensors(
    trials: Sequence[TrialInfo],
    chunk_ids: Sequence[int],
    fields: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    grouped: Dict[str, List[np.ndarray]] = {}
    state_names: Optional[List[str]] = None
    for trial in trials:
        per_state, names = load_trial_state_vectors(trial, chunk_ids, fields)
        if state_names is None:
            state_names = names
        elif names != state_names:
            raise ValueError(f"State field mismatch in trial {trial.trial_id}: {names} vs {state_names}")
        for name in names:
            grouped.setdefault(name, []).append(per_state[name])
    return {name: np.stack(vals, axis=0) for name, vals in grouped.items()}, state_names or list(fields)


def normalized_time_grid(target_length: int) -> np.ndarray:
    if target_length < 2:
        raise ValueError("target_length must be at least 2")
    return np.linspace(0.0, 1.0, target_length, dtype=np.float64)


def linear_interpolate_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float64)
    if sequence.ndim != 2:
        raise ValueError(f"Expected sequence with shape (L, D), got {sequence.shape}")
    length = sequence.shape[0]
    if length < 1:
        raise ValueError("Cannot interpolate an empty sequence")
    if length == target_length:
        return sequence.copy()
    if length == 1:
        return np.repeat(sequence, target_length, axis=0)

    source_index = normalized_time_grid(target_length) * (length - 1)
    left = np.floor(source_index).astype(np.int64)
    right = np.minimum(left + 1, length - 1)
    weight = (source_index - left).reshape(-1, 1)
    return (1.0 - weight) * sequence[left] + weight * sequence[right]


def normalized_source_indices(original_length: int, target_length: int) -> np.ndarray:
    if original_length < 1:
        raise ValueError("original_length must be at least 1")
    return normalized_time_grid(target_length) * (original_length - 1)


def load_trial_normalized_state_vectors(
    trial: TrialInfo,
    target_length: int,
    fields: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    per_state, state_names = load_trial_state_vectors(trial, trial.chunk_ids, fields)
    return {
        name: linear_interpolate_sequence(per_state[name], target_length)
        for name in state_names
    }, state_names


def load_group_normalized_state_tensors(
    trials: Sequence[TrialInfo],
    target_length: int,
    fields: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    grouped: Dict[str, List[np.ndarray]] = {}
    state_names: Optional[List[str]] = None
    for trial in trials:
        per_state, names = load_trial_normalized_state_vectors(trial, target_length, fields)
        if state_names is None:
            state_names = names
        elif names != state_names:
            raise ValueError(f"State field mismatch in trial {trial.trial_id}: {names} vs {state_names}")
        for name in names:
            grouped.setdefault(name, []).append(per_state[name])
    return {name: np.stack(vals, axis=0) for name, vals in grouped.items()}, state_names or list(fields)


def load_group_tensor(
    trials: Sequence[TrialInfo],
    chunk_ids: Sequence[int],
    fields: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    arrays = []
    state_names: Optional[List[str]] = None
    for trial in trials:
        arr, names = load_trial_vectors(trial, chunk_ids, fields)
        if state_names is None:
            state_names = names
        elif names != state_names:
            raise ValueError(f"State field mismatch in trial {trial.trial_id}: {names} vs {state_names}")
        arrays.append(arr)
    return np.stack(arrays, axis=0), state_names or list(fields)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
    return float(1.0 - np.dot(a, b) / denom)


def make_fused(z: np.ndarray) -> np.ndarray:
    zn = l2_normalize(z, axis=-1)
    return zn.reshape(*zn.shape[:-2], zn.shape[-2] * zn.shape[-1])


def make_fused_from_states(states: Dict[str, np.ndarray], state_names: Sequence[str]) -> np.ndarray:
    normalized = [l2_normalize(states[name], axis=-1) for name in state_names]
    return np.concatenate(normalized, axis=-1)


def minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=np.float64)
    if not finite.any():
        return out
    lo = float(np.nanmin(x[finite]))
    hi = float(np.nanmax(x[finite]))
    if math.isclose(lo, hi):
        return out
    out[finite] = (x[finite] - lo) / (hi - lo)
    out[~finite] = np.nan
    return out


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def save_matrix_csv(path: Path, matrix: np.ndarray, row_name: str, col_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join([row_name] + list(col_names))
    lines = [header]
    for i, row in enumerate(matrix):
        vals = ",".join(f"{float(v):.10g}" if np.isfinite(v) else "nan" for v in row)
        lines.append(f"{i},{vals}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def heatmap(path: Path, matrix: np.ndarray, title: str, xlabel: str, ylabel: str, xticklabels: Sequence[str]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(7.0, 0.9 * len(xticklabels))
    fig_h = max(5.0, 0.18 * matrix.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=35, ha="right")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def trial_ids(trials: Sequence[TrialInfo]) -> List[int]:
    return [t.trial_id for t in trials]
