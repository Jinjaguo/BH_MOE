"""
Phase-structure discovery with fused chunk representations.

Purpose
-------
Collect fused normalized-time chunk embeddings from selected success/failure
trials, reduce them with PCA or UMAP, and draw scatter plots colored by
normalized chunk index, success/failure label, and phase label inferred from
the original rollout phase boundaries.

Parameters
----------
--root: chunk-wise task output directory containing trial_* folders.
--out_dir: directory where CSV/PNG outputs are saved.
--n_success: number of successful trials to use.
--n_failure: number of failed trials to use.
--target_length: number of normalized-time chunks after interpolation.
--fields: optional state field names to analyze. Defaults to manifest fields.
--method: pca or umap. UMAP requires the umap-learn package.

Usage
-----
python analysis/chunk_analysis/06_phase_structure_discovery.py \
    --root OOD_exp/dif_start_end_loc/outputs/chunk_wise/put_the_cream_cheese_on_the_plate \
    --out_dir analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate \
    --method pca

Outputs
-------
Saved under --out_dir/phase_structure:
  embedding_points.csv        2D point table with trial/chunk/label/phase
  phase_structure_*.png       scatter plots colored by chunk, label, phase
  phase_structure_metadata.json
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import numpy as np

from chunk_analysis_common import (
    DEFAULT_ROOT,
    discover_trials,
    load_group_normalized_state_tensors,
    load_manifest_fields,
    make_fused_from_states,
    normalized_source_indices,
    normalized_time_grid,
    select_balanced_trials,
    trial_ids,
    write_json,
)


def chunk_phase(record: Dict, chunk_id: int) -> str:
    ranges = record.get("chunk_ranges") or []
    phase_boundaries = record.get("phase_boundaries") or {}
    start_step = None
    for item in ranges:
        if item.get("chunk_id") == chunk_id:
            start_step = item.get("start_step")
            break
    if start_step is None:
        return "unknown"
    for phase in ["approach", "grasp", "move", "place"]:
        bounds = phase_boundaries.get(phase)
        if not bounds or bounds.get("is_empty"):
            continue
        if bounds.get("start_step", -1) <= start_step < bounds.get("end_step_exclusive", -1):
            return phase
    return "unknown"


def chunk_phase_from_source_index(record: Dict, source_index: float) -> tuple[str, int]:
    ranges = record.get("chunk_ranges") or []
    if not ranges:
        return "unknown", int(round(source_index))
    nearest_pos = int(np.clip(round(source_index), 0, len(ranges) - 1))
    chunk_id = int(ranges[nearest_pos].get("chunk_id", nearest_pos))
    return chunk_phase(record, chunk_id), chunk_id


def reduce_2d(x: np.ndarray, method: str, random_state: int) -> np.ndarray:
    if method == "pca":
        x = x.astype(np.float64, copy=False)
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        x_scaled = (x - mean) / std
        _, _, vh = np.linalg.svd(x_scaled, full_matrices=False)
        return x_scaled @ vh[:2].T
    if method == "umap":
        from sklearn.preprocessing import StandardScaler
        import umap

        return umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1).fit_transform(
            StandardScaler().fit_transform(x)
        )
    raise ValueError(f"Unknown method: {method}")


def scatter_plots(out_dir: Path, points: np.ndarray, rows: list[dict], method: str) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_vals = np.asarray([r["chunk_position"] for r in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(points[:, 0], points[:, 1], c=chunk_vals, s=18, cmap="viridis", alpha=0.85)
    ax.set_title(f"{method.upper()} fused chunks colored by chunk index")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.colorbar(sc, ax=ax, label="chunk position")
    fig.tight_layout()
    fig.savefig(out_dir / "phase_structure_by_chunk.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, color in [("success", "tab:blue"), ("failure", "tab:red")]:
        idx = [i for i, r in enumerate(rows) if r["label"] == label]
        ax.scatter(points[idx, 0], points[idx, 1], s=18, alpha=0.8, label=label, color=color)
    ax.set_title(f"{method.upper()} fused chunks colored by outcome")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "phase_structure_by_outcome.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    phases = ["approach", "grasp", "move", "place", "unknown"]
    colors = ["tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:gray"]
    for phase, color in zip(phases, colors):
        idx = [i for i, r in enumerate(rows) if r["phase"] == phase]
        if idx:
            ax.scatter(points[idx, 0], points[idx, 1], s=18, alpha=0.8, label=phase, color=color)
    ax.set_title(f"{method.upper()} fused chunks colored by phase")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "phase_structure_by_phase.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--n_success", type=int, default=20)
    parser.add_argument("--n_failure", type=int, default=20)
    parser.add_argument("--target_length", type=int, default=20)
    parser.add_argument("--fields", nargs="*", default=None)
    parser.add_argument("--method", choices=["pca", "umap"], default="pca")
    parser.add_argument("--random_state", type=int, default=0)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (args.out_dir / "phase_structure").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = args.fields or load_manifest_fields(root)

    trials = discover_trials(root)
    success_trials, failure_trials = select_balanced_trials(trials, args.n_success, args.n_failure)
    selected = success_trials + failure_trials

    success_states, state_names = load_group_normalized_state_tensors(success_trials, args.target_length, fields)
    failure_states, _ = load_group_normalized_state_tensors(failure_trials, args.target_length, fields)
    fused = np.concatenate(
        [
            make_fused_from_states(success_states, state_names),
            make_fused_from_states(failure_states, state_names),
        ],
        axis=0,
    )
    x = fused.reshape(-1, fused.shape[-1])

    rows = []
    ordered_trials = success_trials + failure_trials
    for trial_idx, trial in enumerate(ordered_trials):
        source_indices = normalized_source_indices(trial.num_chunk_files, args.target_length)
        for pos, source_index in enumerate(source_indices):
            phase, chunk_id = chunk_phase_from_source_index(trial.finalize_record, float(source_index))
            rows.append(
                {
                    "trial_id": trial.trial_id,
                    "label": "success" if trial.success else "failure",
                    "chunk_position": pos,
                    "tau": float(pos / (args.target_length - 1)),
                    "source_index": float(source_index),
                    "chunk_id": chunk_id,
                    "phase": phase,
                }
            )

    points = reduce_2d(x, args.method, args.random_state)
    scatter_plots(out_dir, points, rows, args.method)

    with (out_dir / "embedding_points.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "trial_id", "label", "chunk_position", "tau", "source_index", "chunk_id", "phase"])
        for point, row in zip(points, rows):
            writer.writerow([
                point[0],
                point[1],
                row["trial_id"],
                row["label"],
                row["chunk_position"],
                row["tau"],
                row["source_index"],
                row["chunk_id"],
                row["phase"],
            ])

    write_json(
        out_dir / "phase_structure_metadata.json",
        {
            "root": str(root),
            "fields": fields,
            "state_names": state_names,
            "selected_success_trials": trial_ids(success_trials),
            "selected_failure_trials": trial_ids(failure_trials),
            "target_length": args.target_length,
            "normalized_tau": normalized_time_grid(args.target_length).tolist(),
            "original_chunk_counts": {str(t.trial_id): t.num_chunk_files for t in selected},
            "method": args.method,
            "random_state": args.random_state,
        },
    )
    print(f"Wrote phase-structure outputs to {out_dir}")


if __name__ == "__main__":
    main()
