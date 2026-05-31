"""
Intra-trajectory chunk-to-chunk transition analysis.

Purpose
-------
For each selected trial and hidden state, first linearly interpolate the full
trajectory onto tau in [0, 1], then compute adjacent normalized-time changes as
delta[c] = 1 - cosine(h[c], h[c + 1]). The script plots success and failure
mean transition curves for each state and saves the raw mean/std tables.

Parameters
----------
--root: chunk-wise task output directory containing trial_* folders.
--out_dir: directory where CSV/NPY/PNG outputs are saved.
--n_success: number of successful trials to use.
--n_failure: number of failed trials to use.
--target_length: number of normalized-time chunks after interpolation.
--fields: optional state field names to analyze. Defaults to manifest fields.

Usage
-----
python analysis/chunk_analysis/05_transition_analysis.py \
    --root OOD_exp/dif_start_end_loc/outputs/chunk_wise/put_the_cream_cheese_on_the_plate \
    --out_dir analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate

Outputs
-------
Saved under --out_dir/transition:
  transition_delta_mean.csv      success/failure mean delta per transition/state
  transition_delta_std.csv       success/failure std delta per transition/state
  transition_delta_state_*.png   success/failure transition curves
  transition_metadata.json       selected trials and chunk ids
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from chunk_analysis_common import (
    DEFAULT_ROOT,
    cosine_distance,
    discover_trials,
    load_group_normalized_state_tensors,
    load_manifest_fields,
    normalized_time_grid,
    select_balanced_trials,
    trial_ids,
    write_json,
)


def transition_delta(states: dict[str, np.ndarray], state_names: list[str]) -> np.ndarray:
    first = states[state_names[0]]
    n, c_count = first.shape[:2]
    out = np.zeros((n, c_count - 1, len(state_names)), dtype=np.float64)
    for n_idx in range(n):
        for c in range(c_count - 1):
            for s, state in enumerate(state_names):
                z = states[state]
                out[n_idx, c, s] = cosine_distance(z[n_idx, c, :], z[n_idx, c + 1, :])
    return out


def write_transition_csv(path: Path, success_vals: np.ndarray, failure_vals: np.ndarray, state_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["transition_position"]
        for state in state_names:
            header.extend([f"success_{state}", f"failure_{state}"])
        writer.writerow(header)
        for c in range(success_vals.shape[0]):
            row = [c]
            for s in range(success_vals.shape[1]):
                row.extend([success_vals[c, s], failure_vals[c, s]])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--n_success", type=int, default=20)
    parser.add_argument("--n_failure", type=int, default=20)
    parser.add_argument("--target_length", type=int, default=20)
    parser.add_argument("--fields", nargs="*", default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (args.out_dir / "transition").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = args.fields or load_manifest_fields(root)

    trials = discover_trials(root)
    success_trials, failure_trials = select_balanced_trials(trials, args.n_success, args.n_failure, min_chunks=2)
    selected = success_trials + failure_trials
    if args.target_length < 2:
        raise ValueError("Need target_length >= 2 for transition analysis")

    success_states, state_names = load_group_normalized_state_tensors(success_trials, args.target_length, fields)
    failure_states, _ = load_group_normalized_state_tensors(failure_trials, args.target_length, fields)
    delta_success = transition_delta(success_states, state_names)
    delta_failure = transition_delta(failure_states, state_names)
    np.save(out_dir / "transition_delta_success.npy", delta_success)
    np.save(out_dir / "transition_delta_failure.npy", delta_failure)

    mean_success = delta_success.mean(axis=0)
    mean_failure = delta_failure.mean(axis=0)
    std_success = delta_success.std(axis=0)
    std_failure = delta_failure.std(axis=0)
    write_transition_csv(out_dir / "transition_delta_mean.csv", mean_success, mean_failure, state_names)
    write_transition_csv(out_dir / "transition_delta_std.csv", std_success, std_failure, state_names)

    import matplotlib.pyplot as plt

    x = np.arange(args.target_length - 1)
    for s, state in enumerate(state_names):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, mean_success[:, s], label="success", linewidth=2)
        ax.fill_between(x, mean_success[:, s] - std_success[:, s], mean_success[:, s] + std_success[:, s], alpha=0.18)
        ax.plot(x, mean_failure[:, s], label="failure", linewidth=2)
        ax.fill_between(x, mean_failure[:, s] - std_failure[:, s], mean_failure[:, s] + std_failure[:, s], alpha=0.18)
        ax.set_title(f"Chunk-to-chunk delta: {state}")
        ax.set_xlabel("normalized transition position c -> c+1")
        ax.set_ylabel("1 - cosine similarity")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"transition_delta_{state}.png", dpi=180)
        plt.close(fig)

    write_json(
        out_dir / "transition_metadata.json",
        {
            "root": str(root),
            "fields": fields,
            "state_names": state_names,
            "selected_success_trials": trial_ids(success_trials),
            "selected_failure_trials": trial_ids(failure_trials),
            "target_length": args.target_length,
            "normalized_tau": normalized_time_grid(args.target_length).tolist(),
            "original_chunk_counts": {str(t.trial_id): t.num_chunk_files for t in selected},
        },
    )
    print(f"Wrote transition-analysis outputs to {out_dir}")


if __name__ == "__main__":
    main()
