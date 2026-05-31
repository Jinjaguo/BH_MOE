"""
Failure time-stretch analysis.

Purpose
-------
Quantify how much each failed trajectory is stretched relative to the success
reference timeline, then test whether larger stretch ratios are associated
with earlier representation drift. Each trajectory is first linearly
interpolated onto tau in [0, 1] with a fixed target length.

Parameters
----------
--root: chunk-wise task output directory containing trial_* folders.
--out_dir: directory where CSV/PNG outputs are saved.
--n_success: number of successful trials used to build the reference mean.
--n_failure: number of failed trials to analyze.
--target_length: number of normalized-time chunks after interpolation.
--reference_success_length: denominator for stretch_ratio. If omitted, the
                            median selected success chunk count is used.
--early_fraction: prefix of normalized time used for early-drift summaries.
--fields: optional state field names to analyze. Defaults to manifest fields.

Usage
-----
python analysis/chunk_analysis/07_time_stretch_analysis.py \
    --root OOD_exp/dif_start_end_loc/outputs/chunk_wise/put_the_cream_cheese_on_the_plate \
    --out_dir analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate \
    --target_length 20

Outputs
-------
Saved under --out_dir/time_stretch:
  stretch_ratios.csv                 per-failure trajectory stretch ratios
  stretch_drift_correlation.csv/png  corr(stretch, drift[c, s]) heatmap
  early_drift_vs_stretch.csv/png     prefix drift summaries and scatter plots
  time_stretch_metadata.json         selected trials and settings
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
    heatmap,
    load_group_normalized_state_tensors,
    load_manifest_fields,
    normalized_time_grid,
    save_matrix_csv,
    select_balanced_trials,
    trial_ids,
    write_json,
)


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    unique_vals, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    del unique_vals
    for group_idx, count in enumerate(counts):
        if count > 1:
            ranks[inverse == group_idx] = ranks[inverse == group_idx].mean()
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata(x), rankdata(y))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--n_success", type=int, default=20)
    parser.add_argument("--n_failure", type=int, default=20)
    parser.add_argument("--target_length", type=int, default=20)
    parser.add_argument("--reference_success_length", type=float, default=None)
    parser.add_argument("--early_fraction", type=float, default=0.25)
    parser.add_argument("--fields", nargs="*", default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (args.out_dir / "time_stretch").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = args.fields or load_manifest_fields(root)

    trials = discover_trials(root)
    success_trials, failure_trials = select_balanced_trials(trials, args.n_success, args.n_failure)
    success_states, state_names = load_group_normalized_state_tensors(success_trials, args.target_length, fields)
    failure_states, _ = load_group_normalized_state_tensors(failure_trials, args.target_length, fields)

    success_lengths = np.asarray([t.num_chunk_files for t in success_trials], dtype=np.float64)
    reference_length = (
        float(args.reference_success_length)
        if args.reference_success_length is not None
        else float(np.median(success_lengths))
    )
    failure_lengths = np.asarray([t.num_chunk_files for t in failure_trials], dtype=np.float64)
    stretch = failure_lengths / reference_length

    with (out_dir / "stretch_ratios.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "failure_length", "reference_success_length", "stretch_ratio"])
        for trial, length, ratio in zip(failure_trials, failure_lengths, stretch):
            writer.writerow([trial.trial_id, int(length), reference_length, ratio])

    c_count, s_count = args.target_length, len(state_names)
    pearson = np.zeros((c_count, s_count), dtype=np.float64)
    spearman = np.zeros((c_count, s_count), dtype=np.float64)
    failure_drift = {}
    for s, state in enumerate(state_names):
        success_mean = success_states[state].mean(axis=0)
        drift = np.zeros((len(failure_trials), c_count), dtype=np.float64)
        for n in range(len(failure_trials)):
            for c in range(c_count):
                drift[n, c] = cosine_distance(failure_states[state][n, c, :], success_mean[c, :])
        failure_drift[state] = drift
        for c in range(c_count):
            pearson[c, s] = pearson_corr(stretch, drift[:, c])
            spearman[c, s] = spearman_corr(stretch, drift[:, c])

    save_matrix_csv(out_dir / "stretch_drift_correlation_pearson.csv", pearson, "chunk_position", state_names)
    save_matrix_csv(out_dir / "stretch_drift_correlation_spearman.csv", spearman, "chunk_position", state_names)
    heatmap(out_dir / "stretch_drift_correlation_pearson.png", pearson, "Pearson corr(stretch, drift)", "state", "normalized time index", state_names)
    heatmap(out_dir / "stretch_drift_correlation_spearman.png", spearman, "Spearman corr(stretch, drift)", "state", "normalized time index", state_names)

    early_count = max(1, int(np.ceil(args.target_length * args.early_fraction)))
    early_by_state = {
        state: failure_drift[state][:, :early_count].mean(axis=1)
        for state in state_names
    }
    with (out_dir / "early_drift_vs_stretch.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "stretch_ratio"] + [f"early_drift_{state}" for state in state_names])
        for n, trial in enumerate(failure_trials):
            writer.writerow([trial.trial_id, stretch[n]] + [early_by_state[state][n] for state in state_names])

    import matplotlib.pyplot as plt

    for state in state_names:
        y = early_by_state[state]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(stretch, y, s=32, alpha=0.85)
        ax.set_title(f"Early drift vs stretch: {state}")
        ax.set_xlabel("stretch_ratio = L_failure / L_success_ref")
        ax.set_ylabel(f"mean drift over first {early_count} normalized chunks")
        corr = pearson_corr(stretch, y)
        ax.text(0.03, 0.95, f"Pearson r={corr:.3f}", transform=ax.transAxes, va="top")
        fig.tight_layout()
        fig.savefig(out_dir / f"early_drift_vs_stretch_{state}.png", dpi=180)
        plt.close(fig)

    write_json(
        out_dir / "time_stretch_metadata.json",
        {
            "root": str(root),
            "fields": fields,
            "state_names": state_names,
            "target_length": args.target_length,
            "normalized_tau": normalized_time_grid(args.target_length).tolist(),
            "reference_success_length": reference_length,
            "early_fraction": args.early_fraction,
            "early_count": early_count,
            "selected_success_trials": trial_ids(success_trials),
            "selected_failure_trials": trial_ids(failure_trials),
            "success_chunk_counts": {str(t.trial_id): t.num_chunk_files for t in success_trials},
            "failure_chunk_counts": {str(t.trial_id): t.num_chunk_files for t in failure_trials},
        },
    )
    print(f"Wrote time-stretch outputs to {out_dir}")


if __name__ == "__main__":
    main()
