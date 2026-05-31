"""
Chunk-wise hidden-state data health check.

Purpose
-------
Inspect success and failure trials before any chunk-wise alignment. The script
selects up to 20 successful and 20 failed trials, reports each trajectory's
chunk count, verifies state shapes, checks NaN/Inf values, all-zero tensors,
and summarizes L2 norm distributions for each hidden state.

Parameters
----------
--root: chunk-wise task output directory containing trial_* folders.
--out_dir: directory where tables, JSON summaries, and plots are saved.
--n_success: number of successful trials to inspect.
--n_failure: number of failed trials to inspect.
--fields: optional state field names to analyze. Defaults to manifest fields.
--norm_outlier_z: robust z-score threshold used to flag abnormal norms.

Usage
-----
python analysis/chunk_analysis/01_data_health_check.py \
    --root OOD_exp/dif_start_end_loc/outputs/chunk_wise/put_the_cream_cheese_on_the_plate \
    --out_dir analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate

Outputs
-------
Saved under --out_dir/data_health:
  trial_summary.csv              per-trial labels and chunk counts
  state_health_summary.csv       per-state shape/nan/zero/norm summaries
  norm_distribution_state_*.png  success/failure L2 norm histograms
  health_metadata.json           selected trials and common chunk ids
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from chunk_analysis_common import (
    DEFAULT_ROOT,
    common_chunk_ids,
    discover_trials,
    load_chunk_vectors,
    load_manifest_fields,
    select_balanced_trials,
    trial_ids,
    write_json,
)


def robust_z(values: np.ndarray) -> np.ndarray:
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    if not np.isfinite(mad) or mad <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return 0.6745 * (values - med) / mad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--n_success", type=int, default=20)
    parser.add_argument("--n_failure", type=int, default=20)
    parser.add_argument("--fields", nargs="*", default=None)
    parser.add_argument("--norm_outlier_z", type=float, default=5.0)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (args.out_dir / "data_health").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = args.fields or load_manifest_fields(root)

    trials = discover_trials(root)
    success_trials, failure_trials = select_balanced_trials(trials, args.n_success, args.n_failure)
    selected = success_trials + failure_trials
    common_ids = common_chunk_ids(selected)

    with (out_dir / "trial_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "success", "num_chunks_json", "num_chunk_files", "first_chunk", "last_chunk"])
        for t in selected:
            writer.writerow([
                t.trial_id,
                int(t.success),
                t.num_chunks_json,
                t.num_chunk_files,
                min(t.chunk_ids) if t.chunk_ids else "",
                max(t.chunk_ids) if t.chunk_ids else "",
            ])

    records: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    shape_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    nan_counts: Dict[str, int] = defaultdict(int)
    inf_counts: Dict[str, int] = defaultdict(int)
    zero_counts: Dict[str, int] = defaultdict(int)
    total_counts: Dict[str, int] = defaultdict(int)

    for trial in selected:
        label = "success" if trial.success else "failure"
        for chunk_id in trial.chunk_ids:
            chunk_path = trial.path / f"chunk_{chunk_id:05d}.pt"
            vectors, names = load_chunk_vectors(chunk_path, fields)
            for name, vec in zip(names, vectors):
                norm = float(np.linalg.norm(vec))
                shape_counts[name][str(tuple(vec.shape))] += 1
                nan_counts[name] += int(np.isnan(vec).any())
                inf_counts[name] += int(np.isinf(vec).any())
                zero_counts[name] += int(np.allclose(vec, 0.0))
                total_counts[name] += 1
                records[name][f"{label}_norms"].append(norm)
                records[name]["all_norms"].append(norm)

    with (out_dir / "state_health_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "state",
            "shape_counts",
            "num_vectors",
            "nan_vectors",
            "inf_vectors",
            "zero_vectors",
            "success_norm_mean",
            "failure_norm_mean",
            "success_norm_std",
            "failure_norm_std",
            "max_abs_robust_z",
            "num_norm_outliers",
        ])
        for state in sorted(records):
            s_norms = np.asarray(records[state].get("success_norms", []), dtype=np.float64)
            f_norms = np.asarray(records[state].get("failure_norms", []), dtype=np.float64)
            all_norms = np.asarray(records[state].get("all_norms", []), dtype=np.float64)
            rz = robust_z(all_norms)
            writer.writerow([
                state,
                dict(shape_counts[state]),
                total_counts[state],
                nan_counts[state],
                inf_counts[state],
                zero_counts[state],
                float(np.nanmean(s_norms)) if s_norms.size else np.nan,
                float(np.nanmean(f_norms)) if f_norms.size else np.nan,
                float(np.nanstd(s_norms)) if s_norms.size else np.nan,
                float(np.nanstd(f_norms)) if f_norms.size else np.nan,
                float(np.nanmax(np.abs(rz))) if rz.size else np.nan,
                int(np.sum(np.abs(rz) > args.norm_outlier_z)) if rz.size else 0,
            ])

    import matplotlib.pyplot as plt

    for state in sorted(records):
        s_norms = np.asarray(records[state].get("success_norms", []), dtype=np.float64)
        f_norms = np.asarray(records[state].get("failure_norms", []), dtype=np.float64)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(s_norms, bins=35, alpha=0.55, label="success")
        ax.hist(f_norms, bins=35, alpha=0.55, label="failure")
        ax.set_title(f"L2 norm distribution: {state}")
        ax.set_xlabel("L2 norm")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"norm_distribution_{state}.png", dpi=180)
        plt.close(fig)

    write_json(
        out_dir / "health_metadata.json",
        {
            "root": str(root),
            "fields": fields,
            "selected_success_trials": trial_ids(success_trials),
            "selected_failure_trials": trial_ids(failure_trials),
            "common_chunk_ids_across_selected": common_ids,
            "all_selected_chunk_counts": {str(t.trial_id): t.num_chunk_files for t in selected},
        },
    )
    print(f"Wrote data health outputs to {out_dir}")


if __name__ == "__main__":
    main()
