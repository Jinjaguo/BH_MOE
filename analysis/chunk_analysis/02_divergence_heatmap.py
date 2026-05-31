"""
Chunk-wise success-failure cosine divergence heatmap.

Purpose
-------
Build state-wise vectors for 20 successful and 20 failed trials, linearly
interpolate every trajectory onto a normalized time axis tau in [0, 1], and
compute the cosine distance between success and failure mean embeddings for
each normalized chunk/state pair. It also computes a fused representation by
concatenating L2-normalized state vectors.

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
python analysis/chunk_analysis/02_divergence_heatmap.py \
    --root OOD_exp/dif_start_end_loc/outputs/chunk_wise/put_the_cream_cheese_on_the_plate \
    --out_dir analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate

Outputs
-------
Saved under --out_dir/divergence:
  cosine_divergence_statewise.csv/png  D in R^{C x 5}
  cosine_divergence_fused.csv/png      fused D in R^{C x 1}
  Z_success_<state>.npy, Z_failure_<state>.npy
  Z_fused_success.npy, Z_fused_failure.npy
  divergence_metadata.json             selected trials and chunk ids
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chunk_analysis_common import (
    DEFAULT_ROOT,
    cosine_distance,
    discover_trials,
    heatmap,
    load_group_normalized_state_tensors,
    load_manifest_fields,
    make_fused_from_states,
    normalized_time_grid,
    save_matrix_csv,
    select_balanced_trials,
    trial_ids,
    write_json,
)


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
    out_dir = (args.out_dir / "divergence").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = args.fields or load_manifest_fields(root)

    trials = discover_trials(root)
    success_trials, failure_trials = select_balanced_trials(trials, args.n_success, args.n_failure)
    selected = success_trials + failure_trials

    success_states, state_names = load_group_normalized_state_tensors(success_trials, args.target_length, fields)
    failure_states, _ = load_group_normalized_state_tensors(failure_trials, args.target_length, fields)

    c_count, s_count = args.target_length, len(state_names)
    div = np.zeros((c_count, s_count), dtype=np.float64)
    state_shapes = {}
    for s, state in enumerate(state_names):
        z_success = success_states[state]
        z_failure = failure_states[state]
        state_shapes[state] = {"success": list(z_success.shape), "failure": list(z_failure.shape)}
        np.save(out_dir / f"Z_success_{state}.npy", z_success)
        np.save(out_dir / f"Z_failure_{state}.npy", z_failure)
        for c in range(c_count):
            div[c, s] = cosine_distance(z_success[:, c, :].mean(axis=0), z_failure[:, c, :].mean(axis=0))

    z_fused_success = make_fused_from_states(success_states, state_names)
    z_fused_failure = make_fused_from_states(failure_states, state_names)
    np.save(out_dir / "Z_fused_success.npy", z_fused_success)
    np.save(out_dir / "Z_fused_failure.npy", z_fused_failure)

    fused_div = np.zeros((c_count, 1), dtype=np.float64)
    for c in range(c_count):
        fused_div[c, 0] = cosine_distance(z_fused_success[:, c, :].mean(axis=0), z_fused_failure[:, c, :].mean(axis=0))

    save_matrix_csv(out_dir / "cosine_divergence_statewise.csv", div, "chunk_position", state_names)
    save_matrix_csv(out_dir / "cosine_divergence_fused.csv", fused_div, "chunk_position", ["fused_concat"])
    heatmap(
        out_dir / "cosine_divergence_statewise.png",
        div,
        "Success-Failure Cosine Divergence",
        "state",
        "normalized time index",
        state_names,
    )
    heatmap(
        out_dir / "cosine_divergence_fused.png",
        fused_div,
        "Success-Failure Cosine Divergence (Fused)",
        "representation",
        "normalized time index",
        ["fused_concat"],
    )
    write_json(
        out_dir / "divergence_metadata.json",
        {
            "root": str(root),
            "fields": fields,
            "state_names": state_names,
            "selected_success_trials": trial_ids(success_trials),
            "selected_failure_trials": trial_ids(failure_trials),
            "target_length": args.target_length,
            "normalized_tau": normalized_time_grid(args.target_length).tolist(),
            "original_chunk_counts": {str(t.trial_id): t.num_chunk_files for t in selected},
            "state_shapes": state_shapes,
            "Z_fused_success_shape": list(z_fused_success.shape),
            "Z_fused_failure_shape": list(z_fused_failure.shape),
        },
    )
    print(f"Wrote divergence outputs to {out_dir}")


if __name__ == "__main__":
    main()
