"""
Failure-onset detection from divergence and AUC heatmaps.

Purpose
-------
Combine normalized cosine divergence and normalized linear-probe AUC into an
onset score, then find the earliest normalized-time chunk position where the
score stays above a threshold for consecutive chunks. This estimates where
success and failure representations first begin to split on tau in [0, 1].

Parameters
----------
--out_dir: shared analysis result directory containing divergence/ and
           linear_probe_auc/ subdirectories.
--threshold: onset threshold on normalized score.
--consecutive: number of consecutive chunks required to call an onset.

Usage
-----
python analysis/chunk_analysis/04_failure_onset.py \
    --out_dir analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate \
    --threshold 1.2 \
    --consecutive 2

Outputs
-------
Saved under --out_dir/failure_onset:
  onset_score_statewise.csv/png  normalized_D + normalized_AUC
  onset_summary.csv              first onset per state
  onset_metadata.json            threshold and source files
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from chunk_analysis_common import heatmap, minmax01, save_matrix_csv, write_json


def read_matrix_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        names = header[1:]
        rows = []
        for row in reader:
            rows.append([float(x) if x.lower() != "nan" else np.nan for x in row[1:]])
    return np.asarray(rows, dtype=np.float64), names


def first_consecutive(values: np.ndarray, threshold: float, consecutive: int) -> int | None:
    mask = values >= threshold
    for i in range(0, len(mask) - consecutive + 1):
        if bool(np.all(mask[i : i + consecutive])):
            return i
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--threshold", type=float, default=1.2)
    parser.add_argument("--consecutive", type=int, default=2)
    args = parser.parse_args()

    base = args.out_dir.resolve()
    out_dir = base / "failure_onset"
    out_dir.mkdir(parents=True, exist_ok=True)

    div, div_names = read_matrix_csv(base / "divergence" / "cosine_divergence_statewise.csv")
    auc, auc_names = read_matrix_csv(base / "linear_probe_auc" / "auc_statewise.csv")
    if div_names != auc_names or div.shape != auc.shape:
        raise ValueError(f"Divergence/AUC mismatch: {div.shape} {div_names} vs {auc.shape} {auc_names}")

    auc_effect = np.abs(auc - 0.5) * 2.0
    onset = minmax01(div) + minmax01(auc_effect)

    save_matrix_csv(out_dir / "onset_score_statewise.csv", onset, "chunk_position", div_names)
    heatmap(out_dir / "onset_score_statewise.png", onset, "Failure Onset Score", "state", "normalized time index", div_names)

    meta_path = base / "divergence" / "divergence_metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    tau = metadata.get("normalized_tau", [i / max(onset.shape[0] - 1, 1) for i in range(onset.shape[0])])

    with (out_dir / "onset_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "onset_position", "onset_tau", "threshold", "consecutive"])
        for s, state in enumerate(div_names):
            pos = first_consecutive(onset[:, s], args.threshold, args.consecutive)
            onset_tau = tau[pos] if pos is not None and pos < len(tau) else ""
            writer.writerow([state, "" if pos is None else pos, onset_tau, args.threshold, args.consecutive])

    write_json(
        out_dir / "onset_metadata.json",
        {
            "threshold": args.threshold,
            "consecutive": args.consecutive,
            "score_definition": "minmax(cosine_divergence) + minmax(abs(AUC - 0.5) * 2)",
            "target_length": metadata.get("target_length"),
            "normalized_tau": tau,
            "divergence_source": str(base / "divergence" / "cosine_divergence_statewise.csv"),
            "auc_source": str(base / "linear_probe_auc" / "auc_statewise.csv"),
        },
    )
    print(f"Wrote failure-onset outputs to {out_dir}")


if __name__ == "__main__":
    main()
