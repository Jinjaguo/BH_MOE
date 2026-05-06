#!/usr/bin/env python3
"""
Convert template-centroid visual results into paper-reportable metrics.

Purpose
-------
Summarize template margin dynamics with rollout-level metrics and confidence
intervals for three groups:
1. natural success
2. natural failure
3. intervention recovered success

The reported metrics include:
1. Early Mean Margin over chunks 0..4.
2. Post Intervention Mean Margin over chunks 2..4.
3. Nearest Success Template Ratio over early, pre-patch, and post-patch windows.
4. Template Switch Count over full trajectories.
5. Late Success Alignment over the final 20% of each rollout.

Arguments
---------
  --results-dir:
      Directory containing template-centroid CSV outputs, especially
      `sample_margins.csv`, `full_rollout_sample_margins.csv`, and
      `rollout_dynamics_summary.csv`.
  --output-dir:
      Directory where metric CSV/JSON outputs are saved. Defaults to
      `--results-dir`.
  --intervention-chunk:
      Replacement chunk id. Defaults to 2.
  --late-fraction:
      Fraction of each rollout tail used for Late Success Alignment.
  --bootstrap-samples:
      Number of bootstrap resamples for 95% confidence intervals.
  --seed:
      Random seed for bootstrap intervals.

Usage
-----
python analysis/template_centroid/02_paper_report_metrics.py \
    --results-dir analysis/template_centroid/results/put_the_cream_cheese_on_the_plate \
    --intervention-chunk 2 \
    --late-fraction 0.2

Outputs
-------
Saved under:
  analysis/template_centroid/results/put_the_cream_cheese_on_the_plate

Key files:
  paper_metrics_rollout_level.csv
  paper_metrics_summary.csv
  paper_metrics_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


GROUP_ORDER = [
    "natural_success",
    "natural_failure",
    "intervention_recovered_success",
]

GROUP_LABELS = {
    "natural_success": "natural success",
    "natural_failure": "natural failure",
    "intervention_recovered_success": "intervention recovered success",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("analysis/template_centroid/results/put_the_cream_cheese_on_the_plate"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--intervention-chunk", type=int, default=2)
    parser.add_argument("--late-fraction", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values: Iterable[float], rng: np.random.Generator, bootstrap_samples: int) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": "", "ci_low": "", "ci_high": "", "std": "", "n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if arr.size == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "std": std, "n": 1}
    indices = rng.integers(0, arr.size, size=(bootstrap_samples, arr.size))
    boot_means = arr[indices].mean(axis=1)
    return {
        "mean": mean,
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "std": std,
        "n": int(arr.size),
    }


def group_rollout_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["latent_key"], row["rollout_id"], int(row["trial_id"]))
        grouped[key].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: int(row["chunk_id"]))
    return grouped


def window_mean(rows: list[dict[str, str]], start: int, end: int) -> float | None:
    values = [
        float(row["margin_success_minus_bowl"])
        for row in rows
        if start <= int(row["chunk_id"]) <= end
    ]
    if not values:
        return None
    return float(np.mean(values))


def nearest_success_ratio(rows: list[dict[str, str]], start: int, end: int) -> float | None:
    selected = [row for row in rows if start <= int(row["chunk_id"]) <= end]
    if not selected:
        return None
    return float(np.mean([row["nearest_template"] == "success_template" for row in selected]))


def late_mean(rows: list[dict[str, str]], late_fraction: float) -> float | None:
    if not rows:
        return None
    num_late = max(1, int(math.ceil(len(rows) * late_fraction)))
    values = [float(row["margin_success_minus_bowl"]) for row in rows[-num_late:]]
    return float(np.mean(values))


def switch_count(rows: list[dict[str, str]]) -> int:
    if len(rows) < 2:
        return 0
    signs = [float(row["margin_success_minus_bowl"]) >= 0.0 for row in rows]
    return sum(left != right for left, right in zip(signs[:-1], signs[1:]))


def build_rollout_metrics(
    full_rows: list[dict[str, str]],
    *,
    intervention_chunk: int,
    late_fraction: float,
) -> list[dict[str, Any]]:
    rollout_rows: list[dict[str, Any]] = []
    grouped = group_rollout_rows(full_rows)
    post_end = intervention_chunk + 2
    for (latent_key, rollout_id, trial_id), rows in sorted(grouped.items()):
        group = rows[0]["group"]
        early_mean = window_mean(rows, 0, 4)
        post_mean = window_mean(rows, intervention_chunk, post_end)
        pre_ratio = nearest_success_ratio(rows, 0, intervention_chunk - 1)
        post_ratio = nearest_success_ratio(rows, intervention_chunk, post_end)
        early_ratio = nearest_success_ratio(rows, 0, 4)
        rollout_rows.append(
            {
                "latent_key": latent_key,
                "rollout_id": rollout_id,
                "trial_id": trial_id,
                "group": group,
                "group_label": GROUP_LABELS.get(group, group),
                "num_chunks": len(rows),
                "early_mean_margin_chunk_0_4": early_mean,
                "post_intervention_mean_margin_chunk_2_4": post_mean,
                "nearest_success_ratio_chunk_0_4": early_ratio,
                "nearest_success_ratio_pre_patch_chunk_0_1": pre_ratio,
                "nearest_success_ratio_post_patch_chunk_2_4": post_ratio,
                "template_switch_count": switch_count(rows),
                "late_success_alignment_final_20pct": late_mean(rows, late_fraction),
            }
        )
    return rollout_rows


def summarize_rollout_metrics(
    rollout_rows: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    metric_columns = [
        "early_mean_margin_chunk_0_4",
        "post_intervention_mean_margin_chunk_2_4",
        "nearest_success_ratio_chunk_0_4",
        "nearest_success_ratio_pre_patch_chunk_0_1",
        "nearest_success_ratio_post_patch_chunk_2_4",
        "template_switch_count",
        "late_success_alignment_final_20pct",
    ]
    summary_rows: list[dict[str, Any]] = []
    latent_keys = sorted({row["latent_key"] for row in rollout_rows})
    for latent_key in latent_keys:
        for group in GROUP_ORDER:
            group_rows = [
                row for row in rollout_rows if row["latent_key"] == latent_key and row["group"] == group
            ]
            if not group_rows:
                continue
            for metric in metric_columns:
                values = [
                    float(row[metric])
                    for row in group_rows
                    if row[metric] is not None and row[metric] != ""
                ]
                stats = mean_ci(values, rng, bootstrap_samples)
                summary_rows.append(
                    {
                        "latent_key": latent_key,
                        "group": group,
                        "group_label": GROUP_LABELS[group],
                        "metric": metric,
                        "mean": stats["mean"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "std": stats["std"],
                        "n_rollouts": stats["n"],
                    }
                )
    return summary_rows


def compact_json_summary(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for row in summary_rows:
        latent_key = row["latent_key"]
        group = row["group"]
        metric = row["metric"]
        payload.setdefault(latent_key, {}).setdefault(group, {})[metric] = {
            "mean": row["mean"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
            "std": row["std"],
            "n_rollouts": row["n_rollouts"],
        }
    return payload


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    full_rows = read_csv(args.results_dir / "full_rollout_sample_margins.csv")
    rollout_rows = build_rollout_metrics(
        full_rows,
        intervention_chunk=args.intervention_chunk,
        late_fraction=args.late_fraction,
    )
    rng = np.random.default_rng(args.seed)
    summary_rows = summarize_rollout_metrics(
        rollout_rows,
        rng=rng,
        bootstrap_samples=args.bootstrap_samples,
    )

    write_csv(output_dir / "paper_metrics_rollout_level.csv", rollout_rows)
    write_csv(output_dir / "paper_metrics_summary.csv", summary_rows)
    json_payload = {
        "results_dir": args.results_dir.as_posix(),
        "intervention_chunk": args.intervention_chunk,
        "late_fraction": args.late_fraction,
        "bootstrap_samples": args.bootstrap_samples,
        "metrics": compact_json_summary(summary_rows),
    }
    (output_dir / "paper_metrics_summary.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote paper metrics to {output_dir}")


if __name__ == "__main__":
    main()
