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
4. intervention unrecovered failure

The reported metrics include:
1. Early Mean Margin over chunks 0..4.
2. Post Intervention Mean Margin over chunks 2..4.
3. Nearest Success Template Ratio over early, pre-patch, and post-patch windows.
4. Template Switch Count and length-normalized Switch Rate over full trajectories.
5. Template Entropy over early and full-rollout windows.
6. Margin Stability summaries: mean, variance, mean absolute margin, minimum margin,
   final margin, and late final-20% alignment.

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
  template_switch_rate_boxplot.png
  template_entropy_boxplot.png
  margin_variance_boxplot.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


GROUP_ORDER = [
    "natural_success",
    "natural_failure",
    "intervention_recovered_success",
    "intervention_unrecovered_failure",
]

GROUP_LABELS = {
    "natural_success": "natural success",
    "natural_failure": "natural failure",
    "intervention_recovered_success": "intervention recovered success",
    "intervention_unrecovered_failure": "intervention unrecovered failure",
}
GROUP_COLORS = {
    "natural_success": "#1f77b4",
    "natural_failure": "#d62728",
    "intervention_recovered_success": "#2ca02c",
    "intervention_unrecovered_failure": "#9467bd",
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


def switch_rate(rows: list[dict[str, str]]) -> float:
    denominator = max(1, len(rows) - 1)
    return float(switch_count(rows) / denominator)


def template_entropy(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    success_ratio = float(
        np.mean([float(row["margin_success_minus_bowl"]) >= 0.0 for row in rows])
    )
    probs = [success_ratio, 1.0 - success_ratio]
    entropy = 0.0
    for prob in probs:
        if prob > 0.0:
            entropy -= prob * math.log2(prob)
    return float(entropy)


def row_window(rows: list[dict[str, str]], start: int, end: int) -> list[dict[str, str]]:
    return [row for row in rows if start <= int(row["chunk_id"]) <= end]


def margin_values(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([float(row["margin_success_minus_bowl"]) for row in rows], dtype=np.float64)


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
        margins = margin_values(rows)
        early_rows = row_window(rows, 0, 4)
        switches = switch_count(rows)
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
                "template_switch_count": switches,
                "template_switch_rate": float(switches / max(1, len(rows) - 1)),
                "template_entropy_chunk_0_4": template_entropy(early_rows),
                "template_entropy_full_rollout": template_entropy(rows),
                "mean_margin_full_rollout": float(margins.mean()) if margins.size else None,
                "margin_variance_full_rollout": float(margins.var(ddof=1)) if margins.size > 1 else 0.0,
                "abs_margin_mean_full_rollout": float(np.abs(margins).mean()) if margins.size else None,
                "min_margin_full_rollout": float(margins.min()) if margins.size else None,
                "final_margin": float(margins[-1]) if margins.size else None,
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
        "template_switch_rate",
        "template_entropy_chunk_0_4",
        "template_entropy_full_rollout",
        "mean_margin_full_rollout",
        "margin_variance_full_rollout",
        "abs_margin_mean_full_rollout",
        "min_margin_full_rollout",
        "final_margin",
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


def plot_group_boxplot(
    path: Path,
    rollout_rows: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    latent_keys = sorted({row["latent_key"] for row in rollout_rows})
    fig, axes = plt.subplots(1, len(latent_keys), figsize=(5.3 * len(latent_keys), 4.4), sharey=True)
    if len(latent_keys) == 1:
        axes = [axes]
    for axis, latent_key in zip(axes, latent_keys):
        data = []
        labels = []
        colors = []
        for group in GROUP_ORDER:
            values = [
                float(row[metric])
                for row in rollout_rows
                if row["latent_key"] == latent_key and row["group"] == group and row[metric] not in {"", None}
            ]
            data.append(values)
            labels.append(GROUP_LABELS[group].replace(" ", "\n"))
            colors.append(GROUP_COLORS[group])
        box = axis.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.18)
            patch.set_edgecolor(color)
        for median in box["medians"]:
            median.set_color("#222222")
            median.set_linewidth(1.6)
        rng = np.random.default_rng(11)
        for idx, (values, color) in enumerate(zip(data, colors), start=1):
            if not values:
                continue
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            axis.scatter(
                np.full(len(values), idx) + jitter,
                values,
                s=24,
                alpha=0.78,
                color=color,
                edgecolors="white",
                linewidths=0.35,
            )
        axis.set_title(latent_key)
        axis.grid(True, axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelsize=9)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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
    plot_group_boxplot(
        output_dir / "template_switch_rate_boxplot.png",
        rollout_rows,
        metric="template_switch_rate",
        ylabel="switch count / (num chunks - 1)",
        title="Template Switch Rate",
    )
    plot_group_boxplot(
        output_dir / "template_entropy_boxplot.png",
        rollout_rows,
        metric="template_entropy_full_rollout",
        ylabel="template entropy (bits)",
        title="Full-Rollout Template Entropy",
    )
    plot_group_boxplot(
        output_dir / "margin_variance_boxplot.png",
        rollout_rows,
        metric="margin_variance_full_rollout",
        ylabel="variance of success-minus-bowl margin",
        title="Margin Stability",
    )
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
