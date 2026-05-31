#!/usr/bin/env python3
"""
Task-level template dynamics summary pipeline.

Purpose
-------
Run the standard template-centroid analysis for one LIBERO/OOD task from a
single task name. The script automatically locates the task rollout manifest,
chunk manifest, label-review CSV, and latent `.pt` paths recorded in the chunk
manifest. It then writes a compact task-level summary for template dynamics.

Arguments
---------
  --task-name:
      Task name used in manifests and output folders, for example
      `put_the_cream_cheese_on_the_plate`.
  --output-dir:
      Directory for all generated CSV/JSON/PNG outputs. Defaults to
      `analysis/template_centroid/results/<task-name>`.
  --fields:
      Latent keys to summarize. Defaults to `action_head_input chunk_vector_mean`.
  --failure-mode:
      Natural failure mode used as the bowl/compositional template. Defaults to
      `wrong_receptacle`.
  --intervention-task-root:
      Optional chunk-wise intervention trace root. If omitted, the script uses
      `OOD_exp/dif_start_end_loc/outputs/chunk_wise/<task-name>_soft_success` when it exists.
  --intervention-summary:
      Optional intervention summary CSV. If omitted, the script uses
      `OOD_exp/dif_start_end_loc/outputs/videos/soft_success/<task-name>_soft_success/soft_success_summary.csv`
      when it exists.
  --intervention-chunk:
      Replacement chunk id for post-patch metrics. Defaults to 2.
  --late-fraction:
      Tail fraction for late final-margin alignment. Defaults to 0.2.

Usage
-----
python analysis/template_centroid/03_task_level_template_summary.py \\
    --task-name put_the_cream_cheese_on_the_plate

python analysis/template_centroid/03_task_level_template_summary.py \\
    --task-name put_the_cream_cheese_on_the_plate \\
    --fields action_head_input chunk_vector_mean \\
    --intervention-chunk 2 \\
    --output-dir analysis/template_centroid/results/put_the_cream_cheese_on_the_plate

Outputs
-------
Saved under:
  analysis/template_centroid/results/<task-name>

Key files:
  task_summary.csv
  task_summary_metadata.json
  paper_metrics_rollout_level.csv
  paper_metrics_summary.csv
  full_rollout_sample_margins.csv
  cosine_drift_by_transition.csv
  normalized_full_rollout_margin_curve.png
  normalized_cosine_drift_curve.png
  template_switch_rate_boxplot.png
  template_entropy_boxplot.png
  margin_variance_boxplot.png
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_FIELDS = ["action_head_input", "chunk_vector_mean"]
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

SUMMARY_METRICS = {
    "early_margin_gap": "early_mean_margin_chunk_0_4",
    "nearest_success_template_ratio": "nearest_success_ratio_chunk_0_4",
    "post_patch_redirection_ratio": "nearest_success_ratio_post_patch_chunk_2_4",
    "final_margin": "final_margin",
    "late_final_20pct_margin": "late_success_alignment_final_20pct",
    "switch_count": "template_switch_count",
    "switch_rate": "template_switch_rate",
    "template_entropy": "template_entropy_full_rollout",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to analysis/template_centroid/results/<task-name>.",
    )
    parser.add_argument("--fields", nargs="+", default=DEFAULT_FIELDS)
    parser.add_argument("--failure-mode", default="wrong_receptacle")
    parser.add_argument("--max-early-chunk", type=int, default=4)
    parser.add_argument("--full-max-chunk", type=int, default=30)
    parser.add_argument("--normalized-bins", type=int, default=20)
    parser.add_argument("--intervention-task-root", type=Path, default=None)
    parser.add_argument("--intervention-summary", type=Path, default=None)
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


def find_one_csv(root: Path, task_name: str, suffix: str, *, required: bool = True) -> Path | None:
    exact = sorted(root.glob(f"*{task_name}{suffix}"))
    matches = exact or sorted(path for path in root.glob(f"*{suffix}") if task_name in path.name)
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f"Could not find *{task_name}{suffix} under {root}")
    return None


def default_intervention_paths(task_name: str) -> tuple[Path | None, Path | None]:
    task_root = REPO_ROOT / "OOD_exp" / "dif_start_end_loc" / "outputs" / "chunk_wise" / f"{task_name}_soft_success"
    summary = (
        REPO_ROOT
        / "OOD_exp"
        / "dif_start_end_loc"
        / "outputs"
        / "videos"
        / "soft_success"
        / f"{task_name}_soft_success"
        / "soft_success_summary.csv"
    )
    return (task_root if task_root.exists() else None, summary if summary.exists() else None)


def resolve_inputs(args: argparse.Namespace) -> dict[str, Path | None]:
    manifest_dir = REPO_ROOT / "OOD_exp" / "dif_start_end_loc" / "manifests"
    annotation_dir = REPO_ROOT / "OOD_exp" / "dif_start_end_loc" / "annotations"
    rollout_manifest = find_one_csv(manifest_dir, args.task_name, "_rollout_manifest.csv")
    chunk_manifest = find_one_csv(manifest_dir, args.task_name, "_chunk_manifest.csv")
    label_csv = find_one_csv(annotation_dir, args.task_name, "_label_review.csv", required=False)
    default_task_root, default_summary = default_intervention_paths(args.task_name)
    return {
        "rollout_manifest": rollout_manifest,
        "chunk_manifest": chunk_manifest,
        "label_csv": label_csv,
        "intervention_task_root": args.intervention_task_root or default_task_root,
        "intervention_summary": args.intervention_summary or default_summary,
    }


def run_command(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_analysis(args: argparse.Namespace, paths: dict[str, Path | None], output_dir: Path) -> None:
    centroid_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "01_template_centroid_analysis.py"),
        "--chunk-manifest",
        str(paths["chunk_manifest"]),
        "--output-dir",
        str(output_dir),
        "--fields",
        *args.fields,
        "--failure-mode",
        args.failure_mode,
        "--max-early-chunk",
        str(args.max_early_chunk),
        "--full-max-chunk",
        str(args.full_max_chunk),
        "--normalized-bins",
        str(args.normalized_bins),
    ]
    if paths["intervention_task_root"] is not None:
        centroid_cmd.extend(["--intervention-task-root", str(paths["intervention_task_root"])])
    if paths["intervention_summary"] is not None:
        centroid_cmd.extend(["--intervention-summary", str(paths["intervention_summary"])])
    run_command(centroid_cmd)

    metrics_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "02_paper_report_metrics.py"),
        "--results-dir",
        str(output_dir),
        "--intervention-chunk",
        str(args.intervention_chunk),
        "--late-fraction",
        str(args.late_fraction),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--seed",
        str(args.seed),
    ]
    run_command(metrics_cmd)


def summarize_cosine_drift(output_dir: Path, rng: np.random.Generator, bootstrap_samples: int) -> dict[tuple[str, str], dict[str, Any]]:
    path = output_dir / "cosine_drift_by_transition.csv"
    if not path.exists():
        return {}
    rollout_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in read_csv(path):
        rollout_values[(row["latent_key"], row["group"], row["rollout_id"])].append(
            float(row["cosine_drift"])
        )
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (latent_key, group, _), drifts in rollout_values.items():
        values[(latent_key, group)].append(float(np.mean(drifts)))
    return {
        key: mean_ci(group_values, rng, bootstrap_samples)
        for key, group_values in values.items()
    }


def build_task_summary(output_dir: Path, *, bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    paper_rows = read_csv(output_dir / "paper_metrics_summary.csv")
    by_key_metric: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    group_labels: dict[tuple[str, str], str] = {}
    n_rollouts: dict[tuple[str, str], str] = {}
    for row in paper_rows:
        key = (row["latent_key"], row["group"])
        by_key_metric[key][row["metric"]] = row
        group_labels[key] = row["group_label"]
        n_rollouts[key] = row["n_rollouts"]

    rng = np.random.default_rng(seed)
    drift_stats = summarize_cosine_drift(output_dir, rng, bootstrap_samples)
    summary_rows: list[dict[str, Any]] = []
    for latent_key, group in sorted(by_key_metric):
        row: dict[str, Any] = {
            "latent_key": latent_key,
            "group": group,
            "group_label": group_labels[(latent_key, group)],
            "n_rollouts": n_rollouts[(latent_key, group)],
        }
        for public_name, source_metric in SUMMARY_METRICS.items():
            metric_row = by_key_metric[(latent_key, group)].get(source_metric)
            if metric_row is None:
                row[f"{public_name}_mean"] = ""
                row[f"{public_name}_ci_low"] = ""
                row[f"{public_name}_ci_high"] = ""
                continue
            row[f"{public_name}_mean"] = metric_row["mean"]
            row[f"{public_name}_ci_low"] = metric_row["ci_low"]
            row[f"{public_name}_ci_high"] = metric_row["ci_high"]
        drift = drift_stats.get((latent_key, group), {})
        row["mean_cosine_drift_mean"] = drift.get("mean", "")
        row["mean_cosine_drift_ci_low"] = drift.get("ci_low", "")
        row["mean_cosine_drift_ci_high"] = drift.get("ci_high", "")
        summary_rows.append(row)
    return summary_rows


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    inputs: dict[str, Path | None],
    output_dir: Path,
) -> None:
    rollout_rows = read_csv(inputs["rollout_manifest"]) if inputs["rollout_manifest"] else []
    chunk_rows = read_csv(inputs["chunk_manifest"]) if inputs["chunk_manifest"] else []
    label_rows = read_csv(inputs["label_csv"]) if inputs["label_csv"] else []
    latent_paths = [REPO_ROOT / row["latent_pt_path"] for row in chunk_rows if row.get("latent_pt_path")]
    existing_latents = sum(path.exists() for path in latent_paths)
    payload = {
        "task_name": args.task_name,
        "output_dir": output_dir.as_posix(),
        "fields": args.fields,
        "failure_mode": args.failure_mode,
        "inputs": {
            key: value.as_posix() if value is not None else None
            for key, value in inputs.items()
        },
        "counts": {
            "rollout_manifest_rows": len(rollout_rows),
            "chunk_manifest_rows": len(chunk_rows),
            "label_csv_rows": len(label_rows),
            "latent_paths_in_chunk_manifest": len(latent_paths),
            "existing_latent_paths": existing_latents,
        },
        "standard_figures": [
            "normalized_full_rollout_margin_curve.png",
            "normalized_cosine_drift_curve.png",
            "template_switch_rate_boxplot.png",
            "template_entropy_boxplot.png",
            "margin_variance_boxplot.png",
            "nearest_template_distribution.png",
            "rollout_margin_heatmap.png",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (SCRIPT_DIR / "results" / args.task_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = resolve_inputs(args)
    write_metadata(output_dir / "task_summary_metadata.json", args, inputs, output_dir)
    run_analysis(args, inputs, output_dir)
    summary_rows = build_task_summary(
        output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_csv(output_dir / "task_summary.csv", summary_rows)
    print(f"Wrote task-level summary to {output_dir / 'task_summary.csv'}")


if __name__ == "__main__":
    main()
