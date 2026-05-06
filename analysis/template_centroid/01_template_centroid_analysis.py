#!/usr/bin/env python3
"""
Analyze early-chunk template centroids for OOD rollout latent samples.

Purpose
-------
Build two early template centroids for one task:
1. A success template from successful rollout early chunks.
2. A bowl compositional template from wrong-receptacle early chunks.

The script then measures whether early latent samples are closer to the correct
template before action decoding by producing:
1. Early template margin curves over chunks 0..4.
2. Nearest-template distributions for natural success and natural failure.
3. Full-rollout template dynamics curves and heatmaps.
4. Optional PCA scatter plots as supporting visualization.

Arguments
---------
  --chunk-manifest:
      CSV with one row per chunk sample. Required columns include
      `latent_pt_path`, `success`, `dominant_failure_mode`, `valid_for_probe`,
      `is_early_chunk`, `trial_id`, and `chunk_id`.
  --output-dir:
      Directory where plots, CSV summaries, centroids, and metadata are saved.
  --fields:
      Latent keys to analyze from each `.pt` file.
  --failure-mode:
      Failure mode used to define the bowl compositional template.
  --max-early-chunk:
      Highest early chunk id to include in the margin curve.
  --full-max-chunk:
      Highest real chunk id to include in the full-rollout real-time curve.
  --intervention-task-root:
      Optional chunk-wise output root for recovered intervention rollouts.
  --intervention-summary:
      Optional soft-success summary CSV. Rows with `analysis_success=True` are
      treated as intervention successful rollouts.
  --centroid-mode:
      `chunk_aligned` builds one success/bowl centroid pair per chunk id.
      `global` builds one success/bowl centroid pair across all early chunks.

Usage
-----
python analysis/template_centroid/01_template_centroid_analysis.py \
    --chunk-manifest OOD_exp/manifests/pi05_put_the_cream_cheese_on_the_plate_chunk_manifest.csv \
    --output-dir analysis/template_centroid/results/put_the_cream_cheese_on_the_plate \
    --fields action_head_input chunk_vector_mean \
    --failure-mode wrong_receptacle \
    --max-early-chunk 4

Outputs
-------
Results are saved under:
  analysis/template_centroid/results/put_the_cream_cheese_on_the_plate

Key files:
  template_margin_curve.png
  nearest_template_distribution.png
  full_rollout_margin_curve.png
  normalized_full_rollout_margin_curve.png
  rollout_margin_heatmap.png
  rollout_dynamics_summary.csv
  pca_scatter.png
  template_margin_by_chunk.csv
  nearest_template_distribution.csv
  sample_margins.csv
  summary.json
  centroids/*.npy
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CHUNK_MANIFEST = Path(
    "OOD_exp/manifests/pi05_put_the_cream_cheese_on_the_plate_chunk_manifest.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "analysis/template_centroid/results/put_the_cream_cheese_on_the_plate"
)
DEFAULT_FIELDS = ["action_head_input", "chunk_vector_mean"]
GROUPS = ["natural_success", "natural_failure", "intervention_recovered_success"]
GROUP_LABELS = {
    "natural_success": "natural success",
    "natural_failure": "natural failure",
    "intervention_recovered_success": "intervention success",
}
GROUP_COLORS = {
    "natural_success": "#1f77b4",
    "natural_failure": "#d62728",
    "intervention_recovered_success": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-manifest", type=Path, default=DEFAULT_CHUNK_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fields", nargs="+", default=DEFAULT_FIELDS)
    parser.add_argument("--failure-mode", default="wrong_receptacle")
    parser.add_argument("--max-early-chunk", type=int, default=4)
    parser.add_argument("--full-max-chunk", type=int, default=30)
    parser.add_argument("--normalized-bins", type=int, default=20)
    parser.add_argument(
        "--intervention-task-root",
        type=Path,
        default=Path("OOD_exp/outputs/chunk_wise/put_the_cream_cheese_on_the_plate_soft_success"),
    )
    parser.add_argument(
        "--intervention-summary",
        type=Path,
        default=Path(
            "OOD_exp/outputs/videos/soft_success/"
            "put_the_cream_cheese_on_the_plate_soft_success/soft_success_summary.csv"
        ),
    )
    parser.add_argument(
        "--centroid-mode",
        choices=["chunk_aligned", "global"],
        default="chunk_aligned",
    )
    return parser.parse_args()


def load_torch_zip_pickle(path: Path) -> dict[str, Any]:
    """Load torch-saved zip files that only contain pickle-backed numpy arrays."""
    with zipfile.ZipFile(path) as archive:
        data_names = [name for name in archive.namelist() if name.endswith("data.pkl")]
        if not data_names:
            raise ValueError(f"No data.pkl found in {path}")
        obj = pickle.loads(archive.read(data_names[0]))
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj).__name__}")
    return obj


def state_to_vector(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1, arr.shape[-1]).mean(axis=0)


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return vector * 0.0
    return vector / norm


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(l2_normalize(left), l2_normalize(right)))


def parse_int_bool(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n"}:
        return 0
    raise ValueError(f"Cannot parse int bool value: {value!r}")


def parse_csv_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def load_manifest_rows(path: Path, failure_mode: str, max_early_chunk: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected: list[dict[str, Any]] = []
    for row in rows:
        if parse_int_bool(row["valid_for_probe"]) != 1:
            continue
        if parse_int_bool(row["is_early_chunk"]) != 1:
            continue
        chunk_id = int(row["chunk_id"])
        if chunk_id > max_early_chunk:
            continue
        success = parse_int_bool(row["success"])
        dominant_failure_mode = row["dominant_failure_mode"]
        if success != 1 and dominant_failure_mode != failure_mode:
            continue
        selected.append(row)
    return selected


def load_full_manifest_rows(path: Path, failure_mode: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected: list[dict[str, Any]] = []
    for row in rows:
        if parse_int_bool(row["valid_for_probe"]) != 1:
            continue
        success = parse_int_bool(row["success"])
        dominant_failure_mode = row["dominant_failure_mode"]
        if success != 1 and dominant_failure_mode != failure_mode:
            continue
        selected.append(row)
    return selected


def intervention_success_rows(
    task_root: Path,
    summary_path: Path,
    *,
    max_chunk: int | None = None,
) -> list[dict[str, Any]]:
    if not task_root.exists() or not summary_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    success_trials = [
        int(row["trial_id"])
        for row in summary_rows
        if parse_csv_bool(row.get("analysis_success"))
    ]
    for trial_id in success_trials:
        trial_dir = task_root / f"trial_{trial_id}"
        if not trial_dir.exists():
            continue
        chunk_paths = sorted(trial_dir.glob("chunk_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
        for chunk_path in chunk_paths:
            chunk_id = int(chunk_path.stem.split("_")[1])
            if max_chunk is not None and chunk_id > max_chunk:
                continue
            rows.append(
                {
                    "rollout_id": f"intervention_success_trial_{trial_id:03d}",
                    "trial_id": str(trial_id),
                    "chunk_id": str(chunk_id),
                    "latent_pt_path": chunk_path.as_posix(),
                    "success": "1",
                    "dominant_failure_mode": "intervention_recovered_success",
                    "valid_for_probe": "1",
                    "is_early_chunk": "1" if chunk_id <= 4 else "0",
                    "analysis_group": "intervention_recovered_success",
                }
            )
    return rows


def load_field_vectors(rows: Iterable[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    cache: dict[Path, dict[str, Any]] = {}
    for row in rows:
        path = Path(row["latent_pt_path"])
        if path not in cache:
            cache[path] = load_torch_zip_pickle(path)
        payload = cache[path]
        if field not in payload:
            raise KeyError(f"{field} not found in {path}; available={sorted(payload)}")
        vector = state_to_vector(payload[field])
        success = parse_int_bool(row["success"])
        group = row.get("analysis_group") or ("natural_success" if success else "natural_failure")
        samples.append(
            {
                "rollout_id": row["rollout_id"],
                "trial_id": int(row["trial_id"]),
                "chunk_id": int(row["chunk_id"]),
                "group": group,
                "dominant_failure_mode": row["dominant_failure_mode"],
                "vector": vector,
            }
        )
    return samples


def load_field_vectors_with_cache(
    rows: Iterable[dict[str, Any]], field: str, cache: dict[Path, dict[str, Any]]
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["latent_pt_path"])
        if path not in cache:
            cache[path] = load_torch_zip_pickle(path)
        payload = cache[path]
        if field not in payload:
            raise KeyError(f"{field} not found in {path}; available={sorted(payload)}")
        vector = state_to_vector(payload[field])
        success = parse_int_bool(row["success"])
        group = row.get("analysis_group") or ("natural_success" if success else "natural_failure")
        samples.append(
            {
                "rollout_id": row["rollout_id"],
                "trial_id": int(row["trial_id"]),
                "chunk_id": int(row["chunk_id"]),
                "group": group,
                "dominant_failure_mode": row["dominant_failure_mode"],
                "vector": vector,
            }
        )
    return samples


def centroid(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("Cannot build centroid from zero vectors")
    normalized = np.stack([l2_normalize(vector) for vector in vectors], axis=0)
    return l2_normalize(normalized.mean(axis=0))


def build_centroid_pairs(
    samples: list[dict[str, Any]], centroid_mode: str
) -> dict[int | str, tuple[np.ndarray, np.ndarray]]:
    success_vectors = [sample["vector"] for sample in samples if sample["group"] == "natural_success"]
    failure_vectors = [sample["vector"] for sample in samples if sample["group"] == "natural_failure"]
    if centroid_mode == "global":
        return {"global": (centroid(success_vectors), centroid(failure_vectors))}

    pairs: dict[int | str, tuple[np.ndarray, np.ndarray]] = {}
    chunk_ids = sorted({sample["chunk_id"] for sample in samples})
    for chunk_id in chunk_ids:
        chunk_success_vectors = [
            sample["vector"]
            for sample in samples
            if sample["chunk_id"] == chunk_id and sample["group"] == "natural_success"
        ]
        chunk_failure_vectors = [
            sample["vector"]
            for sample in samples
            if sample["chunk_id"] == chunk_id and sample["group"] == "natural_failure"
        ]
        pairs[chunk_id] = (centroid(chunk_success_vectors), centroid(chunk_failure_vectors))
    return pairs


def compute_field_results(
    samples: list[dict[str, Any]], field: str, centroid_mode: str
) -> tuple[dict[int | str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]], list[dict[str, Any]]]:
    centroid_pairs = build_centroid_pairs(samples, centroid_mode)
    sample_rows, margin_rows = score_samples_with_centroids(
        samples, field, centroid_pairs, centroid_mode
    )
    return centroid_pairs, sample_rows, margin_rows


def score_samples_with_centroids(
    samples: list[dict[str, Any]],
    field: str,
    centroid_pairs: dict[int | str, tuple[np.ndarray, np.ndarray]],
    centroid_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rows: list[dict[str, Any]] = []
    for sample in samples:
        centroid_key: int | str = "global" if centroid_mode == "global" else sample["chunk_id"]
        success_centroid, bowl_centroid = centroid_pairs[centroid_key]
        success_cos = cosine(sample["vector"], success_centroid)
        bowl_cos = cosine(sample["vector"], bowl_centroid)
        nearest = "success_template" if success_cos >= bowl_cos else "bowl_compositional_template"
        sample_rows.append(
            {
                "latent_key": field,
                "rollout_id": sample["rollout_id"],
                "trial_id": sample["trial_id"],
                "chunk_id": sample["chunk_id"],
                "group": sample["group"],
                "dominant_failure_mode": sample["dominant_failure_mode"],
                "centroid_mode": centroid_mode,
                "cos_success_centroid": success_cos,
                "cos_bowl_centroid": bowl_cos,
                "margin_success_minus_bowl": success_cos - bowl_cos,
                "nearest_template": nearest,
            }
        )

    margin_rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in sample_rows:
        by_key[(row["group"], row["chunk_id"])].append(row["margin_success_minus_bowl"])
    for (group, chunk_id), values in sorted(by_key.items(), key=lambda item: (item[0][0], item[0][1])):
        arr = np.asarray(values, dtype=np.float64)
        margin_rows.append(
            {
                "latent_key": field,
                "centroid_mode": centroid_mode,
                "group": group,
                "chunk_id": chunk_id,
                "mean_margin": float(arr.mean()),
                "std_margin": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "sem_margin": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
                "n": int(len(arr)),
            }
        )
    return sample_rows, margin_rows


def nearest_distribution_rows(sample_rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in sample_rows:
        counters[row["group"]][row["nearest_template"]] += 1

    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        total = sum(counters[group].values())
        for template in ["success_template", "bowl_compositional_template"]:
            count = counters[group][template]
            rows.append(
                {
                    "latent_key": field,
                    "group": group,
                    "nearest_template": template,
                    "count": count,
                    "proportion": float(count / total) if total else 0.0,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_margin_curve(path: Path, margin_rows: list[dict[str, Any]], fields: list[str]) -> None:
    fig, axes = plt.subplots(1, len(fields), figsize=(6.2 * len(fields), 4.3), sharey=True)
    if len(fields) == 1:
        axes = [axes]
    for axis, field in zip(axes, fields):
        for group in GROUPS:
            rows = [r for r in margin_rows if r["latent_key"] == field and r["group"] == group]
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: int(r["chunk_id"]))
            x = [int(r["chunk_id"]) for r in rows]
            y = [float(r["mean_margin"]) for r in rows]
            sem = [float(r["sem_margin"]) for r in rows]
            axis.plot(x, y, marker="o", linewidth=2.2, color=GROUP_COLORS[group], label=GROUP_LABELS[group])
            axis.fill_between(x, np.asarray(y) - sem, np.asarray(y) + sem, color=GROUP_COLORS[group], alpha=0.15)
        axis.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
        axis.set_title(field)
        axis.set_xlabel("early chunk id")
        axis.grid(True, alpha=0.25)
        axis.set_xticks(sorted({int(r["chunk_id"]) for r in margin_rows if r["latent_key"] == field}))
    axes[0].set_ylabel("cos(z, success centroid) - cos(z, bowl centroid)")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Early Template Margin Curve", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_nearest_distribution(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    fig, axes = plt.subplots(1, len(fields), figsize=(5.6 * len(fields), 4.2), sharey=True)
    if len(fields) == 1:
        axes = [axes]
    group_positions = np.arange(len(GROUPS))
    width = 0.34
    colors = {"success_template": "#1f77b4", "bowl_compositional_template": "#d62728"}
    for axis, field in zip(axes, fields):
        field_rows = [r for r in rows if r["latent_key"] == field]
        for offset, template in [(-width / 2, "success_template"), (width / 2, "bowl_compositional_template")]:
            values = []
            for group in GROUPS:
                match = [
                    r for r in field_rows if r["group"] == group and r["nearest_template"] == template
                ]
                values.append(float(match[0]["proportion"]) if match else 0.0)
            axis.bar(group_positions + offset, values, width=width, color=colors[template], label=template)
            for xpos, value in zip(group_positions + offset, values):
                axis.text(
                    xpos,
                    value + 0.025,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        axis.set_title(field)
        axis.set_xticks(group_positions)
        axis.set_xticklabels(
            ["natural success\nrollouts", "natural failure\nrollouts", "intervention\nsuccess rollouts"]
        )
        axis.set_ylim(0.0, 1.12)
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_ylabel("proportion nearest to template")
    axes[-1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.suptitle("Template Assigned by Early Latent Nearest Centroid", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def centroid_pair_for_sample(
    centroid_pairs: dict[int | str, tuple[np.ndarray, np.ndarray]], chunk_id: int
) -> tuple[np.ndarray, np.ndarray]:
    if "global" in centroid_pairs:
        return centroid_pairs["global"]
    if chunk_id in centroid_pairs:
        return centroid_pairs[chunk_id]
    nearest_key = min(
        [key for key in centroid_pairs if isinstance(key, int)],
        key=lambda key: abs(key - chunk_id),
    )
    return centroid_pairs[nearest_key]


def compute_dynamics_rows(
    full_samples: list[dict[str, Any]],
    centroid_pairs: dict[int | str, tuple[np.ndarray, np.ndarray]],
    field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in full_samples:
        success_centroid, bowl_centroid = centroid_pair_for_sample(
            centroid_pairs, sample["chunk_id"]
        )
        success_cos = cosine(sample["vector"], success_centroid)
        bowl_cos = cosine(sample["vector"], bowl_centroid)
        margin = success_cos - bowl_cos
        nearest = "success_template" if margin >= 0 else "bowl_compositional_template"
        row = {
            "latent_key": field,
            "rollout_id": sample["rollout_id"],
            "trial_id": sample["trial_id"],
            "chunk_id": sample["chunk_id"],
            "group": sample["group"],
            "dominant_failure_mode": sample["dominant_failure_mode"],
            "cos_success_centroid": success_cos,
            "cos_bowl_centroid": bowl_cos,
            "margin_success_minus_bowl": margin,
            "nearest_template": nearest,
        }
        sample_rows.append(row)
        grouped[(sample["rollout_id"], sample["trial_id"])].append(row)

    curve_rows: list[dict[str, Any]] = []
    by_group_chunk: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in sample_rows:
        by_group_chunk[(row["group"], row["chunk_id"])].append(row["margin_success_minus_bowl"])
    for (group, chunk_id), values in sorted(by_group_chunk.items(), key=lambda item: (item[0][0], item[0][1])):
        arr = np.asarray(values, dtype=np.float64)
        curve_rows.append(
            {
                "latent_key": field,
                "group": group,
                "chunk_id": chunk_id,
                "mean_margin": float(arr.mean()),
                "std_margin": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "sem_margin": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
                "n": int(len(arr)),
            }
        )

    rollout_rows: list[dict[str, Any]] = []
    for (_, trial_id), rows in sorted(grouped.items(), key=lambda item: item[0][1]):
        rows = sorted(rows, key=lambda row: row["chunk_id"])
        margins = np.asarray([row["margin_success_minus_bowl"] for row in rows], dtype=np.float64)
        signs = np.asarray([1 if value >= 0 else -1 for value in margins], dtype=np.int8)
        switches = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
        negative_fraction = float(np.mean(margins < 0.0)) if len(margins) else 0.0
        positive_fraction = float(np.mean(margins >= 0.0)) if len(margins) else 0.0
        rollout_rows.append(
            {
                "latent_key": field,
                "rollout_id": rows[0]["rollout_id"],
                "trial_id": trial_id,
                "group": rows[0]["group"],
                "dominant_failure_mode": rows[0]["dominant_failure_mode"],
                "num_chunks": len(rows),
                "mean_margin": float(margins.mean()) if len(margins) else 0.0,
                "min_margin": float(margins.min()) if len(margins) else 0.0,
                "max_margin": float(margins.max()) if len(margins) else 0.0,
                "negative_fraction": negative_fraction,
                "positive_fraction": positive_fraction,
                "zero_crossing_count": switches,
                "template_switch_count": switches,
                "persistent_negative": int(negative_fraction >= 0.8),
                "persistent_positive": int(positive_fraction >= 0.8),
            }
        )
    return sample_rows, curve_rows, rollout_rows


def normalized_curve_rows(
    dynamics_rows: list[dict[str, Any]], field: str, num_bins: int
) -> list[dict[str, Any]]:
    by_rollout: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in dynamics_rows:
        if row["latent_key"] == field:
            by_rollout[(row["rollout_id"], int(row["trial_id"]))].append(row)

    bin_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    target_x = np.linspace(0.0, 1.0, num_bins)
    for rows in by_rollout.values():
        rows = sorted(rows, key=lambda row: int(row["chunk_id"]))
        if not rows:
            continue
        margins = np.asarray([float(row["margin_success_minus_bowl"]) for row in rows], dtype=np.float64)
        source_x = np.linspace(0.0, 1.0, len(margins))
        interpolated = np.interp(target_x, source_x, margins)
        group = rows[0]["group"]
        for bin_id, value in enumerate(interpolated):
            bin_values[(group, bin_id)].append(float(value))

    rows: list[dict[str, Any]] = []
    for (group, bin_id), values in sorted(bin_values.items(), key=lambda item: (item[0][0], item[0][1])):
        arr = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "latent_key": field,
                "group": group,
                "time_bin": bin_id,
                "normalized_time": float(target_x[bin_id]),
                "mean_margin": float(arr.mean()),
                "std_margin": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "sem_margin": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
                "n": int(len(arr)),
            }
        )
    return rows


def plot_full_margin_curve(path: Path, curve_rows: list[dict[str, Any]], fields: list[str]) -> None:
    fig, axes = plt.subplots(len(fields), 1, figsize=(9.2, 3.8 * len(fields)), sharex=True)
    if len(fields) == 1:
        axes = [axes]
    for axis, field in zip(axes, fields):
        for group in GROUPS:
            rows = [r for r in curve_rows if r["latent_key"] == field and r["group"] == group]
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: int(r["chunk_id"]))
            x = np.asarray([int(r["chunk_id"]) for r in rows], dtype=np.int64)
            y = np.asarray([float(r["mean_margin"]) for r in rows], dtype=np.float64)
            sem = np.asarray([float(r["sem_margin"]) for r in rows], dtype=np.float64)
            axis.plot(x, y, linewidth=2.0, color=GROUP_COLORS[group], label=GROUP_LABELS[group])
            axis.fill_between(x, y - sem, y + sem, color=GROUP_COLORS[group], alpha=0.15)
        axis.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
        axis.set_title(field)
        axis.set_ylabel("template margin")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, loc="best")
    axes[-1].set_xlabel("chunk id")
    fig.suptitle("Full Rollout Template Margin Curve", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_margin_curve(path: Path, curve_rows: list[dict[str, Any]], fields: list[str]) -> None:
    fig, axes = plt.subplots(len(fields), 1, figsize=(9.2, 3.8 * len(fields)), sharex=True)
    if len(fields) == 1:
        axes = [axes]
    for axis, field in zip(axes, fields):
        for group in GROUPS:
            rows = [r for r in curve_rows if r["latent_key"] == field and r["group"] == group]
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: int(r["time_bin"]))
            x = np.asarray([float(r["normalized_time"]) for r in rows], dtype=np.float64)
            y = np.asarray([float(r["mean_margin"]) for r in rows], dtype=np.float64)
            sem = np.asarray([float(r["sem_margin"]) for r in rows], dtype=np.float64)
            axis.plot(x, y, linewidth=2.0, color=GROUP_COLORS[group], label=GROUP_LABELS[group])
            axis.fill_between(x, y - sem, y + sem, color=GROUP_COLORS[group], alpha=0.15)
        axis.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
        axis.set_title(field)
        axis.set_ylabel("template margin")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, loc="best")
    axes[-1].set_xlabel("normalized rollout time")
    fig.suptitle("Normalized Full Rollout Template Margin Curve", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_margin_heatmap(path: Path, dynamics_rows: list[dict[str, Any]], fields: list[str]) -> None:
    fig, axes = plt.subplots(len(fields), len(GROUPS), figsize=(16.0, 3.6 * len(fields)), sharex=True)
    if len(fields) == 1:
        axes = np.asarray([axes])
    vmax = max(abs(float(row["margin_success_minus_bowl"])) for row in dynamics_rows)
    vmax = max(vmax, 1e-6)
    for row_idx, field in enumerate(fields):
        for col_idx, group in enumerate(GROUPS):
            axis = axes[row_idx, col_idx]
            rows = [r for r in dynamics_rows if r["latent_key"] == field and r["group"] == group]
            if not rows:
                axis.axis("off")
                continue
            trial_ids = sorted({int(r["trial_id"]) for r in rows})
            max_chunk = max(int(r["chunk_id"]) for r in rows)
            matrix = np.full((len(trial_ids), max_chunk + 1), np.nan, dtype=np.float64)
            trial_to_idx = {trial_id: idx for idx, trial_id in enumerate(trial_ids)}
            for row in rows:
                matrix[trial_to_idx[int(row["trial_id"])], int(row["chunk_id"])] = float(
                    row["margin_success_minus_bowl"]
                )
            masked = np.ma.masked_invalid(matrix)
            cmap = plt.get_cmap("coolwarm").copy()
            cmap.set_bad(color="#f0f0f0")
            im = axis.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-vmax, vmax=vmax)
            axis.axvline(4.5, color="#222222", linewidth=0.8, linestyle=":")
            axis.set_title(f"{field}: {GROUP_LABELS[group]}")
            axis.set_ylabel("rollout")
            axis.set_xlabel("chunk id")
            axis.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="template margin")
    fig.suptitle("Rollout Template Margin Heatmap", y=0.99)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def plot_pca(path: Path, all_samples_by_field: dict[str, list[dict[str, Any]]], fields: list[str]) -> None:
    fig, axes = plt.subplots(1, len(fields), figsize=(5.6 * len(fields), 4.4))
    if len(fields) == 1:
        axes = [axes]
    for axis, field in zip(axes, fields):
        samples = all_samples_by_field[field]
        matrix = np.stack([l2_normalize(sample["vector"]) for sample in samples], axis=0)
        xy = pca_2d(matrix)
        for group in GROUPS:
            idx = [i for i, sample in enumerate(samples) if sample["group"] == group]
            if not idx:
                continue
            axis.scatter(
                xy[idx, 0],
                xy[idx, 1],
                s=18,
                alpha=0.75,
                color=GROUP_COLORS[group],
                label=GROUP_LABELS[group],
            )
        axis.set_title(field)
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(True, alpha=0.2)
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Early Chunk PCA Projection", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    centroid_dir = args.output_dir / "centroids"
    centroid_dir.mkdir(parents=True, exist_ok=True)
    for old_centroid in centroid_dir.glob("*.npy"):
        old_centroid.unlink()

    manifest_rows = load_manifest_rows(args.chunk_manifest, args.failure_mode, args.max_early_chunk)
    intervention_early_rows = intervention_success_rows(
        args.intervention_task_root,
        args.intervention_summary,
        max_chunk=args.max_early_chunk,
    )
    early_eval_rows = manifest_rows + intervention_early_rows
    full_manifest_rows = load_full_manifest_rows(args.chunk_manifest, args.failure_mode)
    intervention_full_rows = intervention_success_rows(
        args.intervention_task_root,
        args.intervention_summary,
        max_chunk=None,
    )
    full_eval_rows = full_manifest_rows + intervention_full_rows
    full_curve_rows_input = [
        row for row in full_eval_rows if int(row["chunk_id"]) <= args.full_max_chunk
    ]
    all_sample_rows: list[dict[str, Any]] = []
    all_margin_rows: list[dict[str, Any]] = []
    all_distribution_rows: list[dict[str, Any]] = []
    all_dynamics_rows: list[dict[str, Any]] = []
    all_dynamics_curve_rows: list[dict[str, Any]] = []
    all_normalized_curve_rows: list[dict[str, Any]] = []
    all_rollout_dynamics_rows: list[dict[str, Any]] = []
    samples_by_field: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {
        "chunk_manifest": args.chunk_manifest.as_posix(),
        "output_dir": args.output_dir.as_posix(),
        "fields": args.fields,
        "failure_mode": args.failure_mode,
        "max_early_chunk": args.max_early_chunk,
        "full_max_chunk": args.full_max_chunk,
        "normalized_bins": args.normalized_bins,
        "centroid_mode": args.centroid_mode,
        "num_manifest_rows_selected": len(manifest_rows),
        "num_intervention_early_rows_selected": len(intervention_early_rows),
        "num_full_manifest_rows_selected": len(full_manifest_rows),
        "num_intervention_full_rows_selected": len(intervention_full_rows),
        "counts": {},
    }
    payload_cache: dict[Path, dict[str, Any]] = {}

    for field in args.fields:
        natural_early_samples = load_field_vectors_with_cache(manifest_rows, field, payload_cache)
        eval_early_samples = load_field_vectors_with_cache(early_eval_rows, field, payload_cache)
        samples_by_field[field] = eval_early_samples
        centroid_pairs, sample_rows, margin_rows = compute_field_results(
            natural_early_samples, field, args.centroid_mode
        )
        eval_sample_rows, eval_margin_rows = score_samples_with_centroids(
            eval_early_samples, field, centroid_pairs, args.centroid_mode
        )
        for key, (success_centroid, bowl_centroid) in centroid_pairs.items():
            suffix = "global" if key == "global" else f"chunk_{int(key):03d}"
            np.save(centroid_dir / f"{field}_{suffix}_success_centroid.npy", success_centroid)
            np.save(
                centroid_dir / f"{field}_{suffix}_bowl_compositional_centroid.npy",
                bowl_centroid,
            )
        distribution_rows = nearest_distribution_rows(eval_sample_rows, field)
        full_samples = load_field_vectors_with_cache(full_eval_rows, field, payload_cache)
        dynamics_rows, dynamics_curve_rows, rollout_dynamics_rows = compute_dynamics_rows(
            full_samples, centroid_pairs, field
        )
        full_curve_samples = load_field_vectors_with_cache(full_curve_rows_input, field, payload_cache)
        _, limited_dynamics_curve_rows, _ = compute_dynamics_rows(
            full_curve_samples, centroid_pairs, field
        )
        norm_rows = normalized_curve_rows(dynamics_rows, field, args.normalized_bins)
        all_sample_rows.extend(eval_sample_rows)
        all_margin_rows.extend(eval_margin_rows)
        all_distribution_rows.extend(distribution_rows)
        all_dynamics_rows.extend(dynamics_rows)
        all_dynamics_curve_rows.extend(limited_dynamics_curve_rows)
        all_rollout_dynamics_rows.extend(rollout_dynamics_rows)
        all_normalized_curve_rows.extend(norm_rows)
        rollout_group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rollout_dynamics_rows:
            rollout_group_rows[row["group"]].append(row)
        summary["counts"][field] = {
            "num_samples": len(eval_early_samples),
            "num_success_samples": sum(s["group"] == "natural_success" for s in eval_early_samples),
            "num_failure_samples": sum(s["group"] == "natural_failure" for s in eval_early_samples),
            "num_intervention_recovered_samples": sum(
                s["group"] == "intervention_recovered_success" for s in eval_early_samples
            ),
            "num_success_rollouts": len({s["trial_id"] for s in eval_early_samples if s["group"] == "natural_success"}),
            "num_failure_rollouts": len({s["trial_id"] for s in eval_early_samples if s["group"] == "natural_failure"}),
            "num_intervention_recovered_rollouts": len(
                {s["trial_id"] for s in eval_early_samples if s["group"] == "intervention_recovered_success"}
            ),
            "nearest_distribution": distribution_rows,
            "dynamics_by_group": {
                group: {
                    "mean_template_switch_count": float(
                        np.mean([r["template_switch_count"] for r in rows])
                    )
                    if rows
                    else 0.0,
                    "mean_negative_fraction": float(
                        np.mean([r["negative_fraction"] for r in rows])
                    )
                    if rows
                    else 0.0,
                    "persistent_negative_rollouts": int(
                        sum(int(r["persistent_negative"]) for r in rows)
                    ),
                    "persistent_positive_rollouts": int(
                        sum(int(r["persistent_positive"]) for r in rows)
                    ),
                    "num_rollouts": len(rows),
                }
                for group, rows in rollout_group_rows.items()
            },
        }

    write_csv(args.output_dir / "sample_margins.csv", all_sample_rows)
    write_csv(args.output_dir / "template_margin_by_chunk.csv", all_margin_rows)
    write_csv(args.output_dir / "nearest_template_distribution.csv", all_distribution_rows)
    write_csv(args.output_dir / "full_rollout_sample_margins.csv", all_dynamics_rows)
    write_csv(args.output_dir / "full_rollout_margin_by_chunk.csv", all_dynamics_curve_rows)
    write_csv(args.output_dir / "normalized_full_rollout_margin_by_bin.csv", all_normalized_curve_rows)
    write_csv(args.output_dir / "rollout_dynamics_summary.csv", all_rollout_dynamics_rows)
    plot_margin_curve(args.output_dir / "template_margin_curve.png", all_margin_rows, args.fields)
    plot_nearest_distribution(
        args.output_dir / "nearest_template_distribution.png", all_distribution_rows, args.fields
    )
    plot_full_margin_curve(
        args.output_dir / "full_rollout_margin_curve.png", all_dynamics_curve_rows, args.fields
    )
    plot_normalized_margin_curve(
        args.output_dir / "normalized_full_rollout_margin_curve.png",
        all_normalized_curve_rows,
        args.fields,
    )
    plot_rollout_margin_heatmap(
        args.output_dir / "rollout_margin_heatmap.png", all_dynamics_rows, args.fields
    )
    plot_pca(args.output_dir / "pca_scatter.png", samples_by_field, args.fields)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote template centroid analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
