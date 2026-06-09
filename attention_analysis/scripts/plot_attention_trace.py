"""
Plot LIBERO / pi0.5 attention tracing summaries.

Purpose
-------
Load attention_summary.parquet files produced by trace_attention_libero.py and
generate diagnostic plots for text sinks, object-word attention, target-object
visual patch mass, visual heatmaps, and cross-prompt visual distribution
distance.

Parameters
----------
--trace_root: root directory containing scene_id=*/seed=*/prompt=*/mode=*/ runs.
--output_dir: directory receiving plot PNG files.
--token_map_path: optional token_map.json for heatmap layout. If omitted, the
                  script uses token_map.json files found in each run directory.

Usage
-----
python attention_analysis/scripts/plot_attention_trace.py \
  --trace_root attention_analysis/outputs/attention_trace \
  --output_dir attention_analysis/outputs/attention_trace_plots

Outputs
-------
PNG figures are saved under --output_dir. The script also writes
plot_manifest.json with counts and source files.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_run_path(path: pathlib.Path, root: pathlib.Path) -> dict[str, str]:
    rel = path.relative_to(root)
    parts = rel.parts
    metadata = {"scene_id": "", "seed": "", "prompt": "", "mode": ""}
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            metadata[key] = value
    if len(parts) >= 2:
        metadata["mode"] = parts[-2]
    return metadata


def load_summaries(trace_root: pathlib.Path) -> pd.DataFrame:
    frames = []
    for summary_path in trace_root.rglob("attention_summary.parquet"):
        frame = pd.read_parquet(summary_path)
        metadata = parse_run_path(summary_path, trace_root)
        for key, value in metadata.items():
            frame[key] = value
        frame["run_dir"] = str(summary_path.parent)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No attention_summary.parquet files found under {trace_root}")
    return pd.concat(frames, ignore_index=True)


def plot_metric_by_layer(df: pd.DataFrame, metric: str, output_path: pathlib.Path, title: str) -> None:
    grouped = (
        df.groupby(["mode", "prompt", "layer"], dropna=False)[metric]
        .mean()
        .reset_index()
        .sort_values(["mode", "prompt", "layer"])
    )
    plt.figure(figsize=(9, 5))
    for (mode, prompt), sub in grouped.groupby(["mode", "prompt"]):
        plt.plot(sub["layer"], sub[metric], marker="o", label=f"{prompt} / {mode}")
    plt.xlabel("layer")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log((p + eps) / (m + eps))) + 0.5 * np.sum(q * np.log((q + eps) / (m + eps))))


def plot_cross_prompt_distance(df: pd.DataFrame, output_path: pathlib.Path) -> None:
    value_col = "target_object_patch_mass"
    rows = []
    grouped = df.groupby(["scene_id", "seed", "mode", "layer"], dropna=False)
    for (scene_id, seed, mode, layer), sub in grouped:
        prompts = sorted(sub["prompt"].unique())
        if len(prompts) < 2:
            continue
        prompt_a, prompt_b = prompts[:2]
        a = sub[sub["prompt"] == prompt_a][value_col].to_numpy()
        b = sub[sub["prompt"] == prompt_b][value_col].to_numpy()
        n = min(len(a), len(b))
        if n == 0:
            continue
        rows.append({"scene_id": scene_id, "seed": seed, "mode": mode, "layer": layer, "js_divergence": js_divergence(a[:n], b[:n])})
    if not rows:
        return
    out = pd.DataFrame(rows).groupby(["mode", "layer"])["js_divergence"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    for mode, sub in out.groupby("mode"):
        plt.plot(sub["layer"], sub["js_divergence"], marker="o", label=mode)
    plt.xlabel("layer")
    plt.ylabel("JS divergence")
    plt.title("Cross-prompt visual distribution distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def read_token_map(run_dir: pathlib.Path) -> list[dict[str, Any]]:
    path = run_dir / "token_map.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["token_map"] if isinstance(data, dict) else data


def plot_heatmaps(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    for run_dir, sub in df.groupby("run_dir"):
        run_path = pathlib.Path(run_dir)
        try:
            token_map = read_token_map(run_path)
        except FileNotFoundError:
            continue
        image_entries = [entry for entry in token_map if entry.get("type") == "image"]
        if not image_entries:
            continue
        best = sub.sort_values(["denoise_step", "layer"]).tail(1)
        if best.empty:
            continue
        label = re.sub(r"[^a-zA-Z0-9_=.-]+", "_", str(run_path))
        for camera in sorted({entry.get("camera") for entry in image_entries}):
            cam_entries = [entry for entry in image_entries if entry.get("camera") == camera]
            rows = max(int(entry.get("patch_row", 0)) for entry in cam_entries) + 1
            cols = max(int(entry.get("patch_col", 0)) for entry in cam_entries) + 1
            grid = np.zeros((rows, cols), dtype=np.float64)
            for entry in cam_entries:
                value = float(best["target_object_patch_mass"].iloc[0])
                grid[int(entry.get("patch_row", 0)), int(entry.get("patch_col", 0))] = value
            plt.figure(figsize=(5, 4))
            plt.imshow(grid, cmap="viridis")
            plt.colorbar(label="target_object_patch_mass")
            plt.title(f"{camera} target patch mass")
            plt.tight_layout()
            plt.savefig(output_dir / f"heatmap_{label}_{camera}.png", dpi=180)
            plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pi0.5 attention trace summaries.")
    parser.add_argument("--trace_root", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)
    parser.add_argument("--token_map_path", type=pathlib.Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_summaries(args.trace_root)
    plot_metric_by_layer(df, "text_sink_mass", args.output_dir / "text_sink_mass_by_layer.png", "Text sink mass by layer")
    plot_metric_by_layer(df, "target_object_word_mass", args.output_dir / "object_word_attention_by_layer.png", "Object word attention by layer")
    plot_metric_by_layer(df, "target_object_patch_mass", args.output_dir / "target_object_patch_mass_by_layer.png", "Target object visual patch mass")
    plot_cross_prompt_distance(df, args.output_dir / "cross_prompt_visual_js_divergence.png")
    plot_heatmaps(df, args.output_dir)
    (args.output_dir / "plot_manifest.json").write_text(
        json.dumps({"rows": int(len(df)), "runs": int(df["run_dir"].nunique())}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
