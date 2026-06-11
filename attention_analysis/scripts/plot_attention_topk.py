"""
Purpose
-------
Plot action-token attention distributions from pi0.5 attention_topk.jsonl files.

This script is intended for the attention_analysis outputs produced by
start_attention_server.py. It reads top-k attention records for continuous action
queries and creates compact diagnostic figures without loading full attention
matrices or all parquet summary files into memory.

Arguments
---------
--trace_root:
  Root directory containing attention traces, usually
  attention_analysis/outputs/attention_trace.
--output_dir:
  Directory where PNG and CSV outputs are saved.
--task_name, --trial_id, --chunk_id, --mode:
  Optional filters for selecting a specific task/trial/chunk/mode.
--layer, --head, --query_position, --denoise_step:
  Optional filters for a single token-level bar plot.
--max_files:
  Maximum number of attention_topk.jsonl files to read. Use 0 to read all.

Usage
-----
python attention_analysis/scripts/plot_attention_topk.py \\
    --trace_root attention_analysis/outputs/attention_trace \\
    --output_dir attention_analysis/outputs/attention_topk_plots \\
    --task_name put_the_wine_bottle_on_top_of_the_cabinet \\
    --trial_id trial_0 \\
    --chunk_id chunk_00000 \\
    --layer 0 \\
    --head 0 \\
    --query_position 0 \\
    --denoise_step 0

Outputs
-------
The script writes:
  topk_key_type_mass_by_layer.png
  top1_attention_by_layer.png
  selected_token_topk_bar.png
  topk_key_type_mass_by_layer.csv
  top1_attention_by_layer.csv
  selected_token_topk.csv
  plot_manifest.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pathlib
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v2 as imageio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pi0.5 action-token top-k attention traces.")
    parser.add_argument("--trace_root", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--trial_id", type=str, default=None)
    parser.add_argument("--chunk_id", type=str, default=None)
    parser.add_argument("--mode", type=str, default="baseline")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--query_position", type=int, default=None)
    parser.add_argument("--denoise_step", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=50)
    return parser.parse_args()


def selected_files(args: argparse.Namespace) -> list[pathlib.Path]:
    root = args.trace_root.expanduser().resolve()
    files = sorted(root.rglob("attention_topk.jsonl"))

    def keep(path: pathlib.Path) -> bool:
        parts = path.parts
        if args.task_name is not None and args.task_name not in parts:
            return False
        if args.trial_id is not None and args.trial_id not in parts:
            return False
        if args.chunk_id is not None and args.chunk_id not in parts:
            return False
        if args.mode is not None and path.parent.name != args.mode:
            return False
        return True

    filtered = [path for path in files if keep(path)]
    if args.max_files > 0:
        filtered = filtered[: args.max_files]
    if not filtered:
        raise FileNotFoundError(f"No attention_topk.jsonl files matched under {root}")
    return filtered


def record_matches(record: dict[str, Any], args: argparse.Namespace) -> bool:
    for field, expected in (
        ("layer", args.layer),
        ("head", args.head),
        ("query_position", args.query_position),
        ("denoise_step", args.denoise_step),
    ):
        if expected is not None and int(record.get(field, -1)) != expected:
            return False
    return True


def stream_records(paths: list[pathlib.Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield path, json.loads(line)


def write_rows(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_key_type_mass(rows: list[dict[str, Any]], output_path: pathlib.Path) -> None:
    layers = sorted({row["layer"] for row in rows})
    key_types = sorted({row["key_type"] for row in rows})
    plt.figure(figsize=(9, 5))
    for key_type in key_types:
        values = []
        for layer in layers:
            layer_rows = [row for row in rows if row["layer"] == layer and row["key_type"] == key_type]
            values.append(sum(row["attention_sum"] for row in layer_rows) / max(sum(row["record_count"] for row in layer_rows), 1))
        plt.plot(layers, values, marker="o", label=key_type)
    plt.xlabel("Layer")
    plt.ylabel("Mean top-k attention mass")
    plt.title("Action-token top-k attention by key type")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_top1(rows: list[dict[str, Any]], output_path: pathlib.Path) -> None:
    layers = sorted({row["layer"] for row in rows})
    values = []
    for layer in layers:
        layer_rows = [row for row in rows if row["layer"] == layer]
        values.append(sum(row["attention_sum"] for row in layer_rows) / max(sum(row["record_count"] for row in layer_rows), 1))
    plt.figure(figsize=(8, 5))
    plt.plot(layers, values, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Mean top-1 attention")
    plt.title("Action-token strongest attended token by layer")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_selected_token(rows: list[dict[str, Any]], output_path: pathlib.Path) -> None:
    if not rows:
        return
    labels = [f"{row['rank']}:{row['key_type']}:{row['key_position']}" for row in rows]
    values = [row["attention"] for row in rows]
    plt.figure(figsize=(max(10, len(rows) * 0.5), 5))
    plt.bar(range(len(rows)), values)
    plt.xticks(range(len(rows)), labels, rotation=75, ha="right", fontsize=8)
    plt.ylabel("Attention")
    plt.title("Selected action token top-k attention")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_selected_image_overlays(rows: list[dict[str, Any]], output_dir: pathlib.Path) -> None:
    image_rows = [
        row
        for row in rows
        if row.get("key_type") == "image" and row.get("raw_image_path") and row.get("patch_box_xyxy")
    ]
    if not image_rows:
        return
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in image_rows:
        source_dir = pathlib.Path(str(row["source_file"])).parent
        raw_image_path = source_dir / str(row["raw_image_path"])
        grouped[(str(raw_image_path), str(row.get("camera") or "camera"))].append(row)

    for (image_path, camera), camera_rows in grouped.items():
        path = pathlib.Path(image_path)
        if not path.exists():
            continue
        image = imageio.imread(path)
        rows_sorted = sorted(camera_rows, key=lambda row: float(row["attention"]), reverse=True)
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        ax = plt.gca()
        for row in rows_sorted[:20]:
            x0, y0, x1, y1 = [float(value) for value in row["patch_box_xyxy"]]
            alpha = min(0.9, 0.25 + float(row["attention"]) * 8.0)
            rect = patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=1.8,
                edgecolor="red",
                facecolor="red",
                alpha=alpha,
            )
            ax.add_patch(rect)
            ax.text(
                x0,
                max(0.0, y0 - 2.0),
                f"{int(row['rank'])}:{float(row['attention']):.3f}",
                color="white",
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 1},
            )
        plt.axis("off")
        plt.title(f"Selected action token top-k image patches: {camera}")
        plt.tight_layout()
        safe_camera = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in camera)
        plt.savefig(output_dir / f"selected_token_image_overlay_{safe_camera}.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = selected_files(args)

    type_sums: dict[tuple[int, str], list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    top1_sums: dict[int, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    selected_rows: list[dict[str, Any]] = []
    records_read = 0

    for source_path, record in stream_records(paths):
        records_read += 1
        layer = int(record["layer"])
        topk = record.get("topk") or []
        if topk:
            top1 = float(topk[0].get("attention", 0.0))
            top1_sums[layer][0] += top1
            top1_sums[layer][1] += 1
        for item in topk:
            key = (layer, str(item.get("key_type", "unknown")))
            type_sums[key][0] += float(item.get("attention", 0.0))
            type_sums[key][1] += 1

        if record_matches(record, args):
            for rank, item in enumerate(topk, start=1):
                selected_rows.append(
                    {
                        "source_file": str(source_path),
                        "denoise_step": int(record.get("denoise_step", -1)),
                        "layer": layer,
                        "head": int(record.get("head", -1)),
                        "query_position": int(record.get("query_position", -1)),
                        "query_index": int(record.get("query_index", -1)),
                        "rank": rank,
                        "key_position": int(item.get("key_position", -1)),
                        "key_index": int(item.get("key_index", -1)),
                        "key_type": item.get("key_type", "unknown"),
                        "token_str": item.get("token_str"),
                        "token_text": item.get("token_text"),
                        "special_role": item.get("special_role"),
                        "camera": item.get("camera"),
                        "patch_row": item.get("patch_row"),
                        "patch_col": item.get("patch_col"),
                        "patch_box_xyxy": item.get("patch_box_xyxy"),
                        "raw_image_path": item.get("raw_image_path"),
                        "raw_image_height": item.get("raw_image_height"),
                        "raw_image_width": item.get("raw_image_width"),
                        "attention": float(item.get("attention", 0.0)),
                    }
                )

    type_rows = [
        {
            "layer": layer,
            "key_type": key_type,
            "attention_sum": values[0],
            "record_count": int(values[1]),
            "mean_attention": values[0] / max(values[1], 1),
        }
        for (layer, key_type), values in sorted(type_sums.items())
    ]
    top1_rows = [
        {
            "layer": layer,
            "attention_sum": values[0],
            "record_count": int(values[1]),
            "mean_attention": values[0] / max(values[1], 1),
        }
        for layer, values in sorted(top1_sums.items())
    ]

    write_rows(args.output_dir / "topk_key_type_mass_by_layer.csv", type_rows)
    write_rows(args.output_dir / "top1_attention_by_layer.csv", top1_rows)
    write_rows(args.output_dir / "selected_token_topk.csv", selected_rows)
    if type_rows:
        plot_key_type_mass(type_rows, args.output_dir / "topk_key_type_mass_by_layer.png")
    if top1_rows:
        plot_top1(top1_rows, args.output_dir / "top1_attention_by_layer.png")
    if selected_rows:
        plot_selected_token(selected_rows[:50], args.output_dir / "selected_token_topk_bar.png")
        plot_selected_image_overlays(selected_rows, args.output_dir)

    (args.output_dir / "plot_manifest.json").write_text(
        json.dumps(
            {
                "files_read": len(paths),
                "records_read": records_read,
                "selected_rows": len(selected_rows),
                "trace_files": [str(path) for path in paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
