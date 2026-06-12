"""
Purpose
-------
Plot per-action-token top-k attention records for pi0.5 attention traces.

For every selected chunk, this script reads attention_topk.jsonl and creates one
figure for each denoise step / action token / layer. Each figure places all
attention heads side by side so the top-k key-token preference of different
heads can be compared directly.

Arguments
---------
--trace_root:
  Root directory containing attention traces.
--task_name, --trial_id, --chunk_id, --mode:
  Optional filters for selecting trace chunks.
--max_files:
  Maximum number of attention_topk.jsonl files to process. Use 0 for all.
--max_heads_per_row:
  Number of head panels per figure row.
--max_label_chars:
  Maximum label length for token labels in the plots.

Usage
-----
python attention_analysis/scripts/attention_topk_token_analysis.py \\
    --trace_root attention_analysis/outputs/attention_trace \\
    --task_name open_the_middle_drawer_of_the_cabinet \\
    --trial_id trial_0 \\
    --chunk_id chunk_00000 \\
    --mode baseline \\
    --max_files 1

Outputs
-------
For each selected chunk directory:

  <chunk>/<mode>/action_token_attention_analysis/
    denoise_step_<num>/
      action_token_<num>/
        layer_<num>.png
        layer_<num>.csv
    analysis_manifest.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import pathlib
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-action-token top-k attention by denoise step/layer/head.")
    parser.add_argument("--trace_root", type=pathlib.Path, required=True)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--trial_id", type=str, default=None)
    parser.add_argument("--chunk_id", type=str, default=None)
    parser.add_argument("--mode", type=str, default="baseline")
    parser.add_argument("--max_files", type=int, default=50)
    parser.add_argument("--max_heads_per_row", type=int, default=4)
    parser.add_argument("--max_label_chars", type=int, default=42)
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


def token_label(item: dict[str, Any], *, max_chars: int) -> str:
    key_type = str(item.get("key_type", "unknown"))
    key_position = item.get("key_position", "?")
    if key_type == "image":
        camera = item.get("camera") or "camera"
        row = item.get("patch_row")
        col = item.get("patch_col")
        text = f"{key_type}:{camera}[{row},{col}]@{key_position}"
    else:
        token_text = item.get("token_text") or item.get("token_str") or ""
        token_text = str(token_text).replace("\n", "\\n")
        role = item.get("special_role")
        if role:
            text = f"{key_type}:{role}:{token_text}@{key_position}"
        elif token_text:
            text = f"{key_type}:{token_text}@{key_position}"
        else:
            text = f"{key_type}@{key_position}"
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def csv_row(source_path: pathlib.Path, record: dict[str, Any], rank: int, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_file": str(source_path),
        "denoise_step": int(record.get("denoise_step", -1)),
        "layer": int(record.get("layer", -1)),
        "head": int(record.get("head", -1)),
        "query_position": int(record.get("query_position", -1)),
        "query_index": int(record.get("query_index", -1)),
        "rank": int(rank),
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


def write_rows(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_layer_heads(
    *,
    rows_by_head: dict[int, list[dict[str, Any]]],
    output_path: pathlib.Path,
    title: str,
    max_heads_per_row: int,
    max_label_chars: int,
) -> None:
    heads = sorted(rows_by_head)
    if not heads:
        return

    cols = max(1, min(max_heads_per_row, len(heads)))
    rows = int(math.ceil(len(heads) / cols))
    fig_width = max(5.0 * cols, 8.0)
    fig_height = max(3.2 * rows, 4.0)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)

    all_attention = [float(row["attention"]) for head_rows in rows_by_head.values() for row in head_rows]
    x_max = max(all_attention) if all_attention else 1.0
    x_max = max(x_max * 1.08, 1e-6)

    for ax in axes.ravel():
        ax.axis("off")

    for panel_idx, head in enumerate(heads):
        ax = axes[panel_idx // cols][panel_idx % cols]
        head_rows = sorted(rows_by_head[head], key=lambda row: int(row["rank"]))
        labels = [token_label(row, max_chars=max_label_chars) for row in head_rows][::-1]
        values = [float(row["attention"]) for row in head_rows][::-1]
        colors = [key_type_color(str(row.get("key_type", "unknown"))) for row in head_rows][::-1]

        ax.axis("on")
        ax.barh(range(len(values)), values, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("attention", fontsize=8)
        ax.set_title(f"head {head}", fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def key_type_color(key_type: str) -> str:
    return {
        "image": "#4C78A8",
        "prompt": "#59A14F",
        "state": "#F28E2B",
        "special": "#E15759",
        "continuous_action": "#B07AA1",
        "fast_action": "#B07AA1",
        "proprio": "#76B7B2",
        "text": "#59A14F",
        "text_padding": "#9D9D9D",
    }.get(key_type, "#9D9D9D")


def process_file(path: pathlib.Path, args: argparse.Namespace) -> dict[str, Any]:
    grouped: dict[tuple[int, int, int], dict[int, list[dict[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    records_read = 0
    topk_rows = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records_read += 1
            record = json.loads(line)
            denoise_step = int(record.get("denoise_step", -1))
            action_token = int(record.get("query_position", -1))
            layer = int(record.get("layer", -1))
            head = int(record.get("head", -1))
            topk = record.get("topk") or []
            for rank, item in enumerate(topk, start=1):
                grouped[(denoise_step, action_token, layer)][head].append(csv_row(path, record, rank, item))
                topk_rows += 1

    output_root = path.parent / "action_token_attention_analysis"
    figures = 0
    csv_files = 0
    for (denoise_step, action_token, layer), rows_by_head in sorted(grouped.items()):
        out_dir = output_root / f"denoise_step_{denoise_step}" / f"action_token_{action_token}"
        out_dir.mkdir(parents=True, exist_ok=True)
        flat_rows = [row for head in sorted(rows_by_head) for row in rows_by_head[head]]
        csv_path = out_dir / f"layer_{layer}.csv"
        write_rows(csv_path, flat_rows)
        csv_files += 1

        title = (
            f"{path.parent.parent.name}/{path.parent.name} "
            f"denoise_step={denoise_step} action_token={action_token} layer={layer}"
        )
        plot_layer_heads(
            rows_by_head=rows_by_head,
            output_path=out_dir / f"layer_{layer}.png",
            title=title,
            max_heads_per_row=args.max_heads_per_row,
            max_label_chars=args.max_label_chars,
        )
        figures += 1

    manifest = {
        "source_file": str(path),
        "output_root": str(output_root),
        "records_read": records_read,
        "topk_rows": topk_rows,
        "figures": figures,
        "csv_files": csv_files,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifests = [process_file(path, args) for path in selected_files(args)]
    print(json.dumps({"processed_files": len(manifests), "manifests": manifests}, indent=2))


if __name__ == "__main__":
    main()
