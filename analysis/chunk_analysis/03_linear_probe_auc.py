"""
Chunk-wise linear-probe AUC heatmap.

Purpose
-------
For every normalized-time chunk and hidden state, train a logistic-regression
linear probe to classify success vs failure from that embedding. Each original
trajectory is first linearly interpolated onto tau in [0, 1]. The script
reports cross-validated AUC for each chunk/state pair and also for the fused
concat representation.

Parameters
----------
--root: chunk-wise task output directory containing trial_* folders.
--out_dir: directory where CSV/NPY/PNG outputs are saved.
--n_success: number of successful trials to use.
--n_failure: number of failed trials to use.
--target_length: number of normalized-time chunks after interpolation.
--fields: optional state field names to analyze. Defaults to manifest fields.
--cv: number of stratified folds. With 20/20, 5 is the default.

Usage
-----
python analysis/chunk_analysis/03_linear_probe_auc.py \
    --root OOD_exp/dif_start_end_loc/outputs/chunk_wise/put_the_cream_cheese_on_the_plate \
    --out_dir analysis/chunk_analysis/results/put_the_cream_cheese_on_the_plate \
    --cv 5

Outputs
-------
Saved under --out_dir/linear_probe_auc:
  auc_statewise.csv/png     AUC in R^{C x 5}
  auc_fused.csv/png         fused concat AUC in R^{C x 1}
  probe_metadata.json       selected trials, chunk ids, and CV settings
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chunk_analysis_common import (
    DEFAULT_ROOT,
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


def probe_auc(x: np.ndarray, y: np.ndarray, cv: int, random_state: int) -> float:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import LeaveOneOut, StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if cv <= 1 or cv >= len(y):
            splitter = LeaveOneOut()
        else:
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = np.zeros(len(y), dtype=np.float64)
        for train_idx, test_idx in splitter.split(x, y):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="liblinear"),
            )
            clf.fit(x[train_idx], y[train_idx])
            scores[test_idx] = clf.predict_proba(x[test_idx])[:, 1]
        return float(roc_auc_score(y, scores))
    except ImportError:
        return torch_probe_auc(x, y, cv, random_state)


def stratified_splits(y: np.ndarray, cv: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(random_state)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_folds = len(y) if cv <= 1 or cv >= len(y) else cv
    if n_folds >= len(y):
        return [(np.setdiff1d(np.arange(len(y)), [i]), np.asarray([i])) for i in range(len(y))]
    pos_folds = np.array_split(pos, n_folds)
    neg_folds = np.array_split(neg, n_folds)
    splits = []
    all_idx = np.arange(len(y))
    for p_fold, n_fold in zip(pos_folds, neg_folds):
        test_idx = np.concatenate([p_fold, n_fold])
        train_idx = np.setdiff1d(all_idx, test_idx)
        splits.append((train_idx, test_idx))
    return splits


def binary_auc(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def torch_probe_auc(x: np.ndarray, y: np.ndarray, cv: int, random_state: int) -> float:
    import torch

    scores = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in stratified_splits(y, cv, random_state):
        x_train = x[train_idx].astype(np.float32, copy=False)
        y_train = y[train_idx].astype(np.float32, copy=False)
        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        x_train = (x_train - mean) / std
        x_test = (x[test_idx].astype(np.float32, copy=False) - mean) / std

        torch.manual_seed(random_state)
        xt = torch.from_numpy(x_train)
        yt = torch.from_numpy(y_train)
        model = torch.nn.Linear(xt.shape[1], 1)
        opt = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=80, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            logits = model(xt).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
            l2 = sum((p * p).sum() for p in model.parameters())
            loss = loss + 1e-4 * l2
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(x_test)).squeeze(1)).cpu().numpy()
        scores[test_idx] = probs
    return binary_auc(y, scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--n_success", type=int, default=20)
    parser.add_argument("--n_failure", type=int, default=20)
    parser.add_argument("--target_length", type=int, default=20)
    parser.add_argument("--fields", nargs="*", default=None)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=0)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (args.out_dir / "linear_probe_auc").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = args.fields or load_manifest_fields(root)

    trials = discover_trials(root)
    success_trials, failure_trials = select_balanced_trials(trials, args.n_success, args.n_failure)
    selected = success_trials + failure_trials

    success_states, state_names = load_group_normalized_state_tensors(success_trials, args.target_length, fields)
    failure_states, _ = load_group_normalized_state_tensors(failure_trials, args.target_length, fields)
    y = np.asarray([1] * len(success_trials) + [0] * len(failure_trials), dtype=np.int64)

    c_count, s_count = args.target_length, len(state_names)
    auc = np.zeros((c_count, s_count), dtype=np.float64)
    state_shapes = {}
    for s, state in enumerate(state_names):
        z_state = np.concatenate([success_states[state], failure_states[state]], axis=0)
        state_shapes[state] = list(z_state.shape)
        for c in range(c_count):
            auc[c, s] = probe_auc(z_state[:, c, :], y, args.cv, args.random_state)

    z_fused = np.concatenate(
        [
            make_fused_from_states(success_states, state_names),
            make_fused_from_states(failure_states, state_names),
        ],
        axis=0,
    )
    auc_fused = np.zeros((c_count, 1), dtype=np.float64)
    for c in range(c_count):
        auc_fused[c, 0] = probe_auc(z_fused[:, c, :], y, args.cv, args.random_state)

    save_matrix_csv(out_dir / "auc_statewise.csv", auc, "chunk_position", state_names)
    save_matrix_csv(out_dir / "auc_fused.csv", auc_fused, "chunk_position", ["fused_concat"])
    heatmap(out_dir / "auc_statewise.png", auc, "Linear Probe AUC", "state", "normalized time index", state_names)
    heatmap(out_dir / "auc_fused.png", auc_fused, "Linear Probe AUC (Fused)", "representation", "normalized time index", ["fused_concat"])
    write_json(
        out_dir / "probe_metadata.json",
        {
            "root": str(root),
            "fields": fields,
            "state_names": state_names,
            "selected_success_trials": trial_ids(success_trials),
            "selected_failure_trials": trial_ids(failure_trials),
            "target_length": args.target_length,
            "normalized_tau": normalized_time_grid(args.target_length).tolist(),
            "original_chunk_counts": {str(t.trial_id): t.num_chunk_files for t in selected},
            "cv": args.cv,
            "random_state": args.random_state,
            "state_shapes": state_shapes,
            "fused_shape": list(z_fused.shape),
        },
    )
    print(f"Wrote linear-probe AUC outputs to {out_dir}")


if __name__ == "__main__":
    main()
