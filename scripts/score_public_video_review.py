"""Score blinded human progress reviews against the proxy labels."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics import cohen_kappa_score


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _bootstrap_accuracy(y_true, y_pred, n=2000, seed=2027):
    if len(y_true) == 0:
        return None
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    values = [
        accuracy_score(y_true[ix], y_pred[ix])
        for ix in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))
    ]
    return [float(x) for x in np.percentile(values, [2.5, 50.0, 97.5])]


def _proxy_metrics(rows):
    if not rows:
        return None
    proxy = np.asarray([r["proxy"] for r in rows], dtype=int)
    human = np.asarray([r["human"] for r in rows], dtype=int)
    has_both_human_classes = len(np.unique(human)) > 1
    return {
        "n": int(len(rows)),
        "human_positive_rate": float(human.mean()),
        "proxy_positive_rate": float(proxy.mean()),
        "majority_human_baseline_accuracy": float(max(human.mean(), 1.0 - human.mean())),
        "accuracy": float(accuracy_score(proxy, human)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(human, proxy))
            if has_both_human_classes
            else None
        ),
        "f1": float(f1_score(proxy, human)) if len(np.unique(proxy)) > 1 else None,
        "auroc": float(roc_auc_score(human, proxy)) if has_both_human_classes else None,
        "confusion_matrix_human_rows_proxy_cols": confusion_matrix(
            human, proxy, labels=[0, 1]
        ).astype(int).tolist(),
        "accuracy_ci95": _bootstrap_accuracy(human, proxy),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--revealed", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    reviews = _read_csv(args.reviews)
    revealed = {x["review_id"]: x for x in json.loads(Path(args.revealed).read_text())["items"]}
    usable = []
    for row in reviews:
        try:
            label = int(row.get("human_label", ""))
        except ValueError:
            continue
        if label not in (0, 1) or row["review_id"] not in revealed:
            continue
        proxy = revealed[row["review_id"]].get("proxy_label")
        if proxy is None:
            continue
        usable.append({**row, "human": label, "proxy": int(proxy)})

    by_item = defaultdict(dict)
    for row in usable:
        by_item[row["review_id"]][row["rater"]] = row
    shared = [v for v in by_item.values() if len(v) >= 2]
    pairs = []
    for item in shared:
        values = list(item.values())
        pairs.append((values[0]["human"], values[1]["human"]))
    consensus = []
    disagreement_items = 0
    for item in by_item.values():
        values = list(item.values())
        labels = {value["human"] for value in values}
        if len(labels) == 1:
            consensus.append(values[0])
        else:
            disagreement_items += 1
    pair_columns = tuple(zip(*pairs)) if pairs else ()
    kappa = (
        float(cohen_kappa_score(*pair_columns))
        if pair_columns and all(len(set(column)) > 1 for column in pair_columns)
        else None
    )
    if kappa is not None and not np.isfinite(kappa):
        kappa = None
    overall = _proxy_metrics(consensus)
    per_rater = {
        rater: _proxy_metrics([row for row in usable if row["rater"] == rater])
        for rater in sorted({row["rater"] for row in reviews})
    }
    output = {
        "status": "ok" if len(usable) else "no_completed_reviews",
        "n_input_rows": len(reviews),
        "n_rated_rows_including_uncertain": sum(
            row.get("human_label") in ("-1", "0", "1") for row in reviews
        ),
        "n_uncertain_rows": sum(row.get("human_label") == "-1" for row in reviews),
        "n_completed_rows": len(usable),
        "n_completed_items": len(by_item),
        "n_shared_items": len(shared),
        "n_consensus_items": len(consensus),
        "n_disagreement_items": disagreement_items,
        "rater_agreement": kappa,
        "proxy_vs_human_accuracy": overall["accuracy"] if overall else None,
        "proxy_vs_human_balanced_accuracy": (
            overall["balanced_accuracy"] if overall else None
        ),
        "proxy_vs_human_f1": overall["f1"] if overall else None,
        "proxy_vs_human_auroc": overall["auroc"] if overall else None,
        "proxy_vs_human_accuracy_ci95": overall["accuracy_ci95"] if overall else None,
        "proxy_metrics": overall,
        "per_rater": per_rater,
        "task_counts": {},
        "limitations": [
            "Human labels measure visible progress, not verified task success.",
            "Uncertain and incomplete rows are excluded.",
            "Agreement is reported only for items reviewed by both raters.",
        ],
    }
    tasks = defaultdict(list)
    for row in consensus:
        tasks[row["task"]].append(row)
    for task, rows in sorted(tasks.items()):
        ty = np.asarray([r["proxy"] for r in rows], dtype=int)
        th = np.asarray([r["human"] for r in rows], dtype=int)
        output["task_counts"][task] = {
            "n": len(rows),
            "accuracy": float(accuracy_score(ty, th)),
            "proxy_positive_rate": float(ty.mean()),
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
