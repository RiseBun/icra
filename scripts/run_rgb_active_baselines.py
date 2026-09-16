"""Compare active-observation gates under controlled RGB occlusion.

This is a diagnostic benchmark: the second view is represented by the clean
image, while the first view is corrupted by a rectangular occluder.  All
policies use the same trained RGB consequence predictor and differ only in
when they request the second view.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier


LABEL = {"drop": 0, "wrong_target": 1, "success": 2}


def center(image: np.ndarray, kind: str) -> tuple[np.ndarray, bool]:
    x = image.astype(np.int16)
    if kind == "yellow":
        mask = (x[..., 0] > 65) & (x[..., 1] > 50) & (x[..., 2] < 35)
    elif kind == "blue":
        mask = (x[..., 2] > 55) & (x[..., 0] < 55) & (x[..., 1] < 80)
    else:
        mask = (x[..., 0] > 55) & (x[..., 1] < 45) & (x[..., 2] < 45)
    yy, xx = np.where(mask)
    if len(xx) == 0:
        return np.array([0.5, 0.5], np.float32), False
    return np.array([xx.mean() / image.shape[1], yy.mean() / image.shape[0]], np.float32), len(xx) > 3


def feature(image: np.ndarray, direction: int) -> np.ndarray:
    return np.r_[np.eye(3, dtype=np.float32)[direction],
                 center(image, "yellow")[0], center(image, "blue")[0],
                 center(image, "red")[0]]


def truth(row: dict, direction: int) -> int:
    seats = np.asarray(row["seat_positions"], np.float32).reshape(3, 2)
    target = np.asarray(row["target_xy"], np.float32)
    target_id = int(np.argmin(np.linalg.norm(seats - target[None], axis=1)))
    return int(direction == target_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--interaction", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = np.load(args.input)
    rows = json.loads(Path(args.interaction).read_text())
    n = min(len(rows), len(data["rgb"]))
    rgb, rows = data["rgb"][:n], rows[:n]
    rng = np.random.default_rng(2027)
    indices = rng.permutation(n)
    train, test = indices[: n // 2], indices[n // 2 :]
    x = np.stack([feature(im, int(row["direction"])) for im, row in zip(rgb, rows)])
    y = np.asarray([LABEL[row["label"]] for row in rows])
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1500,
                        random_state=0).fit(x[train], y[train])

    policies = ("fixed", "visibility", "risk", "random", "risk_or_visibility")
    result = {"n_test": int(len(test)), "levels": {}}
    for frac in (0.0, 0.10, 0.20, 0.30, 0.40):
        cache = []
        for i in test:
            image = rgb[i].copy()
            h, w = image.shape[:2]
            side = int(np.sqrt(frac) * min(h, w))
            if side > 0:
                x0 = int(rng.integers(0, max(1, w - side + 1)))
                y0 = int(rng.integers(0, max(1, h - side + 1)))
                image[y0:y0 + side, x0:x0 + side] = 24
            valid = float(np.mean([center(image, k)[1] for k in ("yellow", "blue", "red")]))
            probs = clf.predict_proba(np.stack([feature(image, d) for d in range(3)]))
            risk = probs[:, 0] + probs[:, 1]
            first_action = int(np.argmin(risk))
            clean_probs = clf.predict_proba(np.stack([feature(rgb[i], d) for d in range(3)]))
            clean_action = int(np.argmin(clean_probs[:, 0] + clean_probs[:, 1]))
            cache.append((i, valid, float(np.min(risk)), first_action, clean_action))

        ours_rate = float(np.mean([r > 0.45 or v < 0.67 for _, v, r, _, _ in cache]))
        level = {"trigger_rate": ours_rate, "policies": {}}
        for policy in policies:
            successes, observations = [], []
            for i, valid, min_risk, first_action, clean_action in cache:
                if policy == "fixed":
                    observe = False
                elif policy == "visibility":
                    observe = valid < 0.67
                elif policy == "risk":
                    observe = min_risk > 0.45
                elif policy == "random":
                    observe = bool(rng.random() < ours_rate)
                else:
                    observe = min_risk > 0.45 or valid < 0.67
                action = clean_action if observe else first_action
                successes.append(truth(rows[i], action))
                observations.append(int(observe))
            success = float(np.mean(successes))
            extra = float(np.mean(observations))
            level["policies"][policy] = {
                "success": success,
                "extra_views": extra,
                "utility": success - 0.10 * extra,
            }
        result["levels"][str(frac)] = level

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
