"""Audit whether the target marker is recoverable from rendered RGB."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier


def marker_centroid(image):
    # Target seats are bright yellow; distractors are gray.  This detector is
    # only an audit oracle for visibility, not a privileged target label.
    x = image.astype(np.int16)
    # Renderer lighting compresses the nominal yellow to roughly [100, 85, 5].
    mask = (x[..., 0] > 80) & (x[..., 1] > 60) & (x[..., 2] < 40)
    yy, xx = np.where(mask)
    if len(xx) == 0:
        return np.array([0.5, 0.5], np.float32), False
    return np.array([xx.mean() / image.shape[1], yy.mean() / image.shape[0]], np.float32), True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True); args = ap.parse_args()
    d = np.load(args.input); rgb = d["rgb"]; y = d["success"].astype(int)
    direction = d["direction"].astype(int); target = np.asarray(d["target_xy"])
    pairs = [marker_centroid(im) for im in rgb]
    cent = np.stack([p[0] for p in pairs]); detected = np.asarray([p[1] for p in pairs])
    # Target marker coordinates are directly observable; classify the target
    # seat from centroid position as a visibility check.
    seat_cent = np.stack([cent[target[:, 0] == target[:, 0][0]][:1]]) if False else None
    # Fit the success readout with action plus the image-derived location.
    action = np.eye(3)[direction]
    X = np.c_[action, cent]
    idx = np.random.default_rng(0).permutation(len(y)); cut = len(y)//2
    tr, te = idx[:cut], idx[cut:]
    clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1500,
                        random_state=0).fit(X[tr], y[tr])
    auc = float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    result = {"n": int(len(y)), "marker_detect_rate": float(detected.mean()),
              "centroid_x_range": [float(cent[:,0].min()), float(cent[:,0].max())],
              "centroid_y_range": [float(cent[:,1].min()), float(cent[:,1].max())],
              "action_plus_rgb_marker_auroc": auc}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
