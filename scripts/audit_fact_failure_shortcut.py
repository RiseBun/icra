"""Amplitude-shortcut A/B test for FACT-style failure data.

FACT (arXiv:2608.10232) does not perturb expert trajectories. Failures are
policy rollouts stored as LeRobot v2 + meta/failure_rollouts.jsonl. This audit
answers whether action amplitude still encodes the success/failure label.

Three synthetic recipes isolate the mechanism:

- expert_perturbation: same direction, larger amplitude -> fail.
- fact_matched_amp: FACT-like directional misses at the expert amplitude.
- fact_mixed_amp: expert successes mixed with larger failed policy moves.

Pass --robotwin_root to run the same scalar test on a real FACT/LeRobot dump.
The synthetic path is stdlib-only; LeRobot loading needs pandas.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


NOMINAL_AMP = 0.16


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def mann_whitney_auroc(y, scores) -> float:
    pos = [s for s, lab in zip(scores, y) if int(lab) == 1]
    neg = [s for s, lab in zip(scores, y) if int(lab) == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def bootstrap_auroc(y, scores, n_boot=400, seed=0):
    rng = random.Random(seed)
    n = len(y)
    vals = []
    for _ in range(int(n_boot)):
        idx = [rng.randrange(n) for _ in range(n)]
        val = mann_whitney_auroc([y[i] for i in idx], [scores[i] for i in idx])
        if math.isfinite(val):
            vals.append(val)
    if not vals:
        return {"median": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    vals.sort()
    def pct(p):
        return vals[min(len(vals) - 1, int(round((p / 100.0) * (len(vals) - 1))))]
    return {"median": pct(50), "ci95": [pct(2.5), pct(97.5)], "n": len(vals)}


def _l2(vec):
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def amplitude_features(action) -> dict[str, float]:
    if not action:
        raise ValueError("empty action")
    if isinstance(action[0], (int, float)):
        chunk = [list(map(float, action))]
    else:
        chunk = [list(map(float, row)) for row in action]
    step_l2 = [_l2(row) for row in chunk]
    return {
        "mean_l2": _mean(step_l2),
        "max_l2": max(step_l2) if step_l2 else float("nan"),
        "std_l2": _std(step_l2),
        "terminal_l2": step_l2[-1] if step_l2 else float("nan"),
    }


def _episode(action, success, recipe, extra=None):
    feats = amplitude_features(action)
    row = {
        "success": int(success),
        "recipe": recipe,
        "action": [[float(x) for x in step] for step in action],
        **feats,
    }
    if extra:
        row.update(extra)
    return row


def make_action(direction_xy, amplitude, horizon=8, noise=0.0, rng=None):
    nrm = _l2(direction_xy)
    if nrm < 1e-8:
        raise ValueError("direction must be nonzero")
    step = [(float(v) / nrm) * (float(amplitude) / horizon) for v in direction_xy]
    chunk = [list(step) for _ in range(horizon)]
    if noise and rng is not None:
        chunk = [[x + rng.gauss(0.0, noise) for x in row] for row in chunk]
    return chunk


def synthetic_fact_recipes(n=240, seed=2027):
    rng = random.Random(seed)
    dirs = ((1.0, 0.0), (-0.5, 0.866), (-0.5, -0.866))
    out = {"expert_perturbation": [], "fact_matched_amp": [], "fact_mixed_amp": []}
    half = n // 2

    for i in range(n):
        amp = NOMINAL_AMP if i < half else rng.choice([0.24, 0.28, 0.32, 0.36])
        success = int(i < half)
        out["expert_perturbation"].append(
            _episode(make_action(dirs[0], amp, rng=rng, noise=0.002), success,
                     "expert_perturbation", {"amplitude": amp, "direction": 0})
        )

    for i in range(n):
        success = int(i < half)
        direction = 0 if success else rng.randrange(1, 3)
        out["fact_matched_amp"].append(
            _episode(make_action(dirs[direction], NOMINAL_AMP, rng=rng, noise=0.002),
                     success, "fact_matched_amp",
                     {"amplitude": NOMINAL_AMP, "direction": direction})
        )

    for i in range(n):
        if i < half:
            amp = NOMINAL_AMP
            direction = 0
            success = 1
        else:
            amp = rng.uniform(0.24, 0.40)
            direction = rng.randrange(0, 3)
            success = 0
        out["fact_mixed_amp"].append(
            _episode(make_action(dirs[direction], amp, rng=rng, noise=0.002),
                     success, "fact_mixed_amp",
                     {"amplitude": amp, "direction": direction})
        )
    return out


def score_recipe(rows, n_boot=400, seed=0):
    y = [int(r["success"]) for r in rows]
    report = {
        "n": len(rows),
        "success_rate": _mean(y) if y else float("nan"),
    }
    for name in ("mean_l2", "max_l2", "std_l2", "terminal_l2"):
        scores = [float(r[name]) for r in rows]
        neg_scores = [-s for s in scores]
        auc = mann_whitney_auroc(y, neg_scores)
        succ = [s for s, lab in zip(scores, y) if lab == 1]
        fail = [s for s, lab in zip(scores, y) if lab == 0]
        report[name] = {
            "auroc_success_from_neg_amp": auc,
            "bootstrap": bootstrap_auroc(y, neg_scores, n_boot=n_boot, seed=seed),
            "mean_success": _mean(succ),
            "mean_fail": _mean(fail),
        }
    amp_auc = report["mean_l2"]["auroc_success_from_neg_amp"]
    report["amplitude_shortcut"] = bool(math.isfinite(amp_auc) and amp_auc >= 0.80)
    report["amplitude_uninformative"] = bool(math.isfinite(amp_auc) and 0.40 <= amp_auc <= 0.60)
    return report


def dump_fact_schema(rows, schema_dir: Path):
    """Write FACT-compatible episode + failure jsonl without parquet."""
    schema_dir = Path(schema_dir)
    meta = schema_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    episodes = []
    failures = []
    action_rows = []
    for i, row in enumerate(rows):
        length = len(row["action"])
        episodes.append({"episode_index": i, "length": length})
        if int(row["success"]) == 0:
            failures.append({
                "episode_index": i,
                "failure_episode": True,
                "failure_active_from_frame": int(row.get("failure_active_from_frame", 0)),
            })
        action_rows.append({
            "episode_index": i,
            "success": int(row["success"]),
            "action": row["action"],
            "recipe": row.get("recipe", ""),
        })
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in episodes), encoding="utf-8")
    (meta / "failure_rollouts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in failures), encoding="utf-8")
    (schema_dir / "episodes_actions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in action_rows), encoding="utf-8")
    return schema_dir


def load_fact_schema(schema_dir: Path, max_episodes=None):
    schema_dir = Path(schema_dir)
    action_path = schema_dir / "episodes_actions.jsonl"
    if not action_path.is_file():
        raise FileNotFoundError(f"missing {action_path}")
    failure_index = load_failure_index(schema_dir)
    rows = []
    for line in action_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        episode_index = int(rec["episode_index"])
        if max_episodes is not None and episode_index >= int(max_episodes):
            break
        fail_row = failure_index.get(episode_index)
        success = int(rec.get("success", 1))
        if fail_row is not None and bool(fail_row.get("failure_episode", True)):
            success = 0
        extra = {"episode_index": episode_index, "recipe": rec.get("recipe", "schema")}
        if fail_row is not None:
            extra["failure_active_from_frame"] = fail_row.get("failure_active_from_frame")
        rows.append(_episode(rec["action"], success, extra["recipe"], extra))
    return rows


def collect_fact_policy_rollouts(n=240, seed=2027, mixed_amplitude=False):
    """FACT-like collection: expert successes + policy misses, no expert noise.

    The 'policy' is a wrong-direction rollout at matched amplitude, or at a
    larger amplitude when mixed_amplitude=True. Labels come from endpoint
    geometry, matching mujoco_strong_multitask.classify.
    """
    rng = random.Random(seed)
    dirs = ((1.0, 0.0), (-0.5, 0.8660254), (-0.5, -0.8660254))
    radius = 0.045
    rows = []
    half = n // 2
    for i in range(n):
        goal = rng.randrange(3)
        target = [dirs[goal][0] * NOMINAL_AMP, dirs[goal][1] * NOMINAL_AMP]
        if i < half:
            direction = goal
            amp = NOMINAL_AMP
        else:
            direction = (goal + 1 + rng.randrange(2)) % 3
            amp = rng.uniform(0.24, 0.40) if mixed_amplitude else NOMINAL_AMP
        action = make_action(dirs[direction], amp, rng=rng, noise=0.001)
        endpoint = [dirs[direction][0] * amp, dirs[direction][1] * amp]
        dist = math.hypot(endpoint[0] - target[0], endpoint[1] - target[1])
        success = int(dist < radius)
        recipe = "fact_policy_mixed" if mixed_amplitude else "fact_policy_matched"
        rows.append(_episode(action, success, recipe, {
            "amplitude": amp, "direction": direction, "goal": goal,
            "endpoint_xy": endpoint, "target_xy": target,
        }))
    return rows


def load_failure_index(task_dir: Path) -> dict[int, dict]:
    path = Path(task_dir) / "meta" / "failure_rollouts.jsonl"
    rows = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[int(row["episode_index"])] = row
    return rows


def load_lerobot_episode_actions(task_dir: Path, max_episodes=None):
    task_dir = Path(task_dir)
    episodes_path = task_dir / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"missing {episodes_path}")
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to audit a LeRobot dump") from exc

    info_path = task_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    data_pattern = str(info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
    chunk_size = int(info.get("chunks_size", 1000))
    failure_index = load_failure_index(task_dir)

    episodes = []
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        episodes.append(json.loads(line))
        if max_episodes is not None and len(episodes) >= int(max_episodes):
            break

    rows = []
    for ep in episodes:
        episode_index = int(ep["episode_index"])
        parquet_path = task_dir / data_pattern.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
        )
        if not parquet_path.is_file():
            continue
        df = pd.read_parquet(parquet_path)
        if "action" not in df.columns:
            raise KeyError(f"{parquet_path} has no action column")
        action = [list(map(float, v)) for v in df["action"].to_numpy()]
        fail_row = failure_index.get(episode_index)
        success = 0 if fail_row is not None and bool(fail_row.get("failure_episode", True)) else 1
        extra = {"episode_index": episode_index, "task_dir": str(task_dir)}
        if fail_row is not None:
            extra["failure_active_from_frame"] = fail_row.get("failure_active_from_frame")
        rows.append(_episode(action, success, "lerobot", extra))
    return rows


def discover_task_dirs(root: Path):
    return sorted(meta.parent.parent for meta in Path(root).rglob("meta/episodes.jsonl"))


def audit_lerobot_root(root: Path, max_episodes=None, n_boot=400):
    task_dirs = discover_task_dirs(root)
    all_rows = []
    per_task = {}
    for task_dir in task_dirs:
        rows = load_lerobot_episode_actions(task_dir, max_episodes=max_episodes)
        if not rows:
            continue
        per_task[str(task_dir)] = score_recipe(rows, n_boot=n_boot)
        all_rows.extend(rows)
    pooled = score_recipe(all_rows, n_boot=n_boot) if all_rows else {}
    return {
        "n_tasks": len(per_task),
        "n_episodes": len(all_rows),
        "pooled": pooled,
        "per_task": per_task,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/diagnostics/fact_failure_shortcut")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--robotwin_root", default="")
    ap.add_argument("--schema_dir", default="")
    ap.add_argument("--mujoco_fact_json", default="")
    ap.add_argument("--max_episodes", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    recipes = synthetic_fact_recipes(n=args.n, seed=args.seed)
    synthetic = {name: score_recipe(rows, n_boot=args.bootstrap, seed=args.seed)
                 for name, rows in recipes.items()}
    report = {
        "fact_protocol": {
            "source": "arXiv:2608.10232 + Bariona/FACT",
            "failure_origin": "policy rollouts, not expert-action perturbation",
            "storage": "LeRobot v2 meta/failure_rollouts.jsonl",
            "labels": ["failure_episode", "failure_active_from_frame"],
            "value_target": "t/T plus VALUE_PENALTY_SCALE after failure onset",
            "action_loss": "masked on failure episodes",
        },
        "synthetic": synthetic,
        "interpretation": {
            "expert_perturbation": "amplitude encodes outcome; this is the original confound",
            "fact_matched_amp": "FACT-like directional misses at matched amplitude; amplitude should be near chance",
            "fact_mixed_amp": "expert success mixed with larger failed moves; mixture itself is an amplitude shortcut",
        },
    }
    policy_matched = collect_fact_policy_rollouts(n=args.n, seed=args.seed, mixed_amplitude=False)
    policy_mixed = collect_fact_policy_rollouts(n=args.n, seed=args.seed, mixed_amplitude=True)
    report["fact_policy_rollouts"] = {
        "matched_amp": score_recipe(policy_matched, n_boot=args.bootstrap, seed=args.seed),
        "mixed_amp": score_recipe(policy_mixed, n_boot=args.bootstrap, seed=args.seed),
    }
    dump_fact_schema(policy_matched, out / "fact_schema_matched")
    dump_fact_schema(policy_mixed, out / "fact_schema_mixed")
    max_ep = args.max_episodes or None
    if args.mujoco_fact_json:
        dumped = json.loads(Path(args.mujoco_fact_json).read_text(encoding="utf-8"))
        physics = {}
        for name in ("matched", "mixed"):
            raw = dumped.get(name, [])
            rows = [_episode(r["action"], r["success"], r.get("recipe", name), {
                "amplitude": r.get("amplitude"), "direction": r.get("direction"),
                "label": r.get("label"),
            }) for r in raw]
            physics[name] = score_recipe(rows, n_boot=args.bootstrap, seed=args.seed)
            dump_fact_schema(rows, out / f"mujoco_schema_{name}")
        report["mujoco_fact_policy"] = physics
    if args.schema_dir:
        schema_rows = load_fact_schema(Path(args.schema_dir), max_episodes=max_ep)
        report["schema"] = score_recipe(schema_rows, n_boot=args.bootstrap, seed=args.seed)
    if args.robotwin_root:
        report["lerobot"] = audit_lerobot_root(
            Path(args.robotwin_root), max_episodes=max_ep, n_boot=args.bootstrap)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, rows in recipes.items():
        (out / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    printable = {
        "fact_protocol": report["fact_protocol"],
        "synthetic": synthetic,
        "fact_policy_rollouts": report["fact_policy_rollouts"],
    }
    if "mujoco_fact_policy" in report:
        printable["mujoco_fact_policy"] = report["mujoco_fact_policy"]
    if "lerobot" in report:
        printable["lerobot"] = report["lerobot"]
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
