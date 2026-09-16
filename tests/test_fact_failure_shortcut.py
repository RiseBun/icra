import json
from pathlib import Path

from scripts.audit_fact_failure_shortcut import (
    collect_fact_policy_rollouts,
    dump_fact_schema,
    load_fact_schema,
    load_failure_index,
    mann_whitney_auroc,
    score_recipe,
    synthetic_fact_recipes,
)


def test_mann_whitney_perfect_and_chance():
    y = [1, 1, 0, 0]
    assert mann_whitney_auroc(y, [3, 2, 1, 0]) == 1.0
    assert mann_whitney_auroc(y, [1, 0, 1, 0]) == 0.5


def test_synthetic_recipes_isolate_amplitude_shortcut():
    recipes = synthetic_fact_recipes(n=80, seed=1)
    pert = score_recipe(recipes["expert_perturbation"], n_boot=50, seed=1)
    matched = score_recipe(recipes["fact_matched_amp"], n_boot=50, seed=1)
    mixed = score_recipe(recipes["fact_mixed_amp"], n_boot=50, seed=1)
    assert pert["amplitude_shortcut"]
    assert pert["mean_l2"]["auroc_success_from_neg_amp"] >= 0.90
    assert matched["amplitude_uninformative"]
    assert mixed["amplitude_shortcut"]
    assert mixed["mean_l2"]["auroc_success_from_neg_amp"] >= 0.90


def test_fact_policy_rollouts_match_synthetic_regimes():
    matched = score_recipe(collect_fact_policy_rollouts(n=80, seed=3, mixed_amplitude=False),
                           n_boot=40, seed=3)
    mixed = score_recipe(collect_fact_policy_rollouts(n=80, seed=3, mixed_amplitude=True),
                         n_boot=40, seed=3)
    assert matched["amplitude_uninformative"]
    assert mixed["amplitude_shortcut"]
    assert matched["success_rate"] == 0.5
    assert mixed["success_rate"] == 0.5


def test_fact_schema_roundtrip(tmp_path: Path):
    rows = collect_fact_policy_rollouts(n=20, seed=4, mixed_amplitude=False)
    dump_fact_schema(rows, tmp_path)
    loaded = load_fact_schema(tmp_path)
    assert len(loaded) == 20
    assert len(load_failure_index(tmp_path)) == 10
    orig = [r["success"] for r in rows]
    got = [r["success"] for r in loaded]
    assert orig == got


def test_failure_rollouts_jsonl(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "failure_rollouts.jsonl").write_text(
        json.dumps({"episode_index": 3, "failure_episode": True,
                    "failure_active_from_frame": 12}) + "\n",
        encoding="utf-8",
    )
    rows = load_failure_index(tmp_path)
    assert rows[3]["failure_active_from_frame"] == 12
    assert load_failure_index(tmp_path / "missing") == {}
