import numpy as np

from scripts.evaluate_omega_scale_anchor import evaluate


def _write_pair(raw_root, exported_root, offset):
    relative = "open_drawer/episode_0000/decision_0000.npz"
    raw_path, exported_path = raw_root / relative, exported_root / relative
    raw_path.parent.mkdir(parents=True)
    exported_path.parent.mkdir(parents=True)
    points = np.zeros((2, 4, 3), np.float32)
    np.savez_compressed(
        raw_path, points=points, robot_base_to_world=np.eye(4, dtype=np.float32),
        history_rgb=np.zeros((2, 3, 256, 256), np.uint8))
    np.savez_compressed(
        exported_path, points=points + offset,
        omega_scale=np.asarray(1.0, np.float32),
        omega_scale_anchor_count=np.asarray(64, np.int32))


def test_omega_gate_passes_small_metric_error(tmp_path):
    raw, exported = tmp_path / "raw", tmp_path / "exported"
    _write_pair(raw, exported, 0.01)
    result = evaluate(raw, exported)
    assert result["passed"]
    assert result["checks"]["source_is_256"]


def test_omega_gate_rejects_large_error(tmp_path):
    raw, exported = tmp_path / "raw", tmp_path / "exported"
    _write_pair(raw, exported, 1.0)
    result = evaluate(raw, exported)
    assert not result["passed"]
    assert not result["checks"]["mean_epe"]
