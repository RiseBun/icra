import sys
from types import SimpleNamespace

import numpy as np

from scripts.peg_geometry_signals import extract_target_pillar_geometry


class FakeShape:
    positions = {
        "pillar0": np.array([0.0, 0.0, 0.832], np.float32),
        "pillar1": np.array([0.1, 0.0, 0.832], np.float32),
        "pillar2": np.array([0.0, 0.1, 0.832], np.float32),
    }

    def __init__(self, name):
        self.name = name

    def get_position(self):
        return self.positions[self.name]


def test_target_geometry_uses_explicit_pillar_id(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrep.objects.shape", SimpleNamespace(Shape=FakeShape))
    ring = SimpleNamespace(get_position=lambda: np.array([0.1, 0.0, 0.897], np.float32))
    task = SimpleNamespace(_square_ring=ring)
    value = extract_target_pillar_geometry(task, 1)
    assert value["target_pillar_index"] == 1
    assert value["nearest_pillar_index"] == 1
    assert value["target_distance"] == 0.0
    assert value["wrong_target_distance"] > 0.0


def test_invalid_target_id_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrep.objects.shape", SimpleNamespace(Shape=FakeShape))
    task = SimpleNamespace(_square_ring=SimpleNamespace(get_position=lambda: np.zeros(3)))
    try:
        extract_target_pillar_geometry(task, 3)
    except ValueError:
        return
    raise AssertionError("invalid target id was accepted")
