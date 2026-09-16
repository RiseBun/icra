from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines import select_with_observation_budget
from dataio import Risk4DDataset
from models import (
    Risk4DEnsemble, RiskAwareController, RiskCalibration,
    RiskConditioned4DModel, ViewValueModel, risk4d_loss,
    rlbench_franka_joint_trajectory,
)


class Risk4DTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.b, self.k, self.m, self.steps = 2, 3, 4, 5
        self.h, self.n, self.q = 2, 12, 6
        self.config = dict(
            point_feature_dim=1, robot_state_dim=6, task_embedding_dim=7,
            hidden_dim=32, future_steps=self.h, layers=2, heads=4, dropout=0.0,
        )
        self.inputs = (
            torch.randn(self.b, self.k, self.n, 3),
            torch.rand(self.b, self.k, self.n, 1),
            torch.randn(self.b, self.k, 6),
            torch.randn(self.b, self.m, self.steps, self.q, 3),
            torch.randn(self.b, 7),
        )

    def test_model_shapes_and_backward(self) -> None:
        model = RiskConditioned4DModel(**self.config)
        output = model(*self.inputs)
        self.assertEqual(output.future_point_flow.shape, (self.b, self.m, self.h, self.n, 3))
        self.assertEqual(output.task_contact_logits.shape, (self.b, self.m, self.h, self.n))
        losses = risk4d_loss(
            output,
            torch.randn(self.b, self.m, self.h, self.n, 3),
            torch.rand(self.b, self.m, self.h, self.n),
            torch.rand(self.b, self.m, self.h, self.n),
            torch.randint(0, 2, (self.b, self.m)).float(),
            torch.randint(0, 2, (self.b, self.m)).float(),
        )
        losses["total"].backward()
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_controller_and_decision_modes(self) -> None:
        members = [RiskConditioned4DModel(**self.config) for _ in range(2)]
        controller = RiskAwareController(
            Risk4DEnsemble(members), ViewValueModel(32, 32), RiskCalibration(),
            risk_threshold=3.0,
        )
        output = controller(
            *self.inputs, torch.randn(self.b, 3, 6), torch.zeros(self.b, 3),
            torch.zeros(self.b, 3), torch.zeros(self.b, dtype=torch.long),
        )
        self.assertEqual(output.calibrated_risk_bound.shape, (self.b, self.m))
        self.assertTrue(torch.equal(output.decision.mode, torch.zeros(self.b, dtype=torch.long)))

        decision = select_with_observation_budget(
            torch.tensor([[0.8, 0.9], [0.8, 0.9]]),
            torch.tensor([[0.4, 0.1], [0.4, 0.1]]), torch.zeros(2, 2),
            torch.zeros(2, 2), torch.zeros(2, 2), torch.tensor([0, 2]),
            risk_threshold=0.35, max_observations=2,
        )
        self.assertEqual(decision.mode.tolist(), [1, 2])
        self.assertEqual(decision.index.tolist(), [0, -1])

    def test_calibration_and_robot_action_mapping(self) -> None:
        success_logits = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        success = torch.tensor([0.0, 0.0, 1.0, 1.0])
        collision_logits = -success_logits
        collision = 1.0 - success
        calibration = RiskCalibration.fit(
            success_logits, collision_logits, success, collision, coverage=0.75,
        )
        risk = calibration.risk_bound(success_logits[None], collision_logits[None])
        self.assertEqual(risk.shape, (1, 4))
        self.assertTrue(torch.isfinite(risk).all())

        actions = torch.zeros(1, 2, 3, 8)
        actions[..., 7] = 1.0
        joints, names = rlbench_franka_joint_trajectory(
            actions, [f"j{i}" for i in range(7)], ["f1", "f2"],
        )
        self.assertEqual(joints.shape, (1, 2, 3, 9))
        self.assertEqual(names[-2:], ["f1", "f2"])
        self.assertTrue(torch.allclose(joints[..., -2:], torch.full_like(joints[..., -2:], 0.04)))

    def test_strict_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez_compressed(
                path,
                points=np.zeros((2, 5, 3), np.float32),
                point_features=np.ones((2, 5, 1), np.float32),
                robot_state=np.zeros((2, 6), np.float32),
                robot_action_flow=np.zeros((3, 4, 2, 3), np.float32),
                task_embedding=np.zeros(7, np.float32),
                future_point_flow=np.zeros((3, 2, 5, 3), np.float32),
                task_contact_map=np.zeros((3, 2, 5), np.float32),
                harmful_collision_map=np.zeros((3, 2, 5), np.float32),
                success=np.zeros(3, np.float32), collision=np.zeros(3, np.float32),
            )
            sample = Risk4DDataset(directory)[0]
            self.assertEqual(tuple(sample["robot_action_flow"].shape), (3, 4, 2, 3))


if __name__ == "__main__":
    unittest.main()
