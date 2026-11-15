# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from agile.rl_env.mdp.actions import JointPositionGUIAction, JointPositionGUIActionCfg


class TestJointPositionGUIAction(unittest.TestCase):
    """Tests for GUI gain mapping and pose transitions."""

    def test_configuration_defaults(self) -> None:
        cfg = JointPositionGUIActionCfg()
        self.assertEqual(cfg.max_stiffness, 200.0)
        self.assertEqual(cfg.pose_transition_duration_s, 1.0)

    @staticmethod
    def _make_actuator(joint_ids: list[int], stiffness: list[float], damping: list[float]):
        return SimpleNamespace(
            joint_indices=torch.tensor(joint_ids),
            joint_names=[f"joint_{joint_id}" for joint_id in joint_ids],
            stiffness=torch.tensor([stiffness]),
            damping=torch.tensor([damping]),
            effort_limit=torch.tensor([[10.0 * (joint_id + 1) for joint_id in joint_ids]]),
        )

    def _make_action(
        self,
        selected_joint_ids: list[int],
        *,
        step_dt: float = 0.25,
        transition_duration_s: float = 1.0,
    ):
        group_a = self._make_actuator([0, 2], [100.0, 300.0], [1.0, 3.0])
        group_b = self._make_actuator([1, 3], [200.0, 400.0], [2.0, 4.0])

        asset = MagicMock()
        asset.num_joints = 4
        asset.actuators = {"group_a": group_a, "group_b": group_b}
        asset.data.joint_pos.torch = torch.zeros(1, 4)
        asset.data.default_joint_pos.torch = torch.zeros(1, 4)
        asset.find_joints.side_effect = lambda names, preserve_order=False: (
            [int(name.removeprefix("joint_")) for name in names],
            list(names),
        )

        env = SimpleNamespace(num_envs=1, device="cpu", step_dt=step_dt, scene={"robot": asset})
        cfg = JointPositionGUIActionCfg(
            asset_name="robot",
            joint_names=[f"joint_{joint_id}" for joint_id in selected_joint_ids],
            preserve_order=True,
            mirror_actions=False,
            pose_transition_duration_s=transition_duration_s,
        )
        with patch("agile.rl_env.mdp.actions.joint_pos_gui_action.threading.Thread.start"):
            action = JointPositionGUIAction(cfg, env)

        return action, asset, group_a, group_b

    def test_maps_interleaved_actuator_groups_by_joint_name(self) -> None:
        action, asset, group_a, group_b = self._make_action([0, 1, 2, 3])

        torch.testing.assert_close(action._desired_stiffness, torch.tensor([[100.0, 200.0, 300.0, 400.0]]))
        torch.testing.assert_close(action._desired_damping, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        torch.testing.assert_close(
            action._read_selected_actuator_property("effort_limit"), torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        )

        action._desired_stiffness = torch.tensor([[11.0, 22.0, 33.0, 44.0]])
        action._desired_damping = torch.tensor([[1.1, 2.2, 3.3, 4.4]])
        action.apply_actions()

        torch.testing.assert_close(group_a.stiffness, torch.tensor([[11.0, 33.0]]))
        torch.testing.assert_close(group_a.damping, torch.tensor([[1.1, 3.3]]))
        torch.testing.assert_close(group_b.stiffness, torch.tensor([[22.0, 44.0]]))
        torch.testing.assert_close(group_b.damping, torch.tensor([[2.2, 4.4]]))
        asset.set_joint_position_target.assert_called_once()

    def test_reordered_subset_leaves_other_joint_gains_unchanged(self) -> None:
        action, _, group_a, group_b = self._make_action([3, 0])
        torch.testing.assert_close(action._desired_stiffness, torch.tensor([[400.0, 100.0]]))

        action._desired_stiffness = torch.tensor([[44.0, 11.0]])
        action._desired_damping = torch.tensor([[4.4, 1.1]])
        action.apply_actions()

        torch.testing.assert_close(group_a.stiffness, torch.tensor([[11.0, 300.0]]))
        torch.testing.assert_close(group_a.damping, torch.tensor([[1.1, 3.0]]))
        torch.testing.assert_close(group_b.stiffness, torch.tensor([[200.0, 44.0]]))
        torch.testing.assert_close(group_b.damping, torch.tensor([[2.0, 4.4]]))

    def test_position_transition_uses_control_timestep(self) -> None:
        action, _, _, _ = self._make_action([0, 1, 2, 3], step_dt=0.25, transition_duration_s=1.0)
        target = torch.tensor([1.0, 2.0, 3.0, 4.0])
        action._start_position_transition(target)

        # Applying multiple physics substeps must not advance transition time.
        for _ in range(4):
            action.apply_actions()
        torch.testing.assert_close(action._desired_pos, torch.zeros(1, 4))

        action.process_actions(torch.empty(1, 4))
        first_step_blend = 0.103515625  # quintic minimum-jerk blend at t=0.25
        torch.testing.assert_close(action._desired_pos, target.unsqueeze(0) * first_step_blend)

        action.process_actions(torch.empty(1, 4))
        torch.testing.assert_close(action._desired_pos, target.unsqueeze(0) * 0.5)

        action.process_actions(torch.empty(1, 4))
        self.assertTrue(action._position_transition_active)
        action.process_actions(torch.empty(1, 4))
        torch.testing.assert_close(action._desired_pos, target.unsqueeze(0))
        self.assertFalse(action._position_transition_active)

    def test_manual_joint_edit_cancels_position_transition(self) -> None:
        action, _, _, _ = self._make_action([0, 1, 2, 3])
        action._start_position_transition(torch.ones(4))
        action.process_actions(torch.empty(1, 4))
        action._set_desired_joint_position(1, -0.5)
        interrupted_position = action._desired_pos.clone()

        for _ in range(8):
            action.process_actions(torch.empty(1, 4))

        self.assertFalse(action._position_transition_active)
        torch.testing.assert_close(action._desired_pos, interrupted_position)
        self.assertEqual(action._desired_pos[0, 1].item(), -0.5)

    def test_new_transition_restarts_from_current_interpolated_position(self) -> None:
        action, _, _, _ = self._make_action([0, 1, 2, 3])
        action._start_position_transition(torch.ones(4))
        action.process_actions(torch.empty(1, 4))
        current_position = action._desired_pos.clone()

        new_target = -torch.ones(4)
        action._start_position_transition(new_target)
        torch.testing.assert_close(action._desired_pos, current_position)
        action.process_actions(torch.empty(1, 4))

        blend = 0.103515625
        torch.testing.assert_close(action._desired_pos, torch.lerp(current_position, new_target.unsqueeze(0), blend))

    def test_default_button_transition_is_smooth_and_restores_gains(self) -> None:
        action, asset, _, _ = self._make_action([0, 1, 2, 3])
        default_position = torch.tensor([[0.2, -0.4, 0.6, -0.8]])
        asset.data.default_joint_pos.torch = default_position
        action._desired_pos.fill_(1.0)
        action._desired_stiffness.zero_()
        action._desired_damping.zero_()

        action._start_default_position_transition()

        # Button press itself must not jump the commanded pose.
        torch.testing.assert_close(action._desired_pos, torch.ones(1, 4))
        torch.testing.assert_close(action._desired_stiffness, action._default_stiffness)
        torch.testing.assert_close(action._desired_damping, action._default_damping)
        for _ in range(4):
            action.process_actions(torch.empty(1, 4))
        torch.testing.assert_close(action._desired_pos, default_position)

    def test_environment_reset_cancels_active_transition(self) -> None:
        action, asset, _, _ = self._make_action([0, 1, 2, 3])
        action._start_position_transition(torch.ones(4))
        action.process_actions(torch.empty(1, 4))
        reset_position = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        asset.data.joint_pos.torch = reset_position

        action.reset(slice(None))
        self.assertFalse(action._position_transition_active)
        torch.testing.assert_close(action._desired_pos, reset_position)
        action.process_actions(torch.empty(1, 4))
        torch.testing.assert_close(action._desired_pos, reset_position)

    def test_transition_duration_must_be_positive(self) -> None:
        for invalid_duration in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(invalid_duration=invalid_duration):
                with self.assertRaisesRegex(ValueError, "pose_transition_duration_s"):
                    self._make_action([0, 1, 2, 3], transition_duration_s=invalid_duration)


if __name__ == "__main__":
    unittest.main()
