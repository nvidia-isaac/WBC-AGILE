# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest
from unittest.mock import MagicMock, patch

import torch

from agile.rl_env.tests.utils import APP_IS_READY

if APP_IS_READY:
    from isaaclab.managers import SceneEntityCfg

from agile.rl_env.mdp.utils import (
    get_body_velocities_and_forces,
    get_contact_sensor_cfg,
    get_joint_indices,
    get_robot_cfg,
    transform_to_body_frame,
)


class TestMDPUtils(unittest.TestCase):
    def setUp(self) -> None:
        """Set up mock objects for testing."""
        # Set device for consistency
        self.device = torch.device("cpu")

        # Create a mock environment
        self.env = MagicMock()

        # Create mock robot data
        robot_data = MagicMock()
        robot_data.joint_pos = torch.ones((2, 10), device=self.device)  # 2 envs, 10 joints
        robot_data.joint_vel = torch.ones((2, 10), device=self.device)
        robot_data.body_pos_w = torch.ones((2, 5, 3), device=self.device)  # 2 envs, 5 bodies, 3D positions
        robot_data.body_vel_w = torch.ones((2, 5, 3), device=self.device)
        robot_data.root_pos_w = torch.zeros((2, 3), device=self.device)
        robot_data.root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=self.device)

        # Create mock robot
        self.robot = MagicMock()
        self.robot.data = robot_data
        self.robot.num_joints = 10
        self.robot.find_bodies = MagicMock(return_value=([0, 1, 2], None))
        self.robot.device = self.device

        # Mock the scene with a proper dictionary access behavior
        self.scene_dict = {"robot": self.robot}
        scene_mock = MagicMock()
        scene_mock.__getitem__ = lambda s, key: self.scene_dict.get(key)  # noqa: ARG005
        scene_mock.get = lambda key, default=None: self.scene_dict.get(key, default)
        self.env.scene = scene_mock

        # Create mock contact sensor data
        contact_data = MagicMock()
        contact_data.net_forces_w = torch.ones((2, 5, 3), device=self.device)  # 2 envs, 5 bodies, 3D forces

        # Create mock contact sensor
        self.contact_sensor = MagicMock()
        self.contact_sensor.data = contact_data

        # Set up sensors in the scene
        self.env.scene.sensors = {
            "contact_sensor": self.contact_sensor,
        }

        # Set up example joint IDs
        self.env.lower_body_ids = [0, 1, 2, 3]
        self.env.upper_body_ids = [4, 5, 6, 7]

    def test_get_joint_indices(self) -> None:
        # Test lower body
        indices = get_joint_indices(self.env, self.robot, "lower_body")
        self.assertEqual(indices, self.env.lower_body_ids)

        # Test upper body
        indices = get_joint_indices(self.env, self.robot, "upper_body")
        self.assertEqual(indices, self.env.upper_body_ids)

        # Test whole body
        indices = get_joint_indices(self.env, self.robot, "whole_body")
        self.assertEqual(indices, list(range(self.robot.num_joints)))

        # Test invalid part
        with self.assertRaises(ValueError):
            get_joint_indices(self.env, self.robot, "invalid_part")

    def test_get_robot_cfg(self) -> None:
        # Mock SceneEntityCfg for default case
        default_cfg = MagicMock()
        default_cfg.name = "robot"

        # Test with default config
        with patch("agile.rl_env.mdp.utils.SceneEntityCfg", return_value=default_cfg):
            robot, cfg = get_robot_cfg(self.env)
            self.assertEqual(robot, self.robot)
            self.assertEqual(cfg.name, "robot")

        # Test with custom config
        if APP_IS_READY:
            custom_cfg = SceneEntityCfg("custom_robot")
            # Update scene_dict to include custom robot
            self.scene_dict["custom_robot"] = self.robot
            robot, cfg = get_robot_cfg(self.env, custom_cfg)
            self.assertEqual(cfg.name, "custom_robot")
            self.assertEqual(robot, self.robot)

    def test_get_contact_sensor_cfg(self) -> None:
        # Mock SceneEntityCfg for default case
        default_cfg = MagicMock()
        default_cfg.name = "contact_sensor"
        default_cfg.body_names = [".*ankle.*link"]

        # Test with default config
        with patch("agile.rl_env.mdp.utils.SceneEntityCfg", return_value=default_cfg):
            sensor, cfg = get_contact_sensor_cfg(self.env)
            self.assertEqual(sensor, self.contact_sensor)
            self.assertEqual(cfg.name, "contact_sensor")
            self.assertEqual(cfg.body_names, [".*ankle.*link"])

        # Test with custom body names
        custom_cfg = MagicMock()
        custom_cfg.name = "contact_sensor"
        custom_cfg.body_names = [".*foot.*"]

        with patch("agile.rl_env.mdp.utils.SceneEntityCfg", return_value=custom_cfg):
            sensor, cfg = get_contact_sensor_cfg(self.env, body_names=[".*foot.*"])
            self.assertEqual(sensor, self.contact_sensor)
            self.assertEqual(cfg.body_names, [".*foot.*"])

    def test_transform_to_body_frame(self) -> None:
        # Create test data
        positions = torch.tensor([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], device=self.device)
        root_pos = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)  # Identity quaternion

        # Test the function directly by using a simple input case
        result = transform_to_body_frame(positions, root_pos, root_quat)

        # For an identity quaternion, the rotation should be identity,
        # so we expect only translation
        expected = positions - root_pos.unsqueeze(1)

        # Allow small numerical differences
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-5))

    def test_get_body_velocities_and_forces(self) -> None:
        # Create test sensor config with body IDs
        sensor_cfg = MagicMock()
        sensor_cfg.body_ids = torch.tensor([0, 1, 2], device=self.device)

        # Test getting velocities and forces
        velocities, forces = get_body_velocities_and_forces(self.robot, self.contact_sensor, sensor_cfg)

        # Check shapes match expected outputs
        self.assertEqual(velocities.shape, (2, 3, 3))  # 2 envs, 3 bodies, 3D velocities
        self.assertEqual(forces.shape, (2, 3, 3))  # 2 envs, 3 bodies, 3D forces

        # Check values match the mocked data sliced by body_ids
        self.assertTrue(torch.equal(velocities, self.robot.data.body_vel_w[:, sensor_cfg.body_ids]))
        self.assertTrue(torch.equal(forces, self.contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]))


if __name__ == "__main__":
    unittest.main()
