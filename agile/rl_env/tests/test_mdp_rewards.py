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

    from agile.rl_env import mdp


class TestRewardsBase(unittest.TestCase):
    """Base class for rewards tests to setup common mock objects."""

    def setUp(self) -> None:
        """Set up test fixtures with common mock objects for rewards testing."""
        self.num_envs = 2
        self.device = "cpu"

        # Create mock environment
        self.env = MagicMock()
        self.env.device = self.device
        self.env.num_envs = self.num_envs

        # Create mock robot
        self.robot = MagicMock()
        self.robot.device = self.device

        # Setup robot data
        self.robot.data = MagicMock()
        self.robot.data.root_pos_w = torch.tensor([[0.0, 0.0, 0.7], [0.0, 0.0, 0.8]], device=self.device)
        self.robot.data.root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=self.device)
        self.robot.data.root_lin_vel_w = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.2, 0.0]], device=self.device)
        self.robot.data.joint_pos = torch.zeros((self.num_envs, 10), device=self.device)
        self.robot.data.joint_vel = torch.zeros((self.num_envs, 10), device=self.device)
        self.robot.data.applied_torque = torch.zeros((self.num_envs, 10), device=self.device)
        self.robot.data.body_pos_w = torch.tensor(
            [
                [[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]],  # env 0: left foot, right foot
                [[0.0, 0.1, 0.1], [0.0, -0.1, 0.0]],  # env 1: left foot, right foot
            ],
            device=self.device,
        )

        # Mock find_bodies method
        self.robot.find_bodies = MagicMock(
            return_value=(
                torch.tensor([0, 1], device=self.device),
                ["left_foot", "right_foot"],
            )
        )

        # Create mock contact sensor
        self.contact_sensor = MagicMock()
        self.contact_sensor.data = MagicMock()
        self.contact_sensor.data.net_forces_w = torch.tensor(
            [
                [[0.0, 0.0, 0.05], [0.0, 0.0, 5.0]],  # env 0: one foot not in contact
                [[0.0, 0.0, 10.0], [0.0, 0.0, 15.0]],  # env 1: both feet in contact
            ],
            device=self.device,
        )
        self.contact_sensor.data.net_forces_w_history = torch.tensor(
            [
                [
                    [[1.0, 0.0, 10.0], [0.0, 0.5, 5.0]],  # history step 1
                    [[2.0, 0.0, 12.0], [0.0, 1.0, 6.0]],  # history step 2
                ],
                [
                    [[0.0, 0.0, 0.0], [3.0, 4.0, 15.0]],  # history step 1
                    [[0.5, 0.5, 1.0], [2.0, 3.0, 12.0]],  # history step 2
                ],
            ],
            device=self.device,
        )

        # Create mock command manager
        self.env.command_manager = MagicMock()
        self.env.command_manager.get_command.return_value = torch.tensor(
            [
                [0.05, 0.0, 0.0, 0.8],  # env 0: low velocity, stance mode
                [0.5, 0.0, 0.0, 0.7],  # env 1: high velocity, not stance mode
            ],
            device=self.device,
        )

        # Create a proper mock scene that behaves like both dict and object
        scene_dict = {
            "robot": self.robot,
        }

        # Create a mock scene object that behaves like both a dict and has sensors attribute
        scene = MagicMock()
        # Make it behave like a dict
        scene.__getitem__ = lambda s, key: scene_dict[key]  # noqa: ARG005
        scene.get = lambda key, default=None: scene_dict.get(key, default)
        # Add sensors attribute
        scene.sensors = {"contact_sensor": self.contact_sensor}

        self.env.scene = scene

        # Create mock sensor config
        self.sensor_cfg = MagicMock()
        self.sensor_cfg.body_ids = torch.tensor([0, 1], device=self.device)
        self.sensor_cfg.name = "contact_sensor"


@unittest.skipIf(not APP_IS_READY, "App is not ready")
class TestTrackBaseHeight(TestRewardsBase):
    """Test cases for the track_base_height reward function."""

    def setUp(self) -> None:
        """Set up test fixtures for track_base_height tests."""
        super().setUp()

        # Update robot body positions for feet
        # Shape: [num_envs, num_bodies, 3]
        self.robot.data.body_pos_w = torch.tensor(
            [
                # Env 0: Two feet at different heights
                [[0.0, 0.1, 0.05], [0.0, -0.1, 0.0]],  # left_foot, right_foot
                # Env 1: Two feet at different heights
                [[0.0, 0.1, 0.1], [0.0, -0.1, 0.05]],  # left_foot, right_foot
            ],
            device=self.device,
        )

        # Setup root positions
        self.robot.data.root_pos_w = torch.tensor([[0.0, 0.0, 0.7], [0.0, 0.0, 0.8]], device=self.device)

        # Setup command
        self.env.command_manager.get_command.return_value = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.74],  # env 0: target height = 0.74
                [0.0, 0.0, 0.0, 0.78],  # env 1: target height = 0.78
            ],
            device=self.device,
        )

    def test_track_base_height_basic(self) -> None:
        """Test basic behavior of track_base_height reward function."""
        # Call the reward function with default parameters
        reward = mdp.track_base_height(self.env)

        # Expected rewards based on:
        # env0: base = 0.7, target = 0.74, error = 0.04
        # reward = exp(-0.04 / 0.25)
        # env1: base = 0.8, target = 0.78, error = 0.02
        # reward = exp(-0.02 / 0.25)
        expected_reward = torch.tensor(
            [
                torch.exp(torch.tensor(-0.04 / 0.25)),
                torch.exp(torch.tensor(-0.02 / 0.25)),
            ],
            device=self.device,
        )

        # Assert the reward is correct
        torch.testing.assert_close(reward, expected_reward)


@unittest.skipIf(not APP_IS_READY, "App is not ready")
class TestFeetStumble(TestRewardsBase):
    """Test cases for the feet_stumble reward function."""

    def test_feet_stumble(self) -> None:
        """Test basic behavior of feet_stumble reward function with different thresholds."""
        # Need to patch get_contact_sensor_cfg due to the internal implementation
        with patch(
            "agile.rl_env.mdp.rewards.get_contact_sensor_cfg",
            return_value=(self.contact_sensor, self.sensor_cfg),
        ):
            # Test with default threshold = 1.0
            sensor_cfg = SceneEntityCfg("contact_sensor")
            reward = mdp.feet_stumble(self.env, sensor_cfg, threshold=1.0)

            # Calculate expected reward
            # env0: max horizontal force = 2.0 (from x-component at history step 2, body 0)
            #       reward = max(2.0 - 1.0, 0) = 1.0
            # env1: max horizontal force = 5.0 (from sqrt(3^2 + 4^2) at history step 1, body 1)
            #       reward = max(5.0 - 1.0, 0) = 4.0
            expected_reward = torch.tensor([1.0, 4.0], device=self.device)

            # Assert the reward is correct
            torch.testing.assert_close(reward, expected_reward)

            # Test with high threshold = 10.0
            reward_high = mdp.feet_stumble(self.env, sensor_cfg, threshold=10.0)

            # With threshold = 10.0, no forces exceed it, so reward should be zero
            expected_reward_high = torch.tensor([0.0, 0.0], device=self.device)

            # Assert the reward is correct
            torch.testing.assert_close(reward_high, expected_reward_high)


@unittest.skipIf(not APP_IS_READY, "App is not ready")
class TestStandStill(TestRewardsBase):
    """Test cases for the stand_still reward function."""

    def setUp(self) -> None:
        """Set up test fixtures for stand_still tests."""
        super().setUp()

        # Update command manager to return different velocity commands
        # The stand_still function rewards when velocity commands are exactly zero (null command)
        self.env.command_manager.get_command.return_value = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.8],  # env 0: null command, stance mode
                [0.5, 0.0, 0.0, 0.8],  # env 1: non-null command, stance mode
                [0.0, 0.0, 0.0, 0.8],  # env 2: null command, stance mode
                [0.15, 0.0, 0.0, 0.8],  # env 3: non-null command, stance mode
            ],
            device=self.device,
        )

        # Update number of environments
        self.num_envs = 4
        self.env.num_envs = self.num_envs

        # Update contact forces for different scenarios
        self.contact_sensor.data.net_forces_w = torch.tensor(
            [
                [[0.0, 0.0, 0.05], [0.0, 0.0, 5.0]],  # env 0: one foot not in contact
                [[0.0, 0.0, 10.0], [0.0, 0.0, 15.0]],  # env 1: both feet in contact
                [[0.0, 0.0, 0.05], [0.0, 0.0, 0.05]],  # env 2: both feet not in contact
                [[0.0, 0.0, 10.0], [0.0, 0.0, 15.0]],  # env 3: both feet in contact
            ],
            device=self.device,
        )

    def test_stand_still(self) -> None:
        """Test the stand_still reward function with different scenarios."""
        sensor_cfg = SceneEntityCfg("contact_sensor")

        # Test with default parameters
        reward = mdp.stand_still(
            self.env,
            contact_threshold=0.1,  # one foot < threshold in env0
            sensor_cfg=sensor_cfg,
        )

        # Expected rewards based on current implementation:
        # env0: null command (True), one foot not in contact (1) -> reward = 1 * 1 = 1
        # env1: non-null command (False), both feet in contact (0) -> reward = 0 * 0 = 0
        # env2: null command (True), both feet not in contact (2) -> reward = 2 * 1 = 2
        # env3: non-null command (False), both feet in contact (0) -> reward = 0 * 0 = 0
        expected_reward = torch.tensor([1, 0, 2, 0], dtype=torch.float32, device=self.device)
        torch.testing.assert_close(reward, expected_reward)


@unittest.skipIf(not APP_IS_READY, "App is not ready")
class TestContactForcesL2(TestRewardsBase):
    """Test cases for the contact_forces_l2 reward function."""

    def setUp(self) -> None:
        """Set up test fixtures for contact_forces_l2 tests."""
        super().setUp()

        # Update for 3 environments
        self.num_envs = 3
        self.env.num_envs = self.num_envs

        # Update net_forces_w for 3 environments
        self.contact_sensor.data.net_forces_w = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],  # env 0
                [[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]],  # env 1
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],  # env 2
            ],
            device=self.device,
        )

    def test_contact_forces_l2(self) -> None:
        """Test contact_forces_l2 with varying contact forces."""
        sensor_cfg = SceneEntityCfg("contact_sensor")
        # Set a threshold of 0 to ensure all forces are penalized
        result = mdp.contact_forces_l2(self.env, contact_force_threshold=0.0, sensor_cfg=sensor_cfg)

        # Expected values:
        # Env 0: max(1^2, 0) + max(2^2, 0) = 1 + 4 = 5
        # Env 1: max(0, 3^2) + max(0, 0) = 9 + 0 = 9
        # Env 2: max(0, 0) + max(4^2, 0) = 0 + 16 = 16
        expected = torch.tensor([5.0, 9.0, 16.0], device=self.device)
        torch.testing.assert_close(result, expected)

    def test_high_threshold(self) -> None:
        """Test contact_forces_l2 with threshold higher than all forces."""
        sensor_cfg = SceneEntityCfg("contact_sensor")
        # Set a very high threshold so all forces are below it
        result = mdp.contact_forces_l2(self.env, contact_force_threshold=10.0, sensor_cfg=sensor_cfg)

        # All forces are below the threshold, so expected is all zeros
        expected = torch.zeros(self.num_envs, device=self.device)
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
