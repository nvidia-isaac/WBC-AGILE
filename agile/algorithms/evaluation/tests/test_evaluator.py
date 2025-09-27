#!/usr/bin/env python

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


import tempfile

from isaaclab.app import AppLauncher

# launch the simulator
try:
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    APP_IS_READY = True
except Exception:
    APP_IS_READY = False

import unittest
from unittest.mock import MagicMock, patch

import torch

if APP_IS_READY:
    from agile.algorithms.evaluation.evaluator import PolicyEvaluator


@unittest.skipIf(not APP_IS_READY, "Application is not ready")
class TestPolicyEvaluator(unittest.TestCase):
    """Test the PolicyEvaluator class."""

    def setUp(self):
        """Set up an environment and evaluator for testing."""
        # Use actual values for everything to avoid type issues
        self.device = torch.device("cpu")
        self.num_envs = 2
        self.max_episode_length = 5
        self.num_joints = 3

        # Create a real environment-like object with all required attributes
        class TestEnv:
            def __init__(self, num_envs, device, max_episode_length):
                self.num_envs = num_envs
                self.device = device
                self.max_episode_length = max_episode_length

                # Add cfg attribute for PolicyEvaluator
                class MockCfg:
                    def __init__(self):
                        self.joint_groups = {"upper_body": ["joint_0", "joint_1"]}

                self.cfg = MockCfg()

                # Create real observation manager with required methods
                class ObsManager:
                    def __init__(self):
                        self.active_terms = {
                            "eval": [
                                "joint_pos",
                                "joint_vel",
                                "joint_acc",
                                "root_lin_vel",
                                "root_ang_vel",
                                "root_lin_vel_robot",
                                "root_rot",
                                "root_pos",
                                "commands",
                                "actions",
                            ]
                        }
                        self.group_obs_term_dim = {
                            "eval": [3, 3, 3, 3, 3, 3, 4, 3, 4, 3]  # dims for each term (added root_lin_vel_robot)
                        }

                self.observation_manager = ObsManager()

                # Mock robot with find_joints method and joint_names attribute
                class MockRobot:
                    def __init__(self):
                        self.joint_names = ["joint_0", "joint_1", "joint_2"]

                    def find_joints(self, patterns: list[str]) -> tuple[list[int], list[str]]:
                        # Match patterns to joints - simple implementation for testing
                        matched_indices = []
                        matched_names = []
                        for pattern in patterns:
                            for idx, name in enumerate(self.joint_names):
                                # Simple matching: exact match or if pattern is in name
                                if pattern == name or pattern.replace(".*", "") in name:
                                    if idx not in matched_indices:
                                        matched_indices.append(idx)
                                        matched_names.append(name)
                        return [matched_indices, matched_names]

                # Mock scene dictionary with robot
                self.scene = {"robot": MockRobot()}

                # Mock cfg attribute with required structure for PolicyEvaluator
                class MockCfg:
                    class Observations:
                        class Policy:
                            class JointPos:
                                params = {"asset_cfg": type("obj", (), {"joint_names": ["joint1", "joint2"]})}

                            joint_pos_upper = JointPos()
                            joint_pos_lower = JointPos()

                        policy = Policy()

                    observations = Observations()

                self.cfg = MockCfg()

        # Create environment with actual integers
        self.env = TestEnv(
            num_envs=self.num_envs,
            device=self.device,
            max_episode_length=self.max_episode_length,
        )

        # Create temporary directory for metrics
        self.temp_dir = tempfile.mkdtemp()

        # Create evaluator
        self.evaluator = PolicyEvaluator(
            self.env,
            task_name="test_task",
            metrics_path=self.temp_dir,
            total_envs_target=4,  # Target 4 environments (2 episodes with 2 envs each)
        )

        # Create real observation data
        # Total dims: 3 + 3 + 3 + 3 + 3 + 3 + 4 + 3 + 4 + 3 = 32 (includes root_lin_vel_robot)
        self.obs_data = torch.ones((self.num_envs, 32))
        self.info = {
            "observations": {"eval": self.obs_data},
            "termination_conditions": {
                "timeout": torch.zeros(self.num_envs, dtype=torch.bool),
                "failure": torch.zeros(self.num_envs, dtype=torch.bool),
            },
        }

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test evaluator initialization."""
        self.assertEqual(self.evaluator._num_envs, self.num_envs)
        self.assertEqual(self.evaluator._max_episode_len, self.max_episode_length)
        self.assertEqual(self.evaluator._metrics_path, self.temp_dir)
        self.assertEqual(self.evaluator._num_envs_evaluated, 0)
        self.assertEqual(self.evaluator._total_envs_target, 4)

    def test_extract_frame_data(self):
        """Test extracting frame data from environment info."""
        # Create real frame data in the info dict
        joint_pos = torch.ones((self.num_envs, self.num_joints, 1))
        joint_vel = torch.ones((self.num_envs, self.num_joints, 1)) * 0.5
        joint_acc = torch.ones((self.num_envs, self.num_joints, 1)) * 0.1

        # Use patch to intercept the _get_term_obs_data method calls and return real tensors
        # This avoids needing to precisely match the observation structure
        term_data_map = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_acc": joint_acc,
            "root_pos": torch.ones((self.num_envs, 3)),
            "root_rot": torch.ones((self.num_envs, 4)),
            "root_lin_vel": torch.ones((self.num_envs, 3)) * 0.1,
            "root_ang_vel": torch.ones((self.num_envs, 3)) * 0.05,
            "commands": torch.ones((self.num_envs, 4)),
            "actions": torch.ones((self.num_envs, self.num_joints, 1)) * 0.2,
        }

        def get_term_side_effect(group, term, obs, manager):
            return term_data_map.get(term)

        with patch.object(self.evaluator, "_get_term_obs_data", side_effect=get_term_side_effect) as mock_get_term:
            frame_data = self.evaluator._extract_frame_data(self.info)

            # Should call get_term_obs_data for 9 required fields (root_lin_vel_robot is generated via transformation)
            self.assertEqual(mock_get_term.call_count, 9)

            # Should include 10 fields in the result (9 extracted + 1 generated via transformation)
            self.assertEqual(len(frame_data), 10)

            # Verify field values
            self.assertTrue(torch.all(torch.eq(frame_data["joint_pos"], joint_pos)))
            self.assertTrue(torch.all(torch.eq(frame_data["joint_vel"], joint_vel)))
            self.assertTrue(torch.all(torch.eq(frame_data["joint_acc"], joint_acc)))

            # Verify that root_lin_vel_robot was generated via transformation
            self.assertIn("root_lin_vel_robot", frame_data)
            self.assertEqual(frame_data["root_lin_vel_robot"].shape, (self.num_envs, 3))

    def test_collect_and_evaluate(self):
        """Test the collection and evaluation process using actual tensors."""
        # Mock metrics since we're testing the evaluator logic, not metrics behavior
        self.evaluator._metrics = MagicMock()
        self.evaluator._metrics.num_envs_successful = 0

        # Create a patch for extract_frame_data to return controlled data
        frame_data = {
            "joint_pos": torch.ones((self.num_envs, self.num_joints, 1)),
            "joint_vel": torch.ones((self.num_envs, self.num_joints, 1)) * 0.5,
            "joint_acc": torch.ones((self.num_envs, self.num_joints, 1)) * 0.1,
            "root_pos": torch.ones((self.num_envs, 3)),
            "root_rot": torch.ones((self.num_envs, 4)),
            "root_lin_vel": torch.ones((self.num_envs, 3)) * 0.1,
            "root_lin_vel_robot": torch.ones((self.num_envs, 3)) * 0.1,  # Robot frame velocity
            "root_ang_vel": torch.ones((self.num_envs, 3)) * 0.05,
            "commands": torch.ones((self.num_envs, 4)),
            "actions": torch.ones((self.num_envs, self.num_joints, 1)) * 0.2,
        }

        with patch.object(self.evaluator, "_extract_frame_data", return_value=frame_data):
            # First step with no terminations
            dones = torch.zeros(self.num_envs, dtype=torch.bool)
            is_complete = self.evaluator.collect(dones, self.info)

            # Should not be complete yet
            self.assertFalse(is_complete)
            self.assertEqual(self.evaluator._num_envs_evaluated, 0)

            # Second step with one environment done
            dones = torch.tensor([True, False], device=self.device)
            is_complete = self.evaluator.collect(dones, self.info)

            # Should have counted one environment
            self.assertEqual(self.evaluator._num_envs_evaluated, 1)
            self.assertFalse(is_complete)

            # Third step with both environments done
            dones = torch.ones(self.num_envs, dtype=torch.bool)
            is_complete = self.evaluator.collect(dones, self.info)

            # Should have counted both environments
            self.assertEqual(self.evaluator._num_envs_evaluated, 3)
            self.assertFalse(is_complete)

            # Final step with one more environment done
            dones = torch.tensor([True, False], device=self.device)
            is_complete = self.evaluator.collect(dones, self.info)

            # Evaluation should be complete
            self.assertEqual(self.evaluator._num_envs_evaluated, 4)
            self.assertTrue(is_complete)

    def test_conclude(self):
        """Test the conclude method."""
        # Mock metrics to isolate testing to just the evaluator functionality
        self.evaluator._metrics = MagicMock()
        self.evaluator._metrics.get_metrics.return_value = {"success_rate": 0.75}

        # Call conclude and check results
        results = self.evaluator.conclude()

        # Verify metrics methods were called
        self.evaluator._metrics.conclude.assert_called_once()
        self.evaluator._metrics.print.assert_called_once()
        self.evaluator._metrics.save.assert_called_once_with(self.temp_dir, "metrics.json")

        # Check results
        self.assertIn("success_rate", results)
        self.assertEqual(results["success_rate"], 0.75)

        # Progress bar should be closed
        self.assertIsNone(self.evaluator._pbar)

    @patch("agile.algorithms.evaluation.evaluator.EpisodeBuffer")
    def test_termination_limiting(self, mock_episode_buffer):
        """Test limiting terminated environments to match the target count.

        This test verifies that when more environments are terminated than needed
        to reach the target evaluation count, only the required number of terminations
        are processed.
        """
        # Setup mock for EpisodeBuffer
        mock_add_frame = MagicMock()
        mock_episode_buffer.return_value.add_frame = mock_add_frame

        # Create evaluator with target of 5 environments
        total_envs_target = 5
        evaluator = PolicyEvaluator(
            env=self.env,
            task_name="test_task",
            metrics_path=self.temp_dir,
            total_envs_target=total_envs_target,
        )

        # Scenario 1: Starting from 0 evaluated environments
        # Create a done tensor with more terminations than the total target
        dones = torch.tensor([True, True, True, True, True, True, True])  # 7 dones
        # Need proper observation structure for _extract_frame_data()
        info = {"observations": {"eval": self.obs_data}}

        # Should only use the first 5 terminations
        evaluator.collect(dones, info)
        args, kwargs = mock_add_frame.call_args
        terminated_ids = args[1]  # Second arg should be terminated_ids
        self.assertEqual(len(terminated_ids), 5)  # Should limit to 5
        self.assertTrue(torch.equal(terminated_ids, torch.tensor([0, 1, 2, 3, 4])))

        # Now reset and create a new evaluator to test partial evaluation
        mock_add_frame.reset_mock()
        evaluator = PolicyEvaluator(
            env=self.env,
            task_name="test_task",
            metrics_path=self.temp_dir,
            total_envs_target=total_envs_target,
        )

        # Scenario 2: Already evaluated 3 environments, need 2 more to reach target of 5
        evaluator._num_envs_evaluated = 3

        # Create a done tensor with more terminations than needed
        dones = torch.tensor([True, True, True, True])  # 4 dones, but only need 2
        evaluator.collect(dones, info)

        # Should only use first 2 terminations to reach total target of 5
        args, kwargs = mock_add_frame.call_args
        terminated_ids = args[1]
        self.assertEqual(len(terminated_ids), 2)  # 5 target - 3 already = 2 needed
        self.assertTrue(torch.equal(terminated_ids, torch.tensor([0, 1])))

        # Verify evaluation is now complete
        self.assertEqual(evaluator._num_envs_evaluated, 5)

        # Scenario 3: Try to evaluate more environments after reaching target
        mock_add_frame.reset_mock()
        dones = torch.tensor([True, True])
        result = evaluator.collect(dones, info)

        # Should return True indicating evaluation is complete
        self.assertTrue(result)

        # Shouldn't have called add_frame again since we're already done
        mock_add_frame.assert_not_called()

    def test_build_joint_groups_simplified_format(self):
        """Test joint groups with simplified format (list of patterns)."""
        # Test with simplified config format (list of patterns)
        joint_group_config = {
            "upper_body": ["joint_0", "joint_1"],
            "lower_body": ["joint_2"],
        }

        evaluator = PolicyEvaluator(
            self.env,
            task_name="test_task",
            metrics_path=self.temp_dir,
            total_envs_target=2,
            joint_group_config=joint_group_config,
        )

        # Check that joint groups were built correctly
        self.assertIn("upper_body", evaluator._joint_groups)
        self.assertIn("lower_body", evaluator._joint_groups)

        # Upper body should have joints 0 and 1 (joint_0, joint_1)
        self.assertEqual(evaluator._joint_groups["upper_body"], [0, 1])

        # Lower body should have joint 2 (joint_2)
        self.assertEqual(evaluator._joint_groups["lower_body"], [2])


if __name__ == "__main__":
    unittest.main(exit=False)
    # Clean up Isaac Sim if we initialized it
    if simulation_app is not None:
        simulation_app.close()
