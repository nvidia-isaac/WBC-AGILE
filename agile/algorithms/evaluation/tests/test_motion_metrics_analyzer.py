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
import unittest
from pathlib import Path

import torch

from agile.algorithms.evaluation.motion_metrics_analyzer import MotionMetricsAnalyzer


class TestMotionMetricsAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.max_episode_length = 5
        self.metrics = MotionMetricsAnalyzer(max_episode_length=self.max_episode_length)

        # Create simulated data in the format that would come from EpisodeBuffer
        self.num_envs = 2
        self.num_joints = 3

        # Create a sample terminated data dictionary with two environments
        # First environment has 5 frames (successful), second has 3 frames (not successful)
        self.terminated_data = self._create_test_terminated_data()

    # noqa: C901
    def _create_test_terminated_data(self):
        """Create test data in the format returned by EpisodeBuffer."""  # noqa: D202
        # Create frame_counts tensor indicating valid frames for each env
        frame_counts = torch.tensor([self.max_episode_length, 3])  # Env 0: success (5), Env 1: failure (3)

        # Joint positions
        joint_pos = torch.zeros((self.max_episode_length, self.num_envs, self.num_joints, 1))
        for i in range(self.max_episode_length):
            for j in range(self.num_envs):
                if i < frame_counts[j]:
                    joint_pos[i, j] = torch.ones((self.num_joints, 1)) * (i + 1)

        # Joint velocities - first derivative of position
        joint_vel = torch.zeros((self.max_episode_length, self.num_envs, self.num_joints, 1))
        for i in range(self.max_episode_length):
            for j in range(self.num_envs):
                if i < frame_counts[j]:
                    joint_vel[i, j] = torch.ones((self.num_joints, 1))

        # Joint accelerations - second derivative
        joint_acc = torch.zeros((self.max_episode_length, self.num_envs, self.num_joints, 1))
        for i in range(self.max_episode_length):
            for j in range(self.num_envs):
                if i < frame_counts[j]:
                    # Make env 0 have smoother acceleration
                    if j == 0:
                        joint_acc[i, j] = torch.ones((self.num_joints, 1)) * 0.5
                    else:
                        joint_acc[i, j] = torch.ones((self.num_joints, 1)) * 2.0

        return {
            "frame_counts": frame_counts,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_acc": joint_acc,
        }

    # noqa: C901
    def _create_second_terminated_data(self, frame_counts):
        """Create another test data with specified frame counts."""  # noqa: D202
        num_envs = len(frame_counts)

        # Joint positions
        joint_pos = torch.zeros((self.max_episode_length, num_envs, self.num_joints, 1))
        for i in range(self.max_episode_length):
            for j in range(num_envs):
                if i < frame_counts[j]:
                    joint_pos[i, j] = torch.ones((self.num_joints, 1)) * (i + 2)  # Different values

        # Joint velocities
        joint_vel = torch.zeros((self.max_episode_length, num_envs, self.num_joints, 1))
        for i in range(self.max_episode_length):
            for j in range(num_envs):
                if i < frame_counts[j]:
                    joint_vel[i, j] = torch.ones((self.num_joints, 1)) * 1.5  # Different values

        # Joint accelerations
        joint_acc = torch.zeros((self.max_episode_length, num_envs, self.num_joints, 1))
        for i in range(self.max_episode_length):
            for j in range(num_envs):
                if i < frame_counts[j]:
                    # Different acceleration values for this data
                    joint_acc[i, j] = torch.ones((self.num_joints, 1)) * (j + 1)

        return {
            "frame_counts": torch.tensor(frame_counts),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_acc": joint_acc,
        }

    def test_initialization(self):
        """Test if SmoothnessMetrics is initialized correctly."""
        self.assertEqual(self.metrics.max_episode_length, self.max_episode_length)
        self.assertEqual(self.metrics.num_envs_evaluated, 0)
        self.assertEqual(self.metrics.num_envs_successful, 0)
        self.assertEqual(self.metrics.success_rate, 0.0)

        # Check that default metrics are registered
        self.assertIn("mean_joint_acc", self.metrics._compute_functions)
        self.assertIn("max_joint_acc", self.metrics._compute_functions)
        self.assertIn("mean_acc_rate", self.metrics._compute_functions)
        self.assertIn("max_acc_rate", self.metrics._compute_functions)

    def test_update_with_terminated_data(self):
        """Test updating metrics with terminated data."""
        # Add the terminated data to metrics
        self.metrics.update(self.terminated_data)

        # Check that environments were counted correctly
        self.assertEqual(self.metrics.num_envs_evaluated, 2)
        self.assertEqual(self.metrics.num_envs_successful, 1)

        # At this point metrics haven't been concluded, so metrics dicts are empty
        self.assertEqual(len(self.metrics._metrics), 0)
        self.assertEqual(len(self.metrics._success_metrics), 0)

        # But metric data should be collected
        self.assertTrue(len(self.metrics._metrics_data["mean_joint_acc"]["values"]) > 0)
        self.assertTrue(len(self.metrics._success_metrics_data["mean_joint_acc"]["values"]) > 0)

    def test_multiple_updates_with_different_frame_counts(self):
        """Test updating metrics multiple times with different frame counts."""
        # First update with the initial test data (2 envs: 1 success, 1 failure)
        self.metrics.update(self.terminated_data)

        # Check initial counts
        self.assertEqual(self.metrics.num_envs_evaluated, 2)
        self.assertEqual(self.metrics.num_envs_successful, 1)

        # Create second terminated data with 3 environments
        # Env 0: success (5), Env 1: failure (2), Env 2: success (5)
        second_data = self._create_second_terminated_data([self.max_episode_length, 2, self.max_episode_length])

        # Update metrics with second data
        self.metrics.update(second_data)

        # Check updated counts (should be cumulative)
        self.assertEqual(self.metrics.num_envs_evaluated, 5)  # 2 from first update + 3 from second
        self.assertEqual(self.metrics.num_envs_successful, 3)  # 1 from first update + 2 from second

        # Create third terminated data with 1 environment that fails
        third_data = self._create_second_terminated_data([4])  # One env with 4 frames (failure)

        # Update metrics with third data
        self.metrics.update(third_data)

        # Check final counts
        self.assertEqual(self.metrics.num_envs_evaluated, 6)  # 5 from before + 1 new
        self.assertEqual(self.metrics.num_envs_successful, 3)  # No new successes

        # Conclude metrics
        self.metrics.conclude()

        # Check success rate
        self.assertEqual(self.metrics.success_rate, 3 / 6)  # 3 successes out of 6 total envs

        # Verify metrics were calculated
        self.assertIn("mean_joint_acc", self.metrics._metrics)
        self.assertIn("mean_joint_acc", self.metrics._success_metrics)

        # Check that we have more data points for all envs than successful envs
        all_data_points = len(self.metrics._metrics_data["mean_joint_acc"]["values"])
        success_data_points = len(self.metrics._success_metrics_data["mean_joint_acc"]["values"])
        self.assertGreater(all_data_points, success_data_points)

        # The success data points should match our successful env count
        self.assertEqual(success_data_points, 3)

    def test_conclude_metrics(self):
        """Test concluding metrics after processing data."""
        # Add data and conclude
        self.metrics.update(self.terminated_data)
        self.metrics.conclude()

        # Check success rate calculation
        self.assertEqual(self.metrics.success_rate, 0.5)  # 1 out of 2 envs were successful

        # Check that metrics were calculated for all environments and successful environments
        self.assertIn("mean_joint_acc", self.metrics._metrics)
        self.assertIn("max_joint_acc", self.metrics._metrics)
        self.assertIn("mean_acc_rate", self.metrics._metrics)
        self.assertIn("max_acc_rate", self.metrics._metrics)

        self.assertIn("mean_joint_acc", self.metrics._success_metrics)
        self.assertIn("max_joint_acc", self.metrics._success_metrics)

        # Since the successful environment had lower acceleration values,
        # success metrics should have lower acceleration values than all metrics
        self.assertLess(
            self.metrics._success_metrics["mean_joint_acc"],
            self.metrics._metrics["mean_joint_acc"],
        )

    def test_custom_metric(self):
        """Test registering and computing a custom metric."""

        # Define a custom metric function
        def compute_min_acc(full_data, env_data):
            env_idx = env_data["env_idx"]
            num_frames = env_data["num_frames"]

            if "joint_acc" not in full_data or full_data["joint_acc"].numel() == 0:
                return None, 0

            acc = full_data["joint_acc"][:num_frames, env_idx]
            min_acc = torch.min(torch.abs(acc)).item()
            return min_acc, 1

        # Register the custom metric
        self.metrics.register_metric("min_joint_acc", compute_min_acc)

        # Process data and conclude
        self.metrics.update(self.terminated_data)
        self.metrics.conclude()

        # Check that the custom metric was computed
        self.assertIn("min_joint_acc", self.metrics._metrics)
        self.assertIn("min_joint_acc", self.metrics._success_metrics)

    def test_get_metrics(self):
        """Test getting all metrics as a dictionary."""
        # Process data and conclude
        self.metrics.update(self.terminated_data)
        self.metrics.conclude()

        # Get metrics dictionary
        metrics_dict = self.metrics.get_metrics()

        # Check that it contains all the expected keys
        self.assertIn("success_rate", metrics_dict)
        self.assertIn("metrics", metrics_dict)
        self.assertIn("success_metrics", metrics_dict)

        # Check that success rate is correct
        self.assertEqual(metrics_dict["success_rate"], 0.5)

    def test_save_metrics(self):
        """Test saving metrics to a file."""
        # Process data and conclude
        self.metrics.update(self.terminated_data)
        self.metrics.conclude()

        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save metrics to the temp directory
            self.metrics.save(temp_dir)

            # Check that a file was created
            files = list(Path(temp_dir).glob("smoothness_metrics_*.json"))
            self.assertEqual(len(files), 1)

    def test_joint_group_env_indexing(self):
        """Test proper environment indexing in joint groups."""
        # Create metrics analyzer with several joint groups using different environment indices
        joint_groups = {
            "first_joint": [0],  # Just the first joint
            "second_joint": [1],  # Just the second joint
            "all_joints": [0, 1, 2],  # All joints
        }

        metrics = MotionMetricsAnalyzer(max_episode_length=self.max_episode_length, joint_groups=joint_groups)

        # Create data with multiple environments with different acceleration values
        # to verify that environment index is correctly handled
        data = self._create_second_terminated_data([self.max_episode_length, self.max_episode_length])

        # Update metrics with multi-environment data
        # This would previously fail if env_idx wasn't reset to 0 for group metrics
        try:
            metrics.update(data)
            metrics.conclude()

            # First verify that metrics were calculated for each group
            self.assertIn("first_joint_mean_joint_acc", metrics._metrics)
            self.assertIn("second_joint_mean_joint_acc", metrics._metrics)
            self.assertIn("all_joints_mean_joint_acc", metrics._metrics)

            # Now check metrics dict to make sure everything was calculated properly
            metrics_dict = metrics.get_metrics()

            # Extract group metrics
            group_metrics = metrics_dict["metrics"]
            self.assertIn("first_joint", group_metrics)
            self.assertIn("second_joint", group_metrics)
            self.assertIn("all_joints", group_metrics)

            # Success flag indicates the test passed if we got here without errors
            success = True
        except IndexError as e:
            # This would happen if env_idx is not reset to 0 when computing group metrics
            self.fail(f"Failed to handle environment indexing in group metrics: {e}")
            success = False

        # If we made it here, the environment index handling worked correctly
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
