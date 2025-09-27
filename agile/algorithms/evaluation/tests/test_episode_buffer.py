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


import unittest

import torch

from agile.algorithms.evaluation.episode_buffer import EpisodeBuffer, Frame


class TestEpisodeBuffer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.num_envs = 2
        self.max_episode_length = 10
        self.device = torch.device("cpu")
        self.buffer = EpisodeBuffer(
            num_envs=self.num_envs,
            max_episode_length=self.max_episode_length,
            device=self.device,
        )

    def test_initialization(self):
        """Test if EpisodeBuffer is initialized correctly."""
        self.assertEqual(self.buffer.num_envs, self.num_envs)
        self.assertEqual(self.buffer.max_episode_length, self.max_episode_length)
        self.assertEqual(self.buffer.device, self.device)
        self.assertFalse(self.buffer._initialized)
        self.assertEqual(self.buffer._num_frames.shape, torch.Size([self.num_envs]))
        self.assertEqual(len(self.buffer._data), 0)
        self.assertTrue(
            torch.equal(
                self.buffer._env_indices,
                torch.arange(self.num_envs, device=self.device),
            )
        )

    def test_add_frame(self):
        """Test adding frames to the buffer."""
        # Create a test frame
        joint_pos = torch.ones((self.num_envs, 3, 1))  # [num_envs, num_joints, 1]
        joint_vel = torch.ones((self.num_envs, 3, 1)) * 2  # [num_envs, num_joints, 1]
        root_pos = torch.ones((self.num_envs, 3)) * 3  # [num_envs, 3]

        frame = Frame(joint_pos=joint_pos, joint_vel=joint_vel, root_pos=root_pos)

        # Add frame to buffer
        result = self.buffer.add_frame(frame)

        # The result should be None when no environments are terminated
        self.assertIsNone(result)

        # Check if buffer was initialized
        self.assertTrue(self.buffer._initialized)

        # Check if data was stored correctly
        self.assertIn("joint_pos", self.buffer._data)
        self.assertIn("joint_vel", self.buffer._data)
        self.assertIn("root_pos", self.buffer._data)

        # Check if frame counters were incremented
        self.assertTrue(torch.all(self.buffer._num_frames == 1))

        # Check stored data shapes
        self.assertEqual(
            self.buffer._data["joint_pos"].shape,
            torch.Size([self.max_episode_length, self.num_envs, 3, 1]),
        )

        # Check stored values (only first frame has values)
        torch.testing.assert_close(self.buffer._data["joint_pos"][0], joint_pos)

    def test_terminated_env(self):
        """Test handling of a terminated environment."""
        # Create and add initial frame
        joint_pos = torch.ones((self.num_envs, 3, 1))
        frame = Frame(joint_pos=joint_pos)
        self.buffer.add_frame(frame)

        # Add another frame and terminate first environment
        frame2 = Frame(joint_pos=torch.ones((self.num_envs, 3, 1)) * 2)
        terminated_ids = torch.tensor([0], device=self.device)
        terminated_data = self.buffer.add_frame(frame2, terminated_ids)

        # Check if terminated data was correctly extracted
        self.assertIsNotNone(terminated_data)
        self.assertIn("joint_pos", terminated_data)
        self.assertIn("frame_counts", terminated_data)

        # Check the shape of the terminated data - should have the full buffer size
        self.assertEqual(
            terminated_data["joint_pos"].shape,
            torch.Size([self.max_episode_length, 1, 3, 1]),
        )

        # Check the frame counts - should be 2 (initial frame + termination frame)
        self.assertEqual(terminated_data["frame_counts"][0].item(), 2)

        # Check values for the first two frames (only valid ones)
        torch.testing.assert_close(terminated_data["joint_pos"][0, 0], torch.ones((3, 1)))
        torch.testing.assert_close(terminated_data["joint_pos"][1, 0], torch.ones((3, 1)) * 2)

        # The frame counter for terminated environment should be reset
        self.assertEqual(self.buffer._num_frames[0].item(), 0)

        # The frame counter for non-terminated environment should be incremented
        self.assertEqual(self.buffer._num_frames[1].item(), 2)

    def test_multiple_frames(self):
        """Test adding multiple frames and handling termination."""
        # Create test frames and add them sequentially
        for i in range(5):
            joint_pos = torch.ones((self.num_envs, 3, 1)) * (i + 1)
            frame = Frame(joint_pos=joint_pos)
            self.buffer.add_frame(frame)

        # Check if frame counters were incremented properly
        self.assertTrue(torch.all(self.buffer._num_frames == 5))

        # Terminate the first environment only
        terminated_ids = torch.tensor([0], device=self.device)
        frame6 = Frame(joint_pos=torch.ones((self.num_envs, 3, 1)) * 6)
        terminated_data = self.buffer.add_frame(frame6, terminated_ids)

        # Check if terminated data was correctly extracted
        self.assertIsNotNone(terminated_data)
        self.assertIn("joint_pos", terminated_data)
        self.assertIn("frame_counts", terminated_data)

        # Check frame count tensor - should be 6 (5 existing + 1 at termination)
        self.assertEqual(terminated_data["frame_counts"][0].item(), 6)

        # Check the shape of terminated data (should be full buffer size)
        self.assertEqual(
            terminated_data["joint_pos"].shape,
            torch.Size([self.max_episode_length, 1, 3, 1]),
        )

        # Check values for each valid frame (1-6) for the terminated environment
        for i in range(6):
            torch.testing.assert_close(terminated_data["joint_pos"][i, 0], torch.ones((3, 1)) * (i + 1))

        # The frame counter for terminated environment should be reset
        self.assertEqual(self.buffer._num_frames[0].item(), 0)

        # The frame counter for non-terminated environment should be incremented
        self.assertEqual(self.buffer._num_frames[1].item(), 6)

    def test_second_env_termination(self):
        """Test handling termination of the second environment."""
        # Create and add frames
        joint_pos1 = torch.ones((self.num_envs, 3, 1))
        joint_pos2 = torch.ones((self.num_envs, 3, 1)) * 2

        self.buffer.add_frame(Frame(joint_pos=joint_pos1))

        # Terminate only the second environment
        terminated_ids = torch.tensor([1], device=self.device)
        terminated_data = self.buffer.add_frame(Frame(joint_pos=joint_pos2), terminated_ids)

        # Check if terminated data was correctly extracted
        self.assertIsNotNone(terminated_data)
        self.assertIn("joint_pos", terminated_data)
        self.assertIn("frame_counts", terminated_data)

        # Check the shape and content of terminated data
        self.assertEqual(
            terminated_data["joint_pos"].shape,
            torch.Size([self.max_episode_length, 1, 3, 1]),
        )

        # Check the frame count
        self.assertEqual(terminated_data["frame_counts"][0].item(), 2)

        # Check values for first and second frame for the terminated environment (env 1)
        torch.testing.assert_close(terminated_data["joint_pos"][0, 0], torch.ones((3, 1)))
        torch.testing.assert_close(terminated_data["joint_pos"][1, 0], torch.ones((3, 1)) * 2)

        # The frame counter for non-terminated environment should be incremented
        self.assertEqual(self.buffer._num_frames[0].item(), 2)

        # The frame counter for terminated environment should be reset
        self.assertEqual(self.buffer._num_frames[1].item(), 0)

    def test_multiple_terminations(self):
        """Test handling of multiple terminated environments at once."""
        # Create and add initial frame
        joint_pos = torch.ones((self.num_envs, 3, 1))
        frame = Frame(joint_pos=joint_pos)
        self.buffer.add_frame(frame)

        # Add another frame and terminate both environments
        frame2 = Frame(joint_pos=torch.ones((self.num_envs, 3, 1)) * 2)
        terminated_ids = torch.tensor([0, 1], device=self.device)
        terminated_data = self.buffer.add_frame(frame2, terminated_ids)

        # Check if terminated data was correctly extracted
        self.assertIsNotNone(terminated_data)
        self.assertIn("joint_pos", terminated_data)
        self.assertIn("frame_counts", terminated_data)

        # Check the shape of the terminated data - full buffer size for both envs
        self.assertEqual(
            terminated_data["joint_pos"].shape,
            torch.Size([self.max_episode_length, 2, 3, 1]),
        )

        # Check frame counts - both should be 2 frames
        self.assertTrue(torch.all(terminated_data["frame_counts"] == 2))

        # Check values for both environments across the first two frames (valid ones)
        for env_idx in range(2):
            # First frame should have value 1
            torch.testing.assert_close(terminated_data["joint_pos"][0, env_idx], torch.ones((3, 1)))
            # Second frame should have value 2
            torch.testing.assert_close(terminated_data["joint_pos"][1, env_idx], torch.ones((3, 1)) * 2)

        # Both environment frame counters should be reset
        self.assertTrue(torch.all(self.buffer._num_frames == 0))

    def test_multiple_envs_different_lengths(self):
        """Test handling multiple terminated environments with different frame counts."""
        # Add 3 frames for both environments
        for i in range(3):
            joint_pos = torch.ones((self.num_envs, 3, 1)) * (i + 1)
            frame = Frame(joint_pos=joint_pos)
            self.buffer.add_frame(frame)

        # Second environment continues with 2 more frames
        for i in range(3, 5):
            # Create a tensor just for environment 1
            joint_pos = torch.zeros((self.num_envs, 3, 1))
            joint_pos[1] = torch.ones((3, 1)) * (i + 1)

            # Terminate environment 0 if we're at the first additional frame
            terminated_ids = torch.tensor([0], device=self.device) if i == 3 else None
            result = self.buffer.add_frame(Frame(joint_pos=joint_pos), terminated_ids)

            # Check env 0 termination
            if i == 3:
                # Check if terminated data was correctly extracted
                self.assertIsNotNone(result)
                self.assertIn("joint_pos", result)
                self.assertIn("frame_counts", result)

                # Check frame count - should be 4 frames (3 + the termination frame)
                self.assertEqual(result["frame_counts"][0].item(), 4)

                # Check the shape of terminated data - full buffer size
                self.assertEqual(
                    result["joint_pos"].shape,
                    torch.Size([self.max_episode_length, 1, 3, 1]),
                )

                # First 3 frames should be (1,2,3), 4th frame should be zeros (for env 0)
                for j in range(3):
                    torch.testing.assert_close(result["joint_pos"][j, 0], torch.ones((3, 1)) * (j + 1))
                # 4th frame should have zeros (the value at termination for env 0)
                torch.testing.assert_close(result["joint_pos"][3, 0], torch.zeros((3, 1)))

                # Only env 0 should be reset to 0, env 1 should continue from 4
                self.assertEqual(self.buffer._num_frames[0].item(), 0)
                self.assertEqual(self.buffer._num_frames[1].item(), 4)

        # Now terminate environment 1 as well with 6th frame
        terminated_ids = torch.tensor([1], device=self.device)
        frame6 = Frame(joint_pos=torch.ones((self.num_envs, 3, 1)) * 6)

        # By now, env 0 should have accumulated 2 new frames (if it was properly reset)
        self.assertEqual(self.buffer._num_frames[0].item(), 1)  # Just 1 frame from the previous loop iteration
        self.assertEqual(self.buffer._num_frames[1].item(), 5)  # 5 frames accumulated so far

        terminated_data = self.buffer.add_frame(frame6, terminated_ids)

        # Check if terminated data was correctly extracted
        self.assertIsNotNone(terminated_data)
        self.assertIn("joint_pos", terminated_data)
        self.assertIn("frame_counts", terminated_data)

        # Check frame count - environment 1 should have 6 frames
        self.assertEqual(terminated_data["frame_counts"][0].item(), 6)

        # Check the shape of terminated data - full buffer size
        self.assertEqual(
            terminated_data["joint_pos"].shape,
            torch.Size([self.max_episode_length, 1, 3, 1]),
        )

        # First 3 frames should be (1,2,3) for env 1
        for i in range(3):
            torch.testing.assert_close(terminated_data["joint_pos"][i, 0], torch.ones((3, 1)) * (i + 1))

        # Frames 4-5 should be (4,5) for env 1
        for i in range(3, 5):
            torch.testing.assert_close(terminated_data["joint_pos"][i, 0], torch.ones((3, 1)) * (i + 1))

        # Frame 6 should be 6 for env 1
        torch.testing.assert_close(terminated_data["joint_pos"][5, 0], torch.ones((3, 1)) * 6)

        # Env 1 should be reset, env 0 should be incremented
        self.assertEqual(self.buffer._num_frames[0].item(), 2)  # Was 1 before, now 2
        self.assertEqual(self.buffer._num_frames[1].item(), 0)  # Reset to 0 after termination

    def test_zero_frames(self):
        """Test that terminating an environment with zero frames works correctly."""
        # Terminate without adding any frames
        terminated_ids = torch.tensor([0], device=self.device)
        result = self.buffer.add_frame(Frame(joint_pos=torch.ones((self.num_envs, 3, 1))), terminated_ids)

        # Result should include frame_counts with 1 for the terminated environment
        self.assertIn("frame_counts", result)
        self.assertEqual(result["frame_counts"][0].item(), 1)

        # Shape should be full buffer size
        self.assertEqual(result["joint_pos"].shape, torch.Size([self.max_episode_length, 1, 3, 1]))

        # First frame should be the one we just added
        torch.testing.assert_close(result["joint_pos"][0, 0], torch.ones((3, 1)))

        # Frame counter should be reset for terminated environment
        self.assertEqual(self.buffer._num_frames[0].item(), 0)

        # Frame counter should be incremented for non-terminated environment
        self.assertEqual(self.buffer._num_frames[1].item(), 1)

        # Now add a frame to verify env 0 increments correctly
        frame2 = Frame(joint_pos=torch.ones((self.num_envs, 3, 1)) * 2)
        self.buffer.add_frame(frame2)

        # Both environments should increment
        self.assertEqual(self.buffer._num_frames[0].item(), 1)  # Restarted from 0
        self.assertEqual(self.buffer._num_frames[1].item(), 2)  # Continued from 1


if __name__ == "__main__":
    unittest.main()
