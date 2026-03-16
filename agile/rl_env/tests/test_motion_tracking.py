# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for motion tracking remapping logic and helper functions.

These tests only import from ``agile.common`` (Isaac-Lab-free) so they can
run without initialising the Isaac Sim runtime.

Run via: ${ISAACLAB_PATH}/isaaclab.sh -p agile/rl_env/tests/test_motion_tracking.py
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch

from agile.common.motion_data import MotionData

# ---------------------------------------------------------------------------
# Test-local helpers (avoid importing from agile.rl_env.mdp which needs
# Isaac Lab / omni.physics at import time)
# ---------------------------------------------------------------------------


class _MockCommand:
    """Minimal mock for MotionCommand with only cfg.body_names."""

    def __init__(self, body_names: list[str]):
        self.cfg = type("Cfg", (), {"body_names": body_names})()


def _get_body_indices(command: _MockCommand, body_names: list[str] | None) -> list[int]:
    """Local copy of ``motion_tracking_rewards._get_body_indices`` to avoid Isaac Lab imports."""
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _make_npz(num_frames: int, num_joints: int, num_bodies: int, *, body_pos_fn=None) -> str:
    """Create a temporary .npz and return its path.  Caller must unlink."""
    joint_pos = np.zeros((num_frames, num_joints), dtype=np.float32)
    joint_vel = np.zeros((num_frames, num_joints), dtype=np.float32)
    for t in range(num_frames):
        for j in range(num_joints):
            joint_pos[t, j] = float(j) + 0.01 * t
            joint_vel[t, j] = float(j) * 10 + 0.1 * t

    body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    if body_pos_fn is not None:
        body_pos_fn(body_pos_w)

    body_quat_w = np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (num_frames, num_bodies, 1))
    body_lin_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez(
        tmp.name,
        fps=50,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# MotionData: joint remapping
# ---------------------------------------------------------------------------


class TestMotionDataJointRemapping(unittest.TestCase):
    """Test that MotionData + joint remapping produce the expected ordering."""

    def setUp(self):
        self.num_frames = 10
        self.num_joints = 4
        self.num_bodies = 3
        self.motion_joint_names = ["joint_A", "joint_B", "joint_C", "joint_D"]
        self.robot_joint_names = ["joint_C", "joint_A", "joint_D", "joint_B"]
        self._path = _make_npz(self.num_frames, self.num_joints, self.num_bodies)

    def tearDown(self):
        os.unlink(self._path)

    def test_joint_remapping_produces_correct_order(self):
        remap_idx = MotionData.build_joint_remap_idx(self.robot_joint_names, self.motion_joint_names)
        motion = MotionData(self._path, joint_remap_idx=remap_idx, device="cpu")

        # After remapping: col 0 = joint_C (orig 2), col 1 = joint_A (orig 0), ...
        expected_cols = [2, 0, 3, 1]
        for robot_col, orig_col in enumerate(expected_cols):
            for t in range(self.num_frames):
                expected_pos = float(orig_col) + 0.01 * t
                self.assertAlmostEqual(
                    motion.joint_pos[t, robot_col].item(),
                    expected_pos,
                    places=5,
                    msg=f"Mismatch at t={t}, robot_col={robot_col}",
                )
                expected_vel = float(orig_col) * 10 + 0.1 * t
                self.assertAlmostEqual(
                    motion.joint_vel[t, robot_col].item(),
                    expected_vel,
                    places=5,
                )

    def test_no_remapping_preserves_order(self):
        motion = MotionData(self._path, device="cpu")
        for j in range(self.num_joints):
            self.assertAlmostEqual(motion.joint_pos[0, j].item(), float(j), places=5)

    def test_remapping_is_invertible(self):
        motion_no_remap = MotionData(self._path, device="cpu")
        original = motion_no_remap.joint_pos.clone()

        fwd = MotionData.build_joint_remap_idx(self.robot_joint_names, self.motion_joint_names)
        motion = MotionData(self._path, joint_remap_idx=fwd, device="cpu")

        inv = MotionData.build_joint_remap_idx(self.motion_joint_names, self.robot_joint_names)
        recovered = motion.joint_pos[:, inv]
        torch.testing.assert_close(recovered, original)


# ---------------------------------------------------------------------------
# MotionData: body index selection
# ---------------------------------------------------------------------------


class TestMotionDataBodyIndexing(unittest.TestCase):
    def setUp(self):
        self.num_frames = 5
        self.num_bodies = 4

        def fill_body_pos(body_pos_w):
            for b in range(self.num_bodies):
                body_pos_w[:, b, :] = [b, b + 0.1, b + 0.2]

        self._path = _make_npz(self.num_frames, 2, self.num_bodies, body_pos_fn=fill_body_pos)

    def tearDown(self):
        os.unlink(self._path)

    def test_body_index_selection(self):
        selected = [2, 0]
        motion = MotionData(self._path, body_indices=selected, device="cpu")

        self.assertEqual(motion.body_pos_w.shape, (self.num_frames, 2, 3))
        torch.testing.assert_close(motion.body_pos_w[0, 0], torch.tensor([2.0, 2.1, 2.2]))
        torch.testing.assert_close(motion.body_pos_w[0, 1], torch.tensor([0.0, 0.1, 0.2]))

    def test_body_index_all(self):
        motion = MotionData(self._path, body_indices=list(range(self.num_bodies)), device="cpu")

        self.assertEqual(motion.body_pos_w.shape, (self.num_frames, self.num_bodies, 3))
        for b in range(self.num_bodies):
            self.assertAlmostEqual(motion.body_pos_w[0, b, 0].item(), float(b), places=5)

    def test_none_body_indices_keeps_all(self):
        motion = MotionData(self._path, body_indices=None, device="cpu")
        self.assertEqual(motion.body_pos_w.shape, (self.num_frames, self.num_bodies, 3))


# ---------------------------------------------------------------------------
# MotionData: static helpers
# ---------------------------------------------------------------------------


class TestMotionDataStaticHelpers(unittest.TestCase):
    def test_build_joint_remap_idx_basic(self):
        idx = MotionData.build_joint_remap_idx(
            target_joint_names=["C", "A"],
            source_joint_names=["A", "B", "C"],
        )
        self.assertEqual(idx.tolist(), [2, 0])

    def test_build_joint_remap_idx_missing_raises(self):
        with self.assertRaises(ValueError):
            MotionData.build_joint_remap_idx(
                target_joint_names=["X"],
                source_joint_names=["A", "B"],
            )

    def test_build_body_indices_basic(self):
        result = MotionData.build_body_indices(
            tracked_body_names=["torso", "pelvis"],
            source_body_names=["pelvis", "knee", "torso"],
        )
        self.assertEqual(result, [2, 0])

    def test_build_body_indices_missing_raises(self):
        with self.assertRaises(ValueError):
            MotionData.build_body_indices(
                tracked_body_names=["missing"],
                source_body_names=["pelvis", "torso"],
            )


# ---------------------------------------------------------------------------
# _get_body_indices helper (tests the logic used by motion_tracking_rewards)
# ---------------------------------------------------------------------------


class TestGetBodyIndices(unittest.TestCase):
    def test_none_body_names_returns_all(self):
        command = _MockCommand(["pelvis", "torso_link", "left_knee_link"])
        self.assertEqual(_get_body_indices(command, None), [0, 1, 2])

    def test_specific_body_names(self):
        command = _MockCommand(["pelvis", "torso_link", "left_knee_link", "right_knee_link"])
        self.assertEqual(_get_body_indices(command, ["torso_link", "right_knee_link"]), [1, 3])

    def test_empty_body_names(self):
        command = _MockCommand(["pelvis", "torso_link"])
        self.assertEqual(_get_body_indices(command, []), [])

    def test_single_body_name(self):
        command = _MockCommand(["pelvis", "torso_link", "left_knee_link"])
        self.assertEqual(_get_body_indices(command, ["torso_link"]), [1])


# ---------------------------------------------------------------------------
# G1 body/joint remapping (self-contained, no Isaac Lab imports needed)
# ---------------------------------------------------------------------------

# These lists mirror the task config in flat_env_cfg.py. They are duplicated
# here so the test has zero Isaac Lab import dependency.
_G1_TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link", "left_shoulder_roll_link", "left_elbow_link",
    "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link",
    "right_wrist_yaw_link",
]  # fmt: skip

_G1_URDF_BODIES = [
    "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
    "left_wrist_yaw_link", "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
    "right_wrist_pitch_link", "right_wrist_yaw_link",
]  # fmt: skip

_G1_MUJOCO_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip


class TestG1BodyRemapping(unittest.TestCase):
    def test_remap_from_urdf_to_usd_order(self):
        npz_indices = MotionData.build_body_indices(_G1_TRACKED_BODIES, _G1_URDF_BODIES)

        self.assertEqual(len(npz_indices), len(set(npz_indices)), "Duplicate indices found in body remapping")
        self.assertEqual(npz_indices[0], 0)  # pelvis is first in both

        torso_tracked_idx = _G1_TRACKED_BODIES.index("torso_link")
        self.assertEqual(npz_indices[torso_tracked_idx], _G1_URDF_BODIES.index("torso_link"))

    def test_joint_names_count_and_uniqueness(self):
        self.assertEqual(len(_G1_MUJOCO_JOINTS), 29)
        self.assertEqual(len(_G1_MUJOCO_JOINTS), len(set(_G1_MUJOCO_JOINTS)))

    def test_tracked_body_names_uniqueness(self):
        self.assertEqual(len(_G1_TRACKED_BODIES), len(set(_G1_TRACKED_BODIES)))

    def test_urdf_body_names_uniqueness(self):
        self.assertEqual(len(_G1_URDF_BODIES), len(set(_G1_URDF_BODIES)))


if __name__ == "__main__":
    unittest.main()
