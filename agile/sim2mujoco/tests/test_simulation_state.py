# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State extraction tests for MuJoCo simulation."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from agile.sim2mujoco.observations import MotionTracker
from agile.sim2mujoco.simulation import MuJocoSimulation

_FLOATING_BASE_MJCF = """
<mujoco>
  <worldbody>
    <body name="base">
      <freejoint/>
      <inertial pos="0 1 0" mass="1" diaginertia="1 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _make_sim(tmp_path):
    mjcf = tmp_path / "floating_base.xml"
    mjcf.write_text(_FLOATING_BASE_MJCF)
    cfg = {
        "scene": {"physics_dt": 0.005, "decimation": 1},
        "articulations": {"robot": {"joint_names": [], "default_joint_pos": []}},
    }
    return MuJocoSimulation(cfg, "cpu", enable_viewer=False, mjcf_path=mjcf)


def test_freejoint_angular_qvel_fallback_is_already_in_root_frame(tmp_path) -> None:
    sim = _make_sim(tmp_path)
    yaw_90 = math.sqrt(0.5)
    sim.mj_data.qpos[3:7] = [yaw_90, 0.0, 0.0, yaw_90]
    sim.mj_data.qvel[3:6] = [1.0, 0.0, 0.0]

    state = sim.get_state()

    assert state.root_ang_vel.tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_freejoint_com_velocity_fallback_uses_world_angular_velocity_for_cross_product(tmp_path) -> None:
    sim = _make_sim(tmp_path)
    yaw_90 = math.sqrt(0.5)
    sim.mj_data.qpos[3:7] = [yaw_90, 0.0, 0.0, yaw_90]
    sim.mj_data.qvel[3:6] = [1.0, 0.0, 0.0]

    state = sim.get_state()

    assert state.root_lin_vel.tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_reset_accepts_reference_motion_state_with_world_velocities(tmp_path) -> None:
    sim = _make_sim(tmp_path)
    yaw_90 = math.sqrt(0.5)

    sim.reset(
        initial_state={
            "root_pos": [1.0, 2.0, 3.0],
            "root_quat": [yaw_90, 0.0, 0.0, yaw_90],
            "root_lin_vel_w": [0.1, 0.2, 0.3],
            "root_ang_vel_w": [1.0, 0.0, 0.0],
            "joint_pos": [],
            "joint_vel": [],
        }
    )
    state = sim.get_state()

    assert state.root_pos.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert state.root_quat.tolist() == pytest.approx([yaw_90, 0.0, 0.0, yaw_90])
    assert state.root_lin_vel.tolist() == pytest.approx([0.2, -0.1, 0.3])
    assert state.root_ang_vel.tolist() == pytest.approx([0.0, -1.0, 0.0], abs=1.0e-6)


def test_motion_tracker_builds_reference_initial_state_in_sim_joint_order(tmp_path) -> None:
    motion_file = tmp_path / "motion.npz"
    np.savez(
        motion_file,
        fps=np.array(50.0, dtype=np.float32),
        joint_pos=np.array([[1.0, 2.0]], dtype=np.float32),
        joint_vel=np.array([[3.0, 4.0]], dtype=np.float32),
        body_pos_w=np.array([[[0.1, 0.2, 0.3], [9.0, 9.0, 9.0]]], dtype=np.float32),
        body_quat_w=np.array([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
        body_lin_vel_w=np.array([[[0.4, 0.5, 0.6], [9.0, 9.0, 9.0]]], dtype=np.float32),
        body_ang_vel_w=np.array([[[0.7, 0.8, 0.9], [9.0, 9.0, 9.0]]], dtype=np.float32),
    )
    tracker = MotionTracker(
        {
            "motion_file": str(motion_file),
            "anchor_body_name": "torso",
            "motion_body_names": ["pelvis", "torso"],
            "motion_joint_names": ["joint_b", "joint_a"],
        },
        target_joint_names=["joint_a", "joint_b"],
        device=torch.device("cpu"),
    )

    initial_state = tracker.get_initial_state(["joint_b", "joint_a"])

    assert initial_state["root_pos"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert initial_state["root_quat"].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert initial_state["root_lin_vel_w"].tolist() == pytest.approx([0.4, 0.5, 0.6])
    assert initial_state["root_ang_vel_w"].tolist() == pytest.approx([0.7, 0.8, 0.9])
    assert initial_state["joint_pos"].tolist() == pytest.approx([1.0, 2.0])
    assert initial_state["joint_vel"].tolist() == pytest.approx([3.0, 4.0])
