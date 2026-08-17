# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for rollout-video camera tracking."""

from types import SimpleNamespace

import numpy as np

from agile.isaaclab_extras.video_camera import configure_robot_tracking_camera
from agile.sim2mujoco.camera import robot_tracking_target


def test_configure_robot_tracking_camera_follows_designated_recorded_robot():
    viewer = SimpleNamespace(
        eye=(7.5, 7.5, 7.5),
        lookat=(0.0, 0.0, 0.0),
        origin_type="world",
        asset_name=None,
        env_index=7,
    )

    configure_robot_tracking_camera(viewer)

    assert viewer.origin_type == "asset_root"
    assert viewer.asset_name == "robot"
    assert viewer.env_index == 0
    assert viewer.eye == (-2.5, -5.0, 2.0)
    assert viewer.lookat == (0.0, 0.0, 0.75)


def test_robot_tracking_target_uses_body_world_position_not_qpos_layout():
    body_world_positions = np.array([[0.0, 0.0, 0.0], [4.0, -3.0, 1.25], [9.0, 8.0, 7.0]])

    target = robot_tracking_target(body_world_positions, root_body_id=1)

    np.testing.assert_array_equal(target, [4.0, -3.0, 1.25])
