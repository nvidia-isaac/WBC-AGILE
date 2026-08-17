# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera configuration for rollout videos."""

from typing import Any


def configure_robot_tracking_camera(viewer_cfg: Any, *, robot_asset_name: str = "robot", env_index: int = 0) -> None:
    """Make the viewport follow one environment's robot root during recorded rollouts.

    Isaac Lab updates ``asset_root`` viewer origins at every render.  Selecting the
    recorded environment directly avoids both a fixed-world camera and a reduction
    over all vectorized environments.  The explicit eye/lookat offset keeps the
    robot large enough in frame instead of inheriting Isaac Lab's terrain-wide
    default camera.
    """
    viewer_cfg.origin_type = "asset_root"
    viewer_cfg.asset_name = robot_asset_name
    viewer_cfg.env_index = env_index
    viewer_cfg.eye = (-2.5, -5.0, 2.0)
    viewer_cfg.lookat = (0.0, 0.0, 0.75)
