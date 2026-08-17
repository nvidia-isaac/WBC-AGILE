# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera helpers for MuJoCo rollout videos."""

import numpy as np


def robot_root_body_id(joint_body_ids: np.ndarray, joint_types: np.ndarray, free_joint_type: int) -> int:
    """Resolve the robot root from its free joint instead of assuming joint zero is the root."""
    free_joints = np.flatnonzero(joint_types == free_joint_type)
    if free_joints.size:
        return int(joint_body_ids[free_joints[0]])
    return int(joint_body_ids[0]) if joint_body_ids.size else 0


def robot_tracking_target(body_world_positions: np.ndarray, root_body_id: int) -> np.ndarray:
    """Return a copy of the robot root's current world-space position."""
    return body_world_positions[root_body_id].copy()
