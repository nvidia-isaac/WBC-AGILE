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


from __future__ import annotations

import torch
from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def get_joint_indices(
    env: ManagerBasedEnv, robot: Articulation, joint_part: str, tensor_shape: torch.Size | None = None
) -> list[int]:
    """Get joint indices based on the specified joint part.

    Args:
        env: Environment instance.
        robot: Robot asset from the scene.
        joint_part: Which part of the joints to consider: "lower_body", "upper_body",
            or "whole_body".
        tensor_shape: Shape of tensor to determine whole_body indices. If None, uses
            robot.num_joints.

    Returns:
        List of joint indices.
    """
    if joint_part == "lower_body":
        return env.lower_body_ids  # type: ignore
    elif joint_part == "upper_body":
        return env.upper_body_ids  # type: ignore
    elif joint_part == "whole_body":
        # Use all joint indices
        size = robot.num_joints if tensor_shape is None else tensor_shape[1]
        return list(range(size))
    else:
        raise ValueError(f"Invalid joint_part: {joint_part}. Must be 'lower_body', 'upper_body', or 'whole_body'")


def get_robot_cfg(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg | None = None) -> tuple[Articulation, SceneEntityCfg]:
    """Get robot configuration and robot asset from the scene.

    Args:
        env: Environment instance.
        robot_cfg: Optional robot configuration.

    Returns:
        Tuple of (robot, robot_cfg)
    """
    if robot_cfg is None:
        robot_cfg = SceneEntityCfg("robot")

    # Get the robot asset from the scene
    robot = env.scene[robot_cfg.name]

    return robot, robot_cfg


def get_contact_sensor_cfg(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg | None = None,
    body_names: list[str] | None = None,
) -> tuple[ContactSensor, SceneEntityCfg]:
    """Get sensor configuration and sensor from the scene.

    Args:
        env: Environment instance.
        sensor_cfg: Optional sensor configuration.
        body_names: List of body name patterns if creating default sensor config.

    Returns:
        Tuple of (sensor, sensor_cfg)
    """
    if sensor_cfg is None:
        if body_names is None:
            body_names = [".*ankle.*link"]
        sensor_cfg = SceneEntityCfg("contact_forces", body_names=body_names)

    # Get the sensor from the scene
    sensor = env.scene.sensors[sensor_cfg.name]

    return sensor, sensor_cfg


def transform_to_body_frame(positions: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    """Transform positions from world frame to body frame.

    Args:
        positions: World positions with shape [num_envs, num_points, 3]
        root_pos: Root positions with shape [num_envs, 3]
        root_quat: Root quaternions with shape [num_envs, 4]

    Returns:
        Positions in body frame with shape [num_envs, num_points, 3]
    """
    # Calculate positions relative to root
    # Shape: [num_envs, num_points, 3]
    pos_relative = positions - root_pos.unsqueeze(1)

    # Transform positions to body frame
    pos_body_frame = torch.zeros_like(pos_relative)
    for i in range(pos_relative.shape[1]):  # For each point
        pos_body_frame[:, i, :] = math_utils.quat_apply_inverse(root_quat, pos_relative[:, i, :])

    return pos_body_frame


def transform_to_asset_frame(positions: torch.Tensor, asset: Articulation | RigidObject) -> torch.Tensor:
    """Transform positions from world frame to asset frame.

    Args:
        positions: World positions with shape [num_envs, num_points, 3]
        asset: Articulation or RigidObject.

    Returns:
        Positions in asset frame with shape [num_envs, num_points, 3]
    """
    # Get asset pose in world frame
    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w

    # Step 1: Translate positions relative to asset center
    translated_positions = positions - root_pos_w.unsqueeze(1)  # [num_envs, num_points, 3]

    # Step 2: Apply inverse rotation to align with asset frame
    # Expand quaternion to match positions shape for proper broadcasting
    quat_expanded = root_quat_w.unsqueeze(1).expand(-1, positions.shape[1], -1)
    asset_frame_positions = math_utils.quat_apply_inverse(quat_expanded, translated_positions)

    return asset_frame_positions


def get_body_velocities_and_forces(
    robot: Articulation, contact_sensor: ContactSensor, sensor_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get body velocities and contact forces for bodies in sensor configuration.

    Args:
        robot: Robot asset from the scene.
        contact_sensor: Contact sensor from the scene.
        sensor_cfg: Sensor configuration containing the body IDs.

    Returns:
        Tuple of (body_velocities, contact_forces) where:
        - body_velocities: Tensor with shape [num_envs, num_bodies, 3]
        - contact_forces: Tensor with shape [num_envs, num_bodies, 3]
    """
    # Get body IDs from sensor config
    body_ids = sensor_cfg.body_ids

    # Get velocities for bodies from robot data
    # Shape: [num_envs, num_bodies, 3]
    body_velocities = robot.data.body_vel_w[:, body_ids]

    # Get contact forces for bodies
    # Shape: [num_envs, num_bodies, 3]
    contact_forces = contact_sensor.data.net_forces_w[:, body_ids]

    return body_velocities, contact_forces


def compute_asset_aabb(prim_path_expr: str, device: str) -> torch.Tensor:
    """Compute the axis-aligned bounding box (AABB) of the given prim paths.

    Args:
        prim_path_expr: Expression to find the prim paths.
        device: Device to compute the bounding box on.

    Returns:
        The bounding box dimensions of the asset. Shape is (num_prims, 3).
    """
    # resolve prim paths for spawning and cloning
    prims = sim_utils.find_matching_prims(prim_path_expr)

    # Initialize scale tensor
    scale = torch.zeros(len(prims), 3, device=device)

    # Create a bbox cache
    bbox_cache = UsdGeom.BBoxCache(
        time=Usd.TimeCode.Default(), useExtentsHint=False, includedPurposes=[UsdGeom.Tokens.default_]
    )

    # Compute bounding box for each prim path
    for i, prim in enumerate(prims):
        bbox_bounds: Gf.BBox3d = bbox_cache.ComputeWorldBound(prim)
        bbox_range = bbox_bounds.GetRange()
        bbox_range_min, bbox_range_max = bbox_range.GetMin(), bbox_range.GetMax()

        scale[i] = torch.tensor([bbox_range_max[j] - bbox_range_min[j] for j in range(3)], device=device)

    return scale
