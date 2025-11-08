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


import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from agile.rl_env.mdp.commands import UniformVelocityBaseHeightCommand
from agile.rl_env.mdp.utils import get_contact_sensor_cfg, get_robot_cfg


# Note: The command gets updated after the reward is computed resulting in a one-step reward delay.
def track_base_height_exp_smooth(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward the agent for tracking the base height."""
    command_term: UniformVelocityBaseHeightCommand = env.command_manager.get_term(command_name)
    base_height_error = torch.square(command_term.base_height - command_term.target_height)
    return torch.exp(-base_height_error / std**2)


def track_lin_vel_xy_yaw_frame_exp_weighted(
    env: ManagerBasedRLEnv,
    command_name: str,
    std_scale: float = 0.5,
    std_offset: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel.

    The std of the exponential kernel is scaled by the commanded velocity magnitude to provide adaptive tolerance:
    std = std_scale * |v_cmd| + std_offset

    This design provides tighter tracking requirements for low velocities and more tolerance for high velocities,
    improving tracking performance across the full velocity range.

    Args:
        env: The RL environment.
        command_name: Name of the command term.
        std_scale: Scaling factor for velocity-based std (k in std = k * |v_cmd| + e). Default is 0.5.
        std_offset: Offset for velocity-based std (e in std = k * |v_cmd| + e). Default is 0.05.
        asset_cfg: Scene entity configuration for the robot.
    """

    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    command_term = env.command_manager.get_term(command_name)
    vel_cmd = command_term.vel_command_b[:, :2]

    # Compute tracking error
    lin_vel_error = torch.sum(torch.square(vel_cmd - vel_yaw[:, :2]), dim=1)

    # Compute adaptive std based on commanded velocity magnitude
    cmd_vel_magnitude = torch.norm(vel_cmd, dim=1)
    adaptive_std = std_scale * cmd_vel_magnitude + std_offset

    # Return exponential reward with adaptive std
    return torch.exp(-lin_vel_error / adaptive_std**2)


def vel_xy_in_threshold(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for tracking the linear velocity."""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])[:, :2]
    vel_cmd = env.command_manager.get_command(command_name)[:, :2]

    lin_vel_error = torch.linalg.vector_norm(vel_cmd - vel_yaw, dim=1)
    return (lin_vel_error < threshold).float()


def track_ang_vel_z_world_exp_weighted(
    env: ManagerBasedRLEnv,
    command_name: str,
    std_scale: float = 0.5,
    std_offset: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel with adaptive std.

    The std of the exponential kernel is scaled by the commanded angular velocity magnitude:
    std = std_scale * |ang_vel_cmd| + std_offset

    This design provides tighter tracking requirements for low angular velocities and more tolerance
    for high angular velocities, improving tracking performance across the full velocity range.

    Args:
        env: The RL environment.
        command_name: Name of the command term.
        std_scale: Scaling factor for velocity-based std (k in std = k * |ang_vel_cmd| + e). Default is 0.5.
        std_offset: Offset for velocity-based std (e in std = k * |ang_vel_cmd| + e). Default is 0.05.
        asset_cfg: Scene entity configuration for the robot.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_cmd = env.command_manager.get_command(command_name)[:, 2]
    ang_vel_error = torch.square(ang_vel_cmd - asset.data.root_ang_vel_w[:, 2])

    # Compute adaptive std based on commanded angular velocity magnitude
    ang_vel_cmd_magnitude = torch.abs(ang_vel_cmd)
    adaptive_std = std_scale * ang_vel_cmd_magnitude + std_offset

    return torch.exp(-ang_vel_error / adaptive_std**2)


def ang_vel_in_threshold(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for tracking angular velocity within threshold.

    Args:
        env: Environment instance
        command_name: Name of the velocity command term
        threshold: Maximum allowed error (rad/s)
        asset_cfg: Asset configuration (uses root angular velocity)

    Returns:
        1.0 if angular velocity error is below threshold, 0.0 otherwise
    """
    asset = env.scene[asset_cfg.name]
    # Use root angular velocity (world frame, z component)
    ang_vel_z = asset.data.root_ang_vel_w[:, 2]
    ang_vel_cmd = env.command_manager.get_command(command_name)[:, 2]

    ang_vel_error = (ang_vel_z - ang_vel_cmd).abs()
    return (ang_vel_error < threshold).float()


def height_reached(
    env: ManagerBasedRLEnv,
    apply_stance_condition: bool = False,
    height_reached_threshold: float = 0.02,
    command_name: str = "base_velocity_height",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Extra bonus reward for reaching the target height.

    Args:
        env: Environment instance.
        apply_stance_condition: Whether to apply the reward only in stance mode.
        command_name: Name of the command generator.
        asset_cfg: Configuration for the robot asset.
    """

    # Get the robot asset from the scene
    robot, _ = get_robot_cfg(env, asset_cfg)

    # Get base height from the robot's root state
    base_height = robot.data.root_pos_w[:, 2]

    # Get the command from the command manager
    command_term = env.command_manager.get_term(command_name)

    # Compute height error
    height_error = torch.abs(base_height - command_term.command[:, -1])

    reward = torch.where(height_error <= height_reached_threshold, 1.0, 0.0)

    if apply_stance_condition:
        is_null_cmd = (command_term.command[:, :3] == 0).all(dim=1)
        reward = reward * is_null_cmd.float()

    return reward


def track_base_height(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    command_name: str = "base_velocity_height",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Reward for tracking a target base height.

    Args:
        env: Environment instance.
        std: Standard deviation for the exponential kernel.
        command_name: Name of the command generator.
        asset_cfg: Configuration for the robot asset.

    Returns:
        Reward tensor.
    """
    # Get the robot asset from the scene
    robot, _ = get_robot_cfg(env, asset_cfg)

    # Get base height from the robot's root state
    base_height = robot.data.root_pos_w[:, 2]

    # Get the command from the command manager
    command = env.command_manager.get_command(command_name)

    is_null_cmd = (command[:, :3] == 0).all(dim=1)

    # Compute height error
    height_error = torch.abs(base_height - command[:, -1])

    # Compute reward (exponential decay with height error)
    reward = torch.exp(-height_error / std**2) * is_null_cmd.float()

    return reward


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward for tracking the target base height with an exponential kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    height_error = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return torch.exp(-height_error / std**2)


def base_height_in_threshold(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Reward the agent for tracking the base height."""
    command_term: UniformVelocityBaseHeightCommand = env.command_manager.get_term(command_name)
    base_height_error = torch.abs(command_term.base_height - command_term.target_height)
    return (base_height_error < threshold).float()


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity_height",
    contact_threshold: float = 0.1,
    sensor_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Reward for standing still when velocity commands are near zero.

    Penalizes motion when command velocity is near zero and the robot should be in stance mode.

    Args:
        env: Environment instance.
        command_name: Name of the command generator.
        velocity_threshold: Threshold for considering velocity commands as zero.
        height_threshold: Height threshold for stance mode.
        contact_threshold: Threshold for determining foot contact.
        sensor_cfg: Contact sensor configuration.

    Returns:
        Reward tensor.
    """
    # Create default sensor_cfg if None is provided
    contact_sensor, sensor_cfg = get_contact_sensor_cfg(env, sensor_cfg)

    # Get velocity command from the command manager
    command = env.command_manager.get_command(command_name)

    # Get feet body IDs from sensor config
    feet_body_ids = sensor_cfg.body_ids

    # Get contact forces for feet
    contact_forces = contact_sensor.data.net_forces_w[:, feet_body_ids, 2]

    # Count feet without contact (contact force < threshold)
    feet_without_contact = torch.sum(contact_forces < contact_threshold, dim=-1)

    # Check if robot is in stance mode
    is_null_cmd = (command[:, :3] == 0).all(dim=1)

    # Compute penalty: apply when in stance mode with zero velocity command
    # Penalty is proportional to the number of feet without contact
    # Ensure the reward has shape [num_envs]
    reward = feet_without_contact * is_null_cmd.float()

    return reward
