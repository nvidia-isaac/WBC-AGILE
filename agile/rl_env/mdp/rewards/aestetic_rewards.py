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

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

from agile.rl_env.mdp.utils import (
    get_body_velocities_and_forces,
    get_contact_sensor_cfg,
    get_robot_cfg,
    transform_to_asset_frame,
)
from agile.rl_env.utils import math_utils as agile_math_utils


class root_acc_l2(ManagerTermBase):
    """Penalize base linear and angular accelerations using velocity history tracking (Isaac Gym style)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # Initialize the base class
        super().__init__(cfg, env)

        # Initialize velocity history buffer
        # Shape: [num_envs, 6] where 6 = 3 (lin_vel) + 3 (ang_vel)
        self.prev_root_vel = torch.zeros(env.num_envs, 6, device=env.device, dtype=torch.float32)

        # Flag to track if this is the first call (skip acceleration computation)
        self.first_call = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Compute root acceleration penalty by tracking velocity changes."""

        # Extract the robot asset
        robot = env.scene[asset_cfg.name]

        # Get current root velocities (both linear and angular)
        current_lin_vel = robot.data.root_lin_vel_w  # [num_envs, 3]
        current_ang_vel = robot.data.root_ang_vel_w  # [num_envs, 3]

        # Concatenate to form 6D velocity vector
        current_root_vel = torch.cat([current_lin_vel, current_ang_vel], dim=-1)  # [num_envs, 6]

        if self.first_call:
            # First call: initialize previous velocity and return zeros
            self.prev_root_vel.copy_(current_root_vel)
            self.first_call = False
            return torch.zeros(env.num_envs, device=env.device)

        # Compute acceleration as velocity difference over timestep
        root_acc = (current_root_vel - self.prev_root_vel) / env.step_dt

        # Update velocity history for next call
        self.prev_root_vel.copy_(current_root_vel)

        # Compute L2 penalty on accelerations (sum of squared accelerations)
        return torch.sum(torch.square(root_acc), dim=-1)


def if_standing(
    env: ManagerBasedRLEnv,
    standing_height_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Check if the robot is standing above a threshold height.

    Args:
        env: Environment instance.
        standing_height_threshold: Height threshold above which the robot is considered standing.
        asset_cfg: Configuration for the robot asset.
        sensor_cfg: Optional configuration for terrain sensor to adjust height measurement.

    Returns:
        Binary float tensor [num_envs] - 1.0 if standing, 0.0 otherwise.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        current_height = asset.data.root_pos_w[:, 2] - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        current_height = asset.data.root_pos_w[:, 2]

    is_standing = current_height > standing_height_threshold
    return is_standing.float()


def feet_roll_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot roll angles (Isaac Gym style).

    Penalizes feet that are not flat (roll rotation around x-axis).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get feet quaternions
    feet_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]  # [num_envs, num_feet, 4]

    # Extract roll angles from quaternions
    # Using Isaac Lab's math utils to extract euler angles
    feet_quat_flat = feet_quat.reshape(-1, 4)  # [num_envs * num_feet, 4]
    roll, _, _ = agile_math_utils.euler_xyz_from_quat(feet_quat_flat)

    # Reshape back to [num_envs, num_feet] and normalize roll to [-pi, pi]
    feet_roll = roll.reshape(env.num_envs, len(asset_cfg.body_ids))
    feet_roll = (feet_roll + torch.pi) % (2 * torch.pi) - torch.pi

    # Return sum of squared roll angles (Isaac Gym style)
    return torch.sum(torch.square(feet_roll), dim=-1)


def feet_yaw_diff_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw difference between left and right feet (Isaac Gym style).

    Encourages both feet to have similar yaw orientation.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get feet quaternions
    feet_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]  # [num_envs, num_feet, 4]

    # Ensure we have exactly 2 feet
    if len(asset_cfg.body_ids) != 2:
        return torch.zeros(env.num_envs, device=env.device)

    # Extract yaw angles for both feet
    feet_quat_flat = feet_quat.reshape(-1, 4)  # [num_envs * 2, 4]
    _, _, yaw = agile_math_utils.euler_xyz_from_quat(feet_quat_flat)

    # Reshape to [num_envs, 2] and normalize yaw to [-pi, pi]
    feet_yaw = yaw.reshape(env.num_envs, 2)
    feet_yaw = (feet_yaw + torch.pi) % (2 * torch.pi) - torch.pi

    # Compute yaw difference between right foot (index 1) and left foot (index 0)
    yaw_diff = (feet_yaw[:, 1] - feet_yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi

    # Return squared yaw difference (Isaac Gym style)
    return torch.square(yaw_diff)


def feet_yaw_mean_vs_base(
    env: ManagerBasedRLEnv,
    feet_asset_cfg: SceneEntityCfg,
    base_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize the squared yaw of each foot relative to the base frame.

    This encourages the feet to stay rotationally aligned with the base's
    forward direction by minimizing the yaw component of their relative orientation.
    """
    asset: Articulation = env.scene[feet_asset_cfg.name]

    # Get feet quaternions and base quaternion
    feet_quat = asset.data.body_quat_w[:, feet_asset_cfg.body_ids]  # [num_envs, 2, 4]
    base_quat = asset.data.body_quat_w[:, base_body_cfg.body_ids].squeeze(1)  # [num_envs, 4]

    # Ensure we have exactly 2 feet
    if len(feet_asset_cfg.body_ids) != 2:
        raise ValueError("Only two feet are supported for feet_yaw_mean_vs_base reward.")

    if len(base_body_cfg.body_ids) != 1:
        raise ValueError("Only one reference body is supported for feet_yaw_mean_vs_base reward.")

    # Express feet quaternions in base frame
    base_quat_inv = math_utils.quat_inv(base_quat)  # [num_envs, 4]
    feet_quat_relative = math_utils.quat_mul(
        base_quat_inv.unsqueeze(1).expand(-1, 2, -1), feet_quat
    )  # [num_envs, 2, 4]

    # Extract yaw from relative quaternions (no reshaping needed)
    _, _, feet_yaw_relative = math_utils.euler_xyz_from_quat(feet_quat_relative.view(-1, 4))
    feet_yaw_relative = feet_yaw_relative.view(env.num_envs, 2)

    # Return squared mean yaw
    return torch.square(feet_yaw_relative).sum(dim=1)


def feet_yaw_mean_vs_base_if_standing(
    env: ManagerBasedRLEnv,
    standing_height_threshold: float,
    feet_asset_cfg: SceneEntityCfg,
    base_body_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize the squared yaw of each foot relative to the base frame if the robot is standing.
    See `feet_yaw_mean_vs_base` for more details."""
    angle_error_squared = feet_yaw_mean_vs_base(env, feet_asset_cfg, base_body_cfg)
    is_standing = if_standing(env, standing_height_threshold, asset_cfg, sensor_cfg)
    return angle_error_squared * is_standing


def feet_distance_from_ref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ref_distance: float = 0.2,
    command_name: str | None = None,
    lateral_velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Penalize feet lateral distance deviation from reference distance.

    This reward encourages maintaining proper lateral spacing between left and right feet.

    Args:
        env: Environment instance.
        asset_cfg: Configuration for the robot asset (should specify foot body names).
        ref_distance: Reference lateral distance between feet (meters).

    Returns:
        Penalty tensor [num_envs] - higher when feet distance deviates from reference.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get feet positions - assumes asset_cfg.body_ids contains left and right foot indices
    # Shape: [num_envs, num_feet, 3]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]

    # Ensure we have exactly 2 feet
    if len(asset_cfg.body_ids) != 2:
        # If not exactly 2 feet specified, return zeros (no penalty)
        return torch.zeros(env.num_envs, device=env.device)

    feet_pos_b = transform_to_asset_frame(feet_pos_w, asset)

    # Get positions of left and right feet
    left_foot_pos = feet_pos_b[:, 0]  # [num_envs, 3]
    right_foot_pos = feet_pos_b[:, 1]  # [num_envs, 3]

    # Calculate lateral (Y-axis) distance between feet
    # In world frame, Y-axis typically represents lateral direction
    lateral_distance = torch.abs(left_foot_pos[:, 1] - right_foot_pos[:, 1])

    # Compute penalty as squared deviation from reference distance
    distance_error = lateral_distance - ref_distance

    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        lateral_velocity_command = command[:, 1].abs()
        large_lateral_velocity = lateral_velocity_command > lateral_velocity_threshold
        distance_error[large_lateral_velocity] = 0

    return torch.square(distance_error)


def feet_distance_from_ref_if_standing(
    env: ManagerBasedRLEnv,
    standing_height_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ref_distance: float = 0.2,
    command_name: str | None = None,
    lateral_velocity_threshold: float = 0.5,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    distance_error_squared = feet_distance_from_ref(
        env, asset_cfg, ref_distance, command_name, lateral_velocity_threshold
    )
    is_standing = if_standing(env, standing_height_threshold, asset_cfg, sensor_cfg)
    return distance_error_squared * is_standing


def jumping(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize if no foot is in contact with the ground.

    Args:
        env: The environment.
        threshold: The force threshold for the jumping.
        sensor_cfg: The configuration for the foot contact sensor.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the penalty
    feet_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=2)
    not_in_contact = feet_forces < threshold
    is_jumping = not_in_contact.all(dim=1)

    return is_jumping.float()


def impact_velocity_l1(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 10.0
) -> torch.Tensor:
    """Penalize large impact velocities.

    Args:
        env: The environment.
        force_threshold: The force threshold for the impact.
        velocity_threshold: The velocity threshold for the impact.
        sensor_cfg: The configuration for the foot contact sensor.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the penalty
    net_contact_forces = contact_sensor.data.net_forces_w_history
    in_contact = (
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > force_threshold
    )

    body_velocities = torch.max(
        torch.norm(contact_sensor.data.velocities_w_history[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1,
    )[0]

    impact_velocities = torch.where(in_contact, body_velocities, 0.0).sum(dim=1)

    return impact_velocities


def no_undersired_base_velocity_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
) -> torch.Tensor:
    """Reward zero base velocity if it is not desired."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_z = torch.square(asset.data.root_lin_vel_b[:, 2])
    ang_vel_xy = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward = torch.exp(-(lin_vel_z + ang_vel_xy) / std**2)
    return reward


def no_undersired_base_velocity_exp_if_null_cmd(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    std: float = 0.1,
) -> torch.Tensor:
    """Reward zero base velocity if it is not desired."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    is_null_cmd = (command_term.command[:, :3] == 0).all(dim=1)

    lin_vel_z_weight = torch.where(
        is_null_cmd,
        torch.full_like(is_null_cmd, 0.1, dtype=torch.float32),
        torch.full_like(is_null_cmd, 1.0, dtype=torch.float32),
    )
    lin_vel_z = torch.square(asset.data.root_lin_vel_b[:, 2]) * lin_vel_z_weight
    ang_vel_xy = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward = torch.exp(-(lin_vel_z + ang_vel_xy) / std**2)
    return reward


def equal_foot_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward if the z-component of the force on each foot is equal.

    If the full force is on one foot, the reward is 0.0.
    If the force is evenly distributed, the reward is 1.0.

    Args:
        env: The environment.
        sensor_cfg: The configuration for the foot contact sensor.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    feet_z_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].abs()
    mean_force = feet_z_forces.mean(dim=1)
    reward = 1.0 - torch.abs(mean_force.unsqueeze(1) - feet_z_forces).mean(dim=1) / (mean_force + 1e-6)

    return reward


def equal_foot_force_if_standing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    standing_height_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_measurement_sensor: SceneEntityCfg = SceneEntityCfg("height_measurement_sensor"),
) -> torch.Tensor:
    """Reward if the z-component of the force on each foot is equal."""
    reward = equal_foot_force(env, sensor_cfg)
    is_standing = if_standing(env, standing_height_threshold, asset_cfg, height_measurement_sensor)
    return reward * is_standing


def equal_foot_force_if_null_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward if the z-component of the force on each foot is equal.

    If the full force is on one foot, the reward is 0.0.
    If the force is evenly distributed, the reward is 1.0.

    Args:
        env: The environment.
        sensor_cfg: The configuration for the foot contact sensor.
    """
    command_term = env.command_manager.get_term(command_name)
    is_null_cmd = (command_term.command[:, :3] == 0).all(dim=1)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    feet_z_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].abs()
    mean_force = feet_z_forces.mean(dim=1)
    reward = 1.0 - torch.abs(mean_force.unsqueeze(1) - feet_z_forces).mean(dim=1) / (mean_force + 1e-6)

    return reward * is_null_cmd.float()


def stand_with_both_feet_if_null_cmd(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: ARG001
) -> torch.Tensor:
    """Reward the agent for standing with both feet if the command is null.

    The reward is 0.0 if the command is not null. If the command is null, the reward is -1.0 if not both
    feet are in contact. Otherwise the reward is dependent on the force distribution on the two feet.
    """
    # check null command
    command_term = env.command_manager.get_term(command_name)
    is_null_cmd = (command_term.command[:, :3] == 0).all(dim=1)

    # check both feet in contact
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_z_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].abs()
    mean_force = feet_z_forces.mean(dim=1)
    reward = 1.0 - torch.abs(mean_force.unsqueeze(1) - feet_z_forces).mean(dim=1) / (mean_force + 1e-6)
    both_feet_in_contact = (feet_z_forces > threshold).all(dim=1)

    reward[~both_feet_in_contact] = -1.0
    reward[~is_null_cmd] = 0.0

    return reward


def foot_orientation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    roll_weight: float = 1.0,
    pitch_weight: float = 1.0,
    yaw_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize the foot orientation."""
    asset: Articulation = env.scene[asset_cfg.name]
    # feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    feet_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    root_quat_w = asset.data.root_quat_w

    feet_quat_b = math_utils.quat_mul(
        math_utils.quat_inv(math_utils.yaw_quat(root_quat_w)).unsqueeze(1).repeat(1, feet_quat_w.shape[1], 1),
        feet_quat_w,
    )
    roll, pitch, yaw = agile_math_utils.euler_xyz_from_quat(feet_quat_b)

    return (
        roll.abs().mean(dim=1) * roll_weight
        + pitch.abs().mean(dim=1) * pitch_weight
        + yaw.abs().mean(dim=1) * yaw_weight
    )


def moving(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, weight_lin: float = 1.0, weight_ang: float = 1.0
) -> torch.Tensor:
    """Penalize the agent for moving."""
    asset = env.scene[asset_cfg.name]
    lin_vels = asset.data.body_lin_vel_w.norm(dim=-1)
    ang_vels = asset.data.body_ang_vel_w.norm(dim=-1)

    penalty = lin_vels.mean(dim=1) * weight_lin + ang_vels.mean(dim=1) * weight_ang
    return penalty


def moving_if_standing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    standing_height_threshold: float,
    weight_lin: float = 1.0,
    weight_ang: float = 1.0,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize the agent for moving if the robot is standing.
    See `moving` for more details."""
    penalty = moving(env, asset_cfg, weight_lin, weight_ang)
    is_standing = if_standing(env, standing_height_threshold, asset_cfg, sensor_cfg)
    return penalty * is_standing


def flat_orientation_if_null_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the agent for non-flat orientation if the command is null."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    is_null_cmd = (command_term.command[:, :3] == 0).all(dim=1)

    orientation_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

    penalty = torch.where(is_null_cmd, orientation_error, 0.0)

    return penalty


def feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward for penalizing feet stumbling (high horizontal forces).

    Args:
        env: Environment instance.
        sensor_cfg: Contact sensor configuration.
        threshold: Force threshold for stumbling.
        scale: Scaling factor.

    Returns:
        Reward tensor.
    """
    # Get the contact sensor from the scene
    contact_sensor, sensor_cfg = get_contact_sensor_cfg(env, sensor_cfg)

    # Get contact forces for these bodies
    # Use the net_forces_w_history which includes the history of contact forces
    net_contact_forces = contact_sensor.data.net_forces_w_history

    # Extract only the horizontal components (x and y) of the forces
    # Shape: [num_envs, history_length, num_bodies, 3] -> [num_envs, history_length, num_bodies, 2]
    horizontal_forces = net_contact_forces[:, :, sensor_cfg.body_ids, :2]

    # Compute the magnitude of horizontal forces
    # Shape: [num_envs, history_length, num_bodies]
    horizontal_force_magnitudes = torch.norm(horizontal_forces, dim=-1)

    # Find the maximum horizontal force for each environment across all bodies and history
    # Shape: [num_envs]
    max_horizontal_forces = torch.max(torch.max(horizontal_force_magnitudes, dim=2)[0], dim=1)[  # Max across bodies
        0
    ]  # Max across history

    # Compute reward
    reward = torch.relu(max_horizontal_forces - threshold)
    return reward


def feet_slip(
    env: ManagerBasedRLEnv,
    contact_threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = None,
    robot_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Reward for penalizing feet slipping.

    Penalizes horizontal velocity of feet when in contact with the ground.

    Args:
        env: Environment instance.
        contact_threshold: Threshold for determining foot contact.
        sensor_cfg: Contact sensor configuration for feet.
        robot_cfg: Configuration for the robot asset.

    Returns:
        Reward tensor.
    """
    # Create default sensor_cfg and robot_cfg if None is provided
    robot, _ = get_robot_cfg(env, robot_cfg)
    contact_sensor, sensor_cfg = get_contact_sensor_cfg(env, sensor_cfg)

    # Get feet body IDs from sensor config
    feet_body_ids = sensor_cfg.body_ids

    # Get contact forces for feet
    net_contact_forces = torch.norm(contact_sensor.data.net_forces_w_history[:, :, feet_body_ids], dim=-1)

    # Count feet without contact (contact force < threshold)
    feet_in_contact = torch.max(net_contact_forces, dim=1)[0] > contact_threshold

    # Calculate horizonta linear velocity magnitude for each foot
    # Shape: [num_envs, num_bodies]
    feet_velocities, _ = get_body_velocities_and_forces(robot, contact_sensor, sensor_cfg)
    horizontal_linear_velocity = torch.norm(feet_velocities[:, :, :2], dim=2)

    # Calculate the slip penalty (horizontal velocity when in contact)
    # Shape: [num_envs, num_bodies]
    slip_penalty = horizontal_linear_velocity * feet_in_contact

    # Sum penalties across all feet
    # Shape: [num_envs]
    reward = torch.sum(slip_penalty, dim=1)

    return reward


def joint_deviation_exp_if_standing(
    env: ManagerBasedRLEnv,
    standing_height_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    std: float = 0.25,
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    is_standing = if_standing(env, standing_height_threshold, asset_cfg, sensor_cfg)
    return torch.sum(torch.exp(-torch.square(angle) / std**2), dim=1) * is_standing
