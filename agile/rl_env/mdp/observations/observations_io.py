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

# ruff: noqa: I001

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

from agile.rl_env.mdp.commands import (
    UniformVelocityBaseHeightCommand,
)
from agile.rl_env.utils import math_utils as agile_math_utils


def is_env_inactive(env: ManagerBasedRLEnv, rest_duration_s: float) -> torch.Tensor:
    """Check if the environment is in the rest phase."""
    # Note: episode_length_buf is initialized after managers, so we check for its existence.
    # This allows the observation manager to be created before the environment is fully initialized.
    if hasattr(env, "episode_length_buf"):
        return (env.episode_length_buf < int(rest_duration_s / env.step_dt)).float().unsqueeze(1)
    else:
        return torch.ones(env.num_envs, 1, device=env.device)


def height_scan_feet(
    env: ManagerBasedEnv,
    sensor_cfg_left: SceneEntityCfg,
    sensor_cfg_right: SceneEntityCfg,
    offset: float = 0.0,
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.0) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor_left: RayCaster = env.scene.sensors[sensor_cfg_left.name]
    sensor_right: RayCaster = env.scene.sensors[sensor_cfg_right.name]
    # height scan: height = sensor height - hit point z - offset
    out = torch.cat(
        (
            (sensor_left.data.pos_w[:, 2].unsqueeze(1) - sensor_left.data.ray_hits_w[..., 2] - offset).unsqueeze(1),
            (sensor_right.data.pos_w[:, 2].unsqueeze(1) - sensor_right.data.ray_hits_w[..., 2] - offset).unsqueeze(1),
        ),
        dim=1,
    )
    return out.reshape(out.shape[0], -1)


def base_height_from_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: ARG001
) -> torch.Tensor:
    """Get the base height from the command."""
    command_term: UniformVelocityBaseHeightCommand = env.command_manager.get_term(command_name)
    return command_term.base_height.unsqueeze(1)


def velocity_height_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: ARG001
) -> torch.Tensor:
    """Get the velocity height command from the command."""
    command_term: UniformVelocityBaseHeightCommand = env.command_manager.get_term(command_name)
    return command_term.command.unsqueeze(1)


def base_height_from_sensor(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: ARG001
) -> torch.Tensor:
    """Get the base height from the sensor."""
    robot = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    base_height = robot.data.root_pos_w.torch[:, 2] - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    return base_height.unsqueeze(1)


"""
Commands.
"""


def joint_pos_action_target(env: ManagerBasedEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Return the internal joint position target from a RelativeJointPositionTargetAction term.

    Shape: [num_envs, num_joints].
    """
    action_term = env.action_manager._terms[action_name]
    return action_term.target


def joint_pos_tracking_error(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the PD tracking error: joint position target minus current position.

    Uses ``asset.data.joint_pos_target`` which is set by any action term
    (absolute, relative, or target-based), making this agnostic to the action type.
    Positive values mean the actuator is lagging behind the commanded target.

    Shape: [num_envs, num_joints].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos_target[:, asset_cfg.joint_ids] - asset.data.joint_pos.torch[:, asset_cfg.joint_ids]


def applied_external_forces(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the external forces currently applied to specified bodies.

    Reads from the permanent wrench composer, which stores forces set by
    ``apply_external_force_torque`` events. Returns forces in the body's link frame.

    Shape: [num_envs, num_bodies * 3].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # out_force_b.torch shape: (num_envs, num_bodies, 3)
    forces = asset.permanent_wrench_composer.out_force_b.torch
    selected = forces[:, asset_cfg.body_ids, :]  # (num_envs, num_selected_bodies, 3)
    return selected.reshape(selected.shape[0], -1)  # (num_envs, num_selected_bodies * 3)


def applied_external_force_torque(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the external forces and torques currently applied to specified bodies.

    Reads from the permanent wrench composer, which stores wrenches set by
    ``apply_external_force_torque`` events. Returns [force, torque] concatenated
    per body in the body's link frame.

    Shape: [num_envs, num_bodies * 6].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    forces = asset.permanent_wrench_composer.out_force_b.torch
    torques = asset.permanent_wrench_composer.out_torque_b.torch
    sel_f = forces[:, asset_cfg.body_ids, :]  # (num_envs, num_selected_bodies, 3)
    sel_t = torques[:, asset_cfg.body_ids, :]  # (num_envs, num_selected_bodies, 3)
    # Concatenate force and torque per body: (num_envs, num_bodies, 6)
    wrench = torch.cat([sel_f, sel_t], dim=-1)
    return wrench.reshape(wrench.shape[0], -1)  # (num_envs, num_bodies * 6)


def body_mass(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the mass of specified bodies.

    Shape: [num_envs, num_selected_bodies].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.default_mass.torch[:, asset_cfg.body_ids]


def joint_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Extract the joint accelerations of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their
    accelerations returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc.torch[:, asset_cfg.joint_ids]


def contact_force_norm(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Extract the norms of the contact forces of the asset."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # Get contact forces for these bodies.
    net_contact_forces = contact_sensor.data.net_forces_w.torch

    # Get forces for the specified bodies
    # Shape: [num_envs, num_bodies]
    body_forces = net_contact_forces[:, sensor_cfg.body_ids].norm(dim=2)

    return body_forces


def feet_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Height of each foot above the ground, measured relative to the ray-cast terrain height.

    Shape: [num_envs, num_feet].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ground_z = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)  # (N,)
    feet_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]  # (N, num_feet)
    return feet_z - ground_z.unsqueeze(1)


def feet_roll_pitch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Roll and pitch of each foot in the world frame.

    Shape: [num_envs, num_feet * 2] — interleaved (roll_0, pitch_0, roll_1, pitch_1, ...).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]  # (N, num_feet, 4)
    roll, pitch, _ = agile_math_utils.euler_xyz_from_quat(feet_quat.reshape(-1, 4))
    num_feet = len(asset_cfg.body_ids)
    roll = roll.reshape(env.num_envs, num_feet)
    pitch = pitch.reshape(env.num_envs, num_feet)
    return torch.stack([roll, pitch], dim=-1).reshape(env.num_envs, num_feet * 2)


def feet_yaw_vs_body(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Yaw of each foot relative to a reference body frame.

    Shape: [num_envs, num_feet].
    """
    asset: Articulation = env.scene[feet_cfg.name]
    feet_quat = asset.data.body_quat_w.torch[:, feet_cfg.body_ids]  # (N, num_feet, 4)
    ref_quat = asset.data.body_quat_w.torch[:, body_cfg.body_ids[0]]  # (N, 4)

    num_feet = len(feet_cfg.body_ids)
    ref_quat_inv = math_utils.quat_inv(ref_quat).unsqueeze(1).expand(-1, num_feet, -1)
    rel_quat = math_utils.quat_mul(ref_quat_inv, feet_quat)  # (N, num_feet, 4)

    _, _, yaw = agile_math_utils.euler_xyz_from_quat(rel_quat.reshape(-1, 4))
    return yaw.reshape(env.num_envs, num_feet)


def end_effector_poses(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the end effector poses (position and quaternion) in world frame.

    Returns poses for specified end effector bodies as [x, y, z, qw, qx, qy, qz] per body.

    Args:
        env: The environment instance.
        asset_cfg: Asset configuration with body_names specifying the end effectors.

    Returns:
        Tensor of shape [num_envs, num_bodies * 7] with pose for each body.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    body_quat_w = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
    body_poses = torch.cat([body_pos_w, body_quat_w], dim=-1)
    return body_poses.reshape(env.num_envs, -1)


def end_effector_poses_in_base_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_body_name: str = "torso_link",
) -> torch.Tensor:
    """Get the end effector poses relative to the robot base frame.

    Returns poses for specified end effector bodies as [x, y, z, qw, qx, qy, qz] per body,
    where positions and orientations are expressed relative to the base frame.

    Args:
        env: The environment instance.
        asset_cfg: Asset configuration with body_names specifying the end effectors.
        base_body_name: Name of the base body to use as reference frame.

    Returns:
        Tensor of shape [num_envs, num_bodies * 7] with relative pose for each body.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    base_idx = asset.find_bodies(base_body_name)[0][0]
    base_pos_w = asset.data.body_pos_w.torch[:, base_idx]
    base_quat_w = asset.data.body_quat_w.torch[:, base_idx]
    ee_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    ee_quat_w = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]

    relative_pos = ee_pos_w - base_pos_w.unsqueeze(1)
    base_quat_expanded = base_quat_w.unsqueeze(1).expand(-1, ee_pos_w.shape[1], -1)
    relative_pos = math_utils.quat_apply_inverse(base_quat_expanded, relative_pos)

    base_quat_inv = math_utils.quat_conjugate(base_quat_w)
    base_quat_inv_expanded = base_quat_inv.unsqueeze(1).expand(-1, ee_quat_w.shape[1], -1)
    relative_quat = math_utils.quat_mul(base_quat_inv_expanded, ee_quat_w)

    relative_poses = torch.cat([relative_pos, relative_quat], dim=-1)
    return relative_poses.reshape(env.num_envs, -1)


def relative_time(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The relative time in the episode in [0, 1]."""
    return env.episode_length_buf.unsqueeze(1) / env.max_episode_length


def command_time_remaining(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Normalized time remaining until next command resample.

    Returns time_left / max_resampling_time so the scale is consistent regardless
    of the sampled interval duration.  Range: ~0 (resample imminent) to 1.0 (just
    resampled with the maximum interval).
    """
    command_term = env.command_manager.get_term(command_name)
    max_time = command_term.cfg.resampling_time_range[1]
    return (command_term.time_left / max_time).unsqueeze(-1)


