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

from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from agile.rl_env.mdp.commands.height_command import SmoothHeightCommand

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
    transform_to_body_frame,
)
from agile.rl_env.utils import math_utils as agile_math_utils


class body_acc_l2(ManagerTermBase):
    """Penalize body linear and angular accelerations using velocity history tracking (Isaac Gym style).

    This reward term computes world-frame accelerations for a specified body/link.
    If no body_names is specified in asset_cfg, it defaults to using the root link.

    Usage:
        # For root acceleration (default):
        body_acc = RewTerm(func=body_acc_l2, weight=-0.01)

        # For a specific link:
        torso_acc = RewTerm(
            func=body_acc_l2,
            weight=-0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
        )
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # Initialize the base class
        super().__init__(cfg, env)

        # Initialize velocity history buffer
        # Shape: [num_envs, 6] where 6 = 3 (lin_vel) + 3 (ang_vel)
        self.prev_body_vel = torch.zeros(env.num_envs, 6, device=env.device, dtype=torch.float32)

        # Flag to track if this is the first call (skip acceleration computation)
        self.first_call = True

        # Resolve body index if body_names is provided
        self._body_idx: int | None = None
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        if asset_cfg.body_names is not None:
            asset: Articulation = env.scene[asset_cfg.name]
            self._body_idx = asset.find_bodies(asset_cfg.body_names)[0][0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Compute body acceleration penalty by tracking velocity changes in world frame."""

        # Extract the robot asset
        robot: Articulation = env.scene[asset_cfg.name]

        # Get current velocities (both linear and angular) in world frame
        if self._body_idx is not None:
            # Use specified body velocities
            current_lin_vel = robot.data.body_lin_vel_w.torch[:, self._body_idx, :]  # [num_envs, 3]
            current_ang_vel = robot.data.body_ang_vel_w.torch[:, self._body_idx, :]  # [num_envs, 3]
        else:
            # Default to root velocities
            current_lin_vel = robot.data.root_lin_vel_w.torch  # [num_envs, 3]
            current_ang_vel = robot.data.root_ang_vel_w.torch  # [num_envs, 3]

        # Concatenate to form 6D velocity vector
        current_body_vel = torch.cat([current_lin_vel, current_ang_vel], dim=-1)  # [num_envs, 6]

        if self.first_call:
            # First call: initialize previous velocity and return zeros
            self.prev_body_vel.copy_(current_body_vel)
            self.first_call = False
            return torch.zeros(env.num_envs, device=env.device)

        # Compute acceleration as velocity difference over timestep
        body_acc = (current_body_vel - self.prev_body_vel) / env.step_dt

        # Update velocity history for next call
        self.prev_body_vel.copy_(current_body_vel)

        # Compute L2 penalty on accelerations (sum of squared accelerations)
        return torch.clamp(torch.sum(torch.square(body_acc), dim=-1), max=1e6)


def _body_tilt_angles(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Compute tilt angle from vertical for specified bodies.

    Args:
        env: Environment instance.
        asset_cfg: Asset config with body_names specifying which bodies to check.
            If no body_names, uses root.

    Returns:
        Tilt angles in radians [num_envs, num_bodies].
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Check if specific bodies were requested via body_names
    if asset_cfg.body_names is None:
        # Use root
        cos_theta = torch.clamp(-asset.data.projected_gravity_b.torch[:, 2], -1.0, 1.0)
        return torch.acos(cos_theta).unsqueeze(-1)  # [num_envs, 1]

    # Get body quaternions and project gravity
    body_quats = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]  # [num_envs, num_bodies, 4]
    if body_quats.dim() == 2:
        # Single body case - add body dimension
        body_quats = body_quats.unsqueeze(1)
    num_bodies = body_quats.shape[1]

    gravity_vec = asset.data.GRAVITY_VEC_W.torch.unsqueeze(1).expand(-1, num_bodies, -1)
    projected_gravity = math_utils.quat_apply_inverse(body_quats, gravity_vec)  # [num_envs, num_bodies, 3]

    # Tilt angle: angle between body z-axis and world z-axis (gravity direction)
    # projected_gravity[:, :, 2] is -1 when upright, +1 when upside down
    cos_theta = torch.clamp(-projected_gravity[..., 2], -1.0, 1.0)
    return torch.acos(cos_theta)  # [num_envs, num_bodies]


class upright_orientation_after_standing(ManagerTermBase):
    """Penalize non-upright orientation only after standing for a minimum duration.

    Tracks how long each environment has been continuously standing above a height
    threshold. Only applies the orientation penalty after the robot has been standing
    for at least `min_standing_duration_s` seconds.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Track standing duration for each environment (in seconds)
        self._standing_duration = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        standing_height_threshold: float,
        min_standing_duration_s: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sensor_cfg: SceneEntityCfg | None = None,
        norm: Literal["l1", "l2"] = "l1",
    ) -> torch.Tensor:
        """Compute orientation penalty only after standing for minimum duration.

        Args:
            standing_height_threshold: Height above which robot is considered standing.
            min_standing_duration_s: Minimum standing duration before penalty applies.
            asset_cfg: Config with body_names for bodies to check (e.g., ["pelvis", "torso_link"]).
                If no body_names, uses root.
            sensor_cfg: Optional height sensor config.
            norm: "l1" returns sum of angles, "l2" returns sum of squared angles.
        """
        # Check current standing state
        is_standing = if_standing(env, standing_height_threshold, asset_cfg, sensor_cfg).bool()

        # Update standing duration: increment if standing, reset if not
        self._standing_duration[is_standing] += env.step_dt
        self._standing_duration[~is_standing] = 0.0

        # Only apply penalty if standing for long enough
        apply_penalty = (self._standing_duration >= min_standing_duration_s).float()

        # Compute orientation penalty (sum over all specified bodies)
        angles = _body_tilt_angles(env, asset_cfg)  # [num_envs, num_bodies]

        if norm == "l2":
            penalty = torch.sum(torch.square(angles), dim=-1)
        else:
            penalty = torch.sum(angles, dim=-1)

        return penalty * apply_penalty

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset standing duration for specified environments."""
        self._standing_duration[env_ids] = 0.0


def severely_tilted_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold_rad: float = 1.5708,  # 90 degrees
) -> torch.Tensor:
    """Penalize when any specified body is tilted beyond threshold from vertical.

    Returns 1.0 if any body exceeds the threshold, 0.0 otherwise.
    This penalty applies from the start of the episode (not gated by standing).

    Args:
        env: Environment instance.
        asset_cfg: Config with body_names for bodies to check (e.g., ["pelvis", "torso_link"]).
            If no body_names, uses root.
        threshold_rad: Tilt angle threshold in radians (default: pi/2 = 90 degrees).

    Returns:
        Binary penalty: 1.0 if any body is severely tilted, 0.0 otherwise.
    """
    angles = _body_tilt_angles(env, asset_cfg)  # [num_envs, num_bodies]
    severely_tilted = (angles > threshold_rad).any(dim=-1)
    return severely_tilted.float()


def body_orientation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    axis: Literal["roll", "pitch"] = "roll",
    direction: Literal["both", "forward", "backward"] = "both",
    kernel: Literal["l1", "l2"] = "l1",
) -> torch.Tensor:
    """Penalize roll or pitch orientation of specified bodies.

    Uses projected gravity in body frame:
    - Roll = Y-component (positive = lean right, negative = lean left)
    - Pitch = X-component (positive = lean forward, negative = lean backward)

    Args:
        env: Environment instance.
        asset_cfg: Config with body_names for bodies to check.
        axis: Which axis to penalize: "roll" (Y-component) or "pitch" (X-component).
        direction: Which direction to penalize:
            "both" = penalize any deviation (absolute value).
            "forward" = only penalize positive component (forward pitch / right roll).
            "backward" = only penalize negative component (backward pitch / left roll).
        kernel: "l1" for absolute value, "l2" for squared.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    body_quats = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    if body_quats.dim() == 2:
        body_quats = body_quats.unsqueeze(1)

    gravity_vec = asset.data.GRAVITY_VEC_W.torch.unsqueeze(1).expand(-1, body_quats.shape[1], -1)
    projected_gravity = math_utils.quat_apply_inverse(body_quats, gravity_vec)

    component = projected_gravity[..., 0] if axis == "pitch" else projected_gravity[..., 1]

    if direction == "forward":
        component = torch.clamp(component, min=0.0)
    elif direction == "backward":
        component = torch.clamp(-component, min=0.0)
    else:
        component = component.abs()

    if kernel == "l2":
        return torch.mean(component**2, dim=-1)
    return torch.mean(component, dim=-1)


def body_ang_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body angular velocity using L2 norm.

    This reward penalizes high angular velocities of a specified body/link,
    useful for reducing shaking/oscillations without affecting linear motion.
    If no body_names is specified in asset_cfg, defaults to using the root link.

    Usage:
        # For root angular velocity:
        root_ang_vel = RewTerm(func=body_ang_vel_l2, weight=-0.01)

        # For a specific link (e.g., torso):
        torso_ang_vel = RewTerm(
            func=body_ang_vel_l2,
            weight=-0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
        )

    Args:
        env: The environment.
        asset_cfg: Asset configuration. Use body_names to specify a link, otherwise uses root.

    Returns:
        L2 norm of the body's angular velocity (sum of squared components).
    """
    # Extract the robot asset
    robot: Articulation = env.scene[asset_cfg.name]

    # Get angular velocity based on whether body_names is specified
    if asset_cfg.body_ids is not None and len(asset_cfg.body_ids) > 0:
        # Use specified body angular velocity
        body_idx = asset_cfg.body_ids[0]
        ang_vel = robot.data.body_ang_vel_w.torch[:, body_idx, :]  # [num_envs, 3]
    else:
        # Default to root angular velocity
        ang_vel = robot.data.root_ang_vel_w.torch  # [num_envs, 3]

    # Compute L2 penalty (sum of squared angular velocities)
    return torch.sum(torch.square(ang_vel), dim=-1)


def bodies_lin_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.0,
) -> torch.Tensor:
    """Penalize linear velocity magnitude of multiple bodies above a deadzone threshold.

    Enforces quasi-static motion: all specified body parts should move slowly.
    Uses 3D velocity magnitude (not just vertical). L2 penalty on excess velocity
    is summed across all specified bodies.

    Usage:
        body_velocity = RewTerm(
            func=bodies_lin_vel_l2,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[
                    "pelvis", "torso_link", ".*_hip_.*_link", ".*_knee_link",
                ]),
                "threshold": 0.3,  # m/s deadzone
            },
        )

    Args:
        env: The environment.
        asset_cfg: Asset configuration with body_names for the bodies to penalize.
        threshold: Deadzone in m/s. Speeds below this are not penalized.

    Returns:
        Sum of L2 penalties on velocity magnitude across all specified bodies.
    """
    robot: Articulation = env.scene[asset_cfg.name]

    if asset_cfg.body_ids is not None and len(asset_cfg.body_ids) > 0:
        body_vel = robot.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :]  # (num_envs, num_bodies, 3)
        vel_magnitude = torch.norm(body_vel, dim=-1)  # (num_envs, num_bodies)
        excess = vel_magnitude - threshold
        return torch.sum(torch.square(torch.clamp(excess, min=0.0)), dim=-1)
    else:
        vel_magnitude = torch.norm(robot.data.root_lin_vel_w.torch[:, :3], dim=-1)
        excess = vel_magnitude - threshold
        return torch.square(torch.clamp(excess, min=0.0))


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
        current_height = asset.data.root_pos_w.torch[:, 2] - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        current_height = asset.data.root_pos_w.torch[:, 2]

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
    feet_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]  # [num_envs, num_feet, 4]

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
    command_name: str | None = None,
    ang_vel_threshold: float = 0.3,
    reduction_scale: float = 2.0,
) -> torch.Tensor:
    """Penalize yaw difference between left and right feet (Isaac Gym style).

    Encourages both feet to have similar yaw orientation, but reduces penalty
    when turning (angular velocity command is high).

    Args:
        env: The environment.
        asset_cfg: Configuration for the robot asset.
        command_name: Name of the command to get angular velocity from. If None, no reduction.
        ang_vel_threshold: Angular velocity threshold (rad/s) above which to start reducing penalty.
        reduction_scale: How strongly to reduce penalty based on angular velocity magnitude.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get feet quaternions
    feet_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]  # [num_envs, num_feet, 4]

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

    # Compute base penalty
    penalty = torch.square(yaw_diff)

    # Reduce penalty when turning (if command_name is provided)
    if command_name is not None:
        ang_vel_cmd = env.command_manager.get_command(command_name)[:, 2]  # Angular z command

        # Compute scaling factor based on angular velocity magnitude
        # When |ang_vel| > threshold, reduce penalty proportionally
        ang_vel_magnitude = torch.abs(ang_vel_cmd)

        # Smooth reduction: 1.0 when |ang_vel| <= threshold, decreasing towards 0 as |ang_vel| increases
        scale_factor = torch.where(
            ang_vel_magnitude <= ang_vel_threshold,
            torch.ones_like(ang_vel_magnitude),
            torch.clamp(1.0 - reduction_scale * (ang_vel_magnitude - ang_vel_threshold), min=0.0),
        )

        penalty = penalty * scale_factor

    return penalty


def feet_yaw_mean_vs_base(
    env: ManagerBasedRLEnv,
    feet_asset_cfg: SceneEntityCfg,
    base_body_cfg: SceneEntityCfg,
    command_name: str | None = None,
    ang_vel_threshold: float = 0.3,
    reduction_scale: float = 2.0,
) -> torch.Tensor:
    """Penalize the squared yaw of each foot relative to the base frame.

    This encourages the feet to stay rotationally aligned with the base's
    forward direction by minimizing the yaw component of their relative orientation.
    Reduces penalty when turning (angular velocity command is high).

    Args:
        env: The environment.
        feet_asset_cfg: Configuration for the feet bodies.
        base_body_cfg: Configuration for the base body.
        command_name: Name of the command to get angular velocity from. If None, no reduction.
        ang_vel_threshold: Angular velocity threshold (rad/s) above which to start reducing penalty.
        reduction_scale: How strongly to reduce penalty based on angular velocity magnitude.
    """
    asset: Articulation = env.scene[feet_asset_cfg.name]

    # Get feet quaternions and base quaternion
    feet_quat = asset.data.body_quat_w.torch[:, feet_asset_cfg.body_ids]  # [num_envs, 2, 4]
    base_quat = asset.data.body_quat_w.torch[:, base_body_cfg.body_ids].squeeze(1)  # [num_envs, 4]

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

    # Compute base penalty
    penalty = torch.square(feet_yaw_relative).sum(dim=1)

    # Reduce penalty when turning (if command_name is provided)
    if command_name is not None:
        ang_vel_cmd = env.command_manager.get_command(command_name)[:, 2]  # Angular z command

        # Compute scaling factor based on angular velocity magnitude
        # When |ang_vel| > threshold, reduce penalty proportionally
        ang_vel_magnitude = torch.abs(ang_vel_cmd)

        # Smooth reduction: 1.0 when |ang_vel| <= threshold, decreasing towards 0 as |ang_vel| increases
        scale_factor = torch.where(
            ang_vel_magnitude <= ang_vel_threshold,
            torch.ones_like(ang_vel_magnitude),
            torch.clamp(1.0 - reduction_scale * (ang_vel_magnitude - ang_vel_threshold), min=0.0),
        )

        penalty = penalty * scale_factor

    return penalty


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
    norm: Literal["l1", "l2"] = "l1",
    error_threshold: float = 0.0,
    distance_mode: Literal["lateral", "absolute"] = "lateral",
    close_multiplier: float = 1.0,
    episode_delay_s: float = 0.0,
    episode_ramp_s: float = 0.0,
) -> torch.Tensor:
    """Penalize feet distance deviation from reference distance.

    This reward encourages maintaining proper spacing between left and right feet.

    Args:
        env: Environment instance.
        asset_cfg: Configuration for the robot asset (should specify foot body names).
        ref_distance: Reference distance between feet (meters).
        command_name: Optional velocity command name for lateral velocity gating.
        lateral_velocity_threshold: Lateral velocity above which penalty is suppressed.
        norm: "l1" or "l2" kernel.
        error_threshold: Dead zone — errors within this threshold produce zero penalty.
        distance_mode: "lateral" uses body-frame Y-axis distance only,
            "absolute" uses full 3D Euclidean distance between feet in world frame.
        close_multiplier: Multiplier applied to the error when feet are too close
            (distance < ref_distance). Makes the penalty asymmetrically steeper for
            feet approaching each other. With L2 norm the effective penalty scales
            by close_multiplier^2.
        episode_delay_s: Seconds at episode start with zero penalty (recovery window).
        episode_ramp_s: Seconds over which penalty linearly ramps from 0 to 1 after delay.

    Returns:
        Penalty tensor [num_envs] - higher when feet distance deviates from reference.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get feet positions - assumes asset_cfg.body_ids contains left and right foot indices
    # Shape: [num_envs, num_feet, 3]
    feet_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]

    # Ensure we have exactly 2 feet
    if len(asset_cfg.body_ids) != 2:
        # If not exactly 2 feet specified, return zeros (no penalty)
        return torch.zeros(env.num_envs, device=env.device)

    if distance_mode == "absolute":
        # Full 3D Euclidean distance in world frame — works regardless of body orientation
        distance = torch.norm(feet_pos_w[:, 0] - feet_pos_w[:, 1], dim=-1)
    else:
        feet_pos_b = transform_to_asset_frame(feet_pos_w, asset)
        left_foot_pos = feet_pos_b[:, 0]  # [num_envs, 3]
        right_foot_pos = feet_pos_b[:, 1]  # [num_envs, 3]
        # Lateral (Y-axis) distance in body frame
        distance = torch.abs(left_foot_pos[:, 1] - right_foot_pos[:, 1])

    # Compute deviation from reference distance
    distance_error = distance - ref_distance

    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        lateral_velocity_command = command[:, 1].abs()
        large_lateral_velocity = lateral_velocity_command > lateral_velocity_threshold
        distance_error[large_lateral_velocity] = 0

    # One-sided hard barrier: too close is penalized immediately,
    # too far is tolerated up to error_threshold, then full error kicks in (discontinuous).
    distance_error = torch.where(
        distance_error > error_threshold,
        distance_error,
        torch.where(distance_error < 0, distance_error, torch.zeros_like(distance_error)),
    )

    # Asymmetric penalty: amplify error when feet are too close
    if close_multiplier != 1.0:
        distance_error = torch.where(
            distance_error < 0,
            distance_error * close_multiplier,
            distance_error,
        )

    if norm == "l1":
        penalty = torch.abs(distance_error)
    elif norm == "l2":
        penalty = distance_error**2
    else:
        raise ValueError(f"Invalid norm: {norm}. Must be 'l1' or 'l2'.")

    # Episode time gate: zero penalty during delay, linear ramp after
    if episode_delay_s > 0 or episode_ramp_s > 0:
        episode_time = env.episode_length_buf * env.step_dt
        if episode_ramp_s > 0:
            scale = torch.clamp((episode_time - episode_delay_s) / episode_ramp_s, 0.0, 1.0)
        else:
            scale = (episode_time >= episode_delay_s).float()
        penalty = penalty * scale

    return penalty


def feet_distance_from_ref_if_standing(
    env: ManagerBasedRLEnv,
    standing_height_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ref_distance: float = 0.2,
    command_name: str | None = None,
    lateral_velocity_threshold: float = 0.5,
    sensor_cfg: SceneEntityCfg | None = None,
    norm: Literal["l1", "l2"] = "l1",
    error_threshold: float = 0.0,
    distance_mode: Literal["lateral", "absolute"] = "lateral",
    close_multiplier: float = 1.0,
    episode_delay_s: float = 0.0,
    episode_ramp_s: float = 0.0,
) -> torch.Tensor:
    distance_error = feet_distance_from_ref(
        env,
        asset_cfg=asset_cfg,
        ref_distance=ref_distance,
        command_name=command_name,
        lateral_velocity_threshold=lateral_velocity_threshold,
        norm=norm,
        error_threshold=error_threshold,
        distance_mode=distance_mode,
        close_multiplier=close_multiplier,
        episode_delay_s=episode_delay_s,
        episode_ramp_s=episode_ramp_s,
    )
    is_standing = if_standing(env, standing_height_threshold, asset_cfg, sensor_cfg)
    return distance_error * is_standing


def small_steps(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the agent for taking small steps.

    If a foot is in the air less than the threshold, it gets penalized depending on how long it is in the air.

    Args:
        env: The environment.
        threshold: The time threshold for the small steps.
        sensor_cfg: The configuration for the contact sensor.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the penalty
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    is_small_step = (air_time < threshold) & ~in_contact
    penalty = torch.where(is_small_step, threshold - air_time, 0.0) / threshold
    return penalty.sum(dim=1)


def jumping(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize if no foot is in contact with the ground.

    Args:
        env: The environment.
        threshold: The force threshold for the jumping.
        sensor_cfg: The configuration for the foot contact sensor.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the penalty
    feet_forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids].norm(dim=2)
    not_in_contact = feet_forces < threshold
    is_jumping = not_in_contact.all(dim=1)

    return is_jumping.float()


class impact_velocity(ManagerTermBase):
    """Penalize current body velocity during contact.

    The contact sensor reports forces but not body velocities, so the monitored sensor bodies are mapped to the
    corresponding articulation bodies by name. The mapping is cached when the reward term is initialized.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        asset: Articulation = env.scene[asset_cfg.name]

        sensor_body_names = contact_sensor.body_names
        if isinstance(sensor_cfg.body_ids, slice):
            selected_body_names = sensor_body_names[sensor_cfg.body_ids]
        else:
            selected_body_names = [sensor_body_names[index] for index in sensor_cfg.body_ids]
        self._asset_body_ids = asset.find_bodies(selected_body_names, preserve_order=True)[0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        force_threshold: float = 10.0,
        kernel: str = "l1",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Compute the summed impact-velocity penalty for each environment.

        Args:
            env: The environment containing the contact sensor and articulation.
            sensor_cfg: Contact sensor and monitored body selection.
            force_threshold: Force in newtons above which a body is considered in contact.
            kernel: Penalty kernel: ``"l1"`` for velocity or ``"l2"`` for squared velocity.
            asset_cfg: Articulation whose body velocities are penalized.

        Returns:
            Per-environment summed impact-velocity penalty.
        """
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        asset: Articulation = env.scene[asset_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history.torch
        in_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > force_threshold
        )
        body_velocities = torch.norm(asset.data.body_lin_vel_w.torch[:, self._asset_body_ids], dim=-1)
        impact_velocities = torch.where(in_contact, body_velocities, 0.0)
        if kernel == "l2":
            impact_velocities = torch.square(impact_velocities)
        return impact_velocities.sum(dim=1)


def no_undersired_base_velocity_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
) -> torch.Tensor:
    """Reward zero base velocity if it is not desired."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_z = torch.square(asset.data.root_lin_vel_b.torch[:, 2])
    ang_vel_xy = torch.sum(torch.square(asset.data.root_ang_vel_b.torch[:, :2]), dim=1)
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
    lin_vel_z = torch.square(asset.data.root_lin_vel_b.torch[:, 2]) * lin_vel_z_weight
    ang_vel_xy = torch.sum(torch.square(asset.data.root_ang_vel_b.torch[:, :2]), dim=1)
    reward = torch.exp(-(lin_vel_z + ang_vel_xy) / std**2)
    return reward


def lin_vel_z(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity_height",
    robot_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Reward for penalizing vertical linear velocity.

    Penalizes z-axis base linear velocity when the robot is in stance mode.

    Args:
        env: Environment instance.
        command_name: Name of the command generator.
        height_threshold: Height threshold for stance mode.
        robot_cfg: Configuration for the robot asset.

    Returns:
        Reward tensor.
    """
    # Create default robot_cfg if None is provided
    robot, _ = get_robot_cfg(env, robot_cfg)

    # Get base linear velocity from the robot data
    base_lin_vel = robot.data.root_lin_vel_w.torch

    # Get vertical component (z-axis)
    lin_vel_z = base_lin_vel[:, 2]

    # Get the command from the command manager
    command = env.command_manager.get_command(command_name)

    is_null_cmd = (command[:, :3] == 0).all(dim=1)

    # Compute reward: square of vertical velocity when in stance mode
    reward = torch.square(lin_vel_z) * is_null_cmd.float()

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
    feet_z_forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2].abs()
    mean_force = feet_z_forces.mean(dim=1)
    reward = 1.0 - torch.abs(mean_force.unsqueeze(1) - feet_z_forces).mean(dim=1) / (mean_force + 1e-6)

    return reward


class ground_unloaded(ManagerTermBase):
    """Penalty for not bearing weight on specified contact bodies. Returns 0-1.

    0 when all weight is on the specified bodies, 1 when no ground contact.
    Use with a negative weight to penalize jumping or lifting off the ground.
    Can be used with feet only, or with additional bodies (knees, hands, etc.)
    to also reward ground contact when the robot is down.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        gravity = abs(env.cfg.sim.gravity[2])
        self._expected_weight = asset.data.default_mass.torch.sum(dim=1).to(env.device) * gravity

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: ARG002
        command_name: str | None = None,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        feet_z_forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2]
        total_feet_force = feet_z_forces.sum(dim=1)
        grounded_ratio = torch.clamp(total_feet_force / (self._expected_weight + 1e-6), 0.0, 1.0)
        penalty = 1.0 - grounded_ratio
        if command_name is not None:
            command_term = env.command_manager.get_term(command_name)
            penalty = penalty * (command_term.target_height >= 0).float()
        return penalty


def completely_airborne(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary penalty when no body has ground contact at all.

    Returns 1.0 when the robot is completely airborne (no body touching the ground),
    0.0 otherwise. Use with a large negative weight to severely penalize jumping/flying.

    Unlike ground_unloaded which only checks feet, this checks ALL specified bodies
    (pass body_names=".*" to check everything). The robot can be lying down on its
    knees/torso and won't be penalized — only fully airborne states are penalized.

    Args:
        env: The environment.
        sensor_cfg: Contact sensor config with body_names specifying which bodies to check.
        threshold: Minimum force magnitude (N) to consider a body as "in contact".
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w shape: (num_envs, num_bodies, 3)
    forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :]
    # Force magnitude per body
    force_magnitude = torch.norm(forces, dim=-1)  # (num_envs, num_bodies)
    # Any body in contact?
    any_contact = (force_magnitude > threshold).any(dim=-1)  # (num_envs,)
    # Penalty = 1.0 when completely airborne, 0.0 when any body touches ground
    return (~any_contact).float()


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
    """
    command_term = env.command_manager.get_term(command_name)
    is_null_cmd = (command_term.command[:, :3] == 0).all(dim=1)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_z_forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2].abs()

    # Calculate variance-based measure of equality
    total_force = feet_z_forces.sum(dim=1, keepdim=True)
    # Avoid division by zero when robot is in the air
    force_distribution = feet_z_forces / (total_force + 1e-6)
    ideal_distribution = 1.0 / feet_z_forces.shape[1]  # Equal distribution

    # Measure how close we are to ideal equal distribution
    reward = 1.0 - torch.abs(force_distribution - ideal_distribution).mean(dim=1) / (2 * ideal_distribution)

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
    feet_z_forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2].abs()
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
    # feet_pos = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    feet_quat_w = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
    root_quat_w = asset.data.root_quat_w.torch

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


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def moving(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, weight_lin: float = 1.0, weight_ang: float = 1.0
) -> torch.Tensor:
    """Penalize the agent for moving."""
    asset = env.scene[asset_cfg.name]
    lin_vels = asset.data.body_lin_vel_w.torch.norm(dim=-1)
    ang_vels = asset.data.body_ang_vel_w.torch.norm(dim=-1)

    penalty = lin_vels.mean(dim=1) * weight_lin + ang_vels.mean(dim=1) * weight_ang
    return penalty


def moving_if_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    error_threshold: float,
    weight_lin: float = 1.0,
    weight_ang: float = 1.0,
) -> torch.Tensor:
    """Penalize the agent for moving only when the height tracking error is below a threshold.

    Once the robot has reached its commanded height (error < threshold), it should stay still.
    While still moving toward the target, no penalty is applied.
    """
    command_term = env.command_manager.get_term(command_name)
    error = torch.abs(command_term.measured_height - command_term.target_height)
    is_tracking = (error < error_threshold).float()
    penalty = moving(env, asset_cfg, weight_lin, weight_ang)
    return penalty * is_tracking


def relaxation_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pos_weight: float = 1.0,
    torque_weight: float = 0.001,
) -> torch.Tensor:
    """Penalty for not being relaxed when height command is negative.

    Weighted sum of squared errors, gated on command < 0:
        intensity * pos_weight * sum(deviation²) + is_negative * torque_weight * sum(torque²)

    The position term is scaled by ``relaxation_intensity`` (0 at command=0,
    1 at the most negative command) so the pressure to reach default joints
    increases with more negative commands.  The torque term uses a binary gate
    (always full when command < 0).

    Use with a negative reward weight. Returns 0 when command is non-negative.

    Args:
        env: The environment.
        command_name: Name of the height command term.
        asset_cfg: Robot asset config (optionally with joint_names to restrict to specific joints).
        pos_weight: Weight for joint position deviation from default.
        torque_weight: Weight for joint torques.
    """
    command_term: SmoothHeightCommand = env.command_manager.get_term(command_name)
    intensity = command_term.relaxation_intensity
    is_relaxation = (command_term.target_height < 0).float()

    asset: Articulation = env.scene[asset_cfg.name]

    deviation = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    torques = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]

    penalty = intensity * pos_weight * torch.sum(deviation**2, dim=1) + is_relaxation * torque_weight * torch.sum(
        torques**2, dim=1
    )

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


def base_height_above_min_if_right_side_up(
    env: ManagerBasedRLEnv,
    threshold_angle: float,
    min_target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize low base height but only if the base is right side up."""
    asset: Articulation = env.scene[asset_cfg.name]

    right_side_up = torch.acos(-asset.data.projected_gravity_b.torch[:, 2]).abs() < threshold_angle

    dist_below_target = torch.clamp(min_target_height - asset.data.root_pos_w.torch[:, 2], min=0.0)

    penalty = torch.where(right_side_up, torch.square(dist_below_target), 0.0)

    return right_side_up * penalty


def feet_to_close(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold_dist: float = 0.115) -> torch.Tensor:
    """Penalize if the feet are too close to each other."""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    feet_dist = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=1)
    penalty = torch.where(feet_dist < threshold_dist, threshold_dist / (feet_dist + 1e-6), 0.0)
    return penalty


def feet_distance_if_standing(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, target_dist: float
) -> torch.Tensor:
    """Penalize if the feet are too close or too far from each other."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    is_null_cmd = (command_term.command[:, :3] == 0).all(dim=1)
    feet_pos = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    feet_dist = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=1)
    error = target_dist - feet_dist
    penalty = torch.where(is_null_cmd, error**2, 0.0)
    return penalty


def dont_move_if_null_cmd(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for not moving if the command is null."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    is_null_cmd = (command_term.command[:, :3] == 0).all(dim=1)

    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    joint_vel_error = torch.sum(joint_vel.abs(), dim=1)

    # base_vel = asset.data.root_lin_vel_w.torch.norm(dim=1)

    reward = torch.where(is_null_cmd, torch.exp(-joint_vel_error / std**2), 0.0)

    return reward


def flat_body_orientation_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward flat orientation using exponential kernel.

    This is computed by rewarding small xy-components of the projected gravity vector.
    Args:
        std: std in rad
        asset_cfg:
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    body_quats = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    gravity_vec_expanded = asset.data.GRAVITY_VEC_W.torch.unsqueeze(1).expand(-1, len(asset_cfg.body_ids), -1)
    projected_gravity = math_utils.quat_apply_inverse(body_quats, gravity_vec_expanded)
    orientation_error = torch.sum(torch.square(projected_gravity[..., :2]), dim=-1)  # sum over x,y
    orientation_error_per_env = torch.sum(orientation_error, dim=-1)  # sum over k bodies

    return torch.exp(-orientation_error_per_env / std**2)


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

    orientation_error = torch.sum(torch.square(asset.data.projected_gravity_b.torch[:, :2]), dim=1)

    penalty = torch.where(is_null_cmd, orientation_error, 0.0)

    return penalty


def joint_deviation_from_default_exp(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg | None = None,
    sigma: float = 0.25,
    command_name: str = "base_velocity_height",
    apply_stance_condition: bool = False,
) -> torch.Tensor:
    """Minimize joint deviation from default angles.

    Args:
        env: Environment instance.
        robot_cfg: Configuration for the robot asset.
        sigma: Sigma for the exponential decay.
        command_name: Name of the command generator.
        height_threshold: Height threshold for stance mode.
        apply_stance_condition: Whether to apply the reward only in stance mode.

    Returns:
        Reward tensor.
    """
    # Get the robot asset with proper configuration
    robot, robot_cfg = get_robot_cfg(env, robot_cfg)

    # Compute joint deviation
    joint_deviation = (
        robot.data.joint_pos.torch[:, robot_cfg.joint_ids] - robot.data.default_joint_pos.torch[:, robot_cfg.joint_ids]
    )

    # Compute squared deviation
    squared_deviation = torch.sum(joint_deviation**2, dim=1)

    # Apply stance condition if requested
    if apply_stance_condition:
        # Get command from the command manager
        command = env.command_manager.get_command(command_name)

        is_null_cmd = (command[:, :3] == 0).all(dim=1)

        # Apply stance mode condition (only consider deviation in stance mode)
        squared_deviation = squared_deviation * is_null_cmd.float()

    # Compute reward (exponential decay based on squared deviation)
    reward = torch.exp(-squared_deviation / sigma)

    return reward


def squat_knee(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg | None = None, command_name: str = "base_velocity"
) -> torch.Tensor:
    """Directional knee-bend reward that ties height error to knee actuation.

    When the robot is above the target height (height_error > 0), the reward
    encourages bent knees (normalized > 0.5). When below target, it encourages
    extended knees. This gives the policy a biomechanical prior for how to
    achieve height changes.

    Uses the command term's sensor-based height (height above terrain) to
    match the height tracking reward.

    Use with a negative weight (e.g. -0.75) so that the correct knee
    configuration is rewarded.
    """
    robot, robot_cfg = get_robot_cfg(env, robot_cfg)

    command_term = env.command_manager.get_term(command_name)
    height_error = command_term.target_height - command_term.base_height

    joint_pos_limits = robot.data.joint_pos_limits.torch
    knee_min = joint_pos_limits[:, robot_cfg.joint_ids, 0]
    knee_max = joint_pos_limits[:, robot_cfg.joint_ids, 1]

    knee_normalized = (robot.data.joint_pos.torch[:, robot_cfg.joint_ids] - knee_min) / (knee_max - knee_min)

    return torch.sum(height_error.unsqueeze(-1) * (knee_normalized - 0.5), dim=-1)


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
    net_contact_forces = contact_sensor.data.net_forces_w_history.torch

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
    net_contact_forces = torch.norm(contact_sensor.data.net_forces_w_history.torch[:, :, feet_body_ids], dim=-1)

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


def link_distance_lateral(
    env: ManagerBasedRLEnv,
    min_distance: float = 0.2,
    max_distance: float = 0.4,
    command_name: str = "base_velocity_height",
    robot_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Reward for maintaining appropriate lateral distance between two links.

    Penalizes when two links are too close or too far apart laterally.

    Args:
        env: Environment instance.
        min_distance: Minimum desired lateral distance between links.
        max_distance: Maximum desired lateral distance between links (only enforced in stance mode).
        command_name: Name of the command generator.
        robot_cfg: Configuration for the robot asset.

    Returns:
        Reward tensor.
    """
    # Create default sensor_cfg and robot_cfg if None is provided
    robot, robot_cfg = get_robot_cfg(env, robot_cfg)

    # Get root state, quaternion, and feet positions
    root_pos = robot.data.root_pos_w.torch  # Shape: [num_envs, 3]
    root_quat = robot.data.root_quat_w.torch  # Shape: [num_envs, 4]

    # Get feet body IDs from sensor config
    link_body_ids = robot_cfg.body_ids

    # Ensure we have exactly 2 feet (left and right)
    if len(link_body_ids) != 2:
        raise ValueError(f"Expected 2 feet, but got {len(link_body_ids)}. Please specify correct body names.")

    # Get feet positions
    # Shape: [num_envs, num_feet, 3]
    link_pos_w = robot.data.body_pos_w.torch[:, link_body_ids]

    # Calculate feet positions relative to root
    link_pos_b = transform_to_body_frame(link_pos_w, root_pos, root_quat)

    # Get lateral (y-axis) distance between feet
    # Shape: [num_envs]
    lateral_distance = torch.abs(link_pos_b[:, 0, 1] - link_pos_b[:, 1, 1])

    # Penalize if feet are too close
    too_close_penalty = torch.clamp(min_distance - lateral_distance, min=0)

    # Penalize if feet are too far apart, but only in stance mode
    # Get command for stance mode check
    command = env.command_manager.get_command(command_name)
    is_null_cmd = (command[:, :3] == 0).all(dim=1)
    too_far_penalty = torch.clamp(lateral_distance - max_distance, min=0) * is_null_cmd.float()

    # Combine penalties
    total_penalty = too_close_penalty + too_far_penalty

    return total_penalty


def feet_lateral_distance_to_ref(
    env: ManagerBasedRLEnv,
    ref_distance: float = 0.22,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Penalize deviation of lateral feet distance from a reference value.

    Projects foot positions into the root body frame and computes squared error
    between the Y-axis separation and the reference distance.

    Args:
        env: Environment instance.
        ref_distance: Target lateral distance between feet (m).
        asset_cfg: Asset config specifying exactly 2 foot links.

    Returns:
        Squared error penalty (positive value, apply with negative weight).
    """
    robot, _ = get_robot_cfg(env, asset_cfg)
    link_pos_w = robot.data.body_pos_w.torch[:, asset_cfg.body_ids]

    if len(asset_cfg.body_ids) != 2:
        raise ValueError("feet_lateral_distance_to_ref requires exactly 2 links")

    root_pos = robot.data.root_pos_w.torch
    root_quat = robot.data.root_quat_w.torch
    link0_b, _ = math_utils.subtract_frame_transforms(root_pos, root_quat, link_pos_w[:, 0])
    link1_b, _ = math_utils.subtract_frame_transforms(root_pos, root_quat, link_pos_w[:, 1])

    lateral_dist = torch.abs(link0_b[:, 1] - link1_b[:, 1])
    return (lateral_dist - ref_distance).square()


def feet_ground_parallel(
    env: ManagerBasedRLEnv,
    contact_threshold: float = 0.5,
    min_contact_duration: float = 0.03,  # 3 * typical dt
    sensor_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Reward for keeping feet parallel to the ground.

    Penalizes feet that are not in contact with the ground or are in contact for
    too short a duration.

    Args:
        env: Environment instance.
        contact_threshold: Threshold for determining foot contact.
        min_contact_duration: Minimum duration for considering a foot in contact.
        sensor_cfg: Contact sensor configuration for feet.

    Returns:
        Reward tensor.
    """
    # Create robot
    contact_sensor, sensor_cfg = get_contact_sensor_cfg(env, sensor_cfg)

    # Get contact forces for these bodies
    contact_forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]

    # Determine which feet are in contact with the ground (vertical force > threshold)
    # Shape: [num_envs, num_feet]
    feet_in_contact = contact_forces[:, :, 2] > contact_threshold

    # Calculate contact duration for each foot
    # Shape: [num_envs, num_feet]
    contact_duration = torch.sum(feet_in_contact, dim=1)

    # Calculate penalty based on contact duration
    # Penalty is proportional to the number of feet without contact or in
    # contact for too short a duration
    penalty = torch.sum(torch.clamp(min_contact_duration - contact_duration, min=0), dim=1)

    return penalty


def feet_swing_height(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.08,
    contact_threshold: float = 1.0,
    robot_cfg: SceneEntityCfg = None,
    sensor_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Reward for feet swing height if the feet are not in contact with the ground.

    This reward encourages each foot to lift above the height_threshold when that foot
    is not in contact with the ground. It penalizes feet that are below the threshold
    during swing phase.

    Args:
        env: Environment instance.
        height_threshold: Target height (in meters) for feet during swing phase.
        contact_threshold: Force threshold to determine if a foot is in contact.
        robot_cfg: Configuration for the robot asset. If None, uses default robot.
        sensor_cfg: Contact sensor configuration containing feet body IDs.
            If None, uses default contact sensor.

    Returns:
        Reward tensor with shape [num_envs]. Higher values indicate feet are
        below the target height during swing phase (should be penalized with negative weight).
    """
    contact_sensor, sensor_cfg = get_contact_sensor_cfg(env, sensor_cfg)

    # Get feet body IDs from sensor config
    feet_body_ids = sensor_cfg.body_ids

    # Assert that there should be two feet
    if len(feet_body_ids) != 2:
        raise ValueError(
            f"Expected 2 feet bodies in sensor config, got {len(feet_body_ids)}. "
            f"Please ensure the sensor configuration includes both feet."
        )

    # Get contact forces for each foot individually
    net_contact_forces = torch.norm(contact_sensor.data.net_forces_w.torch[:, feet_body_ids], dim=-1)

    # Check contact for each foot individually: [num_envs, num_feet]
    # Use current timestep contact forces
    feet_contact = net_contact_forces > contact_threshold

    robot, _ = get_robot_cfg(env, robot_cfg)
    feet_pos = robot.data.body_pos_w.torch[:, feet_body_ids[:2]]

    # For each foot: if not in contact, penalize being below height_threshold
    # feet_pos[:, :, 2] is the z-coordinate (height) of each foot
    feet_height = feet_pos[:, :, 2]  # [num_envs, 2]

    # Only apply penalty when foot is not in contact
    # Penalize when foot height is below threshold during swing
    pos_error = torch.square(feet_height - height_threshold) * ~feet_contact

    return torch.sum(pos_error, dim=1)


def torso_feet_forward_alignment(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Reward for aligning torso forward direction with feet orientation.

    This reward encourages the robot's torso to face the same direction as its feet
    are pointing. The feet forward direction is computed as the perpendicular to the
    line connecting the two feet in the horizontal plane. Only horizontal components
    are considered for alignment to avoid penalizing natural pitch during walking.

    Args:
        env: Environment instance.
        sensor_cfg: Contact sensor configuration containing feet body IDs.
            If None, uses default contact sensor.
        robot_cfg: Configuration for the robot asset. If None, uses default robot.

    Returns:
        Reward tensor with shape [num_envs]. Values range from 0 to 1, where:
        - 1.0 = perfect alignment (torso facing same direction as feet)
        - 0.0 = perpendicular orientations
        - -1.0 = opposite directions
    """
    # Get robot and sensor configurations
    robot, robot_cfg = get_robot_cfg(env, robot_cfg)

    # Find torso body ID
    torso_body_ids, _ = robot.find_bodies([".*torso.*"])
    if len(torso_body_ids) == 0:
        raise ValueError("Could not find torso body. Please check robot model.")
    torso_body_id = torso_body_ids[0]

    # Get feet body IDs from sensor config
    feet_body_ids = robot_cfg.body_ids
    if len(feet_body_ids) < 2:
        raise ValueError(
            f"Expected at least 2 feet bodies in sensor config, got {len(feet_body_ids)}. "
            f"Please ensure the sensor configuration includes both feet."
        )

    # Get feet positions in world frame - [num_envs, 2, 3]
    feet_pos = robot.data.body_pos_w.torch[:, feet_body_ids[:2]]
    left_foot_pos = feet_pos[:, 0, :]  # [num_envs, 3]
    right_foot_pos = feet_pos[:, 1, :]  # [num_envs, 3]

    # Get torso quaternion - [num_envs, num_bodies, 4]
    body_quat_w = robot.data.body_link_quat_w.torch
    torso_quat = body_quat_w[:, torso_body_id, :]  # [num_envs, 4]

    # Convert torso quaternion to forward direction vector
    # In most robot conventions, forward is +x axis in body frame
    # Quaternion rotation of unit x-vector: x' = q * [1,0,0] * q^(-1)
    w = torso_quat[:, 0]
    x = torso_quat[:, 1]
    y = torso_quat[:, 2]
    z = torso_quat[:, 3]

    # Rotated x-axis components
    torso_forward_x = 1.0 - 2.0 * (y * y + z * z)
    torso_forward_y = 2.0 * (x * y + w * z)
    torso_forward_z = 2.0 * (x * z - w * y)
    torso_forward = torch.stack([torso_forward_x, torso_forward_y, torso_forward_z], dim=1)

    # Compute feet forward direction
    # Lateral vector: from right foot to left foot
    feet_lateral = left_foot_pos - right_foot_pos  # [num_envs, 3]

    # Forward direction is perpendicular to lateral in horizontal plane
    # Using cross product: forward = lateral x up
    up_vector = torch.zeros_like(feet_lateral)
    up_vector[:, 2] = 1.0  # z-axis points up

    # Cross product to get forward direction
    feet_forward = torch.cross(feet_lateral, up_vector, dim=1)

    # Project to horizontal plane (only x,y components)
    torso_forward_horizontal = torso_forward[:, :2]
    feet_forward_horizontal = feet_forward[:, :2]

    # Normalize the horizontal vectors
    torso_forward_norm = torch.norm(torso_forward_horizontal, dim=1, keepdim=True)
    feet_forward_norm = torch.norm(feet_forward_horizontal, dim=1, keepdim=True)

    # Avoid division by zero
    torso_forward_horizontal = torso_forward_horizontal / (torso_forward_norm + 1e-6)
    feet_forward_horizontal = feet_forward_horizontal / (feet_forward_norm + 1e-6)

    # Compute alignment using dot product
    # Dot product of unit vectors gives cos(angle between them)
    alignment = torch.sum(torso_forward_horizontal * feet_forward_horizontal, dim=1)

    return alignment


def torso_orientation_target(
    env: ManagerBasedRLEnv,
    target_angles: dict | None = None,
    std: float = 0.25,
    robot_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Reward for maintaining torso orientation close to target angles.

    This reward encourages the robot's torso to maintain specific roll, pitch,
    and/or yaw angles in the world frame. You can track any combination of these
    angles by specifying them in the target_angles dictionary. The reward uses
    exponential kernels to provide smooth gradients around the target angles.

    Args:
        env: Environment instance.
        target_angles: Dictionary specifying which angles to track and their targets.
            Keys can be 'roll', 'pitch', 'yaw' with values in radians.
            Example: {'yaw': 0.0} tracks only yaw
            Example: {'roll': 0.0, 'pitch': 0.1} tracks roll and pitch
            Example: {'roll': 0.0, 'pitch': 0.0, 'yaw': 1.57} tracks all three
            Default is {'yaw': 0.0} for backward compatibility.
        std: Standard deviation for the exponential kernel. Smaller values create
            a sharper peak around the target. Can be a float (same for all angles)
            or a dict with keys 'roll', 'pitch', 'yaw' for different stds.
            Default is 0.25.
        robot_cfg: Configuration for the robot asset. If None, uses default robot.

    Returns:
        Reward tensor with shape [num_envs]. Values range from 0 to 1, with 1
        being perfect alignment with all target angles. When tracking multiple
        angles, the reward is the product of individual angle rewards.
    """
    if target_angles is None:
        target_angles = {"pitch": 0.0}

    # Handle std - convert to dict if needed
    if isinstance(std, int | float):
        std_dict = dict.fromkeys(target_angles.keys(), std)
    else:
        std_dict = std
        # Fill in defaults for any missing keys
        for angle in target_angles.keys():
            if angle not in std_dict:
                std_dict[angle] = 0.25

    # Get robot
    robot, _ = get_robot_cfg(env, robot_cfg)

    # Find torso body ID
    torso_body_ids, _ = robot.find_bodies([".*torso.*"])
    if len(torso_body_ids) == 0:
        raise ValueError("Could not find torso body. Please check robot model.")
    torso_body_id = torso_body_ids[0]

    # Get torso quaternion - [num_envs, num_bodies, 4]
    body_quat_w = robot.data.body_link_quat_w.torch
    torso_quat = body_quat_w[:, torso_body_id, :]  # [num_envs, 4]

    # Convert quaternion to Euler angles (roll, pitch, yaw)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(torso_quat)

    # Store angles in a dict
    angles = {"roll": roll, "pitch": pitch, "yaw": yaw}

    # Initialize reward as ones
    reward = torch.ones(env.num_envs, device=env.device)

    # Compute reward for each specified angle
    for angle_name, target_value in target_angles.items():
        if angle_name not in angles:
            raise ValueError(f"Invalid angle name '{angle_name}'. Must be 'roll', 'pitch', or 'yaw'.")

        # Get current angle
        current_angle = angles[angle_name]

        # Compute error
        angle_error = current_angle - target_value

        # Handle angle wrapping for roll and yaw (they wrap at ±pi)
        # Pitch doesn't wrap (it's limited to ±pi/2)
        if angle_name in ["roll", "yaw"]:
            angle_error = torch.atan2(torch.sin(angle_error), torch.cos(angle_error))

        # Compute individual reward using exponential kernel
        angle_std = std_dict[angle_name]
        angle_reward = torch.exp(-angle_error * angle_error / (angle_std * angle_std))

        # Multiply into total reward (product of individual rewards)
        reward = reward * angle_reward

    return reward


def joint_deviation_if_standing(
    env: ManagerBasedRLEnv,
    standing_height_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    mode: Literal["l1", "l2"] = "l1",
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    is_standing = if_standing(env, standing_height_threshold, asset_cfg, sensor_cfg)
    if mode == "l1":
        return torch.sum(torch.abs(angle), dim=1) * is_standing
    elif mode == "l2":
        return torch.sum(torch.square(angle), dim=1) * is_standing
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'l1' or 'l2'.")


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
    angle = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    is_standing = if_standing(env, standing_height_threshold, asset_cfg, sensor_cfg)
    return torch.sum(torch.exp(-torch.square(angle) / std**2), dim=1) * is_standing


def feet_air_time_positive_biped_command(
    env: ManagerBasedRLEnv, command_name: str, command_slice: slice, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, command_slice], dim=1) > 0.1
    return reward
