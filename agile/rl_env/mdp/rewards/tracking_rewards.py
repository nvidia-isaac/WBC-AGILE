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


import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_error_magnitude

from agile.rl_env.mdp.commands.tracking_commands import TrackingCommand


def motion_global_anchor_position_error_exp(env: ManagerBasedEnv, command_name: str, std: float) -> torch.Tensor:
    command: TrackingCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.command_anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedEnv, command_name: str, std: float) -> torch.Tensor:
    # TODO: We may only care about the yaw angle difference
    command: TrackingCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.command_anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_object_position_error_exp(env: ManagerBasedEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward for tracking object position using exponential kernel.

    Args:
        env: The environment object.
        command_name: The name of the tracking command term.
        std: Standard deviation for the exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    if command.object is None:
        raise ValueError("Object tracking requires object_name to be set in TrackingCommandCfg")
    error = torch.sum(torch.square(command.command_object_pos_w - command.object_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_object_orientation_error_exp(env: ManagerBasedEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward for tracking object orientation using exponential kernel.

    Args:
        env: The environment object.
        command_name: The name of the tracking command term.
        std: Standard deviation for the exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    if command.object is None:
        raise ValueError("Object tracking requires object_name to be set in TrackingCommandCfg")
    error = quat_error_magnitude(command.command_object_quat_w, command.object_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_tracked_joint_pos_error_exp(env: ManagerBasedEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward for tracking joint positions using exponential kernel.

    Args:
        env: The environment object.
        command_name: The name of the tracking command term.
        std: Standard deviation for the exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    error = command.command_tracked_joint_pos - command.robot_tracked_joint_pos
    error = torch.sum(torch.square(error), dim=-1)
    return torch.exp(-error / std**2)


def hand_object_distance_tracking_exp(
    env: ManagerBasedEnv,
    command_name: str,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    release_decay_steps: int = 50,
) -> torch.Tensor:
    """Reward for hand-object proximity with automatic phase detection.

    This reward automatically detects the task phase using the reference trajectory's
    object height profile, without requiring manual progress thresholds. It works for
    any pick-and-place trajectory by detecting the "lift peak".

    Phase detection (fully automatic):
    - BEFORE PEAK: Full proximity reward (approach + grasp + lift phases)
      → Encourages hand to reach and grasp the object
    - AFTER PEAK: Decaying proximity reward over `release_decay_steps`
      → Gradually allows hand to release as object is placed

    The peak is automatically detected as the timestep where the reference object
    height is maximum. This makes the reward generic for different trajectories.

    Timeline visualization:
        Reference Height:
                        ╱╲ peak (auto-detected)
                       ╱  ╲
         ─────────────╱    ╲─────────
         [Full reward]      [Decay → 0]
         (approach/grasp)   (release)

    Args:
        env: The environment object.
        command_name: Name of the tracking command term.
        std: Standard deviation for the exponential kernel.
        ee_frame_cfg: Configuration for the end-effector frame sensor.
        release_decay_steps: Number of steps after peak to decay proximity reward
            from 1.0 to 0.0. Default 50 steps for smooth transition.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    ee_frame: FrameTransformer = env.scene.sensors[ee_frame_cfg.name]

    # Get actual object position in world frame
    object_pos_w = command.object_pos_w  # (num_envs, 3)

    # Get actual hand palm position from ee_frame sensor
    hand_pos_w = ee_frame.data.target_pos_w[:, 0, :]  # (num_envs, 3)

    # Compute actual hand-object distance
    hand_object_dist = torch.norm(object_pos_w - hand_pos_w, dim=-1)  # (num_envs,)

    # Compute proximity reward using exponential kernel
    proximity_reward = torch.exp(-(hand_object_dist**2) / std**2)

    # === Automatic phase detection using reference trajectory ===

    # Use precomputed peak timestep from TrackingCommand (computed once at init)
    peak_timestep = command.object_height_peak_timestep

    # Current timestep for each environment
    current_timestep = command.timestep_counter  # (num_envs,)

    # Compute decay weight based on distance from peak
    # Before/at peak: weight = 1.0 (full proximity reward)
    # After peak: weight decays linearly from 1.0 to 0.0 over release_decay_steps
    steps_after_peak = (current_timestep - peak_timestep).float()
    steps_after_peak = torch.clamp(steps_after_peak, min=0)  # Only count steps after peak

    # Linear decay: 1.0 at peak, 0.0 after release_decay_steps
    decay_weight = 1.0 - steps_after_peak / release_decay_steps
    decay_weight = torch.clamp(decay_weight, min=0.0, max=1.0)

    # Apply decay to proximity reward
    gated_reward = proximity_reward * decay_weight

    return gated_reward
