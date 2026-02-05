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
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms

from agile.rl_env.mdp.commands.tracking_commands import TrackingCommand


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Target anchor position relative to current anchor in body frame. Shape: (num_envs, 3)."""
    command: TrackingCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.command_anchor_pos_w,
        command.command_anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Target anchor orientation relative to current anchor as flattened rotation matrix. Shape: (num_envs, 6)."""
    command: TrackingCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.command_anchor_pos_w,
        command.command_anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_joint_pos_delta(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Target joint positions minus current joint positions. Shape: (num_envs, num_tracked_joints)."""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_tracked_joint_pos - command.robot_tracked_joint_pos


def object_to_hand_pos_b(
    env: ManagerBasedEnv,
    command_name: str,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Observation of object position relative to hand palm in body frame.

    This observation provides the policy with information about where the object is
    relative to the current hand position. Useful for manipulation tasks where the
    policy needs to know the hand-object spatial relationship.

    Args:
        env: The environment object.
        command_name: Name of the tracking command term (to access object position).
        ee_frame_cfg: Configuration for the end-effector frame sensor.

    Returns:
        Tensor of shape (num_envs, 3) with object position relative to hand in body frame.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    ee_frame: FrameTransformer = env.scene.sensors[ee_frame_cfg.name]

    # Get actual object position in world frame
    object_pos_w = command.object_pos_w  # (num_envs, 3)

    # Get actual hand palm position from ee_frame sensor
    # ee_frame.data.target_pos_w shape: (num_envs, num_targets, 3), we take first target
    hand_pos_w = ee_frame.data.target_pos_w[:, 0, :]  # (num_envs, 3)

    # Compute object position relative to hand in world frame
    object_to_hand_w = object_pos_w - hand_pos_w  # (num_envs, 3)

    # Transform to body frame using robot root orientation
    robot = env.scene["robot"]
    root_quat_w = robot.data.root_quat_w  # (num_envs, 4)
    object_to_hand_b = quat_apply_inverse(root_quat_w, object_to_hand_w)

    return object_to_hand_b


def trajectory_progress(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Observation of normalized trajectory progress.

    Returns a value in [0, 1] indicating how far along the reference trajectory
    the current timestep is. This helps the policy learn stage-dependent behavior:
    - 0.0 = beginning of trajectory (approach phase)
    - 0.5 = middle of trajectory (grasp/lift phase)
    - 1.0 = end of trajectory (place phase)

    Args:
        env: The environment object.
        command_name: Name of the tracking command term.

    Returns:
        Tensor of shape (num_envs, 1) with normalized progress in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Compute normalized progress: timestep_counter / (num_timesteps - 1)
    # Using num_timesteps - 1 so that the last timestep gives exactly 1.0
    progress = command.timestep_counter.float() / max(command.num_timesteps - 1, 1)

    # Clamp to [0, 1] for safety
    progress = torch.clamp(progress, 0.0, 1.0)

    return progress.unsqueeze(-1)  # (num_envs, 1)


def object_pos_error(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Observation of object position error (target - current).

    Returns the difference between target object position (from reference trajectory)
    and current object position. This tells the policy "how far is the object from
    where it should be".

    A zero vector means the object is exactly where the reference says it should be.
    Non-zero means the object needs to move in that direction to reach the target.

    Args:
        env: The environment object.
        command_name: Name of the tracking command term.

    Returns:
        Tensor of shape (num_envs, 3) with object position error.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get target object position from reference trajectory
    target_object_pos_w = command.command_object_pos_w  # (num_envs, 3)

    # Get current object position
    current_object_pos_w = command.object_pos_w  # (num_envs, 3)

    # Compute error: target - current (positive means object needs to move in that direction)
    object_error = target_object_pos_w - current_object_pos_w  # (num_envs, 3)

    return object_error
