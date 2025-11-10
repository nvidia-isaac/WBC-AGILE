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


from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.rewards import action_l2
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import RewardTermCfg

from agile.rl_env.mdp.utils import get_contact_sensor_cfg, get_robot_cfg


def relax_if_null_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the agent for torque if the command is null."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    is_null_cmd = (command_term.command[:, :3] == 0).all(dim=1)

    torques = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)

    penalty = torch.where(is_null_cmd, torques, 0.0)

    return penalty


class relax_if_null_cmd_exp(ManagerTermBase):
    """Reward the agent for using low torques when the command is null.

    Returns a reward (not penalty) that is higher when torques are lower.
    The reward is 1.0 when torques are zero and decreases as torques increase.

    This class caches the torque limits during initialization to avoid costly
    CPU-to-GPU transfers during training.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg) -> None:
        """
        Initialize the relaxation reward calculator.

        Args:
            env: Environment instance.
            cfg: Configuration for the reward term.
        """
        super().__init__(cfg, env)

        # Cache torque limits once during initialization (avoid CPU->GPU transfer every step)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset = env.scene[asset_cfg.name]
        self.torque_limits = asset.root_physx_view.get_dof_max_forces()[:, asset_cfg.joint_ids].to(env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float = 0.25,
        asset_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        """
        Calculate reward for low torque usage when command is null.

        Args:
            env: Environment instance.
            command_name: Name of the command term to check.
            std: Standard deviation for Gaussian reward shaping.
            asset_cfg: Configuration for the robot asset. If None, uses default "robot".

        Returns:
            Reward tensor where 1.0 = no torque, 0.0 = high torque (only when cmd is null).
        """
        # Get robot configuration
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg("robot")
        asset = env.scene[asset_cfg.name]

        # Check if command is null
        command_term = env.command_manager.get_term(command_name)
        is_null_cmd = (command_term.command[:, :3] == 0).all(dim=1)

        # Calculate normalized torque magnitude using cached limits
        torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
        normalized_torques = torques / (self.torque_limits + 1e-6)

        # Compute RMS (root mean square) of normalized torques
        # This gives us a single value representing overall torque usage
        rms_norm_torque = torch.sqrt(torch.mean(torch.square(normalized_torques), dim=1))

        # Convert to reward using Gaussian
        reward = torch.exp(-((rms_norm_torque / std) ** 2))

        # Only apply when command is null
        return torch.where(is_null_cmd, reward, torch.zeros_like(reward))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset is not needed for this term as torque limits don't change."""
        pass


def action_rate_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:  # noqa: ARG001
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action),
        dim=1,
    )


def action_rate_l2_if_actor_active(
    env: ManagerBasedRLEnv, rest_duration_s: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    action_rate = action_rate_l2(env, asset_cfg)
    env_in_rest_phase = env.episode_length_buf < int(rest_duration_s / env.step_dt)
    action_rate[env_in_rest_phase] = 0
    return action_rate


def action_l2_if_actor_active(env: ManagerBasedRLEnv, rest_duration_s: float) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    action = action_l2(env)
    env_in_rest_phase = env.episode_length_buf < int(rest_duration_s / env.step_dt)
    action[env_in_rest_phase] = 0
    return action


def joint_deviation_l2(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward for penalizing joint deviation from default angles."""
    robot, robot_cfg = get_robot_cfg(env, robot_cfg)
    return torch.sum(
        torch.square(
            robot.data.joint_pos[:, robot_cfg.joint_ids] - robot.data.default_joint_pos[:, robot_cfg.joint_ids]
        ),
        dim=1,
    )


def contact_forces_l2(
    env: ManagerBasedRLEnv,
    contact_force_threshold: float = 100.0,
    sensor_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Reward for penalizing high contact forces using L2 squared kernel.

    Only computes penalty when force magnitude exceeds the specified threshold.

    Args:
        env: Environment instance.
        contact_force_threshold: Threshold for contact forces before penalization.
                                Only forces above this threshold are penalized.
        sensor_cfg: Contact sensor configuration.

    Returns:
        Reward tensor.
    """
    # Get the contact sensor from the scene
    contact_sensor, sensor_cfg = get_contact_sensor_cfg(env, sensor_cfg)

    # Get contact forces for these bodies.
    net_contact_forces = contact_sensor.data.net_forces_w

    # Get forces for the specified bodies
    # Shape: [num_envs, num_bodies, 3]
    body_forces = net_contact_forces[:, sensor_cfg.body_ids]

    # Compute force magnitudes
    # Shape: [num_envs, num_bodies]
    force_magnitudes = torch.norm(body_forces, dim=-1)

    # Apply threshold - zero out forces below threshold
    # Shape: [num_envs, num_bodies]
    thresholded_magnitudes = torch.where(
        force_magnitudes > contact_force_threshold,
        force_magnitudes - contact_force_threshold,
        torch.zeros_like(force_magnitudes),
    )

    # Square the thresholded magnitudes
    squared_thresholded_magnitudes = thresholded_magnitudes**2

    # Sum over all bodies
    # Shape: [num_envs]
    sum_squared_forces = torch.sum(squared_thresholded_magnitudes, dim=1)

    return sum_squared_forces


def torque_limits(
    env: ManagerBasedRLEnv,
    soft_limit_factor: float = 0.9,  # Options: "lower_body", "upper_body", "whole_body"
    robot_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Reward for penalizing joint torques close to their limits.

    Penalizes joint torques that exceed a soft limit defined as a percentage of the maximum limit.

    Args:
        env: Environment instance.
        soft_limit_factor: Factor to define soft limit relative to the hard limit (0.0 to 1.0).
        robot_cfg: Configuration for the robot asset.

    Returns:
        Reward tensor.
    """
    # Create default robot_cfg if None is provided
    robot, robot_cfg = get_robot_cfg(env, robot_cfg)

    # Get joint torques
    joint_torques = robot.data.applied_torque

    # Get joint torque limits
    joint_torque_limits = robot.root_physx_view.get_dof_max_forces()

    # Extract specified joint torques and limits
    selected_joint_torques = joint_torques[:, robot_cfg.joint_ids]
    selected_joint_torque_limits = joint_torque_limits[robot_cfg.joint_ids]

    # Calculate soft torque limits
    soft_torque_limits = selected_joint_torque_limits * soft_limit_factor

    # Calculate how much the joint torques exceed the soft limits
    # Clip to ensure only violations are penalized
    excess_torque = (torch.abs(selected_joint_torques) - soft_torque_limits).clip(min=0.0)

    # Sum penalties across all joints
    total_penalty = torch.sum(excess_torque, dim=1)

    return total_penalty


def incoming_forces_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = None,
    threshold: float = 100.0,
) -> torch.Tensor:
    """Penalizes large internal forces with the l2 squared kernel.

    Penalizes joint torques that exceed a threshold.

    Args:
        env: Environment instance.
        robot_cfg: Robot entity with specified body names
        threshold: forces above this threshold get penalized with an l2 kernel

    Returns:
        Reward tensor.
    """
    # Create default robot_cfg if None is provided
    robot, robot_cfg = get_robot_cfg(env, robot_cfg)

    # Get joint torques
    incoming_wrench = robot.data.body_incoming_joint_wrench_b[:, robot_cfg.body_ids]
    forces = torch.linalg.vector_norm(incoming_wrench[..., :3], dim=2)

    over_limit = torch.clamp(forces - threshold, min=0)

    return torch.sum(torch.square(over_limit), dim=1)


def max_incoming_forces_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = None,
    threshold: float = 100.0,
) -> torch.Tensor:
    """Penalizes large internal forces with the l2 kernel.

    Penalizes joint torques that exceed a soft limit defined as a percentage of the maximum limit.

    Args:
        env: Environment instance.
        robot_cfg: Robot entity with specified body names
        threshold: forces above this threshold get penalized with an l2 kernel

    Returns:
        Reward tensor.
    """
    # Create default robot_cfg if None is provided
    robot, robot_cfg = get_robot_cfg(env, robot_cfg)

    # Get joint torques
    incoming_wrench = robot.data.body_incoming_joint_wrench_b[:, robot_cfg.body_ids]
    forces = torch.linalg.vector_norm(incoming_wrench[..., :3], dim=2)

    over_limit = torch.clamp(forces - threshold, min=0)

    return torch.square(torch.max(over_limit, dim=1)[0])


class action_rate_rate_l2(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_action_rate = torch.zeros_like(env.action_manager.action)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:  # noqa: ARG002
        """Penalize the rate of change of the rate of change of the actions using L2 squared kernel."""
        action_rate = env.action_manager.action - env.action_manager.prev_action
        action_rate_rate = torch.sum(torch.square(action_rate - self.prev_action_rate), dim=1)
        self.prev_action_rate = action_rate
        return action_rate_rate

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs)
        self.prev_action_rate[env_ids] = 0


class action_rate_rate_l2_if_actor_is_active(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.action_rate_rate_l2 = action_rate_rate_l2(cfg=cfg, env=env)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, rest_duration_s: float) -> torch.Tensor:
        """Penalize the rate of change of the rate of change of the actions using L2 squared kernel."""
        action_rate_rate = self.action_rate_rate_l2(env=env, asset_cfg=asset_cfg)
        env_in_rest_phase = env.episode_length_buf < int(rest_duration_s / env.step_dt)
        action_rate_rate[env_in_rest_phase] = 0
        return action_rate_rate

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.action_rate_rate_l2.reset(env_ids=env_ids)
