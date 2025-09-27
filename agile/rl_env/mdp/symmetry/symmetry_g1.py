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

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from tensordict.tensordict import TensorDict

from .observations import (
    lr_mirror_base_ang_vel,
    lr_mirror_base_lin_vel,
    lr_mirror_projected_gravity,
    mirror_base_com,
    mirror_external_force_torque,
    mirror_gait_cycle_commands,
    mirror_height_scan_feet_left_right,
    mirror_height_scan_left_right,
    mirror_velocity_commands,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lr_mirror_G1(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",  # noqa: ARG001
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Left-right symmetry augmentation for the G1 robot.

    Args:
        env: The environment.
        obs: The observation TensorDict.
        actions: The action tensor.
        obs_type: The observation type.

    Returns:
        A tuple of the augmented observations and actions.
    """

    if actions is not None:
        mirrored_actions = mirror_actions_G1(actions, env)
        augmented_actions = torch.cat([actions, mirrored_actions], dim=0)
    else:
        augmented_actions = None

    # {
    #     name: OBS_TO_MIRROR[name](obs[name], env)
    #     for name, cfg in zip(obs.keys(), env.unwrapped.observation_manager._group_obs_term_cfgs[obs_type])
    # },

    if obs is not None:
        mirrored_obs = TensorDict(
            {name: OBS_TO_MIRROR[name](obs[name], env) for name in obs.keys()},
            batch_size=obs.batch_size,
        )
        augmented_obs = torch.cat([obs, mirrored_obs], dim=0)
    else:
        augmented_obs = None

    return augmented_obs, augmented_actions


def mirror_actions_G1(
    actions: torch.Tensor, env: ManagerBasedRLEnv, action_term_name: str = "joint_pos"
) -> torch.Tensor:
    """Left-right mirroring of the actions. Can be a subset of the joints as defined in the action manager."""

    mirrored_indices, neg_indices = resolve_joint_names_g1(
        tuple(env.unwrapped.action_manager._terms[action_term_name]._joint_names)
    )

    mirrored_actions = actions.clone()
    mirrored_actions[..., mirrored_indices] = actions
    mirrored_actions[..., neg_indices] *= -1

    return mirrored_actions


def mirror_joints_G1(actions: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Left-right mirroring of all the joints of the unitree G1 robot."""

    mirrored_indices, neg_indices = resolve_joint_names_g1(
        tuple(env.unwrapped.scene.articulations["robot"].joint_names)
    )

    mirrored_actions = actions.clone()
    mirrored_actions[..., mirrored_indices] = actions
    mirrored_actions[..., neg_indices] *= -1

    return mirrored_actions


@lru_cache(maxsize=10)
def resolve_joint_names_g1(action_joint_names: tuple[str, ...]) -> tuple[list[int], list[int]]:
    """Resolve the joint names to indices.

    Args:
        action_joint_names: The joint names of the action.

    Returns:
        The indices of the mirrored joints and the indices of the joints that need to be negated on mirror.
    """

    # Mirrored joint names.
    mirrored_indices = []
    for source_joint_name in action_joint_names:
        if "left" in source_joint_name:
            mirrored_joint_name = source_joint_name.replace("left", "right")
        elif "right" in source_joint_name:
            mirrored_joint_name = source_joint_name.replace("right", "left")
        else:
            mirrored_joint_name = source_joint_name

        if mirrored_joint_name not in action_joint_names:
            raise ValueError(f"Mirrored joint name {mirrored_joint_name} not found in action joint names")

        mirrored_indices.append(action_joint_names.index(mirrored_joint_name))

    # Joints that need to be negated on mirror.
    neg_indices = []
    neg_joint_indicators = ["roll", "yaw", "hand"]
    neg_joint_exclude = ["thumb_0"]

    for joint_name in action_joint_names:
        if any(indicator in joint_name for indicator in neg_joint_indicators) and not any(
            exclude in joint_name for exclude in neg_joint_exclude
        ):
            neg_indices.append(action_joint_names.index(joint_name))

    return mirrored_indices, neg_indices


def mirror_actuator_gains(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Mirror the actuator gains.

    obs has shape (..., num_joints, 2)
    """
    mirrored_indices, _ = resolve_joint_names_g1(tuple(env.unwrapped.scene.articulations["robot"].joint_names))
    mirrored_obs = obs.clone()
    mirrored_obs[..., mirrored_indices, :] = obs

    return mirrored_obs


def mirror_joint_parameters(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Mirror the joint parameters.

    obs has shape (..., num_joints, N) where N is the number of parameters per joint.
    """
    mirrored_indices, _ = resolve_joint_names_g1(tuple(env.unwrapped.scene.articulations["robot"].joint_names))
    mirrored_obs = obs.clone()
    mirrored_obs[..., mirrored_indices, :] = obs
    return mirrored_obs


def identity(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:  # noqa: ARG001
    """Identity function."""
    return obs


OBS_TO_MIRROR: dict[str, Callable] = {
    "projected_gravity": lr_mirror_projected_gravity,
    "base_lin_vel": lr_mirror_base_lin_vel,
    "base_ang_vel": lr_mirror_base_ang_vel,
    "joint_pos": mirror_joints_G1,
    "joint_vel": mirror_joints_G1,
    "actions": mirror_actions_G1,
    "controlled_joint_pos": mirror_actions_G1,
    "controlled_joint_vel": mirror_actions_G1,
    "velocity_commands": mirror_velocity_commands,
    "velocity_height_commands": mirror_velocity_commands,
    "height_commands": identity,
    "gait_cycle_commands": mirror_gait_cycle_commands,
    "height_scan": mirror_height_scan_left_right,
    "height_scan_feet": mirror_height_scan_feet_left_right,
    "base_height": identity,
    "external_force_torque": mirror_external_force_torque,
    "base_com": mirror_base_com,
    "actuator_gains": mirror_actuator_gains,
    "joint_parameters": mirror_joint_parameters,
    "base_mass": identity,
    "joint_pos_upper": mirror_joints_G1,
    "joint_pos_lower": mirror_joints_G1,
    "joint_vel_upper": mirror_joints_G1,
    "joint_vel_lower": mirror_joints_G1,
    "last_actions_upper": mirror_joints_G1,
    "last_actions_lower": mirror_joints_G1,
}
"""Mapping of observation names to functions to mirror the observations."""
