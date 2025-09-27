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
    mirror_material,
    mirror_velocity_commands,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lr_mirror_T1(
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
        mirrored_actions = mirror_actions_T1(actions, env)
        augmented_actions = torch.cat([actions, mirrored_actions], dim=0)
    else:
        augmented_actions = None

    if obs is not None:
        mirrored_obs = TensorDict(
            {name: OBS_TO_MIRROR[name](obs[name], env) for name in obs.keys()},
            batch_size=obs.batch_size,
        )
        augmented_obs = torch.cat([obs, mirrored_obs], dim=0)

    else:
        augmented_obs = None

    return augmented_obs, augmented_actions


def mirror_actions_T1(actions: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Left-right mirroring of the actions. Can be a subset of the joints as defined in the action manager."""

    mirrored_indices, neg_indices = resolve_joint_names_t1(
        tuple(env.unwrapped.action_manager._terms["joint_pos"]._joint_names)
    )

    mirrored_actions = actions.clone()
    mirrored_actions[..., mirrored_indices] = actions
    mirrored_actions[..., neg_indices] *= -1

    return mirrored_actions


def mirror_joints_T1(actions: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Left-right mirroring of all the joints of the booster T1 robot."""

    mirrored_indices, neg_indices = resolve_joint_names_t1(
        tuple(env.unwrapped.scene.articulations["robot"].joint_names)
    )

    mirrored_actions = actions.clone()
    mirrored_actions[..., mirrored_indices] = actions
    mirrored_actions[..., neg_indices] *= -1

    return mirrored_actions


def mirror_bodies_T1(bodies: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Left-right mirroring of **all** the bodies of the booster T1 robot."""

    mirrored_indices = resolve_body_names_t1(tuple(env.unwrapped.scene.articulations["robot"].body_names))

    mirrored_bodies = bodies.clone()
    mirrored_bodies[..., mirrored_indices] = bodies

    return mirrored_bodies


@lru_cache(maxsize=10)
def resolve_joint_names_t1(action_joint_names: tuple[str, ...]) -> tuple[list[int], list[int]]:
    """Resolve the joint names to indices.

    Args:
        action_joint_names: The joint names of the action.

    Returns:
        The indices of the mirrored joints and the indices of the joints that need to be negated on mirror.
    """

    # Mirrored joint names.
    mirrored_indices = []
    for source_joint_name in action_joint_names:
        if "Left" in source_joint_name:
            mirrored_joint_name = source_joint_name.replace("Left", "Right")
        elif "Right" in source_joint_name:
            mirrored_joint_name = source_joint_name.replace("Right", "Left")
        else:
            mirrored_joint_name = source_joint_name

        if mirrored_joint_name not in action_joint_names:
            raise ValueError(f"Mirrored joint name {mirrored_joint_name} not found in action joint names")

        mirrored_indices.append(action_joint_names.index(mirrored_joint_name))

    # Joints that need to be negated on mirror.
    neg_indices = []
    neg_joint_indicators = ["Roll", "Yaw", "Waist", "yaw"]
    neg_joint_exclude: list[str] = []

    for joint_name in action_joint_names:
        if any(indicator in joint_name for indicator in neg_joint_indicators) and not any(
            exclude in joint_name for exclude in neg_joint_exclude
        ):
            neg_indices.append(action_joint_names.index(joint_name))

    return mirrored_indices, neg_indices


@lru_cache(maxsize=10)
def resolve_body_names_t1(body_names: tuple[str, ...]) -> list[int]:
    """Resolve the joint names to indices.

    Args:
        action_body_names: The body names of the action.

    Returns:
        The indices of the mirrored bodies.
    """

    # Mirrored joint names.
    mirrored_indices = []
    for source_body_name in body_names:
        if "Left" in source_body_name:
            mirrored_body_name = source_body_name.replace("Left", "Right")
        elif "left" in source_body_name:
            mirrored_body_name = source_body_name.replace("left", "right")
        elif "AL" in source_body_name:
            mirrored_body_name = source_body_name.replace("AL", "AR")
        elif "Right" in source_body_name:
            mirrored_body_name = source_body_name.replace("Right", "Left")
        elif "right" in source_body_name:
            mirrored_body_name = source_body_name.replace("right", "left")
        elif "AR" in source_body_name:
            mirrored_body_name = source_body_name.replace("AR", "AL")
        else:
            mirrored_body_name = source_body_name

        if mirrored_body_name not in body_names:
            raise ValueError(f"Mirrored body name {mirrored_body_name} not found in action body names")

        mirrored_indices.append(body_names.index(mirrored_body_name))

    return mirrored_indices


def mirror_actuator_gains(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Mirror the actuator gains.

    obs has shape (..., num_joints, 2)
    """
    mirrored_indices, _ = resolve_joint_names_t1(tuple(env.unwrapped.scene.articulations["robot"].joint_names))
    mirrored_obs = obs.clone()
    mirrored_obs[..., mirrored_indices, :] = obs

    return mirrored_obs


def mirror_joint_parameters(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Mirror the joint parameters.

    obs has shape (..., num_joints, N) where N is the number of parameters per joint.
    """
    mirrored_indices, _ = resolve_joint_names_t1(tuple(env.unwrapped.scene.articulations["robot"].joint_names))
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
    "joint_pos": mirror_joints_T1,
    "joint_vel": mirror_joints_T1,
    "actions": mirror_actions_T1,
    "controlled_joint_pos": mirror_actions_T1,
    "controlled_joint_vel": mirror_actions_T1,
    "velocity_commands": mirror_velocity_commands,
    "velocity_height_commands": mirror_velocity_commands,
    "height_commands": identity,
    "gait_cycle_commands": mirror_gait_cycle_commands,
    "height_scan": mirror_height_scan_left_right,
    "height_scan_feet": mirror_height_scan_feet_left_right,
    "base_height": identity,
    "material": mirror_material,
    "actuator_gains": mirror_actuator_gains,
    "joint_parameters": mirror_joint_parameters,
    "joint_friction": mirror_joint_parameters,
    "joint_armature": mirror_joint_parameters,
    "external_force_torque": mirror_external_force_torque,
    "base_com": mirror_base_com,
    "base_mass": identity,
    "is_env_inactive": identity,
    "action_smoothing": mirror_actions_T1,
    "contact_forces": mirror_bodies_T1,
}
"""Hashmap of observation names to functions to mirror the observations."""
