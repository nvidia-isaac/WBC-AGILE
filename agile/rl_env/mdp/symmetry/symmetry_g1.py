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

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from tensordict.tensordict import TensorDict

from .observations import (
    lr_mirror_base_ang_vel,
    lr_mirror_base_ang_vel_z,
    lr_mirror_base_lin_vel,
    lr_mirror_projected_gravity,
    mirror_base_com,
    mirror_external_force_torque,
    mirror_feet_height,
    mirror_feet_roll_pitch,
    mirror_feet_yaw_vs_body,
    mirror_flattened_observation_group,
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
            {name: _mirror_observation(name, obs[name], env, obs_type) for name in obs.keys()},
            batch_size=obs.batch_size,
        )
        augmented_obs = torch.cat([obs, mirrored_obs], dim=0)
    else:
        augmented_obs = None

    return augmented_obs, augmented_actions


def _mirror_observation(name: str, obs: torch.Tensor, env: ManagerBasedRLEnv, obs_type: str = "policy") -> torch.Tensor:
    if name in OBS_TO_MIRROR:
        return OBS_TO_MIRROR[name](obs, env)

    if hasattr(env.unwrapped, "observation_manager") and name in env.unwrapped.observation_manager.active_terms:
        return mirror_flattened_observation_group(obs, env, name, OBS_TO_MIRROR)

    return mirror_flattened_observation_group(obs, env, obs_type, OBS_TO_MIRROR)


def mirror_actions_G1(
    actions: torch.Tensor, env: ManagerBasedRLEnv, action_term_name: str = "joint_pos"
) -> torch.Tensor:
    """Left-right mirroring of the actions. Handles single or multiple action terms.

    When the action tensor contains multiple concatenated action terms, splits by
    action dimensions, mirrors each term independently, and concatenates back.
    """
    action_manager = env.unwrapped.action_manager
    term_names = list(action_manager._terms.keys())

    # Single action term or explicitly named
    if len(term_names) == 1 or actions.shape[-1] == action_manager._terms[action_term_name].action_dim:
        mirrored_indices, neg_indices = resolve_joint_names_g1(
            tuple(action_manager._terms[action_term_name]._joint_names)
        )
        mirrored_actions = actions.clone()
        mirrored_actions[..., mirrored_indices] = actions
        mirrored_actions[..., neg_indices] *= -1
        return mirrored_actions

    # Multiple action terms: split, mirror each, concatenate
    mirrored_parts = []
    offset = 0
    for name in term_names:
        term = action_manager._terms[name]
        dim = term.action_dim
        part = actions[..., offset : offset + dim]

        if hasattr(term, "_joint_names"):
            m_indices, n_indices = resolve_joint_names_g1(tuple(term._joint_names))
            mirrored = part.clone()
            mirrored[..., m_indices] = part
            mirrored[..., n_indices] *= -1
        else:
            mirrored = part.clone()

        mirrored_parts.append(mirrored)
        offset += dim

    return torch.cat(mirrored_parts, dim=-1)


def mirror_joints_G1(actions: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Left-right mirroring of all the joints of the unitree G1 robot."""

    mirrored_indices, neg_indices = resolve_joint_names_g1(
        tuple(env.unwrapped.scene.articulations["robot"].joint_names)
    )

    mirrored_actions = actions.clone()
    mirrored_actions[..., mirrored_indices] = actions
    mirrored_actions[..., neg_indices] *= -1

    return mirrored_actions


def mirror_bodies_G1(bodies: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Left-right mirroring of **all** the bodies of the unitree G1 robot."""

    mirrored_indices = resolve_body_names_g1(tuple(env.unwrapped.scene.articulations["robot"].body_names))

    mirrored_bodies = bodies.clone()
    mirrored_bodies[..., mirrored_indices] = bodies

    return mirrored_bodies


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


@lru_cache(maxsize=10)
def resolve_body_names_g1(body_names: tuple[str, ...]) -> list[int]:
    """Resolve the body names to indices for left-right mirroring.

    Args:
        body_names: The body names of the robot.

    Returns:
        The indices of the mirrored bodies.
    """

    mirrored_indices = []
    for source_body_name in body_names:
        if "left" in source_body_name:
            mirrored_body_name = source_body_name.replace("left", "right")
        elif "right" in source_body_name:
            mirrored_body_name = source_body_name.replace("right", "left")
        else:
            mirrored_body_name = source_body_name

        if mirrored_body_name not in body_names:
            raise ValueError(f"Mirrored body name {mirrored_body_name} not found in body names")

        mirrored_indices.append(body_names.index(mirrored_body_name))

    return mirrored_indices


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
    "base_ang_vel_z_world": lr_mirror_base_ang_vel_z,
    "torso_projected_gravity": lr_mirror_projected_gravity,
    "joint_pos": mirror_joints_G1,
    "joint_vel": mirror_joints_G1,
    "actions": mirror_actions_G1,
    "joint_pos_target": mirror_actions_G1,
    "controlled_joint_pos": mirror_actions_G1,
    "controlled_joint_vel": mirror_actions_G1,
    "velocity_commands": mirror_velocity_commands,
    "velocity_height_commands": mirror_velocity_commands,
    "height_command": identity,
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
    "torso_mass": identity,
    "joint_pos_upper": mirror_joints_G1,
    "joint_pos_lower": mirror_joints_G1,
    "joint_vel_upper": mirror_joints_G1,
    "joint_vel_lower": mirror_joints_G1,
    "last_actions_upper": mirror_joints_G1,
    "last_actions_lower": mirror_joints_G1,
    "contact_forces": mirror_bodies_G1,
    "feet_height": mirror_feet_height,
    "feet_roll_pitch": mirror_feet_roll_pitch,
    "feet_yaw_vs_pelvis": mirror_feet_yaw_vs_body,
}
"""Mapping of observation names to functions to mirror the observations."""
