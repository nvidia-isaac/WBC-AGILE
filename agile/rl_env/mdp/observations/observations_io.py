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

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

from agile.isaaclab_extras.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_joint_names,
    record_shape,
)
from agile.rl_env.mdp.commands import UniformVelocityBaseHeightCommand


@generic_io_descriptor(  # type: ignore[arg-type]
    units="unit",
    observation_type="Environment",
    on_inspect=[record_shape, record_dtype],
)
def is_env_inactive(env: ManagerBasedRLEnv, rest_duration_s: float) -> torch.Tensor:
    """Check if the environment is in the rest phase."""
    # Note: episode_length_buf is initialized after managers, so we check for its existence.
    # This allows the observation manager to be created before the environment is fully initialized.
    if hasattr(env, "episode_length_buf"):
        return (env.episode_length_buf < int(rest_duration_s / env.step_dt)).float().unsqueeze(1)
    else:
        return torch.ones(env.num_envs, 1, device=env.device)


@generic_io_descriptor(units="m", observation_type="Sensor", on_inspect=[record_shape, record_dtype])  # type: ignore[arg-type]
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
    # height scan: height = sensor_height - hit_point_z - offset
    out = torch.cat(
        (
            (sensor_left.data.pos_w[:, 2].unsqueeze(1) - sensor_left.data.ray_hits_w[..., 2] - offset).unsqueeze(1),
            (sensor_right.data.pos_w[:, 2].unsqueeze(1) - sensor_right.data.ray_hits_w[..., 2] - offset).unsqueeze(1),
        ),
        dim=1,
    )
    return out.reshape(out.shape[0], -1)


@generic_io_descriptor(units="m", observation_type="Command", on_inspect=[record_shape, record_dtype])  # type: ignore[arg-type]
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
    base_height = robot.data.root_pos_w[:, 2] - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    return base_height.unsqueeze(1)


"""
Commands.
"""


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad/s^2"
)
def joint_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Extract the joint accelerations of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their
    accelerations returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


@generic_io_descriptor(observation_type="Sensor", on_inspect=[record_body_names, record_dtype, record_shape], units="N")
def contact_force_norm(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Extract the norms of the contact forces of the asset."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # Get contact forces for these bodies.
    net_contact_forces = contact_sensor.data.net_forces_w

    # Get forces for the specified bodies
    # Shape: [num_envs, num_bodies]
    body_forces = net_contact_forces[:, sensor_cfg.body_ids].norm(dim=2)

    return body_forces
