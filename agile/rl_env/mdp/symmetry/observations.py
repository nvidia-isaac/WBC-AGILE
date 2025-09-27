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

"""Left-right symmetry augmentation for observations."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def mirror_velocity_commands(actions: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:  # noqa: ARG001
    """Mirror the velocity commands.

    actions has shape (..., 3) (xz, omega_z)
    """
    mirrored_actions = actions.clone()
    mirrored_actions[..., 1] = -actions[..., 1]
    mirrored_actions[..., 2] = -actions[..., 2]

    return mirrored_actions


def lr_mirror_projected_gravity(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:  # noqa: ARG001
    """Mirror the projected gravity.

    obs has shape (..., 3)
    """
    mirrored_obs = obs.clone()
    mirrored_obs[..., 1] = -obs[..., 1]
    return mirrored_obs


def lr_mirror_base_lin_vel(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:  # noqa: ARG001
    """Left-right mirroring of the base linear velocity.

    obs has shape (..., 3)
    """
    mirrored_obs = obs.clone()
    mirrored_obs[..., 1] = -obs[..., 1]
    return mirrored_obs


def lr_mirror_base_ang_vel(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:  # noqa: ARG001
    """Left-right mirroring of the base angular velocity.

    obs has shape (..., 3)
    """
    mirrored_obs = obs.clone()
    mirrored_obs[..., 0] = -obs[..., 0]
    mirrored_obs[..., 2] = -obs[..., 2]
    return mirrored_obs


def mirror_height_scan_left_right(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Left-right mirroring of the height scan.

    obs has shape (..., nx * ny)
    """
    nx, ny = get_height_scan_shape(env, "height_scanner")

    mirrored_obs = obs.clone()
    mirrored_obs = mirrored_obs.view(-1, ny, nx).flip(dims=[1]).view(-1, nx * ny)
    return mirrored_obs


def mirror_height_scan_feet_left_right(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Left-right mirroring of the height scan.

    obs has shape (..., 2 * nx * ny) where the first half is left foot, second half is right foot
    """
    nx, ny = get_height_scan_shape(env, "height_scanner_left_foot")
    scan_size = nx * ny

    # split the observations into left and right
    # First half is left foot, second half is right foot
    left_obs = obs[..., :scan_size]
    right_obs = obs[..., scan_size : 2 * scan_size]

    # mirror the left and right observations
    mirrored_left_obs = left_obs.clone()
    mirrored_right_obs = right_obs.clone()
    mirrored_left_obs = mirrored_left_obs.view(-1, ny, nx).flip(dims=[1]).view(-1, scan_size)
    mirrored_right_obs = mirrored_right_obs.view(-1, ny, nx).flip(dims=[1]).view(-1, scan_size)

    # concatenate the mirrored observations in reverse order (swap left and right)
    mirrored_obs = torch.cat((mirrored_right_obs, mirrored_left_obs), dim=-1)

    return mirrored_obs


def mirror_gait_cycle_commands(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:  # noqa: ARG001
    """Mirror the gait cycle commands.

    obs has shape (..., 2)
    """

    mirrored_obs = obs.clone()
    mirrored_obs[..., 0] = obs[..., 1]
    mirrored_obs[..., 1] = obs[..., 0]

    return mirrored_obs


@lru_cache(maxsize=10)
def get_height_scan_shape(env: ManagerBasedRLEnv, scanner_name: str) -> tuple[int, int]:
    nx = 1 + int(
        np.round(
            env.unwrapped.scene.sensors[scanner_name].cfg.pattern_cfg.size[0]
            / env.unwrapped.scene.sensors[scanner_name].cfg.pattern_cfg.resolution
        )
    )
    ny = 1 + int(
        np.round(
            env.unwrapped.scene.sensors[scanner_name].cfg.pattern_cfg.size[1]
            / env.unwrapped.scene.sensors[scanner_name].cfg.pattern_cfg.resolution
        )
    )

    return nx, ny


##
# Privileged observations
##


def mirror_material(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:  # noqa: ARG001
    """Mirror the base height.

    obs has shape (..., 2, 3) with 2 bodies
    """
    mirrored_obs = obs.clone()
    mirrored_obs[..., 0, :] = obs[..., 0, :]
    mirrored_obs[..., 1, :] = obs[..., 1, :]

    return mirrored_obs


def mirror_external_force_torque(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:  # noqa: ARG001
    """Mirror the external force torque.

    obs has shape (..., 6) 3d force and torque
    """
    mirrored_obs = obs.clone()
    # mirror the force
    mirrored_obs[..., 1] = -obs[..., 1]

    # mirror the torque
    mirrored_obs[..., 3] = -obs[..., 3]
    mirrored_obs[..., 5] = -obs[..., 5]

    return mirrored_obs


def mirror_base_com(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:  # noqa: ARG001
    """Mirror the base com.

    obs has shape (..., 3)
    """
    mirrored_obs = obs.clone()
    mirrored_obs[..., 1] = -obs[..., 1]
    return mirrored_obs
