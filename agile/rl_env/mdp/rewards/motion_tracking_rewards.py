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


from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_error_magnitude

from agile.rl_env.mdp.commands.motion_tracking_commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "wb_motion_relative_body_position_error_exp",
    "wb_motion_relative_body_orientation_error_exp",
    "wb_motion_global_body_linear_velocity_error_exp",
    "wb_motion_global_body_angular_velocity_error_exp",
]


def _get_body_indices(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def wb_motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = _get_body_indices(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indices] - command.robot_body_pos_w[:, body_indices]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def wb_motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = _get_body_indices(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indices], command.robot_body_quat_w[:, body_indices])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def wb_motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = _get_body_indices(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indices] - command.robot_body_lin_vel_w[:, body_indices]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def wb_motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = _get_body_indices(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indices] - command.robot_body_ang_vel_w[:, body_indices]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)
