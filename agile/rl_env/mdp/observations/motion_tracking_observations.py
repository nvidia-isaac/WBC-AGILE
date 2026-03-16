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

from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_dtype, record_shape
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from agile.rl_env.mdp.commands.motion_tracking_commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["robot_body_pos_b", "robot_body_ori_b"]


@generic_io_descriptor(observation_type="MotionTracking", on_inspect=[record_shape, record_dtype])
def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot tracked body positions in anchor body frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


@generic_io_descriptor(observation_type="MotionTracking", on_inspect=[record_shape, record_dtype])
def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot tracked body orientations in anchor body frame (first 2 columns of rotation matrix)."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)
