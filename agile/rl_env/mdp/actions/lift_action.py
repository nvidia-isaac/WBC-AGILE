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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import LiftActionCfg


class LiftAction(ActionTerm):
    """
    Lift action to help a bipedal robot to stand up.

    Applies external forces to lift the robot up by a simple pd law on a target height
    that increases linearly over time.
    We then use a curiculum to reduce the forces applied to the robot.
    This way we teach the robot to stand up and walk without falling over.
    """

    cfg: LiftActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: LiftActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)
        self.stiffness_forces = cfg.stiffness_forces
        self.damping_forces = cfg.damping_forces
        self._force_limit = cfg.force_limit
        # height sensor
        self._height_sensor: RayCaster = env.scene.sensors[cfg.height_sensor]
        self._lift_link_id, _ = self._asset.find_bodies(cfg.link_to_lift)
        self._is_disabled = False

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.empty(0, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.empty(0, device=self.device)

    def scale_forces(self, scale: float) -> None:
        """Scale the force and torque limits, useful for curriculum."""
        self.stiffness_forces = self.cfg.stiffness_forces * scale
        self.damping_forces = self.cfg.damping_forces * scale
        self._force_limit = self.cfg.force_limit * scale
        self._is_disabled = scale <= 0

    def process_actions(self, actions: torch.Tensor) -> None:
        # store the raw actions
        self._raw_actions = actions

    def apply_actions(self) -> None:
        if self._is_disabled:
            return

        # find current desired height above ground
        time_passed = self._env.episode_length_buf * self._env.step_dt
        ratio = torch.clamp(
            (time_passed - self.cfg.start_lifting_time_s) / self.cfg.lifting_duration_s, min=0.0, max=1.0
        )
        target_height = ratio * self.cfg.target_height

        # find the error in local frame of root
        forces = torch.zeros_like(self._asset.data.root_lin_vel_b)

        height = self._asset.data.root_pos_w[:, 2].unsqueeze(1) - self._height_sensor.data.ray_hits_w[..., 2]
        height = torch.mean(height, dim=-1)
        # calculate the height error
        height_error = target_height - height  # (N, 1)
        # apply the height error to the forces
        forces[:, 2] = self.stiffness_forces * height_error
        # limit the forces
        forces = torch.clamp(forces, 0.0, self._force_limit).unsqueeze(1)

        # rotate forces to body frame
        link_quat = self._asset.data.body_quat_w[:, self._lift_link_id].squeeze(1)

        forces_b = math_utils.quat_apply_inverse(link_quat, forces)

        self._asset.set_external_force_and_torque(
            forces=forces_b, torques=torch.zeros_like(forces_b), body_ids=self._lift_link_id
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass  # No reset needed for this action term
