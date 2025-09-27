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

    from .actions_cfg import HarnessActionCfg


class HarnessAction(ActionTerm):
    """
    Harness action to help a bipedal robot walk.

    We are using the sensor base class but this is not a sensor per seay.
    We measure from the height sensor the height and
    Orientatation from the articaulation root.
    We apply corrictive force torque to the root of the robot
    to keep the robot upright and at the desired height.
    We then use a curiculum to reduce the forces applied to the robot.
    This way we teach the robot to stand up and walk without falling over.
    """

    cfg: HarnessActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: HarnessActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)
        self.stiffness_torques = cfg.stiffness_torques
        self.damping_torques = cfg.damping_torques
        self.stiffness_forces = cfg.stiffness_forces
        self.damping_forces = cfg.damping_forces
        self._force_limit = cfg.force_limit
        self._torque_limit = cfg.torque_limit
        # quat
        self._target_quat = torch.tensor(cfg.target_quat, dtype=torch.float32, device=self.device)
        # height sensor
        self._height_sensor: RayCaster = env.scene.sensors[cfg.height_sensor]

        # Command integration for height
        if cfg.command_name is not None:
            self._command_term = env.command_manager.get_term(cfg.command_name)
        else:
            self._command_term = None

        self._root_id, _ = self._asset.find_bodies(cfg.root_name)
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
        self.stiffness_torques = self.cfg.stiffness_torques * scale
        self.damping_torques = self.cfg.damping_torques * scale
        self.stiffness_forces = self.cfg.stiffness_forces * scale
        self.damping_forces = self.cfg.damping_forces * scale
        self._force_limit = self.cfg.force_limit * scale
        self._torque_limit = self.cfg.torque_limit * scale
        self._is_disabled = scale <= 0

    def process_actions(self, actions: torch.Tensor) -> None:
        # store the raw actions
        self._raw_actions = actions

    def apply_actions(self) -> None:
        if self._is_disabled:
            return

        # Compute orientation stabilization torque (roll/pitch only)
        torque_stabilization = self._compute_orientation_torque()

        # Compute height control force (vertical)
        force_height = self._compute_height_force()

        # Apply combined wrench to root
        self._asset.set_external_force_and_torque(
            forces=force_height, torques=torque_stabilization, body_ids=self._root_id
        )

    def _compute_orientation_torque(self) -> torch.Tensor:
        """Compute torque for roll/pitch stabilization.

        Returns:
            Torque tensor [N, 1, 3] with z-component = 0
        """
        target_quat = self._target_quat.repeat(self.num_envs, 1)
        current_quat = self._asset.data.root_quat_w
        q_err = math_utils.quat_mul(math_utils.quat_conjugate(current_quat), target_quat)
        error = 2.0 * torch.sign(q_err[:, 0:1]) * q_err[:, 1:]

        # PD control
        torque = self.stiffness_torques * error - self.damping_torques * self._asset.data.root_ang_vel_b
        torque[:, 2] = 0.0  # No yaw assistance from stabilization
        torque = torch.clamp(torque, -self._torque_limit, self._torque_limit)

        return torque.view(self.num_envs, 1, 3)

    def _compute_height_force(self) -> torch.Tensor:
        """Compute vertical force for height control.

        Returns:
            Force tensor [N, 1, 3] with only z-component non-zero
        """
        # Measure current height
        height_rays = self._height_sensor.data.ray_hits_w[..., 2]
        root_z = self._asset.data.root_pos_w[:, 2].unsqueeze(1)
        current_height = -torch.mean(height_rays - root_z, dim=-1).unsqueeze(1)

        # Get target height from command or config
        if self._command_term is not None:
            target_height = self._command_term.command[:, -1].unsqueeze(1)
        else:
            target_height = self.cfg.target_height

        # Height error and velocity
        height_error = target_height - current_height
        z_velocity = self._asset.data.root_lin_vel_b[:, 2].unsqueeze(1)

        # PD control
        force_z = self.stiffness_forces * height_error - self.damping_forces * z_velocity
        force_z = torch.clamp(force_z, -self._force_limit, self._force_limit)

        # Return as [N, 1, 3]
        forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        forces[:, 0, 2] = force_z.squeeze(1)
        return forces

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass  # No reset needed for this action term
