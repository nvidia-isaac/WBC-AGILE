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

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import LiftActionCfg


class LiftAction(ActionTerm):
    """
    Lift action to help a bipedal robot to stand up.

    Applies external forces to lift the robot up by a simple PD law on a target height
    that increases linearly over time. Also applies angular velocity damping to prevent
    spinning.

    Use a curriculum (e.g., `remove_harness` or `adaptive_lift_curriculum`) to reduce
    the forces over time as the robot learns to stand up on its own.
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
        self.damping_torques = cfg.damping_torques
        self._torque_limit = cfg.torque_limit
        # height sensor
        self._height_sensor: RayCaster = env.scene.sensors[cfg.height_sensor]
        self._lift_link_id, _ = self._asset.find_bodies(cfg.link_to_lift)
        self._is_disabled = False

        # Force scale for curriculum (1.0 = full force, 0.0 = disabled)
        self._force_scale = 1.0

        # Track max height achieved per environment during episode (for curriculum)
        self._max_heights = torch.zeros(env.num_envs, device=env.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.empty(0, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.empty(0, device=self.device)

    @property
    def force_scale(self) -> float:
        """Current force scale for logging/monitoring."""
        return self._force_scale

    @property
    def max_heights(self) -> torch.Tensor:
        """Max height achieved per environment during current episode.

        Used by adaptive_lift_curriculum to determine if robot successfully
        stood up at any point (even if it fell afterwards).
        """
        return self._max_heights

    def scale_forces(self, scale: float) -> None:
        """Scale all force and torque parameters by the given scale.

        Called by curriculum terms to adjust the lift assistance over training.

        Args:
            scale: Scale factor in [0, 1]. 0 = disabled, 1 = full force.
        """
        self._force_scale = scale
        self.stiffness_forces = self.cfg.stiffness_forces * self._force_scale
        self.damping_forces = self.cfg.damping_forces * self._force_scale
        self._force_limit = self.cfg.force_limit * self._force_scale
        self.damping_torques = self.cfg.damping_torques * self._force_scale
        self._torque_limit = self.cfg.torque_limit * self._force_scale
        self._is_disabled = self._force_scale <= 0

    def process_actions(self, actions: torch.Tensor) -> None:
        # store the raw actions
        self._raw_actions = actions

    def apply_actions(self) -> None:
        # Always compute and track height (even when disabled) for curriculum
        height = self._asset.data.root_pos_w[:, 2].unsqueeze(1) - self._height_sensor.data.ray_hits_w[..., 2]
        height = torch.mean(height, dim=-1)

        # Track max height achieved during episode
        self._max_heights = torch.maximum(self._max_heights, height)

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
        # calculate the height error
        height_error = target_height - height  # (N, 1)
        # apply the height error to the forces
        forces[:, 2] = self.stiffness_forces * height_error
        # limit the forces
        forces = torch.clamp(forces, 0.0, self._force_limit).unsqueeze(1)

        # Angular velocity damping (D term) - only on z-axis (yaw) in world frame
        # This prevents fast spinning while allowing roll/pitch for balance
        torques_w = torch.zeros_like(self._asset.data.root_ang_vel_w)
        if self.damping_torques > 0:
            # Get angular velocity in world frame and damp only z-component
            ang_vel_z = self._asset.data.root_ang_vel_w[:, 2]
            torques_w[:, 2] = -self.damping_torques * ang_vel_z
            # Clamp torques
            torques_w[:, 2] = torch.clamp(torques_w[:, 2], -self._torque_limit, self._torque_limit)

        # Apply forces and torques in world frame
        self._asset.set_external_force_and_torque(
            forces=forces, torques=torques_w.unsqueeze(1), body_ids=self._lift_link_id, is_global=True
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Reset max heights for environments that are resetting
        if env_ids is None:
            self._max_heights[:] = 0.0
        else:
            self._max_heights[env_ids] = 0.0
