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

"""Exponential Moving Average (EMA) velocity profile implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import VelocityProfileBase

if TYPE_CHECKING:
    from .configs import EMAVelocityProfileCfg


class EMAVelocityProfile(VelocityProfileBase):
    """Exponential Moving Average (EMA) velocity profile.

    This profile implements smooth position transitions using the EMA formula:
    pos = scale * target + (1 - scale) * current

    This creates an exponential approach to the target position, where the
    velocity decreases exponentially as the position approaches the target.
    """

    cfg: EMAVelocityProfileCfg
    """Configuration for EMA velocity profile."""

    def _initialize_states(self) -> None:
        """Initialize EMA profile specific states."""
        # Velocity scale (EMA coefficient) for each joint
        self._velocity_scale = torch.zeros((self._num_envs, self._num_joints), device=self._device)

        # Initial position when trajectory starts (for progress tracking)
        self._initial_position: torch.Tensor = torch.zeros_like(self._current_position)

        # Store previous position for velocity calculation
        self._previous_position: torch.Tensor = torch.zeros_like(self._current_position)

        # Estimated velocity (computed from position changes)
        self._current_velocity: torch.Tensor = torch.zeros_like(self._current_position)

    def initialize_state(self, position: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Initialize profile state from the current joint position without sampling."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self._current_position[env_ids] = position
        self._previous_position[env_ids] = position
        self._initial_position[env_ids] = position
        self._target_position[env_ids] = position
        self._current_velocity[env_ids] = 0.0
        self._velocity_scale[env_ids] = 0.0
        self._is_active[env_ids] = False

    def redirect_target(self, target_pos: torch.Tensor, env_mask: torch.Tensor) -> None:
        """Redirect masked trajectories without resetting their EMA state.

        Args:
            target_pos: Full ``[num_envs, num_joints]`` target tensor.
            env_mask: Boolean ``[num_envs]`` tensor selecting trajectories to redirect.
        """
        target_pos = self._clamp_to_limits(target_pos)
        self._target_position.copy_(torch.where(env_mask.unsqueeze(-1), target_pos, self._target_position))
        self._is_active.logical_or_(env_mask)

    def _clamp_ema_targets(self, target_pos: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """Clamp selected targets without synchronizing device state to the host."""
        if self.cfg.enable_position_limits and self._joint_limits is not None:
            limits = self._joint_limits[env_ids]
            return torch.clamp(target_pos, limits[..., 0], limits[..., 1])
        return target_pos

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset profile state for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        # Reset states for specified environments
        self._is_active[env_ids] = False
        self._current_position[env_ids] = 0.0
        self._target_position[env_ids] = 0.0
        self._initial_position[env_ids] = 0.0
        self._previous_position[env_ids] = 0.0
        self._current_velocity[env_ids] = 0.0
        self._velocity_scale[env_ids] = 0.0

    def set_target(
        self,
        current_pos: torch.Tensor,
        target_pos: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Set new target and initialize trajectory."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        # Validate and potentially clamp target positions
        target_pos = self._clamp_ema_targets(target_pos, env_ids)

        # Update positions
        self._current_position[env_ids] = current_pos
        self._target_position[env_ids] = target_pos
        self._initial_position[env_ids] = current_pos
        self._previous_position[env_ids] = current_pos

        # Sample velocity scales
        self._sample_velocity_scales(env_ids)

        # Mark trajectories as active
        self._is_active[env_ids] = True

        # Reset velocity
        self._current_velocity[env_ids] = 0.0

    def _sample_velocity_scales(self, env_ids: torch.Tensor) -> None:
        """Sample random EMA coefficients for specified environments.

        Args:
            env_ids: Environment indices to update.
        """
        num_envs_to_update = len(env_ids)

        if self.cfg.synchronize_joints:
            # All joints in an environment use the same coefficient
            scales = torch.rand(num_envs_to_update, 1, device=self._device)
            scales = scales * (self.cfg.ema_coefficient_range[1] - self.cfg.ema_coefficient_range[0])
            scales += self.cfg.ema_coefficient_range[0]
            # Broadcast to all joints
            self._velocity_scale[env_ids] = scales.expand(-1, self._num_joints)
        else:
            # Each joint gets its own coefficient
            scales = torch.rand(num_envs_to_update, self._num_joints, device=self._device)
            scales = scales * (self.cfg.ema_coefficient_range[1] - self.cfg.ema_coefficient_range[0])
            scales += self.cfg.ema_coefficient_range[0]
            self._velocity_scale[env_ids] = scales

    def compute_next_position(self, dt: float) -> torch.Tensor:
        """Compute next position using exponential moving average."""
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        self._previous_position.copy_(self._current_position)
        active = self._is_active.unsqueeze(-1)
        candidate = self._velocity_scale * self._target_position + (1.0 - self._velocity_scale) * self._current_position
        candidate = torch.where(active, self._clamp_to_limits(candidate), self._current_position)
        velocity = torch.where(
            active,
            (candidate - self._previous_position) / dt,
            torch.zeros_like(self._current_velocity),
        )
        complete = self._is_active & ((candidate - self._target_position).abs() < self.cfg.position_tolerance).all(
            dim=1
        )
        complete &= (velocity.abs() < self.cfg.velocity_tolerance).all(dim=1)
        complete_joints = complete.unsqueeze(-1)
        self._current_position.copy_(torch.where(complete_joints, self._target_position, candidate))
        self._current_velocity.copy_(torch.where(complete_joints, torch.zeros_like(velocity), velocity))
        self._is_active.logical_and_(~complete)
        return self._current_position.clone()

    def is_trajectory_complete(self) -> torch.Tensor:
        """Check if trajectories are complete."""
        return ~self._is_active

    def get_current_velocity(self) -> torch.Tensor:
        """Get current velocity estimate."""
        return self._current_velocity.clone()

    def get_current_acceleration(self) -> torch.Tensor:
        """Get current acceleration for EMA profile.

        For EMA dynamics: x_new = α*target + (1-α)*x_old
        The velocity is: v = (x_new - x_old) / dt = α*(target - x_old) / dt
        The acceleration is: a = dv/dt = -α*v / dt = -α²*(target - x) / dt²

        Since EMA has exponential decay, acceleration is proportional to the
        negative of current velocity (deceleration as approaching target).
        """
        # We approximate dt as 1 timestep since we don't store it.
        acceleration = -self._velocity_scale * self._current_velocity
        return torch.where(self._is_active.unsqueeze(-1), acceleration, torch.zeros_like(acceleration))

    def get_time_remaining(self) -> torch.Tensor:
        """Estimate time remaining based on current convergence rate."""
        with torch.no_grad():
            # Estimate based on exponential decay
            # Time constant tau = -dt / ln(1 - scale)
            # Approximate time to reach 99% of target: 5 * tau

            avg_scale = self._velocity_scale.mean(dim=1)
            # Avoid log(0) by clamping
            time_constant = -1.0 / torch.log(1.0 - avg_scale.clamp(max=0.99))

            # Scale by how far we are from target
            position_error = torch.abs(self._current_position - self._target_position)
            max_error = position_error.max(dim=1)[0]
            initial_error = torch.abs(self._initial_position - self._target_position).max(dim=1)[0]

            # Avoid division by zero
            progress_ratio = torch.where(
                initial_error > 1e-6, max_error / (initial_error + 1e-8), torch.zeros_like(initial_error)
            )

            # Estimate remaining time
            time_remaining = time_constant * (-torch.log(progress_ratio.clamp(min=1e-6)))

            # Set to zero for inactive trajectories
            time_remaining = torch.where(self._is_active, time_remaining, torch.zeros_like(time_remaining))

            return time_remaining.clamp(min=0.0)
