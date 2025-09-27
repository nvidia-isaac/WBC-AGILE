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

"""Base class for velocity profiles used in trajectory generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .configs import VelocityProfileBaseCfg


class VelocityProfileBase(ABC):
    """Abstract base class for velocity profiles.

    This class defines the interface that all velocity profiles must implement.
    Velocity profiles control how joint positions transition from current to target values.
    """

    def __init__(
        self,
        cfg: VelocityProfileBaseCfg,
        num_envs: int,
        num_joints: int,
        device: torch.device,
        joint_limits: torch.Tensor | None = None,
        velocity_limits: torch.Tensor | None = None,
    ):
        """Initialize the velocity profile.

        Args:
            cfg: Profile configuration.
            num_envs: Number of parallel environments.
            num_joints: Number of joints per environment.
            device: Torch device for computations.
            joint_limits: Optional [num_envs, num_joints, 2] tensor of position limits.
            velocity_limits: Optional [num_envs, num_joints] tensor of velocity limits.
        """
        self.cfg = cfg
        self._num_envs = num_envs
        self._num_joints = num_joints
        self._device = device

        # Store limits if provided
        self._joint_limits = joint_limits
        self._velocity_limits = velocity_limits

        # Common state variables
        self._initial_position: torch.Tensor = torch.zeros((num_envs, num_joints), device=device)
        self._current_position: torch.Tensor = torch.zeros((num_envs, num_joints), device=device)
        self._target_position: torch.Tensor = torch.zeros((num_envs, num_joints), device=device)
        self._is_active: torch.Tensor = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Initialize profile-specific states
        self._initialize_states()

    @abstractmethod
    def _initialize_states(self) -> None:
        """Initialize profile-specific state variables."""
        pass

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset profile state for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        pass

    @abstractmethod
    def set_target(
        self,
        current_pos: torch.Tensor,
        target_pos: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Set new target and plan trajectory from current position.

        Args:
            current_pos: [num_envs, num_joints] or [len(env_ids), num_joints] current positions.
            target_pos: [num_envs, num_joints] or [len(env_ids), num_joints] target positions.
            env_ids: Optional environment indices to update.

        Note:
            Always starts from current_pos regardless of any ongoing trajectory.
        """
        pass

    @abstractmethod
    def compute_next_position(self, dt: float) -> torch.Tensor:
        """Compute next position for all environments.

        Args:
            dt: Time step in seconds.

        Returns:
            [num_envs, num_joints] tensor of joint positions.
        """
        pass

    @abstractmethod
    def is_trajectory_complete(self) -> torch.Tensor:
        """Check if trajectories are complete.

        Returns:
            [num_envs] boolean tensor indicating completion status.
        """
        pass

    @abstractmethod
    def get_current_velocity(self) -> torch.Tensor:
        """Get current velocity for monitoring.

        Returns:
            [num_envs, num_joints] tensor of velocities in rad/s.
        """
        pass

    @abstractmethod
    def get_current_acceleration(self) -> torch.Tensor:
        """Get current acceleration for monitoring.

        Returns:
            [num_envs, num_joints] tensor of accelerations in rad/s^2.
        """
        pass

    def get_trajectory_progress(self) -> torch.Tensor:
        """Get normalized progress [0, 1] for each trajectory.

        Returns:
            [num_envs] tensor of progress values.
        """
        # Default implementation based on position
        with torch.no_grad():
            total_distance = torch.abs(self._target_position - self._initial_position)
            current_distance = torch.abs(self._current_position - self._initial_position)
            # Avoid division by zero
            progress = torch.where(
                total_distance > 1e-6, current_distance / (total_distance + 1e-8), torch.ones_like(total_distance)
            )
            # Average across joints for each environment
            return progress.mean(dim=1).clamp(0, 1)

    @abstractmethod
    def get_time_remaining(self) -> torch.Tensor:
        """Get estimated time remaining for each trajectory.

        Returns:
            [num_envs] tensor of time remaining in seconds.
        """
        pass

    # Properties
    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._num_envs

    @property
    def num_joints(self) -> int:
        """Number of joints being controlled."""
        return self._num_joints

    @property
    def device(self) -> torch.device:
        """Device for tensor operations."""
        return self._device

    # Helper methods
    def _clamp_to_limits(self, positions: torch.Tensor) -> torch.Tensor:
        """Clamp positions to joint limits if enabled.

        Args:
            positions: [num_envs, num_joints] tensor of positions.

        Returns:
            Clamped positions.
        """
        if self.cfg.enable_position_limits and self._joint_limits is not None:
            return torch.clamp(positions, self._joint_limits[..., 0], self._joint_limits[..., 1])
        return positions

    def _validate_target_positions(self, target_pos: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Validate and potentially clamp target positions.

        Args:
            target_pos: Target positions to validate.
            env_ids: Environment indices being updated.

        Returns:
            Validated target positions.
        """
        if self.cfg.enable_position_limits and self._joint_limits is not None:
            if env_ids is not None:
                limits = self._joint_limits[env_ids]
            else:
                limits = self._joint_limits

            clamped = torch.clamp(target_pos, limits[..., 0], limits[..., 1])

            # Check if any clamping occurred
            if not torch.allclose(target_pos, clamped, atol=1e-6):
                num_clamped = (~torch.isclose(target_pos, clamped, atol=1e-6)).sum().item()
                print(f"Warning: {num_clamped} target positions were clamped to joint limits")

            return clamped

        return target_pos
