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

"""True linear velocity profile implementation with constant velocity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import VelocityProfileBase

if TYPE_CHECKING:
    from .configs import LinearVelocityProfileCfg


class LinearVelocityProfile(VelocityProfileBase):
    """Linear velocity profile with constant velocity motion.

    This profile implements true linear motion where the position changes
    at a constant velocity: pos = initial_pos + velocity * time

    The profile features:
    - Constant velocity throughout the trajectory
    - Instantaneous start and stop (infinite acceleration/deceleration)
    - Linear position change over time
    - Deterministic completion time based on distance and velocity
    """

    cfg: LinearVelocityProfileCfg
    """Configuration for linear velocity profile."""

    def _initialize_states(self) -> None:
        """Initialize linear profile specific states."""
        # Velocity for each joint (constant during trajectory)
        self._trajectory_velocity = torch.zeros((self._num_envs, self._num_joints), device=self._device)

        # Initial position when trajectory starts
        self._initial_position = torch.zeros_like(self._current_position)  # type: ignore[has-type]

        # Time elapsed since trajectory start
        self._elapsed_time = torch.zeros(self._num_envs, device=self._device)

        # Total time needed to complete trajectory
        self._trajectory_duration = torch.zeros(self._num_envs, device=self._device)

        # Direction of motion for each joint
        self._trajectory_direction = torch.zeros((self._num_envs, self._num_joints), device=self._device)

        # Total distance to travel for each joint
        self._trajectory_distance = torch.zeros((self._num_envs, self._num_joints), device=self._device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset profile state for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        # Reset states for specified environments
        self._is_active[env_ids] = False
        self._current_position[env_ids] = 0.0
        self._target_position[env_ids] = 0.0
        self._initial_position[env_ids] = 0.0
        self._trajectory_velocity[env_ids] = 0.0
        self._elapsed_time[env_ids] = 0.0
        self._trajectory_duration[env_ids] = 0.0
        self._trajectory_direction[env_ids] = 0.0
        self._trajectory_distance[env_ids] = 0.0

    def set_target(
        self,
        current_pos: torch.Tensor,
        target_pos: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Set new target and initialize linear trajectory."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        # Validate and potentially clamp target positions
        target_pos = self._validate_target_positions(target_pos, env_ids)

        # Store positions
        self._current_position[env_ids] = current_pos
        self._target_position[env_ids] = target_pos
        self._initial_position[env_ids] = current_pos

        # Calculate trajectory parameters
        self._trajectory_distance[env_ids] = target_pos - current_pos
        self._trajectory_direction[env_ids] = torch.sign(self._trajectory_distance[env_ids])

        # Sample random velocities for each environment
        self._sample_velocities(env_ids)

        # Plan synchronized trajectories if needed
        if self.cfg.synchronize_joints:
            self._plan_synchronized_trajectories(env_ids)
        else:
            self._plan_independent_trajectories(env_ids)

        # Reset timing
        self._elapsed_time[env_ids] = 0.0

        # Mark as active
        self._is_active[env_ids] = True

    def _sample_velocities(self, env_ids: torch.Tensor) -> None:
        """Sample random velocities for specified environments.

        Args:
            env_ids: Environment indices to update.
        """
        num_envs_to_update = len(env_ids)

        if self.cfg.synchronize_joints:
            # All joints use the same velocity magnitude
            velocities = torch.rand(num_envs_to_update, 1, device=self._device)
            velocities = velocities * (self.cfg.velocity_range[1] - self.cfg.velocity_range[0])
            velocities += self.cfg.velocity_range[0]
            # Broadcast to all joints
            base_velocity = velocities.expand(-1, self._num_joints)
        else:
            # Each joint gets its own velocity
            velocities = torch.rand(num_envs_to_update, self._num_joints, device=self._device)
            velocities = velocities * (self.cfg.velocity_range[1] - self.cfg.velocity_range[0])
            velocities += self.cfg.velocity_range[0]
            base_velocity = velocities

        # Apply velocity limits if enabled
        if self.cfg.enable_velocity_limits and self._velocity_limits is not None:
            base_velocity = torch.min(base_velocity, self._velocity_limits[env_ids])

        # Apply direction to get signed velocity
        self._trajectory_velocity[env_ids] = base_velocity * self._trajectory_direction[env_ids]

    def _plan_synchronized_trajectories(self, env_ids: torch.Tensor) -> None:
        """Plan trajectories with all joints synchronized to finish simultaneously.

        Args:
            env_ids: Environment indices to plan for.
        """
        # Get distances and velocities for all selected environments [N, num_joints]
        distances = torch.abs(self._trajectory_distance[env_ids])
        velocities = torch.abs(self._trajectory_velocity[env_ids])

        # Find the joint that will take the longest time for each environment
        joint_times = torch.where(distances > 1e-6, distances / (velocities + 1e-8), torch.zeros_like(distances))

        # Max time for each environment [N]
        max_times = joint_times.max(dim=1)[0]

        # Mask for environments that need planning (not already at target)
        needs_planning = max_times > 1e-6

        if needs_planning.any():
            # Adjust velocities so all joints finish at the same time
            # v = d / t for each joint [N, num_joints]
            max_times_exp = max_times.unsqueeze(-1)  # [N, 1] for broadcasting
            adjusted_velocities = distances / (max_times_exp + 1e-8)

            # Apply velocity limits if needed
            if self.cfg.enable_velocity_limits and self._velocity_limits is not None:
                adjusted_velocities = torch.min(adjusted_velocities, self._velocity_limits[env_ids])
                # Recalculate max_time if velocities were clamped
                joint_times = torch.where(
                    distances > 1e-6, distances / (adjusted_velocities + 1e-8), torch.zeros_like(distances)
                )
                max_times = joint_times.max(dim=1)[0]

            # Update velocities with correct sign for environments that need planning
            planning_envs = env_ids[needs_planning]
            self._trajectory_velocity[planning_envs] = (
                adjusted_velocities[needs_planning] * self._trajectory_direction[planning_envs]
            )
            self._trajectory_duration[planning_envs] = max_times[needs_planning]

        # Set duration to zero for environments already at target
        at_target_envs = env_ids[~needs_planning]
        if len(at_target_envs) > 0:
            self._trajectory_velocity[at_target_envs] = 0.0
            self._trajectory_duration[at_target_envs] = 0.0

    def _plan_independent_trajectories(self, env_ids: torch.Tensor) -> None:
        """Plan trajectories with each joint moving independently.

        Args:
            env_ids: Environment indices to plan for.
        """
        # Get distances and velocities for all selected environments [N, num_joints]
        distances = torch.abs(self._trajectory_distance[env_ids])
        velocities = torch.abs(self._trajectory_velocity[env_ids])

        # Each joint finishes based on its own velocity
        joint_times = torch.where(distances > 1e-6, distances / (velocities + 1e-8), torch.zeros_like(distances))

        # Use maximum time as overall trajectory duration for each environment [N]
        max_times = joint_times.max(dim=1)[0]
        self._trajectory_duration[env_ids] = max_times

    def compute_next_position(self, dt: float) -> torch.Tensor:
        """Compute next position using constant velocity motion."""
        # Validate dt
        if dt <= 0:
            print(f"Warning: Invalid dt={dt}, returning current position")
            return self._current_position.clone()

        # Update elapsed time for active trajectories
        self._elapsed_time = torch.where(self._is_active, self._elapsed_time + dt, self._elapsed_time)

        # Check which trajectories are complete
        trajectory_complete = (
            (self._elapsed_time >= self._trajectory_duration) & (self._trajectory_duration > 0) & self._is_active
        )

        # Set completed trajectories to target
        if trajectory_complete.any():
            self._current_position[trajectory_complete] = self._target_position[trajectory_complete]
            self._is_active[trajectory_complete] = False

        # Update positions for active trajectories
        if self._is_active.any():
            # Compute linear motion: pos = initial_pos + velocity * time
            # Broadcast elapsed_time to match joint dimensions
            elapsed_time_expanded = self._elapsed_time.unsqueeze(-1)  # [num_envs, 1]
            new_positions = self._initial_position + self._trajectory_velocity * elapsed_time_expanded

            # Clamp to targets based on direction
            # For positive direction, don't exceed target
            positive_direction = self._trajectory_direction > 0
            new_positions = torch.where(
                positive_direction, torch.min(new_positions, self._target_position), new_positions
            )

            # For negative direction, don't go below target
            negative_direction = self._trajectory_direction < 0
            new_positions = torch.where(
                negative_direction, torch.max(new_positions, self._target_position), new_positions
            )

            # For zero distance joints (already at target), keep target position
            at_target = torch.abs(self._trajectory_distance) < 1e-6
            new_positions = torch.where(at_target, self._target_position, new_positions)

            # Only update active trajectories
            active_mask = self._is_active.unsqueeze(-1)  # [num_envs, 1]
            self._current_position = torch.where(active_mask, new_positions, self._current_position)

        # Clamp positions to joint limits
        self._current_position = self._clamp_to_limits(self._current_position)

        return self._current_position.clone()

    def is_trajectory_complete(self) -> torch.Tensor:
        """Check if trajectories are complete."""
        return ~self._is_active

    def get_current_velocity(self) -> torch.Tensor:
        """Get current velocity (constant during trajectory)."""
        # Return velocity for active trajectories, zero for inactive
        velocity = torch.zeros_like(self._trajectory_velocity)
        velocity[self._is_active] = self._trajectory_velocity[self._is_active]
        return velocity

    def get_current_acceleration(self) -> torch.Tensor:
        """Get current acceleration (always zero for constant velocity)."""
        # Linear profile has zero acceleration during motion
        # (infinite acceleration at start/stop is not modeled)
        return torch.zeros_like(self._current_position)

    def get_time_remaining(self) -> torch.Tensor:
        """Get time remaining for each trajectory."""
        time_remaining = self._trajectory_duration - self._elapsed_time
        time_remaining = torch.where(self._is_active, time_remaining.clamp(min=0.0), torch.zeros_like(time_remaining))
        return time_remaining
