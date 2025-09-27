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

"""Trapezoidal velocity profile implementation with fully vectorized operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import VelocityProfileBase

if TYPE_CHECKING:
    from .configs import TrapezoidalVelocityProfileCfg


class TrapezoidalVelocityProfile(VelocityProfileBase):
    """Trapezoidal velocity profile with acceleration, cruise, and deceleration phases.

    This implementation uses fully vectorized operations for computational efficiency
    in batched RL training environments.

    The profile consists of three phases:
    1. Acceleration: Constant acceleration from initial to max velocity
    2. Cruise: Constant velocity at max velocity
    3. Deceleration: Constant deceleration from max velocity to zero

    For short distances, a triangular profile (no cruise phase) is used.
    All joints are synchronized to complete their trajectories simultaneously.
    """

    cfg: TrapezoidalVelocityProfileCfg
    """Configuration for trapezoidal velocity profile."""

    def _initialize_states(self) -> None:
        """Initialize trapezoidal profile specific states."""
        # Current kinematic states
        self._current_velocity: torch.Tensor = torch.zeros((self._num_envs, self._num_joints), device=self._device)
        self._current_acceleration: torch.Tensor = torch.zeros((self._num_envs, self._num_joints), device=self._device)

        # Initial velocity when trajectory starts
        self._initial_velocity: torch.Tensor = torch.zeros_like(self._current_velocity)

        # Profile parameters (randomized per joint)
        self._max_acceleration: torch.Tensor = torch.zeros((self._num_envs, self._num_joints), device=self._device)
        self._max_velocity: torch.Tensor = torch.zeros((self._num_envs, self._num_joints), device=self._device)
        self._max_deceleration: torch.Tensor = torch.zeros((self._num_envs, self._num_joints), device=self._device)

        # Trajectory planning
        self._trajectory_distance: torch.Tensor = torch.zeros((self._num_envs, self._num_joints), device=self._device)
        self._trajectory_direction: torch.Tensor = torch.zeros((self._num_envs, self._num_joints), device=self._device)

        # Synchronized timing (same for all joints in an environment)
        self._trajectory_duration = torch.zeros(self._num_envs, device=self._device)
        self._elapsed_time = torch.zeros(self._num_envs, device=self._device)

        # Phase durations (synchronized across joints)
        self._accel_duration = torch.zeros(self._num_envs, device=self._device)
        self._cruise_duration = torch.zeros(self._num_envs, device=self._device)
        self._decel_duration = torch.zeros(self._num_envs, device=self._device)

        # Effective velocities after synchronization
        self._sync_max_velocity: torch.Tensor = torch.zeros((self._num_envs, self._num_joints), device=self._device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset profile state for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        # Reset all states for specified environments
        self._is_active[env_ids] = False
        self._current_position[env_ids] = 0.0
        self._target_position[env_ids] = 0.0
        self._current_velocity[env_ids] = 0.0
        self._current_acceleration[env_ids] = 0.0

        self._initial_position[env_ids] = 0.0
        self._initial_velocity[env_ids] = 0.0

        self._elapsed_time[env_ids] = 0.0
        self._trajectory_duration[env_ids] = 0.0

        self._accel_duration[env_ids] = 0.0
        self._cruise_duration[env_ids] = 0.0
        self._decel_duration[env_ids] = 0.0

    def set_target(
        self,
        current_pos: torch.Tensor,
        target_pos: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Set new target and plan synchronized trajectory."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        # Validate and potentially clamp target positions
        target_pos = self._validate_target_positions(target_pos, env_ids)

        # Store current state as initial state
        self._current_position[env_ids] = current_pos
        self._initial_position[env_ids] = current_pos
        self._target_position[env_ids] = target_pos

        # Inherit current velocity if smooth start is enabled
        if self.cfg.use_smooth_start and self._is_active[env_ids].any():
            self._initial_velocity[env_ids] = self._current_velocity[env_ids]
        else:
            self._initial_velocity[env_ids] = 0.0
            self._current_velocity[env_ids] = 0.0

        # Calculate trajectory parameters
        self._trajectory_distance[env_ids] = target_pos - current_pos
        self._trajectory_direction[env_ids] = torch.sign(self._trajectory_distance[env_ids])

        # Sample random profile parameters per joint
        self._sample_profile_parameters(env_ids)

        # Plan synchronized trajectories (vectorized)
        self._plan_synchronized_trajectories_batched(env_ids)

        # Reset timing
        self._elapsed_time[env_ids] = 0.0

        # Mark as active
        self._is_active[env_ids] = True

    def _sample_profile_parameters(self, env_ids: torch.Tensor) -> None:
        """Sample random acceleration and velocity parameters for each joint.

        Args:
            env_ids: Environment indices to update.
        """
        num_envs_to_update = len(env_ids)

        # Sample acceleration
        accel = torch.rand(num_envs_to_update, self._num_joints, device=self._device)
        accel = accel * (self.cfg.acceleration_range[1] - self.cfg.acceleration_range[0])
        accel += self.cfg.acceleration_range[0]
        self._max_acceleration[env_ids] = accel

        # Sample max velocity
        vel = torch.rand(num_envs_to_update, self._num_joints, device=self._device)
        vel = vel * (self.cfg.max_velocity_range[1] - self.cfg.max_velocity_range[0])
        vel += self.cfg.max_velocity_range[0]
        self._max_velocity[env_ids] = vel

        # Sample deceleration (or use acceleration if not specified)
        if self.cfg.deceleration_range is not None:
            decel = torch.rand(num_envs_to_update, self._num_joints, device=self._device)
            decel = decel * (self.cfg.deceleration_range[1] - self.cfg.deceleration_range[0])
            decel += self.cfg.deceleration_range[0]
            self._max_deceleration[env_ids] = decel
        else:
            self._max_deceleration[env_ids] = self._max_acceleration[env_ids]

        # Apply velocity limits if enabled
        if self.cfg.enable_velocity_limits and self._velocity_limits is not None:
            self._max_velocity[env_ids] = torch.min(self._max_velocity[env_ids], self._velocity_limits[env_ids])

    def _plan_synchronized_trajectories_batched(self, env_ids: torch.Tensor) -> None:
        """Plan synchronized trajectories using fully vectorized operations.

        Args:
            env_ids: Environment indices to plan for.
        """
        # Get parameters for selected environments
        distances = torch.abs(self._trajectory_distance[env_ids])  # [N, num_joints]
        max_vels = self._max_velocity[env_ids]
        max_accels = self._max_acceleration[env_ids]
        max_decels = self._max_deceleration[env_ids]

        # Calculate time for each joint to complete with its own parameters
        # Time to reach max velocity
        t_to_max_vel = max_vels / max_accels

        # Distance during acceleration
        d_acc = 0.5 * max_accels * t_to_max_vel * t_to_max_vel

        # Distance during deceleration
        t_to_zero_vel = max_vels / max_decels
        d_dec = 0.5 * max_decels * t_to_zero_vel * t_to_zero_vel

        # Check if trapezoidal or triangular profile is needed
        needs_triangular = (d_acc + d_dec) > distances

        # For triangular profiles, calculate peak velocity
        # d = v_peak^2 / (2*a) + v_peak^2 / (2*d) = v_peak^2 * (1/(2*a) + 1/(2*d))
        # v_peak = sqrt(2*d / (1/a + 1/d))
        v_peak = torch.sqrt(2 * distances / (1 / max_accels + 1 / max_decels))

        # Calculate times for both profile types
        # Triangular
        t_acc_tri = v_peak / max_accels
        t_dec_tri = v_peak / max_decels
        t_total_tri = t_acc_tri + t_dec_tri

        # Trapezoidal
        d_cruise = distances - d_acc - d_dec
        d_cruise = torch.max(d_cruise, torch.zeros_like(d_cruise))  # Ensure non-negative
        t_cruise_trap = d_cruise / (max_vels + 1e-8)
        t_total_trap = t_to_max_vel + t_cruise_trap + t_to_zero_vel

        # Select appropriate times based on profile type
        joint_times = torch.where(needs_triangular, t_total_tri, t_total_trap)

        # Handle zero-distance joints
        joint_times = torch.where(distances < 1e-6, torch.zeros_like(joint_times), joint_times)

        # Synchronize: all joints in an environment use the same total time
        if self.cfg.synchronize_joints:
            if self.cfg.time_scaling_method == "max_time":
                # Use the slowest joint's time
                sync_times = joint_times.max(dim=1)[0]  # [N]
            else:  # average_time
                # Use average time (excluding zero times)
                non_zero_mask = joint_times > 1e-6
                sums = (joint_times * non_zero_mask).sum(dim=1)
                counts = non_zero_mask.sum(dim=1).clamp(min=1)
                sync_times = sums / counts
        else:
            # No synchronization, use max time per environment anyway
            sync_times = joint_times.max(dim=1)[0]

        # Store synchronized durations
        self._trajectory_duration[env_ids] = sync_times

        # Compute synchronized profile parameters for each environment
        self._compute_synchronized_profile_batched(env_ids, distances, sync_times)

    def _compute_synchronized_profile_batched(
        self,
        env_ids: torch.Tensor,
        distances: torch.Tensor,
        total_times: torch.Tensor,
    ) -> None:
        """Compute synchronized profile parameters using vectorized operations.

        Args:
            env_ids: Environment indices [N].
            distances: Absolute distances for each joint [N, num_joints].
            total_times: Total time for synchronized motion [N].
        """
        # Use average acceleration/deceleration for simplicity
        avg_accel = self._max_acceleration[env_ids].mean(dim=1, keepdim=True)  # [N, 1]
        avg_decel = self._max_deceleration[env_ids].mean(dim=1, keepdim=True)  # [N, 1]

        # Find maximum distance to travel in each environment
        max_distance = distances.max(dim=1, keepdim=True)[0]  # [N, 1]

        # Solve for velocity that achieves the desired total time
        # For trapezoidal: total_time = v/a + (d - 0.5*v^2/a - 0.5*v^2/d) / v
        # This is a quadratic: v^2 * (0.5/a + 0.5/d) - v*total_time + d = 0

        a = 0.5 * (1 / avg_accel + 1 / avg_decel)
        b = -total_times.unsqueeze(-1)  # [N, 1]
        c = max_distance

        discriminant = b * b - 4 * a * c

        # For positive discriminant, use trapezoidal; otherwise use triangular
        use_trapezoidal = (discriminant >= 0) & (max_distance > 1e-6)

        # Initialize velocities
        v_max = torch.zeros_like(max_distance)  # [N, 1]

        # Trapezoidal case
        if use_trapezoidal.any():
            v_trap = (-b - torch.sqrt(discriminant.clamp(min=0))) / (2 * a)
            v_max = torch.where(use_trapezoidal, v_trap, v_max)

        # Triangular case (or fallback)
        use_triangular = ~use_trapezoidal & (max_distance > 1e-6)
        if use_triangular.any():
            # For triangular: total_time = v_peak * (1/a + 1/d)
            v_tri = total_times.unsqueeze(-1) / (1 / avg_accel + 1 / avg_decel)
            v_max = torch.where(use_triangular, v_tri, v_max)

        # Calculate phase durations for each environment
        t_acc = v_max / avg_accel  # [N, 1]
        t_dec = v_max / avg_decel  # [N, 1]

        # Distance during accel and decel
        d_acc = 0.5 * avg_accel * t_acc * t_acc
        d_dec = 0.5 * avg_decel * t_dec * t_dec
        d_cruise = max_distance - d_acc - d_dec
        d_cruise = torch.max(d_cruise, torch.zeros_like(d_cruise))

        # Cruise time
        t_cruise = d_cruise / (v_max + 1e-8)

        # Set to zero for triangular profiles
        t_cruise = torch.where(use_trapezoidal, t_cruise, torch.zeros_like(t_cruise))

        # Store phase durations (squeeze to [N])
        self._accel_duration[env_ids] = t_acc.squeeze(-1)
        self._cruise_duration[env_ids] = t_cruise.squeeze(-1)
        self._decel_duration[env_ids] = t_dec.squeeze(-1)

        # Set synchronized max velocity for all joints
        # Scale based on each joint's distance ratio
        distance_ratio = distances / (max_distance + 1e-8)  # [N, num_joints]
        self._sync_max_velocity[env_ids] = v_max * distance_ratio  # Broadcasting [N, 1] * [N, num_joints]

    def compute_next_position(self, dt: float) -> torch.Tensor:
        """Compute next position based on trapezoidal velocity profile using vectorized operations."""
        # Validate dt
        if dt <= 0:
            print(f"Warning: Invalid dt={dt}, returning current position")
            return self._current_position.clone()

        # Update elapsed time for active trajectories
        self._elapsed_time = torch.where(self._is_active, self._elapsed_time + dt, self._elapsed_time)

        if not self._is_active.any():
            return self._current_position.clone()

        # Get timing parameters [num_envs]
        t = self._elapsed_time
        t_acc = self._accel_duration
        t_cruise = self._cruise_duration
        t_dec = self._decel_duration

        # Determine phase for each environment
        in_accel = (t <= t_acc) & self._is_active
        in_cruise = (t > t_acc) & (t <= t_acc + t_cruise) & self._is_active
        in_decel = (t > t_acc + t_cruise) & (t <= t_acc + t_cruise + t_dec) & self._is_active
        is_complete = (t > t_acc + t_cruise + t_dec) & self._is_active

        # Expand time dimensions for broadcasting [num_envs] -> [num_envs, 1]
        t_exp = t.unsqueeze(-1)

        # === Vectorized phase computations ===

        # Acceleration phase: s = s0 + v0*t + 0.5*a*t^2
        if in_accel.any():
            t_phase = t_exp[in_accel]
            v_max = self._sync_max_velocity[in_accel]
            a = self._max_acceleration[in_accel]
            v0 = self._initial_velocity[in_accel]

            # Velocity: v = v0 + a*t (clamped to max)
            v = v0 + a * t_phase
            v = torch.min(v, v_max)

            # Position
            displacement = v0 * t_phase + 0.5 * a * t_phase * t_phase
            self._current_position[in_accel] = (
                self._initial_position[in_accel] + self._trajectory_direction[in_accel] * displacement
            )
            self._current_velocity[in_accel] = self._trajectory_direction[in_accel] * v
            self._current_acceleration[in_accel] = self._trajectory_direction[in_accel] * a

        # Cruise phase: constant velocity
        if in_cruise.any():
            t_acc_exp = t_acc.unsqueeze(-1)[in_cruise]
            t_cruise_elapsed = t_exp[in_cruise] - t_acc_exp
            v_max = self._sync_max_velocity[in_cruise]
            a = self._max_acceleration[in_cruise]
            v0 = self._initial_velocity[in_cruise]

            # Distance during acceleration
            d_acc = v0 * t_acc_exp + 0.5 * a * t_acc_exp * t_acc_exp

            # Total displacement
            displacement = d_acc + v_max * t_cruise_elapsed
            self._current_position[in_cruise] = (
                self._initial_position[in_cruise] + self._trajectory_direction[in_cruise] * displacement
            )
            self._current_velocity[in_cruise] = self._trajectory_direction[in_cruise] * v_max
            self._current_acceleration[in_cruise] = 0.0

        # Deceleration phase
        if in_decel.any():
            t_acc_exp = t_acc.unsqueeze(-1)[in_decel]
            t_cruise_exp = t_cruise.unsqueeze(-1)[in_decel]
            t_decel_elapsed = t_exp[in_decel] - t_acc_exp - t_cruise_exp

            v_max = self._sync_max_velocity[in_decel]
            a_acc = self._max_acceleration[in_decel]
            a_dec = self._max_deceleration[in_decel]
            v0 = self._initial_velocity[in_decel]

            # Distances during acceleration and cruise
            d_acc = v0 * t_acc_exp + 0.5 * a_acc * t_acc_exp * t_acc_exp
            d_cruise = v_max * t_cruise_exp

            # Current velocity during deceleration: v = v_max - a_dec * t
            v = v_max - a_dec * t_decel_elapsed
            v = torch.max(v, torch.zeros_like(v))

            # Distance during deceleration: d = v_max*t - 0.5*a_dec*t^2
            d_decel = v_max * t_decel_elapsed - 0.5 * a_dec * t_decel_elapsed * t_decel_elapsed

            # Total displacement
            displacement = d_acc + d_cruise + d_decel
            self._current_position[in_decel] = (
                self._initial_position[in_decel] + self._trajectory_direction[in_decel] * displacement
            )
            self._current_velocity[in_decel] = self._trajectory_direction[in_decel] * v
            self._current_acceleration[in_decel] = -self._trajectory_direction[in_decel] * a_dec

        # Complete trajectories
        if is_complete.any():
            self._current_position[is_complete] = self._target_position[is_complete]
            self._current_velocity[is_complete] = 0.0
            self._current_acceleration[is_complete] = 0.0
            self._is_active[is_complete] = False

        # Clamp positions to joint limits
        self._current_position = self._clamp_to_limits(self._current_position)

        return self._current_position.clone()

    def is_trajectory_complete(self) -> torch.Tensor:
        """Check if trajectories are complete."""
        return ~self._is_active

    def get_current_velocity(self) -> torch.Tensor:
        """Get current velocity."""
        return self._current_velocity.clone()

    def get_current_acceleration(self) -> torch.Tensor:
        """Get current acceleration."""
        return self._current_acceleration.clone()

    def get_time_remaining(self) -> torch.Tensor:
        """Get time remaining for each trajectory."""
        time_remaining = self._trajectory_duration - self._elapsed_time
        time_remaining = torch.where(self._is_active, time_remaining.clamp(min=0.0), torch.zeros_like(time_remaining))
        return time_remaining
