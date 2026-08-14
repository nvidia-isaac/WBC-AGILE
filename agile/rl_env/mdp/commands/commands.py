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

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.sensors import RayCaster
from isaaclab.utils import math as math_utils

# Only import the command class during type checking
if TYPE_CHECKING:
    from .commands_cfg import (
        UniformNullVelocityCommandCfg,
        UniformVelocityBaseHeightCommandCfg,
        UniformVelocityGaitBaseHeightCommandCfg,
    )


class UniformNullVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with min velocity command and traveled distance metric."""

    def __init__(self, cfg: UniformNullVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.metrics_cumulative: dict[str, torch.Tensor] = {}
        for k, v in self.metrics.items():  # type: ignore
            if "xy" in k:
                # store vectors not norms
                self.metrics_cumulative[k] = torch.zeros(env.num_envs, 2, device=self.device)
            else:
                self.metrics_cumulative[k] = torch.zeros_like(v)

        # for performance estimate
        self.metrics["traveled_distance"] = torch.zeros(env.num_envs, device=self.device)  # type: ignore
        self.traveled_distance = torch.zeros(env.num_envs, device=self.device)
        self.start_positions = self.robot.data.root_pos_w.torch[:, :2].clone()

        self.min_vel_norm = cfg.min_vel_norm

        # smoothed velocity estimate
        self.smoothing_param = cfg.ema_smoothing_param
        self.vel_xy_smoothed = torch.zeros_like(self.robot.data.root_lin_vel_b.torch[:, :2])
        self.angvel_smoothed = torch.zeros_like(self.robot.data.root_ang_vel_b.torch[:, 2])

        # Initialize command filtering
        self._setup_command_filtering(cfg, env)

    def _setup_command_filtering(self, cfg: UniformNullVelocityCommandCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize command filtering state and configuration."""
        # Dynamically build axis mapping based on available velocity command dimensions
        # The base class has 3 velocity commands: [lin_vel_x, lin_vel_y, ang_vel_z]
        # Subclasses may override this to add more axes
        self.filter_axis_map = self._get_command_axis_mapping()

        # Store filter configuration
        self.command_filter_config = cfg.command_filter_alpha_ranges if cfg.command_filter_alpha_ranges else {}

        # Get number of command dimensions from the actual command tensor
        num_command_dims = self.vel_command_b.shape[1]

        # Initialize filter coefficients (default to 1.0 = no filtering)
        self.command_filter_alphas = torch.ones(env.num_envs, num_command_dims, device=self.device)

        # Initialize target command buffer (unfiltered commands)
        self.vel_command_target_b = torch.zeros_like(self.vel_command_b)

    def _get_command_axis_mapping(self) -> dict[str, int]:
        """Get the mapping from axis names to indices in the command tensor.

        This method can be overridden by subclasses to define their own mappings.

        Returns:
            Dictionary mapping axis names to indices in vel_command_b tensor.

        Example for subclass with additional axes:
            def _get_command_axis_mapping(self):
                base_mapping = super()._get_command_axis_mapping()
                # Add height as 4th dimension if using combined tensor
                base_mapping["base_height"] = 3
                return base_mapping
        """
        # Base velocity commands mapping
        return {
            "lin_vel_x": 0,
            "lin_vel_y": 1,
            "ang_vel_z": 2,
        }

    def _sample_filter_coefficients(self, env_ids: Sequence[int] | None = None) -> None:
        """Sample new filter coefficients for specified environments.

        Args:
            env_ids: Indices of environments to sample new coefficients for.
                     If None, no operation is performed.
        """
        # Early return if no environments specified
        if env_ids is None:
            return

        if not self.command_filter_config:
            # No filtering configured - set all alphas to 1.0 (instant response)
            self.command_filter_alphas[env_ids, :] = 1.0
            return

        # Sample per-axis filter coefficients
        for axis_name, axis_idx in self.filter_axis_map.items():
            if axis_name in self.command_filter_config and self.command_filter_config[axis_name] is not None:
                alpha_range = self.command_filter_config[axis_name]
                if alpha_range is None:
                    raise ValueError(f"Alpha range is None for axis {axis_name}. Please check your configuration.")
                # Sample uniformly from the specified range
                self.command_filter_alphas[env_ids, axis_idx] = torch.empty(len(env_ids), device=self.device).uniform_(
                    alpha_range[0], alpha_range[1]
                )
            else:
                # No filtering for this axis
                self.command_filter_alphas[env_ids, axis_idx] = 1.0

    def _update_metrics(self) -> None:
        # update smoothed velocity estimate

        vel_xy = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.robot.data.root_quat_w.torch),
            self.robot.data.root_lin_vel_w.torch[:, :3],
        )[:, :2]
        self.vel_xy_smoothed = self.smoothing_param * vel_xy + (1 - self.smoothing_param) * self.vel_xy_smoothed
        self.angvel_smoothed = (
            self.smoothing_param * self.robot.data.root_ang_vel_w.torch[:, 2]
            + (1 - self.smoothing_param) * self.angvel_smoothed
        )

        # logs data
        self.metrics_cumulative["error_vel_xy"] += self.vel_command_b[:, :2] - self.vel_xy_smoothed
        self.metrics_cumulative["error_vel_yaw"] += torch.abs(self.vel_command_b[:, 2] - self.angvel_smoothed)

        current_positions = self.robot.data.root_pos_w.torch[:, :2]
        traveled_dist = torch.norm(current_positions - self.start_positions, dim=1)

        # norm of vector sum, not sum of norms
        normalizer = torch.clamp(self._env.episode_length_buf, min=1.0)

        self.metrics = {
            k: (v / normalizer if "xy" not in k else torch.norm(v, dim=-1) / normalizer)
            for k, v in self.metrics_cumulative.items()
        }
        self.metrics["traveled_distance"] = self.traveled_distance + traveled_dist

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = {k: v[env_ids].mean().item() for k, v in self.metrics.items()}

        self.traveled_distance[env_ids] = 0.0
        self.start_positions[env_ids] = self.robot.data.root_pos_w.torch[env_ids, :2].clone()

        # Reset filter state: copy targets directly to filtered commands
        # This is correct for reset - we want to start fresh with no history from previous episode
        self.vel_command_b[env_ids] = self.vel_command_target_b[env_ids].clone()
        # Resample filter coefficients for reset environments
        self._sample_filter_coefficients(env_ids)

        super().reset(env_ids)
        for _, v in self.metrics_cumulative.items():
            v[env_ids] = 0.0
        return extras

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        # Reimplement parent's _resample_command with bias sampling
        r = torch.empty(len(env_ids), device=self.device)

        # Helper function for piecewise uniform sampling
        def sample_with_bias(r: torch.Tensor, full_range: tuple[float, float], bias_speed: float) -> torch.Tensor:
            """Sample with bias toward low speeds if bias_sampling is enabled."""
            if self.cfg.bias_sampling:
                # Create mask for which samples go to low-speed range
                bias_mask = r.uniform_(0.0, 1.0) < self.cfg.bias_sampling_probability

                # Initialize with full range sampling
                samples = r.uniform_(*full_range)

                # For biased samples, resample from low-speed range
                if bias_mask.any():
                    # Clamp bias_speed to not exceed the full range
                    bias_range_min = max(full_range[0], -bias_speed)
                    bias_range_max = min(full_range[1], bias_speed)
                    samples[bias_mask] = r[bias_mask].uniform_(bias_range_min, bias_range_max)

                return samples
            else:
                # Standard uniform sampling when bias is disabled
                return r.uniform_(*full_range)

        # Sample new target commands (not filtered yet)
        # -- linear velocity - x direction
        self.vel_command_target_b[env_ids, 0] = sample_with_bias(
            r, self.cfg.ranges.lin_vel_x, self.cfg.bias_sampling_speed
        )

        # -- linear velocity - y direction
        self.vel_command_target_b[env_ids, 1] = sample_with_bias(
            r, self.cfg.ranges.lin_vel_y, self.cfg.bias_sampling_speed
        )

        # -- ang vel yaw - rotation around z
        # For angular velocity, we might want to use a scaled bias speed or the same one
        self.vel_command_target_b[env_ids, 2] = sample_with_bias(
            r, self.cfg.ranges.ang_vel_z, self.cfg.bias_sampling_speed
        )

        # -- heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        # -- update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        # Sample new filter coefficients for these environments
        self._sample_filter_coefficients(env_ids)

        # Update traveled distance tracking
        current_positions = self.robot.data.root_pos_w.torch[env_ids, :2]
        self.traveled_distance[env_ids] += torch.norm(current_positions - self.start_positions[env_ids], dim=1)
        self.start_positions[env_ids] = current_positions.clone()

        # set small velocity samples to zero (apply to both target and filtered)
        too_small_envs = torch.zeros(self._env.num_envs, dtype=torch.bool, device=self.device)
        too_small_envs[env_ids] = self.vel_command_target_b[env_ids].norm(dim=1) < self.min_vel_norm
        self.vel_command_target_b[too_small_envs] = 0
        self.vel_command_b[too_small_envs] = 0

    def _update_command(self) -> None:
        """Update command with low-pass filtering applied per-axis.

        This method is called every simulation step to smoothly transition commands.
        """
        # First apply the low-pass filter to smoothly transition to target commands
        # Filter formula: filtered = alpha * target + (1 - alpha) * previous_filtered
        # Shape: command_filter_alphas is [num_envs, 3], vel commands are [num_envs, 3]
        self.vel_command_b = (
            self.command_filter_alphas * self.vel_command_target_b
            + (1 - self.command_filter_alphas) * self.vel_command_b
        )

        # Call parent to handle heading control and standing envs
        # Parent may modify vel_command_b for heading/standing envs
        super()._update_command()

        # Re-apply standing constraint to target as well (for consistency)
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(standing_env_ids) > 0:
            self.vel_command_target_b[standing_env_ids, :] = 0.0


class UniformVelocityBaseHeightCommand(UniformNullVelocityCommand):
    """Uniform velocity command generator with height command."""

    cfg: UniformVelocityBaseHeightCommandCfg
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: UniformVelocityBaseHeightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.target_height = torch.zeros(env.num_envs, device=self.device)
        self.base_height = torch.zeros(env.num_envs, device=self.device)

        # Initialize height filtering if configured
        # Note: Height uses a separate tensor, so needs separate filter state
        self._setup_height_filtering(cfg, env)

        self.error_height_log = 0.0
        self.random_height_during_walking = cfg.random_height_during_walking
        self.normal_walking_height = cfg.default_height

        # Track previous standing state for height randomization while standing.
        self.prev_stand_normal_height = torch.ones(env.num_envs, dtype=torch.bool, device=self.device)
        self.prev_stand_squat_height = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self.prev_walk = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

        self.current_stand_normal_height = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self.current_stand_squat_height = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self.current_walk = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

        # Raycaster to measure base height
        self._height_sensor: RayCaster = env.scene.sensors[cfg.height_sensor]  # type: ignore
        self._root_id, _ = self.robot.find_bodies(cfg.root_name)

    def _setup_height_filtering(self, cfg: UniformVelocityBaseHeightCommandCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize height command filtering.

        Height is handled separately from velocity commands, so it needs its own filter state.
        Always initializes filter infrastructure - uses alpha=1.0 (no filtering) when not configured.
        """
        # Always initialize filter state
        self.height_filter_alpha = torch.ones(env.num_envs, device=self.device)  # Default 1.0 = no filtering
        self.target_height_unfiltered = torch.zeros(env.num_envs, device=self.device)

        # Store height filter config if available (used for sampling)
        if cfg.command_filter_alpha_ranges and "base_height" in cfg.command_filter_alpha_ranges:
            self.height_filter_config = cfg.command_filter_alpha_ranges.get("base_height")
        else:
            self.height_filter_config = None

    def _sample_filter_coefficients(self, env_ids: Sequence[int] | None = None) -> None:
        """Sample filter coefficients including height."""
        # Early return if no environments specified
        if env_ids is None:
            return

        super()._sample_filter_coefficients(env_ids)

        # Sample height filter coefficients
        if self.height_filter_config:
            # Sample from configured range
            self.height_filter_alpha[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
                self.height_filter_config[0], self.height_filter_config[1]
            )
        else:
            # No filtering - set alpha to 1.0 (pass-through)
            self.height_filter_alpha[env_ids] = 1.0

    def _update_command(self) -> None:
        """Update commands including filtered height."""
        # Always apply height filtering (when alpha=1.0, this is just pass-through)
        self.target_height = (
            self.height_filter_alpha * self.target_height_unfiltered
            + (1 - self.height_filter_alpha) * self.target_height
        )

        # Update height sensor readings
        height = self.robot.data.body_pos_w.torch[:, self._root_id, 2] - self._height_sensor.data.ray_hits_w[..., 2]
        self.base_height = torch.clamp(torch.mean(height, dim=-1), min=0.0, max=5.0)  # clamp to prevent inf values

        # Call parent to handle velocity filtering and other updates
        super()._update_command()

        # Apply walking height constraint (after filtering)
        if not self.random_height_during_walking:
            non_standing_env_ids = (~self.is_standing_env).nonzero(as_tuple=False).flatten()
            # Update both filtered and unfiltered to maintain consistency
            self.target_height[non_standing_env_ids] = self.normal_walking_height
            self.target_height_unfiltered[non_standing_env_ids] = self.normal_walking_height

    def __str__(self) -> str:
        msg = "UniformVelocityBaseHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        msg += f"\tBase height: {self.cfg.ranges.base_height}"
        msg += f"\tRoot name: {self.cfg.root_name}"
        msg += f"\tDefault hight: {self.cfg.default_height}"
        return msg

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample command with strict transition rules to prevent direct walking<->squatting."""
        super()._resample_command(env_ids)

        env_ids_tensor = torch.as_tensor(env_ids, device=self.device)

        # Start everyone with default height (set unfiltered target only)
        self.target_height_unfiltered[env_ids] = self.normal_walking_height
        # Do NOT copy to filtered - let the filter handle smooth transition

        # Sample height for standing environments
        current_stand_mask = self.is_standing_env[env_ids]
        current_stand_envs_ids = env_ids_tensor[current_stand_mask]
        if current_stand_mask.any():
            sampled_height = self._sample_target_height(current_stand_mask.sum())
            self.target_height_unfiltered[current_stand_envs_ids] = sampled_height
            # Do NOT copy to filtered - let the filter handle smooth transition
        self.current_stand_normal_height, self.current_stand_squat_height, self.current_walk = self._update_state(
            self.is_standing_env, self.target_height, self.cfg.squatting_threshold
        )

        # Case 1: previous walk
        if self.prev_walk[env_ids].any():
            # Sub case 1: Current walk: No action
            # Sub case 2: Current stand at normal: no action
            # Sub case 3: Current stand at sqaut: set the squat height to the normal height
            walk_to_stand_mask = self.prev_walk[env_ids] & self.current_stand_squat_height[env_ids]
            if walk_to_stand_mask.any() and not self.random_height_during_walking:
                walk_to_stand_ids = env_ids_tensor[walk_to_stand_mask]
                # Always update unfiltered target (filtered will follow via _update_command)
                self.target_height_unfiltered[walk_to_stand_ids] = self.normal_walking_height

        # Case 2: previous stand at normal. All transitions are allowed.

        # Case 3: previous stand at squat.
        if self.prev_stand_squat_height[env_ids].any():
            # Sub case 1: Current stand at normal: no action
            # Sub case 2: Current stand at squat: no action
            # Sub case 3: Current walk: set this env to stand and resample the height
            squat_to_walk_mask = self.prev_stand_squat_height[env_ids] & self.current_walk[env_ids]
            if squat_to_walk_mask.any() and not self.random_height_during_walking:
                squat_to_walk_ids = env_ids_tensor[squat_to_walk_mask]
                self.is_standing_env[squat_to_walk_ids] = True
                sampled_height = self._sample_target_height(squat_to_walk_mask.sum())
                # Always update unfiltered target (filtered will follow via _update_command)
                self.target_height_unfiltered[squat_to_walk_ids] = sampled_height
                # Update the current state.
                self.current_stand_normal_height, self.current_stand_squat_height, self.current_walk = (
                    self._update_state(self.is_standing_env, self.target_height, self.cfg.squatting_threshold)
                )

        # Make sure all the standing envs have zero velocity command.
        standing_envs_mask = self.is_standing_env[env_ids]
        standing_envs_ids = env_ids_tensor[standing_envs_mask]
        # Set both target and filtered to zero for standing envs (no motion desired)
        self.vel_command_target_b[standing_envs_ids, :] = 0.0
        self.vel_command_b[standing_envs_ids, :] = 0.0

        # Check whether we need to randomize the height for walking envs.
        if self.random_height_during_walking:
            current_walk_mask = self.current_walk[env_ids]
            if current_walk_mask.any():
                current_walk_ids = env_ids_tensor[current_walk_mask]
                sampled_height = self._sample_target_height(current_walk_mask.sum())
                # Update unfiltered target only - let filter handle smooth transition
                self.target_height_unfiltered[current_walk_ids] = sampled_height

                crouching_envs = self.target_height[current_walk_ids] < self.cfg.min_walk_height
                if crouching_envs.any():
                    crouching_env_ids = current_walk_ids[crouching_envs]
                    scale = 1 - (self.cfg.min_walk_height - self.target_height[crouching_env_ids]) / (  # type: ignore[call-overload]
                        self.cfg.min_walk_height - self.cfg.ranges.base_height[0]
                    )
                    self.vel_command_b[crouching_env_ids, :] *= scale.unsqueeze(1)  # type: ignore[call-overload]

        # Update tracking for next resample
        self.prev_stand_normal_height = self.current_stand_normal_height
        self.prev_stand_squat_height = self.current_stand_squat_height
        self.prev_walk = self.current_walk

    def _update_metrics(self) -> None:
        super()._update_metrics()
        if self.random_height_during_walking:
            self.error_height_log = torch.abs(self.target_height - self.base_height).abs().mean().item()
        else:
            self.error_height_log = (
                torch.abs(self.target_height - self.base_height)[self.is_standing_env].abs().mean().item()
            )

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame and base height. Shape is (num_envs, 4)."""
        return torch.cat([self.vel_command_b, self.target_height.unsqueeze(-1)], dim=-1)

    # helpers
    def _update_state(
        self, is_standing_env: torch.Tensor, target_height: torch.Tensor, squat_height_threshold: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Based on the given sampled standing env and target height, compute the state of the robot."""
        current_stand_normal_height = is_standing_env & (target_height >= squat_height_threshold)
        current_stand_squat_height = is_standing_env & (target_height < squat_height_threshold)
        current_walk = ~is_standing_env
        return current_stand_normal_height, current_stand_squat_height, current_walk

    def _sample_target_height(self, num_envs_to_sample: int) -> torch.Tensor:
        target_height = torch.zeros(num_envs_to_sample, device=self.device)
        # Create a 1 to num_envs_to_sample tensor
        local_ids = torch.arange(num_envs_to_sample, device=self.device)
        """Sample the height for the given envs."""
        if self.cfg.bias_height_randomization:
            # Biased sampling
            use_lower = torch.rand(num_envs_to_sample, device=self.device) < self.cfg.lower_height_bias

            if use_lower.any():
                lower_ids = local_ids[use_lower]
                target_height[lower_ids] = torch.empty(use_lower.sum(), device=self.device).uniform_(
                    self.cfg.ranges.base_height[0], self.cfg.sample_middle_height
                )

            if (~use_lower).any():
                upper_ids = local_ids[~use_lower]
                target_height[upper_ids] = torch.empty((~use_lower).sum(), device=self.device).uniform_(
                    self.cfg.sample_middle_height, self.cfg.ranges.base_height[1]
                )
        else:
            # Uniform sampling
            target_height = torch.empty(num_envs_to_sample, device=self.device).uniform_(*self.cfg.ranges.base_height)

        return target_height

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids=env_ids)
        extras["error_height"] = self.error_height_log

        # Reset height filter state: copy unfiltered targets directly to filtered
        # This is correct for reset - we want to start fresh with no history from previous episode
        if env_ids is not None:
            self.target_height[env_ids] = self.target_height_unfiltered[env_ids].clone()
            # Resample height filter coefficients for reset environments (will be 1.0 if no config)
            self._sample_filter_coefficients(env_ids)

        # Reset the previous standing state for the reset envs
        if env_ids is not None:
            self.prev_stand_normal_height[env_ids] = True
            self.prev_stand_squat_height[env_ids] = False
            self.prev_walk[env_ids] = False
        else:
            self.prev_stand_normal_height[:] = True
            self.prev_stand_squat_height[:] = False
            self.prev_walk[:] = False

        return extras


class UniformVelocityGaitBaseHeightCommand(UniformVelocityBaseHeightCommand):
    """Velocity height command with gait phase."""

    cfg: UniformVelocityGaitBaseHeightCommandCfg

    def __init__(self, cfg: UniformVelocityGaitBaseHeightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.gait_frequency = torch.zeros(env.num_envs, device=self.device)
        self.gait_process = torch.zeros(env.num_envs, device=self.device)
        # the gait process is the time since the cycle started

        self.gait_cycle = torch.zeros(env.num_envs, 2, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        self.gait_frequency[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.gait_frequency_range
        )
        self.gait_process[env_ids] = 0

        # standing envs get frequency 0
        null_velocity = (self.vel_command_b[:, :3] == 0).all(dim=1)
        self.gait_frequency[self.is_standing_env | null_velocity] = 0.0

    def _update_command(self) -> None:
        super()._update_command()
        self.gait_process = torch.fmod(self.gait_process + self._env.step_dt * self.gait_frequency, 1.0)

        self.gait_cycle[:, 0] = torch.sin(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()
        self.gait_cycle[:, 1] = torch.cos(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()
