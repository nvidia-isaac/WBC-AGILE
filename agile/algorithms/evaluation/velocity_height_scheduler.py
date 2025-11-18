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

"""Time-based environment override scheduler for deterministic evaluation.

This scheduler is specifically designed for velocity+height commands (linear velocity + yaw rate + height).
For other command types, a new scheduler implementation will be needed in future versions.
"""

from __future__ import annotations

import numpy as np
import torch

from agile.algorithms.evaluation.eval_config import EvalConfig, ScheduleStep


class VelocityHeightScheduler:
    """Manages time-based command overrides during evaluation.

    This scheduler applies deterministic commands to specific environments at scheduled
    times during evaluation episodes. Automatically detects available command fields
    from the environment configuration.

    **Supported Command Type:** base_velocity
        - Velocity-only tasks: lin_vel_x, lin_vel_y, ang_vel_z (3 fields)
        - Velocity+Height tasks: lin_vel_x, lin_vel_y, ang_vel_z, base_height (4 fields)
        - Automatically adapts based on task configuration

    **Command Fields:**
        - lin_vel_x: Forward/backward velocity (m/s)
        - lin_vel_y: Lateral velocity (m/s)
        - ang_vel_z: Yaw rate (rad/s)
        - base_height: Target standing height (m) [optional, only for height-tracking tasks]

    **Not supported:** Other command types (end effector poses, joint positions, etc.)
        will require extending the scheduler. See class docstring for extension ideas.

    Note: Commands must be reapplied after env.step() because the command_manager
    resamples commands during the step, which would overwrite our scheduled values.
    """

    def __init__(self, env, eval_config: EvalConfig, verbose: bool = True):
        """Initialize command scheduler.

        Automatically detects command structure from environment (velocity or velocity+height).

        Args:
            env: IsaacLab environment instance (may be wrapped)
            eval_config: Evaluation configuration with schedules
            verbose: Whether to print schedule application info and detected command structure
        """
        # Unwrap environment if needed to access command_manager
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.config = eval_config
        self.verbose = verbose

        self.current_time = 0.0
        self.device = self.env.device

        # Extract command ranges from environment config
        self.command_ranges = self._extract_command_ranges()

        # Build schedule index: env_id -> list of (time, ScheduleStep)
        self.schedules = {}
        self._build_schedules()

        # Validate that all scheduled env_ids exist in the environment
        self._validate_env_ids()

        # Track which updates have been applied
        self.applied_steps = {env_id: set() for env_id in self.schedules.keys()}

        # Store current active commands for each scheduled environment
        # These will be reapplied after command_manager.compute() overwrites them
        self.active_commands = dict.fromkeys(self.schedules.keys())

        if self.verbose:
            self._print_schedule_summary()

    def _extract_command_ranges(self) -> dict:
        """Extract min/max command ranges dynamically from environment configuration.

        Detects available command fields automatically, supporting both:
        - Velocity-only tasks: lin_vel_x, lin_vel_y, ang_vel_z (3 fields)
        - Velocity+Height tasks: lin_vel_x, lin_vel_y, ang_vel_z, base_height (4 fields)

        Returns:
            Dictionary with command field ranges (only fields that exist in the task)

        Raises:
            RuntimeError: If command ranges cannot be extracted from environment config
        """
        try:
            cfg = self.env.cfg.commands.base_velocity.ranges

            # Dynamically extract all available command fields
            command_ranges = {}

            # Check for each possible field (exclude 'heading' as it's computed, not directly commanded)
            possible_fields = ["lin_vel_x", "lin_vel_y", "ang_vel_z", "base_height"]

            for field_name in possible_fields:
                if hasattr(cfg, field_name):
                    field_value = getattr(cfg, field_name)
                    # Check if it's a valid range (tuple/list with 2 elements)
                    if field_value is not None and isinstance(field_value, tuple | list) and len(field_value) == 2:
                        command_ranges[field_name] = (field_value[0], field_value[1])

            if not command_ranges:
                raise RuntimeError("No valid command fields detected from environment config")

            if self.verbose:
                print(f"[INFO] Detected {len(command_ranges)} command fields: {list(command_ranges.keys())}")

            return command_ranges

        except AttributeError as e:
            raise RuntimeError(
                f"Failed to extract command ranges from environment config. "
                f"Ensure the environment has 'cfg.commands.base_velocity.ranges' defined. "
                f"Original error: {e}"
            ) from e

    def _build_schedules(self):
        """Build schedule index for each environment."""
        for env_cfg in self.config.environments:
            full_schedule = env_cfg.get_full_schedule(self.config.episode_length_s)

            # Validate and clamp all commands
            for step in full_schedule:
                if step.commands:
                    self._validate_and_clamp_commands(step.commands)

            # Assign to each env_id
            for env_id in env_cfg.env_ids:
                self.schedules[env_id] = full_schedule

    def _validate_env_ids(self):
        """Validate that all scheduled env_ids actually exist in the environment.

        Raises:
            ValueError: If any env_id in the schedule is >= the actual number of environments
        """
        actual_num_envs = self.env.num_envs
        invalid_env_ids = [env_id for env_id in self.schedules.keys() if env_id >= actual_num_envs]

        if invalid_env_ids:
            raise ValueError(
                f"Evaluation config references environment IDs {invalid_env_ids} that don't exist. "
                f"Only {actual_num_envs} environments available (IDs 0-{actual_num_envs - 1}). "
                f"The config specifies num_envs={self.config.num_envs} but the simulation was created with "
                f"--num_envs {actual_num_envs}. Please use --num_envs {self.config.num_envs} or update the config."
            )

    def _validate_and_clamp_commands(self, commands: dict):
        """Validate command structure and clamp values to allowed ranges.

        Dynamically validates based on fields detected from environment.

        Args:
            commands: Command dictionary to validate and modify in-place
        """
        if "base_velocity" not in commands:
            raise ValueError("Commands must contain 'base_velocity' key")

        cmd = commands["base_velocity"]

        # Get required fields from detected command structure
        required_fields = list(self.command_ranges.keys())

        # Check all required fields present
        missing = [f for f in required_fields if f not in cmd]
        if missing:
            raise ValueError(
                f"Command spec incomplete. Missing: {missing}. Required fields for this task: {required_fields}"
            )

        # Warn about extra fields (not in task but provided in YAML)
        extra_fields = set(cmd.keys()) - set(required_fields)
        if extra_fields and self.verbose:
            print(f"[WARNING] Command contains fields not used by this task: {extra_fields}")

        # Clamp to valid ranges
        for field in required_fields:
            value = cmd[field]
            min_val, max_val = self.command_ranges[field]

            if not (min_val <= value <= max_val):
                clamped = np.clip(value, min_val, max_val)
                if self.verbose:
                    print(
                        f"[WARNING] Command {field}={value:.3f} outside range "
                        f"[{min_val:.3f}, {max_val:.3f}]. Clamped to {clamped:.3f}"
                    )
                cmd[field] = clamped

    def _print_schedule_summary(self):
        """Print summary of loaded schedules."""
        print("\n" + "=" * 80)
        print("Evaluation Schedule Summary")
        print("=" * 80)
        print(f"Episode length: {self.config.episode_length_s}s")
        print(f"Number of episodes: {self.config.num_episodes}")
        print(f"Total environments: {self.config.num_envs}")
        print(f"Scheduled environments: {len(self.schedules)}")
        print("Command ranges:")
        for field, (min_val, max_val) in self.command_ranges.items():
            print(f"  {field}: [{min_val:.2f}, {max_val:.2f}]")

        for env_id, schedule in self.schedules.items():
            env_cfg = self.config.get_env_config(env_id)
            print(f"\n  Env {env_id} ({env_cfg.name}):")
            print(f"    Schedule steps: {len(schedule)}")
            if schedule:
                print(f"    Time range: {schedule[0].time:.1f}s - {schedule[-1].time:.1f}s")
                # Show first few commands
                print("    Commands preview:")
                for _i, step in enumerate(schedule[:3]):
                    if step.commands and "base_velocity" in step.commands:
                        cmd = step.commands["base_velocity"]
                        # Dynamically format all available command fields
                        cmd_str_parts = []
                        for field in self.command_ranges.keys():
                            if field in cmd:
                                cmd_str_parts.append(f"{field}={cmd[field]:.2f}")
                        cmd_str = ", ".join(cmd_str_parts)
                        print(f"      t={step.time:.1f}s: {cmd_str}")

        print("=" * 80 + "\n")

    def reset(self, env_ids: list[int] | None = None):
        """Reset scheduler time and tracking.

        Args:
            env_ids: Specific environment IDs to reset. If None, resets all.
        """
        if env_ids is None:
            # Full reset
            self.current_time = 0.0
            self.applied_steps = {env_id: set() for env_id in self.schedules.keys()}
            if self.verbose:
                print("[VelocityHeightScheduler] Reset for new episode")
        else:
            # Partial reset for specific environments
            for env_id in env_ids:
                if env_id in self.applied_steps:
                    self.applied_steps[env_id] = set()
                    if self.verbose:
                        print(f"[VelocityHeightScheduler] Reset env {env_id} (environment terminated)")

    def update(self, dt: float):
        """Update scheduler and apply any pending overrides.

        Call this every simulation step to check if scheduled updates should be applied.

        Args:
            dt: Time step in seconds
        """
        self.current_time += dt

        # Check each scheduled environment
        for env_id, schedule in self.schedules.items():
            # Find steps that should be applied now
            for i, step in enumerate(schedule):
                # Skip if already applied
                if i in self.applied_steps[env_id]:
                    continue

                # Check if time has been reached
                if self.current_time >= step.time:
                    self._apply_step(env_id, step, step_index=i)
                    self.applied_steps[env_id].add(i)

    def _apply_step(self, env_id: int, step: ScheduleStep, step_index: int = -1):
        """Apply scheduled overrides for a specific environment.

        Args:
            env_id: Environment ID to apply overrides to
            step: Schedule step with overrides to apply
            step_index: Index of this step in the schedule (for debugging)
        """
        if self.verbose:
            env_cfg = self.config.get_env_config(env_id)
            total_steps = len(self.schedules[env_id])
            print(
                f"[VelocityHeightScheduler] t={self.current_time:.2f}s | Env {env_id} ({env_cfg.name}) | Step {step_index + 1}/{total_steps}"
            )

        # Apply command overrides
        if step.commands:
            self._apply_command_override(env_id, step.commands)

        # Apply terrain overrides
        if step.terrain:
            self._apply_terrain_override(env_id, step.terrain)

        # Apply event overrides
        if step.events:
            self._apply_event_override(env_id, step.events)

        # Apply physics overrides
        if step.physics:
            self._apply_physics_override(env_id, step.physics)

    def _apply_command_override(self, env_id: int, commands: dict):
        """Apply command overrides to specific environment.

        Args:
            env_id: Environment ID
            commands: Command dictionary with 'base_velocity' key
        """
        cmd = commands["base_velocity"]

        # Create command tensor dynamically based on available fields
        # Order matches command_ranges keys (typically: lin_vel_x, lin_vel_y, ang_vel_z, [base_height])
        command_values = []
        for field in self.command_ranges.keys():
            command_values.append(cmd[field])

        command_tensor = torch.tensor(
            command_values,
            dtype=torch.float32,
            device=self.device,
        )

        # Store this command for reapplication after env.step()
        self.active_commands[env_id] = command_tensor.clone()

        # Apply command immediately
        self._set_command(env_id, command_tensor)

        if self.verbose:
            # Dynamically format command fields for printing
            cmd_str_parts = [f"{field}={cmd[field]:.2f}" for field in self.command_ranges.keys()]
            cmd_str = ", ".join(cmd_str_parts)
            print(f"    Command: {cmd_str}")

    def _set_command(self, env_id: int, command_tensor: torch.Tensor):
        """Set command for a specific environment.

        Dynamically sets velocity commands (always 3 fields) and height (if available).

        Args:
            env_id: Environment ID
            command_tensor: Command tensor with [lin_vel_x, lin_vel_y, ang_vel_z, [base_height]]
                          Length is 3 for velocity-only, 4 for velocity+height tasks
        """
        cmd_manager = self.env.command_manager
        base_vel_term = cmd_manager.get_term("base_velocity")

        # The 'command' property is read-only (computed from vel_command_b + target_height)
        # We must set the underlying storage directly:
        # - vel_command_b: [lin_vel_x, lin_vel_y, ang_vel_z] (shape: [num_envs, 3])
        # - target_height: [base_height] (shape: [num_envs]) - only if height commands exist

        # Set velocity commands (always first 3 elements)
        base_vel_term.vel_command_b[env_id, 0] = command_tensor[0]  # lin_vel_x
        base_vel_term.vel_command_b[env_id, 1] = command_tensor[1]  # lin_vel_y
        base_vel_term.vel_command_b[env_id, 2] = command_tensor[2]  # ang_vel_z

        # Set height command if available (4th element)
        if len(command_tensor) >= 4 and hasattr(base_vel_term, "target_height"):
            base_vel_term.target_height[env_id] = command_tensor[3]  # base_height

        # CRITICAL: Disable heading control for scheduled environments
        # When heading_command is enabled, ang_vel_z is computed from a heading target
        # using a PD controller. We need to disable this to use direct ang_vel_z commands.
        if hasattr(base_vel_term, "is_heading_env"):
            # Disable heading control for this environment to allow direct ang_vel_z control
            base_vel_term.is_heading_env[env_id] = False

        # CRITICAL: Prevent command_manager from resampling this environment's commands
        # by setting time_left to a large value. This keeps our scheduled commands active.
        if hasattr(base_vel_term, "time_left"):
            base_vel_term.time_left[env_id] = 1000.0  # Large value to prevent resampling

    def reapply_commands(self):
        """Reapply all active scheduled commands and recompute observations.

        This should be called AFTER env.step() to override any command resampling
        that happened during the step. We also need to recompute observations because
        they were computed using the random commands before we restored the scheduled ones.
        """
        for env_id, command_tensor in self.active_commands.items():
            if command_tensor is not None:
                self._set_command(env_id, command_tensor)

        # Recompute observations to reflect the corrected commands
        # This is necessary because observation_manager.compute() was called inside env.step()
        # BEFORE we restored the scheduled commands
        if self.active_commands:  # Only if we have scheduled environments
            self.env.obs_buf = self.env.observation_manager.compute(update_history=False)

            # CRITICAL: _update_command() inside observation_manager.compute() may have
            # modified our commands (e.g., resetting target_height for walking envs).
            # Re-apply scheduled commands AGAIN after observation computation.
            for env_id, command_tensor in self.active_commands.items():
                if command_tensor is not None:
                    self._set_command(env_id, command_tensor)

    def _apply_terrain_override(self, env_id: int, terrain: dict):
        """Apply terrain overrides to specific environment.

        Args:
            env_id: Environment ID
            terrain: Terrain configuration dictionary
        """
        if "terrain_level" in terrain:
            level = terrain["terrain_level"]
            # Set terrain level for this environment
            if hasattr(self.env, "terrain_levels"):
                self.env.terrain_levels[env_id] = level
                if self.verbose:
                    print(f"    Terrain level: {level}")
            else:
                print("[WARNING] Environment does not support terrain levels")

    def _apply_event_override(self, env_id: int, events: dict):
        """Apply event overrides to specific environment.

        Args:
            env_id: Environment ID
            events: Event configuration dictionary
        """
        # This is a placeholder for future event override implementation
        if self.verbose:
            print(f"    Events: {events} (not yet implemented)")

    def _apply_physics_override(self, env_id: int, physics: dict):
        """Apply physics overrides to specific environment.

        Args:
            env_id: Environment ID
            physics: Physics configuration dictionary
        """
        # This is a placeholder for future physics override implementation
        if self.verbose:
            print(f"    Physics: {physics} (not yet implemented)")
