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

"""Random command scheduler for Isaac Lab evaluation.

Resamples velocity (+height) commands uniformly at random at a fixed interval,
using the same ``.update(dt)`` / ``.reapply_commands()`` interface as
:class:`VelocityHeightScheduler` so the two are interchangeable in the eval loop.
"""

from __future__ import annotations

import random

import torch

# User-facing field names and the order they appear in the command tensor.
FIELD_NAMES = ["lin_vel_x", "lin_vel_y", "ang_vel_z", "base_height"]


class RandomCommandScheduler:
    """Randomly resamples commands at a fixed interval for all Isaac Lab environments.

    Only the dimensions listed in *randomize_fields* are randomized; the rest
    stay at mid-range defaults.  This enables controlled experiments that vary
    one dimension at a time.

    Implements the same interface as :class:`VelocityHeightScheduler`
    (``update``, ``reapply_commands``, ``reset``, ``active_commands``) so it
    can be used as a drop-in replacement in ``eval.py``.

    Args:
        env: Isaac Lab environment instance (may be wrapped).
        randomize_fields: Which dimensions to randomize.  Each element must be
            one of ``"lin_vel_x"``, ``"lin_vel_y"``, ``"ang_vel_z"``,
            ``"base_height"``, or ``"all"`` (expands to every detected field).
        interval: Seconds between resamples.
        seed: RNG seed for reproducibility (``None`` = non-deterministic).
        verbose: Print each new command to stdout.
    """

    def __init__(
        self,
        env,
        randomize_fields: list[str],
        interval: float = 2.0,
        seed: int | None = None,
        verbose: bool = True,
    ):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.device = self.env.device
        self.num_envs = self.env.num_envs
        self.interval = interval
        self.verbose = verbose
        self.current_time = 0.0
        self._next_resample = 0.0
        self._rng = random.Random(seed)

        # Detect available command fields and their ranges from env config.
        self.command_ranges = self._extract_command_ranges()
        self._ordered_fields = [f for f in FIELD_NAMES if f in self.command_ranges]

        # Expand "all" and validate field names.
        valid_fields = set(self.command_ranges.keys())
        if "all" in randomize_fields:
            self._randomize_fields = list(valid_fields)
        else:
            unknown = set(randomize_fields) - valid_fields
            if unknown:
                raise ValueError(
                    f"Unknown randomize fields: {unknown}. Valid options for this task: {sorted(valid_fields)} or 'all'"
                )
            self._randomize_fields = list(randomize_fields)

        # Mid-range defaults for non-randomized dimensions.
        self._defaults: dict[str, float] = {}
        for field, (lo, hi) in self.command_ranges.items():
            self._defaults[field] = (lo + hi) / 2.0

        # active_commands mirrors VelocityHeightScheduler: env_id -> tensor | None
        self.active_commands: dict[int, torch.Tensor | None] = dict.fromkeys(range(self.num_envs))

        if self.verbose:
            self._print_summary(seed)

        # Apply initial commands immediately.
        self._resample()
        self._next_resample = self.interval

    # ------------------------------------------------------------------
    # Command range extraction (shared pattern with VelocityHeightScheduler)
    # ------------------------------------------------------------------

    def _extract_command_ranges(self) -> dict[str, tuple[float, float]]:
        """Extract min/max command ranges from environment configuration."""
        try:
            cfg = self.env.cfg.commands.base_velocity.ranges
        except AttributeError as e:
            raise RuntimeError(
                "Failed to extract command ranges from environment config. "
                "Ensure the environment has 'cfg.commands.base_velocity.ranges' defined. "
                f"Original error: {e}"
            ) from e

        command_ranges: dict[str, tuple[float, float]] = {}
        for field_name in FIELD_NAMES:
            if hasattr(cfg, field_name):
                value = getattr(cfg, field_name)
                if value is not None and isinstance(value, tuple | list) and len(value) == 2:
                    command_ranges[field_name] = (value[0], value[1])

        if not command_ranges:
            raise RuntimeError("No valid command fields detected from environment config")
        return command_ranges

    # ------------------------------------------------------------------
    # Public interface (matches VelocityHeightScheduler)
    # ------------------------------------------------------------------

    def reset(self, env_ids: list[int] | None = None):
        """Reset scheduler time and tracking.

        Args:
            env_ids: Specific environment IDs to reset. If None, resets all.
        """
        if env_ids is None:
            self.current_time = 0.0
            self._resample()
            self._next_resample = self.interval
        # Per-env reset is a no-op because all envs share the same random schedule.

    def update(self, dt: float):
        """Advance time and resample if the interval has elapsed.

        Args:
            dt: Control timestep in seconds.
        """
        self.current_time += dt
        if self.current_time >= self._next_resample:
            self._resample()
            self._next_resample += self.interval

    def reapply_commands(self):
        """Reapply active commands after env.step() and recompute observations.

        Mirrors :meth:`VelocityHeightScheduler.reapply_commands`.
        """
        for env_id, command_tensor in self.active_commands.items():
            if command_tensor is not None:
                self._set_command(env_id, command_tensor)

        # Recompute observations to reflect the corrected commands.
        self.env.obs_buf = self.env.observation_manager.compute(update_history=False)

        # Re-apply after observation computation (observation_manager may mutate commands).
        for env_id, command_tensor in self.active_commands.items():
            if command_tensor is not None:
                self._set_command(env_id, command_tensor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resample(self):
        """Draw new random values for the active fields and broadcast to all envs."""
        values: dict[str, float] = {}
        for field in self._ordered_fields:
            if field in self._randomize_fields:
                lo, hi = self.command_ranges[field]
                values[field] = self._rng.uniform(lo, hi)
            else:
                values[field] = self._defaults[field]

        command_tensor = torch.tensor(
            [values[f] for f in self._ordered_fields],
            dtype=torch.float32,
            device=self.device,
        )

        for env_id in range(self.num_envs):
            self.active_commands[env_id] = command_tensor.clone()
            self._set_command(env_id, command_tensor)

        if self.verbose:
            parts = []
            for field in self._ordered_fields:
                marker = "*" if field in self._randomize_fields else " "
                parts.append(f"{field}={values[field]:+.2f}{marker}")
            print(f"[RandomCmd] t={self.current_time:.2f}s | {', '.join(parts)}")

    def _set_command(self, env_id: int, command_tensor: torch.Tensor):
        """Set command for a specific environment (same logic as VelocityHeightScheduler)."""
        cmd_manager = self.env.command_manager
        base_vel_term = cmd_manager.get_term("base_velocity")

        base_vel_term.vel_command_b[env_id, 0] = command_tensor[0]  # lin_vel_x
        base_vel_term.vel_command_b[env_id, 1] = command_tensor[1]  # lin_vel_y
        base_vel_term.vel_command_b[env_id, 2] = command_tensor[2]  # ang_vel_z

        if len(command_tensor) >= 4 and hasattr(base_vel_term, "target_height"):
            base_vel_term.target_height[env_id] = command_tensor[3]

        if hasattr(base_vel_term, "is_heading_env"):
            base_vel_term.is_heading_env[env_id] = False

        if hasattr(base_vel_term, "time_left"):
            base_vel_term.time_left[env_id] = 1000.0

    def _print_summary(self, seed: int | None):
        print("\n" + "=" * 80)
        print("Isaac Lab Random Command Scheduler")
        print("=" * 80)
        print(f"Randomized fields: {self._randomize_fields}")
        for field in self._randomize_fields:
            lo, hi = self.command_ranges[field]
            print(f"  {field}: [{lo:.2f}, {hi:.2f}]")
        fixed = [f for f in self._ordered_fields if f not in self._randomize_fields]
        if fixed:
            print(f"Fixed fields (mid-range): {fixed}")
            for f in fixed:
                print(f"  {f}: {self._defaults[f]:.2f}")
        print(f"Resample interval: {self.interval}s")
        print(f"Seed: {seed}")
        print(f"Num environments: {self.num_envs}")
        print("=" * 80 + "\n")
