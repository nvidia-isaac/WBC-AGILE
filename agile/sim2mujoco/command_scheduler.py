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

"""Command schedulers for sim2mujoco evaluation.

Provides two scheduler types that share the same ``.update(dt)`` interface:
  - :class:`Sim2MuJoCoCommandScheduler`: Deterministic commands from EvalConfig YAML.
  - :class:`RandomCommandScheduler`: Uniformly-random commands resampled at a fixed interval,
    with per-dimension control over which fields are randomized.
"""

from __future__ import annotations

import random

from agile.algorithms.evaluation.eval_config import EvalConfig
from agile.sim2mujoco.commands import CommandManager

# Mapping from user-facing CLI names to CommandManager internals.
FIELD_SPECS: dict[str, tuple[str, str]] = {
    "vx": ("linear_x", "linear_x_range"),
    "vy": ("linear_y", "linear_y_range"),
    "wz": ("angular_z", "angular_z_range"),
    "height": ("height", "height_range"),
}


class Sim2MuJoCoCommandScheduler:
    """Applies time-based command schedule from EvalConfig to CommandManager.

    Sim2MuJoCo has a single environment (env_id=0). The scheduler uses the schedule
    or sweep from the first environment in the eval config.

    **Supported command format** (same as IsaacLab eval_config):
        commands:
          base_velocity:
            lin_vel_x: 0.2
            lin_vel_y: 0.0
            ang_vel_z: 0.0
            base_height: 0.72
    """

    def __init__(
        self,
        eval_config: EvalConfig,
        command_manager: CommandManager,
        duration: float,
        command_dim: int = 4,
        verbose: bool = True,
    ):
        """Initialize the scheduler.

        Args:
            eval_config: Evaluation config loaded from YAML.
            command_manager: CommandManager to inject commands into.
            duration: Total run duration in seconds (for sweep step generation).
            command_dim: 3 for velocity-only, 4 for velocity+height.
            verbose: Whether to print schedule application info.
        """
        self.command_manager = command_manager
        self.command_dim = command_dim
        self.verbose = verbose

        env_cfg = eval_config.get_env_config(0)
        if env_cfg is None:
            raise ValueError(
                "EvalConfig must have an environment with env_ids: [0] for sim2mujoco. "
                f"Available env_ids: {[e.env_ids for e in eval_config.environments]}"
            )

        self.schedule = env_cfg.get_full_schedule(duration)
        self.applied_indices: set[int] = set()
        self.current_time = 0.0
        self._last_printed_cmd: tuple | None = None

        if self.verbose:
            self._print_schedule_summary(duration)

    def _print_schedule_summary(self, duration: float) -> None:
        """Print summary of loaded schedule."""
        print("\n" + "=" * 80)
        print("Sim2MuJoCo Evaluation Schedule")
        print("=" * 80)
        print(f"Duration: {duration}s")
        print(f"Schedule steps: {len(self.schedule)}")
        if self.schedule:
            print("Preview:")
            for _i, step in enumerate(self.schedule[:5]):
                if step.commands and "base_velocity" in step.commands:
                    cmd = step.commands["base_velocity"]
                    parts = [f"{k}={v}" for k, v in cmd.items()]
                    print(f"  t={step.time:.1f}s: {', '.join(parts)}")
            if len(self.schedule) > 5:
                print(f"  ... and {len(self.schedule) - 5} more steps")
        print("=" * 80 + "\n")

    def update(self, dt: float) -> None:
        """Update scheduler and apply any pending commands.

        Call this every control step before policy inference.

        Args:
            dt: Control timestep in seconds.
        """
        self.current_time += dt

        for i, step in enumerate(self.schedule):
            if i in self.applied_indices:
                continue
            if self.current_time >= step.time and step.commands:
                self._apply_step(step, i)
                self.applied_indices.add(i)

    def _apply_step(self, step, step_index: int) -> None:
        """Apply a schedule step to CommandManager."""
        if "base_velocity" not in step.commands:
            return

        cmd = step.commands["base_velocity"]
        linear_x = float(cmd.get("lin_vel_x", 0.0))
        linear_y = float(cmd.get("lin_vel_y", 0.0))
        angular_z = float(cmd.get("ang_vel_z", 0.0))
        height = float(cmd["base_height"]) if "base_height" in cmd else None

        self.command_manager.set_command(
            linear_x=linear_x,
            linear_y=linear_y,
            angular_z=angular_z,
            height=height,
        )

        current_cmd = (linear_x, linear_y, angular_z, height)
        if current_cmd != self._last_printed_cmd:
            parts = [f"vx={linear_x:.2f}", f"vy={linear_y:.2f}", f"wz={angular_z:.2f}"]
            if height is not None:
                parts.append(f"h={height:.2f}")
            print(f"[Command] t={self.current_time:.2f}s | {', '.join(parts)}")
            self._last_printed_cmd = current_cmd


class RandomCommandScheduler:
    """Resamples commands uniformly at random at a fixed interval.

    Only the dimensions listed in *randomize_fields* are randomized; the rest
    stay at the :class:`CommandManager` defaults.  This enables apple-to-apple
    comparison with deterministic sweeps that vary one dimension at a time.

    Uses the same ``.update(dt)`` interface as :class:`Sim2MuJoCoCommandScheduler`
    so the two are interchangeable in the evaluation loop.

    Args:
        command_manager: Target :class:`CommandManager` to inject commands into.
        randomize_fields: Which dimensions to randomize.  Each element must be
            one of ``"vx"``, ``"vy"``, ``"wz"``, ``"height"``, or ``"all"``
            (expands to all four).
        interval: Seconds between resamples.
        seed: RNG seed for reproducibility (``None`` = non-deterministic).
        verbose: Print each new command to stdout.
    """

    def __init__(
        self,
        command_manager: CommandManager,
        randomize_fields: list[str],
        interval: float = 2.0,
        seed: int | None = None,
        verbose: bool = True,
    ):
        self.command_manager = command_manager
        self.interval = interval
        self.verbose = verbose
        self.current_time = 0.0
        self._next_resample = 0.0
        self._rng = random.Random(seed)

        # Expand "all" and validate field names.
        if "all" in randomize_fields:
            self._fields = list(FIELD_SPECS.keys())
        else:
            unknown = set(randomize_fields) - set(FIELD_SPECS.keys())
            if unknown:
                raise ValueError(
                    f"Unknown randomize fields: {unknown}. Valid options: {list(FIELD_SPECS.keys())} or 'all'"
                )
            self._fields = list(randomize_fields)

        # Read ranges from CommandManager.
        self._ranges: dict[str, tuple[float, float]] = {}
        for field in self._fields:
            _, range_attr = FIELD_SPECS[field]
            self._ranges[field] = getattr(command_manager, range_attr)

        # Defaults for non-randomized dimensions (read once from CommandManager).
        cm_defaults = command_manager.get_defaults()
        self._defaults = {
            "vx": cm_defaults["linear_x"],
            "vy": cm_defaults["linear_y"],
            "wz": cm_defaults["angular_z"],
            "height": cm_defaults["height"],
        }

        self._print_summary(seed)
        # Apply initial command immediately.
        self._resample()
        self._next_resample = self.interval

    def _print_summary(self, seed: int | None) -> None:
        print("\n" + "=" * 80)
        print("Random Command Scheduler")
        print("=" * 80)
        print(f"Randomized fields: {self._fields}")
        for field in self._fields:
            lo, hi = self._ranges[field]
            print(f"  {field}: [{lo:.2f}, {hi:.2f}]")
        fixed = [f for f in FIELD_SPECS if f not in self._fields]
        if fixed:
            print(f"Fixed fields: {fixed}")
            for f in fixed:
                print(f"  {f}: {self._defaults[f]:.2f}")
        print(f"Resample interval: {self.interval}s")
        print(f"Seed: {seed}")
        print("=" * 80 + "\n")

    def update(self, dt: float) -> None:
        """Advance time and resample if the interval has elapsed."""
        self.current_time += dt
        if self.current_time >= self._next_resample:
            self._resample()
            self._next_resample += self.interval

    def _resample(self) -> None:
        """Draw new random values for the active fields and set the command."""
        values: dict[str, float] = {}
        for field in FIELD_SPECS:
            if field in self._fields:
                lo, hi = self._ranges[field]
                values[field] = self._rng.uniform(lo, hi)
            else:
                values[field] = self._defaults[field]

        self.command_manager.set_command(
            linear_x=values["vx"],
            linear_y=values["vy"],
            angular_z=values["wz"],
            height=values["height"],
        )

        if self.verbose:
            parts = []
            for field in FIELD_SPECS:
                marker = "*" if field in self._fields else " "
                parts.append(f"{field}={values[field]:+.2f}{marker}")
            print(f"[Random] t={self.current_time:.2f}s | {', '.join(parts)}")
