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

"""Configuration loading for deterministic evaluation scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ScheduleStep:
    """A single time-based update in the evaluation schedule."""

    time: float
    commands: dict | None = None
    terrain: dict | None = None
    events: dict | None = None
    physics: dict | None = None

    def __post_init__(self):
        """Validate that at least one override is specified."""
        if not any([self.commands, self.terrain, self.events, self.physics]):
            raise ValueError(f"ScheduleStep at time={self.time} has no overrides specified")


@dataclass
class SweepConfig:
    """Configuration for sweep-based command generation."""

    interval: float
    commands: dict | None = None
    terrain: dict | None = None
    events: dict | None = None

    def to_schedule(self, max_time: float) -> list[ScheduleStep]:
        """Convert sweep config to explicit schedule steps.

        Args:
            max_time: Maximum episode time to generate schedule for

        Returns:
            List of ScheduleStep objects
        """
        schedule = []

        # Extract sweep values (lists) and fixed values
        sweep_fields = {}
        fixed_values = {}

        if self.commands:
            for cmd_type, cmd_dict in self.commands.items():
                sweep_fields[cmd_type] = {}
                fixed_values[cmd_type] = {}

                for field, value in cmd_dict.items():
                    if isinstance(value, list):
                        sweep_fields[cmd_type][field] = value
                    else:
                        fixed_values[cmd_type][field] = value

        # Generate schedule steps
        current_time = 0.0
        step_idx = 0

        while current_time <= max_time:
            step_commands = {}

            # Build command dict for this step
            for cmd_type in sweep_fields:
                step_commands[cmd_type] = fixed_values[cmd_type].copy()

                # Add swept values (cycle through list)
                for field, value_list in sweep_fields[cmd_type].items():
                    idx = step_idx % len(value_list)
                    step_commands[cmd_type][field] = value_list[idx]

            schedule.append(
                ScheduleStep(
                    time=current_time,
                    commands=step_commands if step_commands else None,
                    terrain=self.terrain,
                    events=self.events,
                )
            )

            current_time += self.interval
            step_idx += 1

        return schedule


@dataclass
class EnvConfig:
    """Configuration for a specific environment or group of environments."""

    env_ids: list[int]
    name: str
    schedule: list[ScheduleStep] = field(default_factory=list)
    sweep: SweepConfig | None = None

    def get_full_schedule(self, max_time: float) -> list[ScheduleStep]:
        """Get complete schedule including expanded sweep.

        Args:
            max_time: Maximum episode time

        Returns:
            Combined schedule from both explicit steps and sweep
        """
        full_schedule = list(self.schedule)

        # Add sweep-generated steps
        if self.sweep:
            sweep_steps = self.sweep.to_schedule(max_time)
            full_schedule.extend(sweep_steps)

        # Sort by time
        full_schedule.sort(key=lambda s: s.time)

        return full_schedule


@dataclass
class EventOverrides:
    """Event override configuration for evaluation."""

    disable_all: bool = False
    disable_interval_events: bool = False
    disable_specific: list[str] = field(default_factory=list)


@dataclass
class EnvOverrides:
    """Environment configuration overrides for evaluation."""

    episode_length_s: float | None = None  # Override episode length
    num_envs: int | None = None  # Override number of environments
    events: EventOverrides | None = None
    # Future additions:
    # observations: ObservationOverrides | None = None
    # physics: PhysicsOverrides | None = None


@dataclass
class EvalConfig:
    """Complete evaluation scenario configuration."""

    task_name: str
    num_envs: int
    episode_length_s: float
    num_episodes: int = 1
    global_overrides: dict = field(default_factory=dict)
    environments: list[EnvConfig] = field(default_factory=list)
    env_overrides: EnvOverrides | None = None

    def __post_init__(self):
        """Validate configuration."""
        # Check that all env_ids are unique and cover [0, num_envs)
        assigned_ids = set()
        for env_cfg in self.environments:
            for env_id in env_cfg.env_ids:
                if env_id in assigned_ids:
                    raise ValueError(f"Environment ID {env_id} assigned to multiple configurations")
                if env_id >= self.num_envs:
                    raise ValueError(f"Environment ID {env_id} exceeds num_envs={self.num_envs}")
                assigned_ids.add(env_id)

        # Warn about unassigned environments
        unassigned = set(range(self.num_envs)) - assigned_ids
        if unassigned:
            print(f"[WARNING] Environments {sorted(unassigned)} have no schedule and will use random commands")

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        """Load evaluation config from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            EvalConfig instance

        Example YAML:
            evaluation:
              task_name: "MyTask-v0"
              num_envs: 4
              episode_length_s: 50.0
              num_episodes: 5
              environments:
                - env_ids: [0]
                  name: "x_velocity_test"
                  schedule:
                    - time: 0.0
                      commands:
                        base_velocity:
                          lin_vel_x: 1.0
                          lin_vel_y: 0.0
                          ang_vel_z: 0.0
                          base_height: 0.75
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation config not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if "evaluation" not in data:
            raise ValueError("YAML must have top-level 'evaluation' key")

        eval_data = data["evaluation"]

        # Parse environment configurations
        environments = []
        for env_data in eval_data.get("environments", []):
            # Parse schedule steps
            schedule = []
            for step_data in env_data.get("schedule", []):
                schedule.append(
                    ScheduleStep(
                        time=step_data["time"],
                        commands=step_data.get("commands"),
                        terrain=step_data.get("terrain"),
                        events=step_data.get("events"),
                        physics=step_data.get("physics"),
                    )
                )

            # Parse sweep config if present
            sweep = None
            if "sweep" in env_data:
                sweep_data = env_data["sweep"]
                sweep = SweepConfig(
                    interval=sweep_data["interval"],
                    commands=sweep_data.get("commands"),
                    terrain=sweep_data.get("terrain"),
                    events=sweep_data.get("events"),
                )

            environments.append(
                EnvConfig(
                    env_ids=env_data["env_ids"],
                    name=env_data["name"],
                    schedule=schedule,
                    sweep=sweep,
                )
            )

        # Parse env_overrides if present
        env_overrides = None
        if "env_overrides" in eval_data:
            overrides_data = eval_data["env_overrides"]

            # Parse event overrides
            event_overrides = None
            if "events" in overrides_data:
                events_data = overrides_data["events"]
                event_overrides = EventOverrides(
                    disable_all=events_data.get("disable_all", False),
                    disable_interval_events=events_data.get("disable_interval_events", False),
                    disable_specific=events_data.get("disable_specific", []),
                )

            # Episode length and num_envs can be in env_overrides or at top level (backward compat)
            episode_length_override = overrides_data.get("episode_length_s")
            num_envs_override = overrides_data.get("num_envs")

            env_overrides = EnvOverrides(
                episode_length_s=episode_length_override,
                num_envs=num_envs_override,
                events=event_overrides,
            )

        return cls(
            task_name=eval_data["task_name"],
            num_envs=eval_data["num_envs"],
            episode_length_s=eval_data["episode_length_s"],
            num_episodes=eval_data.get("num_episodes", 1),
            global_overrides=eval_data.get("global_overrides", {}),
            environments=environments,
            env_overrides=env_overrides,
        )

    def get_env_config(self, env_id: int) -> EnvConfig | None:
        """Get configuration for specific environment ID.

        Args:
            env_id: Environment ID to look up

        Returns:
            EnvConfig if found, None otherwise
        """
        for env_cfg in self.environments:
            if env_id in env_cfg.env_ids:
                return env_cfg
        return None
