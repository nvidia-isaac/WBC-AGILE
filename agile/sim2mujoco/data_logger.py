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

"""Data logger for sim2mujoco evaluation.

Records per-step simulation data to parquet files compatible with the existing
plotting utilities in agile.algorithms.evaluation.plotting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from agile.sim2mujoco.simulation import JointCommand, SimState

if TYPE_CHECKING:
    from agile.sim2mujoco.command_provider import CommandProvider


class Sim2MuJoCoDataLogger:
    """Records per-step simulation data for offline analysis.

    Produces output compatible with :func:`agile.algorithms.evaluation.plotting.load_episode`
    and :func:`agile.algorithms.evaluation.plotting.load_metadata`.
    """

    def __init__(
        self,
        output_dir: str | Path,
        config: dict,
        joint_names: list[str],
        control_dt: float,
        provenance: dict | None = None,
        command_provider: CommandProvider | None = None,
    ):
        """Initialize the data logger.

        Args:
            output_dir: Directory where ``trajectories/`` subfolder will be created.
            config: Full YAML configuration dictionary (used for metadata extraction).
            joint_names: Joint names from the MuJoCo simulation.
            control_dt: Control timestep in seconds (physics_dt * decimation).
            provenance: Optional dict with paths used to produce this run
                        (e.g. checkpoint, config, eval_config).
            command_provider: Optional :class:`CommandProvider` that supplies
                command metadata (type, dim, names).  When ``None``, no command
                columns are recorded.
        """
        self.output_dir = Path(output_dir)
        self.trajectories_dir = self.output_dir / "trajectories"
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.joint_names = joint_names
        self.num_joints = len(joint_names)
        self.control_dt = control_dt
        self.provenance = provenance or {}

        if command_provider is not None:
            self.command_type = command_provider.command_type
            self.command_names = command_provider.command_names
            self.command_dim = command_provider.command_dim
        else:
            self.command_type = "none"
            self.command_names: list[str] = []
            self.command_dim = 0

        self._prev_vel: np.ndarray | None = None
        self._rows: list[dict[str, float]] = []
        self._step_idx = 0

        self._save_metadata()
        print(f"Sim2MuJoCoDataLogger: saving to {self.trajectories_dir}")
        print(f"  Command type: {self.command_type} (dim={self.command_dim})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def has_data(self) -> bool:
        """Whether there is buffered data that hasn't been saved yet."""
        return len(self._rows) > 0

    def record_step(
        self,
        sim_state: SimState,
        joint_cmd: JointCommand,
        actions: torch.Tensor,
        commands: torch.Tensor | None = None,
        episode_id: int = 0,
    ) -> None:
        """Record one control step of data.

        Args:
            sim_state: Simulation state *after* the physics steps.  Joint torques
                       are read from ``sim_state.joint_effort``.
            joint_cmd: Desired joint command sent to the simulation.
            actions: Raw policy output.
            commands: Command tensor from :meth:`CommandProvider.get_commands`.
                      Pass ``None`` when no command provider is active.
            episode_id: Current episode identifier (for multi-episode runs).
        """
        row: dict[str, float] = {}

        # Metadata columns (compatible with agile.algorithms.evaluation.plotting)
        row["episode_id"] = episode_id
        row["env_id"] = 0
        row["frame_idx"] = self._step_idx
        row["timestep"] = self._step_idx * self.control_dt
        row["is_success"] = 1.0

        # Joint position, velocity, acceleration
        jp = sim_state.joint_pos.detach().cpu().numpy()
        jv = sim_state.joint_vel.detach().cpu().numpy()

        for i in range(self.num_joints):
            row[f"joint_pos_{i}"] = float(jp[i])
            row[f"joint_vel_{i}"] = float(jv[i])

        if self._prev_vel is not None:
            acc = (jv - self._prev_vel) / self.control_dt
            for i in range(self.num_joints):
                row[f"joint_acc_{i}"] = float(acc[i])
        else:
            for i in range(self.num_joints):
                row[f"joint_acc_{i}"] = 0.0
        self._prev_vel = jv.copy()

        # Joint torques
        if sim_state.joint_effort is not None:
            jt = sim_state.joint_effort.detach().cpu().numpy()
            for i in range(self.num_joints):
                row[f"joint_torque_{i}"] = float(jt[i])

        # Desired joint positions (from action processor)
        jp_des = joint_cmd.position.detach().cpu().numpy()
        for i in range(self.num_joints):
            row[f"joint_pos_des_{i}"] = float(jp_des[i])

        # Root state
        rp = sim_state.root_pos.detach().cpu().numpy()
        rlv = sim_state.root_lin_vel.detach().cpu().numpy()
        rav = sim_state.root_ang_vel.detach().cpu().numpy()
        for i in range(3):
            row[f"root_pos_{i}"] = float(rp[i])
            row[f"root_lin_vel_robot_{i}"] = float(rlv[i])
            row[f"root_ang_vel_robot_{i}"] = float(rav[i])

        # Commands (only log up to command_dim to match policy semantics)
        if commands is not None and self.command_dim > 0:
            cmd = commands.detach().cpu().numpy()
            n = min(len(cmd), self.command_dim)
            for i in range(n):
                row[f"commands_{i}"] = float(cmd[i])

        # Raw actions
        act = actions.detach().cpu().numpy()
        for i in range(len(act)):
            row[f"actions_{i}"] = float(act[i])

        self._rows.append(row)
        self._step_idx += 1

    def save_episode(self, episode_id: int) -> Path | None:
        """Write collected data to ``episode_{episode_id:03d}.parquet`` and return its path.

        Returns ``None`` if there is no data to save.
        """
        if not self._rows:
            print(f"No data to save for episode {episode_id}")
            return None
        df = pd.DataFrame(self._rows)
        filepath = self.trajectories_dir / f"episode_{episode_id:03d}.parquet"
        df.to_parquet(filepath, compression="snappy", index=False)
        print(f"Saved {len(self._rows)} steps to {filepath}")
        return filepath

    def reset(self) -> None:
        """Clear buffers for the next episode. Call after save_episode when episode boundary is reached."""
        self._rows = []
        self._step_idx = 0
        self._prev_vel = None

    def save(self) -> Path | None:
        """Write collected data to ``episode_000.parquet`` and return its path. Alias for single-episode runs."""
        return self.save_episode(0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_limits_to_sim_order(self, limits: list, config_joint_names: list[str]) -> list:
        """Map limits from config (YAML) joint order to simulation (MuJoCo) joint order."""
        if not limits or not config_joint_names:
            return []
        is_pos_limits = isinstance(limits[0], list | tuple)
        result = []
        for sim_joint_name in self.joint_names:
            if sim_joint_name in config_joint_names:
                cfg_idx = config_joint_names.index(sim_joint_name)
                if cfg_idx < len(limits):
                    result.append(limits[cfg_idx])
                else:
                    result.append([-float("inf"), float("inf")] if is_pos_limits else 1000.0)
            else:
                result.append([-float("inf"), float("inf")] if is_pos_limits else 1000.0)
        return result

    def _save_metadata(self) -> None:
        """Extract metadata from config and write ``metadata.json``."""
        robot_cfg = self.config.get("articulations", {}).get("robot", {})
        scene_cfg = self.config.get("scene", {})

        physics_dt = scene_cfg.get("physics_dt", scene_cfg.get("dt", 0.005))
        config_joint_names = robot_cfg.get("joint_names", [])
        raw_pos_limits = robot_cfg.get("default_joint_pos_limits", [])
        raw_vel_limits = robot_cfg.get("soft_joint_vel_limits", [])

        joint_pos_limits = self._map_limits_to_sim_order(raw_pos_limits, config_joint_names)
        joint_vel_limits = self._map_limits_to_sim_order(raw_vel_limits, config_joint_names)

        metadata: dict = {
            "physics_dt": physics_dt,
            "control_dt": self.control_dt,
            "num_joints": self.num_joints,
            "joint_names": self.joint_names,
            "joint_pos_limits": joint_pos_limits,
            "joint_vel_limits": joint_vel_limits,
            "command_type": self.command_type,
            "command_names": self.command_names,
            "command_dim": self.command_dim,
            "task_name": "sim2mujoco",
            "noise_scale": self.provenance.get("noise_scale", 0.0),
            "noise_seed": self.provenance.get("noise_seed"),
            "provenance": self.provenance,
        }

        metadata_path = self.trajectories_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
