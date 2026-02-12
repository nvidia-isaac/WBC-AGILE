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

"""Pre-collected fallen state dataset for efficient stand-up task resets."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class FallenStateDatasetCfg:
    """Configuration for fallen state dataset collection."""

    num_spawns_per_level: int = 2
    """Number of times to spawn all envs and let them fall per terrain level.

    Total states collected = num_envs x num_spawns_per_level x num_terrain_levels.
    Each spawn distributes envs uniformly across all terrain types within the level.
    """

    fall_duration_s: float = 2.5
    """Duration to simulate falling with disabled joints (long enough for zero velocity)."""

    cache_enabled: bool = True
    """Whether to enable disk caching of collected states."""

    cache_dir: str = "fallen_states_cache"
    """Directory to store cached fallen states."""


@dataclass
class FallenState:
    """Container for a single fallen robot state."""

    # Root state relative to terrain origin
    root_pos_rel: torch.Tensor  # (3,) position relative to terrain origin
    root_quat: torch.Tensor  # (4,) quaternion orientation
    root_lin_vel: torch.Tensor  # (3,) linear velocity
    root_ang_vel: torch.Tensor  # (3,) angular velocity

    # Joint state
    joint_pos: torch.Tensor  # (num_joints,)
    joint_vel: torch.Tensor  # (num_joints,)


@dataclass
class FallenStateDataset:
    """Stores pre-collected fallen robot states organized by terrain level.

    This dataset collects diverse fallen poses by simulating the robot falling
    with disabled joints on different terrain levels. States are stored relative
    to terrain origins so they can be applied to any spawn location.
    """

    cfg: FallenStateDatasetCfg = field(default_factory=FallenStateDatasetCfg)

    # Storage organized by terrain level
    # Each entry is a dict with keys: root_pos_rel, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel
    _states_by_level: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    _num_terrain_levels: int = 0
    _num_joints: int = 0
    _device: str = "cpu"
    _terrain_cell_size: tuple[float, float] = (8.0, 8.0)  # Default, will be updated from terrain config

    def __post_init__(self) -> None:
        self._states_by_level = {}

    @property
    def is_collected(self) -> bool:
        """Check if states have been collected."""
        return len(self._states_by_level) > 0

    @property
    def num_terrain_levels(self) -> int:
        """Number of terrain levels with collected states."""
        return self._num_terrain_levels

    def get_num_states(self, terrain_level: int) -> int:
        """Get number of collected states for a given terrain level."""
        if terrain_level not in self._states_by_level:
            return 0
        return int(self._states_by_level[terrain_level]["root_pos_rel"].shape[0])

    def collect(self, env: ManagerBasedRLEnv, verbose: bool = True) -> None:
        """Run collection algorithm to gather fallen states.

        For each terrain level, spawns all envs uniformly across terrain types,
        simulates falling for fall_duration_s, then captures the final resting state.
        This is repeated num_spawns_per_level times per level.

        Args:
            env: The environment to collect states from.
            verbose: Whether to print progress information.
        """
        # Get terrain and robot info
        terrain: TerrainImporter = env.scene.terrain
        robot: Articulation = env.scene["robot"]

        self._device = "cpu"  # Store on CPU to save VRAM
        self._num_joints = robot.num_joints
        self._num_terrain_levels = terrain.cfg.terrain_generator.num_rows
        self._terrain_cell_size = terrain.cfg.terrain_generator.size

        # Calculate collection parameters
        dt = env.step_dt
        fall_steps = int(self.cfg.fall_duration_s / dt)
        num_terrain_types = terrain.terrain_origins.shape[1]
        states_per_level = env.num_envs * self.cfg.num_spawns_per_level
        total_states = states_per_level * self._num_terrain_levels

        if verbose:
            logger.info("[FallenStateDataset] Starting collection:")
            logger.info(f"  - Terrain grid: {self._num_terrain_levels} levels x {num_terrain_types} types")
            logger.info(f"  - Num envs: {env.num_envs}")
            logger.info(f"  - Spawns per level: {self.cfg.num_spawns_per_level}")
            logger.info(f"  - States per level: {states_per_level}")
            logger.info(f"  - Total states: {total_states}")
            logger.info(f"  - Fall duration: {self.cfg.fall_duration_s}s ({fall_steps} steps)")

        # Initialize storage for all terrain levels
        for level in range(self._num_terrain_levels):
            self._states_by_level[level] = {
                "root_pos_rel": [],
                "root_quat": [],
                "root_lin_vel": [],
                "root_ang_vel": [],
                "joint_pos": [],
                "joint_vel": [],
                "terrain_type": [],  # Track which terrain type each state came from
            }

        # Disable terminations during collection to allow full falls without interruption
        # Requires the monkey patch in manager_based_rl_env_patch.py to be active
        env._disable_terminations = True
        try:
            # Collect states for each terrain level
            for level in range(self._num_terrain_levels):
                if verbose:
                    logger.info(f"  Collecting level {level + 1}/{self._num_terrain_levels}...")

                for _spawn_idx in range(self.cfg.num_spawns_per_level):
                    # Reset envs distributed across terrain columns
                    # With num_envs >> num_cols, all terrain types are covered via modulo wrap-around
                    self._reset_envs_to_terrain_cells(env, level)

                    # Simulate falling for full duration
                    for _step in range(fall_steps):
                        env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))

                    # Capture final resting state (only once at the end)
                    self._capture_states(env, level, terrain)

                # Finalize level (concatenate tensors)
                self._finalize_level(level)

                if verbose:
                    actual = self.get_num_states(level)
                    logger.info(f"  Level {level + 1} done ({actual} states)")

            if verbose:
                total_states = sum(self.get_num_states(lvl) for lvl in range(self._num_terrain_levels))
                logger.info(f"[FallenStateDataset] Collection complete: {total_states} total states")
        finally:
            # Re-enable terminations for normal training
            env._disable_terminations = False
            # Reset terrain levels to 0 (easiest) so training starts from curriculum beginning
            terrain.terrain_levels[:] = 0

    def _reset_envs_to_terrain_cells(self, env: ManagerBasedRLEnv, level: int) -> None:
        """Reset environments to specific terrain cells with random initial poses.

        Args:
            env: The environment.
            level: The terrain level (row) to collect from.
        """
        terrain: TerrainImporter = env.scene.terrain
        robot: Articulation = env.scene["robot"]
        num_envs = env.num_envs

        # terrain_origins has shape (num_levels, num_cols, 3)
        terrain_origins = terrain.terrain_origins  # (num_levels, num_cols, 3)
        num_cols = terrain_origins.shape[1]

        # Update terrain tracking tensors
        # Envs are distributed across columns via modulo - with num_envs >> num_cols,
        # all terrain types are covered multiple times per spawn
        terrain.terrain_levels[:] = level
        terrain.terrain_types[:] = torch.arange(num_envs, device=env.device) % num_cols

        # Get env origins directly from terrain_origins using level and type indices
        env_origins = terrain_origins[level, terrain.terrain_types.long()]  # (num_envs, 3)
        env.scene.env_origins[:] = env_origins

        # Sample random initial poses for falling
        root_states = robot.data.default_root_state.clone()
        env_ids = torch.arange(num_envs, device=env.device)

        # Random yaw, roll, pitch
        yaw = torch.rand(num_envs, device=env.device) * 2 * math.pi - math.pi
        roll = (torch.rand(num_envs, device=env.device) * 2 - 1) * math.radians(20)
        pitch = (torch.rand(num_envs, device=env.device) * 2 - 1) * math.radians(20)
        quat_delta = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        root_states[:, 3:7] = math_utils.quat_mul(root_states[:, 3:7], quat_delta)

        # Add random linear and angular velocity
        root_states[:, 7:10] = (torch.rand(num_envs, 3, device=env.device) * 2 - 1) * 5.0  # lin vel
        root_states[:, 10:13] = (torch.rand(num_envs, 3, device=env.device) * 2 - 1) * 5.0  # ang vel

        # Set position at env origin (default_root_state has relative offset from origin)
        root_states[:, 0:3] = env_origins + root_states[:, 0:3]

        # Write root state
        robot.write_root_pose_to_sim(root_states[:, 0:7], env_ids)
        robot.write_root_velocity_to_sim(root_states[:, 7:13], env_ids)

        # Random joint positions within limits
        joint_pos_limits = robot.data.soft_joint_pos_limits
        joint_pos = torch.rand(num_envs, robot.num_joints, device=env.device)
        joint_pos = joint_pos * (joint_pos_limits[:, :, 1] - joint_pos_limits[:, :, 0]) + joint_pos_limits[:, :, 0]
        joint_vel = (torch.rand(num_envs, robot.num_joints, device=env.device) * 2 - 1) * 1.0
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Reset episode counters to enable joint disabling during fall
        env.episode_length_buf[:] = 0

    def _capture_states(self, env: ManagerBasedRLEnv, level: int, terrain: TerrainImporter) -> None:
        """Capture current robot states and add to the dataset."""
        robot: Articulation = env.scene["robot"]

        # Get current root state
        root_pos_w = robot.data.root_pos_w.clone()  # world position
        root_quat = robot.data.root_quat_w.clone()
        root_lin_vel = robot.data.root_lin_vel_w.clone()
        root_ang_vel = robot.data.root_ang_vel_w.clone()

        # Convert position to relative (relative to env origin / terrain origin)
        root_pos_rel = root_pos_w - env.scene.env_origins

        # Clamp relative position to stay within terrain cell bounds
        # This prevents issues when robots roll outside their original cell during falling
        half_size_x = self._terrain_cell_size[0] / 2.0 - 0.5  # Leave 0.5m margin
        half_size_y = self._terrain_cell_size[1] / 2.0 - 0.5
        root_pos_rel[:, 0] = torch.clamp(root_pos_rel[:, 0], -half_size_x, half_size_x)
        root_pos_rel[:, 1] = torch.clamp(root_pos_rel[:, 1], -half_size_y, half_size_y)

        # Get joint states
        joint_pos = robot.data.joint_pos.clone()
        joint_vel = robot.data.joint_vel.clone()

        # Get terrain type for each env (needed for correct reset)
        terrain_type = terrain.terrain_types.clone()

        # Store on CPU
        storage = self._states_by_level[level]
        storage["root_pos_rel"].append(root_pos_rel.cpu())
        storage["root_quat"].append(root_quat.cpu())
        storage["root_lin_vel"].append(root_lin_vel.cpu())
        storage["root_ang_vel"].append(root_ang_vel.cpu())
        storage["joint_pos"].append(joint_pos.cpu())
        storage["joint_vel"].append(joint_vel.cpu())
        storage["terrain_type"].append(terrain_type.cpu())

    def _finalize_level(self, level: int) -> None:
        """Convert lists to tensors."""
        storage = self._states_by_level[level]

        for key in storage:
            # Concatenate all collected tensors
            storage[key] = torch.cat(storage[key], dim=0)

    def sample(self, num_samples: int, terrain_levels: torch.Tensor, device: str = "cuda") -> dict[str, torch.Tensor]:
        """Sample fallen states for given terrain levels.

        Args:
            num_samples: Number of samples to return (should match len(terrain_levels))
            terrain_levels: Terrain level for each sample (shape: num_samples,)
            device: Device to return tensors on

        Returns:
            Dictionary with keys: root_pos_rel, root_quat, root_lin_vel, root_ang_vel,
            joint_pos, joint_vel, terrain_type.
            Each tensor has shape (num_samples, ...).
            terrain_type indicates which terrain cell the state was collected from.
        """
        if not self.is_collected:
            raise RuntimeError("Dataset has not been collected yet. Call collect() first.")

        terrain_levels_cpu = terrain_levels.cpu()

        # Pre-allocate output tensors
        result = {
            "root_pos_rel": torch.zeros(num_samples, 3, device=device),
            "root_quat": torch.zeros(num_samples, 4, device=device),
            "root_lin_vel": torch.zeros(num_samples, 3, device=device),
            "root_ang_vel": torch.zeros(num_samples, 3, device=device),
            "joint_pos": torch.zeros(num_samples, self._num_joints, device=device),
            "joint_vel": torch.zeros(num_samples, self._num_joints, device=device),
            "terrain_type": torch.zeros(num_samples, dtype=torch.long, device=device),
        }

        # Sample for each unique terrain level
        unique_levels = terrain_levels_cpu.unique()
        for level in unique_levels:
            level_int = level.item()
            mask_cpu = terrain_levels_cpu == level

            # Clamp level to valid range
            level_clamped = max(0, min(level_int, self._num_terrain_levels - 1))
            storage = self._states_by_level[level_clamped]
            num_available = storage["root_pos_rel"].shape[0]

            # Random indices for this level
            indices = torch.randint(0, num_available, (mask_cpu.sum().item(),))

            # Copy sampled states (mask must be on same device as result tensors)
            mask = mask_cpu.to(device)
            for key in result:
                result[key][mask] = storage[key][indices].to(device)

        return result

    def save(self, path: str) -> None:
        """Save dataset to disk.

        Args:
            path: File path to save to (should end with .pt)
        """
        save_dict = {
            "cfg": {
                "num_spawns_per_level": self.cfg.num_spawns_per_level,
                "fall_duration_s": self.cfg.fall_duration_s,
            },
            "num_terrain_levels": self._num_terrain_levels,
            "num_joints": self._num_joints,
            "terrain_cell_size": self._terrain_cell_size,
            "states_by_level": self._states_by_level,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(save_dict, path)

    def load(self, path: str) -> bool:
        """Load dataset from disk.

        Args:
            path: File path to load from

        Returns:
            True if load was successful, False otherwise
        """
        if not os.path.exists(path):
            return False

        try:
            save_dict = torch.load(path, weights_only=False)
            self._num_terrain_levels = save_dict["num_terrain_levels"]
            self._num_joints = save_dict["num_joints"]
            self._terrain_cell_size = save_dict.get("terrain_cell_size", (8.0, 8.0))
            self._states_by_level = save_dict["states_by_level"]
            return True
        except Exception:
            logger.exception(f"[FallenStateDataset] Failed to load from {path}")
            return False
