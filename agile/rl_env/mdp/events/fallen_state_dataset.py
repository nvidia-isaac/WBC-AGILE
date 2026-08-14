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

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.configclass import configclass

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

    fall_duration_s: float = 1.0
    """Duration to simulate falling with disabled joints (long enough for zero velocity)."""

    spawn_height_offset: float = 2.0
    """Extra height (meters) above default spawn height when dropping robots."""

    spawn_xy_range: float = 3.0
    """Random xy offset (meters) when spawning. Sampled uniformly from [-range, range] per axis."""

    initial_lin_vel_range: float = 1.0
    """Maximum linear velocity (m/s) per axis when spawning. Sampled uniformly from [-range, range]."""

    initial_ang_vel_range: float = 1.0
    """Maximum angular velocity (rad/s) per axis when spawning. Sampled uniformly from [-range, range]."""

    spawn_orientation: Literal["random", "on_back"] = "random"
    """How to orient robots when spawning for collection.

    'random': Uniform random over SO(3) (default, for general fallen states).
    'on_back': Lying on back with random yaw (for back-to-standing tasks).
    """

    spawn_pitch_range: tuple[float, float] = (-1.7, -1.4)
    """Pitch range in radians for 'on_back' orientation mode. Default centered around -pi/2."""

    spawn_joint_mode: Literal["random", "default"] = "random"
    """How to initialize joints when spawning.

    'random': Random within soft joint limits (default).
    'default': Default joint positions with zero velocity.
    """

    cache_enabled: bool = True
    """Whether to enable disk caching of collected states."""

    cache_dir: str = "fallen_states_cache"
    """Directory to store cached fallen states."""

    # Stability thresholds to detect and reset exploding robots
    max_height_above_spawn: float = 1.0
    """Maximum height above spawn origin before robot is considered unstable (m)."""

    max_lin_vel: float = 20.0
    """Maximum linear velocity magnitude before robot is considered unstable (m/s)."""

    max_ang_vel: float = 50.0
    """Maximum angular velocity magnitude before robot is considered unstable (rad/s)."""

    max_joint_vel: float = 100.0
    """Maximum joint velocity magnitude before robot is considered unstable (rad/s)."""


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
    _is_flat_terrain: bool = False

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

        For flat terrain (no terrain_generator), uses a single level with no xy offset.

        Args:
            env: The environment to collect states from.
            verbose: Whether to print progress information.
        """
        # Get terrain and robot info
        terrain: TerrainImporter = env.scene.terrain
        robot: Articulation = env.scene["robot"]

        self._device = "cpu"  # Store on CPU to save VRAM
        self._num_joints = robot.num_joints

        # Check if using generated terrain or flat plane
        self._is_flat_terrain = terrain.cfg.terrain_generator is None
        if self._is_flat_terrain:
            self._num_terrain_levels = 1
            self._terrain_cell_size = (100.0, 100.0)  # Large size, won't clamp much
        else:
            self._num_terrain_levels = terrain.cfg.terrain_generator.num_rows
            self._terrain_cell_size = terrain.cfg.terrain_generator.size

        # Calculate collection parameters
        dt = env.step_dt
        fall_steps = int(self.cfg.fall_duration_s / dt)
        num_terrain_types = 1 if self._is_flat_terrain else terrain.terrain_origins.shape[1]
        states_per_level = env.num_envs * self.cfg.num_spawns_per_level
        total_states = states_per_level * self._num_terrain_levels

        if verbose:
            print("[FallenStateDataset] Starting collection:")
            if self._is_flat_terrain:
                print("  - Terrain: flat plane (single level)")
            else:
                print(f"  - Terrain grid: {self._num_terrain_levels} levels x {num_terrain_types} types")
            print(f"  - Num envs: {env.num_envs}")
            print(f"  - Spawns per level: {self.cfg.num_spawns_per_level}")
            print(f"  - States per level: {states_per_level}")
            print(f"  - Total states: {total_states}")
            print(f"  - Fall duration: {self.cfg.fall_duration_s}s ({fall_steps} steps)")

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
        env._disable_terminations = True

        decimation = env.cfg.decimation

        try:
            # Collect states for each terrain level
            for level in range(self._num_terrain_levels):
                if verbose:
                    print(f"  Level {level + 1}/{self._num_terrain_levels}", flush=True)

                level_reset_count = 0
                for spawn_idx in range(self.cfg.num_spawns_per_level):
                    # Reset envs distributed across terrain columns
                    # With num_envs >> num_cols, all terrain types are covered via modulo wrap-around
                    self._reset_envs_to_terrain_cells(env, level)

                    # Simulate falling for full duration with joints disabled (zero torques)
                    # We use a manual sim loop to ensure efforts are zeroed before each physics step
                    for _step in range(fall_steps):
                        if verbose:
                            progress = f"    Spawn {spawn_idx + 1}/{self.cfg.num_spawns_per_level}, step {_step + 1}/{fall_steps}"
                            print(f"{progress:<50}", end="\r", flush=True)

                        # Step simulation with disabled joints
                        for _ in range(decimation):
                            # Zero out joint efforts before physics (same as disable_joints event)
                            wp.to_torch(robot._joint_effort_target_sim)[:] = 0.0
                            robot.root_view.set_dof_actuation_forces(robot._joint_effort_target_sim, robot._ALL_INDICES)
                            # Step physics with rendering enabled
                            env.sim.step()

                        # Update scene after decimation steps
                        env.scene.update(dt=env.step_dt)

                        # Check for unstable robots and reset them
                        # They continue falling with remaining steps (no counter restart)
                        unstable = self._check_unstable_envs(env)
                        if unstable.any():
                            reset_ids = torch.where(unstable)[0]
                            for env_id in reset_ids:
                                self._reset_single_env(env, env_id.item(), level)
                            level_reset_count += unstable.sum().item()

                    # Capture final resting state (only once at the end)
                    self._capture_states(env, level, terrain)

                # Finalize level (concatenate tensors)
                self._finalize_level(level)

                if verbose:
                    # Clear progress line and print level summary
                    actual = self.get_num_states(level)
                    reset_info = f", {level_reset_count} resets" if level_reset_count > 0 else ""
                    print(f"    Done: {actual} states{reset_info:<30}")

            if verbose:
                total_states = sum(self.get_num_states(lvl) for lvl in range(self._num_terrain_levels))
                print(f"[FallenStateDataset] Collection complete: {total_states} total states")
        finally:
            # Re-enable terminations for normal training
            env._disable_terminations = False
            # Reset terrain levels to 0 (easiest) so training starts from curriculum beginning
            # Only for generated terrain - plane terrain doesn't have terrain_levels
            if not self._is_flat_terrain:
                terrain.terrain_levels[:] = 0
            # Re-enable rendering by doing a render step
            if env.sim.has_gui:
                env.sim.render()

    def _reset_envs_to_terrain_cells(self, env: ManagerBasedRLEnv, level: int) -> None:
        """Reset environments to specific terrain cells with random initial poses.

        Args:
            env: The environment.
            level: The terrain level (row) to collect from.
        """
        terrain: TerrainImporter = env.scene.terrain
        robot: Articulation = env.scene["robot"]
        num_envs = env.num_envs

        if self._is_flat_terrain:
            # Flat terrain: use existing env_origins (set by scene), no xy offset
            # Note: plane terrain doesn't have terrain_levels/terrain_types attributes
            env_origins = env.scene.env_origins.clone()
        else:
            # Generated terrain: distribute across terrain cells
            terrain_origins = terrain.terrain_origins  # (num_levels, num_cols, 3)
            num_cols = terrain_origins.shape[1]

            # Update terrain tracking tensors
            terrain.terrain_levels[:] = level
            terrain.terrain_types[:] = torch.arange(num_envs, device=env.device) % num_cols

            # Get env origins directly from terrain_origins
            env_origins = terrain_origins[level, terrain.terrain_types.long()]  # (num_envs, 3)
            env.scene.env_origins[:] = env_origins

        # Sample initial poses for falling
        root_states = robot.data.default_root_state.torch.clone()
        env_ids = torch.arange(num_envs, device=env.device)

        # Set orientation based on spawn mode
        if self.cfg.spawn_orientation == "on_back":
            # Lying on back: backward pitch + random yaw
            pitch = torch.empty(num_envs, device=env.device).uniform_(*self.cfg.spawn_pitch_range)
            yaw = torch.rand(num_envs, device=env.device) * 2 * math.pi - math.pi
            roll = torch.zeros(num_envs, device=env.device)
            quat_delta = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
            root_states[:, 3:7] = math_utils.quat_mul(root_states[:, 3:7], quat_delta)
        else:
            # Uniform random orientation over SO(3) using Shoemake's algorithm
            u1 = torch.rand(num_envs, device=env.device)
            u2 = torch.rand(num_envs, device=env.device)
            u3 = torch.rand(num_envs, device=env.device)
            sqrt_u1 = torch.sqrt(u1)
            sqrt_1_minus_u1 = torch.sqrt(1.0 - u1)
            two_pi_u2 = 2.0 * math.pi * u2
            two_pi_u3 = 2.0 * math.pi * u3
            random_quat = torch.stack(
                [
                    sqrt_1_minus_u1 * torch.sin(two_pi_u2),
                    sqrt_1_minus_u1 * torch.cos(two_pi_u2),
                    sqrt_u1 * torch.sin(two_pi_u3),
                    sqrt_u1 * torch.cos(two_pi_u3),
                ],
                dim=-1,
            )
            root_states[:, 3:7] = random_quat

        # Add random linear and angular velocity (configurable range)
        lin_vel_range = self.cfg.initial_lin_vel_range
        ang_vel_range = self.cfg.initial_ang_vel_range
        root_states[:, 7:10] = (torch.rand(num_envs, 3, device=env.device) * 2 - 1) * lin_vel_range
        root_states[:, 10:13] = (torch.rand(num_envs, 3, device=env.device) * 2 - 1) * ang_vel_range

        # Set position at env origin (default_root_state has relative offset from origin)
        # Add configurable extra height to drop from above terrain features
        root_states[:, 0:3] = env_origins + root_states[:, 0:3]
        root_states[:, 2] += self.cfg.spawn_height_offset

        # Add random xy offset only for generated terrain (not flat)
        if not self._is_flat_terrain:
            xy_offset = (torch.rand(num_envs, 2, device=env.device) * 2 - 1) * self.cfg.spawn_xy_range
            root_states[:, 0:2] += xy_offset

        # Write root state
        robot.write_root_pose_to_sim(root_states[:, 0:7], env_ids)
        robot.write_root_velocity_to_sim(root_states[:, 7:13], env_ids)

        # Set joint state based on spawn mode
        if self.cfg.spawn_joint_mode == "default":
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos.torch.clone(),
                robot.data.default_joint_vel.torch.clone(),
                env_ids=env_ids,
            )
        else:
            joint_pos_limits = robot.data.soft_joint_pos_limits.torch
            joint_pos = torch.rand(num_envs, robot.num_joints, device=env.device)
            joint_pos = joint_pos * (joint_pos_limits[:, :, 1] - joint_pos_limits[:, :, 0]) + joint_pos_limits[:, :, 0]
            joint_vel = (torch.rand(num_envs, robot.num_joints, device=env.device) * 2 - 1) * 1.0
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Reset episode counters to enable joint disabling during fall
        env.episode_length_buf[:] = 0

    def _reset_single_env(self, env: ManagerBasedRLEnv, env_id: int, level: int) -> None:
        """Reset a single environment that became unstable during falling.

        Args:
            env: The environment.
            env_id: The environment index to reset.
            level: The terrain level for this env.
        """
        terrain: TerrainImporter = env.scene.terrain
        robot: Articulation = env.scene["robot"]
        device = env.device

        if self._is_flat_terrain:
            # Flat terrain: use existing env_origin
            env_origin = env.scene.env_origins[env_id]
        else:
            # Generated terrain: get origin from terrain grid
            terrain_origins = terrain.terrain_origins
            terrain_type = terrain.terrain_types[env_id].long()
            env_origin = terrain_origins[level, terrain_type]
            env.scene.env_origins[env_id] = env_origin

        # Sample new initial pose
        root_state = robot.data.default_root_state.torch[env_id].clone()

        # Set orientation based on spawn mode
        if self.cfg.spawn_orientation == "on_back":
            pitch = torch.empty(1, device=device).uniform_(*self.cfg.spawn_pitch_range).squeeze()
            yaw = torch.rand(1, device=device).squeeze() * 2 * math.pi - math.pi
            roll = torch.zeros(1, device=device).squeeze()
            quat_delta = math_utils.quat_from_euler_xyz(
                roll.unsqueeze(0), pitch.unsqueeze(0), yaw.unsqueeze(0)
            ).squeeze(0)
            root_state[3:7] = math_utils.quat_mul(root_state[3:7].unsqueeze(0), quat_delta.unsqueeze(0)).squeeze(0)
        else:
            u1, u2, u3 = torch.rand(3, device=device)
            sqrt_u1 = torch.sqrt(u1)
            sqrt_1_minus_u1 = torch.sqrt(1.0 - u1)
            two_pi_u2 = 2.0 * math.pi * u2
            two_pi_u3 = 2.0 * math.pi * u3
            random_quat = torch.tensor(
                [
                    sqrt_1_minus_u1 * torch.sin(two_pi_u2),
                    sqrt_1_minus_u1 * torch.cos(two_pi_u2),
                    sqrt_u1 * torch.sin(two_pi_u3),
                    sqrt_u1 * torch.cos(two_pi_u3),
                ],
                device=device,
            )
            root_state[3:7] = random_quat

        # Random velocities
        root_state[7:10] = (torch.rand(3, device=device) * 2 - 1) * self.cfg.initial_lin_vel_range
        root_state[10:13] = (torch.rand(3, device=device) * 2 - 1) * self.cfg.initial_ang_vel_range

        # Position at env origin with offset
        root_state[0:3] = env_origin + root_state[0:3]
        root_state[2] += self.cfg.spawn_height_offset

        # Add random xy offset only for generated terrain
        if not self._is_flat_terrain:
            xy_offset = (torch.rand(2, device=device) * 2 - 1) * self.cfg.spawn_xy_range
            root_state[0:2] += xy_offset

        # Write root state
        env_ids_tensor = torch.tensor([env_id], device=device)
        robot.write_root_pose_to_sim(root_state[0:7].unsqueeze(0), env_ids_tensor)
        robot.write_root_velocity_to_sim(root_state[7:13].unsqueeze(0), env_ids_tensor)

        # Set joint state based on spawn mode
        if self.cfg.spawn_joint_mode == "default":
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos.torch[env_id].unsqueeze(0),
                robot.data.default_joint_vel.torch[env_id].unsqueeze(0),
                env_ids=env_ids_tensor,
            )
        else:
            joint_pos_limits = robot.data.soft_joint_pos_limits.torch[env_id]
            joint_pos = torch.rand(robot.num_joints, device=device)
            joint_pos = joint_pos * (joint_pos_limits[:, 1] - joint_pos_limits[:, 0]) + joint_pos_limits[:, 0]
            joint_vel = (torch.rand(robot.num_joints, device=device) * 2 - 1) * 1.0
            robot.write_joint_state_to_sim(joint_pos.unsqueeze(0), joint_vel.unsqueeze(0), env_ids=env_ids_tensor)

    def _check_unstable_envs(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Check which environments have become unstable (exploding physics).

        Args:
            env: The environment.

        Returns:
            Boolean tensor of shape (num_envs,) indicating unstable envs.
        """
        robot: Articulation = env.scene["robot"]

        # Get current state
        root_pos_w = robot.data.root_pos_w.torch
        root_lin_vel = robot.data.root_lin_vel_w.torch
        root_ang_vel = robot.data.root_ang_vel_w.torch
        joint_vel = robot.data.joint_vel.torch

        # Check height above spawn position (env_origin + spawn_height_offset)
        spawn_height = env.scene.env_origins[:, 2] + self.cfg.spawn_height_offset
        height_above_spawn = root_pos_w[:, 2] - spawn_height
        too_high = height_above_spawn > self.cfg.max_height_above_spawn

        # Check linear velocity magnitude
        lin_vel_mag = torch.norm(root_lin_vel, dim=-1)
        lin_vel_too_high = lin_vel_mag > self.cfg.max_lin_vel

        # Check angular velocity magnitude
        ang_vel_mag = torch.norm(root_ang_vel, dim=-1)
        ang_vel_too_high = ang_vel_mag > self.cfg.max_ang_vel

        # Check joint velocity magnitude (max across all joints)
        joint_vel_mag = torch.abs(joint_vel).max(dim=-1).values
        joint_vel_too_high = joint_vel_mag > self.cfg.max_joint_vel

        # Combine all checks
        unstable = too_high | lin_vel_too_high | ang_vel_too_high | joint_vel_too_high

        return unstable

    def _capture_states(self, env: ManagerBasedRLEnv, level: int, terrain: TerrainImporter) -> None:
        """Capture current robot states and add to the dataset."""
        robot: Articulation = env.scene["robot"]

        # Get current root state
        root_pos_w = robot.data.root_pos_w.torch.clone()  # world position
        root_quat = robot.data.root_quat_w.torch.clone()
        root_lin_vel = robot.data.root_lin_vel_w.torch.clone()
        root_ang_vel = robot.data.root_ang_vel_w.torch.clone()

        # Convert position to relative (relative to env origin / terrain origin)
        root_pos_rel = root_pos_w - env.scene.env_origins

        # Clamp relative position to stay within terrain cell bounds (skip for flat terrain)
        if not self._is_flat_terrain:
            half_size_x = self._terrain_cell_size[0] / 2.0 - 0.5  # Leave 0.5m margin
            half_size_y = self._terrain_cell_size[1] / 2.0 - 0.5
            root_pos_rel[:, 0] = torch.clamp(root_pos_rel[:, 0], -half_size_x, half_size_x)
            root_pos_rel[:, 1] = torch.clamp(root_pos_rel[:, 1], -half_size_y, half_size_y)

        # Get joint states
        joint_pos = robot.data.joint_pos.torch.clone()
        joint_vel = robot.data.joint_vel.torch.clone()

        # Get terrain type for each env (0 for flat terrain)
        if self._is_flat_terrain:
            terrain_type = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        else:
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
                "spawn_height_offset": self.cfg.spawn_height_offset,
                "spawn_xy_range": self.cfg.spawn_xy_range,
                "initial_lin_vel_range": self.cfg.initial_lin_vel_range,
                "initial_ang_vel_range": self.cfg.initial_ang_vel_range,
                "spawn_orientation": self.cfg.spawn_orientation,
                "spawn_pitch_range": self.cfg.spawn_pitch_range,
                "spawn_joint_mode": self.cfg.spawn_joint_mode,
            },
            "num_terrain_levels": self._num_terrain_levels,
            "num_joints": self._num_joints,
            "terrain_cell_size": self._terrain_cell_size,
            "is_flat_terrain": self._is_flat_terrain,
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
            self._is_flat_terrain = save_dict.get("is_flat_terrain", False)
            self._states_by_level = save_dict["states_by_level"]
            return True
        except Exception as e:
            print(f"[FallenStateDataset] Failed to load from {path}: {e}")
            return False
