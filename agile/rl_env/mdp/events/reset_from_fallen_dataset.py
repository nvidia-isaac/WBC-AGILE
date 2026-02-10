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

"""Reset event that samples from pre-collected fallen states."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

from .fallen_state_dataset import FallenStateDataset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class reset_from_fallen_dataset(ManagerTermBase):
    """Reset event that samples from pre-collected fallen states.

    This reset event samples fallen robot states from a pre-collected dataset
    instead of simulating the robot falling with disabled joints each episode.
    This can significantly speed up training by eliminating the 2-second fall
    simulation at the start of each episode.

    The dataset must be injected after collection using set_dataset().
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        """Initialize the reset event.

        Args:
            cfg: Event term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        self._dataset: FallenStateDataset | None = None

    def set_dataset(self, dataset: FallenStateDataset) -> None:
        """Inject the fallen state dataset after collection.

        Args:
            dataset: The pre-collected fallen state dataset.
        """
        self._dataset = dataset

    @property
    def has_dataset(self) -> bool:
        """Check if a dataset has been set and is collected."""
        return self._dataset is not None and self._dataset.is_collected

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        standing_ratio: float = 0.1,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        """Reset environments to sampled fallen states or standing poses.

        Args:
            env: The environment instance.
            env_ids: Environment indices to reset.
            standing_ratio: Fraction of envs to reset to standing pose (default 0.1).
            asset_cfg: Asset configuration for the robot.

        Raises:
            RuntimeError: If no dataset has been set and fallen envs need to be reset.
        """
        if len(env_ids) == 0:
            return

        # Get robot and terrain
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        terrain: TerrainImporter = env.scene.terrain

        # Determine which envs reset to standing vs fallen
        standing_mask = torch.rand(len(env_ids), device=env.device) < standing_ratio
        fallen_mask = ~standing_mask

        standing_env_ids = env_ids[standing_mask]
        fallen_env_ids = env_ids[fallen_mask]

        # Reset standing envs to default pose with random yaw
        if len(standing_env_ids) > 0:
            self._reset_to_standing(env, standing_env_ids, asset)

        # Reset fallen envs from dataset
        if len(fallen_env_ids) > 0:
            if not self.has_dataset:
                # Dataset not ready (e.g., during collection phase) - fall back to standing
                self._reset_to_standing(env, fallen_env_ids, asset)
            else:
                self._reset_from_dataset(env, fallen_env_ids, asset, terrain)

    def _reset_to_standing(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        asset: RigidObject | Articulation,
    ) -> None:
        """Reset environments to standing pose with random yaw."""
        root_states = asset.data.default_root_state[env_ids].clone()

        # Random yaw only (no roll/pitch for standing)
        yaw = torch.rand(len(env_ids), device=env.device) * 2 * torch.pi - torch.pi
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)
        quat_delta = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        root_states[:, 3:7] = math_utils.quat_mul(root_states[:, 3:7], quat_delta)

        # Zero velocity for standing
        root_states[:, 7:13] = 0

        # Position at env origin
        root_states[:, 0:3] = env.scene.env_origins[env_ids] + root_states[:, 0:3]

        # Write to simulation
        asset.write_root_pose_to_sim(root_states[:, 0:7], env_ids)
        asset.write_root_velocity_to_sim(root_states[:, 7:13], env_ids)

        # Reset joints to default
        if isinstance(asset, Articulation):
            asset.write_joint_state_to_sim(
                asset.data.default_joint_pos[env_ids],
                asset.data.default_joint_vel[env_ids],
                env_ids=env_ids,
            )

    def _reset_from_dataset(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        asset: RigidObject | Articulation,
        terrain: TerrainImporter,
    ) -> None:
        """Reset environments to sampled fallen states from dataset."""
        if self._dataset is None:
            raise RuntimeError("Dataset not set. Call set_dataset() before _reset_from_dataset().")

        # Get terrain levels for each env
        terrain_levels = terrain.terrain_levels[env_ids]

        # Sample states from dataset (includes terrain_type where state was collected)
        states = self._dataset.sample(len(env_ids), terrain_levels, device=env.device)

        # Update terrain types and env origins to match where states were collected
        # This ensures the relative position is applied to the correct terrain cell
        sampled_terrain_types = states["terrain_type"]
        terrain.terrain_types[env_ids] = sampled_terrain_types

        # Get the correct env origins from terrain_origins using level and sampled type
        # terrain_origins has shape (num_levels, num_types, 3)
        env_origins = terrain.terrain_origins[terrain_levels.long(), sampled_terrain_types.long()]
        env.scene.env_origins[env_ids] = env_origins

        # Convert relative position to world position using the correct origin
        root_pos_w = states["root_pos_rel"] + env_origins

        # Construct root pose and velocity
        root_pose = torch.cat([root_pos_w, states["root_quat"]], dim=-1)
        root_vel = torch.cat([states["root_lin_vel"], states["root_ang_vel"]], dim=-1)

        # Write to simulation
        asset.write_root_pose_to_sim(root_pose, env_ids)
        asset.write_root_velocity_to_sim(root_vel, env_ids)

        # Write joint state
        if isinstance(asset, Articulation):
            asset.write_joint_state_to_sim(
                states["joint_pos"],
                states["joint_vel"],
                env_ids=env_ids,
            )
