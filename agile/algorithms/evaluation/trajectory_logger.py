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

import json
from pathlib import Path

import pandas as pd
import torch


class TrajectoryLogger:
    """Saves episode trajectory data to parquet files for offline analysis.

    This logger converts episode trajectory tensors into structured pandas DataFrames
    and saves them incrementally as episodes complete. Each episode is saved as a
    separate parquet file for efficient storage and easy loading.
    """

    def __init__(
        self,
        output_dir: str | Path,
        physics_dt: float,
        env=None,
        fields_to_save: list[str] | None = None,
        verbose: bool = False,
    ):
        """Initialize trajectory logger.

        Args:
            output_dir: Base directory where trajectory files will be saved.
                        A 'trajectories/' subfolder will be created inside.
            physics_dt: Physics timestep in seconds (for computing time column)
            env: Environment instance (optional). If provided, extracts metadata like joint names/limits.
            fields_to_save: List of field names to save. If None, saves all available fields.
                           Example: ["joint_pos", "joint_vel", "root_pos"]
            verbose: Whether to print detailed logging information
        """
        self.output_dir = Path(output_dir)
        self.trajectories_dir = self.output_dir / "trajectories"
        self.physics_dt = physics_dt
        self.fields_to_save = fields_to_save  # None means save all
        self.verbose = verbose

        # Create output directory
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)

        # Extract and save metadata if environment provided
        if env is not None:
            self._save_metadata(env)

        if self.verbose:
            print(f"TrajectoryLogger initialized. Saving to: {self.trajectories_dir}")
            if self.fields_to_save:
                print(f"  Fields to save: {self.fields_to_save}")
            else:
                print("  Fields to save: All available fields")

    def _save_metadata(self, env):
        """Extract and save metadata about the robot and environment.

        Args:
            env: Environment instance to extract metadata from
        """

        # Get robot from environment
        robot = env.unwrapped.scene["robot"] if hasattr(env, "unwrapped") else env.scene["robot"]

        # Extract joint information
        joint_names = robot.joint_names
        num_joints = len(joint_names)

        # Get joint limits (take from first environment as they should be the same)
        joint_pos_limits = robot.data.soft_joint_pos_limits[0].detach().cpu().numpy().tolist()  # [num_joints, 2]
        joint_vel_limits = robot.data.soft_joint_vel_limits[0].detach().cpu().numpy().tolist()  # [num_joints]

        # Create metadata dictionary
        metadata = {
            "physics_dt": self.physics_dt,
            "num_joints": num_joints,
            "joint_names": joint_names,
            "joint_pos_limits": joint_pos_limits,  # [num_joints, 2] - [min, max] for each joint
            "joint_vel_limits": joint_vel_limits,  # [num_joints] - max absolute velocity
            "max_episode_length": getattr(env, "max_episode_length", None),
            "task_name": getattr(env.cfg, "name", "unknown") if hasattr(env, "cfg") else "unknown",
        }

        # Save to JSON file
        metadata_path = self.trajectories_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if self.verbose:
            print(f"  Saved metadata to: {metadata_path}")
            print(f"    Joints: {num_joints}")

    def log_episodes(
        self,
        episode_data: dict[str, torch.Tensor],
        env_ids: torch.Tensor,
        episode_numbers: list[int],
        is_success: torch.Tensor,
    ):
        """Log multiple terminated episodes to disk.

        Args:
            episode_data: Dictionary containing trajectory tensors for all terminated episodes.
                         Keys are field names (e.g., "joint_pos", "joint_vel").
                         Values are tensors of shape [max_frames, num_terminated_envs, ...]
            env_ids: Tensor of environment IDs that terminated [num_terminated_envs]
            episode_numbers: List of unique episode identifiers [num_terminated_envs]
            is_success: Boolean tensor indicating success for each episode [num_terminated_envs]
        """
        # Get frame counts to know valid data length for each episode
        frame_counts = episode_data.get("frame_counts")
        if frame_counts is None:
            if self.verbose:
                print("Warning: No frame_counts in episode_data, cannot determine valid data length")
            return

        num_episodes = len(env_ids)

        # Process each terminated episode
        for i in range(num_episodes):
            env_id = env_ids[i].item()
            episode_id = episode_numbers[i]
            num_frames = frame_counts[i].item()
            success = is_success[i].item() if is_success.dim() > 0 else is_success.item()

            # Convert this episode's data to DataFrame
            df = self._convert_to_dataframe(
                episode_data=episode_data,
                episode_index=i,
                env_id=env_id,
                episode_id=episode_id,
                num_frames=num_frames,
                is_success=success,
            )

            # Save to file
            self._save_dataframe(df, episode_id)

            if self.verbose:
                status = "success" if success else "failed"
                print(f"  Saved episode {episode_id} (env {env_id}, {num_frames} frames, {status})")

    def _convert_to_dataframe(
        self,
        episode_data: dict[str, torch.Tensor],
        episode_index: int,
        env_id: int,
        episode_id: int,
        num_frames: int,
        is_success: bool,
    ) -> pd.DataFrame:
        """Convert trajectory tensors for one episode into a DataFrame.

        Args:
            episode_data: Full dictionary of trajectory data
            episode_index: Which episode index in the batch
            env_id: Environment ID
            episode_id: Unique episode identifier
            num_frames: Number of valid frames in this episode
            is_success: Whether episode completed successfully

        Returns:
            DataFrame with one row per frame, columns for metadata and all trajectory fields
        """
        # Start with metadata columns
        data_dict = {
            "episode_id": [episode_id] * num_frames,
            "env_id": [env_id] * num_frames,
            "frame_idx": list(range(num_frames)),
            "timestep": [i * self.physics_dt for i in range(num_frames)],
            "is_success": [is_success] * num_frames,
        }

        # Process each trajectory field
        for field_name, field_tensor in episode_data.items():
            # Skip metadata fields
            if field_name in ["frame_counts"]:
                continue

            # Skip if not in fields_to_save (when filter is specified)
            if self.fields_to_save is not None and field_name not in self.fields_to_save:
                continue

            # Skip non-tensor data
            if not isinstance(field_tensor, torch.Tensor) or field_tensor.numel() == 0:
                continue

            # Extract data for this specific episode: [num_frames, episode_index, ...]
            episode_field_data = field_tensor[:num_frames, episode_index]

            # Flatten the data and create columns
            # Shape: [num_frames, feature_dim_1, feature_dim_2, ...]
            if episode_field_data.dim() == 1:
                # Scalar per frame (e.g., single value)
                data_dict[field_name] = episode_field_data.cpu().numpy()
            elif episode_field_data.dim() == 2:
                # Vector per frame (e.g., joint_pos with shape [num_frames, num_joints])
                num_features = episode_field_data.shape[1]
                for j in range(num_features):
                    column_name = f"{field_name}_{j}"
                    data_dict[column_name] = episode_field_data[:, j].cpu().numpy()
            elif episode_field_data.dim() == 3:
                # Matrix per frame (e.g., [num_frames, num_joints, 1])
                # Squeeze last dimension if it's 1
                if episode_field_data.shape[-1] == 1:
                    episode_field_data = episode_field_data.squeeze(-1)
                num_features = episode_field_data.shape[1]
                for j in range(num_features):
                    column_name = f"{field_name}_{j}"
                    data_dict[column_name] = episode_field_data[:, j].cpu().numpy()
            else:
                # Higher dimensional data - flatten all non-frame dimensions
                flat_data = episode_field_data.reshape(num_frames, -1)
                num_features = flat_data.shape[1]
                for j in range(num_features):
                    column_name = f"{field_name}_{j}"
                    data_dict[column_name] = flat_data[:, j].cpu().numpy()

        # Create DataFrame
        df = pd.DataFrame(data_dict)
        return df

    def _save_dataframe(self, df: pd.DataFrame, episode_id: int):
        """Save DataFrame to parquet file.

        Args:
            df: DataFrame containing episode trajectory data
            episode_id: Unique episode identifier for filename
        """
        filename = f"episode_{episode_id:03d}.parquet"
        filepath = self.trajectories_dir / filename

        # Save to parquet with good compression
        df.to_parquet(filepath, compression="snappy", index=False)

    def get_episode_path(self, episode_id: int) -> Path:
        """Get the file path for a specific episode.

        Args:
            episode_id: Episode identifier

        Returns:
            Path to the episode's parquet file
        """
        filename = f"episode_{episode_id:03d}.parquet"
        return self.trajectories_dir / filename

    def load_episode(self, episode_id: int) -> pd.DataFrame:
        """Load a specific episode's trajectory data.

        Args:
            episode_id: Episode identifier to load

        Returns:
            DataFrame containing the episode's trajectory data
        """
        filepath = self.get_episode_path(episode_id)
        if not filepath.exists():
            raise FileNotFoundError(f"Episode {episode_id} not found at {filepath}")
        return pd.read_parquet(filepath)

    def load_all_episodes(self) -> pd.DataFrame:
        """Load all saved episodes into a single DataFrame.

        Returns:
            DataFrame containing all episodes concatenated together
        """
        parquet_files = sorted(self.trajectories_dir.glob("episode_*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No episode files found in {self.trajectories_dir}")

        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True)
