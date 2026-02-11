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

"""Data recorder utilities for recording observations to HDF5 format.

This module provides classes and utilities for recording robot observations
from Isaac Lab environments into HDF5 files, compatible with robotics learning
frameworks like GR00T, robomimic, and LIBERO.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


@dataclass
class EpisodeBuffer:
    """Buffer to store observations for a single episode/trajectory.

    Attributes:
        obs: Dictionary mapping observation term names to lists of numpy arrays.
        actions: List of action arrays.
    """

    obs: dict[str, list[np.ndarray]] = field(default_factory=dict)
    actions: list[np.ndarray] = field(default_factory=list)

    def append_obs(self, obs_dict: dict[str, torch.Tensor | np.ndarray]) -> None:
        """Append observation terms to the buffer.

        Args:
            obs_dict: Dictionary of observation terms (term_name -> tensor/array).
        """
        for term_name, obs_value in obs_dict.items():
            if term_name not in self.obs:
                self.obs[term_name] = []
            if isinstance(obs_value, torch.Tensor):
                obs_value = obs_value.cpu().numpy()
            self.obs[term_name].append(obs_value.copy())

    def append_action(self, action: torch.Tensor | np.ndarray) -> None:
        """Append an action to the buffer."""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        self.actions.append(action.copy())

    def __len__(self) -> int:
        """Return the number of timesteps in the buffer."""
        if not self.obs:
            return 0
        first_key = next(iter(self.obs))
        return len(self.obs[first_key])

    def clear(self) -> None:
        """Clear all data in the buffer."""
        self.obs.clear()
        self.actions.clear()

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Convert buffer contents to stacked numpy arrays.

        Returns:
            Dictionary with stacked observations and actions.
        """
        result = {}

        # Stack observations
        for term_name, obs_list in self.obs.items():
            result[f"obs/{term_name}"] = np.stack(obs_list, axis=0)

        # Stack actions if available
        if self.actions:
            result["actions"] = np.stack(self.actions, axis=0)

        return result


class HDF5DataRecorder:
    """Records observation data from Isaac Lab environments to HDF5 format.

    This recorder handles multi-environment observations and stores them as
    separate episodes/trajectories in an HDF5 file. The file structure follows
    the robomimic/LIBERO convention:

    ```
    data/
        demo_0/
            obs/
                image (N, H, W, C)
                base_lin_vel (N, 3)
                joint_pos (N, num_joints)
                ...
            actions (N, action_dim)
        demo_1/
            ...
    ```

    Attributes:
        output_path: Path to the output HDF5 file.
        obs_group_name: Name of the observation group to record (e.g., "record").
        compress: Whether to use gzip compression for datasets.
        compress_images: Whether to use compression for image data.
    """

    def __init__(
        self,
        output_path: str | Path,
        obs_group_name: str = "record",
        compress: bool = True,
        compress_images: bool = False,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize the HDF5 data recorder.

        Args:
            output_path: Path to save the HDF5 file.
            obs_group_name: Name of the observation group to record from env.
            compress: Whether to use gzip compression for non-image data.
            compress_images: Whether to compress image data (slower but smaller).
            metadata: Optional metadata to store in the file attributes.
        """
        self.output_path = Path(output_path)
        self.obs_group_name = obs_group_name
        self.compress = compress
        self.compress_images = compress_images
        self.metadata = metadata or {}

        # Episode buffers for each environment
        self._env_buffers: dict[int, EpisodeBuffer] = {}
        self._episode_counter = 0
        self._h5_file: h5py.File | None = None
        self._data_group: h5py.Group | None = None

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def open(self) -> None:
        """Open the HDF5 file for writing."""
        self._h5_file = h5py.File(self.output_path, "w")
        self._data_group = self._h5_file.create_group("data")

        # Write metadata
        self._h5_file.attrs["date"] = datetime.datetime.now().strftime("%Y-%m-%d")
        self._h5_file.attrs["time"] = datetime.datetime.now().strftime("%H:%M:%S")
        for key, value in self.metadata.items():
            self._h5_file.attrs[key] = value

    def close(self) -> None:
        """Flush any remaining episodes and close the HDF5 file."""
        # Flush remaining episode buffers
        for env_idx in list(self._env_buffers.keys()):
            self._flush_episode(env_idx, discard_incomplete=False)

        # Write final metadata
        if self._data_group is not None:
            self._data_group.attrs["num_demos"] = self._episode_counter
            total_samples = sum(
                self._data_group[f"demo_{i}"].attrs.get("num_samples", 0) for i in range(self._episode_counter)
            )
            self._data_group.attrs["total"] = total_samples

        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._data_group = None

    def __enter__(self) -> HDF5DataRecorder:
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def record_step(
        self,
        obs: dict[str, torch.Tensor | np.ndarray],
        actions: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        """Record a single step of observations for all environments.

        Args:
            obs: Dictionary of observation terms from the record observation group.
                  Each term is expected to have shape (num_envs, ...).
            actions: Actions taken, shape (num_envs, action_dim). Optional.
        """
        if self._h5_file is None:
            raise RuntimeError("Recorder not opened. Call open() or use context manager.")

        # Get number of environments from first observation term
        first_term = next(iter(obs.values()))
        if isinstance(first_term, torch.Tensor):
            num_envs = first_term.shape[0]
        else:
            num_envs = first_term.shape[0]

        # Initialize buffers for new environments
        for env_idx in range(num_envs):
            if env_idx not in self._env_buffers:
                self._env_buffers[env_idx] = EpisodeBuffer()

        # Record data for each environment
        for env_idx in range(num_envs):
            buffer = self._env_buffers[env_idx]

            # Extract per-environment observations
            env_obs = {
                term_name: (
                    term_value[env_idx].cpu().numpy() if isinstance(term_value, torch.Tensor) else term_value[env_idx]
                )
                for term_name, term_value in obs.items()
            }
            buffer.append_obs(env_obs)

            # Record action if provided
            if actions is not None:
                action = actions[env_idx]
                buffer.append_action(action)

    def record_from_env(
        self,
        env_obs: dict[str, Any],
        actions: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        """Record observations directly from environment step output.

        This method extracts the specified observation group from the full
        environment observation dictionary.

        Args:
            env_obs: Full observation dictionary from env.step() or env.reset().
            actions: Actions taken, shape (num_envs, action_dim). Optional.
        """
        # Extract the record observation group
        if self.obs_group_name not in env_obs:
            return

        record_obs = env_obs[self.obs_group_name]

        # Handle TensorDict by converting to regular dict
        if hasattr(record_obs, "to_dict"):
            # TensorDict.to_dict() returns a nested dict
            record_obs = dict(record_obs.items())
        elif not isinstance(record_obs, dict):
            raise TypeError(
                f"Expected observation group '{self.obs_group_name}' to be a dict "
                f"(concatenate_terms=False), but got {type(record_obs)}. "
                "Set concatenate_terms=False in RecordObservationsCfg."
            )

        self.record_step(record_obs, actions)

    def flush_all_episodes(self) -> None:
        """Force flush all current episode buffers to the HDF5 file."""
        for env_idx in list(self._env_buffers.keys()):
            self._flush_episode(env_idx, discard_incomplete=False)

    def _flush_episode(self, env_idx: int, discard_incomplete: bool = True) -> None:
        """Flush an episode buffer to the HDF5 file.

        Args:
            env_idx: Environment index to flush.
            discard_incomplete: If True, discard episodes with less than 2 steps.
        """
        if env_idx not in self._env_buffers:
            return

        buffer = self._env_buffers[env_idx]

        # Skip empty or very short episodes
        if len(buffer) < 2 and discard_incomplete:
            buffer.clear()
            return

        if len(buffer) == 0:
            del self._env_buffers[env_idx]
            return

        # Create episode group
        ep_name = f"demo_{self._episode_counter}"
        ep_group = self._data_group.create_group(ep_name)
        ep_group.attrs["num_samples"] = len(buffer)
        ep_group.attrs["env_idx"] = env_idx

        # Convert buffer to numpy and write
        data = buffer.to_numpy()

        for key, value in data.items():
            # Determine if this is image data
            is_image = "image" in key.lower() or "rgb" in key.lower()

            # Convert images from float32 to uint8 (4x smaller file size)
            if is_image and value.dtype in (np.float32, np.float64):
                value = (value * 255).clip(0, 255).astype(np.uint8)

            # Determine compression settings
            use_compression = self.compress
            if is_image:
                use_compression = self.compress_images

            if "/" in key:
                # Create subgroup for nested keys (e.g., "obs/image")
                parts = key.split("/")
                parent = ep_group
                for part in parts[:-1]:
                    if part not in parent:
                        parent = parent.create_group(part)
                    else:
                        parent = parent[part]
                dataset_name = parts[-1]
            else:
                parent = ep_group
                dataset_name = key

            if use_compression:
                parent.create_dataset(dataset_name, data=value, compression="gzip")
            else:
                parent.create_dataset(dataset_name, data=value)

        self._episode_counter += 1

        # Clear and reset buffer for new episode
        buffer.clear()


class MultiEnvDataRecorder(HDF5DataRecorder):
    """Extended data recorder with support for multi-environment episode tracking.

    This recorder provides additional methods for handling reset signals and
    properly tracking episodes across multiple parallel environments.
    """

    def __init__(
        self,
        output_path: str | Path,
        obs_group_name: str = "record",
        compress: bool = True,
        compress_images: bool = False,
        metadata: dict[str, Any] | None = None,
        min_episode_length: int = 10,
    ):
        """Initialize the multi-environment data recorder.

        Args:
            output_path: Path to save the HDF5 file.
            obs_group_name: Name of the observation group to record.
            compress: Whether to use gzip compression.
            compress_images: Whether to compress image data.
            metadata: Optional metadata to store.
            min_episode_length: Minimum episode length to save.
        """
        super().__init__(
            output_path=output_path,
            obs_group_name=obs_group_name,
            compress=compress,
            compress_images=compress_images,
            metadata=metadata,
        )
        self.min_episode_length = min_episode_length

    def handle_resets(self, reset_mask: torch.Tensor | np.ndarray) -> None:
        """Handle environment resets by flushing completed episodes.

        Args:
            reset_mask: Boolean tensor/array of shape (num_envs,) indicating
                        which environments have reset.
        """
        if isinstance(reset_mask, torch.Tensor):
            reset_mask = reset_mask.cpu().numpy()

        for env_idx in range(len(reset_mask)):
            if reset_mask[env_idx]:
                self._flush_episode(env_idx, discard_incomplete=True)

    def _flush_episode(self, env_idx: int, discard_incomplete: bool = True) -> None:
        """Override to enforce minimum episode length."""
        if env_idx not in self._env_buffers:
            return

        buffer = self._env_buffers[env_idx]

        # Check minimum length
        if len(buffer) < self.min_episode_length and discard_incomplete:
            buffer.clear()
            return

        # Call parent implementation
        super()._flush_episode(env_idx, discard_incomplete=False)


def create_recorder_from_env_cfg(
    env_cfg: Any,
    output_dir: str | Path,
    filename: str | None = None,
    obs_group_name: str = "record",
    **kwargs,
) -> HDF5DataRecorder:
    """Create a data recorder from an environment configuration.

    Args:
        env_cfg: Environment configuration object with observations attribute.
        output_dir: Directory to save the HDF5 file.
        filename: Optional filename. If None, generates timestamp-based name.
        **kwargs: Additional arguments to pass to HDF5DataRecorder.

    Returns:
        Configured HDF5DataRecorder instance.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_{timestamp}.hdf5"

    output_path = output_dir / filename

    # Extract metadata from env config
    metadata = {
        "env_class": type(env_cfg).__name__,
    }
    if hasattr(env_cfg, "scene"):
        metadata["num_envs"] = getattr(env_cfg.scene, "num_envs", 1)
    if hasattr(env_cfg, "decimation"):
        metadata["decimation"] = env_cfg.decimation
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "dt"):
        metadata["sim_dt"] = env_cfg.sim.dt

    return HDF5DataRecorder(
        output_path=output_path,
        obs_group_name=obs_group_name,
        metadata=metadata,
        **kwargs,
    )
