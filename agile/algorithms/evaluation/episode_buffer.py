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

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Frame:
    """A frame contains a set of trajectory data for a single timestep."""

    joint_pos: torch.Tensor = torch.tensor([])  # [num_envs, num_joints, 1]
    joint_vel: torch.Tensor = torch.tensor([])  # [num_envs, num_joints, 1]
    joint_acc: torch.Tensor = torch.tensor([])  # [num_envs, num_joints, 1]
    root_pos: torch.Tensor = torch.tensor([])  # [num_envs, 3]
    root_rot: torch.Tensor = torch.tensor([])  # [num_envs, 4]
    root_lin_vel: torch.Tensor = torch.tensor([])  # [num_envs, 3] - World frame
    root_ang_vel: torch.Tensor = torch.tensor([])  # [num_envs, 3] - World frame (yaw rate is same in both frames)
    root_lin_vel_robot: torch.Tensor = torch.tensor([])  # [num_envs, 3] - Robot yaw-aligned frame [forward, left, up]
    commands: torch.Tensor = torch.tensor([])  # [num_envs, 4]
    actions: torch.Tensor = torch.tensor([])  # [num_envs, num_joints, 1]

    # Add custom fields as needed

    @staticmethod
    def from_dict(data: dict) -> Frame:
        """Create a Frame from a dictionary."""
        return Frame(**data)


class EpisodeBuffer:
    """An episode buffer that contains trajectories current running environments.

    Each time a new environment terminates, the episode buffer should do the following:
      1. reset the num_frame for the newly terminated environments to 0.
      2. extract all the trajectory data of the newly terminated environments.
    """

    def __init__(
        self,
        num_envs: int,
        max_episode_length: int,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize an episode for storing trajectory data.

        Args:
            num_envs: Number of parallel environments
            max_episode_length: Maximum number of frames per episode
            device: Device to store tensors on
        """
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        # Ensure device is torch.device object (handle string or device input)
        self.device = torch.device(device) if isinstance(device, str) else device

        # Storage for trajectory data - initialized after first frame
        self._data = {}

        # Track number of frames per environment
        self._num_frames = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._initialized = False

        # Cache for efficiency
        self._env_indices = torch.arange(num_envs, device=device)

    def add_frame(self, frame: Frame, newly_terminated_ids: torch.Tensor | None = None) -> dict[str, Any] | None:
        """Add a frame of data to the episode and handle environment termination.

        Args:
            frame: Frame containing trajectory data for all environments
            newly_terminated_ids: IDs of environments that just terminated

        Returns:
            Dictionary of trajectory data and frame counts for terminated environments,
            or None if no terminations
        """
        if not self._initialized:
            self._initialize_storage(frame)

        # Store all non-empty tensors from the frame for all environments first
        # attr_data is a tensor of shape [num_envs, ...]
        for attr_name, attr_data in vars(frame).items():
            if isinstance(attr_data, torch.Tensor) and attr_data.numel() > 0:
                # Use advanced indexing to store data for all environments at once
                # _num_frames contains the current frame index for each environment
                self._data[attr_name][self._num_frames, self._env_indices] = attr_data

        # Increment frame counters for all environments
        self._num_frames += 1

        # Extract terminated data after storing the current frame
        terminated_data = None
        has_terminations = newly_terminated_ids is not None and newly_terminated_ids.numel() > 0

        if has_terminations:
            terminated_data = self._extract_terminated_data(newly_terminated_ids)
            # Reset frame counters for newly terminated environments
            self._num_frames[newly_terminated_ids] = 0

        return terminated_data

    def _extract_terminated_data(self, terminated_ids: torch.Tensor) -> dict[str, Any]:
        """Extract trajectory data for terminated environments.

        Args:
            terminated_ids: IDs of environments that have terminated

        Returns:
            Dictionary containing:
            - Each tensor attribute with shape [max_episode_length, num_terminated_envs, ...]
            - 'frame_counts': Tensor of size [num_terminated_envs] with the number of valid frames
              for each environment
        """
        terminated_data = {}

        # Check whether terminated_ids has any elements in it
        if terminated_ids.numel() == 0:
            return terminated_data

        # Store the frame counts for each terminated environment
        frame_counts = self._num_frames[terminated_ids].clone()
        terminated_data["frame_counts"] = frame_counts

        # For efficiency, extract all frames for all terminated environments
        # Even though some might not be valid, this is more computationally efficient
        for attr_name, attr_data in self._data.items():
            # Get full frames for all terminated environments
            # Shape: [max_episode_length, num_terminated_envs, ...]
            env_data = attr_data[:, terminated_ids].clone()
            terminated_data[attr_name] = env_data

        return terminated_data

    def _initialize_storage(self, frame: Frame) -> None:
        """Initialize storage tensors based on first frame dimensions.

        Args:
            frame: First frame containing trajectory data
        """
        for attr_name, tensor in vars(frame).items():
            if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                # Create storage tensor with shape [max_episode_length, num_envs, *tensor.shape[1:]]
                # Note: tensor.shape[0] is num_envs
                self._data[attr_name] = torch.zeros(
                    (self.max_episode_length, self.num_envs, *tensor.shape[1:]),
                    dtype=tensor.dtype,
                    device=self.device,
                )

        self._initialized = True

    def __getattr__(self, name: str) -> Any:
        """Provide attribute access to trajectory data.

        Args:
            name: Name of the attribute to access

        Returns:
            Tensor data for the requested attribute

        Raises:
            AttributeError: If attribute doesn't exist
        """
        # Check if attribute is in trajectory data
        if name in self._data:
            # Return data for all environments up to their current frame count
            return self._data[name]

        # Default behavior for unknown attributes
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
