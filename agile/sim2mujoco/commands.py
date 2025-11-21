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

"""Command manager for interactive control of robot commands."""

import threading

import torch


class CommandManager:
    """
    Thread-safe manager for velocity and height commands with keyboard control.

    Command ranges:
        - linear_x: [-0.5, 0.5] m/s
        - linear_y: [-0.5, 0.5] m/s
        - angular_z: [-1.0, 1.0] rad/s
        - height: [0.3, 0.8] m
    """

    def __init__(self, device: torch.device, defaults: dict = None):
        """
        Initialize command manager.

        Args:
            device: Torch device for tensor creation.
            defaults: Dictionary with default values for commands.
        """
        self.device = device

        # Thread-safe lock (viewer runs in separate thread)
        self._lock = threading.Lock()

        # Default values
        defaults = defaults or {}
        self._default_linear_x = defaults.get("linear_x", 0.0)
        self._default_linear_y = defaults.get("linear_y", 0.0)
        self._default_angular_z = defaults.get("angular_z", 0.0)
        self._default_height = defaults.get("height", 0.72)

        # Current values
        self._linear_x = self._default_linear_x
        self._linear_y = self._default_linear_y
        self._angular_z = self._default_angular_z
        self._height = self._default_height

        # Step sizes for incremental control
        self.vel_step = 0.1  # m/s per key press
        self.ang_step = 0.2  # rad/s per key press
        self.height_step = 0.05  # m per key press

        # Command limits
        self.linear_x_range = (-0.5, 0.5)
        self.linear_y_range = (-0.5, 0.5)
        self.angular_z_range = (-1.0, 1.0)
        self.height_range = (0.3, 0.8)

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value to range."""
        return max(min_val, min(max_val, value))

    def update_linear_x(self, delta: float):
        """Increment/decrement forward velocity."""
        with self._lock:
            self._linear_x = self._clamp(self._linear_x + delta, *self.linear_x_range)

    def update_linear_y(self, delta: float):
        """Increment/decrement sideways velocity."""
        with self._lock:
            self._linear_y = self._clamp(self._linear_y + delta, *self.linear_y_range)

    def update_angular_z(self, delta: float):
        """Increment/decrement turning velocity."""
        with self._lock:
            self._angular_z = self._clamp(self._angular_z + delta, *self.angular_z_range)

    def update_height(self, delta: float):
        """Increment/decrement target height."""
        with self._lock:
            self._height = self._clamp(self._height + delta, *self.height_range)

    def stop(self):
        """Reset all commands to default values (STOP button)."""
        with self._lock:
            self._linear_x = self._default_linear_x
            self._linear_y = self._default_linear_y
            self._angular_z = self._default_angular_z
            self._height = self._default_height
        print("🛑 STOP: All commands reset to defaults")
        self.print_status()

    def get_command(self) -> torch.Tensor:
        """
        Get current command as tensor.

        Returns:
            Tensor of shape (4,) with [vx, vy, wz, height].
        """
        with self._lock:
            return torch.tensor(
                [self._linear_x, self._linear_y, self._angular_z, self._height], device=self.device, dtype=torch.float32
            )

    def get_navigation_command(self) -> torch.Tensor:
        """
        Get navigation command (3D version without height).

        Returns:
            Tensor of shape (3,) with [vx, vy, wz].
        """
        with self._lock:
            return torch.tensor(
                [self._linear_x, self._linear_y, self._angular_z], device=self.device, dtype=torch.float32
            )

    def print_status(self):
        """Print current command values."""
        with self._lock:
            print(
                f"📡 Commands: "
                f"vel_x={self._linear_x:+.2f} m/s, "
                f"vel_y={self._linear_y:+.2f} m/s, "
                f"ang_z={self._angular_z:+.2f} rad/s, "
                f"height={self._height:.2f} m"
            )
