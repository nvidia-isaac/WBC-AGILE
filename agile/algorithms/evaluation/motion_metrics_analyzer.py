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
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch


class MotionMetricsAnalyzer:
    """Computes, tracks and aggregates motion metrics across episodes.

    Supports:
    - Registering custom metrics with compute functions
    - Tracking metrics for all episodes vs. successful episodes
    - Body part specific metrics (e.g., upper body, lower body)

    Compatible with the memory-efficient EpisodeBuffer implementation that returns
    full tensors with a frame_counts field to indicate valid frame counts.
    """

    def __init__(
        self,
        max_episode_length: int = 0,
        joint_groups: dict[str, list[int]] | None = None,
        verbose: bool = False,
    ):
        """Initialize a new smoothness measures calculator.

        Args:
            max_episode_length: Maximum possible episode length. Used to determine success.
                                If set to 0, all episodes are considered non-successful.
            joint_groups: Dictionary mapping group names to lists of joint indices
                          Example: {'upper_body': [0, 1, 2], 'lower_body': [3, 4, 5]}
            verbose: Whether to print detailed debug information
        """
        self.num_envs_evaluated = 0
        self.num_envs_successful = 0
        self.success_rate = 0.0
        self.max_episode_length = max_episode_length
        self.joint_groups = joint_groups or {}
        self.verbose = verbose

        # Print joint groups for debugging
        if self.joint_groups and self.verbose:
            print("Initialized SmoothnessMetrics with joint groups:")
            for group_name, indices in self.joint_groups.items():
                print(
                    f"  {group_name}: {len(indices)} joints (indices {indices[:3]}{'...' if len(indices) > 3 else ''})"
                )

        # Storage for aggregated metrics
        self._metrics = {}  # Metrics for all episodes
        self._success_metrics = {}  # Metrics for successful episodes only

        # Storage for per-environment metric data (for weighted averaging)
        self._metrics_data = {}
        self._success_metrics_data = {}

        # Registry of metric computation functions
        self._compute_functions = {}

        # Register default metrics
        self._register_default_metrics()

        # Cache for processed tensor data to avoid redundant calculations
        self._tensor_cache = {}

    def _register_default_metrics(self):
        """Register the default set of smoothness metrics."""
        # Define joint-specific metrics that work with body parts
        base_joint_metrics = {
            "mean_joint_acc": self._compute_mean_joint_acc,
            "max_joint_acc": self._compute_max_joint_acc,
            "mean_acc_rate": self._compute_mean_acc_rate,
            "max_acc_rate": self._compute_max_acc_rate,
            "mean_joint_vel": self._compute_mean_joint_vel,
            "max_joint_vel": self._compute_max_joint_vel,
        }

        # Non-joint metrics that don't make sense for body parts
        base_non_joint_metrics = {
            # Example for future metrics:
        }

        # Combined metrics dictionary with all metrics
        base_metrics = {}
        base_metrics.update(base_joint_metrics)
        base_metrics.update(base_non_joint_metrics)

        # Register whole body metrics for all metric types
        for name, func in base_metrics.items():
            self.register_metric(name, func)

        # Register body part specific metrics if joint groups provided,
        # but only for joint-specific metrics
        for group_name, joint_indices in self.joint_groups.items():
            if not joint_indices:  # Skip empty groups
                if self.verbose:
                    print(f"Warning: Joint group '{group_name}' has no joints, skipping")
                continue

            # Register only joint-specific metrics for this joint group
            for name, func in base_joint_metrics.items():
                self.register_metric(
                    f"{group_name}_{name}",
                    self._create_group_metric(func, joint_indices),
                )

    def _create_group_metric(self, base_metric_fn: Callable, joint_indices: list[int]) -> Callable:
        """Create a joint-group specific metric function from a base metric function.

        Args:
            base_metric_fn: The original metric function that operates on all joints
            joint_indices: Indices of joints to include in the group

        Returns:
            A new metric function that computes metrics only for specified joints
        """

        def group_metric_fn(full_data: dict[str, torch.Tensor], env_data: dict) -> tuple[float, float]:
            env_idx = env_data["env_idx"]
            num_frames = env_data["num_frames"]

            # Create a cache key for this specific data slice
            cache_key = (env_idx, num_frames, tuple(joint_indices))

            # Check if we have already processed this data
            if cache_key in self._tensor_cache:
                filtered_data = self._tensor_cache[cache_key]
            else:
                # Create filtered data dictionary for all joint-related fields
                filtered_data = {}

                # Filter all joint-related data fields (joint_acc, joint_vel, joint_pos, etc.)
                for data_key in ["joint_acc", "joint_vel", "joint_pos"]:
                    if data_key not in full_data or full_data[data_key].numel() == 0:
                        continue

                    # Get data for this environment, limited to valid frames
                    data_tensor = full_data[data_key][:num_frames, env_idx]  # [num_frames, num_joints, ...]

                    # Make sure we have enough dimensions
                    if data_tensor.dim() < 2:
                        if self.verbose:
                            print(f"Warning: {data_key} tensor has insufficient dimensions: {data_tensor.shape}")
                        continue

                    # Get number of joints and check if we have any valid joints to process
                    num_joints = data_tensor.shape[1] if data_tensor.dim() > 1 else 1
                    valid_indices = [idx for idx in joint_indices if idx < num_joints]

                    if not valid_indices:
                        if self.verbose:
                            print(
                                f"No valid joint indices for environment {env_idx}. Shape: {data_tensor.shape},"
                                f" Max index: {num_joints - 1}, Indices requested: {joint_indices[:5]}..."
                            )
                        continue

                    # Select only the joints in this group
                    filtered_tensor = data_tensor[:, valid_indices]

                    # Add env dimension for compatibility (make it [num_frames, 1, num_joints_in_group, ...])
                    filtered_data[data_key] = filtered_tensor.unsqueeze(1)

                # If we didn't get any data, return None
                if not filtered_data:
                    return None, 0

                # Store in cache for reuse
                self._tensor_cache[cache_key] = filtered_data

            # Call the base metric function with filtered data
            # Set env_idx to 0 as we already filtered data for this env.
            env_data["env_idx"] = 0
            return base_metric_fn(filtered_data, env_data)

        return group_metric_fn

    def register_metric(
        self,
        name: str,
        compute_fn: Callable[[dict[str, torch.Tensor], dict], tuple[float | None, float]],
    ):
        """Register a new metric to compute.

        Args:
            name: Name of the metric
            compute_fn: Function that computes the metric from trajectory data
                        Should accept (data_dict, env_data) and return (value, weight)
        """
        if name in self._metrics_data:
            if self.verbose:
                print(f"Warning: Overwriting existing metric '{name}'")

        # Initialize storage for metric data
        self._metrics_data[name] = {"values": [], "weights": []}
        self._success_metrics_data[name] = {"values": [], "weights": []}

        # Store compute function
        self._compute_functions[name] = compute_fn

    def update(self, terminated_data: dict[str, Any]):
        """Update metrics with data from terminated environments.

        Args:
            terminated_data: Dictionary of trajectory data for terminated environments,
                             including 'frame_counts' tensor indicating valid frame counts
                             for each environment
        """
        if not terminated_data or "frame_counts" not in terminated_data:
            return

        # Get frame counts for each terminated environment
        frame_counts = terminated_data["frame_counts"]
        num_terminated = frame_counts.shape[0]

        if num_terminated == 0:
            return

        # Clear tensor cache to prevent memory build-up
        self._tensor_cache.clear()

        self.num_envs_evaluated += num_terminated

        # Identify successful environments (those that reached max episode length)
        if self.max_episode_length > 0:
            success_mask = frame_counts == self.max_episode_length

            # Handle scalar case (single terminated env)
            if success_mask.dim() == 0:
                num_success = 1 if success_mask.item() else 0
            else:
                success_indices = torch.nonzero(success_mask).squeeze(-1)
                num_success = success_indices.numel()

            self.num_envs_successful += num_success

        # Compute and store metrics for all terminated environments
        for env_idx in range(num_terminated):
            # Extract valid frame count for this environment
            num_frames = frame_counts[env_idx].item()

            if num_frames > 0:
                # Compute and store metrics for this environment
                self._compute_and_store_env_metrics(terminated_data, env_idx, num_frames, self._metrics_data)

                # Also store metrics for successful environments
                if self.max_episode_length > 0 and num_frames == self.max_episode_length:
                    self._compute_and_store_env_metrics(
                        terminated_data, env_idx, num_frames, self._success_metrics_data
                    )

    def _compute_and_store_env_metrics(
        self,
        terminated_data: dict[str, torch.Tensor],
        env_idx: int,
        num_frames: int,
        storage: dict[str, dict[str, list]],
    ):
        """Compute all registered metrics for a single environment and store results.

        Args:
            terminated_data: Dictionary of trajectory data for terminated environments
            env_idx: Index of the environment to compute metrics for
            num_frames: Number of valid frames for this environment
            storage: Dictionary to store metric results
        """
        # Create a data dict specific to this computation
        env_metrics_data = {
            "env_idx": env_idx,
            "num_frames": num_frames,
        }

        # Compute each registered metric
        for metric_name, compute_fn in self._compute_functions.items():
            try:
                value, weight = compute_fn(terminated_data, env_metrics_data)
                if value is not None and weight > 0:
                    storage[metric_name]["values"].append(value)
                    storage[metric_name]["weights"].append(weight)
            except Exception as e:
                print(f"Error computing metric '{metric_name}': {e}")
                raise e

    def conclude(self):
        """Calculate final metrics as weighted averages across all episodes."""
        # Compute weighted averages for all episodes
        self._metrics = {}
        for metric_name, data in self._metrics_data.items():
            if data["values"] and data["weights"]:
                try:
                    self._metrics[metric_name] = np.average(data["values"], weights=data["weights"])
                except Exception as e:
                    print(f"Error calculating weighted average for metric '{metric_name}': {e}")
                    print(f"Values: {data['values'][:5]}... ({len(data['values'])} items)")
                    print(f"Weights: {data['weights'][:5]}... ({len(data['weights'])} items)")

        # Calculate success rate
        if self.num_envs_evaluated > 0:
            self.success_rate = self.num_envs_successful / self.num_envs_evaluated
        else:
            self.success_rate = 0.0

        # Compute weighted averages for successful episodes
        self._success_metrics = {
            metric_name: np.average(data["values"], weights=data["weights"])
            for metric_name, data in self._success_metrics_data.items()
            if data["values"] and data["weights"]
        }

        # Release memory
        self._tensor_cache.clear()

    def print(self, metrics_per_line: int = 6):
        """Print the current metrics to console.

        Args:
            metrics_per_line: Number of metrics to display per line (default: 6)
        """
        print(f"Number of environments evaluated: {self.num_envs_evaluated}")
        print(f"Success Rate: {self.success_rate:.2f}")

        # Pre-compute grouped metrics to avoid repetition
        grouped_all = self._group_metrics_by_part(self._metrics)
        grouped_success = self._group_metrics_by_part(self._success_metrics)

        # Helper function to print metrics in compact format
        def print_compact_metrics(metrics_group, header):
            print(f"{header}:")

            for group_name, group_metrics in metrics_group.items():
                if not group_metrics:
                    continue

                print(f"  {group_name.replace('_', ' ').title()}:", end="")

                # Sort metrics to ensure consistent ordering
                sorted_metrics = sorted(group_metrics.items())

                # Print metrics in chunks of metrics_per_line
                for i in range(0, len(sorted_metrics), metrics_per_line):
                    chunk = sorted_metrics[i : i + metrics_per_line]
                    # First chunk continues on same line, subsequent chunks get indentation
                    if i > 0:
                        print("\n    ", end="")

                    # Print each metric in the chunk
                    metrics_str = " | ".join([f"{name}: {value:.2f}" for name, value in chunk])
                    print(f" {metrics_str}")

        # Print all episodes metrics
        if self._metrics:
            print_compact_metrics(grouped_all, "All episodes metrics")

        # Print successful episodes metrics
        if self._success_metrics:
            print_compact_metrics(grouped_success, "Successful episodes metrics")

    def _group_metrics_by_part(self, metrics_dict):
        """Group metrics by body part for better organization.

        Args:
            metrics_dict: Dictionary of metrics to group

        Returns:
            Dictionary of metrics grouped by body part
        """
        # Group metrics by body part for better organization
        grouped = {}

        # First add the global metrics (no prefix)
        global_metrics = {
            name: value
            for name, value in metrics_dict.items()
            if not any(name.startswith(f"{group}_") for group in self.joint_groups)
        }
        if global_metrics:
            grouped["full_body"] = global_metrics

        # Then add body part specific metrics
        for part_name in self.joint_groups:
            part_metrics = {
                name.replace(f"{part_name}_", ""): value  # Remove prefix for cleaner display
                for name, value in metrics_dict.items()
                if name.startswith(f"{part_name}_")
            }
            if part_metrics:
                grouped[part_name] = part_metrics

        return grouped

    def save(self, directory: str, filename: str | None = None):
        """Save metrics to a file.

        Args:
            directory: Directory to save metrics file to
            filename: Specific filename to use (optional). If provided, will use this exact
                      filename. If not provided, will check for existing metrics files or
                      create a new timestamped one.
        """
        file_dir = Path(directory)
        file_dir.mkdir(parents=True, exist_ok=True)

        if filename:
            # Use the specified filename
            file_path = file_dir / filename
        else:
            # Create a new file with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            file_name = f"smoothness_metrics_{timestamp}.json"
            file_path = file_dir / file_name

        # Organize metrics by body part for cleaner JSON
        all_metrics_grouped = self._group_metrics_by_part(self._metrics)
        success_metrics_grouped = self._group_metrics_by_part(self._success_metrics)

        # For debugging - include raw metrics too
        if self.verbose:
            raw_metrics = {
                "all_raw": {name: float(value) for name, value in self._metrics.items()},
                "success_raw": {name: float(value) for name, value in self._success_metrics.items()},
            }
            debug_info = {"raw_metrics": raw_metrics}
        else:
            debug_info = {}

        content = {
            "num_environments": self.num_envs_evaluated,
            "success_rate": self.success_rate,
            "all": all_metrics_grouped,
            "success": success_metrics_grouped,
            "joint_groups": self.joint_groups,
            **debug_info,
        }

        with open(file_path, "w") as fh:
            json.dump(content, fh, indent=2)

        print(f"Metrics saved to {file_path}")

    def get_metrics(self) -> dict[str, Any]:
        """Get all metrics as a dictionary.

        Returns:
            Dictionary containing all metric data
        """
        # Group metrics by body part for better organization
        all_metrics_grouped = self._group_metrics_by_part(self._metrics)
        success_metrics_grouped = self._group_metrics_by_part(self._success_metrics)

        # Include the raw metrics for debugging
        if self.verbose:
            raw_metrics = {
                "all_raw": self._metrics,
                "success_raw": self._success_metrics,
            }
            debug_info = {"raw_metrics": raw_metrics}
        else:
            debug_info = {}

        return {
            "success_rate": self.success_rate,
            "metrics": all_metrics_grouped,
            "success_metrics": success_metrics_grouped,
            "joint_groups": self.joint_groups,
            **debug_info,
        }

    def _compute_mean_joint_acc(self, full_data: dict[str, torch.Tensor], env_data: dict) -> tuple[float, float]:
        """Compute mean joint acceleration magnitude.

        Args:
            full_data: Full dictionary of trajectory data for all terminated environments
            env_data: Dictionary with environment-specific metadata (env_idx, num_frames)

        Returns:
            Tuple of (metric_value, weight)
        """
        env_idx = env_data["env_idx"]
        num_frames = env_data["num_frames"]

        if "joint_acc" not in full_data or full_data["joint_acc"].numel() == 0:
            return None, 0

        # Get acceleration data for this environment, limited to valid frames
        acc = full_data["joint_acc"][:num_frames, env_idx]

        # Compute absolute acceleration magnitude
        acc_mag = torch.abs(acc)

        # Debug: Print statistics if there are very high values
        max_acc = torch.max(acc_mag).item()
        if max_acc > 500 and self.verbose:
            print(f"\nHigh acceleration detected in env {env_idx}:")
            print(f"  Max: {max_acc:.2f} rad/s², Mean: {torch.mean(acc_mag).item():.2f} rad/s²")
            print(f"  Shape: {acc.shape}, Num frames: {num_frames}")
            # Find which frame and joint has max acceleration
            max_frame_joint = torch.argmax(acc_mag.flatten())
            max_frame = max_frame_joint // (acc.shape[1] if acc.dim() > 1 else 1)
            print(f"  Occurs at frame {max_frame}/{num_frames}")

        # Compute mean across all dimensions
        mean_acc = torch.mean(acc_mag).item()

        # Weight by number of frames
        return mean_acc, num_frames

    def _compute_max_joint_acc(self, full_data: dict[str, torch.Tensor], env_data: dict) -> tuple[float, float]:
        """Compute maximum joint acceleration magnitude.

        Args:
            full_data: Full dictionary of trajectory data for all terminated environments
            env_data: Dictionary with environment-specific metadata (env_idx, num_frames)

        Returns:
            Tuple of (metric_value, weight)
        """
        env_idx = env_data["env_idx"]
        num_frames = env_data["num_frames"]

        if "joint_acc" not in full_data or full_data["joint_acc"].numel() == 0:
            return None, 0

        # Get acceleration data for this environment, limited to valid frames
        acc = full_data["joint_acc"][:num_frames, env_idx]

        # Compute absolute acceleration magnitude
        acc_mag = torch.abs(acc)

        # Compute maximum across all dimensions
        max_acc = torch.max(acc_mag).item()

        # Weight by 1 (each max value has equal weight)
        return max_acc, 1

    def _compute_mean_acc_rate(self, full_data: dict[str, torch.Tensor], env_data: dict) -> tuple[float, float]:
        """Compute mean acceleration rate (jerk) magnitude.

        Args:
            full_data: Full dictionary of trajectory data for all terminated environments
            env_data: Dictionary with environment-specific metadata (env_idx, num_frames)

        Returns:
            Tuple of (metric_value, weight)
        """
        env_idx = env_data["env_idx"]
        num_frames = env_data["num_frames"]

        if "joint_acc" not in full_data or full_data["joint_acc"].numel() == 0 or num_frames < 2:
            return None, 0

        # Get acceleration data for this environment, limited to valid frames
        acc = full_data["joint_acc"][:num_frames, env_idx]

        # Compute jerk as the difference between consecutive accelerations
        jerk = acc[1:] - acc[:-1]

        # Compute absolute jerk magnitude
        jerk_mag = torch.abs(jerk)

        # Compute mean across all dimensions
        mean_jerk = torch.mean(jerk_mag).item()

        # Weight by number of jerk frames
        return mean_jerk, num_frames - 1

    def _compute_max_acc_rate(self, full_data: dict[str, torch.Tensor], env_data: dict) -> tuple[float, float]:
        """Compute maximum acceleration rate (jerk) magnitude.

        Args:
            full_data: Full dictionary of trajectory data for all terminated environments
            env_data: Dictionary with environment-specific metadata (env_idx, num_frames)

        Returns:
            Tuple of (metric_value, weight)
        """
        env_idx = env_data["env_idx"]
        num_frames = env_data["num_frames"]

        if "joint_acc" not in full_data or full_data["joint_acc"].numel() == 0 or num_frames < 2:
            return None, 0

        # Get acceleration data for this environment, limited to valid frames
        acc = full_data["joint_acc"][:num_frames, env_idx]

        # Compute jerk as the difference between consecutive accelerations
        jerk = acc[1:] - acc[:-1]

        # Compute absolute jerk magnitude
        jerk_mag = torch.abs(jerk)

        # Compute maximum across all dimensions
        max_jerk = torch.max(jerk_mag).item()

        # Weight by 1 (each max value has equal weight)
        return max_jerk, 1

    def _compute_mean_joint_vel(self, full_data: dict[str, torch.Tensor], env_data: dict) -> tuple[float, float]:
        """Compute mean joint velocity magnitude.

        Args:
            full_data: Full dictionary of trajectory data for all terminated environments
            env_data: Dictionary with environment-specific metadata (env_idx, num_frames)

        Returns:
            Tuple of (metric_value, weight)
        """
        env_idx = env_data["env_idx"]
        num_frames = env_data["num_frames"]

        if "joint_vel" not in full_data:
            if self.verbose and env_idx == 0:  # Only print once
                print(f"Warning: 'joint_vel' not found in episode data. Available keys: {list(full_data.keys())}")
            return None, 0

        if full_data["joint_vel"].numel() == 0:
            return None, 0

        # Get velocity data for this environment, limited to valid frames
        vel = full_data["joint_vel"][:num_frames, env_idx]

        # Compute absolute velocity magnitude
        vel_mag = torch.abs(vel)

        # Compute mean across all dimensions
        mean_vel = torch.mean(vel_mag).item()

        # Weight by number of frames
        return mean_vel, num_frames

    def _compute_max_joint_vel(self, full_data: dict[str, torch.Tensor], env_data: dict) -> tuple[float, float]:
        """Compute maximum joint velocity magnitude.

        Args:
            full_data: Full dictionary of trajectory data for all terminated environments
            env_data: Dictionary with environment-specific metadata (env_idx, num_frames)

        Returns:
            Tuple of (metric_value, weight)
        """
        env_idx = env_data["env_idx"]
        num_frames = env_data["num_frames"]

        if "joint_vel" not in full_data or full_data["joint_vel"].numel() == 0:
            return None, 0

        # Get velocity data for this environment, limited to valid frames
        vel = full_data["joint_vel"][:num_frames, env_idx]

        # Compute absolute velocity magnitude
        vel_mag = torch.abs(vel)

        # Compute maximum across all dimensions
        max_vel = torch.max(vel_mag).item()

        # Weight by 1 (each max value has equal weight)
        return max_vel, 1
