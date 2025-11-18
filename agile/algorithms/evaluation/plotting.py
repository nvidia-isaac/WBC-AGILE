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


"""Visualization tools for trajectory data analysis.

This module is standalone and can be used without Isaac Sim or other dependencies.
It only requires: pandas, matplotlib, seaborn, numpy
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Set seaborn style
sns.set_style("whitegrid")

# Define consistent color palette for joints
JOINT_COLORS = sns.color_palette("husl", 30)  # Up to 30 unique colors for joints


# ============================================================================
# Data Loading Functions (Standalone - No TrajectoryLogger dependency)
# ============================================================================


def load_metadata(trajectory_dir: str | Path) -> dict:
    """Load metadata about joints and robot configuration.

    Args:
        trajectory_dir: Path to directory containing trajectory files

    Returns:
        Dictionary with metadata including joint_names, joint_pos_limits, etc.

    Example:
        >>> metadata = load_metadata("logs/rsl_rl/experiment")
        >>> joint_names = metadata['joint_names']
        >>> print(f"Joint 0: {joint_names[0]}")
    """
    trajectory_dir = Path(trajectory_dir)

    # Check if we're in the parent directory or trajectories directory
    if (trajectory_dir / "trajectories").exists():
        trajectory_dir = trajectory_dir / "trajectories"

    metadata_path = trajectory_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"Warning: metadata.json not found at {metadata_path}")
        print("Metadata is available for evaluations run after adding metadata logging.")
        return {}

    with open(metadata_path) as f:
        return json.load(f)


def load_episode(trajectory_dir: str | Path, episode_id: int) -> pd.DataFrame:
    """Load a single episode's trajectory data directly from parquet file.

    Args:
        trajectory_dir: Path to directory containing trajectory parquet files
                       (should have trajectories/ subfolder or be the trajectories folder itself)
        episode_id: Episode ID to load

    Returns:
        DataFrame with episode trajectory data

    Example:
        >>> df = load_episode("logs/rsl_rl/experiment/trajectories", episode_id=0)
        >>> # or
        >>> df = load_episode("logs/rsl_rl/experiment", episode_id=0)
    """
    trajectory_dir = Path(trajectory_dir)

    # Check if we're in the parent directory or trajectories directory
    if (trajectory_dir / "trajectories").exists():
        trajectory_dir = trajectory_dir / "trajectories"

    filename = f"episode_{episode_id:03d}.parquet"
    filepath = trajectory_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Episode {episode_id} not found at {filepath}")

    return pd.read_parquet(filepath)


def load_all_episodes(trajectory_dir: str | Path) -> pd.DataFrame:
    """Load all episodes from trajectory directory.

    Args:
        trajectory_dir: Path to directory containing trajectory parquet files

    Returns:
        DataFrame with all episodes concatenated

    Example:
        >>> all_df = load_all_episodes("logs/rsl_rl/experiment/trajectories")
    """
    trajectory_dir = Path(trajectory_dir)

    # Check if we're in the parent directory or trajectories directory
    if (trajectory_dir / "trajectories").exists():
        trajectory_dir = trajectory_dir / "trajectories"

    parquet_files = sorted(trajectory_dir.glob("episode_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No episode files found in {trajectory_dir}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True)


# ============================================================================
# Plotting Functions
# ============================================================================


def plot_joint_trajectories(
    data: pd.DataFrame,
    joint_names: list[str] | None = None,
    joint_indices: list[int] | None = None,
    metadata: dict | None = None,
    show_limits: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot position, velocity, and acceleration for multiple joints.

    Creates a figure with 3 rows (position, velocity, acceleration) and one column
    per joint. Each joint maintains a consistent color across all subplots.

    Args:
        data: DataFrame with trajectory data (use load_episode() to load from file)
        joint_names: List of joint names to plot (e.g., ['left_hip_yaw_joint', 'right_knee_joint'])
                    Either joint_names or joint_indices must be provided.
        joint_indices: List of joint indices to plot (e.g., [0, 1, 2]). Used if joint_names not provided.
        metadata: Metadata dictionary from load_metadata(). Required if using joint_names or show_limits.
        show_limits: Whether to draw horizontal limit lines (requires metadata)
        figsize: Figure size (width, height). Auto-calculated if None

    Returns:
        (fig, axes): Matplotlib figure and 2D array of axes [3, num_joints]
                     rows: [pos, vel, acc], cols: [joint0, joint1, ...]

    Example:
        >>> from agile.algorithms.evaluation.plotting import load_episode, load_metadata, plot_joint_trajectories
        >>> metadata = load_metadata("logs/rsl_rl/experiment")
        >>> df = load_episode("logs/rsl_rl/experiment", episode_id=0)
        >>> # Use joint names (recommended)
        >>> fig, axes = plot_joint_trajectories(
        ...     df,
        ...     joint_names=['left_hip_yaw_joint', 'right_knee_joint'],
        ...     metadata=metadata,
        ...     show_limits=True
        ... )
        >>> # Or use indices
        >>> fig, axes = plot_joint_trajectories(df, joint_indices=[0, 5, 10], metadata=metadata)
        >>> plt.show()
    """
    df = data

    # Convert joint names to indices if provided
    joint_name_map = {}
    if joint_names is not None:
        if metadata is None:
            raise ValueError("metadata must be provided when using joint_names")

        # Convert joint names to indices
        all_joint_names = metadata.get("joint_names", [])
        if not all_joint_names:
            raise ValueError("Metadata does not contain 'joint_names'")

        joint_indices = []
        for name in joint_names:
            if name not in all_joint_names:
                raise ValueError(f"Joint name '{name}' not found in metadata. Available: {all_joint_names}")
            idx = all_joint_names.index(name)
            joint_indices.append(idx)
            joint_name_map[idx] = name
    elif joint_indices is not None:
        # Create name map from metadata if available
        if metadata and "joint_names" in metadata:
            all_joint_names = metadata["joint_names"]
            for idx in joint_indices:
                if idx < len(all_joint_names):
                    joint_name_map[idx] = all_joint_names[idx]
    else:
        raise ValueError("Either joint_names or joint_indices must be provided")

    num_joints = len(joint_indices)

    # Auto-calculate figure size if not provided
    if figsize is None:
        width = max(12, num_joints * 4)
        height = 10
        figsize = (width, height)

    # Create subplots: 3 rows (pos, vel, acc) x num_joints columns
    fig, axes = plt.subplots(3, num_joints, figsize=figsize, sharex=True)

    # Handle single joint case (axes won't be 2D)
    if num_joints == 1:
        axes = axes.reshape(-1, 1)

    timestep = df["timestep"].values

    # Get limits from metadata if available
    joint_pos_limits_data = metadata.get("joint_pos_limits", []) if metadata else []
    joint_vel_limits_data = metadata.get("joint_vel_limits", []) if metadata else []

    for col_idx, joint_idx in enumerate(joint_indices):
        color = JOINT_COLORS[joint_idx % len(JOINT_COLORS)]

        # Get display name (use joint name if available, otherwise index)
        if joint_idx in joint_name_map:
            display_name = joint_name_map[joint_idx]
        else:
            display_name = f"Joint {joint_idx}"

        # Position
        pos_col = f"joint_pos_{joint_idx}"
        if pos_col in df.columns:
            axes[0, col_idx].plot(timestep, df[pos_col], color=color, linewidth=1.5)
            axes[0, col_idx].set_ylabel("Position (rad)", fontsize=10)
            axes[0, col_idx].set_title(display_name, fontsize=11, fontweight="bold")

            # Add limits from metadata if available
            if show_limits and joint_idx < len(joint_pos_limits_data):
                limits = joint_pos_limits_data[joint_idx]
                lower_limit, upper_limit = limits[0], limits[1]
                axes[0, col_idx].axhline(upper_limit, color="r", linestyle="--", linewidth=1, alpha=0.7, label="Limits")
                axes[0, col_idx].axhline(lower_limit, color="r", linestyle="--", linewidth=1, alpha=0.7)

        # Velocity
        vel_col = f"joint_vel_{joint_idx}"
        if vel_col in df.columns:
            axes[1, col_idx].plot(timestep, df[vel_col], color=color, linewidth=1.5)
            axes[1, col_idx].set_ylabel("Velocity (rad/s)", fontsize=10)

            # Add velocity limits from metadata if available
            if show_limits and joint_idx < len(joint_vel_limits_data):
                vel_limit = joint_vel_limits_data[joint_idx]
                axes[1, col_idx].axhline(vel_limit, color="r", linestyle="--", linewidth=1, alpha=0.7)
                axes[1, col_idx].axhline(-vel_limit, color="r", linestyle="--", linewidth=1, alpha=0.7)

        # Acceleration
        acc_col = f"joint_acc_{joint_idx}"
        if acc_col in df.columns:
            axes[2, col_idx].plot(timestep, df[acc_col], color=color, linewidth=1.5)
            axes[2, col_idx].set_ylabel("Acceleration (rad/s²)", fontsize=10)
            axes[2, col_idx].set_xlabel("Time (s)", fontsize=10)

    # Overall title
    episode_id = df["episode_id"].iloc[0] if "episode_id" in df.columns else "Unknown"
    success = df["is_success"].iloc[0] if "is_success" in df.columns else False
    status = "Success" if success else "Failed"
    fig.suptitle(f"Joint Trajectories - Episode {episode_id} ({status})", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig, axes


def plot_tracking_performance(
    data: pd.DataFrame,
    quantities: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, np.ndarray] | tuple[None, None]:
    """Plot actual vs commanded for tracking performance.

    Args:
        data: DataFrame with trajectory data (use load_episode() to load from file)
        quantities: Which quantities to plot. Options:
                   - 'root_lin_vel_x': root linear velocity x
                   - 'root_lin_vel_y': root linear velocity y
                   - 'root_ang_vel_z': root angular velocity z
                   - 'root_height': root position z
                   If None, plots all available quantities
        figsize: Figure size (width, height). Auto-calculated if None

    Returns:
        (fig, axes): Figure and axes array, or (None, None) if command data unavailable

    Example:
        >>> from agile.algorithms.evaluation.plotting import load_episode, plot_tracking_performance
        >>> df = load_episode("logs/rsl_rl/experiment", episode_id=0)
        >>> fig, axes = plot_tracking_performance(df, quantities=['root_lin_vel_x', 'root_lin_vel_y'])
        >>> if fig: plt.show()
    """
    df = data

    # Check if command data is available (try both singular and plural naming)
    command_prefix = None
    if "commands_0" in df.columns:
        command_prefix = "commands"
    elif "command_0" in df.columns:
        command_prefix = "command"
    else:
        print("Warning: Command data not available in trajectory. Cannot plot tracking performance.")
        return None, None

    # Define quantity mappings: quantity_name -> (actual_column, command_index, ylabel, title)
    # Prefer robot frame velocities (aligned with commands), fallback to world frame for backward compatibility
    quantity_map = {
        "root_lin_vel_x": (
            "root_lin_vel_robot_0" if "root_lin_vel_robot_0" in df.columns else "root_lin_vel_0",
            0,
            "Velocity (m/s)",
            "Linear Velocity X (Forward)",
        ),
        "root_lin_vel_y": (
            "root_lin_vel_robot_1" if "root_lin_vel_robot_1" in df.columns else "root_lin_vel_1",
            1,
            "Velocity (m/s)",
            "Linear Velocity Y (Left)",
        ),
        "root_ang_vel_z": (
            "root_ang_vel_2",  # Yaw rate is same in both world and robot frames
            2,
            "Angular Vel (rad/s)",
            "Angular Velocity Z (Yaw Rate)",
        ),
        "root_height": ("root_pos_2", 3, "Height (m)", "Root Height"),
    }

    # If no quantities specified, plot all available
    if quantities is None:
        quantities = list(quantity_map.keys())

    # Filter to only quantities that have data available
    available_quantities = []
    for qty in quantities:
        if qty in quantity_map:
            actual_col, _, _, _ = quantity_map[qty]
            if actual_col in df.columns:
                available_quantities.append(qty)

    if not available_quantities:
        print("Warning: No tracking quantities available in data")
        return None, None

    num_plots = len(available_quantities)

    # Auto-calculate figure size
    if figsize is None:
        figsize = (14, 4 * num_plots)

    # Create subplots
    fig, axes = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)

    # Handle single plot case
    if num_plots == 1:
        axes = [axes]

    timestep = df["timestep"].values

    for idx, qty in enumerate(available_quantities):
        actual_col, cmd_idx, ylabel, title = quantity_map[qty]
        cmd_col = f"{command_prefix}_{cmd_idx}"

        # Plot actual vs commanded
        axes[idx].plot(timestep, df[actual_col], label="Actual", color="blue", linewidth=2, alpha=0.8)
        axes[idx].plot(timestep, df[cmd_col], label="Commanded", color="red", linewidth=1.5, linestyle="--", alpha=0.7)

        axes[idx].set_ylabel(ylabel, fontsize=11)
        axes[idx].set_title(title, fontsize=12, fontweight="bold")
        axes[idx].legend(loc="upper right", fontsize=10)
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=11)

    # Overall title
    episode_id = df["episode_id"].iloc[0] if "episode_id" in df.columns else "Unknown"
    success = df["is_success"].iloc[0] if "is_success" in df.columns else False
    status = "Success" if success else "Failed"
    fig.suptitle(f"Tracking Performance - Episode {episode_id} ({status})", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig, axes


def calculate_tracking_errors(df: pd.DataFrame) -> dict:
    """Calculate tracking error statistics for any command structure.

    Works with velocity-only (3 fields) or velocity+height (4 fields) commands.
    Only calculates errors for command fields that exist in the DataFrame.

    Args:
        df: DataFrame with trajectory data including commands and actual state

    Returns:
        Dictionary with error statistics for each tracking quantity:
        {"lin_vel_x": {"mean": ..., "max": ..., "rms": ...}, ...}
        Returns empty dict if command data not available.

    Example:
        >>> df = load_episode("logs/evaluation/task", episode_id=0)
        >>> errors = calculate_tracking_errors(df)
        >>> print(f"X velocity mean error: {errors['lin_vel_x']['mean']:.3f}")
    """
    errors = {}

    # Check if command data available
    command_prefix = "commands" if "commands_0" in df.columns else ("command" if "command_0" in df.columns else None)

    if command_prefix:
        # Prefer robot frame velocities (aligned with commands), fallback to world frame
        # Calculate errors for each tracking quantity
        for name, (actual_col, cmd_idx) in [
            (
                "lin_vel_x",
                ("root_lin_vel_robot_0" if "root_lin_vel_robot_0" in df.columns else "root_lin_vel_0", 0),
            ),
            (
                "lin_vel_y",
                ("root_lin_vel_robot_1" if "root_lin_vel_robot_1" in df.columns else "root_lin_vel_1", 1),
            ),
            (
                "ang_vel_z",
                ("root_ang_vel_2", 2),  # Yaw rate is same in both world and robot frames
            ),
            ("height", ("root_pos_2", 3)),
        ]:
            cmd_col = f"{command_prefix}_{cmd_idx}"
            if actual_col in df.columns and cmd_col in df.columns:
                error = (df[actual_col] - df[cmd_col]).abs()
                errors[name] = {
                    "mean": error.mean(),
                    "std": error.std(),
                    "max": error.max(),
                    "rms": (error**2).mean() ** 0.5,
                }

    return errors


def plot_all_episodes_summary(
    trajectory_dir: str | Path,
    metric: str = "joint_vel_0",
    figsize: tuple[float, float] = (12, 6),
) -> tuple[Figure, Axes]:
    """Create summary plot comparing a metric across all episodes.

    Args:
        trajectory_dir: Path to directory containing trajectory parquet files
        metric: Column name to plot (e.g., 'joint_vel_0', 'joint_acc_5')
        figsize: Figure size

    Returns:
        (fig, ax): Figure and axes

    Example:
        >>> from agile.algorithms.evaluation.plotting import plot_all_episodes_summary
        >>> fig, ax = plot_all_episodes_summary("logs/rsl_rl/experiment", metric='joint_acc_0')
        >>> plt.show()
    """
    # Load all episodes
    all_data = load_all_episodes(trajectory_dir)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each episode separately
    for episode_id in all_data["episode_id"].unique():
        episode_data = all_data[all_data["episode_id"] == episode_id]
        success = episode_data["is_success"].iloc[0]
        color = "green" if success else "red"
        alpha = 0.7 if success else 0.4

        ax.plot(
            episode_data["timestep"],
            episode_data[metric],
            color=color,
            alpha=alpha,
            linewidth=1,
            label=f"Ep {episode_id}" if episode_id < 5 else "",  # Only label first few
        )

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"All Episodes: {metric}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add legend for success/failure
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", alpha=0.7, label="Success"),
        Patch(facecolor="red", alpha=0.4, label="Failed"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    return fig, ax
