# mypy: ignore-errors
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

#!/usr/bin/env python3
"""Comparison test for all three velocity profiles using actual implementations.

This test imports the actual velocity profile implementations directly,
avoiding code duplication.
"""

# ruff: noqa

import argparse
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


# Direct file imports to avoid Isaac Sim dependencies
def load_module_from_file(module_name: str, file_path: str, package_name: str | None = None):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path, submodule_search_locations=[])
    module = importlib.util.module_from_spec(spec)

    # Set package for relative imports
    if package_name:
        module.__package__ = package_name

    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Get the directory of velocity_profiles
vp_dir = os.path.dirname(os.path.abspath(__file__))

# Create a fake package structure for relative imports to work
package_name = "agile.rl_env.mdp.actions.velocity_profiles"

# Load modules in dependency order
configs = load_module_from_file(f"{package_name}.configs", os.path.join(vp_dir, "configs.py"), package_name)
base = load_module_from_file(f"{package_name}.base", os.path.join(vp_dir, "base.py"), package_name)
ema = load_module_from_file(f"{package_name}.ema", os.path.join(vp_dir, "ema.py"), package_name)
linear = load_module_from_file(f"{package_name}.linear", os.path.join(vp_dir, "linear.py"), package_name)
trap = load_module_from_file(f"{package_name}.trapezoidal", os.path.join(vp_dir, "trapezoidal.py"), package_name)

# Extract the classes we need
EMAVelocityProfileCfg = configs.EMAVelocityProfileCfg
LinearVelocityProfileCfg = configs.LinearVelocityProfileCfg
TrapezoidalVelocityProfileCfg = configs.TrapezoidalVelocityProfileCfg

EMAVelocityProfile = ema.EMAVelocityProfile
LinearVelocityProfile = linear.LinearVelocityProfile
TrapezoidalVelocityProfile = trap.TrapezoidalVelocityProfile


def simulate_profile(
    profile: EMAVelocityProfile | LinearVelocityProfile | TrapezoidalVelocityProfile,
    dt: float = 0.01,
    max_steps: int = 500,
):
    """Simulate a single velocity profile and collect data.

    Args:
        profile: Velocity profile instance.
        dt: Time step in seconds.
        max_steps: Maximum simulation steps.

    Returns:
        Dictionary with pos, vel, acc, and time arrays.
    """
    data = {"pos": [], "vel": [], "acc": [], "time": []}

    for i in range(max_steps):
        if not profile._is_active.any():
            break

        t = i * dt
        pos = profile.compute_next_position(dt)
        vel = profile.get_current_velocity()
        acc = profile.get_current_acceleration()

        data["pos"].append(pos.clone())
        data["vel"].append(vel.clone())
        data["acc"].append(acc.clone())
        data["time"].append(t)

    # Convert to numpy
    if len(data["pos"]) > 0:
        data["pos"] = torch.stack(data["pos"]).numpy()
        data["vel"] = torch.stack(data["vel"]).numpy()
        data["acc"] = torch.stack(data["acc"]).numpy()

    return data


def run_comparison_test() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], torch.Tensor]:
    """Run comparison test for all three profiles."""

    print("=" * 70)
    print("Velocity Profile Comparison: EMA vs Linear vs Trapezoidal")
    print("Using actual implementations from velocity_profiles module")
    print("=" * 70)

    # Test parameters
    num_envs = 1
    num_joints = 2
    device = torch.device("cpu")

    # Joint limits
    joint_limits = torch.tensor([[[-2.0, 2.0], [-1.5, 1.5]]])

    # Velocity limits
    velocity_limits = torch.tensor([[1.5, 1.0]])

    # Initial and target positions
    current_pos = torch.tensor([[0.0, 0.0]])
    target_pos = torch.tensor([[1.5, -1.0]])  # Different distances for each joint

    print("\nTest Setup:")
    print(f"  Number of joints: {num_joints}")
    print(f"  Initial positions: {current_pos[0].tolist()}")
    print(f"  Target positions: {target_pos[0].tolist()}")
    print(f"  Distances: {(target_pos - current_pos)[0].tolist()}")

    # Create configurations (fixed parameters for fair comparison)
    ema_cfg = EMAVelocityProfileCfg(ema_coefficient_range=(0.05, 0.05))
    linear_cfg = LinearVelocityProfileCfg(velocity_range=(1.0, 1.0))
    trap_cfg = TrapezoidalVelocityProfileCfg(
        acceleration_range=(2.0, 2.0),
        max_velocity_range=(1.2, 1.2),
        deceleration_range=(2.0, 2.0),
        use_smooth_start=False,
    )

    # Create profiles using actual implementations
    print("\nCreating profiles...")
    ema_profile = EMAVelocityProfile(ema_cfg, num_envs, num_joints, device, joint_limits, velocity_limits)
    linear_profile = LinearVelocityProfile(linear_cfg, num_envs, num_joints, device, joint_limits, velocity_limits)
    trap_profile = TrapezoidalVelocityProfile(trap_cfg, num_envs, num_joints, device, joint_limits, velocity_limits)

    # Set targets
    ema_profile.set_target(current_pos, target_pos)
    linear_profile.set_target(current_pos, target_pos)
    trap_profile.set_target(current_pos, target_pos)

    # Simulate all profiles
    print("\nSimulating profiles...")
    dt = 0.01

    ema_data = simulate_profile(ema_profile, dt)
    linear_data = simulate_profile(linear_profile, dt)
    trap_data = simulate_profile(trap_profile, dt)

    print(f"  EMA profile: {len(ema_data['time'])} steps, {ema_data['time'][-1]:.2f}s")
    print(f"  Linear profile: {len(linear_data['time'])} steps, {linear_data['time'][-1]:.2f}s")
    print(f"  Trapezoidal profile: {len(trap_data['time'])} steps, {trap_data['time'][-1]:.2f}s")

    return ema_data, linear_data, trap_data, target_pos


def plot_comparison(
    ema_data: dict[str, np.ndarray],
    linear_data: dict[str, np.ndarray],
    trap_data: dict[str, np.ndarray],
    target_pos: torch.Tensor,
    save_figure: bool = False,
):
    """Create comprehensive comparison plots.

    Args:
        ema_data: EMA profile trajectory data.
        linear_data: Linear profile trajectory data.
        trap_data: Trapezoidal profile trajectory data.
        target_pos: Target positions.
        save_figure: Whether to save the figure to file.
    """

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Velocity Profile Comparison: EMA vs Linear vs Trapezoidal (2 Joints)", fontsize=16, fontweight="bold")

    joint_names = ["Joint 0", "Joint 1"]
    profile_colors = {"EMA": "blue", "Linear": "green", "Trapezoidal": "red"}

    # Position plots for each joint
    for j in range(2):
        ax = plt.subplot(3, 4, j + 1)
        ax.plot(
            ema_data["time"],
            ema_data["pos"][:, 0, j],
            "-",
            color=profile_colors["EMA"],
            linewidth=2,
            label="EMA",
            alpha=0.8,
        )
        ax.plot(
            linear_data["time"],
            linear_data["pos"][:, 0, j],
            "-",
            color=profile_colors["Linear"],
            linewidth=2,
            label="Linear",
            alpha=0.8,
        )
        ax.plot(
            trap_data["time"],
            trap_data["pos"][:, 0, j],
            "-",
            color=profile_colors["Trapezoidal"],
            linewidth=2,
            label="Trapezoidal",
            alpha=0.8,
        )
        ax.axhline(y=target_pos[0, j].item(), color="k", linestyle="--", alpha=0.3, label="Target")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (rad)")
        ax.set_title(f"{joint_names[j]} - Position")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Velocity plots for each joint
    for j in range(2):
        ax = plt.subplot(3, 4, j + 5)
        ax.plot(
            ema_data["time"],
            ema_data["vel"][:, 0, j],
            "-",
            color=profile_colors["EMA"],
            linewidth=2,
            label="EMA",
            alpha=0.8,
        )
        ax.plot(
            linear_data["time"],
            linear_data["vel"][:, 0, j],
            "-",
            color=profile_colors["Linear"],
            linewidth=2,
            label="Linear",
            alpha=0.8,
        )
        ax.plot(
            trap_data["time"],
            trap_data["vel"][:, 0, j],
            "-",
            color=profile_colors["Trapezoidal"],
            linewidth=2,
            label="Trapezoidal",
            alpha=0.8,
        )
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (rad/s)")
        ax.set_title(f"{joint_names[j]} - Velocity")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Acceleration plots for each joint
    for j in range(2):
        ax = plt.subplot(3, 4, j + 9)
        ax.plot(
            ema_data["time"],
            ema_data["acc"][:, 0, j],
            "-",
            color=profile_colors["EMA"],
            linewidth=2,
            label="EMA",
            alpha=0.8,
        )
        ax.plot(
            linear_data["time"],
            linear_data["acc"][:, 0, j],
            "-",
            color=profile_colors["Linear"],
            linewidth=2,
            label="Linear",
            alpha=0.8,
        )
        ax.plot(
            trap_data["time"],
            trap_data["acc"][:, 0, j],
            "-",
            color=profile_colors["Trapezoidal"],
            linewidth=2,
            label="Trapezoidal",
            alpha=0.8,
        )
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (rad/s²)")
        ax.set_title(f"{joint_names[j]} - Acceleration")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Phase portraits (Velocity vs Position)
    for j in range(2):
        ax = plt.subplot(3, 4, j + 3)
        ax.plot(
            ema_data["pos"][:, 0, j],
            ema_data["vel"][:, 0, j],
            "-",
            color=profile_colors["EMA"],
            linewidth=2,
            label="EMA",
            alpha=0.7,
        )
        ax.plot(
            linear_data["pos"][:, 0, j],
            linear_data["vel"][:, 0, j],
            "-",
            color=profile_colors["Linear"],
            linewidth=2,
            label="Linear",
            alpha=0.7,
        )
        ax.plot(
            trap_data["pos"][:, 0, j],
            trap_data["vel"][:, 0, j],
            "-",
            color=profile_colors["Trapezoidal"],
            linewidth=2,
            label="Trapezoidal",
            alpha=0.7,
        )
        ax.scatter([0], [0], color="green", s=100, zorder=5, marker="o", label="Start")
        ax.scatter([target_pos[0, j].item()], [0], color="red", s=100, zorder=5, marker="x", label="Target")
        ax.set_xlabel("Position (rad)")
        ax.set_ylabel("Velocity (rad/s)")
        ax.set_title(f"{joint_names[j]} - Phase Portrait")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Summary statistics
    ax = plt.subplot(3, 4, 11)
    ax.axis("off")

    stats_text = "Profile Characteristics:\n\n"
    stats_text += "EMA (Exponential):\n"
    stats_text += f"  • Completion: {ema_data['time'][-1]:.2f}s\n"
    stats_text += f"  • Max vel J0: {np.abs(ema_data['vel'][:, 0, 0]).max():.2f} rad/s\n"
    stats_text += f"  • Max vel J1: {np.abs(ema_data['vel'][:, 0, 1]).max():.2f} rad/s\n"
    stats_text += "  • Smooth exponential decay\n\n"

    stats_text += "Linear (Constant Vel):\n"
    stats_text += f"  • Completion: {linear_data['time'][-1]:.2f}s\n"
    stats_text += f"  • Max vel J0: {np.abs(linear_data['vel'][:, 0, 0]).max():.2f} rad/s\n"
    stats_text += f"  • Max vel J1: {np.abs(linear_data['vel'][:, 0, 1]).max():.2f} rad/s\n"
    stats_text += "  • Constant velocity\n\n"

    stats_text += "Trapezoidal:\n"
    stats_text += f"  • Completion: {trap_data['time'][-1]:.2f}s\n"
    stats_text += f"  • Max vel J0: {np.abs(trap_data['vel'][:, 0, 0]).max():.2f} rad/s\n"
    stats_text += f"  • Max vel J1: {np.abs(trap_data['vel'][:, 0, 1]).max():.2f} rad/s\n"
    stats_text += "  • Controlled accel/decel\n"

    ax.text(
        0.1,
        0.5,
        stats_text,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),  # noqa: C408
    )

    # Comparison summary
    ax = plt.subplot(3, 4, 12)
    ax.axis("off")

    summary = "Key Differences:\n\n"
    summary += "1. EMA Profile:\n"
    summary += "   - Exponential approach\n"
    summary += "   - Velocity decreases over time\n"
    summary += "   - Smooth, natural motion\n\n"

    summary += "2. Linear Profile:\n"
    summary += "   - Constant velocity\n"
    summary += "   - Predictable timing\n"
    summary += "   - Sharp start/stop\n\n"

    summary += "3. Trapezoidal:\n"
    summary += "   - Three distinct phases\n"
    summary += "   - Smooth acceleration\n"
    summary += "   - Time-optimal motion\n"

    ax.text(
        0.1,
        0.5,
        summary,
        fontsize=9,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),  # noqa: C408
    )

    plt.tight_layout()

    if save_figure:
        plt.savefig("velocity_profiles_comparison.png", dpi=150, bbox_inches="tight")
        print("\n" + "=" * 70)
        print("Plot saved as 'velocity_profiles_comparison.png'")
        print("=" * 70)

    plt.show()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compare velocity profiles: EMA, Linear, and Trapezoidal")
    parser.add_argument(
        "-s",
        "--save-figure",
        action="store_true",
        default=False,
        help="Save the comparison figure to file (default: False)",
    )
    args = parser.parse_args()

    # Run the test
    ema_data, linear_data, trap_data, target_pos = run_comparison_test()
    plot_comparison(ema_data, linear_data, trap_data, target_pos, save_figure=args.save_figure)

    print("\nTest completed successfully!")
    print("\nKey Takeaways:")
    print("- EMA: Best for smooth, natural-looking motion")
    print("- Linear: Best for predictable, time-based control")
    print("- Trapezoidal: Best for efficient, physically realistic motion")
