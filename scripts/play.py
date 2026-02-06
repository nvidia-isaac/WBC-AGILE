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
"""Play an environment with sinusoidal actions to validate configuration (no policy required)."""

# flake8: noqa

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#

parser = argparse.ArgumentParser(description="Play an IsaacLab environment with sinusoidal actions (no policy).")
parser.add_argument("--task", type=str, required=True, help="Gym task ID to load (if registered).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric (use USD I/O).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run close to real-time if possible.")
parser.add_argument(
    "--validate-fallen-states",
    action="store_true",
    default=False,
    help="Validate fallen state dataset by visualizing collected poses with zero actions.",
)

# Isaac Sim app args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Always enable cameras for video
if args_cli.video:
    args_cli.enable_cameras = True

# -----------------------------------------------------------------------------#
# Launch app
# -----------------------------------------------------------------------------#

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Omniverse/Isaac imports must happen AFTER SimulationApp instantiation
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402

# Ensure task modules are imported/registered
import agile.rl_env.tasks  # noqa: F401, E402
import isaaclab_tasks  # noqa: F401, E402

# -----------------------------------------------------------------------------#
# Helper Functions
# -----------------------------------------------------------------------------#


def prepare_env_for_playing(env_cfg: ManagerBasedRLEnvCfg) -> ManagerBasedRLEnvCfg:
    """Prepare environment for interactive playing (remove training-specific components)."""
    # Remove curriculum
    env_cfg.curriculum = None

    # Remove harness
    if hasattr(env_cfg.actions, "harness"):
        del env_cfg.actions.harness

    # Remove random upper body motion
    if hasattr(env_cfg.actions, "random_pos"):
        del env_cfg.actions.random_pos

    return env_cfg


def generate_sinusoidal_actions(timestep: int, num_envs: int, action_dim: int, dt: float) -> np.ndarray:
    """Generate smooth sinusoidal trajectory for arm actions.

    Args:
        timestep: Current timestep counter
        num_envs: Number of parallel environments
        action_dim: Dimension of action space
        dt: Time step in seconds

    Returns:
        Array of shape (num_envs, action_dim) with sinusoidal actions in range [-0.5, 0.5]
    """
    time_elapsed = timestep * dt
    actions = np.zeros((num_envs, action_dim), dtype=np.float32)

    # Generate sinusoidal motion with different frequencies for each joint
    # to create more natural, non-repetitive motion
    for i in range(action_dim):
        # Use different frequencies for different joints (0.5 Hz to 1.5 Hz)
        frequency = 0.5 + (i % 5) * 0.2
        # Use different phase offsets to avoid all joints moving in sync
        phase_offset = i * np.pi / action_dim
        # Generate sinusoidal values in range [-0.5, 0.5]
        actions[:, i] = 0.5 * np.sin(2 * np.pi * frequency * time_elapsed + phase_offset)

    return actions


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#


def main() -> None:
    # Parse env cfg from registry
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Apply eval mode if available
    if hasattr(env_cfg, "eval"):
        env_cfg.eval()

    # Cleanup env for playing (remove training-specific components)
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg = prepare_env_for_playing(env_cfg)

    # Create environment directly from cfg (no RL wrappers)
    render_mode = "rgb_array" if args_cli.video else None
    env = ManagerBasedRLEnv(env_cfg, render_mode=render_mode)

    # Call pre_learn hook if the task provides one (e.g., to load fallen state dataset)
    _call_pre_learn_hook(env, args_cli.task)

    # Setup for fallen state validation mode (set standing_ratio=0)
    if args_cli.validate_fallen_states:
        _setup_fallen_state_validation(env)

    # Optional video recording (uses gym wrapper over our Env)
    if args_cli.video:
        video_dir = os.path.abspath(os.path.join("logs", "videos", "play"))
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Use step_dt (control timestep) instead of physics_dt for proper timing
    dt = env.unwrapped.step_dt
    timestep = 0

    # Get action space dimensions
    num_envs = env.unwrapped.num_envs
    action_dim = env.unwrapped.action_manager.total_action_dim

    print(f"[INFO] Environment loaded: {args_cli.task}")
    print(f"[INFO] Number of environments: {num_envs}")
    print(f"[INFO] Action dimension: {action_dim}")
    print(f"[INFO] Control timestep: {dt:.4f}s ({1.0 / dt:.1f} Hz)")

    if args_cli.validate_fallen_states:
        print("[INFO] Validating fallen states with zero actions (resets every second)...")
        reset_interval_steps = int(1.0 / dt)  # Reset every second
    else:
        print("[INFO] Generating sinusoidal actions for environment validation...")
        reset_interval_steps = None

    # Wrap entire simulation in inference_mode for performance
    with torch.inference_mode():
        # Initial reset
        obs, _ = env.reset()

        while simulation_app.is_running():
            start = time.time()

            if args_cli.validate_fallen_states:
                # Zero actions for fallen state validation
                actions = torch.zeros(num_envs, action_dim, device=env.unwrapped.device)
                # Periodic reset to cycle through different fallen states
                if reset_interval_steps and timestep > 0 and timestep % reset_interval_steps == 0:
                    obs, _ = env.reset()
                    print(f"[INFO] Reset at step {timestep} to show new fallen states")
            else:
                # Generate smooth sinusoidal actions
                actions_np = generate_sinusoidal_actions(timestep, num_envs, action_dim, dt)
                actions = torch.as_tensor(actions_np, dtype=torch.float32, device=env.unwrapped.device)

            # Step
            obs, _, _, _, _ = env.step(actions)

            # Increment timestep for sinusoidal trajectory
            timestep += 1

            if args_cli.video:
                if timestep == args_cli.video_length:
                    break

            # Sleep to approximate real-time, if requested
            if args_cli.real_time:
                sleep_time = dt - (time.time() - start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    env.close()


def _call_pre_learn_hook(env, task_name: str, agent_cfg=None) -> None:
    """Call pre_learn hook if the task provides one.

    This is needed for tasks that require setup before the first reset
    (e.g., loading fallen state datasets for stand-up tasks).
    """
    import importlib

    pre_learn_entry_point = gym.spec(task_name).kwargs.get("pre_learn_entry_point")
    if pre_learn_entry_point is None:
        return  # No pre_learn hook for this task

    if agent_cfg is None:
        # Construct agent config from task spec
        agent_cfg_entry_point = gym.spec(task_name).kwargs.get("rsl_rl_cfg_entry_point")
        if agent_cfg_entry_point is None:
            print(f"[WARN] Task {task_name} has pre_learn but no rsl_rl_cfg_entry_point, skipping")
            return
        mod_name, class_name = agent_cfg_entry_point.split(":")
        mod = importlib.import_module(mod_name)
        agent_cfg = getattr(mod, class_name)()

    # Call pre_learn
    mod_name, fn_name = pre_learn_entry_point.split(":")
    mod = importlib.import_module(mod_name)
    pre_learn_fn = getattr(mod, fn_name)
    pre_learn_fn(env, task_name, agent_cfg)


def _setup_fallen_state_validation(env: ManagerBasedRLEnv) -> None:
    """Configure environment for fallen state validation.

    Sets standing_ratio=0 so all resets use fallen states from the dataset.
    """
    # Check if reset_base term exists
    term_exists = any("reset_base" in terms for terms in env.event_manager.active_terms.values())
    if not term_exists:
        print("[WARN] Fallen-state validation: 'reset_base' term not found; skipping standing_ratio override")
        return

    reset_term_cfg = env.event_manager.get_term_cfg("reset_base")
    reset_term_cfg.params["standing_ratio"] = 0.0
    print("[INFO] Fallen state validation mode: standing_ratio set to 0")


if __name__ == "__main__":
    main()
    simulation_app.close()
