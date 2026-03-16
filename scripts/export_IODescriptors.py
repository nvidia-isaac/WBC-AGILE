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


"""Export IO descriptor YAML for sim2mujoco evaluation."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--output_dir", type=str, default=None, help="Path to the output directory.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import agile.isaaclab_extras.monkey_patches  # noqa: F401

# agile imports
import agile.rl_env.tasks  # noqa: F401


def convert_to_serializable(obj):
    """Recursively convert tensors and numpy arrays to Python types for YAML serialization."""
    if isinstance(obj, torch.Tensor):
        # Convert tensor to Python list or scalar
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.cpu().tolist()
    elif isinstance(obj, np.ndarray):
        # Convert numpy array to Python list or scalar
        if obj.size == 1:
            return obj.item()
        else:
            return obj.tolist()
    elif isinstance(obj, np.integer | np.floating):
        # Convert numpy scalars to Python scalars
        return obj.item()
    elif isinstance(obj, dict):
        # Recursively process dictionary
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively process list
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        # Recursively process tuple (convert to list for YAML)
        return [convert_to_serializable(item) for item in obj]
    else:
        # Return as-is for other types
        return obj


def export_motion_tracking_data(env):
    """Export motion tracking command configuration if present.

    Iterates over the environment's command manager terms and extracts the
    MotionCommand configuration needed by the sim2mujoco MotionTracker.

    Args:
        env: The unwrapped IsaacLab environment.

    Returns:
        A dictionary with motion tracking config, or None if no MotionCommand is found.
    """
    if not hasattr(env, "command_manager"):
        return None

    for term_name in env.command_manager.active_terms:
        term = env.command_manager.get_term(term_name)
        cfg = getattr(term, "cfg", None)
        # Check if this is a MotionCommand by looking for the motion_file attribute.
        if cfg is not None and hasattr(cfg, "motion_file"):
            # Try to make the motion file path relative to the current working
            # directory for portability.  Fall back to the absolute path.
            motion_file = cfg.motion_file
            try:
                rel_path = os.path.relpath(motion_file)
                if not rel_path.startswith(".."):
                    motion_file = rel_path
            except ValueError:
                pass  # e.g. different drives on Windows

            motion_data = {
                "motion_file": motion_file,
                "anchor_body_name": cfg.anchor_body_name,
            }
            if cfg.motion_body_names is not None:
                motion_data["motion_body_names"] = list(cfg.motion_body_names)
            if cfg.motion_joint_names is not None:
                motion_data["motion_joint_names"] = list(cfg.motion_joint_names)
            print(f"[INFO]: Found MotionCommand '{term_name}' with motion_file: {motion_file}")
            return motion_data

    return None


def _inject_noise_config(env, outs: dict) -> None:
    """Extract per-term noise config from the observation manager into the export dict.

    Isaac Lab's ``get_IO_descriptors`` deliberately omits noise.  This function
    reads the noise config from each ``ObsTerm`` and attaches it to the
    corresponding entry in ``outs["observations"]``.
    """
    from isaaclab.utils.noise import noise_cfg as noise_cfg_mod

    obs_mgr = getattr(env, "observation_manager", None)
    if obs_mgr is None:
        return

    for group_name in obs_mgr._group_obs_term_names:
        if group_name not in outs.get("observations", {}):
            continue
        term_names = obs_mgr._group_obs_term_names[group_name]
        term_cfgs = obs_mgr._group_obs_term_cfgs[group_name]
        exported_terms = outs["observations"][group_name]

        name_to_export = {t["name"]: t for t in exported_terms}
        for term_name, term_cfg in zip(term_names, term_cfgs, strict=False):
            ncfg = getattr(term_cfg, "noise", None)
            if ncfg is None or term_name not in name_to_export:
                continue

            noise_dict = None
            if isinstance(ncfg, noise_cfg_mod.UniformNoiseCfg):
                noise_dict = {
                    "type": "uniform",
                    "n_min": float(ncfg.n_min) if not isinstance(ncfg.n_min, torch.Tensor) else ncfg.n_min.tolist(),
                    "n_max": float(ncfg.n_max) if not isinstance(ncfg.n_max, torch.Tensor) else ncfg.n_max.tolist(),
                }
            elif isinstance(ncfg, noise_cfg_mod.GaussianNoiseCfg):
                noise_dict = {
                    "type": "gaussian",
                    "mean": float(ncfg.mean) if not isinstance(ncfg.mean, torch.Tensor) else ncfg.mean.tolist(),
                    "std": float(ncfg.std) if not isinstance(ncfg.std, torch.Tensor) else ncfg.std.tolist(),
                }

            if noise_dict is not None:
                name_to_export[term_name]["noise"] = noise_dict
                print(f"[INFO]: Exported noise for '{term_name}': {noise_dict}")


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)

    # Disable startup randomization events so the export captures deterministic
    # base values (e.g. clean default_joint_pos from init_state, not randomized).
    # Tasks like Velocity-G1-v0 have no startup randomization and export cleanly.
    # Tasks like Tracking-Flat-G1-v0 have randomize_joint_default_pos which
    # perturbs default_joint_pos before export, causing wrong offsets/scales.
    if hasattr(env_cfg, "events"):
        disabled = []
        for attr_name in list(vars(env_cfg.events).keys()):
            event = getattr(env_cfg.events, attr_name, None)
            if event is not None and hasattr(event, "mode") and event.mode == "startup":
                setattr(env_cfg.events, attr_name, None)
                disabled.append(attr_name)
        if disabled:
            print(f"[INFO]: Disabled startup events for deterministic export: {disabled}")

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    outs = env.unwrapped.get_IO_descriptors
    out_observations = outs["observations"]
    out_actions = outs["actions"]
    out_articulations = outs["articulations"]
    out_scene = outs["scene"]

    # Export motion tracking config if present.
    motion_tracking = export_motion_tracking_data(env.unwrapped)
    if motion_tracking is not None:
        outs["motion_tracking"] = motion_tracking

    # Inject per-term noise config from the observation manager (not exported by IsaacLab).
    _inject_noise_config(env.unwrapped, outs)

    # Supplement articulation data with soft joint velocity limits
    # (not exported by IsaacLab's export_articulations_data).
    for name, art in env.unwrapped.scene.articulations.items():
        vel_limits = art.data.soft_joint_vel_limits
        if vel_limits is not None and name in outs["articulations"]:
            outs["articulations"][name]["soft_joint_vel_limits"] = vel_limits[0].detach().cpu().tolist()

    # Convert all data to be YAML-serializable
    outs_serializable = convert_to_serializable(outs)

    # Make a yaml file with the output
    import yaml

    # Custom dumper to fix sequence indentation
    class ProperIndentDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)

    name = args_cli.task.lower().replace("-", "_")
    name = name.replace(" ", "_")

    if not os.path.exists(args_cli.output_dir):
        os.makedirs(args_cli.output_dir)

    with open(os.path.join(args_cli.output_dir, f"{name}_IO_descriptors.yaml"), "w") as f:
        print(f"[INFO]: Exporting IO descriptors to {os.path.join(args_cli.output_dir, f'{name}_IO_descriptors.yaml')}")
        yaml.dump(
            outs_serializable,
            f,
            Dumper=ProperIndentDumper,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
            width=float("inf"),  # Prevent line wrapping
            allow_unicode=True,
        )

    for k in out_actions:
        print(f"--- Action term: {k['name']} ---")
        k.pop("name")
        for k1, v1 in k.items():
            print(f"{k1}: {v1}")

    for obs_group_name, obs_group in out_observations.items():
        print(f"--- Obs group: {obs_group_name} ---")
        for k in obs_group:
            print(f"--- Obs term: {k['name']} ---")
            k.pop("name")
            for k1, v1 in k.items():
                print(f"{k1}: {v1}")

    for articulation_name, articulation_data in out_articulations.items():
        print(f"--- Articulation: {articulation_name} ---")
        for k1, v1 in articulation_data.items():
            print(f"{k1}: {v1}")

    for k1, v1 in out_scene.items():
        print(f"{k1}: {v1}")

    env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
