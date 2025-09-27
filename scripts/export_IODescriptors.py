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


"""Script to an environment with random action agent."""

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


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
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
