#!/usr/bin/env python3

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

"""Minimal script to export a trained policy to JIT format."""

import argparse
import os

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# Parse arguments
parser = argparse.ArgumentParser(description="Export trained policy to JIT.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: checkpoint_dir/exported).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Force headless mode
args_cli.headless = True

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg

import agile.isaaclab_extras.monkey_patches  # noqa: F401
import agile.rl_env.tasks  # noqa: F401
from agile.rl_env.rsl_rl import RslRlVecEnvWrapper, make_rsl_rl_runner

# Create minimal environment
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=False)
# Set to evaluation mode if available
if hasattr(env_cfg, "eval"):
    env_cfg.eval()

env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
if isinstance(env.unwrapped, DirectMARLEnv):
    env = multi_agent_to_single_agent(env)
env = RslRlVecEnvWrapper(env)

# Load checkpoint
agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
ppo_runner = make_rsl_rl_runner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
ppo_runner.load(args_cli.checkpoint)

# Export to JIT
output_dir = args_cli.output_dir or os.path.join(os.path.dirname(args_cli.checkpoint), "exported")
os.makedirs(output_dir, exist_ok=True)
ppo_runner.export_policy_to_jit(path=output_dir, filename="policy.pt")
print(f"Exported to: {os.path.join(output_dir, 'policy.pt')}")

env.close()
simulation_app.close()
