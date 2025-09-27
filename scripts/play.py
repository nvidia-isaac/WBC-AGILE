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
"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

# flake8: noqa

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
parser.add_argument(
    "--use_mirroring",
    action="store_true",
    default=False,
    help="Mirror observations/actions for side-by-side visualization (supports G1 and T1).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import os
import pickle
import time
import torch

# from isaaclab_rl.rsl_rl import (
from agile.rl_env.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
import agile.isaaclab_extras.monkey_patches

import agile.rl_env.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from agile.rl_env.mdp.symmetry import lr_mirror_G1, lr_mirror_T1


def prepare_env_for_playing(env_cfg: ManagerBasedRLEnvCfg) -> ManagerBasedRLEnvCfg:
    # use flat terrain
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # remove curriculum
    env_cfg.curriculum = None

    # remove harness
    if hasattr(env_cfg.actions, "harness"):
        del env_cfg.actions.harness

    # remove random upper body motion
    if hasattr(env_cfg.actions, "random_pos"):
        del env_cfg.actions.random_pos

    return env_cfg


def load_policy(resume_path, env, agent_cfg):
    """Load policy from either TorchScript or regular checkpoint.

    NOTE: Recurrent TorchScript policies are skipped because they're exported for
    single-env inference and don't work well with batched evaluation.
    """

    device = env.unwrapped.device

    try:
        policy = torch.jit.load(resume_path, map_location=device)
        policy.eval()

        # Check if it's a recurrent policy - if so, skip TorchScript and use regular checkpoint
        # Recurrent TorchScript policies are exported for single-env inference
        if hasattr(policy, "is_recurrent") and policy.is_recurrent:
            print(
                f"[INFO] Detected recurrent TorchScript policy, falling back to regular checkpoint for batched evaluation"
            )
            # Fall through to regular checkpoint loading
        else:
            print(f"[INFO] Loaded TorchScript policy from: {resume_path}")
            print("[INFO] TorchScript policies are self-contained (include normalizer)")
            return policy, None
    except (RuntimeError, AttributeError, pickle.UnpicklingError) as e:
        print(f"[INFO] Not a TorchScript file (error: {type(e).__name__}), loading as regular checkpoint...")

    try:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        ppo_runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        ppo_runner.load(resume_path)
        policy = ppo_runner.get_inference_policy(device=device)
        print("[INFO] Successfully loaded regular checkpoint")
        return policy, ppo_runner
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load checkpoint from {resume_path}. "
            f"Tried both TorchScript and regular checkpoint formats. Error: {e}"
        ) from e


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # cleanup env for playing
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg = prepare_env_for_playing(env_cfg)

    # setup
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    mirror_fn = None
    if args_cli.use_mirroring:
        if not args_cli.task:
            print("[WARNING] --use_mirroring requires --task to infer the robot type. Disabling flag.")
        else:
            task_lower = args_cli.task.lower()
            if "t1" in task_lower:
                mirror_fn = lr_mirror_T1
            elif "g1" in task_lower:
                mirror_fn = lr_mirror_G1
            else:
                print("[WARNING] --use_mirroring currently supports only G1 or T1 tasks. Disabling flag.")

        if mirror_fn and not hasattr(env.unwrapped, "action_manager"):
            print("[WARNING] Environment does not expose an action_manager. Disabling --use_mirroring.")
            mirror_fn = None

    use_mirroring = mirror_fn is not None

    policy, ppo_runner = load_policy(resume_path, env, agent_cfg)

    if ppo_runner is not None:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.policy,
            ppo_runner.obs_normalizer,
            path=export_model_dir,
            filename="policy.pt",
        )
        export_policy_as_onnx(
            ppo_runner.alg.policy,
            normalizer=ppo_runner.obs_normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )
    else:
        print("[INFO] Skipping export (policy already in TorchScript format)")

    dt = env.unwrapped.physics_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    # Check if we need to convert TensorDict to tensor for exported policies
    # Note: We check if it's a dict-like object, not just if it has "values" attribute
    # (regular tensors have .values() method for sparse tensors, which would cause false positives)
    is_tensordict_obs = isinstance(obs, dict) or (
        hasattr(obs, "values") and callable(getattr(obs, "values", None)) and not isinstance(obs, torch.Tensor)
    )

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # Convert TensorDict to tensor if needed (for exported TorchScript policies)
            if is_tensordict_obs and ppo_runner is None:
                # Flatten TensorDict to tensor for exported policy
                obs_tensor = torch.cat([v.flatten(start_dim=1) for v in obs.values()], dim=-1)
            else:
                obs_tensor = obs

            # agent stepping
            if use_mirroring:
                augmented_obs, _ = mirror_fn(env, obs_tensor, None, "policy")
                mirrored_actions = policy(augmented_obs[env.num_envs :])
                _, augmented_actions = mirror_fn(env, None, mirrored_actions, "policy")
                actions = augmented_actions[env.num_envs :]
            else:
                actions = policy(obs_tensor)
            # env stepping
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
