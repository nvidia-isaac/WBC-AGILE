# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Script for dataset recording and GR00T policy inference.

- Dataset Recording: Record observations and actions to HDF5 files from RL policy rollouts.
- GR00T Inference: Connect to a GR00T inference server and execute policies with action chunking.
"""

# flake8: noqa

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Record dataset from an RL or GR00T policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during execution.")
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
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
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
# Add a new argument for number of steps to run
parser.add_argument(
    "--num_steps",
    type=int,
    default=10000,
    help="Number of steps to run the agent.",
)
# Add curriculum ratio for upper body action
parser.add_argument(
    "--curriculum_ratio",
    "-c",
    type=float,
    default=0.5,
    help="Curriculum ratio for upper body action.",
)
# Add arguments for HDF5 data recording
parser.add_argument(
    "--record",
    action="store_true",
    default=False,
    help="Record observations and actions to HDF5 file for dataset generation.",
)
parser.add_argument(
    "--record_output",
    type=str,
    default="/tmp/eval_recording",
    help="Output directory for recorded HDF5 data.",
)
parser.add_argument(
    "--record_episode_length",
    type=int,
    default=10,
    help="Minimum episode length to save (shorter episodes are discarded).",
)
parser.add_argument(
    "--record_obs_group",
    type=str,
    default="record",
    help="Name of the observation group to record (default: 'record').",
)
# Add arguments for GR00T policy inference
parser.add_argument(
    "--gr00t",
    action="store_true",
    default=False,
    help="Use GR00T policy for inference instead of RL checkpoint.",
)
parser.add_argument(
    "--gr00t_host",
    type=str,
    default="0.0.0.0",
    help="GR00T inference server host.",
)
parser.add_argument(
    "--gr00t_port",
    type=int,
    default=6666,
    help="GR00T inference server port.",
)
parser.add_argument(
    "--gr00t_action_horizon",
    type=int,
    default=8,
    help="Number of steps to execute from each GR00T action chunk.",
)
parser.add_argument(
    "--gr00t_task_description",
    type=str,
    default="Pick up apple",
    help="Task description for GR00T policy.",
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
import numpy as np
import os
import pickle
import time
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import agile.rl_env.tasks  # noqa: F401
import agile.isaaclab_extras.monkey_patches
from rsl_rl.runners import OnPolicyRunner
from agile.rl_env.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from agile.rl_env.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from data_recording.data_recorder import MultiEnvDataRecorder

# Image dimensions for GR00T inference (from TiledCameraCfg)
GR00T_IMAGE_HEIGHT = 480
GR00T_IMAGE_WIDTH = 640


def gr00t_policy_process(obs_dict, gr00t_client, task_description: str = "Pick up apple"):
    """Process observations from the record group for GR00T policy inference.

    Args:
        obs_dict: Observation dictionary containing 'record' group with:
            - image: RGB camera image (B, H, W, C)
            - base_lin_vel: Base linear velocity (B, 3)
            - base_ang_vel: Base angular velocity (B, 3)
            - joint_pos: Upper body joint positions (B, num_joints)
        gr00t_client: GR00T inference client.
        task_description: Task description for the policy.

    Returns:
        Action tensor of shape (B, T, action_dim) for the environment.
    """
    record_obs = obs_dict["record"]

    # Handle TensorDict
    if hasattr(record_obs, "to_dict"):
        record_obs = dict(record_obs.items())

    b = record_obs["image"].shape[0]
    device = record_obs["image"].device

    # Reshape image: (B, H, W, C) -> (B, 1, H, W, C) for video format
    # Convert from float32 [0, 1] to uint8 [0, 255] as expected by GR00T
    rgb = record_obs["image"].reshape(b, 1, GR00T_IMAGE_HEIGHT, GR00T_IMAGE_WIDTH, 3).cpu().numpy()
    rgb = (rgb * 255).astype(np.uint8)

    # Prepare observations for GR00T policy
    observations = {
        "annotation.human.action.task_description": np.repeat(np.array([[task_description]]), b, axis=0),
        "video.image": rgb,
        "state.base_ang_vel": record_obs["base_ang_vel"].reshape(b, 1, 3).cpu().numpy(),
        "state.joint_pos": record_obs["joint_pos"].reshape(b, 1, -1).cpu().numpy(),
        "state.joint_vel": record_obs["joint_vel"].reshape(b, 1, -1).cpu().numpy(),
    }

    # Get action from GR00T policy
    outputs = gr00t_client.get_action(observations)

    # Extract joint position actions: (B, T, action_dim)
    action_tensor = torch.from_numpy(outputs["action.action"]).to(device=device, dtype=torch.float32)
    return action_tensor


def load_policy(resume_path, env, agent_cfg):
    """Load policy from either TorchScript or regular checkpoint.

    This function intelligently detects the checkpoint format and loads accordingly:
    - TorchScript (.pt): Directly loads the exported policy (includes normalizer)
      * NOTE: Recurrent TorchScript policies are skipped because they're exported for
        single-env inference and don't work well with batched evaluation
    - Regular checkpoint (.pt): Loads through OnPolicyRunner (includes optimizer state, etc.)

    Args:
        resume_path: Path to the checkpoint file
        env: The wrapped environment (RslRlVecEnvWrapper)
        agent_cfg: Agent configuration (RslRlOnPolicyRunnerCfg)

    Returns:
        tuple: (policy, ppo_runner)
            - policy: Callable policy for inference
            - ppo_runner: OnPolicyRunner instance (None if TorchScript)
    """
    device = env.unwrapped.device

    # Try loading as TorchScript first (exported policies)
    try:
        policy = torch.jit.load(resume_path, map_location=device)
        policy.eval()

        # Check if it's a recurrent policy - if so, skip TorchScript and use regular checkpoint
        # Recurrent TorchScript policies are exported for single-env inference, which doesn't
        # work well with batched evaluation (would require per-env policy calls)
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
        # Not a valid TorchScript file, try regular checkpoint
        print(f"[INFO] Not a TorchScript file (error: {type(e).__name__}), loading as regular checkpoint...")

    # Load as regular checkpoint through OnPolicyRunner
    try:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        ppo_runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        ppo_runner.load(resume_path)

        # Obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=device)
        print("[INFO] Successfully loaded regular checkpoint")
        return policy, ppo_runner

    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint from {resume_path}. "
            f"Tried both TorchScript and regular checkpoint formats. Error: {e}"
        ) from e


def main():
    """Record dataset from RL or GR00T policy."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Skip checkpoint loading if using GR00T policy
    resume_path = None
    log_dir = None

    if not args_cli.gr00t:
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
        video_folder = os.path.join(log_dir, "videos", "play") if log_dir else "/tmp/eval_videos"
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Initialize GR00T client or load RL policy
    gr00t_client = None
    policy = None
    ppo_runner = None

    if args_cli.gr00t:
        # Use GR00T policy inference
        from data_recording.gr00t_service import ExternalRobotInferenceClient

        gr00t_client = ExternalRobotInferenceClient(host=args_cli.gr00t_host, port=args_cli.gr00t_port)
        print(f"[INFO] Using GR00T policy at {args_cli.gr00t_host}:{args_cli.gr00t_port}")
        print(f"[INFO] Action horizon: {args_cli.gr00t_action_horizon}")
        print(f"[INFO] Task description: {args_cli.gr00t_task_description}")
    else:
        # Load RL policy (supports both TorchScript and regular checkpoints)
        policy, ppo_runner = load_policy(resume_path, env, agent_cfg)

    # Export policy to onnx/jit if we loaded from a regular checkpoint
    # (Skip if already TorchScript or if export fails)
    if ppo_runner is not None:
        try:
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
            print("[INFO] Successfully exported policy to JIT and ONNX")
        except Exception as e:
            print(f"[WARNING] Failed to export policy (continuing anyway): {e}")
    else:
        print("[INFO] Skipping export (policy already in TorchScript format)")

    # Get the control timestep (not physics timestep - accounts for decimation)
    dt = env.unwrapped.step_dt

    # Initialize data recorder if requested
    recorder = None
    if args_cli.record:
        from pathlib import Path

        output_path = Path(args_cli.record_output) / "data.h5"
        recorder = MultiEnvDataRecorder(
            output_path=output_path,
            obs_group_name=args_cli.record_obs_group,
            compress=True,
            compress_images=False,
            metadata={"task": args_cli.task, "num_envs": args_cli.num_envs},
            min_episode_length=args_cli.record_episode_length,
        )
        print(f"[INFO] Recording to: {recorder.output_path}")
        print(f"[INFO] Min episode length: {args_cli.record_episode_length} steps")
        recorder.open()

    env.reset()
    print("[INFO] Running recording...")
    obs, _ = env.get_observations()
    timestep = 0
    num_steps = 0

    # Check if we need to convert TensorDict to tensor for exported policies. This is necessary when we are loading
    # a TorchScript policy instead of a regular checkpoint.
    # Note: We check if it's a dict-like object, not just if it has "values" attribute
    # (regular tensors have .values() method for sparse tensors, which would cause false positives)
    is_tensordict_obs = isinstance(obs, dict) or (
        hasattr(obs, "values") and callable(getattr(obs, "values", None)) and not isinstance(obs, torch.Tensor)
    )

    # GR00T action chunking state
    gr00t_action_chunk = None
    gr00t_chunk_step = 0

    # simulate environment
    while simulation_app.is_running() and num_steps < args_cli.num_steps:
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            if gr00t_client is not None:
                # GR00T policy inference with action chunking
                # Get new action chunk if needed
                if gr00t_action_chunk is None or gr00t_chunk_step >= args_cli.gr00t_action_horizon:
                    # Get full observations including the record group
                    _, obs_extras = env.get_observations()
                    if "observations" in obs_extras:
                        gr00t_action_chunk = gr00t_policy_process(
                            obs_extras["observations"],
                            gr00t_client,
                            task_description=args_cli.gr00t_task_description,
                        )
                        gr00t_chunk_step = 0

                # Execute current step from action chunk
                if gr00t_action_chunk is not None:
                    actions = gr00t_action_chunk[:, gr00t_chunk_step, :]
                    upper_body_term = env.unwrapped.action_manager.get_term("upper_body_joint_pos")
                    if upper_body_term is None:
                        raise RuntimeError(
                            "Action term 'upper_body_joint_pos' not found. "
                            "Ensure the environment config includes this action term."
                        )
                    actions[:, :-4] = (
                        actions[:, :-4] - upper_body_term.prev_targets
                    ) / upper_body_term._scale
                    gr00t_chunk_step += 1
                else:
                    # Fallback to zero actions if no chunk available
                    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            elif policy is not None:
                # RL policy inference
                # Convert TensorDict to tensor if needed (for exported TorchScript policies)
                if is_tensordict_obs and ppo_runner is None:
                    # Flatten TensorDict to tensor for exported policy
                    obs_tensor = torch.cat([v.flatten(start_dim=1) for v in obs.values()], dim=-1)
                else:
                    obs_tensor = obs

                # agent stepping
                actions = policy(obs_tensor)
            else:
                # No policy loaded - use zero actions
                actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

            # Record observations BEFORE env.step (captures obs that led to action)
            record_obs = None
            action_manager = env.unwrapped.action_manager
            if recorder is not None:
                _, record_extras = env.get_observations()
                if "observations" in record_extras:
                    record_obs = record_extras["observations"]

            # env stepping
            obs, _, _, extras = env.step(actions)

            # Record with upper_body_joint_pos overridden to processed_actions (absolute positions)
            if recorder is not None and record_obs is not None:
                # Get the processed actions for upper_body (absolute joint positions)
                upper_body_term = action_manager.get_term("upper_body_joint_pos")
                upper_body_dim = upper_body_term.action_dim
                # Clone actions and override upper_body portion with processed actions
                actions_to_record = actions.clone()
                actions_to_record[:, :upper_body_dim] = upper_body_term.processed_actions

                recorder.record_from_env(record_obs, actions_to_record)

        # Handle episode boundaries for recording (only for timeout terminations)
        if recorder is not None:
            if "time_outs" in extras and extras["time_outs"].any():
                timeout_mask = extras["time_outs"]
                recorder.handle_resets(timeout_mask)
                print(f"[INFO] Episodes timed out for envs: {timeout_mask.nonzero().flatten().tolist()}")

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time mode
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        num_steps += 1

    # Close the data recorder if it was opened
    if recorder is not None:
        recorder.close()
        print(f"[INFO] Recording saved to: {recorder.output_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function with propery error handling for CI/CD.
    exit_code = 0
    try:
        main()
    except Exception as e:
        import traceback

        print(f"\n[ERROR] Recording failed with exception: {e}", flush=True)
        traceback.print_exc()
        exit_code = 1
    finally:
        # close sim app
        simulation_app.close()

    # Exit with appropriate code
    if exit_code != 0:
        import sys

        sys.exit(exit_code)
