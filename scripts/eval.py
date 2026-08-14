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

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

# flake8: noqa

import argparse
import sys
from pathlib import Path

# Prefer this checkout over an editable AGILE installation from another workspace.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent trained with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_length_s",
    type=float,
    default=None,
    help="Recorded video length in seconds; overrides --video_length using the env control rate.",
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
    "--run_evaluation",
    action="store_true",
    help="Run evaluation.",
)
parser.add_argument(
    "--fail_on_non_timeout_dones",
    action="store_true",
    default=False,
    help="Fail if the rollout produces any non-timeout termination. Use this for videos that must be continuous.",
)
parser.add_argument(
    "--non_timeout_done_warmup_steps",
    type=int,
    default=0,
    help="Ignore non-timeout terminations before this rollout step when --fail_on_non_timeout_dones is set.",
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
# Add argument for direct metrics file output
parser.add_argument(
    "--metrics_file",
    type=str,
    default=None,
    help="Path to save metrics JSON file directly.",
)
# Add arguments for trajectory logging
parser.add_argument(
    "--save_trajectories",
    action="store_true",
    default=False,
    help="Save episode trajectory data to parquet files for offline analysis.",
)
parser.add_argument(
    "--trajectory_fields",
    type=str,
    nargs="+",
    default=None,
    help="Specific fields to save in trajectories (e.g., joint_pos joint_vel root_pos). Default: save all fields.",
)
# Add argument for evaluation scenario config
parser.add_argument(
    "--eval_config",
    type=str,
    default=None,
    help="Path to YAML file with deterministic evaluation scenario configuration.",
)
# Add argument for automatic report generation
parser.add_argument(
    "--generate_report",
    action="store_true",
    default=False,
    help="Automatically generate HTML report after evaluation (requires --save_trajectories).",
)
# Random command scheduling
parser.add_argument(
    "--random_commands",
    type=str,
    nargs="+",
    default=None,
    help=(
        "Enable random command scheduling. Specify which fields to randomize: "
        "lin_vel_x, lin_vel_y, ang_vel_z, base_height, or 'all'. "
        "Example: --random_commands lin_vel_x ang_vel_z"
    ),
)
parser.add_argument(
    "--random_interval",
    type=float,
    default=2.0,
    help="Seconds between random command resamples (default: 2.0).",
)
parser.add_argument(
    "--random_seed",
    type=int,
    default=None,
    help="RNG seed for random command scheduling (default: non-deterministic).",
)
# Observation noise injection
parser.add_argument(
    "--noise_scale",
    type=float,
    default=None,
    help="Gaussian noise standard deviation to add to observations before policy inference. "
    "Useful for stress-testing policy robustness.",
)
parser.add_argument(
    "--noise_seed",
    type=int,
    default=None,
    help="RNG seed for observation noise injection (default: non-deterministic).",
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

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import agile.rl_env.tasks  # noqa: F401
import agile.isaaclab_extras.monkey_patches
from agile.isaaclab_extras.record_video import EfficientRecordVideo
from agile.isaaclab_extras.video_camera import configure_robot_tracking_camera
from agile.algorithms.evaluation.evaluator import PolicyEvaluator
from agile.rl_env.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    make_rsl_rl_inference_load_cfg,
    make_rsl_rl_runner,
)
from agile.rl_env.rsl_rl.export_pruning import prepare_training_only_actions_for_evaluation
from agile.rl_env.rsl_rl_observations import policy_observation


def _apply_env_overrides(env_cfg, eval_config):
    """Apply environment overrides from eval config.

    Args:
        env_cfg: Environment configuration object
        eval_config: EvalConfig with overrides (can be None)
    """
    if eval_config is None:
        return

    overrides = eval_config.env_overrides

    # Override episode length (from overrides or top-level for backward compatibility)
    episode_length = None
    if overrides and overrides.episode_length_s is not None:
        episode_length = overrides.episode_length_s
    elif eval_config.episode_length_s is not None:
        episode_length = eval_config.episode_length_s

    if episode_length is not None and hasattr(env_cfg, "episode_length_s"):
        original_length = env_cfg.episode_length_s
        env_cfg.episode_length_s = episode_length
        print(f"[INFO] Overriding episode length: {original_length:.1f}s -> {episode_length:.1f}s")

    # Override num_envs (from overrides or top-level for backward compatibility)
    num_envs = None
    if overrides and overrides.num_envs is not None:
        num_envs = overrides.num_envs
    elif eval_config.num_envs is not None:
        num_envs = eval_config.num_envs

    if num_envs is not None and hasattr(env_cfg.scene, "num_envs"):
        original_num_envs = env_cfg.scene.num_envs
        env_cfg.scene.num_envs = num_envs
        print(f"[INFO] Overriding num_envs: {original_num_envs} -> {num_envs}")

    # Handle event overrides
    if overrides and overrides.events and hasattr(env_cfg, "events") and env_cfg.events is not None:
        if overrides.events.disable_all:
            env_cfg.events = None
            print("[INFO] Disabled all environment events")

        elif overrides.events.disable_interval_events:
            # Remove all interval-mode events
            events_to_remove = []
            for event_name in dir(env_cfg.events):
                if not event_name.startswith("_"):
                    event = getattr(env_cfg.events, event_name, None)
                    if event and hasattr(event, "mode") and event.mode == "interval":
                        events_to_remove.append(event_name)

            for event_name in events_to_remove:
                delattr(env_cfg.events, event_name)

            if events_to_remove:
                print(f"[INFO] Disabled interval events: {events_to_remove}")

        elif overrides.events.disable_specific:
            # Remove specifically named events
            disabled_events = []
            for event_name in overrides.events.disable_specific:
                if hasattr(env_cfg.events, event_name):
                    delattr(env_cfg.events, event_name)
                    disabled_events.append(event_name)
                else:
                    print(f"[WARNING] Event '{event_name}' not found in env config")

            if disabled_events:
                print(f"[INFO] Disabled events: {disabled_events}")


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
        ppo_runner = make_rsl_rl_runner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path, load_cfg=make_rsl_rl_inference_load_cfg(agent_cfg))

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
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Set the environment to evaluation mode
    if hasattr(env_cfg, "eval"):
        env_cfg.eval()

    removed_actions, default_position_actions = prepare_training_only_actions_for_evaluation(env_cfg)
    for removed_action in removed_actions:
        print(f"[INFO] Removed training-only action for evaluation: {removed_action}")
    for action_name in default_position_actions:
        print(f"[INFO] Holding evaluation action at the default joint positions: {action_name}")

    # Isaac Lab 3.0.0b2's plane USD has neither the legacy collision nor shader prims its
    # terrain importer expects. The compatibility patch retains the plane's built-in setup.
    terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    if terrain_cfg is not None and terrain_cfg.terrain_type == "plane":
        terrain_cfg.physics_material = None
        terrain_cfg.visual_material = None

    # Load evaluation scenario config early to override episode length before env creation
    eval_config = None
    if args_cli.eval_config:
        from agile.algorithms.evaluation.eval_config import EvalConfig

        print(f"[INFO] Loading evaluation scenario from: {args_cli.eval_config}")
        eval_config = EvalConfig.from_yaml(args_cli.eval_config)

        # Apply environment overrides from eval config BEFORE environment is created
        # This includes episode length, num_envs, event disabling, etc.
        _apply_env_overrides(env_cfg, eval_config)

    if args_cli.video:
        configure_robot_tracking_camera(env_cfg.viewer)
        print("[INFO] Recording video with the camera tracking robot in environment 0.")

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        try:
            from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
        except ImportError:
            print("[ERROR] Pretrained checkpoint feature not available in this Isaac Lab version.")
            return

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

    # Call pre_learn hook if the task provides one (e.g., to load fallen state dataset)
    _call_pre_learn_hook(env.unwrapped, args_cli.task, agent_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        # A seconds-based length is robust across tasks with different control rates. Convert it to
        # steps using the env control dt (sim.dt * decimation), which is also the recorded frame rate.
        if args_cli.video_length_s is not None:
            control_dt = env_cfg.sim.dt * env_cfg.decimation
            args_cli.video_length = max(1, round(args_cli.video_length_s / control_dt))
            print(
                f"[INFO] video_length_s={args_cli.video_length_s}s -> {args_cli.video_length} steps "
                f"(control_dt={control_dt:.4f}s)"
            )
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = EfficientRecordVideo(env, app_launcher=app_launcher, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Load policy (supports both TorchScript and regular checkpoints)
    policy, ppo_runner = load_policy(resume_path, env, agent_cfg)

    # Export policy to onnx/jit if we loaded from a regular checkpoint
    # (Skip if already TorchScript or if export fails)
    if ppo_runner is not None:
        try:
            export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
            ppo_runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            ppo_runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            print("[INFO] Successfully exported policy to JIT and ONNX")
        except Exception as e:
            print(f"[WARNING] Failed to export policy (continuing evaluation anyway): {e}")
            # This is not critical for evaluation, so we continue
    else:
        print("[INFO] Skipping export (policy already in TorchScript format)")

    # Get the control timestep (not physics timestep - accounts for decimation)
    dt = env.unwrapped.step_dt

    # Create scheduler: random commands take priority over deterministic eval_config
    scheduler = None
    if args_cli.random_commands is not None:
        from agile.algorithms.evaluation.random_command_scheduler import RandomCommandScheduler

        scheduler = RandomCommandScheduler(
            env,
            randomize_fields=args_cli.random_commands,
            interval=args_cli.random_interval,
            seed=args_cli.random_seed,
            verbose=True,
        )
    elif eval_config is not None:
        from agile.algorithms.evaluation.velocity_height_scheduler import VelocityHeightScheduler

        # Validate num_envs matches
        if eval_config.num_envs != args_cli.num_envs:
            print(f"[INFO] Config specifies {eval_config.num_envs} envs but {args_cli.num_envs} was used.")
            if env.num_envs != eval_config.num_envs:
                print(
                    f"[WARNING] Config specifies {eval_config.num_envs} envs but "
                    f"{env.num_envs} were created. Using {env.num_envs}."
                )
                eval_config.num_envs = env.num_envs

        # Create scheduler
        scheduler = VelocityHeightScheduler(env, eval_config, verbose=True)

    # Metrics path setup - use direct file if specified
    metrics_path = None
    if args_cli.metrics_file:
        # Extract both directory and filename from the metrics_file path
        metrics_path = os.path.dirname(args_cli.metrics_file)
        os.makedirs(metrics_path, exist_ok=True)

    # Motion metrics require an "eval" observation group; other tasks still record video.
    if args_cli.run_evaluation and "eval" not in env.unwrapped.observation_manager.active_terms:
        print(
            "[WARNING] --run_evaluation requested but this task has no 'eval' observation group; "
            "skipping motion-metrics evaluation and recording a rollout video only."
        )
        args_cli.run_evaluation = False

    if args_cli.run_evaluation:
        print("[INFO] Running default motion metrics evaluator.")
        if args_cli.save_trajectories and args_cli.trajectory_fields:
            print(f"[INFO] Saving fields: {args_cli.trajectory_fields}")
        else:
            print("[INFO] Saving all trajectory fields.")

        # Extract joint group config if available
        joint_group_config = None
        if eval_config is not None and eval_config.joint_groups:
            joint_group_config = eval_config.joint_groups
            print(f"[INFO] Using joint groups from config: {list(joint_group_config.keys())}")
        else:
            print("[INFO] No joint groups specified, using 'default' group with all joints")

        # Calculate total episodes to collect
        # If eval_config is provided, use num_envs * num_episodes, otherwise just num_envs
        if eval_config is not None:
            total_episodes = eval_config.num_envs * eval_config.num_episodes
            print(
                f"[INFO] Will collect {total_episodes} episodes ({eval_config.num_envs} envs x {eval_config.num_episodes} episodes each)"
            )
        else:
            total_episodes = args_cli.num_envs
            print(f"[INFO] Will collect {total_episodes} episodes")

        # Build provenance metadata for reproducibility
        import sys

        provenance = {
            "checkpoint": str(resume_path),
            "task": args_cli.task,
            "eval_config": args_cli.eval_config,
            "num_envs": env.num_envs,
            "num_steps": args_cli.num_steps,
            "noise_scale": args_cli.noise_scale,
            "noise_seed": args_cli.noise_seed,
            "random_commands": args_cli.random_commands,
            "random_interval": args_cli.random_interval,
            "random_seed": args_cli.random_seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "command_line": " ".join(sys.argv),
        }

        evaluator = PolicyEvaluator(
            env,
            task_name=args_cli.task,
            metrics_path=metrics_path,
            total_envs_target=total_episodes,
            verbose=True,
            save_trajectories=args_cli.save_trajectories,
            trajectory_fields=args_cli.trajectory_fields,
            joint_group_config=joint_group_config,
            provenance=provenance,
        )

    env.reset()
    # Reset scheduler after burn-in if using scenarios
    if scheduler:
        scheduler.reset()

    print("[INFO] Running evaluation...")
    obs = env.get_observations()
    timestep = 0
    num_steps = 0
    non_timeout_done_count = 0
    non_timeout_done_steps: list[int] = []

    # Check if we need to convert TensorDict to tensor for exported policies. This is necessary when we are loading
    # a TorchScript policy instead of a regular checkpoint.
    # Note: We check if it's a dict-like object, not just if it has "values" attribute
    # (regular tensors have .values() method for sparse tensors, which would cause false positives)
    is_tensordict_obs = isinstance(obs, dict) or (
        hasattr(obs, "values") and callable(getattr(obs, "values", None)) and not isinstance(obs, torch.Tensor)
    )

    # Set up observation noise generator if requested
    noise_generator = None
    if args_cli.noise_scale is not None and args_cli.noise_scale > 0:
        noise_generator = torch.Generator(device=env.unwrapped.device)
        if args_cli.noise_seed is not None:
            noise_generator.manual_seed(args_cli.noise_seed)
        print(f"[INFO] Observation noise enabled: scale={args_cli.noise_scale}, seed={args_cli.noise_seed}")

    # simulate environment
    while simulation_app.is_running() and num_steps < args_cli.num_steps:
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # Check if we need to update scheduled commands based on time
            if scheduler:
                scheduler.update(dt)
            # Convert TensorDict to tensor if needed (for exported TorchScript policies)
            if is_tensordict_obs and ppo_runner is None:
                obs_tensor = policy_observation(obs)
            else:
                obs_tensor = obs

            # Inject observation noise for robustness stress-testing
            if noise_generator is not None:
                if isinstance(obs_tensor, torch.Tensor):
                    obs_tensor = obs_tensor + args_cli.noise_scale * torch.randn(
                        obs_tensor.shape, device=obs_tensor.device, generator=noise_generator
                    )
                else:
                    # TensorDict / dict: apply noise to each value independently
                    for key in obs_tensor:
                        val = obs_tensor[key]
                        obs_tensor[key] = val + args_cli.noise_scale * torch.randn(
                            val.shape, device=val.device, generator=noise_generator
                        )

            # agent stepping
            actions = policy(obs_tensor)
            # env stepping
            obs, _, dones, extras = env.step(actions)

            if args_cli.fail_on_non_timeout_dones:
                done_mask = dones.to(dtype=torch.bool)
                time_outs = extras.get("time_outs")
                if time_outs is None:
                    timeout_mask = torch.zeros_like(done_mask, dtype=torch.bool)
                else:
                    timeout_mask = time_outs.to(device=done_mask.device, dtype=torch.bool)
                non_timeout_done_mask = done_mask & ~timeout_mask
                non_timeout_count = int(non_timeout_done_mask.sum().item())
                if non_timeout_count and num_steps >= args_cli.non_timeout_done_warmup_steps:
                    non_timeout_done_count += non_timeout_count
                    if len(non_timeout_done_steps) < 10:
                        non_timeout_done_steps.append(num_steps)

            # Reapply scheduled commands after env.step()
            # This is necessary because command_manager.compute() inside env.step()
            # resamples commands, which would overwrite our scheduled values.
            # The scheduler also recomputes observations to reflect the corrected commands.
            if scheduler:
                scheduler.reapply_commands()
                # Get the recomputed observations and extras
                obs, extras = env.get_observations_with_extras()

                # CRITICAL FIX: _update_command() inside observation_manager.compute() may have
                # modified our scheduled commands before they were captured in observations.
                # Directly inject the correct scheduled commands into the observations dict.
                if "observations" in extras:
                    obs_dict = extras["observations"]
                    if "eval" in obs_dict:
                        eval_obs = obs_dict["eval"]
                        obs_manager = env.unwrapped.observation_manager

                        # eval_obs can be either a dict or a concatenated tensor
                        if isinstance(eval_obs, dict) and "commands" in eval_obs:
                            # Dict format: eval_obs["commands"] is a tensor
                            commands_tensor = eval_obs["commands"]
                            for env_id, command_tensor in scheduler.active_commands.items():
                                if command_tensor is not None:
                                    if commands_tensor.dim() == 3:
                                        commands_tensor[env_id, 0, :] = command_tensor
                                    elif commands_tensor.dim() == 2:
                                        commands_tensor[env_id, :] = command_tensor
                        elif isinstance(eval_obs, torch.Tensor) and "eval" in obs_manager.active_terms:
                            # Concatenated tensor format: need to find the slice for "commands"
                            term_names = obs_manager.active_terms["eval"]
                            term_dims = obs_manager.group_obs_term_dim["eval"]

                            if "commands" in term_names:
                                # Find the start index for commands in the concatenated tensor
                                cmd_idx = term_names.index("commands")
                                start_idx = sum((d[0] if isinstance(d, tuple) else d) for d in term_dims[:cmd_idx])
                                cmd_dim = (
                                    term_dims[cmd_idx][0]
                                    if isinstance(term_dims[cmd_idx], tuple)
                                    else term_dims[cmd_idx]
                                )

                                # Inject scheduled commands at the correct slice
                                for env_id, command_tensor in scheduler.active_commands.items():
                                    if command_tensor is not None:
                                        eval_obs[env_id, start_idx : start_idx + cmd_dim] = command_tensor

        if args_cli.video:
            timestep += 1
            # A video limits recording only. Metric evaluation must continue until all scheduled
            # episodes finish, otherwise a short report video would silently produce partial metrics.
            if timestep == args_cli.video_length and not args_cli.run_evaluation:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        num_steps += 1

        if args_cli.run_evaluation:
            # The RSL-RL wrapper's step() does not put raw observations into ``extras``, but the
            # evaluator needs ``extras["observations"]`` (the "eval" group). The scheduler branch
            # above refreshes ``extras`` via ``get_observations_with_extras()``; when no scheduler
            # is active, populate it here so evaluation works regardless of the eval config.
            if "observations" not in extras:
                extras["observations"] = env.get_observations_with_extras()[1]["observations"]
            # Update the evaluator with extras that contain the observations (and corrected commands).
            done = evaluator.collect(dones, extras)
            if done:
                break

    # Finalize evaluation if it was running
    # This is called whether the loop completed normally or was interrupted
    if args_cli.run_evaluation:
        # Check if evaluation completed
        if evaluator._num_envs_evaluated < evaluator._total_envs_target:
            print(
                f"[INFO] Evaluation incomplete: {evaluator._num_envs_evaluated}/{evaluator._total_envs_target} episodes completed"
            )
        # Always conclude to save metrics and trajectories
        evaluator.conclude()

        # Compute and save aggregated tracking metrics if trajectories were saved
        if args_cli.save_trajectories and args_cli.generate_report:
            # Generate HTML report if requested
            print("\n[INFO] Generating HTML report...")
            try:
                from agile.algorithms.evaluation.report_generator import TrajectoryReportGenerator

                # Use the metrics_path from evaluator (where trajectories are saved)
                if evaluator._metrics_path:
                    generator = TrajectoryReportGenerator(evaluator._metrics_path, task_name=args_cli.task)
                    report_path = generator.generate_full_report(
                        episode_ids="all",
                        include_all_joints=True,
                        open_browser=False,  # Don't open browser in headless mode
                    )
                    print(f"[INFO] Report generated: {report_path}")
                else:
                    print("[WARNING] Cannot generate report: no metrics path available")
            except Exception as e:
                print(f"[ERROR] Failed to generate report: {e}")
                import traceback

                traceback.print_exc()

    # close the simulator
    env.close()
    if args_cli.fail_on_non_timeout_dones and non_timeout_done_count:
        raise RuntimeError(
            "Isaac Lab rollout had "
            f"{non_timeout_done_count} non-timeout termination(s); "
            f"first steps: {non_timeout_done_steps}"
        )


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


if __name__ == "__main__":
    from agile.evaluation.cli_exit import run_main_with_simulation_app

    run_main_with_simulation_app(main, simulation_app)
