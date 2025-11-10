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
parser = argparse.ArgumentParser(description="Play an RL agent trained with RSL-RL.")
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
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import agile.rl_env.tasks  # noqa: F401
import agile.isaaclab_extras.monkey_patches
from rsl_rl.runners import OnPolicyRunner
from agile.algorithms.evaluation.evaluator import PolicyEvaluator
from agile.rl_env.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from agile.rl_env.rsl_rl import export_policy_as_jit, export_policy_as_onnx


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

    # Load evaluation scenario config early to override episode length before env creation
    eval_config = None
    if args_cli.eval_config:
        from agile.algorithms.evaluation.eval_config import EvalConfig

        print(f"[INFO] Loading evaluation scenario from: {args_cli.eval_config}")
        eval_config = EvalConfig.from_yaml(args_cli.eval_config)

        # Apply environment overrides from eval config BEFORE environment is created
        # This includes episode length, num_envs, event disabling, etc.
        _apply_env_overrides(env_cfg, eval_config)

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

    # Load policy (supports both TorchScript and regular checkpoints)
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
            print(f"[WARNING] Failed to export policy (continuing evaluation anyway): {e}")
            # This is not critical for evaluation, so we continue
    else:
        print("[INFO] Skipping export (policy already in TorchScript format)")

    # Get the control timestep (not physics timestep - accounts for decimation)
    dt = env.unwrapped.step_dt

    # Create scheduler if eval config was loaded
    scheduler = None
    if eval_config is not None:
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

    if args_cli.run_evaluation:
        print("[INFO] Running default motion metrics evaluator.")
        if args_cli.save_trajectories and args_cli.trajectory_fields:
            print(f"[INFO] Saving fields: {args_cli.trajectory_fields}")
        else:
            print("[INFO] Saving all trajectory fields.")

        # Calculate total episodes to collect
        # If eval_config is provided, use num_envs * num_episodes, otherwise just num_envs
        if eval_config is not None:
            total_episodes = eval_config.num_envs * eval_config.num_episodes
            print(
                f"[INFO] Will collect {total_episodes} episodes ({eval_config.num_envs} envs × {eval_config.num_episodes} episodes each)"
            )
        else:
            total_episodes = args_cli.num_envs
            print(f"[INFO] Will collect {total_episodes} episodes")

        evaluator = PolicyEvaluator(
            env,
            task_name=args_cli.task,
            metrics_path=metrics_path,
            total_envs_target=total_episodes,
            verbose=True,
            save_trajectories=args_cli.save_trajectories,
            trajectory_fields=args_cli.trajectory_fields,
        )

    env.reset()
    # Reset scheduler after burn-in if using scenarios
    if scheduler:
        scheduler.reset()

    print("[INFO] Running evaluation...")
    obs, _ = env.get_observations()
    timestep = 0
    num_steps = 0
    # simulate environment
    while simulation_app.is_running() and num_steps < args_cli.num_steps:
        start_time = time.time()

        # Check if we need to update scheduled commands based on time
        if scheduler:
            scheduler.update(dt)

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, extras = env.step(actions)

            # Reapply scheduled commands after env.step()
            # This is necessary because command_manager.compute() inside env.step()
            # resamples commands, which would overwrite our scheduled values.
            # The scheduler also recomputes observations to reflect the corrected commands.
            if scheduler:
                scheduler.reapply_commands()
                # Get the recomputed observations
                obs, extras = env.get_observations()

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
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        num_steps += 1

        if args_cli.run_evaluation:
            # Update the evaluator with corrected extras that contain the right commands
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
                    generator = TrajectoryReportGenerator(evaluator._metrics_path)
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


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
