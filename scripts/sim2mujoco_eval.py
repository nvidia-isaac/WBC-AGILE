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

"""Main entry point for sim2mujoco evaluation.

Examples:
    # Without eval config (manual keyboard control):
    python scripts/sim2mujoco_eval.py \
        --checkpoint agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt \
        --config agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.yaml \
        --mjcf agile/rl_env/assets/robot_menagerie/unitree/g1/mujoco/scene_29dof.xml \
        --duration 100.0

    # Motion tracking task (uses training checkpoint directly, no JIT export needed):
    python scripts/sim2mujoco_eval.py \
        --checkpoint agile/data/policy/tracking_flat_g1/tracking_flat_g1.pt \
        --config agile/data/policy/tracking_flat_g1/tracking_flat_g1.yaml \
        --mjcf agile/rl_env/assets/robot_menagerie/unitree/g1/mujoco/scene_29dof.xml \
        --duration 100.0

    # With eval config (deterministic command schedule, duration from eval config):
    python scripts/sim2mujoco_eval.py \
        --checkpoint agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt \
        --config agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.yaml \
        --mjcf agile/rl_env/assets/robot_menagerie/unitree/g1/mujoco/scene_29dof.xml \
        --eval-config agile/sim2mujoco/configs/x_velocity_sweep.yaml \
        --save-data --no-viewer

    # Random commands (randomize only vx, for comparison with deterministic sweep):
    python scripts/sim2mujoco_eval.py \
        --checkpoint agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_teacher.pt \
        --config agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_teacher.yaml \
        --mjcf agile/rl_env/assets/robot_menagerie/unitree/g1/mujoco/scene_29dof.xml \
        --random-commands vx --random-interval 2.0 --random-seed 42 \
        --duration 50.0 --save-data --no-viewer

    # Random commands (randomize all dimensions):
    python scripts/sim2mujoco_eval.py \
        --checkpoint agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt \
        --config agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.yaml \
        --mjcf agile/rl_env/assets/robot_menagerie/unitree/g1/mujoco/scene_29dof.xml \
        --random-commands all --random-interval 2.0 --random-seed 0 \
        --duration 50.0 --save-data --no-viewer
"""

import argparse
import signal
import time
from datetime import datetime
from pathlib import Path

import torch

from agile.sim2mujoco.actions import ActionProcessor
from agile.sim2mujoco.command_provider import VelocityCommandProvider, create_command_provider
from agile.sim2mujoco.data_logger import Sim2MuJoCoDataLogger
from agile.sim2mujoco.observations import ObservationProcessor
from agile.sim2mujoco.policy import PolicyWrapper
from agile.sim2mujoco.simulation import MuJocoSimulation
from agile.sim2mujoco.utils import default_device, load_config


def main():
    """Run sim2sim evaluation."""
    parser = argparse.ArgumentParser(description="Sim2Sim Policy Evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to policy checkpoint (.pt or .onnx)")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--mjcf", type=Path, default=None, help="Path to MJCF file (optional, overrides config)")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration (seconds)")
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, cpu, or auto")
    parser.add_argument("--no-viewer", action="store_true", help="Disable MuJoCo viewer")
    parser.add_argument("--log-freq", type=int, default=100, help="Logging frequency (control steps)")
    parser.add_argument(
        "--pd-scale", type=float, default=1.0, help="Scale factor for PD gains (use 0.3-0.5 for stability)"
    )
    parser.add_argument(
        "--disable-keyboard", action="store_true", help="Disable keyboard control for interactive commands"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable step-by-step logging output")
    parser.add_argument(
        "--no-real-time", action="store_true", help="Disable real-time pacing (runs as fast as possible)"
    )
    parser.add_argument(
        "--eval-config", type=Path, default=None, help="Path to eval config YAML (deterministic command schedule)"
    )
    parser.add_argument("--save-data", action="store_true", help="Save evaluation data to disk")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for saved data")
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.0,
        help="Scale factor for observation noise (0=off, 1=match training, >1=stress test)",
    )
    parser.add_argument("--noise-seed", type=int, default=None, help="Random seed for reproducible observation noise")
    parser.add_argument(
        "--random-commands",
        type=str,
        nargs="+",
        default=None,
        metavar="FIELD",
        help="Randomize commands uniformly (resample every --random-interval seconds). "
        "Fields: vx, vy, wz, height, or 'all'. Non-listed fields stay at defaults. "
        "Mutually exclusive with --eval-config. "
        "Example: --random-commands vx  (only forward velocity randomized)",
    )
    parser.add_argument(
        "--random-interval",
        type=float,
        default=2.0,
        help="Seconds between random command resamples (default: 2.0)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="RNG seed for reproducible random commands (default: non-deterministic)",
    )

    args = parser.parse_args()

    if args.random_commands and args.eval_config:
        parser.error("--random-commands and --eval-config are mutually exclusive")

    # Setup device.
    if args.device == "auto":
        device = default_device()
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Setup noise seed for reproducibility.
    if args.noise_seed is not None:
        torch.manual_seed(args.noise_seed)
    if args.noise_scale > 0:
        seed_info = f", seed={args.noise_seed}" if args.noise_seed is not None else ""
        print(f"Observation noise: scale={args.noise_scale}{seed_info}")

    # Load config.
    print(f"\nLoading config from {args.config}...")
    config = load_config(args.config)

    # Override MJCF path if provided.
    if args.mjcf:
        config["mjcf_path"] = str(args.mjcf)

    # Scale PD gains if requested.
    if args.pd_scale != 1.0:
        print(f"Scaling PD gains by {args.pd_scale}...")
        robot_config = config["articulations"]["robot"]
        robot_config["default_joint_stiffness"] = [kp * args.pd_scale for kp in robot_config["default_joint_stiffness"]]
        robot_config["default_joint_damping"] = [kd * args.pd_scale for kd in robot_config["default_joint_damping"]]

    # Load eval config if provided (YAML-defined command schedule).
    eval_config = None
    if args.eval_config is not None:
        if not args.eval_config.exists():
            raise FileNotFoundError(f"Eval config not found: {args.eval_config}")
        from agile.algorithms.evaluation.eval_config import EvalConfig

        eval_config = EvalConfig.from_yaml(args.eval_config)
        if eval_config.num_envs != 1:
            raise ValueError(
                f"sim2mujoco only supports num_envs=1, got num_envs={eval_config.num_envs} "
                f"in eval config {args.eval_config}"
            )
        if eval_config.num_episodes != 1:
            raise ValueError(
                f"sim2mujoco only supports num_episodes=1, got num_episodes={eval_config.num_episodes} "
                f"in eval config {args.eval_config}"
            )
        if eval_config.get_env_config(0) is None:
            raise ValueError(
                "Eval config must have an environment with env_ids: [0] for sim2mujoco. "
                f"Found env_ids: {[e.env_ids for e in eval_config.environments]}"
            )
        args.duration = eval_config.episode_length_s
        print(f"\n✓ Loaded eval config from {args.eval_config} (duration={args.duration}s)")

    # Load policy.
    print(f"\nLoading policy from {args.checkpoint}...")
    policy = PolicyWrapper.from_config(args.checkpoint, config, device)
    print(f"  Policy type: {type(policy).__name__}")

    # Create simulation (command_manager will be attached after provider creation).
    print("\nCreating simulation...")
    sim = MuJocoSimulation(config, device, enable_viewer=not args.no_viewer, mjcf_path=args.mjcf)
    print(f"  Num joints: {sim.num_joints}")
    print(f"  Fixed base: {sim.fixed_base}")
    print(f"  Physics dt: {sim.physics_dt}s ({1.0 / sim.physics_dt:.0f} Hz)")
    print(f"  Control dt: {sim.dt}s ({1.0 / sim.dt:.1f} Hz)")
    print(f"  Decimation: {sim.decimation}")

    # Create observation processor first — it builds the MotionTracker if needed.
    print("\nSetting up observation processor...")
    obs_processor = ObservationProcessor(config, sim.joint_names, device)
    print(f"  Total observation dim: {obs_processor.total_obs_dim}")
    print("  Observation terms:")
    for term in obs_processor.terms:
        hist_info = f" (history={term.history_length})" if term.history_length > 0 else ""
        noise_info = f" (noise={term.noise_type})" if term.noise_type else ""
        print(f"    - {term.name}: {term.output_dim()}{hist_info}{noise_info}")

    # Create the unified command provider (factory decides velocity vs motion tracking).
    command_provider = create_command_provider(config, device, motion_tracker=obs_processor.motion_tracker)

    # Wire up CommandManager-based features (keyboard, scheduler) for velocity providers.
    command_manager = None
    command_scheduler = None
    if isinstance(command_provider, VelocityCommandProvider):
        command_manager = command_provider.manager
        sim.command_manager = command_manager
        obs_processor.command_manager = command_manager

        if eval_config is not None:
            from agile.sim2mujoco.command_scheduler import Sim2MuJoCoCommandScheduler

            command_scheduler = Sim2MuJoCoCommandScheduler(
                eval_config=eval_config,
                command_manager=command_manager,
                duration=args.duration,
                command_dim=command_provider.command_dim,
                verbose=args.verbose,
            )
            print("\n✓ Eval config active (command schedule from YAML)")
        elif args.random_commands is not None:
            from agile.sim2mujoco.command_scheduler import RandomCommandScheduler

            command_scheduler = RandomCommandScheduler(
                command_manager=command_manager,
                randomize_fields=args.random_commands,
                interval=args.random_interval,
                seed=args.random_seed,
                verbose=True,
            )
            print("\n✓ Random commands active")
        elif not args.disable_keyboard and not args.no_viewer:
            print("\n✓ Keyboard control enabled")
        else:
            print("\n✓ Keyboard control disabled (command manager active for default commands)")
    elif command_provider is not None:
        print(f"\n✓ Command provider: {command_provider.command_type} (dim={command_provider.command_dim})")
    else:
        print("\n✓ No command terms in policy")

    print("\nSetting up action processor...")
    act_processor = ActionProcessor(config, sim.joint_names, device)
    print(f"  Total action dim: {act_processor.total_action_dim}")
    print("  Action terms:")
    for term in act_processor.action_terms:
        print(f"    - {term.name}: {term.action_dim} joints (scale: {term.scale})")

    # Reset.
    sim.reset()
    obs_processor.reset()
    policy.reset()

    # Setup data logger.
    data_logger = None
    if args.save_data:
        if args.output_dir is not None:
            output_dir = args.output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if eval_config is not None:
                task_name = eval_config.task_name
                eval_stem = args.eval_config.stem
                output_dir = Path("logs/sim2mujoco") / task_name / f"{eval_stem}_{timestamp}"
            elif args.random_commands is not None:
                fields_tag = "_".join(args.random_commands)
                seed_tag = f"_s{args.random_seed}" if args.random_seed is not None else ""
                output_dir = Path("logs/sim2mujoco") / f"random_{fields_tag}{seed_tag}_{timestamp}"
            else:
                output_dir = Path("logs/sim2mujoco") / f"{args.config.stem}_{timestamp}"

        provenance = {
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "eval_config": str(args.eval_config) if args.eval_config else None,
            "random_commands": args.random_commands,
            "random_interval": args.random_interval if args.random_commands else None,
            "random_seed": args.random_seed if args.random_commands else None,
            "noise_scale": args.noise_scale,
            "noise_seed": args.noise_seed,
        }
        data_logger = Sim2MuJoCoDataLogger(
            output_dir, config, sim.joint_names, sim.dt, provenance=provenance, command_provider=command_provider
        )

    # Evaluation loop parameters.
    control_dt = sim.dt  # This is physics_dt * decimation
    physics_dt = sim.physics_dt
    num_steps = int(args.duration / control_dt)

    # Real-time pacing: sync viewer at 30 Hz and sleep to match wall-clock time.
    real_time = not args.no_real_time
    render_dt = 1.0 / 30.0 if real_time else 0.0

    print(f"\nRunning evaluation for {args.duration}s ({num_steps} control steps)...")
    print(f"  Control frequency: {1.0 / control_dt:.1f} Hz")
    print(f"  Physics frequency: {1.0 / physics_dt:.1f} Hz")
    if real_time:
        print("  Viewer sync: 30 Hz (real-time pacing)")
    else:
        print(f"  Viewer sync: {1.0 / control_dt:.1f} Hz (no pacing)")
    print("-" * 80)

    total_steps = 0
    interrupted = False

    def _raise_keyboard_interrupt(*_args):
        """Convert SIGTERM to KeyboardInterrupt so finally block runs and data is saved."""
        raise KeyboardInterrupt

    if args.save_data and hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    try:
        wall_start = time.time()
        last_render = wall_start

        for step in range(num_steps):
            # Wait while paused (viewer stays responsive).
            was_paused = sim.paused or sim.step_once
            while sim.paused and not sim.step_once:
                was_paused = True
                time.sleep(0.01)
                if not sim.viewer.is_running():
                    raise KeyboardInterrupt
            # Reset wall clock reference after unpausing to avoid a burst of catch-up steps.
            if was_paused or sim.step_once:
                wall_start = time.time() - step * control_dt
                last_render = time.time()
            sim.step_once = False

            # Apply scheduled commands (before obs so policy sees updated commands).
            if command_scheduler is not None:
                command_scheduler.update(control_dt)

            # Get observations.
            sim_state = sim.get_state()
            obs = obs_processor.compute(sim_state, noise_scale=args.noise_scale)

            # Policy inference.
            with torch.no_grad():
                actions = policy(obs)

            # Update last action.
            obs_processor.set_last_action(actions)

            # Process actions.
            joint_cmd = act_processor.process(actions)

            # Step simulation (decimation times).
            for _ in range(sim.decimation):
                sim.step(joint_cmd)

            # Record data for analysis.
            if data_logger is not None:
                post_state = sim.get_state()
                commands = command_provider.get_commands() if command_provider is not None else None
                data_logger.record_step(post_state, joint_cmd, actions, commands)

            # Sync viewer at target frame rate.
            now = time.time()
            if now - last_render >= render_dt:
                sim.viewer.sync()
                last_render = now

            # Advance motion tracker for next step.
            obs_processor.step_motion()

            # Real-time pacing: sleep to match simulation time to wall-clock time.
            if real_time:
                target_wall = wall_start + (step + 1) * control_dt
                sleep_time = target_wall - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Logging (get fresh state AFTER simulation steps).
            if args.verbose and total_steps % args.log_freq == 0:
                current_state = post_state if data_logger is not None else sim.get_state()
                print(
                    f"Step {total_steps:4d} | "
                    f"Root pos: [{current_state.root_pos[0]:6.3f}, {current_state.root_pos[1]:6.3f}, {current_state.root_pos[2]:6.3f}] | "
                    f"Root vel: [{current_state.root_lin_vel[0]:6.3f}, {current_state.root_lin_vel[1]:6.3f}, {current_state.root_lin_vel[2]:6.3f}] | "
                    f"Action mean: {actions.mean().item():7.4f}, std: {actions.std().item():7.4f}"
                )

            total_steps += 1

        print("-" * 80)
        print(f"\nEvaluation complete! Ran {total_steps} steps.")

    except KeyboardInterrupt:
        interrupted = True
        print("\n\nInterrupted by user (Ctrl+C).")

    finally:
        if data_logger is not None and data_logger.has_data:
            if interrupted:
                print("Saving buffered data before exit...")
            data_logger.save_episode(0)
        sim.close()
        print("Simulation closed.")


if __name__ == "__main__":
    main()
