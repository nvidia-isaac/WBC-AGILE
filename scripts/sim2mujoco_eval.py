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

"""Main entry point for sim2mujoco evaluation.

Examples:
    # LEAPP YAML with interactive keyboard control:
    uv run agile-download-assets
    uv run scripts/sim2mujoco_eval.py \
        --leapp-yaml logs/rsl_rl/velocity_height_g1_lower/<run>/Velocity-Height-G1-History-v0/Velocity-Height-G1-History-v0.yaml \
        --mjcf external_assets/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
        --duration 100.0

    # With eval config (deterministic command schedule, duration from eval config):
    uv run scripts/sim2mujoco_eval.py \
        --leapp-yaml logs/rsl_rl/velocity_height_g1_lower/<run>/Velocity-Height-G1-History-v0/Velocity-Height-G1-History-v0.yaml \
        --mjcf external_assets/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
        --eval-config agile/sim2mujoco/configs/x_velocity_sweep.yaml \
        --save-data --no-viewer

    # Random commands (randomize only vx, for comparison with deterministic sweep):
    uv run scripts/sim2mujoco_eval.py \
        --leapp-yaml logs/rsl_rl/velocity_height_g1_lower/<run>/Velocity-Height-G1-History-v0/Velocity-Height-G1-History-v0.yaml \
        --mjcf external_assets/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
        --random-commands vx --random-interval 2.0 --random-seed 42 \
        --duration 50.0 --save-data --no-viewer
"""

import argparse
import signal
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import torch
import yaml

from agile.sim2mujoco.command_provider import HeightCommandProvider, VelocityCommandProvider
from agile.sim2mujoco.data_logger import Sim2MuJoCoDataLogger
from agile.sim2mujoco.leapp import (
    LeappPolicyController,
    create_leapp_command_provider,
    load_leapp_description,
    resolve_leapp_bundle,
    synthesize_sim_config,
)
from agile.sim2mujoco.simulation import MuJocoSimulation
from agile.sim2mujoco.utils import default_device

_DEFAULT_DURATION_S = 10.0


@dataclass
class BalanceResult:
    passed: bool
    min_pelvis_height: float | None
    message: str


class BalanceMonitor:
    """Validate that the floating base stays above a minimum pelvis height."""

    def __init__(self, min_pelvis_height: float | None, start_time_s: float = 0.0):
        self.min_pelvis_height = min_pelvis_height
        self.start_time_s = start_time_s
        self._min_observed: float | None = None
        self._min_time_s: float | None = None

    @property
    def enabled(self) -> bool:
        return self.min_pelvis_height is not None

    def update(self, time_s: float, pelvis_height: float) -> None:
        if not self.enabled or time_s < self.start_time_s:
            return
        if self._min_observed is None or pelvis_height < self._min_observed:
            self._min_observed = pelvis_height
            self._min_time_s = time_s

    def result(self) -> BalanceResult:
        if not self.enabled:
            return BalanceResult(True, None, "balance validation disabled")
        if self._min_observed is None:
            return BalanceResult(True, None, "balance validation had no samples")
        assert self.min_pelvis_height is not None
        if self._min_observed < self.min_pelvis_height:
            return BalanceResult(
                False,
                self._min_observed,
                (
                    f"pelvis height dropped to {self._min_observed:.3f} m at "
                    f"{self._min_time_s:.2f} s; required >= {self.min_pelvis_height:.3f} m"
                ),
            )
        return BalanceResult(
            True,
            self._min_observed,
            f"minimum pelvis height {self._min_observed:.3f} m >= {self.min_pelvis_height:.3f} m",
        )


def _resolve_duration(cli_duration: float | None, eval_config: object | None) -> float:
    """Prefer an explicit CLI duration, then the scenario duration, then the default."""
    if cli_duration is not None:
        return cli_duration
    if eval_config is not None:
        return float(eval_config.episode_length_s)
    return _DEFAULT_DURATION_S


def _load_sim2mujoco_options(eval_config_path: Path | None) -> dict:
    """Load optional MuJoCo-only scenario options from an EvalConfig YAML."""
    if eval_config_path is None:
        return {}
    data = yaml.safe_load(eval_config_path.read_text())
    evaluation = data.get("evaluation", {}) if isinstance(data, dict) else {}
    options = evaluation.get("sim2mujoco", {})
    if not isinstance(options, dict):
        raise ValueError("evaluation.sim2mujoco must be a mapping when present")
    return options


def _load_policy_config(path: str | Path) -> dict:
    policy_path = Path(path)
    if not policy_path.is_absolute():
        policy_path = Path.cwd() / policy_path
    if not policy_path.is_file():
        raise FileNotFoundError(f"Policy config not found: {policy_path}")
    data = yaml.safe_load(policy_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Policy config must be a mapping: {policy_path}")
    return data


def _resolve_initial_state(initial_state: str | None, command_provider: object, sim_joint_names: list[str]):
    """Resolve a scenario initial state to a MuJoCo reset argument."""
    if initial_state == "reference_motion":
        tracker = getattr(command_provider, "_tracker", None)
        if tracker is None or not hasattr(tracker, "get_initial_state"):
            raise ValueError("initial_state=reference_motion requires a motion-tracking command provider")
        return tracker.get_initial_state(sim_joint_names)
    return initial_state


def main():
    """Run sim2sim evaluation."""
    parser = argparse.ArgumentParser(description="Sim2Sim Policy Evaluation")
    parser.add_argument(
        "--leapp-yaml",
        type=Path,
        required=True,
        help="Path to the LEAPP YAML exported by scripts/export_policy_leapp.py.",
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=None,
        help="Path to the MuJoCo MJCF (required; joint names and physics dt are read from it)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Simulation duration in seconds (defaults to the eval config duration, or 10 without a config)",
    )
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
    parser.add_argument(
        "--eval-env-id",
        type=int,
        default=None,
        help="Select one environment from a multi-environment eval config and remap it to MuJoCo environment 0.",
    )
    parser.add_argument("--save-data", action="store_true", help="Save evaluation data to disk")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for saved data")
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
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="If set, write an mp4 of the rollout to this path (headless offscreen render).",
    )

    args = parser.parse_args()

    if args.random_commands and args.eval_config:
        parser.error("--random-commands and --eval-config are mutually exclusive")
    if args.leapp_yaml.is_dir():
        parser.error("--leapp-yaml must point to the exported LEAPP YAML file, not the bundle directory")

    # Setup device.
    if args.device == "auto":
        device = default_device()
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load the bundle and build the simulation config. The bundle is self-contained: the policy
    # joints' gains ride in the graph and the control frequency in ``pipeline.configs.frequency``;
    # the rest (reset pose, non-policy gains, decimation) is synthesized from the bundle + MJCF +
    # defaults.
    leapp_yaml_path = resolve_leapp_bundle(args.leapp_yaml)
    leapp_desc = load_leapp_description(leapp_yaml_path)
    print(f"\nLoading LEAPP YAML from {leapp_yaml_path}...")
    if args.mjcf is None:
        raise ValueError("--mjcf is required: joint names and the physics timestep are read from it.")
    config = synthesize_sim_config(leapp_desc, args.mjcf)

    # Override MJCF path if provided.
    if args.mjcf:
        config["mjcf_path"] = str(args.mjcf)

    # Scale PD gains if requested.
    if args.pd_scale != 1.0:
        print(f"Scaling PD gains by {args.pd_scale}...")

    # Load eval config if provided (YAML-defined command schedule).
    eval_config = None
    sim2mujoco_options = _load_sim2mujoco_options(args.eval_config)
    if args.eval_config is not None:
        if not args.eval_config.exists():
            raise FileNotFoundError(f"Eval config not found: {args.eval_config}")
        from agile.algorithms.evaluation.eval_config import EvalConfig

        eval_config = EvalConfig.from_yaml(args.eval_config)
        if args.eval_env_id is not None:
            selected_env = eval_config.get_env_config(args.eval_env_id)
            if selected_env is None:
                raise ValueError(
                    f"Eval config has no schedule for env id {args.eval_env_id}. "
                    f"Found env_ids: {[e.env_ids for e in eval_config.environments]}"
                )
            eval_config = replace(
                eval_config,
                num_envs=1,
                num_episodes=1,
                environments=[replace(selected_env, env_ids=[0])],
            )
        elif eval_config.num_envs != 1 or eval_config.num_episodes != 1 or eval_config.get_env_config(0) is None:
            raise ValueError(
                "sim2mujoco requires a single-environment, one-episode eval config. "
                "Pass --eval-env-id to select one environment from a multi-environment scenario."
            )
    args.duration = _resolve_duration(args.duration, eval_config)
    if eval_config is not None:
        print(f"\n✓ Loaded eval config from {args.eval_config} (duration={args.duration}s)")

    motion_policy_config = sim2mujoco_options.get("motion_tracking_policy_config")
    if motion_policy_config is not None:
        policy_config = _load_policy_config(motion_policy_config)
        motion_tracking = policy_config.get("motion_tracking")
        if not isinstance(motion_tracking, dict):
            raise ValueError(f"Policy config has no motion_tracking section: {motion_policy_config}")
        config["motion_tracking"] = motion_tracking
        print(f"\n✓ Loaded motion-tracking config from {motion_policy_config}")

    validation_options = sim2mujoco_options.get("validation", {})
    if validation_options is None:
        validation_options = {}
    if not isinstance(validation_options, dict):
        raise ValueError("evaluation.sim2mujoco.validation must be a mapping when present")
    balance_options = validation_options.get("balance", {})
    if balance_options is None:
        balance_options = {}
    if not isinstance(balance_options, dict):
        raise ValueError("evaluation.sim2mujoco.validation.balance must be a mapping when present")
    min_pelvis_height = balance_options.get("min_pelvis_height")
    balance_monitor = BalanceMonitor(
        min_pelvis_height=None if min_pelvis_height is None else float(min_pelvis_height),
        start_time_s=float(balance_options.get("start_time_s", 0.0)),
    )
    if balance_monitor.enabled:
        print(
            "\n✓ Balance validation active "
            f"(min_pelvis_height={balance_monitor.min_pelvis_height:.3f} m, "
            f"start_time_s={balance_monitor.start_time_s:.2f})"
        )

    # Create simulation (command_manager will be attached after provider creation).
    print("\nCreating simulation...")
    sim = MuJocoSimulation(config, device, enable_viewer=not args.no_viewer, mjcf_path=args.mjcf)
    print(f"  Num joints: {sim.num_joints}")
    print(f"  Fixed base: {sim.fixed_base}")
    print(f"  Physics dt: {sim.physics_dt}s ({1.0 / sim.physics_dt:.0f} Hz)")
    print(f"  Control dt: {sim.dt}s ({1.0 / sim.dt:.1f} Hz)")
    print(f"  Decimation: {sim.decimation}")

    print("\nLoading LEAPP policy controller...")
    command_provider = create_leapp_command_provider(leapp_desc, device, config=config)
    leapp_controller = LeappPolicyController(
        leapp_yaml_path,
        config,
        sim.joint_names,
        device,
        command_provider=command_provider,
    )
    print(f"  LEAPP nodes: {', '.join(leapp_controller.model_names)}")
    for model_name, model_path in leapp_controller.model_paths.items():
        print(f"  {model_name} model file: {model_path}")
    print(f"  Action dim: {leapp_controller.action_dim}")

    # Wire up CommandManager-based features (keyboard, scheduler) for velocity providers.
    command_manager = None
    command_scheduler = None
    if isinstance(command_provider, VelocityCommandProvider):
        command_manager = command_provider.manager
        sim.command_manager = command_manager

        if eval_config is not None:
            from agile.sim2mujoco.command_scheduler import Sim2MuJoCoCommandScheduler

            command_scheduler = Sim2MuJoCoCommandScheduler(
                eval_config=eval_config,
                duration=args.duration,
                command_manager=command_manager,
                command_provider=command_provider,
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
    elif isinstance(command_provider, HeightCommandProvider):
        if eval_config is not None:
            from agile.sim2mujoco.command_scheduler import Sim2MuJoCoCommandScheduler

            command_scheduler = Sim2MuJoCoCommandScheduler(
                eval_config=eval_config,
                duration=args.duration,
                command_provider=command_provider,
                command_dim=command_provider.command_dim,
                verbose=args.verbose,
            )
            print("\n✓ Eval config active (height schedule from YAML)")
        else:
            print("\n✓ Command provider: height")
    elif command_provider is not None:
        print(f"\n✓ Command provider: {command_provider.command_type} (dim={command_provider.command_dim})")
    else:
        print("\n✓ No command terms in policy")

    if hasattr(command_provider, "reset"):
        command_provider.reset()
    # Reset.
    sim.reset(
        initial_state=_resolve_initial_state(
            sim2mujoco_options.get("initial_state"),
            command_provider,
            sim.joint_names,
        )
    )
    leapp_controller.reset()

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
                output_dir = Path("logs/sim2mujoco") / f"{leapp_yaml_path.stem}_{timestamp}"

        provenance = {
            "leapp_yaml": str(leapp_yaml_path),
            "eval_config": str(args.eval_config) if args.eval_config else None,
            "random_commands": args.random_commands,
            "random_interval": args.random_interval if args.random_commands else None,
            "random_seed": args.random_seed if args.random_commands else None,
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

    frames = []
    capture_every = max(1, round((1.0 / 30.0) / (sim.physics_dt * sim.decimation))) if args.video else 0
    step_idx = 0

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

            # Run the LEAPP graph on the current MuJoCo state.
            sim_state = sim.get_state()
            joint_cmd = leapp_controller.process(sim_state)
            actions = leapp_controller.last_action
            if args.pd_scale != 1.0:
                joint_cmd.kp = joint_cmd.kp * args.pd_scale
                joint_cmd.kd = joint_cmd.kd * args.pd_scale

            # Step simulation (decimation times).
            for _ in range(sim.decimation):
                sim.step(joint_cmd)

            post_state = sim.get_state()
            balance_monitor.update((step + 1) * control_dt, float(post_state.root_pos[2].item()))

            # Record data for analysis.
            if data_logger is not None:
                commands = command_provider.get_commands() if command_provider is not None else None
                data_logger.record_step(post_state, joint_cmd, actions, commands)
            if hasattr(command_provider, "step"):
                command_provider.step()

            if args.video and step_idx % capture_every == 0:
                frames.append(sim.capture_frame())
            step_idx += 1

            # Sync viewer at target frame rate.
            now = time.time()
            if now - last_render >= render_dt:
                sim.viewer.sync()
                last_render = now

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
        balance_result = balance_monitor.result()
        print(f"\nBalance validation: {balance_result.message}")
        if not balance_result.passed:
            raise RuntimeError(f"MuJoCo balance validation failed: {balance_result.message}")
        print(f"\nEvaluation complete! Ran {total_steps} steps.")

    except KeyboardInterrupt:
        interrupted = True
        print("\n\nInterrupted by user (Ctrl+C).")

    finally:
        if data_logger is not None and data_logger.has_data:
            if interrupted:
                print("Saving buffered data before exit...")
            data_logger.save_episode(0)
        if args.video and frames:
            import imageio.v3 as iio

            args.video.parent.mkdir(parents=True, exist_ok=True)
            # Play back at the true capture rate. capture_every rounds the 30 Hz target to an
            # integer step stride, so the effective rate usually differs from 30; writing 30 fps
            # then makes the video run fast (e.g. 3 s of sim -> 2.5 s clip). This keeps it real-time.
            effective_fps = 1.0 / (capture_every * control_dt)
            iio.imwrite(args.video, frames, fps=effective_fps)
            print(f"[INFO] Wrote sim2sim video ({len(frames)} frames @ {effective_fps:.1f} fps) to {args.video}")
        sim.close()
        print("Simulation closed.")


if __name__ == "__main__":
    main()
