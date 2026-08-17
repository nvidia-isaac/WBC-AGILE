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

"""Export an AGILE RSL-RL policy with Isaac Lab's LEAPP exporter.

This launcher reuses Isaac Lab's official LEAPP export logic, which AGILE vendors at
``third_party/isaaclab_leapp_export/export.py`` (Isaac Lab ships that driver only in its repo
``scripts/`` tree, which is not part of the wheels AGILE installs via uv -- see the vendored
file's header). It follows Isaac Lab's standard import ordering: launch the simulation app
FIRST (importing only ``isaaclab.app`` beforehand), and only then import Isaac Lab task
utilities and AGILE. This avoids (a) a native protobuf/gRPC clash from importing AGILE's deps
(e.g. ``wandb``) before Isaac Sim, and (b) class-identity mismatches from importing Isaac Lab
sub-packages before the app. Isaac Lab's own ``export.py`` entry point cannot be used directly
for AGILE because it resolves the task cfg via Hydra before launching the app.

Requirements:
    - A working AGILE uv environment (``isaaclab[isaacsim]`` + ``leapp`` are pulled by
      ``pyproject.toml``). No Isaac Lab source checkout / ``ISAACLAB_PATH`` is needed.

Example:
    uv run scripts/export_policy_leapp.py \\
        --task Velocity-G1-History-v0 \\
        --checkpoint logs/checkpoints/Velocity-G1-History-v0/model_<n>.pt

Arguments are those of Isaac Lab's LEAPP export script (e.g. ``--export_method``,
``--export_save_path``, ``--validation_steps``, ``--disable_graph_visualization``).
"""

import argparse
import sys
from pathlib import Path

# AGILE vendors Isaac Lab's LEAPP export driver so this works from a pure uv install.
_export_dir = Path(__file__).resolve().parents[1] / "third_party" / "isaaclab_leapp_export"
if not (_export_dir / "export.py").is_file():
    raise SystemExit(f"Vendored LEAPP export driver not found in: {_export_dir}")

# --- Launch the simulation app FIRST, importing only ``isaaclab.app`` beforehand. ---
from isaaclab.app import AppLauncher  # noqa: E402

_pre_parser = argparse.ArgumentParser(add_help=False)
AppLauncher.add_app_launcher_args(_pre_parser)
_pre_args, _ = _pre_parser.parse_known_args()
_pre_args.headless = True
app_launcher = AppLauncher(_pre_args)
simulation_app = app_launcher.app

# --- Now (post-app) import Isaac Lab task utilities, AGILE, and the LEAPP export module. ---
sys.path.insert(0, str(_export_dir))
import export  # noqa: E402

# Isaac Lab's export migrates IsaacLab's own rsl_rl cfgs across versions via
# ``handle_deprecated_rsl_rl_cfg``. AGILE's custom cfg + runner work together without it (it is
# never run during AGILE training) and migrating would corrupt the cfg, so bypass it here.
import isaaclab_rl.rsl_rl as _isaaclab_rl_rsl_rl  # noqa: E402

import agile.isaaclab_extras.monkey_patches  # noqa: E402, F401
import agile.rl_env.tasks  # noqa: E402, F401

_isaaclab_rl_rsl_rl.handle_deprecated_rsl_rl_cfg = lambda agent_cfg, *args, **kwargs: agent_cfg

# Route export through AGILE's wrapper so termination metadata and train-only action filtering match training.
from agile.evaluation.cli_exit import close_simulation_app  # noqa: E402
from agile.isaaclab_extras.leapp_export_fallback_patch import install_leapp_export_fallback_patch  # noqa: E402
from agile.rl_env.rsl_rl import make_rsl_rl_inference_load_cfg  # noqa: E402
from agile.rl_env.rsl_rl.export_pruning import remove_training_only_actions  # noqa: E402
from agile.rl_env.rsl_rl.leapp_export_metadata import update_leapp_export_metadata  # noqa: E402
from agile.rl_env.rsl_rl.rl_cfg import rsl_rl_cfg_to_dict  # noqa: E402
from agile.rl_env.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper as _AgileVecEnvWrapper  # noqa: E402

_isaaclab_rl_rsl_rl.RslRlVecEnvWrapper = _AgileVecEnvWrapper
install_leapp_export_fallback_patch()

# Parse the full LEAPP export CLI args (the app is already running; launcher args are unused here).
args_cli, _hydra_args = export.parse_export_args()

# Resolve the task's env/agent configs and run Isaac Lab's core LEAPP export function.
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)

# Drop training-only action terms before export. They consume no policy actions, so the trained
# policy's action space is unchanged.
for _removed_action in remove_training_only_actions(env_cfg):
    print(f"[INFO] Removed training-only action for export: {_removed_action}")

# Disable domain-randomization events so the exported bundle captures the nominal deployment
# values (gains, mass, ...) rather than one random per-episode draw. Detect them generically: the
# Isaac Lab / AGILE randomization MDP terms are all named ``randomize_*`` (both the event-term func
# and, by convention, the term itself). A hand-maintained name list is fragile because configs name
# the terms differently (e.g. randomize_actuator_gains vs randomize_lower/upper_actuator_gains).
# These events only perturb the env at startup/reset; the trained policy network is unchanged.
if getattr(env_cfg, "events", None) is not None:
    for _event_name, _event_term in list(vars(env_cfg.events).items()):
        _func = getattr(_event_term, "func", None)
        _func_name = getattr(_func, "__name__", "") if _func is not None else ""
        if _func_name.startswith("randomize") or _event_name.startswith("randomize"):
            setattr(env_cfg.events, _event_name, None)
            print(f"[INFO] Disabled domain-randomization event for nominal export: {_event_name}")

# Isaac Lab's LEAPP export selects the runner from ``agent_cfg.class_name``.
agent_cfg.class_name = rsl_rl_cfg_to_dict(agent_cfg)["class_name"]
agent_cfg.to_dict = lambda: rsl_rl_cfg_to_dict(agent_cfg)


# ---------------------------------------------------------------------------------------------
# TEMPORARY workarounds. These two helpers make the exported bundle self-contained and are
# inlined here because they are transitional -- each goes away once its upstream change lands:
#   * _populate_default_gains_from_actuators: AGILE's G1 uses explicit (DC-motor) actuators, so the
#     sim joint stiffness/damping Isaac Lab's LEAPP exporter reads for the bundle's kp/kd outputs
#     are 0. Copy the real actuator gains into those buffers just before annotation so the graph
#     carries them. Remove once isaac-sim/IsaacLab#6252 (export gains from the actuators) is in
#     AGILE's Isaac Lab version. NB: it writes data.default_joint_stiffness/damping, which Isaac Lab
#     deprecates for removal in 4.0, so this shim must go before any 4.0 bump even if #6252 has not
#     landed.
#   * update_leapp_export_metadata: record the policy control rate in the bundle under
#     ``pipeline.configs.frequency`` (the LEAPP ``GraphConfigs`` schema, nvidia-isaac/leapp#6).
#     Remove the frequency annotation once AGILE uses a LEAPP with ``GraphConfigs`` and passes
#     ``graph_configs`` to ``compile_graph()`` directly (the produced frequency metadata is
#     identical).
# ---------------------------------------------------------------------------------------------
def _actuator_gains(robot):
    """Per-joint ``(kp, kd)`` from the robot's actuators (sim-level buffers are 0 for DC-motors)."""
    n = robot.num_joints
    kp = [0.0] * n
    kd = [0.0] * n
    for actuator in robot.actuators.values():
        joint_indices = actuator.joint_indices
        if isinstance(joint_indices, slice):
            indices = list(range(*joint_indices.indices(n)))
        else:
            indices = [int(i) for i in (joint_indices.tolist() if hasattr(joint_indices, "tolist") else joint_indices)]
        stiffness = actuator.stiffness[0].detach().cpu().tolist()
        damping = actuator.damping[0].detach().cpu().tolist()
        for local, joint in enumerate(indices):
            kp[joint] = float(stiffness[local])
            kd[joint] = float(damping[local])
    return kp, kd


_robot_articulation_defaults = None


def _tensor_row_to_list(value):
    tensor = value.torch if hasattr(value, "torch") else value
    if getattr(tensor, "ndim", 0) > 1:
        tensor = tensor[0]
    return [float(v) for v in tensor.detach().cpu().tolist()]


def _populate_default_gains_from_actuators(env):
    """Write actuator gains into ``data.default_joint_stiffness``/``damping`` so they reach the bundle."""
    global _robot_articulation_defaults

    import torch

    robot = env.unwrapped.scene["robot"]
    kp, kd = _actuator_gains(robot)
    kp_t = torch.tensor(kp, device=robot.device, dtype=torch.float32).unsqueeze(0)
    kd_t = torch.tensor(kd, device=robot.device, dtype=torch.float32).unsqueeze(0)
    robot.data.default_joint_stiffness.torch[:] = kp_t
    robot.data.default_joint_damping.torch[:] = kd_t
    _robot_articulation_defaults = {
        "joint_names": list(robot.joint_names),
        "default_joint_pos": _tensor_row_to_list(robot.data.default_joint_pos),
        "default_joint_stiffness": kp,
        "default_joint_damping": kd,
    }


exported = export.export_rsl_rl_agent(
    args_cli,
    env_cfg,
    agent_cfg,
    simulation_app=simulation_app,
    on_env_ready=_populate_default_gains_from_actuators,
    checkpoint_load_cfg=make_rsl_rl_inference_load_cfg(agent_cfg),
)

# Record the policy control frequency in the LEAPP bundle (``pipeline.configs.frequency``, the
# LEAPP GraphConfigs schema). The graph itself does not carry the policy's execution rate, and the
# Sim2MuJoCo runner needs it to set the control decimation. Everything else the runner needs (reset
# pose, non-policy-joint gains, armature) it defaults or takes from the MJCF, so no companion file
# is required.
if exported:
    # Mirror export_rsl_rl_agent's save-path logic to locate the bundle dir (LEAPP writes the graph
    # to ``<save_path>/<graph_name>/``). We cover the cases whose path is knowable up front; a remote
    # (Nucleus/HTTP) ``--checkpoint`` is downloaded to a temp dir, so pass ``--export_save_path`` for
    # those.
    _task_name = args_cli.task.split(":")[-1]
    _graph_name = args_cli.export_task_name if args_cli.export_task_name is not None else _task_name
    if args_cli.export_save_path is not None:
        _bundle_root = Path(args_cli.export_save_path)
    elif args_cli.use_pretrained_checkpoint:
        _bundle_root = Path(".pretrained_checkpoints") / "rsl_rl" / _task_name.replace("-Play", "")
    elif args_cli.checkpoint and "://" not in args_cli.checkpoint:
        _bundle_root = Path(args_cli.checkpoint).expanduser().resolve().parent
    else:
        _bundle_root = None

    _leapp_yaml = (_bundle_root / _graph_name / f"{_graph_name}.yaml") if _bundle_root is not None else None
    if _leapp_yaml is not None and _leapp_yaml.is_file():
        _frequency_hz = 1.0 / (env_cfg.decimation * env_cfg.sim.dt)
        update_leapp_export_metadata(
            _leapp_yaml,
            frequency_hz=_frequency_hz,
            robot_articulation=_robot_articulation_defaults,
        )
        print(f"[INFO] Ensured runtime metadata in {_leapp_yaml}")
    else:
        print(
            "[WARNING] Could not locate the exported LEAPP bundle; skipping control-frequency "
            "annotation. Use a local --checkpoint or pass --export_save_path."
        )

close_simulation_app(simulation_app, exit_code=0 if exported else 1)
