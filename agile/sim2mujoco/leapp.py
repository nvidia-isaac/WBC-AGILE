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

"""LEAPP bundle execution for Sim2MuJoCo.

Isaac Lab's LEAPP exporter writes a YAML pipeline description next to the exported
TorchScript or ONNX model. This module delegates LEAPP graph execution to
``leapp.InferenceManager`` and maps semantic inputs/outputs to the Sim2MuJoCo
state and command loop.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from agile.sim2mujoco.command_provider import (
    CommandProvider,
    HeightCommandProvider,
    MotionCommandProvider,
    VelocityCommandProvider,
)
from agile.sim2mujoco.commands import CommandManager
from agile.sim2mujoco.observations import MotionTracker
from agile.sim2mujoco.simulation import JointCommand, SimState
from agile.sim2mujoco.utils import quat_rotate_inverse

_LEAPP_REQUIRED_TOP_LEVEL_KEYS = {"models", "pipeline", "system information"}


@dataclass(frozen=True)
class _TensorBinding:
    key: str
    node_name: str
    tensor_name: str
    desc: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def is_leapp_description(path: Path) -> bool:
    """Return whether *path* looks like a LEAPP pipeline YAML."""
    try:
        data = _load_yaml(path)
    except OSError:
        return False
    return _LEAPP_REQUIRED_TOP_LEVEL_KEYS.issubset(data.keys())


def resolve_leapp_bundle(path: Path) -> Path:
    """Validate and return the path to an exported LEAPP YAML.

    Args:
        path: LEAPP ``.yaml`` description exported by Isaac Lab.

    Returns:
        The validated LEAPP YAML path.
    """
    if not path.exists():
        raise FileNotFoundError(f"LEAPP YAML path not found: {path}")
    if not path.is_file():
        raise ValueError(f"LEAPP path must be the exported YAML file, not a directory: {path}")
    if not is_leapp_description(path):
        raise ValueError(f"Not a LEAPP pipeline description: {path}")
    return path


def synthesize_sim_config(
    leapp_desc: dict[str, Any],
    mjcf_path: Path | str,
    default_kp: float = 100.0,
    default_kd: float = 1.0,
) -> dict[str, Any]:
    """Build a minimal Sim2MuJoCo config from the bundle + MJCF when there is no companion file.

    A self-contained bundle carries the policy joints' gains (in the graph) and the control
    frequency (``pipeline.configs.frequency``, the LEAPP ``GraphConfigs`` schema). Everything else
    the runner needs is defaulted here:

    - reset pose: all zeros (the policy drives the joints toward its own default at run time),
    - per-joint gains: ``default_kp`` / ``default_kd`` for every joint; the LEAPP graph overrides
      the policy-controlled joints at run time, so this only sets the un-controlled ones,
    - armature: left to the MJCF,
    - decimation: derived from the MJCF physics timestep so the policy runs at its trained rate
      regardless of the MJCF's step size.

    Raises:
        ValueError: if the bundle has no positive ``pipeline.configs.frequency`` (AGILE only runs
            bundles exported by ``scripts/export_policy_leapp.py``, which always records it).
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    n = len(joint_names)
    physics_dt = float(model.opt.timestep)
    frequency = leapp_desc.get("pipeline", {}).get("configs", {}).get("frequency")
    if not frequency or float(frequency) <= 0:
        raise ValueError(
            "LEAPP bundle has no positive 'pipeline.configs.frequency'. AGILE only runs bundles "
            "exported by scripts/export_policy_leapp.py, which records the control frequency; "
            "re-export this checkpoint."
        )
    decimation = max(1, round((1.0 / float(frequency)) / physics_dt))
    effective_hz = 1.0 / (physics_dt * decimation)
    if abs(effective_hz - float(frequency)) > 1e-3:
        warnings.warn(
            f"MJCF physics timestep {physics_dt}s does not evenly divide the bundle control period; "
            f"running at {effective_hz:.2f} Hz instead of the trained {float(frequency)} Hz.",
            stacklevel=2,
        )
    default_joint_pos = [0.0] * n
    default_joint_stiffness = [float(default_kp)] * n
    default_joint_damping = [float(default_kd)] * n
    robot_defaults = _exported_robot_articulation(
        leapp_desc,
        joint_names,
        {
            "default_joint_pos": default_joint_pos,
            "default_joint_stiffness": default_joint_stiffness,
            "default_joint_damping": default_joint_damping,
        },
    )
    return {
        "mjcf_path": str(mjcf_path),
        "scene": {"physics_dt": physics_dt, "decimation": decimation},
        "articulations": {
            "robot": {
                "joint_names": joint_names,
                "default_joint_pos": robot_defaults.get("default_joint_pos", default_joint_pos),
                "default_joint_stiffness": robot_defaults.get("default_joint_stiffness", default_joint_stiffness),
                "default_joint_damping": robot_defaults.get("default_joint_damping", default_joint_damping),
            }
        },
    }


def _exported_robot_articulation(
    leapp_desc: dict[str, Any],
    sim_joint_names: list[str],
    defaults: dict[str, list[float]],
) -> dict[str, list[float]]:
    """Return full-robot defaults exported in AGILE LEAPP metadata, mapped to MJCF order."""
    robot = leapp_desc.get("agile", {}).get("articulations", {}).get("robot", {})
    exported_joint_names = robot.get("joint_names", [])
    if not exported_joint_names:
        return {}

    result: dict[str, list[float]] = {}
    for key in ("default_joint_pos", "default_joint_stiffness", "default_joint_damping"):
        values = robot.get(key)
        if values is None:
            continue
        if len(values) != len(exported_joint_names):
            raise ValueError(
                f"LEAPP bundle agile.articulations.robot.{key} has {len(values)} values for "
                f"{len(exported_joint_names)} joint names"
            )
        mapped = list(defaults.get(key, [0.0] * len(sim_joint_names)))
        for sim_index, joint_name in enumerate(sim_joint_names):
            try:
                mapped[sim_index] = float(values[exported_joint_names.index(joint_name)])
            except ValueError:
                pass
        result[key] = mapped
    return result


def load_leapp_description(path: Path) -> dict[str, Any]:
    """Load and validate a LEAPP pipeline description."""
    data = _load_yaml(path)
    missing = sorted(_LEAPP_REQUIRED_TOP_LEVEL_KEYS.difference(data.keys()))
    if missing:
        raise ValueError(f"LEAPP YAML {path} is missing required keys: {missing}")
    if not isinstance(data["models"], dict) or not data["models"]:
        raise ValueError(f"LEAPP YAML {path} must contain at least one model")
    if not isinstance(data["pipeline"], dict):
        raise ValueError(f"LEAPP YAML {path} has invalid pipeline metadata")
    return data


def create_leapp_command_provider(
    leapp_desc: dict[str, Any],
    device: torch.device,
    config: dict[str, Any] | None = None,
) -> CommandProvider | None:
    """Create the command provider declared by the LEAPP model."""
    for binding in _external_input_bindings(leapp_desc):
        input_desc = binding.desc
        connection = input_desc.get("isaaclab_connection", "")
        if not connection.startswith("command:"):
            continue
        if _is_motion_tracking_reference_input(input_desc):
            continue

        shape = input_desc.get("shape", [])
        dim = _last_dim(shape)
        if _is_motion_tracking_command_input(input_desc):
            if config is None or not config.get("motion_tracking"):
                continue
            target_joint_names = _motion_target_joint_names(input_desc, config, leapp_desc)
            tracker = MotionTracker(config["motion_tracking"], target_joint_names, device)
            return MotionCommandProvider(tracker)
        if dim == 1:
            return HeightCommandProvider(device=device, default_height=0.72)
        if dim not in (3, 4):
            continue

        defaults = {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0, "height": 0.74 if dim == 4 else 0.72}
        return VelocityCommandProvider(CommandManager(device=device, defaults=defaults), dim)

    return None


def _is_motion_tracking_command_input(input_desc: dict[str, Any]) -> bool:
    """Return whether an input is explicitly connected to the motion-tracking command namespace."""
    connection = input_desc.get("isaaclab_connection", "")
    command_name = (
        connection.removeprefix("command:") if connection.startswith("command:") else input_desc.get("name", "")
    )
    return command_name in {
        "motion",
        "generated_commands",
        "reference_motion",
    }


def _is_motion_tracking_reference_input(input_desc: dict[str, Any]) -> bool:
    """Return whether an input is a motion-tracking reference observation, not the command vector."""
    connection = input_desc.get("isaaclab_connection", "")
    command_name = (
        connection.removeprefix("command:") if connection.startswith("command:") else input_desc.get("name", "")
    )
    return command_name in {"motion_anchor_pos_b", "motion_anchor_ori_b"}


def _motion_target_joint_names(
    input_desc: dict[str, Any], config: dict[str, Any], leapp_desc: dict[str, Any]
) -> list[str]:
    """Return target joint names for a motion command input."""
    dim = _last_dim(input_desc.get("shape", []))
    command_joint_count = dim // 2
    source_joint_names = config.get("motion_tracking", {}).get("motion_joint_names", [])
    exported_names = _element_names(input_desc)
    exported_joint_names = _motion_reference_joint_names(exported_names, command_joint_count)
    if (
        command_joint_count > 0
        and exported_joint_names
        and source_joint_names
        and all(name in source_joint_names for name in exported_joint_names)
    ):
        return exported_joint_names

    policy_joint_names = _exported_policy_joint_names(leapp_desc, command_joint_count)
    if policy_joint_names:
        return policy_joint_names
    return config.get("articulations", {}).get("robot", {}).get("joint_names", [])[:command_joint_count]


def _exported_policy_joint_names(leapp_desc: dict[str, Any], joint_count: int) -> list[str]:
    """Infer policy joint order from exported robot joint state inputs."""
    if joint_count <= 0:
        return []
    for binding in _external_input_bindings(leapp_desc):
        desc = binding.desc
        connection = desc.get("isaaclab_connection")
        if connection not in {"state:robot:joint_pos", "state:robot:joint_vel"}:
            continue
        names = _element_names(desc)
        if len(names) == joint_count:
            return names
    return []


def _motion_reference_joint_names(element_names: list[str], joint_count: int) -> list[str]:
    """Return joint names from motion-reference command element names."""
    if len(element_names) == joint_count:
        return element_names
    if len(element_names) != 2 * joint_count:
        return []

    position_names = [_strip_motion_reference_prefix(name, "ref_joint_pos/") for name in element_names[:joint_count]]
    velocity_names = [_strip_motion_reference_prefix(name, "ref_joint_vel/") for name in element_names[joint_count:]]
    if position_names == velocity_names and all(position_names):
        return position_names
    return []


def _strip_motion_reference_prefix(name: str, prefix: str) -> str:
    if name.startswith(prefix):
        return name[len(prefix) :]
    return ""


class LeappPolicyController:
    """Run a LEAPP-exported policy graph and return MuJoCo joint commands."""

    def __init__(
        self,
        yaml_path: Path,
        config: dict[str, Any],
        sim_joint_names: list[str],
        device: torch.device,
        command_provider: CommandProvider | None = None,
    ):
        self.yaml_path = yaml_path
        self.bundle_dir = yaml_path.parent
        self.desc = load_leapp_description(yaml_path)
        self.config = config
        self.sim_joint_names = sim_joint_names
        self.device = device
        self.command_provider = command_provider
        self.model_names = list(self.desc["models"])
        self.model_paths = {}
        for model_name, model_desc in self.desc["models"].items():
            model_path = self._resolve_model_path(model_desc)
            if model_path is not None:
                self.model_paths[model_name] = model_path

        try:
            from leapp import InferenceManager
        except ImportError as exc:
            raise ImportError(
                "LEAPP bundle execution requires the `leapp` package. "
                "Install AGILE with `uv sync --frozen` or install `leapp==0.5.2`."
            ) from exc

        self.manager = InferenceManager(str(yaml_path))
        self.input_bindings = _external_input_bindings(self.desc)
        self.input_devices = {
            binding.key: torch.device(self.manager.nodes[binding.node_name].device) for binding in self.input_bindings
        }
        self.output_bindings = _pipeline_output_bindings(self.desc)

        self.default_position, self.default_kp, self.default_kd = self._load_default_joint_command()
        self.position_output = self._find_output(kind="target/joint/position", write_method="set_joint_position")
        self.kp_output = self._find_output(kind="kp", write_method="write_joint_stiffness")
        self.kd_output = self._find_output(kind="kd", write_method="write_joint_damping")
        if self.position_output is None:
            raise ValueError(f"LEAPP YAML {yaml_path} has no joint position target output")

        self.last_action = torch.zeros(_last_dim(self.position_output.desc.get("shape", [])), device=device)

    def reset(self) -> None:
        """Reset feedback tensors and logged action state."""
        self.manager.reset()
        self.last_action.zero_()

    def process(self, sim_state: SimState) -> JointCommand:
        """Run policy inference and map LEAPP outputs to a MuJoCo joint command."""
        inputs = {
            binding.key: self._build_input(binding.desc, sim_state, self.input_devices[binding.key])
            for binding in self.input_bindings
        }
        outputs = self.manager.run_policy(inputs)

        position = self.default_position.clone()
        kp = self.default_kp.clone()
        kd = self.default_kd.clone()

        target_values = _flatten_output(outputs[self.position_output.key], self.device)
        self._scatter_joint_output(position, target_values, self.position_output.desc)
        self.last_action = target_values.detach().clone()

        if self.kp_output is not None and self.kp_output.key in outputs:
            self._scatter_joint_output(
                kp, _flatten_output(outputs[self.kp_output.key], self.device), self.kp_output.desc
            )
        if self.kd_output is not None and self.kd_output.key in outputs:
            self._scatter_joint_output(
                kd, _flatten_output(outputs[self.kd_output.key], self.device), self.kd_output.desc
            )

        return JointCommand(position=position, kp=kp, kd=kd)

    @property
    def action_dim(self) -> int:
        """Dimension of the primary LEAPP joint-target output."""
        return int(self.last_action.numel())

    def _resolve_model_path(self, model_desc: dict[str, Any]) -> Path | None:
        params = model_desc.get("parameters", {})
        model_path = params.get("model_path")
        if not model_path:
            return None

        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = self.bundle_dir / model_path
        return model_path

    def _build_input(
        self, input_desc: dict[str, Any], sim_state: SimState, target_device: torch.device
    ) -> torch.Tensor:
        name = input_desc["name"]
        connection = input_desc.get("isaaclab_connection")
        if connection is None:
            raise ValueError(f"External LEAPP input '{name}' has no isaaclab_connection metadata")

        value = self._read_connection(connection, input_desc, sim_state)
        return _reshape_like_desc(value, input_desc, target_device)

    def _read_connection(self, connection: str, input_desc: dict[str, Any], sim_state: SimState) -> torch.Tensor:
        parts = connection.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid LEAPP isaaclab_connection: {connection}")

        if parts[0] == "command":
            return self._read_command(parts[1], input_desc, sim_state)

        if parts[0] != "state" or len(parts) != 3:
            raise ValueError(f"Unsupported LEAPP input connection: {connection}")

        entity_name, property_name = parts[1], parts[2]
        if entity_name != "robot":
            raise ValueError(f"Sim2MuJoCo only supports robot state inputs, got: {connection}")

        if property_name == "root_lin_vel_b":
            return sim_state.root_lin_vel.float()
        if property_name == "root_ang_vel_b":
            return sim_state.root_ang_vel.float()
        if property_name == "projected_gravity_b":
            gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
            return quat_rotate_inverse(sim_state.root_quat.float(), gravity_world)
        if property_name == "root_quat_w":
            return self._root_quat_for_desc(sim_state.root_quat.float(), input_desc)
        if property_name == "joint_pos":
            return self._gather_joint_state(sim_state.joint_pos.float(), input_desc)
        if property_name == "joint_vel":
            return self._gather_joint_state(sim_state.joint_vel.float(), input_desc)

        raise ValueError(f"Unsupported LEAPP robot state input: {connection}")

    def _read_command(self, command_name: str, input_desc: dict[str, Any], sim_state: SimState) -> torch.Tensor:
        dim = _last_dim(input_desc.get("shape", []))
        if self.command_provider is None:
            return torch.zeros(dim, device=self.device, dtype=torch.float32)

        if hasattr(self.command_provider, "get_named_command"):
            command = self.command_provider.get_named_command(command_name, sim_state).to(
                self.device, dtype=torch.float32
            )
        else:
            command = self.command_provider.get_commands().to(self.device, dtype=torch.float32)
        if command.numel() == dim:
            return command
        if command.numel() > dim:
            return command[:dim]
        return torch.cat([command, torch.zeros(dim - command.numel(), device=self.device, dtype=torch.float32)])

    def _root_quat_for_desc(self, root_quat_wxyz: torch.Tensor, input_desc: dict[str, Any]) -> torch.Tensor:
        names = _element_names(input_desc)
        if names in (["qx", "qy", "qz", "qw"], ["x", "y", "z", "w"]):
            return torch.stack([root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]])
        return root_quat_wxyz

    def _gather_joint_state(self, values: torch.Tensor, input_desc: dict[str, Any]) -> torch.Tensor:
        joint_names = _element_names(input_desc)
        if not joint_names:
            return values
        indices = [self.sim_joint_names.index(name) for name in joint_names]
        return values[indices]

    def _load_default_joint_command(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        robot_cfg = self.config.get("articulations", {}).get("robot", {})
        config_joint_names = robot_cfg.get("joint_names", [])

        position = self._map_joint_values(robot_cfg.get("default_joint_pos"), config_joint_names, default=0.0)
        kp = self._map_joint_values(robot_cfg.get("default_joint_stiffness"), config_joint_names, default=0.0)
        kd = self._map_joint_values(robot_cfg.get("default_joint_damping"), config_joint_names, default=0.0)
        return position, kp, kd

    def _map_joint_values(
        self,
        values: list[float] | None,
        value_joint_names: list[str],
        default: float,
    ) -> torch.Tensor:
        result = torch.full((len(self.sim_joint_names),), default, device=self.device, dtype=torch.float32)
        if values is None or not value_joint_names:
            return result
        for idx, joint_name in enumerate(value_joint_names):
            if joint_name in self.sim_joint_names and idx < len(values):
                result[self.sim_joint_names.index(joint_name)] = float(values[idx])
        return result

    def _find_output(self, *, kind: str, write_method: str) -> _TensorBinding | None:
        for output in self.output_bindings:
            output_desc = output.desc
            if output_desc.get("kind") == kind:
                return output
            connection = output_desc.get("isaaclab_connection", "")
            if connection.startswith("write:") and write_method in connection:
                return output
        return None

    def _scatter_joint_output(
        self,
        target: torch.Tensor,
        values: torch.Tensor,
        output_desc: dict[str, Any],
    ) -> None:
        joint_names = _element_names(output_desc)
        if not joint_names:
            if values.numel() != target.numel():
                raise ValueError(
                    f"Cannot map LEAPP output '{output_desc['name']}' without element_names: "
                    f"{values.numel()} values for {target.numel()} MuJoCo joints"
                )
            target[:] = values
            return

        if values.numel() != len(joint_names):
            raise ValueError(
                f"LEAPP output '{output_desc['name']}' has {values.numel()} values but {len(joint_names)} element names"
            )
        for value, joint_name in zip(values, joint_names, strict=True):
            target[self.sim_joint_names.index(joint_name)] = value


def _external_input_bindings(desc: dict[str, Any]) -> list[_TensorBinding]:
    bindings: list[_TensorBinding] = []
    for node_name, input_names in desc.get("pipeline", {}).get("inputs", {}).items():
        model_desc = _model_desc(desc, node_name)
        input_descs = _descs_by_name(model_desc.get("inputs", []))
        for input_name in input_names:
            bindings.append(
                _TensorBinding(
                    key=f"{node_name}/{input_name}",
                    node_name=node_name,
                    tensor_name=input_name,
                    desc=input_descs[input_name],
                )
            )
    return bindings


def _pipeline_output_bindings(desc: dict[str, Any]) -> list[_TensorBinding]:
    bindings: list[_TensorBinding] = []
    for node_name, output_names in desc.get("pipeline", {}).get("outputs", {}).items():
        model_desc = _model_desc(desc, node_name)
        output_descs = _descs_by_name(model_desc.get("outputs", []))
        for output_name in output_names:
            bindings.append(
                _TensorBinding(
                    key=f"{node_name}/{output_name}",
                    node_name=node_name,
                    tensor_name=output_name,
                    desc=output_descs[output_name],
                )
            )
    return bindings


def _model_desc(desc: dict[str, Any], node_name: str) -> dict[str, Any]:
    try:
        return desc["models"][node_name]
    except KeyError as exc:
        raise ValueError(f"LEAPP pipeline references unknown model node: {node_name}") from exc


def _descs_by_name(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in items}


def _element_names(desc: dict[str, Any]) -> list[str]:
    names = desc.get("element_names")
    if not names or not isinstance(names, list):
        return []
    first_axis = names[0]
    if isinstance(first_axis, str):
        return names
    if isinstance(first_axis, list):
        return [str(item) for item in first_axis]
    return []


def _last_dim(shape: Any) -> int:
    if isinstance(shape, str):
        shape = json.loads(shape)
    if not isinstance(shape, list) or not shape:
        return 0
    return int(shape[-1])


def _prod(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _reshape_like_desc(value: torch.Tensor, desc: dict[str, Any], device: torch.device) -> torch.Tensor:
    shape = desc.get("shape", [])
    if isinstance(shape, str):
        shape = json.loads(shape)
    value = value.to(device, dtype=torch.float32)
    if tuple(value.shape) == tuple(shape):
        return value
    if value.ndim == 1 and len(shape) == 2 and shape[0] == 1 and value.numel() == shape[1]:
        return value.unsqueeze(0)
    if value.numel() == _prod(shape):
        return value.reshape(tuple(shape))
    raise ValueError(f"Cannot reshape value with shape {tuple(value.shape)} to LEAPP shape {shape}")


def _flatten_output(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device, dtype=torch.float32).reshape(-1)
