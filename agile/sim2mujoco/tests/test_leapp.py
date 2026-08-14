# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LEAPP bundle support."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from agile.sim2mujoco.leapp import (
    LeappPolicyController,
    create_leapp_command_provider,
    resolve_leapp_bundle,
    synthesize_sim_config,
)
from agile.sim2mujoco.simulation import SimState

# Minimal MJCF: a floating base (free joint, excluded) plus two hinge joints.
_MJCF = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <body name="base">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body name="l0">
        <joint name="j0" type="hinge"/>
        <geom type="sphere" size="0.1" mass="1"/>
        <body name="l1">
          <joint name="j1" type="hinge"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _TinyLeappModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("joint_pos_target", torch.tensor([[0.1, -0.2]], dtype=torch.float32))
        self.register_buffer("kp", torch.tensor([[10.0, 20.0]], dtype=torch.float32))
        self.register_buffer("kd", torch.tensor([[1.0, 2.0]], dtype=torch.float32))

    def forward(self, base_velocity, robot_joint_pos, last_action_in):
        dependency = base_velocity[:, :1] * 0.0 + robot_joint_pos[:, :1] * 0.0 + last_action_in[:, :1] * 0.0
        return (
            self.joint_pos_target + dependency,
            self.joint_pos_target + dependency,
            self.kp + dependency,
            self.kd + dependency,
        )


class TestLeappBundleSupport(unittest.TestCase):
    def test_resolve_leapp_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            yaml_path = bundle_dir / "Policy.yaml"
            yaml_path.write_text(
                yaml.safe_dump(
                    {
                        "models": {"Policy": {"inputs": [], "outputs": [], "parameters": {"model_path": "p.pt"}}},
                        "pipeline": {"inputs": {"Policy": []}, "outputs": {"Policy": []}},
                        "system information": {"leapp version": "test"},
                    }
                )
            )

            self.assertEqual(resolve_leapp_bundle(yaml_path), yaml_path)

    def test_resolve_rejects_bundle_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "exported YAML file"):
                resolve_leapp_bundle(Path(tmp))

    def test_synthesize_sim_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            mjcf_path = Path(tmp) / "robot.xml"
            mjcf_path.write_text(_MJCF)

            cfg = synthesize_sim_config(
                {"pipeline": {"configs": {"frequency": 50}}}, mjcf_path, default_kp=100.0, default_kd=1.0
            )

            robot = cfg["articulations"]["robot"]
            self.assertEqual(robot["joint_names"], ["j0", "j1"])  # free joint excluded
            self.assertEqual(robot["default_joint_pos"], [0.0, 0.0])
            self.assertEqual(robot["default_joint_stiffness"], [100.0, 100.0])
            self.assertEqual(robot["default_joint_damping"], [1.0, 1.0])
            self.assertEqual(cfg["scene"]["physics_dt"], 0.005)
            # decimation = round((1 / 50) / 0.005) = 4
            self.assertEqual(cfg["scene"]["decimation"], 4)

    def test_synthesize_sim_config_uses_exported_full_robot_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            mjcf_path = Path(tmp) / "robot.xml"
            mjcf_path.write_text(_MJCF)

            cfg = synthesize_sim_config(
                {
                    "pipeline": {"configs": {"frequency": 50}},
                    "agile": {
                        "articulations": {
                            "robot": {
                                "joint_names": ["j1", "j0"],
                                "default_joint_pos": [0.2, -0.1],
                                "default_joint_stiffness": [220.0, 110.0],
                                "default_joint_damping": [22.0, 11.0],
                            }
                        }
                    },
                },
                mjcf_path,
            )

            robot = cfg["articulations"]["robot"]
            self.assertEqual(robot["joint_names"], ["j0", "j1"])
            self.assertEqual(robot["default_joint_pos"], [-0.1, 0.2])
            self.assertEqual(robot["default_joint_stiffness"], [110.0, 220.0])
            self.assertEqual(robot["default_joint_damping"], [11.0, 22.0])

    def test_synthesize_sim_config_uses_defaults_for_joints_missing_from_exported_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            mjcf_path = Path(tmp) / "robot.xml"
            mjcf_path.write_text(_MJCF)

            cfg = synthesize_sim_config(
                {
                    "pipeline": {"configs": {"frequency": 50}},
                    "agile": {
                        "articulations": {
                            "robot": {
                                "joint_names": ["j0"],
                                "default_joint_stiffness": [110.0],
                                "default_joint_damping": [11.0],
                            }
                        }
                    },
                },
                mjcf_path,
                default_kp=100.0,
                default_kd=1.0,
            )

            robot = cfg["articulations"]["robot"]
            self.assertEqual(robot["default_joint_stiffness"], [110.0, 100.0])
            self.assertEqual(robot["default_joint_damping"], [11.0, 1.0])

    def test_synthesize_sim_config_missing_frequency_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            mjcf_path = Path(tmp) / "robot.xml"
            mjcf_path.write_text(_MJCF)

            with self.assertRaisesRegex(ValueError, "frequency"):
                synthesize_sim_config({"pipeline": {}}, mjcf_path)

    def test_controller_maps_torchscript_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            model_path = bundle_dir / "Policy.pt"
            model = _TinyLeappModel()
            traced = torch.jit.trace(
                model,
                (
                    torch.zeros(1, 3),
                    torch.zeros(1, 2),
                    torch.zeros(1, 2),
                ),
            )
            traced.save(model_path)

            leapp_yaml = {
                "models": {
                    "Policy": {
                        "inputs": [
                            {
                                "name": "base_velocity",
                                "dtype": "float32",
                                "shape": [1, 3],
                                "type": "tensor",
                                "kind": "command/body/velocity",
                                "element_names": [["lin_vel_x", "lin_vel_y", "ang_vel_z"]],
                                "isaaclab_connection": "command:base_velocity",
                            },
                            {
                                "name": "robot_joint_pos",
                                "dtype": "float32",
                                "shape": [1, 2],
                                "type": "tensor",
                                "kind": "state/joint/position",
                                "element_names": [["b_joint", "a_joint"]],
                                "isaaclab_connection": "state:robot:joint_pos",
                            },
                            {"name": "last_action_in", "dtype": "float32", "shape": [1, 2], "type": "tensor"},
                        ],
                        "outputs": [
                            {
                                "name": "joint_pos",
                                "dtype": "float32",
                                "shape": [1, 2],
                                "type": "tensor",
                                "kind": "target/joint/position",
                                "element_names": [["b_joint", "a_joint"]],
                                "isaaclab_connection": "write:robot:set_joint_position_target_index",
                            },
                            {"name": "last_action_out", "dtype": "float32", "shape": [1, 2], "type": "tensor"},
                            {
                                "name": "joint_pos_kp_gains",
                                "dtype": "float32",
                                "shape": [1, 2],
                                "type": "tensor",
                                "kind": "kp",
                                "element_names": [["b_joint", "a_joint"]],
                                "isaaclab_connection": "write:robot:write_joint_stiffness_to_sim_index",
                            },
                            {
                                "name": "joint_pos_kd_gains",
                                "dtype": "float32",
                                "shape": [1, 2],
                                "type": "tensor",
                                "kind": "kd",
                                "element_names": [["b_joint", "a_joint"]],
                                "isaaclab_connection": "write:robot:write_joint_damping_to_sim_index",
                            },
                        ],
                        "parameters": {
                            "model_path": model_path.name,
                            "sha256sum": sha256sum(model_path),
                            "md5sum": "unused",
                            "backend": "jit",
                        },
                    }
                },
                "pipeline": {
                    "inputs": {"Policy": ["base_velocity", "robot_joint_pos"]},
                    "outputs": {"Policy": ["joint_pos", "joint_pos_kp_gains", "joint_pos_kd_gains"]},
                    "feedback_flow": {"Policy/last_action_out": ["Policy/last_action_in"]},
                },
                "system information": {"leapp version": "test"},
            }
            yaml_path = bundle_dir / "Policy.yaml"
            yaml_path.write_text(yaml.safe_dump(leapp_yaml, sort_keys=False))

            config = {
                "articulations": {
                    "robot": {
                        "joint_names": ["a_joint", "b_joint"],
                        "default_joint_pos": [0.0, 0.0],
                        "default_joint_stiffness": [0.0, 0.0],
                        "default_joint_damping": [0.0, 0.0],
                    }
                }
            }
            device = torch.device("cpu")
            controller = LeappPolicyController(yaml_path, config, ["a_joint", "b_joint"], device)
            state = SimState(
                joint_pos=torch.tensor([1.0, 2.0]),
                joint_vel=torch.zeros(2),
                root_pos=torch.zeros(3),
                root_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                root_lin_vel=torch.zeros(3),
                root_ang_vel=torch.zeros(3),
            )

            command = controller.process(state)
            torch.testing.assert_close(command.position, torch.tensor([-0.2, 0.1]))
            torch.testing.assert_close(command.kp, torch.tensor([20.0, 10.0]))
            torch.testing.assert_close(command.kd, torch.tensor([2.0, 1.0]))

    def test_motion_command_without_element_names_uses_exported_joint_state_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            motion_path = Path(tmp) / "motion.npz"
            np.savez(
                motion_path,
                fps=np.array(50.0, dtype=np.float32),
                joint_pos=np.array([[1.0, 2.0]], dtype=np.float32),
                joint_vel=np.array([[3.0, 4.0]], dtype=np.float32),
                body_pos_w=np.zeros((1, 1, 3), dtype=np.float32),
                body_quat_w=np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
                body_lin_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
                body_ang_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
            )
            leapp_desc = {
                "models": {
                    "Policy": {
                        "inputs": [
                            {
                                "name": "motion",
                                "dtype": "float32",
                                "shape": [1, 4],
                                "type": "tensor",
                                "isaaclab_connection": "command:motion",
                            },
                            {
                                "name": "robot_joint_pos",
                                "dtype": "float32",
                                "shape": [1, 2],
                                "type": "tensor",
                                "element_names": [["policy_b", "policy_a"]],
                                "isaaclab_connection": "state:robot:joint_pos",
                            },
                        ],
                        "outputs": [],
                    }
                },
                "pipeline": {"inputs": {"Policy": ["motion", "robot_joint_pos"]}, "outputs": {"Policy": []}},
                "system information": {"leapp version": "test"},
            }
            config = {
                "articulations": {"robot": {"joint_names": ["policy_a", "policy_b"]}},
                "motion_tracking": {
                    "motion_file": str(motion_path),
                    "anchor_body_name": "pelvis",
                    "motion_body_names": ["pelvis"],
                    "motion_joint_names": ["policy_a", "policy_b"],
                },
            }

            provider = create_leapp_command_provider(leapp_desc, torch.device("cpu"), config=config)

            torch.testing.assert_close(provider.get_commands(), torch.tensor([2.0, 1.0, 4.0, 3.0]))

    def test_motion_command_strips_reference_prefixes_from_element_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            motion_path = Path(tmp) / "motion.npz"
            np.savez(
                motion_path,
                fps=np.array(50.0, dtype=np.float32),
                joint_pos=np.array([[1.0, 2.0]], dtype=np.float32),
                joint_vel=np.array([[3.0, 4.0]], dtype=np.float32),
                body_pos_w=np.zeros((1, 1, 3), dtype=np.float32),
                body_quat_w=np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
                body_lin_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
                body_ang_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
            )
            leapp_desc = {
                "models": {
                    "Policy": {
                        "inputs": [
                            {
                                "name": "motion",
                                "dtype": "float32",
                                "shape": [1, 4],
                                "type": "tensor",
                                "element_names": [
                                    [
                                        "ref_joint_pos/policy_b",
                                        "ref_joint_pos/policy_a",
                                        "ref_joint_vel/policy_b",
                                        "ref_joint_vel/policy_a",
                                    ]
                                ],
                                "isaaclab_connection": "command:motion",
                            },
                        ],
                        "outputs": [],
                    }
                },
                "pipeline": {"inputs": {"Policy": ["motion"]}, "outputs": {"Policy": []}},
                "system information": {"leapp version": "test"},
            }
            config = {
                "articulations": {"robot": {"joint_names": ["policy_a", "policy_b"]}},
                "motion_tracking": {
                    "motion_file": str(motion_path),
                    "anchor_body_name": "pelvis",
                    "motion_body_names": ["pelvis"],
                    "motion_joint_names": ["policy_a", "policy_b"],
                },
            }

            provider = create_leapp_command_provider(leapp_desc, torch.device("cpu"), config=config)

            torch.testing.assert_close(provider.get_commands(), torch.tensor([2.0, 1.0, 4.0, 3.0]))

    def test_motion_anchor_command_without_motion_config_is_not_velocity_command(self):
        leapp_desc = {
            "models": {
                "Policy": {
                    "inputs": [
                        {
                            "name": "motion_anchor_pos_b",
                            "dtype": "float32",
                            "shape": [1, 3],
                            "type": "tensor",
                            "isaaclab_connection": "command:motion_anchor_pos_b",
                        },
                    ],
                    "outputs": [],
                }
            },
            "pipeline": {"inputs": {"Policy": ["motion_anchor_pos_b"]}, "outputs": {"Policy": []}},
            "system information": {"leapp version": "test"},
        }

        provider = create_leapp_command_provider(leapp_desc, torch.device("cpu"), config={})

        self.assertIsNone(provider)

    def test_motion_anchor_input_does_not_preempt_motion_command_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            motion_path = Path(tmp) / "motion.npz"
            np.savez(
                motion_path,
                fps=np.array(50.0, dtype=np.float32),
                joint_pos=np.array([[1.0, 2.0]], dtype=np.float32),
                joint_vel=np.array([[3.0, 4.0]], dtype=np.float32),
                body_pos_w=np.zeros((1, 1, 3), dtype=np.float32),
                body_quat_w=np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
                body_lin_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
                body_ang_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
            )
            leapp_desc = {
                "models": {
                    "Policy": {
                        "inputs": [
                            {
                                "name": "motion_anchor_ori_b",
                                "dtype": "float32",
                                "shape": [1, 6],
                                "type": "tensor",
                                "isaaclab_connection": "command:motion_anchor_ori_b",
                            },
                            {
                                "name": "generated_commands",
                                "dtype": "float32",
                                "shape": [1, 4],
                                "type": "tensor",
                                "element_names": [
                                    [
                                        "ref_joint_pos/policy_a",
                                        "ref_joint_pos/policy_b",
                                        "ref_joint_vel/policy_a",
                                        "ref_joint_vel/policy_b",
                                    ]
                                ],
                                "isaaclab_connection": "command:generated_commands",
                            },
                        ],
                        "outputs": [],
                    }
                },
                "pipeline": {
                    "inputs": {"Policy": ["motion_anchor_ori_b", "generated_commands"]},
                    "outputs": {"Policy": []},
                },
                "system information": {"leapp version": "test"},
            }
            config = {
                "motion_tracking": {
                    "motion_file": str(motion_path),
                    "anchor_body_name": "pelvis",
                    "motion_body_names": ["pelvis"],
                    "motion_joint_names": ["policy_a", "policy_b"],
                },
            }

            provider = create_leapp_command_provider(leapp_desc, torch.device("cpu"), config=config)

            self.assertEqual(provider.command_type, "motion_tracking")
            torch.testing.assert_close(provider.get_commands(), torch.tensor([1.0, 2.0, 3.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
