#!/usr/bin/env python3

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

"""Unit tests for Sim2MuJoCoDataLogger."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from agile.sim2mujoco.command_provider import VelocityCommandProvider, create_command_provider
from agile.sim2mujoco.commands import CommandManager
from agile.sim2mujoco.data_logger import Sim2MuJoCoDataLogger
from agile.sim2mujoco.simulation import JointCommand, SimState


def _make_sim_state(num_joints: int = 4, device: torch.device | None = None) -> SimState:
    """Create a minimal SimState for testing."""
    device = device or torch.device("cpu")
    return SimState(
        joint_pos=torch.zeros(num_joints, device=device),
        joint_vel=torch.zeros(num_joints, device=device),
        root_pos=torch.tensor([0.0, 0.0, 0.72], device=device),
        root_quat=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
        root_lin_vel=torch.zeros(3, device=device),
        root_ang_vel=torch.zeros(3, device=device),
    )


def _make_joint_cmd(num_joints: int = 4, device: torch.device | None = None) -> JointCommand:
    """Create a minimal JointCommand for testing."""
    device = device or torch.device("cpu")
    return JointCommand(
        position=torch.zeros(num_joints, device=device),
        kp=torch.ones(num_joints, device=device) * 50.0,
        kd=torch.ones(num_joints, device=device) * 2.0,
    )


def _make_velocity_provider(dim: int = 3) -> VelocityCommandProvider:
    """Create a VelocityCommandProvider for testing."""
    mgr = CommandManager(device=torch.device("cpu"))
    return VelocityCommandProvider(mgr, dim)


class TestCreateCommandProvider(unittest.TestCase):
    """Tests for create_command_provider factory."""

    def test_velocity_3d(self):
        """Velocity-only policy (locomotion_command) creates VelocityCommandProvider with dim=3."""
        config = {
            "observations": {
                "policy": [
                    {"name": "locomotion_command", "shape": [3]},
                ]
            }
        }
        provider = create_command_provider(config, torch.device("cpu"))
        self.assertIsNotNone(provider)
        self.assertEqual(provider.command_type, "velocity")
        self.assertEqual(provider.command_names, ["vx", "vy", "wz"])
        self.assertEqual(provider.command_dim, 3)

    def test_velocity_height_4d(self):
        """Velocity+height policy (generated_commands) creates VelocityCommandProvider with dim=4."""
        config = {
            "observations": {
                "policy": [
                    {"name": "generated_commands", "shape": [4]},
                ]
            }
        }
        provider = create_command_provider(config, torch.device("cpu"))
        self.assertIsNotNone(provider)
        self.assertEqual(provider.command_type, "velocity_height")
        self.assertEqual(provider.command_names, ["vx", "vy", "wz", "height"])
        self.assertEqual(provider.command_dim, 4)

    def test_no_command_term(self):
        """Policy without command term returns None."""
        config = {
            "observations": {
                "policy": [
                    {"name": "joint_pos_rel", "shape": [14]},
                ]
            }
        }
        provider = create_command_provider(config, torch.device("cpu"))
        self.assertIsNone(provider)


class TestSim2MuJoCoDataLogger(unittest.TestCase):
    """Tests for Sim2MuJoCoDataLogger."""

    def setUp(self):
        """Create temp dir for each test."""
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp dir."""
        import shutil

        if hasattr(self, "tmpdir") and Path(self.tmpdir).exists():
            shutil.rmtree(self.tmpdir)

    def test_velocity_3d_logs_commands_0_2_only(self):
        """Velocity-only policy logs exactly commands_0, commands_1, commands_2."""
        config = {
            "articulations": {"robot": {}},
            "scene": {"physics_dt": 0.005},
        }
        provider = _make_velocity_provider(3)
        logger = Sim2MuJoCoDataLogger(
            self.tmpdir, config, joint_names=["j0", "j1", "j2", "j3"], control_dt=0.02, command_provider=provider
        )
        self.assertEqual(logger.command_dim, 3)

        sim_state = _make_sim_state(4)
        joint_cmd = _make_joint_cmd(4)
        actions = torch.zeros(4)
        commands_3d = torch.tensor([0.1, 0.2, 0.3])

        logger.record_step(sim_state, joint_cmd, actions, commands=commands_3d, episode_id=0)
        logger.save_episode(0)

        df = __import__("pandas").read_parquet(Path(self.tmpdir) / "trajectories" / "episode_000.parquet")
        self.assertIn("commands_0", df.columns)
        self.assertIn("commands_1", df.columns)
        self.assertIn("commands_2", df.columns)
        self.assertNotIn("commands_3", df.columns)
        self.assertAlmostEqual(df["commands_0"].iloc[0], 0.1, places=6)
        self.assertAlmostEqual(df["commands_1"].iloc[0], 0.2, places=6)
        self.assertAlmostEqual(df["commands_2"].iloc[0], 0.3, places=6)

    def test_velocity_height_4d_logs_commands_0_3(self):
        """Velocity+height policy logs commands_0 through commands_3."""
        config = {
            "articulations": {"robot": {}},
            "scene": {"physics_dt": 0.005},
        }
        provider = _make_velocity_provider(4)
        logger = Sim2MuJoCoDataLogger(
            self.tmpdir, config, joint_names=["j0", "j1", "j2", "j3"], control_dt=0.02, command_provider=provider
        )
        self.assertEqual(logger.command_dim, 4)

        sim_state = _make_sim_state(4)
        joint_cmd = _make_joint_cmd(4)
        actions = torch.zeros(4)
        commands_4d = torch.tensor([0.1, 0.2, 0.3, 0.72])

        logger.record_step(sim_state, joint_cmd, actions, commands=commands_4d, episode_id=0)
        logger.save_episode(0)

        df = __import__("pandas").read_parquet(Path(self.tmpdir) / "trajectories" / "episode_000.parquet")
        for i in range(4):
            self.assertIn(f"commands_{i}", df.columns)
        self.assertAlmostEqual(df["commands_3"].iloc[0], 0.72, places=6)

    def test_metadata_has_command_dim(self):
        """Metadata includes command_dim and command_names from provider."""
        config = {
            "articulations": {"robot": {}},
            "scene": {"physics_dt": 0.005},
        }
        provider = _make_velocity_provider(3)
        _logger = Sim2MuJoCoDataLogger(
            self.tmpdir, config, joint_names=["j0", "j1"], control_dt=0.02, command_provider=provider
        )
        meta_path = Path(self.tmpdir) / "trajectories" / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        self.assertEqual(meta["command_dim"], 3)
        self.assertEqual(meta["command_names"], ["vx", "vy", "wz"])
        self.assertEqual(meta["command_type"], "velocity")

    def test_reset_clears_buffers(self):
        """reset() clears _rows, _step_idx, _prev_vel."""
        config = {
            "articulations": {"robot": {}},
            "scene": {"physics_dt": 0.005},
        }
        logger = Sim2MuJoCoDataLogger(self.tmpdir, config, joint_names=["j0", "j1"], control_dt=0.02)
        sim_state = _make_sim_state(2)
        joint_cmd = _make_joint_cmd(2)
        logger.record_step(sim_state, joint_cmd, torch.zeros(2), episode_id=0)
        self.assertEqual(len(logger._rows), 1)
        self.assertEqual(logger._step_idx, 1)

        logger.reset()
        self.assertEqual(len(logger._rows), 0)
        self.assertEqual(logger._step_idx, 0)
        self.assertIsNone(logger._prev_vel)

    def test_save_episode_and_reset_episode_boundary(self):
        """save_episode then reset produces separate episode files."""
        config = {
            "articulations": {"robot": {}},
            "scene": {"physics_dt": 0.005},
        }
        logger = Sim2MuJoCoDataLogger(self.tmpdir, config, joint_names=["j0"], control_dt=0.02)
        sim_state = _make_sim_state(1)
        joint_cmd = _make_joint_cmd(1)

        logger.record_step(sim_state, joint_cmd, torch.zeros(1), episode_id=0)
        logger.save_episode(0)
        logger.reset()

        logger.record_step(sim_state, joint_cmd, torch.zeros(1), episode_id=1)
        logger.save_episode(1)

        traj_dir = Path(self.tmpdir) / "trajectories"
        self.assertTrue((traj_dir / "episode_000.parquet").exists())
        self.assertTrue((traj_dir / "episode_001.parquet").exists())
        df0 = __import__("pandas").read_parquet(traj_dir / "episode_000.parquet")
        df1 = __import__("pandas").read_parquet(traj_dir / "episode_001.parquet")
        self.assertEqual(df0["episode_id"].iloc[0], 0)
        self.assertEqual(df1["episode_id"].iloc[0], 1)

    def test_schema_compat_env_id_is_success(self):
        """Output has env_id and is_success for report compatibility."""
        config = {
            "articulations": {"robot": {}},
            "scene": {"physics_dt": 0.005},
        }
        logger = Sim2MuJoCoDataLogger(self.tmpdir, config, joint_names=["j0"], control_dt=0.02)
        sim_state = _make_sim_state(1)
        joint_cmd = _make_joint_cmd(1)
        logger.record_step(sim_state, joint_cmd, torch.zeros(1), episode_id=0)
        logger.save_episode(0)

        df = __import__("pandas").read_parquet(Path(self.tmpdir) / "trajectories" / "episode_000.parquet")
        self.assertIn("env_id", df.columns)
        self.assertIn("is_success", df.columns)
        self.assertEqual(df["env_id"].iloc[0], 0)
        self.assertEqual(df["is_success"].iloc[0], 1.0)

    def test_joint_limits_mapped_to_sim_order(self):
        """Joint pos/vel limits are reordered from config (YAML) order to sim (MuJoCo) order."""
        # Config order: j0, j1, j2. Sim (MuJoCo) order: j2, j0, j1.
        config = {
            "articulations": {
                "robot": {
                    "joint_names": ["j0", "j1", "j2"],
                    "default_joint_pos_limits": [[0, 1], [10, 11], [20, 21]],
                    "soft_joint_vel_limits": [1.0, 2.0, 3.0],
                }
            },
            "scene": {"physics_dt": 0.005},
        }
        sim_joint_names = ["j2", "j0", "j1"]  # MuJoCo order differs from config
        _logger = Sim2MuJoCoDataLogger(self.tmpdir, config, joint_names=sim_joint_names, control_dt=0.02)

        meta_path = Path(self.tmpdir) / "trajectories" / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        # Limits must be in sim order: j2 -> [20,21], j0 -> [0,1], j1 -> [10,11]
        self.assertEqual(meta["joint_pos_limits"], [[20, 21], [0, 1], [10, 11]])
        self.assertEqual(meta["joint_vel_limits"], [3.0, 1.0, 2.0])

    def test_plotting_load_episode_compat(self):
        """Output can be loaded by agile.algorithms.evaluation.plotting.load_episode."""
        config = {
            "articulations": {"robot": {}},
            "scene": {"physics_dt": 0.005},
        }
        provider = _make_velocity_provider(3)
        logger = Sim2MuJoCoDataLogger(
            self.tmpdir, config, joint_names=["j0", "j1"], control_dt=0.02, command_provider=provider
        )
        sim_state = _make_sim_state(2)
        joint_cmd = _make_joint_cmd(2)
        commands = torch.tensor([0.0, 0.0, 0.0])  # 3D for locomotion_command
        logger.record_step(sim_state, joint_cmd, torch.zeros(2), commands=commands, episode_id=0)
        logger.save_episode(0)

        from agile.algorithms.evaluation.plotting import load_episode, load_metadata

        traj_dir = Path(self.tmpdir)
        df = load_episode(traj_dir, 0)
        meta = load_metadata(traj_dir)
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        self.assertIn("joint_pos_0", df.columns)
        self.assertIn("joint_vel_0", df.columns)
        self.assertIn("commands_0", df.columns)
        self.assertIn("joint_names", meta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
