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


"""End-to-end test for scripts/export_IODescriptors.py.

Validates that the IO descriptor export produces valid YAML with all
expected fields for different task types (velocity, tracking, etc.).
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


class TestExportIODescriptors(unittest.TestCase):
    """Test case for IO descriptor export across task types."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).parent.parent.absolute()
        cls.export_script = cls.project_root / "scripts" / "export_IODescriptors.py"

        cls.isaaclab_path = os.environ.get("ISAACLAB_PATH")
        if not cls.isaaclab_path:
            raise unittest.SkipTest("ISAACLAB_PATH environment variable is not set")

        cls.isaaclab_script = os.path.join(cls.isaaclab_path, "isaaclab.sh")

        if not cls.export_script.exists():
            raise unittest.SkipTest(f"Export script not found at {cls.export_script}")
        if not os.path.exists(cls.isaaclab_script):
            raise unittest.SkipTest(f"IsaacLab script not found at {cls.isaaclab_script}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_export(self, task: str, output_dir: str) -> subprocess.CompletedProcess:
        """Run the export script for a given task and return the result."""
        cmd = [
            self.isaaclab_script,
            "-p",
            str(self.export_script),
            "--task",
            task,
            "--output_dir",
            output_dir,
            "--headless",
        ]

        env = dict(os.environ)
        env["OMNI_HEADLESS"] = "1"
        env["DISPLAY"] = ":1"

        print(f"\n{'=' * 60}")
        print(f"Exporting IO descriptors for: {task}")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 60)

        return subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)

    def _load_yaml(self, output_dir: str, task: str) -> dict:
        """Load the exported YAML and return it as a dict."""
        name = task.lower().replace("-", "_").replace(" ", "_")
        yaml_path = Path(output_dir) / f"{name}_IO_descriptors.yaml"
        self.assertTrue(yaml_path.exists(), f"Expected YAML not found: {yaml_path}")
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    def _assert_common_structure(self, data: dict, task: str):
        """Assert the YAML has the required top-level sections and articulation fields."""
        for section in ("observations", "actions", "articulations", "scene"):
            self.assertIn(section, data, f"Missing top-level section '{section}' for {task}")

        self.assertIn("robot", data["articulations"], f"Missing 'robot' articulation for {task}")
        robot = data["articulations"]["robot"]

        required_fields = [
            "joint_names",
            "default_joint_pos",
            "default_joint_vel",
            "default_joint_pos_limits",
            "default_joint_damping",
            "default_joint_stiffness",
            "default_joint_armature",
        ]
        for field in required_fields:
            self.assertIn(field, robot, f"Missing articulation field '{field}' for {task}")

        num_joints = len(robot["joint_names"])
        self.assertGreater(num_joints, 0, f"No joints found for {task}")
        self.assertEqual(len(robot["default_joint_pos"]), num_joints, f"default_joint_pos length mismatch for {task}")
        self.assertEqual(
            len(robot["default_joint_pos_limits"]), num_joints, f"default_joint_pos_limits length mismatch for {task}"
        )

        scene = data["scene"]
        for field in ("physics_dt", "dt", "decimation"):
            self.assertIn(field, scene, f"Missing scene field '{field}' for {task}")
        self.assertGreater(scene["decimation"], 0)

        obs = data["observations"]
        self.assertIn("policy", obs, f"Missing 'policy' observation group for {task}")
        self.assertGreater(len(obs["policy"]), 0, f"No observation terms for {task}")

        self.assertGreater(len(data["actions"]), 0, f"No action terms for {task}")

    def _assert_has_vel_limits(self, data: dict, task: str):
        """Assert soft_joint_vel_limits is present and has correct length."""
        robot = data["articulations"]["robot"]
        self.assertIn(
            "soft_joint_vel_limits",
            robot,
            f"Missing 'soft_joint_vel_limits' for {task}. Available keys: {list(robot.keys())}",
        )
        num_joints = len(robot["joint_names"])
        self.assertEqual(
            len(robot["soft_joint_vel_limits"]),
            num_joints,
            f"soft_joint_vel_limits length mismatch for {task}",
        )

    # ------------------------------------------------------------------
    # Test cases
    # ------------------------------------------------------------------

    def test_velocity_task(self):
        """Export IO descriptors for a velocity task and validate structure."""
        task = "Velocity-G1-History-v0"
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run_export(task, tmpdir)
            self.assertEqual(result.returncode, 0, f"Export failed:\n{result.stderr[-2000:]}")

            data = self._load_yaml(tmpdir, task)
            self._assert_common_structure(data, task)
            self._assert_has_vel_limits(data, task)

            self.assertNotIn("motion_tracking", data, "Velocity task should not have motion_tracking")
            print(f"✅ {task} export validated")

    def test_velocity_height_task(self):
        """Export IO descriptors for a velocity+height task and validate structure."""
        task = "Velocity-Height-G1-v0"
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run_export(task, tmpdir)
            self.assertEqual(result.returncode, 0, f"Export failed:\n{result.stderr[-2000:]}")

            data = self._load_yaml(tmpdir, task)
            self._assert_common_structure(data, task)
            self._assert_has_vel_limits(data, task)
            print(f"✅ {task} export validated")


if __name__ == "__main__":
    unittest.main(verbosity=2)
