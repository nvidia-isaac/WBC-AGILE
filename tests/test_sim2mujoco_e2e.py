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


"""End-to-end test for Sim2MuJoCo evaluation."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestSim2MuJoCo(unittest.TestCase):
    """Test case for running Sim2MuJoCo evaluation on different robots."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all tests."""
        # Get the project root directory
        cls.project_root = Path(__file__).parent.parent.absolute()

        # Path to the sim2mujoco eval script
        cls.eval_script = os.path.join(cls.project_root, "scripts", "sim2mujoco_eval.py")

        # Path to policy directory
        policy_dir = os.path.join(cls.project_root, "agile", "data", "policy")

        # Local assets directory (for T1)
        local_assets_dir = os.path.join(cls.project_root, "agile", "rl_env", "assets", "robot_menagerie")

        # Setup temporary directory for Unitree assets
        cls.temp_dir = tempfile.mkdtemp()
        print(f"\nCreated temporary directory for Unitree assets: {cls.temp_dir}")

        # Pull Unitree MuJoCo repo on the fly
        cls.unitree_assets_dir = cls._pull_unitree_repo(cls.temp_dir)

        # --- G1 Configuration ---
        cls.g1_ckpt = os.path.join(policy_dir, "velocity_height_g1", "unitree_g1_velocity_height_teacher.pt")
        cls.g1_cfg = os.path.join(policy_dir, "velocity_height_g1", "unitree_g1_velocity_height_teacher.yaml")

        # Use G1 from the pulled Unitree repo if available, otherwise fall back to local
        # Note: unitree_mujoco/unitree_robots/g1/g1_29dof.xml
        if cls.unitree_assets_dir:
            cls.g1_mjcf = os.path.join(cls.unitree_assets_dir, "unitree_robots", "g1", "g1_29dof.xml")
            print(f"Using G1 MJCF from Unitree repo: {cls.g1_mjcf}")
        else:
            # Fallback to local if pull failed
            cls.g1_mjcf = os.path.join(local_assets_dir, "unitree", "g1", "mujoco", "g1_29dof.xml")
            print(f"Using G1 MJCF from local assets: {cls.g1_mjcf}")

        # --- G1 Velocity History Configuration ---
        cls.g1_hist_ckpt = os.path.join(policy_dir, "velocity_g1", "unitree_g1_velocity_history.pt")
        cls.g1_hist_cfg = os.path.join(policy_dir, "velocity_g1", "unitree_g1_velocity_history.yaml")
        cls.g1_hist_mjcf = cls.g1_mjcf

        # Check if script exists
        if not os.path.exists(cls.eval_script):
            raise unittest.SkipTest(f"Sim2MuJoCo script not found at {cls.eval_script}")

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, "temp_dir") and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
            print(f"\nCleaned up temporary directory: {cls.temp_dir}")

    @staticmethod
    def _pull_unitree_repo(target_dir):
        """Clone the Unitree MuJoCo repo to the target directory."""
        repo_url = "https://github.com/unitreerobotics/unitree_mujoco.git"
        print(f"Cloning {repo_url} to {target_dir}...")

        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, target_dir],
                check=True,
                capture_output=True,
                timeout=120,  # 2 mins timeout for clone
            )
            return target_dir
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"Failed to clone Unitree repo: {e}")
            return None

    def test_g1_velocity_height(self):
        """Test G1 velocity height teacher policy in Sim2MuJoCo."""
        self._run_eval(self.g1_ckpt, self.g1_cfg, self.g1_mjcf, "G1 Velocity Height")

    def test_g1_velocity_history(self):
        """Test G1 velocity policy with history in Sim2MuJoCo."""
        self._run_eval(self.g1_hist_ckpt, self.g1_hist_cfg, self.g1_hist_mjcf, "G1 Velocity History")

    def _run_eval(self, ckpt, cfg, mjcf, name):
        """Helper to run the evaluation script."""
        # Skip if checkpoint doesn't exist
        if not os.path.exists(ckpt):
            print(f"\n⚠️ Skipping {name}: Checkpoint not found at {ckpt}")
            # We don't fail here to allow running tests even without all checkpoints locally
            return

        if not os.path.exists(mjcf):
            print(f"❌ MJCF file not found at {mjcf}")
            # If we expected it from the Unitree repo and it's missing, that's a fail
            self.fail(f"MJCF file not found at {mjcf}")

        print(f"\n{'=' * 60}")
        print(f"Testing Sim2MuJoCo: {name}")
        print("=" * 60)

        # Determine if we're in CI environment (use isaaclab.sh) or local (use python directly)
        isaaclab_path = os.environ.get("ISAACLAB_PATH")
        if isaaclab_path and os.path.exists(os.path.join(isaaclab_path, "isaaclab.sh")):
            # CI environment - use isaaclab.sh wrapper
            cmd = [
                os.path.join(isaaclab_path, "isaaclab.sh"),
                "-p",
                self.eval_script,
                "--checkpoint",
                ckpt,
                "--config",
                cfg,
                "--mjcf",
                mjcf,
                "--duration",
                "5.0",  # Run for 5 seconds
                "--no-viewer",  # Headless
                "--device",
                "cpu",  # Use CPU for consistent testing
                "--verbose",
            ]
        else:
            # Local environment - use python directly
            cmd = [
                "python",
                self.eval_script,
                "--checkpoint",
                ckpt,
                "--config",
                cfg,
                "--mjcf",
                mjcf,
                "--duration",
                "5.0",  # Run for 5 seconds
                "--no-viewer",  # Headless
                "--device",
                "cpu",  # Use CPU for consistent testing
                "--verbose",
            ]

        print(f"Running command: {' '.join(cmd)}")

        try:
            # Run the command with timeout
            result = subprocess.run(
                cmd,
                check=True,
                timeout=60,  # 1 minute timeout should be plenty for 5s sim
                capture_output=True,
                text=True,
                env=os.environ,  # Pass current environment
            )

            print(f"✅ {name} passed")

            # Verify output contains success message or steps
            if "Evaluation complete!" not in result.stdout:
                print("⚠️ Warning: 'Evaluation complete!' not found in stdout")

        except subprocess.CalledProcessError as e:
            print(f"❌ {name} failed with return code {e.returncode}")
            print("STDERR:")
            print(e.stderr[-2000:] if e.stderr else "No stderr")
            print("STDOUT:")
            print(e.stdout[-2000:] if e.stdout else "No stdout")
            self.fail(f"{name} failed execution")

        except subprocess.TimeoutExpired as e:
            print(f"❌ {name} timed out")
            print("Partial STDOUT:")
            print(e.stdout[-2000:] if e.stdout else "No output")
            self.fail(f"{name} timed out")


if __name__ == "__main__":
    unittest.main(verbosity=2)
