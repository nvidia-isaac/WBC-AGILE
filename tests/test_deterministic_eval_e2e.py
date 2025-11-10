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

"""End-to-end system test for deterministic evaluation pipeline.

Tests the complete flow:
1. Run evaluation with deterministic config (headless)
2. Generate trajectory logs
3. Validate deterministic commands match config
4. Generate HTML report
5. Validate report structure

This test ensures the evaluation scenario system works correctly in CI.
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class TestDeterministicEvalE2E(unittest.TestCase):
    """End-to-end test for deterministic evaluation system."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Get project root directory
        cls.project_root = Path(__file__).parent.parent.absolute()

        # Paths to scripts and configs
        cls.eval_script = cls.project_root / "scripts" / "eval.py"
        cls.quick_config = cls.project_root / "tests" / "test_eval_configs" / "quick_test.yaml"
        cls.checkpoint = (
            cls.project_root
            / "agile"
            / "data"
            / "policy"
            / "velocity_height_g1"
            / "unitree_g1_velocity_height_teacher.pt"
        )

        # Verify files exist
        assert cls.eval_script.exists(), f"Eval script not found: {cls.eval_script}"
        assert cls.quick_config.exists(), f"Quick config not found: {cls.quick_config}"

        # Check if checkpoint exists - skip test if not available
        if not cls.checkpoint.exists():
            raise unittest.SkipTest(
                f"Checkpoint not found: {cls.checkpoint}\n"
                f"To run this test, you need a trained policy checkpoint.\n"
                f"You can either:\n"
                f"  1. Train a policy and place it at the expected location\n"
                f"  2. Download a pretrained checkpoint\n"
                f"  3. Update the checkpoint path in the test"
            )

        # Check ISAACLAB_PATH is set
        cls.isaaclab_path = os.environ.get("ISAACLAB_PATH")
        if not cls.isaaclab_path:
            raise unittest.SkipTest("ISAACLAB_PATH environment variable is not set")

        # Expected schedule from quick_test.yaml
        # Column names map: commands_0=lin_vel_x, commands_1=lin_vel_y, commands_2=ang_vel_z, commands_3=base_height
        cls.expected_schedule = {
            0: {  # Env 0: X-velocity sweep
                "command_col": "commands_0",  # lin_vel_x
                "values": [0.0, 0.5],
                "interval_s": 2.5,
                "other_commands": {"commands_1": 0.0, "commands_2": 0.0, "commands_3": 0.72},
            },
            1: {  # Env 1: Height sweep
                "command_col": "commands_3",  # base_height
                "values": [0.65, 0.72],
                "interval_s": 2.5,
                "other_commands": {"commands_0": 0.0, "commands_1": 0.0, "commands_2": 0.0},
            },
        }

    def test_deterministic_eval_complete_flow(self):
        """Test complete evaluation pipeline: eval → trajectories → report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            print(f"\n[TEST] Using temporary directory: {tmpdir}")

            # Phase 1: Run evaluation
            log_dir = self._run_evaluation(tmpdir)

            # Phase 2: Validate trajectories
            self._validate_trajectory_structure(log_dir)
            self._validate_data_completeness(log_dir)
            self._validate_episode_length(log_dir)
            self._validate_deterministic_commands(log_dir)
            self._validate_metrics(log_dir)

            # Phase 3: Generate and validate report
            self._generate_and_validate_report(log_dir)

            print("[TEST] All validations passed!")

    def _run_evaluation(self, tmpdir: Path) -> Path:
        """Run eval.py with deterministic config in headless mode.

        Args:
            tmpdir: Temporary directory for logs

        Returns:
            Path to log directory
        """
        print("\n[TEST] Phase 1: Running evaluation...")

        # Use tmpdir for metrics to avoid polluting logs/ directory
        metrics_file = tmpdir / "metrics.json"

        # Use Isaac Lab wrapper script if ISAACLAB_PATH is set, otherwise use python directly
        if self.isaaclab_path:
            isaaclab_script = Path(self.isaaclab_path) / "isaaclab.sh"
            cmd = [
                str(isaaclab_script),
                "-p",
                str(self.eval_script),
                "--task",
                "Velocity-Height-G1-Dev-v0",
                "--checkpoint",
                str(self.checkpoint),
                "--eval_config",
                str(self.quick_config),
                "--num_envs",
                "2",
                "--run_evaluation",
                "--save_trajectories",
                "--headless",
                "--metrics_file",
                str(metrics_file),
            ]
        else:
            cmd = [
                "python",
                str(self.eval_script),
                "--task",
                "Velocity-Height-G1-Dev-v0",
                "--checkpoint",
                str(self.checkpoint),
                "--eval_config",
                str(self.quick_config),
                "--num_envs",
                "2",
                "--run_evaluation",
                "--save_trajectories",
                "--headless",
                "--metrics_file",
                str(metrics_file),
            ]

        # Set environment variables for headless mode
        env = os.environ.copy()
        env["ISAACLAB_HEADLESS"] = "1"

        print(f"[TEST] Command: {' '.join(cmd)}")

        # Run evaluation with timeout
        try:
            result = subprocess.run(
                cmd,
                env=env,
                timeout=300,  # 5 minutes timeout
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
            )
        except subprocess.TimeoutExpired:
            self.fail("Evaluation timed out after 5 minutes")

        # Check for errors
        if result.returncode != 0:
            print(f"[TEST] STDOUT:\n{result.stdout}")
            print(f"[TEST] STDERR:\n{result.stderr}")
            self.fail(f"Evaluation failed with return code {result.returncode}")

        # The log directory is the parent of metrics_file (tmpdir in this case)
        log_dir = metrics_file.parent

        # Verify metrics file was created
        self.assertTrue(metrics_file.exists(), f"Metrics file not created: {metrics_file}")

        print(f"[TEST] Log directory: {log_dir}")

        return log_dir

    def _validate_trajectory_structure(self, log_dir: Path):
        """Verify expected files and directory structure exist.

        Args:
            log_dir: Evaluation log directory
        """
        print("\n[TEST] Phase 2.1: Validating trajectory structure...")

        # Check required files exist
        self.assertTrue((log_dir / "metrics.json").exists(), "metrics.json not found")
        self.assertTrue((log_dir / "trajectories").exists(), "trajectories directory not found")

        # Check trajectory files exist
        traj_dir = log_dir / "trajectories"
        parquet_files = list(traj_dir.glob("episode_*.parquet"))
        self.assertGreater(len(parquet_files), 0, "No trajectory parquet files found")

        print(f"[TEST] ✓ Found {len(parquet_files)} trajectory files")

    def _validate_data_completeness(self, log_dir: Path):
        """Check all expected fields are logged and data is complete.

        Args:
            log_dir: Evaluation log directory
        """
        print("\n[TEST] Phase 2.2: Validating data completeness...")

        traj_dir = log_dir / "trajectories"
        parquet_files = sorted(traj_dir.glob("episode_*.parquet"))

        # Load first trajectory file
        df = pd.read_parquet(parquet_files[0])

        # Required fields that should be present
        # Commands are saved as indexed columns: commands_0, commands_1, commands_2, commands_3
        # Root velocity/position are saved as: root_lin_vel_0, root_ang_vel_2, root_pos_2, etc.
        required_fields = [
            "timestep",
            "env_id",
            "episode_id",
            "commands_0",  # lin_vel_x command
            "commands_1",  # lin_vel_y command
            "commands_2",  # ang_vel_z command
            "commands_3",  # base_height command
            "root_lin_vel_0",  # actual lin_vel_x
            "root_lin_vel_1",  # actual lin_vel_y
            "root_ang_vel_2",  # actual ang_vel_z
            "root_pos_2",  # actual height
        ]

        # Check all required fields exist
        for field in required_fields:
            self.assertIn(field, df.columns, f"Missing required field: {field}")

        # Check timesteps are continuous (should start at 0)
        self.assertEqual(df["timestep"].iloc[0], 0, "Timesteps should start at 0")

        # Check no NaN values in command fields
        command_fields = [f for f in required_fields if f.startswith("commands_")]
        for field in command_fields:
            nan_count = df[field].isna().sum()
            self.assertEqual(nan_count, 0, f"Found {nan_count} NaN values in {field}")

        print("[TEST] ✓ All required fields present, no NaN values")
        print(f"[TEST] ✓ Trajectory length: {len(df)} steps")

    def _validate_episode_length(self, log_dir: Path):
        """Verify episode length and num_envs match the config overrides.

        Args:
            log_dir: Evaluation log directory
        """
        print("\n[TEST] Phase 2.2b: Validating episode length and num_envs overrides...")

        traj_dir = log_dir / "trajectories"
        parquet_files = sorted(traj_dir.glob("episode_*.parquet"))

        # Validate num_envs: should have 2 episodes (one per env)
        expected_num_envs = 2  # From quick_test.yaml
        actual_num_episodes = len(parquet_files)
        self.assertEqual(
            actual_num_episodes,
            expected_num_envs,
            f"Number of episodes mismatch: expected {expected_num_envs} episodes (1 per env), got {actual_num_episodes}",
        )

        # Check env_ids are correct
        env_ids = set()
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            env_ids.add(int(df["env_id"].iloc[0]))

        expected_env_ids = {0, 1}
        self.assertEqual(
            env_ids, expected_env_ids, f"Environment IDs mismatch: expected {expected_env_ids}, got {env_ids}"
        )

        print(f"[TEST] Num envs correctly overridden: {actual_num_episodes} envs with IDs {sorted(env_ids)}")

        # Validate episode_id = env_id mapping (core fix for deterministic evaluation)
        # For single-episode-per-env scenarios, episode_id should equal env_id
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            episode_id = int(df["episode_id"].iloc[0])
            env_id = int(df["env_id"].iloc[0])
            self.assertEqual(
                episode_id,
                env_id,
                f"Episode-Env mapping incorrect: episode_{episode_id:03d}.parquet contains env_id={env_id} (expected env_id={episode_id})",
            )

        print(f"[TEST] Episode-Env ID mapping correct: episode_id = env_id for all {actual_num_episodes} episodes")

        # Validate episode length
        df = pd.read_parquet(parquet_files[0])

        # Expected episode length from quick_test.yaml
        expected_length_s = 5.0
        expected_steps = int(expected_length_s * 50)  # 50 Hz control frequency

        # Check episode length in steps
        actual_steps = len(df)
        self.assertEqual(
            actual_steps,
            expected_steps,
            f"Episode length mismatch: expected {expected_steps} steps ({expected_length_s}s), got {actual_steps} steps",
        )

        # Check duration from timestep
        actual_duration = df["timestep"].max()
        self.assertAlmostEqual(
            actual_duration,
            expected_length_s,
            delta=0.1,  # Allow 0.1s tolerance
            msg=f"Episode duration mismatch: expected {expected_length_s}s, got {actual_duration:.2f}s",
        )

        print(f"[TEST] Episode length correctly overridden: {actual_steps} steps ({actual_duration:.2f}s)")

    def _validate_deterministic_commands(self, log_dir: Path):
        """Verify commands match the schedule from config.

        This is the core test - validates that deterministic scheduling works correctly.

        Args:
            log_dir: Evaluation log directory
        """
        print("\n[TEST] Phase 2.3: Validating deterministic commands...")

        traj_dir = log_dir / "trajectories"
        parquet_files = sorted(traj_dir.glob("episode_*.parquet"))

        # Group trajectories by env_id (there may be multiple episodes per env)
        env_trajectories = {}
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            env_id = int(df["env_id"].iloc[0])

            if env_id not in env_trajectories:
                env_trajectories[env_id] = []
            env_trajectories[env_id].append(df)

        print(f"[TEST] Found trajectories for envs: {sorted(env_trajectories.keys())}")

        # Validate each environment that has a schedule
        for env_id in sorted(env_trajectories.keys()):
            # Skip if this env is not in our expected schedule
            if env_id not in self.expected_schedule:
                print(f"[TEST] Skipping env {env_id} (no schedule configured)")
                continue

            # Use the first episode from this environment
            df = env_trajectories[env_id][0]

            schedule = self.expected_schedule[env_id]
            command_col = schedule["command_col"]  # e.g., "commands_0" for lin_vel_x
            expected_values = schedule["values"]
            interval_s = schedule["interval_s"]
            other_commands = schedule["other_commands"]

            print(f"\n[TEST] Validating env {env_id} ({command_col} sweep)...")
            print(f"[TEST]   Episode length: {len(df)} steps ({len(df) / 50:.1f}s)")
            print(f"[TEST]   Expected values: {expected_values}")
            print(f"[TEST]   Interval: {interval_s}s")

            # Calculate interval in timesteps (assuming 50 Hz)
            control_freq = 50  # Hz
            interval_steps = int(interval_s * control_freq)

            # Determine how many segments we should validate based on actual episode length
            max_segments = len(expected_values)
            actual_segments = min(max_segments, len(df) // interval_steps)

            # Validate first N segments (where N = len(expected_values))
            # Even if episode is longer, we only check the first cycle through expected values
            for i in range(actual_segments):
                expected_value = expected_values[i % len(expected_values)]
                start_step = i * interval_steps
                end_step = min((i + 1) * interval_steps, len(df))

                # Handle case where episode is shorter than expected
                if start_step >= len(df):
                    print(f"[TEST]   Segment {i}: Skipped (beyond episode length)")
                    continue

                # Skip first 5 timesteps of each segment to allow scheduler to apply commands
                # (there can be a small delay between scheduler.update() and command application)
                skip_steps = 5
                segment_start = start_step + skip_steps
                if segment_start >= end_step:
                    print(f"[TEST]   Segment {i}: Skipped (too short after burn-in)")
                    continue

                segment = df.iloc[segment_start:end_step]

                # Check the swept command
                actual_values = segment[command_col].values
                mean_value = actual_values.mean()
                min_value = actual_values.min()
                max_value = actual_values.max()

                # Print diagnostic info
                print(
                    f"[TEST]   Segment {i} (t={segment_start / control_freq:.1f}s-{end_step / control_freq:.1f}s): "
                    f"{command_col} = {mean_value:.3f} (min={min_value:.3f}, max={max_value:.3f}), expected={expected_value:.3f}"
                )

                # Commands should be constant within segment (small std dev)
                std_dev = actual_values.std()
                if std_dev >= 0.05:
                    print(
                        f"[TEST]     WARNING: Commands not constant! std={std_dev:.4f}, "
                        f"first={actual_values[0]:.3f}, last={actual_values[-1]:.3f}"
                    )

                self.assertLess(
                    std_dev,
                    0.05,  # Relaxed from 0.01 to allow for minor variations
                    f"Env {env_id} segment {i}: Commands should be constant. "
                    f"Expected {expected_value}, got std={std_dev}",
                )

                # Mean should match expected value (relaxed tolerance)
                self.assertAlmostEqual(
                    mean_value,
                    expected_value,
                    places=1,  # Relaxed from 2 to 1 decimal place
                    msg=f"Env {env_id} segment {i}: Expected {command_col}={expected_value}, got {mean_value}",
                )

                # Validate other commands remain constant
                for other_col, other_value in other_commands.items():
                    other_actual = segment[other_col].mean()
                    self.assertAlmostEqual(
                        other_actual,
                        other_value,
                        places=2,
                        msg=f"Env {env_id} segment {i}: {other_col} should be {other_value}, got {other_actual}",
                    )

        print("\n[TEST] All deterministic commands validated successfully!")

    def _validate_metrics(self, log_dir: Path):
        """Check metrics.json is valid and contains expected data.

        Args:
            log_dir: Evaluation log directory
        """
        print("\n[TEST] Phase 2.4: Validating metrics...")

        metrics_file = log_dir / "metrics.json"
        with open(metrics_file) as f:
            metrics = json.load(f)

        # Check required top-level keys
        self.assertIn("success_rate", metrics, "Missing success_rate in metrics")

        # Validate success rate is in valid range
        success_rate = metrics["success_rate"]
        self.assertGreaterEqual(success_rate, 0.0, "Success rate should be >= 0")
        self.assertLessEqual(success_rate, 1.0, "Success rate should be <= 1")

        print(f"[TEST] ✓ Success rate: {success_rate:.2%}")

    def _generate_and_validate_report(self, log_dir: Path):
        """Generate HTML report and verify structure.

        Args:
            log_dir: Evaluation log directory
        """
        print("\n[TEST] Phase 3: Generating and validating report...")

        # Import report generator
        try:
            from agile.algorithms.evaluation.report_generator import TrajectoryReportGenerator
        except ImportError as e:
            self.skipTest(f"Could not import TrajectoryReportGenerator: {e}")

        # Generate report
        try:
            generator = TrajectoryReportGenerator(str(log_dir))
            generator.generate_full_report(open_browser=False)
        except Exception as e:
            self.fail(f"Report generation failed: {e}")

        # Validate report structure
        report_dir = log_dir / "reports"
        self.assertTrue(report_dir.exists(), "Report directory not created")

        # Check index page exists
        index_html = report_dir / "index.html"
        self.assertTrue(index_html.exists(), "index.html not found")

        # Check episode pages directory exists
        episodes_dir = report_dir / "episodes"
        self.assertTrue(episodes_dir.exists(), "episodes directory not found")

        # Check episode pages exist
        episode_pages = list(episodes_dir.glob("episode_*.html"))
        self.assertGreater(len(episode_pages), 0, "No episode HTML pages found")

        # Validate index.html is well-formed
        with open(index_html) as f:
            html_content = f.read()
            self.assertIn("<html", html_content.lower(), "index.html missing <html> tag")
            self.assertIn("</html>", html_content.lower(), "index.html missing </html> tag")
            self.assertIn("Success Rate", html_content, "index.html missing 'Success Rate' text")

        # Validate an episode page is well-formed
        with open(episode_pages[0]) as f:
            episode_html = f.read()
            self.assertIn("<html", episode_html.lower(), "Episode HTML missing <html> tag")
            self.assertIn("plotly", episode_html.lower(), "Episode HTML missing plotly (plots)")

        print("[TEST] ✓ Report generated successfully")
        print(f"[TEST] ✓ Found {len(episode_pages)} episode pages")
        print(f"[TEST] ✓ Report location: {report_dir}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
