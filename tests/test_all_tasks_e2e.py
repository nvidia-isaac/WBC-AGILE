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


"""End-to-end test for all registered tasks - ensures they can be trained."""

import os
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path

import torch


class TestAllTasks(unittest.TestCase):
    """Test case for running training on all registered tasks."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all tests."""
        # Get the project root directory
        cls.project_root = Path(__file__).parent.parent.absolute()

        # Path to the train script
        cls.train_script = os.path.join(cls.project_root, "scripts", "train.py")

        # Check if ISAACLAB_PATH is set
        cls.isaaclab_path = os.environ.get("ISAACLAB_PATH")
        if not cls.isaaclab_path:
            raise unittest.SkipTest("ISAACLAB_PATH environment variable is not set")

        # Path to the isaaclab.sh script
        cls.isaaclab_script = os.path.join(cls.isaaclab_path, "isaaclab.sh")

        # Check if the scripts exist
        if not os.path.exists(cls.train_script):
            raise unittest.SkipTest(f"Train script not found at {cls.train_script}")
        if not os.path.exists(cls.isaaclab_script):
            raise unittest.SkipTest(f"IsaacLab script not found at {cls.isaaclab_script}")

        # Check for GPU availability
        cls.gpu_available = False

        if torch.cuda.is_available():
            cls.gpu_available = True
        else:
            warnings.warn(
                "\n"
                + "=" * 60
                + "\nWARNING: CUDA not available - E2E tests will likely fail!\n"
                + "Tests will still run to identify specific failures.\n"
                + "=" * 60,
                RuntimeWarning,
                stacklevel=2,
            )

        # List of all tasks to test
        cls.tasks = cls.get_all_tasks()

    @classmethod
    def get_all_tasks(cls) -> list[str]:
        """Get all registered tasks from the codebase.

        ⚠️ IMPORTANT: When you add a new task to the codebase, add it here too!

        This ensures your task doesn't break in future updates.
        Tasks should match the gym.register() calls in your task's __init__.py file.
        """
        tasks = [
            # ====================================================================
            # LOCOMOTION TASKS
            # ====================================================================
            # G1 Robot - Basic Locomotion
            "Velocity-G1-History-v0",
            # G1 Robot - Locomotion with Height Map
            "Velocity-Height-G1-v0",
            # T1 Robot - Basic Locomotion
            "Velocity-T1-v0",
            # T1 Robot - Stand Up
            "StandUp-T1-v0",
            # ====================================================================
            # 🆕 ADD YOUR NEW TASKS HERE!
            # ====================================================================
        ]
        return tasks

    def test_task_training(self):
        """Test that each task can be trained for a few iterations."""
        # Number of iterations for testing (keep small for CI)
        # Need at least 5 iterations to ensure update() is called in distillation
        num_iterations = 5
        num_envs = 4  # Small number of environments for testing

        failed_tasks = []

        for task in self.tasks:
            with self.subTest(task=task):
                print(f"\n{'=' * 60}", flush=True)
                print(f"Testing task: {task}", flush=True)
                print("=" * 60, flush=True)

                # Create a temporary directory for this task's logs
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Command to run the train script with IsaacLab
                    cmd = [
                        self.isaaclab_script,
                        "-p",
                        self.train_script,
                        "--task",
                        task,
                        "--max_iterations",
                        str(num_iterations),
                        "--num_envs",
                        str(num_envs),
                        "--headless",
                        "--logger",
                        "wandb",
                        "--log_project_name",
                        "e2e-test",
                        "--run_name",
                        f"test_{task}",
                        # Hydra override to use temp directory for outputs
                        f"hydra.run.dir={temp_dir}",
                    ]

                    # Environment variables for command execution
                    env = dict(os.environ)
                    env["WANDB_MODE"] = "disabled"  # Disable wandb - mocked mode for testing
                    env["OMNI_HEADLESS"] = "1"
                    env["DISPLAY"] = ":1"

                    print(f"Running command: {' '.join(cmd)}", flush=True)

                    # Distillation tasks need more time to load teacher models
                    timeout = 180 if "Distillation" in task else 120

                    try:
                        # Run the command with timeout
                        result = subprocess.run(
                            cmd,
                            check=True,  # Will raise exception if process returns non-zero
                            timeout=timeout,  # 2-3 minutes timeout per task
                            capture_output=True,
                            text=True,
                            env=env,
                        )

                        print(f"✅ Task {task} passed", flush=True)

                        # Optionally print output for debugging
                        if os.environ.get("VERBOSE_E2E_TESTS") == "true":
                            print("STDOUT:")
                            print(result.stdout[-1000:])  # Last 1000 chars

                    except subprocess.CalledProcessError as e:
                        failed_tasks.append(task)
                        print(f"❌ Task {task} failed with return code {e.returncode}", flush=True)
                        print("STDERR:", flush=True)
                        print(e.stderr[-2000:] if e.stderr else "No stderr", flush=True)
                        # Don't fail immediately, continue testing other tasks

                    except subprocess.TimeoutExpired as e:
                        failed_tasks.append(task)
                        print(f"❌ Task {task} timed out", flush=True)
                        print("Partial output:", flush=True)
                        print(e.stdout[-2000:] if e.stdout else "No output", flush=True)

        # Report summary
        print(f"\n{'=' * 60}")
        print("Test Summary")
        print("=" * 60)

        # Show GPU status
        if not self.gpu_available:
            print("⚠️  WARNING: Tests ran WITHOUT GPU - failures expected!")
        else:
            print("✅ GPU available")

        print(f"\nTotal tasks tested: {len(self.tasks)}")
        print(f"Passed: {len(self.tasks) - len(failed_tasks)}")
        print(f"Failed: {len(failed_tasks)}")

        if failed_tasks:
            print("\nFailed tasks:")
            for task in failed_tasks:
                print(f"  - {task}")

            # Different failure message depending on GPU availability
            if not self.gpu_available:
                msg = f"{len(failed_tasks)} task(s) failed - GPU not available, failures expected"
                self.fail(msg)
            else:
                self.fail(f"{len(failed_tasks)} task(s) failed training test")
        else:
            print("\n✅ All tasks passed!")


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
