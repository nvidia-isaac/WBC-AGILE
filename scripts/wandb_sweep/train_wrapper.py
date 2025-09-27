#!/usr/bin/env python
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

"""
WandB Sweep Wrapper for Isaac Lab Training.

This script acts as a wrapper between WandB sweeps and Isaac Lab training scripts.
It processes sweep parameters and launches the training script with the appropriate arguments.

Key features:
- Handles scaled-dictionary parameters for complex robot configurations
- Supports WandB logging integration
- Passes through any command-line arguments to the training script
- Simple retry logic for Isaac Sim cold start failures (2 attempts with 10s delay)
"""

import argparse
import json
import os
import subprocess
import sys
import time

import wandb
import yaml


def main() -> None:
    """Process WandB sweep parameters and launch the training script."""
    # Parse command line arguments - pass through everything to the training script
    parser = argparse.ArgumentParser(description="WandB sweep wrapper for Isaac Lab training")
    args, remaining_args = parser.parse_known_args()

    # Check if --logger wandb is being used
    using_wandb_logger = "--logger" in sys.argv and "wandb" in sys.argv

    # Initialize WandB for sweep coordination
    run = wandb.init()
    if run is None:
        print("Failed to initialize WandB", file=sys.stderr)
        sys.exit(1)

    try:
        cfg = dict(run.config)

        # Build the training command
        train_script = os.path.join(os.path.dirname(__file__), "..", "train.py")
        cmd = [sys.executable, train_script] + sys.argv[1:]

        # If using wandb logger, add the project name to match the sweep
        if using_wandb_logger:
            cmd.extend(["--log_project_name", run.project or "default-project"])

        # Process sweep parameters
        processed_keys = set()
        for key, value in cfg.items():
            if key in processed_keys or key.startswith("_"):
                continue

            # Handle scaled-dictionary parameters (pattern: param, param_cli_path, param_base_dict)
            cli_path_key = f"{key}_cli_path"
            base_dict_key = f"{key}_base_dict"
            type_key = f"{key}_type"

            if cli_path_key in cfg and base_dict_key in cfg:
                scale = float(value)
                cli_path = cfg[cli_path_key]
                base_dict_raw = cfg[base_dict_key]

                # Parse the base dictionary (JSON or YAML format)
                try:
                    base_dict: dict[str, float] = json.loads(base_dict_raw)
                except (json.JSONDecodeError, TypeError):
                    base_dict = yaml.safe_load(base_dict_raw)

                # Scale values and format for Hydra override
                scaled_dict = {}
                for k, v in base_dict.items():
                    if isinstance(v, list | tuple):
                        # Handle tuples/lists by scaling each element
                        scaled_tuple = tuple(float(x) * scale for x in v)
                        scaled_dict[k] = scaled_tuple
                    else:
                        # Handle regular floats
                        scaled_dict[k] = float(v) * scale

                        # Format for Hydra override
                inner_parts = []
                for k, v in scaled_dict.items():
                    if isinstance(v, tuple):
                        # Format tuple as [val1, val2, ...] for Hydra compatibility
                        list_str = "[" + ",".join(f"{x:.6f}" for x in v) + "]"
                        inner_parts.append(f"{k}:{list_str}")
                    else:
                        # Format regular float
                        inner_parts.append(f"{k}:{v:.6f}")

                inner = ",".join(inner_parts)
                override = f"{cli_path}={{{inner}}}"
                cmd.append(override)

                processed_keys.update([key, cli_path_key, base_dict_key])
                continue

            # Handle regular Hydra overrides
            if not key.endswith("_cli_path") and not key.endswith("_base_dict") and not key.endswith("_type"):
                # Check if this parameter has explicit type information
                if type_key in cfg and cfg[type_key] == "float":
                    # Force float formatting
                    cmd.append(f"{key}={float(value):.1f}")
                    processed_keys.add(type_key)
                elif isinstance(value, float):
                    # Regular float handling
                    cmd.append(f"{key}={value:.1f}")
                else:
                    cmd.append(f"{key}={value}")

        # Log the constructed command for debugging
        if wandb.run:
            wandb.run.config.update({"_constructed_command": " ".join(cmd)}, allow_val_change=True)

        print(f"[WRAPPER] Executing: {' '.join(cmd)}")

        # If using wandb logger, finish the wrapper's run to prevent conflicts
        if using_wandb_logger and wandb.run:
            wandb.finish()

        # Execute the training script with simple retry for Isaac Sim cold start issues
        max_retries = 2
        retry_delay = 10  # seconds between retries

        for attempt in range(max_retries):
            if attempt > 0:
                print(f"[WRAPPER] Retry attempt {attempt + 1}/{max_retries}")

            # Simple subprocess call - let it run normally
            exit_code = subprocess.call(cmd)

            # If successful, we're done
            if exit_code == 0:
                break

            # If failed and not last attempt, retry
            if attempt < max_retries - 1:
                print(f"[WRAPPER] Process exited with code {exit_code}.")
                print(f"[WRAPPER] Retrying in {retry_delay} seconds...")
                print("[WRAPPER] (Isaac Sim sometimes has initialization issues in containers)")
                time.sleep(retry_delay)
            else:
                print(f"[WRAPPER] Process failed with exit code {exit_code} after {max_retries} attempts.")

        # Finish wandb if we haven't already
        if not using_wandb_logger and wandb.run:
            wandb.finish(exit_code=exit_code)

        sys.exit(exit_code)

    except Exception as e:
        print(f"Wrapper error: {e}", file=sys.stderr)
        if wandb.run:
            wandb.finish(exit_code=1)
        sys.exit(1)


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def _build_scaled_dict_arg_from_path(cli_path: str, base: dict[str, float], scale: float) -> str:
    """Create the CLI override string for a scaled-dictionary parameter.

    Parameters
    ----------
    cli_path : str
        The full dot-notation path understood by the underlying Hydra/CLI, e.g.
        ``env.scene.robot.actuators.legs.stiffness``.
    base : dict[str, float]
        Baseline mapping from joint-regex to value. Values can be floats or tuples.
    scale : float
        Multiplicative factor applied to every value in *base*.
    """

    scaled = {}
    for k, v in base.items():
        if isinstance(v, list | tuple):
            # Handle tuples/lists by scaling each element
            scaled[k] = tuple(round(float(x) * scale, 6) for x in v)
        else:
            # Handle regular floats
            scaled[k] = round(float(v) * scale, 6)

        # Format for Hydra override
    inner_parts = []
    for k, v in scaled.items():
        if isinstance(v, tuple):
            # Format tuple as [val1, val2, ...] for Hydra compatibility
            list_str = "[" + ",".join(f"{x}" for x in v) + "]"
            inner_parts.append(f"{k}:{list_str}")
        else:
            # Format regular float
            inner_parts.append(f"{k}:{v}")

    inner = ",".join(inner_parts)
    return f"{cli_path}={{" + inner + "}}"
