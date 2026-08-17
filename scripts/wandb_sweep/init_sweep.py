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
WandB Sweep Initialization Script.

This script creates a new WandB sweep based on the configuration in sweep.yaml
and stores the sweep ID in sweep_ids.json for later use by agents.

Usage:
    python scripts/wandb_sweep/init_sweep.py --project_name <project_name>


"""

import argparse
import copy
import json
import os

import wandb
import yaml


def load_sweep_config(file_path: str | os.PathLike) -> dict:
    """Load and return the YAML sweep configuration as a dictionary."""
    with open(file_path) as file:
        return yaml.safe_load(file)  # type: ignore[no-any-return]


def transform_sweep_config(config: dict) -> dict:
    """Transform a sweep config with nested parameters into a flat one for WandB."""
    transformed_config = copy.deepcopy(config)
    flat_params = {}

    for param_name, param_config in config.get("parameters", {}).items():
        if isinstance(param_config, dict):
            # This is a potential scaled-dictionary parameter
            cli_path_key = f"{param_name}_cli_path"
            base_dict_key = f"{param_name}_base_dict"
            type_key = f"{param_name}_type"

            # Check for nested keys
            if cli_path_key in param_config and base_dict_key in param_config:
                # Extract the nested parts
                cli_path_config = param_config.pop(cli_path_key)
                base_dict_config = param_config.pop(base_dict_key)

                # Add the main parameter and the extracted parts to the flat dict
                flat_params[param_name] = param_config
                flat_params[cli_path_key] = cli_path_config
                flat_params[base_dict_key] = base_dict_config
            else:
                # Check if this parameter has a type specification
                if "type" in param_config and param_config["type"] == "float":
                    # Extract type information and store it separately
                    param_config_copy = param_config.copy()
                    param_config_copy.pop("type")  # Remove type from the config sent to WandB

                    # Add the parameter without type
                    flat_params[param_name] = param_config_copy
                    # Store type information separately
                    flat_params[type_key] = {"value": "float"}
                else:
                    # Regular parameter
                    flat_params[param_name] = param_config
        else:
            flat_params[param_name] = param_config

    transformed_config["parameters"] = flat_params
    return transformed_config


def main() -> None:
    """Initialize a WandB sweep and persist its ID to *sweep_ids.json*."""
    parser = argparse.ArgumentParser(description="Initialize a WandB sweep.")
    parser.add_argument("--project_name", type=str, required=True, help="Name of the WandB project.")
    parser.add_argument(
        "--entity_name",
        type=str,
        default=None,
        help="Name of the WandB entity/team (can also be provided in sweep.yaml).",
    )
    args = parser.parse_args()

    # Determine paths relative to this script's directory
    sweep_dir = os.path.dirname(os.path.abspath(__file__))
    sweep_config_path = os.path.join(sweep_dir, "sweep.yaml")
    sweep_ids_path = os.path.join(sweep_dir, "sweep_ids.json")

    # Load the sweep configuration YAML
    sweep_config = load_sweep_config(sweep_config_path)

    # Transform the config to be WandB-compatible
    transformed_sweep_config = transform_sweep_config(sweep_config)

    # Derive entity: CLI flag takes precedence over YAML
    entity = args.entity_name or sweep_config.get("entity")

    if entity is None:
        raise ValueError("Entity name must be provided via --entity_name or top-level 'entity' in sweep.yaml.")

    # Register the sweep with WandB; this returns a unique sweep ID
    sweep_id = wandb.sweep(transformed_sweep_config, project=args.project_name, entity=entity)
    print(f"Sweep '{args.project_name}' initialized with ID: {sweep_id}")

    # Persist the sweep ID so that other machines can join the sweep later on
    if os.path.exists(sweep_ids_path):
        with open(sweep_ids_path, encoding="utf-8") as file:
            sweep_ids: dict[str, str] = json.load(file)
    else:
        sweep_ids = {}

    sweep_ids[args.project_name] = sweep_id

    with open(sweep_ids_path, "w", encoding="utf-8") as file:
        json.dump(sweep_ids, file, indent=4)


if __name__ == "__main__":
    main()
