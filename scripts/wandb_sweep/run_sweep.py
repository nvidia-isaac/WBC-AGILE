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
WandB Sweep Agent Runner.

This script starts WandB sweep agents that pull jobs from an existing sweep
and execute training runs with the sampled hyperparameters.

Usage:
    python scripts/wandb_sweep/run_sweep.py --project_name <project_name> --agent_count 15
"""

import argparse
import json
import os

import wandb
import yaml


def load_sweep_id(project_name: str, sweep_ids_path: str | os.PathLike) -> str:
    """Return the sweep ID corresponding to *project_name* stored in *sweep_ids_path*.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    ValueError
        If *project_name* is not found in the JSON mapping.
    """
    if not os.path.exists(sweep_ids_path):
        raise FileNotFoundError(f"Sweep IDs file not found at {sweep_ids_path}.")

    with open(sweep_ids_path, encoding="utf-8") as file:
        sweep_ids: dict[str, str] = json.load(file)

    try:
        return sweep_ids[project_name]
    except KeyError as exc:
        raise ValueError(f"No sweep ID found for project '{project_name}' in {sweep_ids_path}.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Start WandB sweep agents using an existing sweep ID.")
    parser.add_argument("--project_name", type=str, required=True, help="Name of the WandB project.")
    parser.add_argument(
        "--entity_name",
        type=str,
        default=None,
        help="Name of the WandB entity/team (can also be provided in sweep.yaml).",
    )
    parser.add_argument(
        "--agent_count",
        type=int,
        default=1,
        help="Number of sweep trial executions before agent exits.",
    )
    args = parser.parse_args()

    # Paths
    sweep_dir = os.path.dirname(os.path.abspath(__file__))
    sweep_ids_path = os.path.join(sweep_dir, "sweep_ids.json")
    sweep_yaml_path = os.path.join(sweep_dir, "sweep.yaml")

    # Retrieve sweep ID
    sweep_id = load_sweep_id(args.project_name, sweep_ids_path)

    # Determine entity name
    if args.entity_name is not None:
        entity = args.entity_name
    else:
        # fallback to YAML top-level 'entity'
        if not os.path.exists(sweep_yaml_path):
            raise FileNotFoundError("sweep.yaml not found to infer entity name.")
        with open(sweep_yaml_path, encoding="utf-8") as f_yaml:
            sweep_cfg = yaml.safe_load(f_yaml)  # type: ignore[no-any-return]
        entity = sweep_cfg.get("entity")
        if entity is None:
            raise ValueError("Entity name must be provided via --entity_name or 'entity' in sweep.yaml.")

    print(f"Starting sweep agents for project '{args.project_name}' (sweep ID: {sweep_id})")

    # Launch agent(s)
    wandb.agent(
        sweep_id,
        entity=entity,
        project=args.project_name,
        count=args.agent_count,
    )


if __name__ == "__main__":
    main()
