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
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.e2e

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "export_policy_leapp.py"
SIM2MUJOCO_SCRIPT = PROJECT_ROOT / "scripts" / "sim2mujoco_eval.py"
CKPT = PROJECT_ROOT / "agile" / "data" / "policy" / "velocity_g1" / "unitree_g1_velocity_history_state_dict.pt"
MJCF = PROJECT_ROOT / "external_assets" / "unitree_mujoco" / "unitree_robots" / "g1" / "scene_29dof.xml"
TASK = "Velocity-G1-History-v0"


def test_sim2mujoco_save_data_logs_commands(tmp_path: Path) -> None:
    if not CKPT.exists():
        pytest.skip("checkpoint unavailable")
    assert MJCF.is_file(), "E2E setup must run agile-download-assets"
    env = dict(os.environ)
    process_tmp = tmp_path / "tmp"
    process_tmp.mkdir()
    env["TMPDIR"] = str(process_tmp)
    completed = subprocess.run(
        [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--task",
            TASK,
            "--checkpoint",
            str(CKPT),
            "--export_save_path",
            str(tmp_path),
            "--disable_graph_visualization",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    bundle = tmp_path / TASK / f"{TASK}.yaml"
    if not bundle.is_file():
        candidates = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*.yaml"))
        pytest.fail(
            f"LEAPP bundle was not produced at {bundle}; YAML files under export root: {candidates}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(SIM2MUJOCO_SCRIPT),
            "--leapp-yaml",
            str(bundle),
            "--mjcf",
            str(MJCF),
            "--duration",
            "1.0",
            "--no-viewer",
            "--disable-keyboard",
            "--device",
            "cpu",
            "--no-real-time",
            "--save-data",
            "--output-dir",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    parquets = list((out / "trajectories").glob("episode_*.parquet"))
    assert parquets, "no parquet trajectories produced"
    df = pd.read_parquet(parquets[0])
    assert "commands_0" in df.columns
    assert "commands_2" in df.columns
    assert "commands_3" not in df.columns  # velocity-only policy logs x, y, yaw commands
