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

"""End-to-end smoke tests for AGILE task training, evaluation, and play entrypoints."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import pytest
import torch

pytestmark = pytest.mark.e2e

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval.py"
PLAY_SCRIPT = PROJECT_ROOT / "scripts" / "play.py"
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "export_policy_leapp.py"
SIM2MUJOCO_SCRIPT = PROJECT_ROOT / "scripts" / "sim2mujoco_eval.py"
POLICY_DIR = PROJECT_ROOT / "agile" / "data" / "policy"

# Use the scene MJCF documented by the public repository. It includes the robot and adds the ground
# plane, lights, and skybox; without the floor the robot has no ground contact in Sim2MuJoCo.
G1_MJCF = PROJECT_ROOT / "external_assets" / "unitree_mujoco" / "unitree_robots" / "g1" / "scene_29dof.xml"


@dataclass(frozen=True)
class E2ETask:
    """One e2e-covered task. checkpoint enables eval+sim2sim; mjcf additionally enables sim2sim."""

    task_id: str
    checkpoint: Path | None = None  # RSL-RL state-dict checkpoint
    mjcf: Path | None = None  # external robot MJCF


# Single source of truth. Add a task here to cover it; attach a checkpoint (+mjcf) to extend
# coverage from training-only to eval and sim2sim.
E2E_TASKS: list[E2ETask] = [
    # Training-only for now (checkpoints to be added later).
    E2ETask("Velocity-G1-History-v0"),
    E2ETask("Velocity-Height-G1-History-v0"),
    E2ETask("Velocity-Height-G1-Student-Recurrent-v0"),
    E2ETask("Velocity-Height-G1-Student-History-v0"),
    E2ETask("Velocity-G1-Teacher-v0"),
    E2ETask("Velocity-Height-G1-Teacher-v0"),
    E2ETask("Velocity-T1-v0"),
    E2ETask("StandUp-T1-v0"),
    E2ETask("HeightTracking-G1-v0"),
    E2ETask("PickPlace-G1-v0"),
]

# Registered trainable tasks intentionally NOT in the matrix, with reasons.
E2E_SKIP: dict[str, str] = {
    "Debug-G1-v0": "debug environment",
    "Debug-G1-Object-v0": "debug environment",
    "Debug-T1-v0": "debug environment",
    "MotionTracking-G1-v0": "requires a caller-provided motion dataset",
    "PickPlace-G1-Debug-v0": "debug environment",
    "PickPlace-G1-Record-v0": "recording environment",
    "PickPlace-G1-GR00T-Inference-v0": "manual inference environment",
}

_EVAL = [t for t in E2E_TASKS if t.checkpoint and t.checkpoint.exists()]
_SIM2SIM = [t for t in E2E_TASKS if t.checkpoint and t.checkpoint.exists() and t.mjcf and t.mjcf.exists()]


def _warn_if_cuda_is_unavailable() -> None:
    if torch.cuda.is_available():
        return

    warnings.warn(
        "\n"
        + "=" * 60
        + "\nWARNING: CUDA not available - E2E tests will likely fail!\n"
        + "Tests will still run to identify specific failures.\n"
        + "=" * 60,
        RuntimeWarning,
        stacklevel=2,
    )


def _e2e_env() -> dict[str, str]:
    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    env["OMNI_HEADLESS"] = "1"
    env["DISPLAY"] = ":1"
    return env


def _fail_from_process_error(label: str, exc: subprocess.CalledProcessError) -> None:
    print(f"{label} failed with return code {exc.returncode}", flush=True)
    print("STDOUT:", flush=True)
    print(exc.stdout[-2000:] if exc.stdout else "No stdout", flush=True)
    print("STDERR:", flush=True)
    print(exc.stderr[-2000:] if exc.stderr else "No stderr", flush=True)
    pytest.fail(label)


def _fail_from_timeout(label: str, exc: subprocess.TimeoutExpired) -> None:
    print(f"{label} timed out", flush=True)
    print("Partial STDOUT:", flush=True)
    print(exc.stdout[-2000:] if exc.stdout else "No output", flush=True)
    print("Partial STDERR:", flush=True)
    print(exc.stderr[-2000:] if exc.stderr else "No output", flush=True)
    pytest.fail(label)


def _training_overrides(task: str) -> list[str]:
    overrides = []
    if "StandUp" in task or "HeightTracking" in task:
        overrides.extend(
            [
                "agent.fallen_state_dataset_cfg.num_spawns_per_level=1",
                "agent.fallen_state_dataset_cfg.fall_duration_s=0.1",
                "agent.fallen_state_dataset_cfg.cache_enabled=False",
            ]
        )
    if "HeightTracking" in task:
        overrides.extend(
            [
                "agent.fallen_state_dataset_secondary_cfg.num_spawns_per_level=1",
                "agent.fallen_state_dataset_secondary_cfg.fall_duration_s=0.1",
                "agent.fallen_state_dataset_secondary_cfg.cache_enabled=False",
            ]
        )

    return overrides


def test_every_registered_task_is_covered_or_skipped() -> None:
    """Force new tasks into the e2e matrix: every registered trainable AGILE task must be in
    E2E_TASKS or explicitly listed in E2E_SKIP."""
    import agile.rl_env.tasks  # noqa: F401

    registered = {
        task_id
        for task_id, spec in gym.registry.items()
        if str((spec.kwargs or {}).get("env_cfg_entry_point", "")).startswith("agile.rl_env.tasks.")
        and "rsl_rl_cfg_entry_point" in (spec.kwargs or {})
    }
    covered = {t.task_id for t in E2E_TASKS} | set(E2E_SKIP)
    missing = sorted(registered - covered)
    assert missing == [], f"Tasks neither covered nor skipped: {missing}"


@pytest.mark.parametrize("task", [t.task_id for t in E2E_TASKS], ids=str)
def test_task_training_smoke(task: str) -> None:
    """Smoke-test that each table task can train for a few iterations."""
    _warn_if_cuda_is_unavailable()

    if not TRAIN_SCRIPT.exists():
        pytest.skip(f"Train script not found at {TRAIN_SCRIPT}")

    num_iterations = 5
    num_envs = 4
    timeout = 180 if "Distillation" in task else 120

    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
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
            f"hydra.run.dir={temp_dir}",
            *_training_overrides(task),
        ]

        print(f"Running command: {' '.join(cmd)}", flush=True)
        try:
            subprocess.run(
                cmd,
                check=True,
                timeout=timeout,
                capture_output=True,
                text=True,
                env=_e2e_env(),
            )
        except subprocess.CalledProcessError as exc:
            _fail_from_process_error(f"Training smoke test for {task}", exc)
        except subprocess.TimeoutExpired as exc:
            _fail_from_timeout(f"Training smoke test for {task}", exc)


@pytest.mark.parametrize(
    "task",
    _EVAL or [pytest.param(None, marks=pytest.mark.skip(reason="no checkpoints available"))],
    ids=[t.task_id for t in _EVAL] or ["no-checkpoints"],
)
def test_task_evaluation_smoke(task: E2ETask | None) -> None:
    """Smoke-test evaluation for table tasks that ship a checkpoint."""
    _warn_if_cuda_is_unavailable()
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--task",
        task.task_id,
        "--checkpoint",
        str(task.checkpoint),
        "--num_envs",
        "2",
        "--num_steps",
        "50",
        "--headless",
    ]
    print(f"Running command: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True, timeout=120, capture_output=True, text=True, env=_e2e_env())
    except subprocess.CalledProcessError as exc:
        _fail_from_process_error(f"Evaluation smoke test for {task.task_id}", exc)
    except subprocess.TimeoutExpired as exc:
        _fail_from_timeout(f"Evaluation smoke test for {task.task_id}", exc)


def test_play_script_smoke() -> None:
    """Smoke-test scripts/play.py using a single environment."""
    _warn_if_cuda_is_unavailable()

    if not PLAY_SCRIPT.exists():
        pytest.skip(f"Play script not found at {PLAY_SCRIPT}")

    task = "Velocity-G1-History-v0"
    cmd = [
        sys.executable,
        str(PLAY_SCRIPT),
        "--task",
        task,
        "--num_envs",
        "2",
        "--num_steps",
        "10",
        "--headless",
    ]

    print(f"Running command: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=180,
            capture_output=True,
            text=True,
            env=_e2e_env(),
        )
    except subprocess.CalledProcessError as exc:
        _fail_from_process_error(f"Play script smoke test for {task}", exc)
    except subprocess.TimeoutExpired as exc:
        _fail_from_timeout(f"Play script smoke test for {task}", exc)


@pytest.mark.parametrize(
    "task",
    _SIM2SIM or [pytest.param(None, marks=pytest.mark.skip(reason="no checkpoint+mjcf available"))],
    ids=[t.task_id for t in _SIM2SIM] or ["no-sim2sim"],
)
def test_task_sim2sim_smoke(task: E2ETask | None, tmp_path: Path) -> None:
    """Export a LEAPP bundle from the checkpoint, then run it in MuJoCo (cross-sim smoke)."""
    _warn_if_cuda_is_unavailable()

    export_cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--task",
        task.task_id,
        "--checkpoint",
        str(task.checkpoint),
        "--export_save_path",
        str(tmp_path),
        "--disable_graph_visualization",
    ]
    print(f"Running command: {' '.join(export_cmd)}", flush=True)
    try:
        subprocess.run(export_cmd, check=True, timeout=300, capture_output=True, text=True, env=_e2e_env())
    except subprocess.CalledProcessError as exc:
        _fail_from_process_error(f"LEAPP export for {task.task_id}", exc)
    except subprocess.TimeoutExpired as exc:
        _fail_from_timeout(f"LEAPP export for {task.task_id}", exc)

    bundle = tmp_path / task.task_id / f"{task.task_id}.yaml"
    assert bundle.is_file(), f"LEAPP bundle not produced at {bundle}"

    sim_cmd = [
        sys.executable,
        str(SIM2MUJOCO_SCRIPT),
        "--leapp-yaml",
        str(bundle),
        "--mjcf",
        str(task.mjcf),
        "--no-viewer",
        "--device",
        "cpu",
        "--no-real-time",
        "--duration",
        "2",
    ]
    print(f"Running command: {' '.join(sim_cmd)}", flush=True)
    try:
        subprocess.run(sim_cmd, check=True, timeout=120, capture_output=True, text=True, env=_e2e_env())
    except subprocess.CalledProcessError as exc:
        _fail_from_process_error(f"Sim2MuJoCo for {task.task_id}", exc)
    except subprocess.TimeoutExpired as exc:
        _fail_from_timeout(f"Sim2MuJoCo for {task.task_id}", exc)
