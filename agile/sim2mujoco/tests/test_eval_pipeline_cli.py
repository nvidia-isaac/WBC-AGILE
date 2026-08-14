# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""eval_pipeline --dry-run prints the 4 stage commands in order without running them."""

import subprocess
import sys
from pathlib import Path

from scripts.eval_pipeline import _headless_mujoco_env

ROOT = Path(__file__).resolve().parents[3]


def test_dry_run_lists_all_stages(tmp_path):
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_pipeline.py"),
            "--task",
            "Velocity-Height-G1-History-v0",
            "--checkpoint",
            "/tmp/ckpt.pt",
            "--mjcf",
            "/tmp/g1.xml",
            "--output-dir",
            str(tmp_path),
            "--run-label",
            "local",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    body = out.stdout
    assert "eval.py" in body and "--video" in body
    assert "--fail_on_non_timeout_dones" in body
    assert "export_policy_leapp.py" in body
    assert "sim2mujoco_eval.py" in body and "--video" in body and "--eval-env-id 0" in body
    assert "generate_report.py" in body and "--eval_video" in body and "--sim2sim_video" in body


def test_dry_run_requires_an_exact_wandb_checkpoint(tmp_path):
    command = [
        sys.executable,
        str(ROOT / "scripts" / "eval_pipeline.py"),
        "--task",
        "Velocity-Height-G1-History-v0",
        "--wandb_run",
        "entity/project/run",
        "--mjcf",
        "/tmp/g1.xml",
        "--output-dir",
        str(tmp_path),
        "--run-label",
        "remote",
        "--dry-run",
    ]
    out = subprocess.run(command, capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode != 0
    assert "exact" in out.stderr


def test_video_only_report_uses_eval_reports_directory(tmp_path):
    spec = tmp_path / "video-only.yaml"
    spec.write_text("video_only: true\n")
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_pipeline.py"),
            "--task",
            "Velocity-Height-G1-History-v0",
            "--checkpoint",
            "/tmp/ckpt.pt",
            "--mjcf",
            "/tmp/g1.xml",
            "--output-dir",
            str(tmp_path / "out"),
            "--run-label",
            "video-only",
            "--evaluation-spec",
            str(spec),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    expected = tmp_path / "out" / "Velocity-Height-G1-History-v0" / "video-only" / "eval" / "reports"
    assert f"--output-dir {expected}" in out.stdout


def test_video_only_still_exports_leapp_and_runs_sim2mujoco(tmp_path):
    spec = tmp_path / "video-only.yaml"
    spec.write_text("video_only: true\n")
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_pipeline.py"),
            "--task",
            "PickPlace-G1-v0",
            "--checkpoint",
            "/tmp/ckpt.pt",
            "--mjcf",
            "/tmp/g1.xml",
            "--output-dir",
            str(tmp_path / "out"),
            "--run-label",
            "video-only",
            "--evaluation-spec",
            str(spec),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    body = out.stdout
    assert "eval.py" in body
    assert "--run_evaluation" not in body
    assert "generate_report.py" in body
    assert "--eval_video" in body
    assert "export_policy_leapp.py" in body
    assert "--validation_steps 0" in body
    assert "sim2mujoco_eval.py" in body
    assert "--sim2sim_video" in body


def test_spec_can_require_continuous_isaac_lab_rollout(tmp_path):
    spec = tmp_path / "continuous-rollout.yaml"
    spec.write_text("video_only: true\nfail_on_non_timeout_dones: true\nnon_timeout_done_warmup_steps: 1\n")
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_pipeline.py"),
            "--task",
            "MotionTracking-G1-v0",
            "--checkpoint",
            "/tmp/ckpt.pt",
            "--mjcf",
            "/tmp/g1.xml",
            "--output-dir",
            str(tmp_path / "out"),
            "--run-label",
            "tracking",
            "--evaluation-spec",
            str(spec),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    assert "--fail_on_non_timeout_dones" in out.stdout
    assert "--non_timeout_done_warmup_steps 1" in out.stdout


def test_sim2mujoco_video_defaults_to_egl_without_overriding_user_backend(monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    assert _headless_mujoco_env()["MUJOCO_GL"] == "egl"

    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    assert _headless_mujoco_env()["MUJOCO_GL"] == "osmesa"


def test_spec_can_skip_sim2mujoco_but_still_export_leapp(tmp_path):
    spec = tmp_path / "video-only-no-sim2mujoco.yaml"
    spec.write_text("video_only: true\nsim2mujoco: false\n")
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_pipeline.py"),
            "--task",
            "PickPlace-G1-v0",
            "--checkpoint",
            "/tmp/ckpt.pt",
            "--mjcf",
            "/tmp/g1.xml",
            "--output-dir",
            str(tmp_path / "out"),
            "--run-label",
            "video-only",
            "--evaluation-spec",
            str(spec),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    body = out.stdout
    assert "eval.py" in body
    assert "export_policy_leapp.py" in body
    assert "--validation_steps 0" in body
    assert "generate_report.py" in body
    assert "--eval_video" in body
    assert "sim2mujoco_eval.py" not in body
    assert "--sim2sim_video" not in body
