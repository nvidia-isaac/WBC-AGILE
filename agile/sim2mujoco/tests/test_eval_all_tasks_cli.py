# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
import textwrap
from pathlib import Path

from scripts.eval_all_tasks import _local_image_key

ROOT = Path(__file__).resolve().parents[3]


def test_local_image_key_distinguishes_bundle_basenames(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"same checkpoint")
    second.write_bytes(b"same checkpoint")

    assert _local_image_key("eval-all", first) != _local_image_key("eval-all", second)


def test_dry_run_local_lists_pipeline_per_task(tmp_path):
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        textwrap.dedent("""
        runs:
          - label: remote-10000
            task_id: Velocity-Height-G1-History-v0
            checkpoint: {wandb_run: nvidia-isaac/Velocity-Height-G1-Lower/abc, iteration: 10000}
          - label: local
            task_id: Velocity-G1-History-v0
            checkpoint: {local_path: /tmp/g1.pt}
    """)
    )
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_all_tasks.py"),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    assert "eval_pipeline.py" in out.stdout
    assert "Velocity-Height-G1-History-v0" in out.stdout and "Velocity-G1-History-v0" in out.stdout
    assert "--run-label remote-10000" in out.stdout and "--wandb-iteration 10000" in out.stdout


def test_dry_run_osmo_includes_name(tmp_path):
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        textwrap.dedent("""
        runs:
          - label: remote-10000
            task_id: Velocity-Height-G1-History-v0
            checkpoint: {wandb_run: nvidia-isaac/Velocity-Height-G1-Lower/abc, iteration: 10000}
    """)
    )
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_all_tasks.py"),
            "--manifest",
            str(manifest),
            "--osmo",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    assert "run.py" in out.stdout
    assert "eval-pipeline" in out.stdout
    assert "--name" in out.stdout


def test_dry_run_forwards_manifest_spec_and_makes_osmo_names_label_distinct(tmp_path):
    spec = tmp_path / "custom.yaml"
    spec.write_text("video_only: true\n")
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        textwrap.dedent(f"""
        runs:
          - label: baseline
            task_id: Velocity-G1-History-v0
            evaluation_spec: {spec}
            checkpoint: {{local_path: /tmp/g1.pt}}
          - label: candidate
            task_id: Velocity-G1-History-v0
            checkpoint: {{local_path: /tmp/g1.pt}}
    """)
    )
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_all_tasks.py"), "--manifest", str(manifest), "--osmo"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    assert f"--evaluation-spec {spec}" in out.stdout
    assert "--name eval-velocity-g1-history-v0-baseline" in out.stdout
    assert "--name eval-velocity-g1-history-v0-candidate" in out.stdout
    assert out.stdout.count("--image-key eval-all-local-") == 2
    assert out.stdout.count("--rebuild") == 1
    assert out.stdout.count("--use-existing") == 1


def test_osmo_batch_sanitizes_names_and_reuses_one_remote_image(tmp_path):
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        textwrap.dedent("""
        runs:
          - label: run_1.2
            task_id: Velocity-Height-G1-History-v0
            checkpoint: {wandb_run: nvidia-isaac/project/first, iteration: 10000}
          - label: candidate
            task_id: Velocity-G1-History-v0
            checkpoint: {wandb_run: nvidia-isaac/project/second, iteration: 20000}
    """)
    )
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_all_tasks.py"), "--manifest", str(manifest), "--osmo"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    assert "--name eval-velocity-height-g1-history-v0-run-1-2" in out.stdout
    assert out.stdout.count("--image-key eval-all") == 2
    assert out.stdout.count("--rebuild") == 1
    assert out.stdout.count("--use-existing") == 1


def test_osmo_batch_rejects_names_that_collide_after_sanitizing(tmp_path):
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        textwrap.dedent("""
        runs:
          - label: run_1.2
            task_id: Velocity-G1-History-v0
            checkpoint: {wandb_run: nvidia-isaac/project/first, iteration: 10000}
          - label: run-1-2
            task_id: Velocity-G1-History-v0
            checkpoint: {wandb_run: nvidia-isaac/project/second, iteration: 20000}
    """)
    )
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_all_tasks.py"), "--manifest", str(manifest), "--osmo"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 2
    assert "workflow name collision" in out.stderr
