#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one named, reproducible evaluation pipeline per manifest entry."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run as remote_run  # noqa: E402
from agile.evaluation.evaluation_manifest import LocalCheckpoint, WandbCheckpoint, parse_manifest  # noqa: E402
from agile.evaluation.task_catalog import TASK_CATALOG  # noqa: E402

_MJCF = {entry.task_id: entry.mjcf for entry in TASK_CATALOG if entry.mjcf is not None}
_RUN_PY_EVAL_PIPELINE_PREFIX = "agile_eval_pipeline_"
_RUN_PY_EVAL_BATCH_PREFIX = "agile_eval_batch_"
_OSMO_WORKFLOW_NAME_LIMIT = 90
_EVAL_STORAGE_URL_PREFIX = "swift://pdx.s8k.io/AUTH_team-isaac/datasets/agile-eval"


def _args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--exclude", nargs="+", default=[])
    p.add_argument("--output-dir", type=Path, default=ROOT / "logs" / "e2e_eval")
    p.add_argument("--osmo", action="store_true", help="Submit one OSMO job per task instead of running locally.")
    p.add_argument("--submit", action="store_true", help="Actually run/launch (default is dry-run).")
    p.add_argument("--image-key", default="eval-all", help="Base key for reusable OSMO evaluation images.")
    p.add_argument(
        "--priority",
        default=None,
        help=(
            "OSMO workflow priority passed to run.py eval-pipeline (HIGH, NORMAL, LOW). "
            "LOW is preemptible and can run when non-preemptible pool quota is full."
        ),
    )
    p.add_argument(
        "--no-build",
        action="store_true",
        help="Use an existing image for the first run in each image group instead of rebuilding it.",
    )
    p.add_argument(
        "--aggregate-report",
        action="store_true",
        help="Submit one OSMO map/reduce workflow: per-task eval jobs plus a final aggregate report job.",
    )
    p.add_argument(
        "--batch-name",
        default=None,
        help="Name suffix for the aggregate report workflow. Defaults to --image-key.",
    )
    return p.parse_args()


def _osmo_workflow_name(task: str, label: str) -> str:
    """Return a deterministic OSMO-safe name that fits after run.py adds its prefix."""
    raw = f"eval-{task}-{label}".lower()
    slug = "".join(char if char.isascii() and char.isalnum() else "-" for char in raw)
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")
    max_len = _OSMO_WORKFLOW_NAME_LIMIT - len(_RUN_PY_EVAL_PIPELINE_PREFIX)
    if len(slug) <= max_len:
        return slug
    digest = hashlib.sha256(slug.encode()).hexdigest()[:10]
    return f"{slug[: max_len - len(digest) - 1].rstrip('-')}-{digest}"


def _local_image_key(base_key: str, checkpoint: Path) -> str:
    """Group local runs only when the same content-addressed bundle file can be present."""
    digest = hashlib.sha256()
    if checkpoint.is_file():
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    else:
        # Dry runs commonly use placeholder paths. Keep them distinct without requiring the file.
        digest.update(str(checkpoint.resolve()).encode())
    # The in-image name retains the source basename, so equal bytes under different names still
    # require distinct images.
    digest.update(b"\0")
    digest.update(checkpoint.name.encode())
    return f"{base_key}-local-{digest.hexdigest()[:16]}"


def _slug(value: str) -> str:
    slug = "".join(char if char.isascii() and char.isalnum() else "-" for char in value.lower())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _batch_workflow_name(batch_name: str) -> str:
    slug = _slug(batch_name)
    max_len = _OSMO_WORKFLOW_NAME_LIMIT - len(_RUN_PY_EVAL_BATCH_PREFIX)
    if len(slug) <= max_len:
        return f"{_RUN_PY_EVAL_BATCH_PREFIX}{slug}"
    digest = hashlib.sha256(slug.encode()).hexdigest()[:10]
    return f"{_RUN_PY_EVAL_BATCH_PREFIX}{slug[: max_len - len(digest) - 1].rstrip('-')}-{digest}"


def _task_name(task: str, label: str, used: set[str]) -> str:
    slug = _slug(f"eval-{task}-{label}")
    max_len = 63
    if len(slug) > max_len:
        digest = hashlib.sha256(slug.encode()).hexdigest()[:10]
        slug = f"{slug[: max_len - len(digest) - 1].rstrip('-')}-{digest}"
    if slug in used:
        raise ValueError(f"OSMO task name collision after sanitizing: {slug}")
    used.add(slug)
    return slug


def _checkpoint_args(checkpoint: WandbCheckpoint) -> list[str]:
    args = ["--wandb_run", checkpoint.run]
    if checkpoint.iteration is not None:
        return args + ["--wandb-iteration", str(checkpoint.iteration)]
    if checkpoint.file_name is not None:
        return args + ["--wandb-checkpoint-file", checkpoint.file_name]
    return args + ["--wandb-artifact-version", str(checkpoint.artifact_version)]


def _eval_entry_script(
    *,
    task: str,
    label: str,
    checkpoint: WandbCheckpoint,
    specification: Path,
    mjcf: Path,
) -> str:
    cmd = [
        "timeout",
        "4h",
        "uv",
        "run",
        "--frozen",
        "--offline",
        "--no-sync",
        "scripts/eval_pipeline.py",
        "--task",
        task,
        "--run-label",
        label,
        "--evaluation-spec",
        str(specification),
        *_checkpoint_args(checkpoint),
        "--mjcf",
        str(mjcf),
        "--output-dir",
        '"{{output}}"',
    ]
    return "\n".join(
        [
            "set -e",
            "export MUJOCO_GL=egl",
            "set +e",
            'mkdir -p "{{output}}"',
            "setup_status=$?",
            "if [ ${setup_status} -eq 0 ]; then",
            "  uv run --frozen --no-sync agile-download-assets",
            "  setup_status=$?",
            "fi",
            "if [ ${setup_status} -eq 0 ]; then",
            f"  {' '.join(cmd)}",
            "  eval_status=$?",
            "else",
            "  eval_status=${setup_status}",
            "fi",
            'echo "${eval_status}" > "{{output}}/_exit_code"',
            'chmod -R a+rX "{{output}}"',
            "exit 0",
        ]
    )


def _aggregate_entry_script(input_count: int) -> str:
    lines = ["set -e", "mkdir -p batch"]
    lines += [f'cp -a "{{{{input:{idx}}}}}/." batch/' for idx in range(input_count)]
    lines += [
        "uv run --frozen --offline --no-sync scripts/build_eval_index.py --batch-dir batch",
        'mkdir -p "{{output}}"',
        'cp -a batch/. "{{output}}"/',
        'chmod -R a+rX "{{output}}"',
    ]
    return "\n".join(lines)


def _task_outputs(task_name: str) -> list[dict[str, str]]:
    return [{"url": f"{_EVAL_STORAGE_URL_PREFIX}/{{{{workflow_id}}}}/map/{task_name}/"}]


def _aggregate_outputs() -> list[dict[str, str]]:
    return [{"url": f"{_EVAL_STORAGE_URL_PREFIX}/{{{{workflow_id}}}}/"}]


def _build_map_reduce_workflow(runs, *, batch_name: str, image: str, omni_server: str) -> dict:
    used_names: set[str] = set()
    tasks = []
    eval_task_names = []
    for run in runs:
        if isinstance(run.checkpoint, LocalCheckpoint):
            raise ValueError(
                "--aggregate-report currently requires W&B checkpoints; use single-task debug for local checkpoints"
            )
        mjcf = _MJCF.get(run.task_id)
        if mjcf is None:
            raise ValueError(f"no MJCF for {run.task_id}; sim2sim is not supported")
        task_name = _task_name(run.task_id, run.label, used_names)
        eval_task_names.append(task_name)
        tasks.append(
            {
                "name": task_name,
                "resource": "eval",
                "image": image,
                "command": ["/bin/bash"],
                "args": ["/tmp/entry.sh"],
                "environment": {
                    "ACCEPT_EULA": "Y",
                    "OMNI_SERVER": omni_server,
                    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
                },
                "credentials": {
                    "omni-auth": {"OMNI_PASS": "omni_pass", "OMNI_USER": "omni_user"},
                    "wandb": {"WANDB_API_KEY": "wandb_api_key", "WANDB_USERNAME": "wandb_user"},
                },
                "files": [
                    {
                        "path": "/tmp/entry.sh",
                        "contents": _eval_entry_script(
                            task=run.task_id,
                            label=run.label,
                            checkpoint=run.checkpoint,
                            specification=run.specification,
                            mjcf=mjcf,
                        ),
                    }
                ],
                "outputs": _task_outputs(task_name),
            }
        )
    tasks.append(
        {
            "name": "aggregate-report",
            "resource": "aggregate",
            "image": image,
            "command": ["/bin/bash"],
            "args": ["/tmp/entry.sh"],
            "inputs": [{"task": task_name} for task_name in eval_task_names],
            "files": [{"path": "/tmp/entry.sh", "contents": _aggregate_entry_script(len(eval_task_names))}],
            "outputs": _aggregate_outputs(),
        }
    )
    return {
        "version": 2,
        "workflow": {
            "name": _batch_workflow_name(batch_name),
            "resources": {
                "eval": {"cpu": 4, "gpu": 1, "memory": "100Gi", "storage": "100Gi"},
                "aggregate": {"cpu": 4, "gpu": 0, "memory": "32Gi", "storage": "200Gi"},
            },
            "tasks": tasks,
        },
    }


def _write_map_reduce_workflow(workflow: dict, batch_name: str) -> Path:
    workflow_dir = ROOT / "logs" / "e2e_eval_workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    path = workflow_dir / f"{_slug(batch_name)}.yaml"
    path.write_text(yaml.safe_dump(workflow, sort_keys=False))
    return path


def _submit_map_reduce_workflow(runs, *, batch_name: str, image_key: str, rebuild: bool, priority: str | None) -> Path:
    run_config = remote_run.RunConfig.load_from_path(ROOT / "workflows" / "run_config.yaml")
    image = None
    if not rebuild:
        image = remote_run.get_existing_image(image_key)
    if image is None or rebuild:
        image = remote_run.build_docker_image(run_config=run_config)
        remote_run.store_image_mapping(image_key, image)
    workflow = _build_map_reduce_workflow(
        runs,
        batch_name=batch_name,
        image=image,
        omni_server=run_config.omni_server_url,
    )
    workflow_path = _write_map_reduce_workflow(workflow, batch_name)
    remote_run.submit_osmo_workflow(
        workflow_path,
        [],
        pool=run_config.osmo_pools.get("eval", next(iter(run_config.osmo_pools.values()))),
        priority=priority,
    )
    return workflow_path


def main() -> int:
    a = _args()
    runs = list(
        parse_manifest(
            a.manifest,
            only_task_ids=set(a.only) if a.only is not None else None,
            excluded_task_ids=set(a.exclude),
        )
    )
    if not runs:
        print("[ERROR] no tasks selected", file=sys.stderr)
        return 2
    if a.aggregate_report and not a.osmo:
        print("[ERROR] --aggregate-report requires --osmo", file=sys.stderr)
        return 2
    if a.aggregate_report and a.osmo:
        batch_name = a.batch_name or a.image_key
        if not a.submit:
            workflow = _build_map_reduce_workflow(
                runs,
                batch_name=batch_name,
                image="<image-built-on-submit>",
                omni_server="<omni-server-from-run-config>",
            )
            workflow_path = _write_map_reduce_workflow(workflow, batch_name)
            print(f"=== DRY RUN: {len(runs)} evaluation run(s), osmo=True, aggregate_report=True ===")
            print(f"\nGenerated map/reduce workflow: {workflow_path}")
            print("\n(Dry run — re-run with --submit to execute.)")
            return 0
        workflow_path = _submit_map_reduce_workflow(
            runs,
            batch_name=batch_name,
            image_key=a.image_key,
            rebuild=not a.no_build,
            priority=a.priority,
        )
        print(f"\nSubmitted map/reduce workflow: {workflow_path}")
        print("Final report will be uploaded by the aggregate-report task.")
        return 0

    print(f"=== {'SUBMIT' if a.submit else 'DRY RUN'}: {len(runs)} evaluation run(s), osmo={a.osmo} ===")
    used_workflow_names: set[str] = set()
    used_image_keys: set[str] = set()
    for run in runs:
        task = run.task_id
        if isinstance(run.checkpoint, LocalCheckpoint):
            src = ["--checkpoint", str(run.checkpoint.path)]
        else:
            assert isinstance(run.checkpoint, WandbCheckpoint)
            src = ["--wandb_run", run.checkpoint.run]
            if run.checkpoint.iteration is not None:
                src += ["--wandb-iteration", str(run.checkpoint.iteration)]
            elif run.checkpoint.file_name is not None:
                src += ["--wandb-checkpoint-file", run.checkpoint.file_name]
            else:
                src += ["--wandb-artifact-version", str(run.checkpoint.artifact_version)]
        mjcf = _MJCF.get(task)
        if mjcf is None:
            print(
                f"[WARN] no MJCF for {task}; sim2sim is not supported — skipping {run.label}",
                file=sys.stderr,
            )
            continue
        if a.osmo:
            # one OSMO job per task: run.py eval-pipeline submits the e2e eval workflow via OSMO
            osmo_name = _osmo_workflow_name(task, run.label)
            if osmo_name in used_workflow_names:
                print(f"[ERROR] OSMO workflow name collision after sanitizing: {osmo_name}", file=sys.stderr)
                return 2
            used_workflow_names.add(osmo_name)
            image_key = (
                _local_image_key(a.image_key, run.checkpoint.path)
                if isinstance(run.checkpoint, LocalCheckpoint)
                else a.image_key
            )
            reuse_image = a.no_build or image_key in used_image_keys
            used_image_keys.add(image_key)
            cmd = [
                "./run.py",
                "eval-pipeline",
                "--name",
                osmo_name,
                "--task",
                task,
                "--run-label",
                run.label,
                "--evaluation-spec",
                str(run.specification),
                *src,
                "--mjcf",
                str(mjcf),
                "--image-key",
                image_key,
                "--use-existing" if reuse_image else "--rebuild",
            ]  # run.py subcommand wraps the OSMO submit
            if a.priority is not None:
                cmd += ["--priority", a.priority]
        else:
            cmd = [
                sys.executable,
                "scripts/eval_pipeline.py",
                "--task",
                task,
                "--run-label",
                run.label,
                "--evaluation-spec",
                str(run.specification),
                *src,
                "--mjcf",
                str(mjcf),
                "--output-dir",
                str(a.output_dir),
            ]
        print(f"\n[{task}/{run.label}]")
        print("  " + " ".join(str(c) for c in cmd))
        if a.submit:
            result = subprocess.run(cmd, check=False, cwd=str(ROOT))
            if result.returncode != 0:
                print(f"[ERROR] evaluation submission failed: {task}/{run.label}", file=sys.stderr)
                return 1
    if not a.submit:
        print("\n(Dry run — re-run with --submit to execute.)")
    print(f"\nReports will be under: {a.output_dir}/<task>/<run-label>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
