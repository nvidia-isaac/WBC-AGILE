"""Validated, reproducible evaluation-batch manifests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from agile.evaluation.task_catalog import TASK_CATALOG


@dataclass(frozen=True)
class LocalCheckpoint:
    path: Path


@dataclass(frozen=True)
class WandbCheckpoint:
    run: str
    file_name: str | None = None
    iteration: int | None = None
    artifact_version: str | None = None


@dataclass(frozen=True)
class EvaluationRun:
    label: str
    task_id: str
    checkpoint: LocalCheckpoint | WandbCheckpoint
    specification: Path


@dataclass(frozen=True)
class EvaluationSpec:
    metric_suite: str | None
    video_only: bool
    scenario: Path | None
    sim2mujoco_scenario: Path | None
    sim2mujoco: bool
    fail_on_non_timeout_dones: bool
    non_timeout_done_warmup_steps: int


_CATALOG = {entry.task_id: entry for entry in TASK_CATALOG}
_SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
SUPPORTED_METRIC_SUITES = frozenset({"motion_tracking"})


def validate_run_label(label: str) -> str:
    """Return a label safe to use as one output-directory component."""
    if not _SAFE_LABEL.fullmatch(label):
        raise ValueError("evaluation run label must be a safe path component")
    return label


def load_evaluation_spec(path: Path) -> EvaluationSpec:
    """Load a task specification with an explicit metric or video-only decision."""
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"evaluation specification must be a mapping: {path}")
    metric_suite = data.get("metric_suite")
    video_only = data.get("video_only") is True
    if (metric_suite is None) == (not video_only):
        raise ValueError("evaluation specification must select exactly one metric_suite or video_only")
    if metric_suite is not None and str(metric_suite) not in SUPPORTED_METRIC_SUITES:
        raise ValueError(f"unsupported metric_suite: {metric_suite}")
    scenario = data.get("scenario")
    sim2mujoco_scenario = data.get("sim2mujoco_scenario")
    return EvaluationSpec(
        str(metric_suite) if metric_suite is not None else None,
        video_only,
        Path(scenario) if scenario else None,
        Path(sim2mujoco_scenario) if sim2mujoco_scenario else None,
        data.get("sim2mujoco") is not False,
        data.get("fail_on_non_timeout_dones") is not False,
        int(data.get("non_timeout_done_warmup_steps", 0)),
    )


def _parse_checkpoint(data: object) -> LocalCheckpoint | WandbCheckpoint:
    if not isinstance(data, dict):
        raise ValueError("checkpoint must be a mapping")
    if "local_path" in data and len(data) == 1:
        return LocalCheckpoint(Path(data["local_path"]))
    run = data.get("wandb_run")
    exact = {key: data.get(key) for key in ("file_name", "iteration", "artifact_version") if data.get(key) is not None}
    if run is None:
        raise ValueError("checkpoint must specify local_path or wandb_run")
    if len(exact) != 1:
        raise ValueError(
            "a W&B checkpoint must name exactly one exact checkpoint (file_name, iteration, or artifact_version)"
        )
    return WandbCheckpoint(str(run), **exact)


def parse_manifest(
    path: Path,
    *,
    only_task_ids: set[str] | None = None,
    excluded_task_ids: set[str] | None = None,
) -> tuple[EvaluationRun, ...]:
    """Load named evaluation runs, rejecting ambiguous checkpoint sources."""
    excluded_task_ids = excluded_task_ids or set()
    data = yaml.safe_load(path.read_text())
    entries = data.get("runs") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must contain a non-empty 'runs' list")
    runs: list[EvaluationRun] = []
    labels: set[str] = set()
    for raw in entries:
        if not isinstance(raw, dict):
            raise ValueError("each run must be a mapping")
        label, task_id = raw.get("label"), raw.get("task_id")
        if not isinstance(label, str) or not label:
            raise ValueError("each run requires a non-empty label")
        if only_task_ids is not None and task_id not in only_task_ids:
            continue
        if task_id in excluded_task_ids:
            continue
        validate_run_label(label)
        if label in labels:
            raise ValueError(f"duplicate evaluation run label: {label}")
        entry = _CATALOG.get(task_id)
        if entry is None:
            raise ValueError(f"unknown task: {task_id}")
        if entry.eligibility != "trainable":
            raise ValueError(f"task is excluded from automation: {task_id}: {entry.exclusion_reason}")
        if not entry.public_evaluation:
            raise ValueError(f"task is not part of public evaluation automation: {task_id}")
        specification = Path(raw.get("evaluation_spec", entry.evaluation_spec or ""))
        if not specification or not specification.is_file():
            raise ValueError(f"task has no evaluation specification: {task_id}")
        load_evaluation_spec(specification)
        runs.append(EvaluationRun(label, task_id, _parse_checkpoint(raw.get("checkpoint")), specification))
        labels.add(label)
    return tuple(runs)
