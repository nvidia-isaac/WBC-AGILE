from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from scripts import eval_all_tasks, submit_all_tasks


def test_eval_all_osmo_name_fits_prefixed_osmo_limit() -> None:
    name = eval_all_tasks._osmo_workflow_name(
        "Velocity-Height-G1-Student-Recurrent-v0",
        "main-f33da1b3-20260707-velocity-height-g1-distillation-recurrent-v0-model-49999",
    )

    assert len(f"agile_eval_pipeline_{name}") <= 90


def _manifest(path: Path) -> Path:
    path.write_text(
        """runs:
  - label: one
    task_id: Velocity-G1-History-v0
    checkpoint: {local_path: /tmp/model.pt}
"""
    )
    return path


def test_eval_all_returns_a_child_submission_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_all_tasks.py", "--manifest", str(_manifest(tmp_path / "manifest.yaml")), "--submit"],
    )
    monkeypatch.setattr(eval_all_tasks.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    assert eval_all_tasks.main() == 1


def test_eval_all_forwards_osmo_priority(tmp_path: Path, monkeypatch) -> None:
    submitted: list[list[str]] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_all_tasks.py",
            "--manifest",
            str(_manifest(tmp_path / "manifest.yaml")),
            "--osmo",
            "--submit",
            "--priority",
            "LOW",
        ],
    )
    monkeypatch.setattr(
        eval_all_tasks.subprocess,
        "run",
        lambda cmd, **_kwargs: submitted.append(cmd) or SimpleNamespace(returncode=0),
    )

    assert eval_all_tasks.main() == 0
    assert "--priority" in submitted[0]
    assert submitted[0][submitted[0].index("--priority") + 1] == "LOW"


def test_eval_all_osmo_aggregate_report_submits_one_map_reduce_workflow(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """runs:
  - label: one
    task_id: Velocity-G1-History-v0
    checkpoint: {wandb_run: nvidia-isaac/project/run-a, file_name: model_49999.pt}
  - label: two
    task_id: Velocity-T1-v0
    checkpoint: {wandb_run: nvidia-isaac/project/run-b, file_name: model_49999.pt}
"""
    )
    submitted: list[tuple[Path, list[str], str, str | None]] = []

    monkeypatch.setattr(eval_all_tasks.remote_run, "build_docker_image", lambda **_kwargs: "registry/agile:test")
    monkeypatch.setattr(eval_all_tasks.remote_run, "store_image_mapping", lambda *_args: None)
    monkeypatch.setattr(
        eval_all_tasks.remote_run,
        "submit_osmo_workflow",
        lambda workflow, set_args, pool, priority=None: submitted.append((workflow, set_args, pool, priority)),
    )
    monkeypatch.setattr(
        eval_all_tasks.remote_run.RunConfig,
        "load_from_path",
        lambda _path: SimpleNamespace(
            image_name="registry/agile",
            dockerfile=Path("Dockerfile"),
            osmo_pools={"eval": "eval-pool"},
            omni_server_url="omniverse://server",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_all_tasks.py",
            "--manifest",
            str(manifest),
            "--osmo",
            "--submit",
            "--aggregate-report",
            "--batch-name",
            "public-latest",
            "--priority",
            "LOW",
        ],
    )

    assert eval_all_tasks.main() == 0
    assert len(submitted) == 1
    workflow, set_args, pool, priority = submitted[0]
    workflow_text = workflow.read_text()
    workflow_yaml = yaml.safe_load(workflow_text)
    eval_scripts = [task["files"][0]["contents"] for task in workflow_yaml["workflow"]["tasks"][:-1]]
    aggregate_script = workflow_yaml["workflow"]["tasks"][-1]["files"][0]["contents"]
    assert set_args == []
    assert pool == "eval-pool"
    assert priority == "LOW"
    assert workflow_text.count("scripts/eval_pipeline.py") == 2
    assert (
        sum(
            "timeout 4h uv run --frozen --offline --no-sync scripts/eval_pipeline.py" in script
            for script in eval_scripts
        )
        == 2
    )
    assert all('echo "${eval_status}" > "{{output}}/_exit_code"' in script for script in eval_scripts)
    assert all(script.rstrip().endswith("exit 0") for script in eval_scripts)
    assert all(script.index("set +e") < script.index("agile-download-assets") for script in eval_scripts)
    assert "name: aggregate-report" in workflow_text
    assert "inputs:" in workflow_text
    assert "task: eval-velocity-g1-history-v0-one" in workflow_text
    assert "task: eval-velocity-t1-v0-two" in workflow_text
    assert "{{input:0}}" in aggregate_script
    assert "{{output}}" in aggregate_script
    assert "scripts/build_eval_index.py --batch-dir batch" in aggregate_script


def test_eval_all_only_filters_before_manifest_validation(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """runs:
  - label: public
    task_id: HeightTracking-G1-v0
    checkpoint: {local_path: /tmp/model.pt}
  - label: internal
    task_id: EETracking-G1-v0
    checkpoint: {local_path: /tmp/model.pt}
"""
    )
    submitted: list[list[str]] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_all_tasks.py",
            "--manifest",
            str(manifest),
            "--only",
            "HeightTracking-G1-v0",
            "--submit",
        ],
    )
    monkeypatch.setattr(
        eval_all_tasks.subprocess,
        "run",
        lambda cmd, **_kwargs: submitted.append(cmd) or SimpleNamespace(returncode=0),
    )

    assert eval_all_tasks.main() == 0
    assert len(submitted) == 1
    assert "HeightTracking-G1-v0" in submitted[0]
    assert "EETracking-G1-v0" not in submitted[0]


def test_submit_all_uses_the_production_catalog() -> None:
    assert "tests.test_task_smoke_e2e" not in Path(submit_all_tasks.__file__).read_text()
