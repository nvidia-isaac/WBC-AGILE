from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import run
from scripts.eval_pipeline import _metric_suite_args, _required_videos, _validate_leapp_bundle


def test_checkpoint_bundle_name_includes_content_digest(tmp_path: Path) -> None:
    first = tmp_path / "first" / "model.pt"
    second = tmp_path / "second" / "model.pt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    assert run._checkpoint_bundle_name(first) != run._checkpoint_bundle_name(second)
    assert run._checkpoint_bundle_name(first).startswith(hashlib.sha256(b"first").hexdigest()[:16])


def test_eval_pipeline_submits_content_addressed_bundled_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"exact checkpoint")
    submitted: dict[str, object] = {}

    monkeypatch.setattr(run, "build_docker_image", lambda **_kwargs: "registry/agile:test")
    monkeypatch.setattr(run, "store_image_mapping", lambda *_args: None)

    def capture_submit(_workflow: Path, set_args: list[str], pool: str, priority: str | None = None) -> None:
        submitted.update(set_args=set_args, pool=pool, priority=priority)

    monkeypatch.setattr(run, "submit_osmo_workflow", capture_submit)
    config = SimpleNamespace(
        image_name="registry/agile",
        omni_server_url="omniverse://server",
        eval_pipeline_workflow=Path("workflows/e2e_eval_workflow.yaml"),
        osmo_pools={"eval": "pool"},
    )

    run.handle_eval_pipeline(
        name="test",
        task_name="Velocity-G1-History-v0",
        wandb_run=None,
        wandb_iteration=None,
        wandb_checkpoint_file=None,
        wandb_artifact_version=None,
        checkpoint_path=str(checkpoint),
        run_label="test",
        mjcf="scene.xml",
        evaluation_spec=None,
        use_existing=False,
        rebuild=False,
        priority=None,
        set_args=[],
        run_config=config,
    )

    expected_name = run._checkpoint_bundle_name(checkpoint)
    assert f"checkpoint_path=/workspace/agile/policy/resume/{expected_name}" in submitted["set_args"]


def test_submit_osmo_workflow_omits_empty_set_args(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: list[list[str]] = []

    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **_kwargs: submitted.append(cmd))

    run.submit_osmo_workflow(Path("workflow.yaml"), [], pool="pool", priority="LOW")

    assert submitted == [["osmo", "workflow", "submit", "workflow.yaml", "--pool=pool", "--priority=LOW"]]


def test_eval_pipeline_workflow_forwards_every_exact_wandb_selector() -> None:
    workflow = Path("workflows/e2e_eval_workflow.yaml").read_text()
    assert "--wandb-checkpoint-file {{wandb_checkpoint_file}}" in workflow
    assert "--wandb-artifact-version {{wandb_artifact_version}}" in workflow
    assert "--evaluation-spec {{evaluation_spec}}" in workflow


def test_eval_pipeline_workflow_defaults_to_official_unitree_scene() -> None:
    workflow = Path("workflows/e2e_eval_workflow.yaml").read_text()
    assert "agile-download-assets" in workflow
    assert "mjcf: external_assets/unitree_mujoco/unitree_robots/g1/scene_29dof.xml" in workflow


def test_eval_pipeline_workflow_uses_current_osmo_schema() -> None:
    workflow = yaml.safe_load(Path("workflows/e2e_eval_workflow.yaml").read_text())

    assert "timeout" not in workflow
    outputs = workflow["workflow"]["tasks"][0]["outputs"]
    assert outputs == [{"url": "{{storage_url_prefix}}/{{workflow_id}}/"}]
    assert workflow["default-values"]["storage_url_prefix"] == "swift://pdx.s8k.io/AUTH_team-isaac/datasets/agile-eval"


def test_eval_pipeline_workflow_uses_shell_timeout_for_eval_command() -> None:
    workflow = Path("workflows/e2e_eval_workflow.yaml").read_text()

    assert "timeout 4h uv run --frozen --offline --no-sync scripts/eval_pipeline.py" in workflow


def test_eval_pipeline_workflow_stages_reports_for_osmo_upload() -> None:
    workflow = Path("workflows/e2e_eval_workflow.yaml").read_text()

    assert "mkdir -p /osmo/data/output" in workflow
    assert "cp -a outputs/. /osmo/data/output/" in workflow
    assert "chmod -R a+rX /osmo/data/output" in workflow


def test_required_videos_rejects_a_missing_isaac_or_mujoco_video(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Isaac"):
        _required_videos(tmp_path / "isaac", tmp_path / "sim2sim.mp4")

    isaac = tmp_path / "isaac"
    isaac.mkdir()
    (isaac / "rollout.mp4").write_bytes(b"video")
    with pytest.raises(RuntimeError, match="MuJoCo"):
        _required_videos(isaac, tmp_path / "sim2sim.mp4")


def test_motion_tracking_suite_is_explicitly_dispatched() -> None:
    assert _metric_suite_args("motion_tracking") == ["--run_evaluation", "--save_trajectories"]


def test_validate_leapp_bundle_rejects_missing_yaml(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="LEAPP YAML was not produced"):
        _validate_leapp_bundle(tmp_path / "Task-v0.yaml")


def test_validate_leapp_bundle_rejects_missing_frequency(tmp_path: Path) -> None:
    bundle_yaml = tmp_path / "Task-v0.yaml"
    bundle_yaml.write_text(
        yaml.safe_dump({"models": {"Task-v0": {"parameters": {"model_path": "Task-v0.onnx"}}}, "pipeline": {}})
    )
    (tmp_path / "Task-v0.onnx").write_bytes(b"onnx")

    with pytest.raises(RuntimeError, match="pipeline.configs.frequency"):
        _validate_leapp_bundle(bundle_yaml)


def test_validate_leapp_bundle_rejects_missing_model_file(tmp_path: Path) -> None:
    bundle_yaml = tmp_path / "Task-v0.yaml"
    bundle_yaml.write_text(
        yaml.safe_dump(
            {
                "models": {"Task-v0": {"parameters": {"model_path": "Task-v0.onnx"}}},
                "pipeline": {"configs": {"frequency": 50.0}},
            }
        )
    )

    with pytest.raises(RuntimeError, match="model file"):
        _validate_leapp_bundle(bundle_yaml)


def test_validate_leapp_bundle_accepts_complete_bundle(tmp_path: Path) -> None:
    bundle_yaml = tmp_path / "Task-v0.yaml"
    bundle_yaml.write_text(
        yaml.safe_dump(
            {
                "models": {"Task-v0": {"parameters": {"model_path": "Task-v0.onnx"}}},
                "pipeline": {
                    "configs": {"frequency": 50.0},
                    "initial_values": "Task-v0_initial_values.safetensors",
                },
            }
        )
    )
    (tmp_path / "Task-v0.onnx").write_bytes(b"onnx")
    (tmp_path / "Task-v0_initial_values.safetensors").write_bytes(b"initial")

    _validate_leapp_bundle(bundle_yaml)
