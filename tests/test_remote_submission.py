from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import jinja2

import run


def test_main_does_not_forward_environment_secrets_as_template_values(tmp_path: Path, monkeypatch) -> None:
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "run_config.yaml").write_text("placeholder: true\n")
    config = SimpleNamespace(wandb_team_name="example-team")
    submitted: dict[str, object] = {}

    monkeypatch.setenv("WANDB_API_KEY", "example-wandb-secret")
    monkeypatch.setenv("HF_TOKEN", "example-hf-secret")
    monkeypatch.setattr(run, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(run.RunConfig, "load_from_path", classmethod(lambda _cls, _path: config))
    monkeypatch.setattr(run, "handle_train", lambda **kwargs: submitted.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "train",
            "--name",
            "test",
            "--task_name",
            "Velocity-T1-v0",
            "--use-existing",
            "--set",
            "custom=value",
        ],
    )

    run.main()

    assert submitted["set_args"] == ["custom=value"]


def test_training_workflow_renders_without_a_huggingface_token() -> None:
    template = jinja2.Environment(undefined=jinja2.StrictUndefined).from_string(
        Path("workflows/train_workflow.yaml").read_text()
    )

    rendered = template.render(
        workflow_name="test",
        memory="120Gi",
        storage="200Gi",
        image="registry/agile:test",
        image_default="registry/agile:latest",
        omni_server="omniverse://localhost",
        num_envs=4,
        task_name="Velocity-T1-v0",
        logger="wandb",
        project_name="test-project",
        run_name="test-run",
        max_iterations=5,
        video=False,
        video_length=200,
        video_interval_iter=200,
        resume=False,
        output="/osmo/data/output",
        storage_url_prefix="swift://storage.example/agile",
        workflow_id="workflow-id",
    )

    assert "--task Velocity-T1-v0" in rendered


def test_example_run_config_provides_an_output_url_prefix() -> None:
    config = run.RunConfig.load_from_path(Path("workflows/run_config.example.yaml"))

    assert config.osmo_output_url_prefix == "swift://your-osmo-storage-url/agile"


def test_train_and_sweep_forward_the_configured_output_url_prefix(monkeypatch) -> None:
    submissions: list[list[str]] = []
    config = SimpleNamespace(
        image_name="registry/agile",
        omni_server_url="omniverse://localhost",
        osmo_output_url_prefix="swift://storage.example/agile",
        osmo_pools={"train": "train-pool", "sweep": "sweep-pool"},
        train_workflow=Path("workflows/train_workflow.yaml"),
        sweep_workflow=Path("workflows/sweep_workflow.yaml"),
    )

    monkeypatch.setattr(run, "get_existing_image", lambda _name: "registry/agile:test")
    monkeypatch.setattr(
        run,
        "submit_osmo_workflow",
        lambda _workflow, set_args, _pool=None, **_kwargs: submissions.append(set_args),
    )

    run.handle_train_single(
        name="train-test",
        run_config=config,
        task_name="Velocity-T1-v0",
        project_name="test-project",
        resume_checkpoint=None,
        use_existing=True,
        set_args=[],
    )
    run.handle_sweep(
        name="sweep-test",
        run_config=config,
        sweep_name="test-sweep",
        use_existing=True,
        set_args=[],
    )

    expected = "storage_url_prefix=swift://storage.example/agile"
    assert expected in submissions[0]
    assert expected in submissions[1]
