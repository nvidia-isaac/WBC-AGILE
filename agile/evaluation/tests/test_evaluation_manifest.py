from pathlib import Path

import pytest

from agile.evaluation.evaluation_manifest import load_evaluation_spec, parse_manifest

def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(content)
    return path


def test_manifest_supports_two_labeled_runs_for_one_task(tmp_path: Path) -> None:
    runs = parse_manifest(
        _write(
            tmp_path,
            """
runs:
  - label: iter-10000
    task_id: Velocity-G1-History-v0
    checkpoint: {local_path: /tmp/model_10000.pt}
  - label: iter-20000
    task_id: Velocity-G1-History-v0
    checkpoint: {local_path: /tmp/model_20000.pt}
""",
        )
    )
    assert [(run.task_id, run.label) for run in runs] == [
        ("Velocity-G1-History-v0", "iter-10000"),
        ("Velocity-G1-History-v0", "iter-20000"),
    ]


def test_manifest_rejects_bare_wandb_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact checkpoint"):
        parse_manifest(
            _write(
                tmp_path,
                """
runs:
  - label: not-reproducible
    task_id: Velocity-G1-History-v0
    checkpoint: {wandb_run: entity/project/run}
""",
            )
        )


def test_evaluation_spec_requires_metric_suite_or_video_only(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("scenario: example.yaml\n")
    with pytest.raises(ValueError, match="metric_suite or video_only"):
        load_evaluation_spec(path)


def test_evaluation_spec_rejects_unsupported_metric_suite(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("metric_suite: made_up\n")
    with pytest.raises(ValueError, match="unsupported metric_suite"):
        load_evaluation_spec(path)


def test_evaluation_spec_runs_sim2mujoco_by_default(tmp_path: Path) -> None:
    path = tmp_path / "video-only.yaml"
    path.write_text("video_only: true\n")

    assert load_evaluation_spec(path).sim2mujoco is True


def test_evaluation_spec_can_disable_sim2mujoco(tmp_path: Path) -> None:
    path = tmp_path / "video-only-no-sim2mujoco.yaml"
    path.write_text("video_only: true\nsim2mujoco: false\n")

    assert load_evaluation_spec(path).sim2mujoco is False


def test_evaluation_spec_can_require_continuous_isaac_lab_rollout(tmp_path: Path) -> None:
    path = tmp_path / "continuous-rollout.yaml"
    path.write_text("video_only: true\nfail_on_non_timeout_dones: true\nnon_timeout_done_warmup_steps: 1\n")

    spec = load_evaluation_spec(path)
    assert spec.fail_on_non_timeout_dones is True
    assert spec.non_timeout_done_warmup_steps == 1


def test_manifest_rejects_run_label_that_escapes_output_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="safe path component"):
        parse_manifest(
            _write(
                tmp_path,
                """
runs:
  - label: ../outside
    task_id: Velocity-G1-History-v0
    checkpoint: {local_path: /tmp/model.pt}
""",
            )
        )


def test_standup_t1_evaluation_is_video_only() -> None:
    spec = load_evaluation_spec(Path("agile/evaluation/specs/StandUp-T1-v0.yaml"))

    assert spec.video_only is True
    assert spec.metric_suite is None


def test_height_tracking_g1_evaluation_is_video_only() -> None:
    spec = load_evaluation_spec(Path("agile/evaluation/specs/HeightTracking-G1-v0.yaml"))

    assert spec.video_only is True
    assert spec.metric_suite is None
    assert spec.sim2mujoco is True


def test_pickplace_g1_skips_sim2mujoco_until_public_hand_mjcf_exists() -> None:
    spec = load_evaluation_spec(Path("agile/evaluation/specs/PickPlace-G1-v0.yaml"))

    assert spec.video_only is True
    assert spec.metric_suite is None
    assert spec.sim2mujoco is False


def test_evaluation_spec_rejects_non_timeout_dones_by_default(tmp_path: Path) -> None:
    path = tmp_path / "default.yaml"
    path.write_text("video_only: true\n")

    spec = load_evaluation_spec(path)

    assert spec.fail_on_non_timeout_dones is True


def test_evaluation_spec_can_allow_non_timeout_dones_when_explicitly_opted_out(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opt_out.yaml"
    path.write_text("video_only: true\nfail_on_non_timeout_dones: false\n")

    spec = load_evaluation_spec(path)

    assert spec.fail_on_non_timeout_dones is False
