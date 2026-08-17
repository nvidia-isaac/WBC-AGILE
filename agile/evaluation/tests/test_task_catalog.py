from agile.evaluation.task_catalog import TASK_CATALOG, public_evaluation_tasks, registered_agile_trainable_task_ids


def test_every_registered_trainable_task_is_catalogued() -> None:
    assert registered_agile_trainable_task_ids() == {entry.task_id for entry in TASK_CATALOG}


def test_excluded_tasks_have_a_reason() -> None:
    assert all(entry.exclusion_reason for entry in TASK_CATALOG if entry.eligibility == "excluded")


def test_public_evaluation_tasks_match_public_release_boundary() -> None:
    assert {entry.task_id for entry in public_evaluation_tasks()} == {
        "Velocity-G1-History-v0",
        "Velocity-Height-G1-History-v0",
        "Velocity-Height-G1-Student-Recurrent-v0",
        "Velocity-Height-G1-Student-History-v0",
        "HeightTracking-G1-v0",
        "PickPlace-G1-v0",
        "MotionTracking-G1-v0",
        "Velocity-T1-v0",
        "StandUp-T1-v0",
    }


def test_public_evaluation_task_names_are_deployable_and_consistent() -> None:
    task_ids = {entry.task_id for entry in public_evaluation_tasks()}

    assert "Velocity-G1-v0" not in task_ids
    assert "Velocity-Height-G1-v0" not in task_ids
    assert "Velocity-G1-Teacher-v0" not in task_ids
    assert "Velocity-Height-G1-Teacher-v0" not in task_ids
    assert all("Distillation" not in task_id for task_id in task_ids)
    assert "Tracking-Flat-G1-v0" not in task_ids
    assert "G1-PickPlace-Tracking-v0" not in task_ids


def test_privileged_teacher_tasks_are_catalogued_but_not_public_evaluation() -> None:
    entries = {entry.task_id: entry for entry in TASK_CATALOG}
    public_task_ids = {entry.task_id for entry in public_evaluation_tasks()}

    for task_id in ("Velocity-G1-Teacher-v0", "Velocity-Height-G1-Teacher-v0"):
        assert entries[task_id].eligibility == "trainable"
        assert not entries[task_id].public_evaluation
        assert task_id not in public_task_ids


def test_distillation_task_ids_are_named_as_students() -> None:
    trainable_task_ids = {entry.task_id for entry in TASK_CATALOG if entry.eligibility == "trainable"}

    assert "Velocity-Height-G1-Student-Recurrent-v0" in trainable_task_ids
    assert "Velocity-Height-G1-Student-History-v0" in trainable_task_ids
    assert all("Distillation" not in task_id for task_id in trainable_task_ids)


def test_catalog_evaluation_specs_exist_when_set() -> None:
    for entry in TASK_CATALOG:
        if entry.evaluation_spec is not None:
            assert entry.evaluation_spec.exists(), entry.task_id
