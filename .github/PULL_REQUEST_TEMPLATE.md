## Summary

<!-- One-line description of what this PR does -->

## Motivation

<!-- Why is this change needed? -->

## Changes

<!-- List the key changes -->
-

## PR Checklist

Before merging, confirm you have completed the following:

### Pre-commit checks

Run pre-commit on all files and ensure all hooks pass:

```bash
pre-commit run --all-files
```

- [ ] All pre-commit hooks pass (ruff, ruff-format, mypy, yamlfmt, etc.)

### End-to-end and unit tests

Run the following tests to verify correctness. Requires a machine with GPU and `ISAACLAB_PATH` set.

```bash
isaaclab -p -m pytest tests/test_all_tasks_e2e.py -v
isaaclab -p -m pytest tests/test_sim2mujoco_e2e.py -v
./run_unit_tests.sh
```

- [ ] `test_all_tasks_e2e.py` - All registered tasks train for 5 iterations without errors
- [ ] `test_sim2mujoco_e2e.py` - All sim2mujoco evaluations pass with provided checkpoints
- [ ] `run_unit_tests.sh` - All unit tests pass without errors

> **Note:** If you added a new task, make sure it is registered in
> `tests/test_all_tasks_e2e.py::get_all_tasks()` and, if applicable,
> `get_tasks_with_checkpoints()`.

## Testing

<!-- How was this PR verified beyond the checklist above? -->
-
