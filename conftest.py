from __future__ import annotations

from pathlib import Path


def pytest_configure(config) -> None:
    """Reject accidental Isaac application startup in the unit-test process."""
    if (config.option.markexpr or "").strip() == "e2e":
        return

    from isaaclab.app import AppLauncher

    def fail_unit_app_launch(*_args, **_kwargs):
        raise RuntimeError("Unit tests must not launch AppLauncher; mark simulator tests e2e")

    AppLauncher.__init__ = fail_unit_app_launch


def pytest_ignore_collect(collection_path: Path, config) -> bool:
    """Route default and E2E pytest entrypoints to disjoint test modules."""
    if (config.option.markexpr or "").strip() == "e2e":
        return collection_path.name.startswith("test_") and not collection_path.name.endswith("_e2e.py")

    return collection_path.name.endswith("_e2e.py")
