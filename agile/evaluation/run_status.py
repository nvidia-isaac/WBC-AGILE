"""Persistent status for an evaluation run that may fail part-way through."""

from __future__ import annotations

import json
from pathlib import Path


def write_status(run_dir: Path, stage: str, state: str, message: str = "") -> Path:
    """Atomically update a run's current stage status."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"stage": stage, "state": state, "message": message}, indent=2) + "\n")
    temporary.replace(path)
    return path
