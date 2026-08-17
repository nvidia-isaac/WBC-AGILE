"""Post-process AGILE LEAPP export metadata."""

from pathlib import Path
from typing import Any

import yaml


def update_leapp_export_metadata(
    leapp_yaml_path: Path | str,
    *,
    frequency_hz: float,
    robot_articulation: dict[str, Any] | None = None,
) -> None:
    """Ensure exported LEAPP bundle metadata matches the upstream Isaac Lab schema."""

    path = Path(leapp_yaml_path)
    desc = yaml.safe_load(path.read_text())
    configs = desc.setdefault("pipeline", {}).setdefault("configs", {})
    existing_frequency = configs.get("frequency")
    try:
        has_positive_frequency = float(existing_frequency) > 0.0
    except (TypeError, ValueError):
        has_positive_frequency = False
    if not has_positive_frequency:
        configs["frequency"] = float(frequency_hz)

    if robot_articulation is not None:
        robot = desc.setdefault("agile", {}).setdefault("articulations", {}).setdefault("robot", {})
        for key in ("joint_names", "default_joint_pos", "default_joint_stiffness", "default_joint_damping"):
            if key not in robot_articulation:
                continue
            values = robot_articulation[key]
            if key == "joint_names":
                robot[key] = [str(value) for value in values]
            else:
                robot[key] = [float(value) for value in values]
    path.write_text(yaml.safe_dump(desc, sort_keys=False))
