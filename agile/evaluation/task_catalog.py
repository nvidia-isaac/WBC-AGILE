"""Production catalog of every registered AGILE task eligible for automation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agile.evaluation.external_assets import G1_MJCF, T1_MJCF


@dataclass(frozen=True)
class TaskCatalogEntry:
    """One registered task and its automation eligibility."""

    task_id: str
    eligibility: Literal["trainable", "excluded"]
    exclusion_reason: str | None = None
    mjcf: Path | None = None
    evaluation_spec: Path | None = None
    public_evaluation: bool = True


_G1_MJCF = G1_MJCF
_T1_MJCF = T1_MJCF
_SPECS = Path("agile/evaluation/specs")

TASK_CATALOG: tuple[TaskCatalogEntry, ...] = (
    *(
        TaskCatalogEntry(task, "trainable", mjcf=_G1_MJCF, evaluation_spec=_SPECS / f"{task}.yaml")
        for task in (
            "Velocity-G1-History-v0",
            "Velocity-Height-G1-History-v0",
            "Velocity-Height-G1-Student-Recurrent-v0",
            "Velocity-Height-G1-Student-History-v0",
            "HeightTracking-G1-v0",
            "PickPlace-G1-v0",
            "MotionTracking-G1-v0",
        )
    ),
    *(
        TaskCatalogEntry(
            task,
            "trainable",
            mjcf=_G1_MJCF,
            evaluation_spec=_SPECS / f"{task}.yaml",
            public_evaluation=False,
        )
        for task in (
            "Velocity-G1-Teacher-v0",
            "Velocity-Height-G1-Teacher-v0",
        )
    ),
    *(
        TaskCatalogEntry(task, "trainable", mjcf=_T1_MJCF, evaluation_spec=_SPECS / f"{task}.yaml")
        for task in ("Velocity-T1-v0", "StandUp-T1-v0")
    ),
    *(
        TaskCatalogEntry(task, "excluded", reason)
        for task, reason in (
            ("Debug-G1-v0", "debug environment"),
            ("Debug-G1-Object-v0", "debug environment"),
            ("Debug-T1-v0", "debug environment"),
            ("PickPlace-G1-Debug-v0", "debug environment"),
            ("PickPlace-G1-Record-v0", "recording environment"),
            ("PickPlace-G1-GR00T-Inference-v0", "manual inference environment"),
        )
    ),
)


def trainable_tasks() -> tuple[TaskCatalogEntry, ...]:
    """Return tasks included in automated training."""
    return tuple(entry for entry in TASK_CATALOG if entry.eligibility == "trainable")


def public_evaluation_tasks() -> tuple[TaskCatalogEntry, ...]:
    """Return trainable tasks included in public evaluation automation."""
    return tuple(entry for entry in trainable_tasks() if entry.public_evaluation)


def registered_agile_trainable_task_ids() -> set[str]:
    """Return the registered AGILE tasks that expose an RSL-RL training config."""
    import gymnasium as gym

    import agile.rl_env.tasks  # noqa: F401

    return {
        task_id
        for task_id, spec in gym.registry.items()
        if str((spec.kwargs or {}).get("env_cfg_entry_point", "")).startswith("agile.rl_env.tasks.")
        and "rsl_rl_cfg_entry_point" in (spec.kwargs or {})
    }
