"""Helpers for pruning training-only environment terms before policy export."""

from __future__ import annotations

TRAINING_ONLY_ACTIONS = ("harness", "lift", "random_pos", "random_upper_body_pos")
ASSISTANCE_ACTIONS = ("harness", "lift")
DEFAULT_POSITION_ACTIONS = ("random_pos", "random_upper_body_pos")
TRAINING_ONLY_CURRICULA = ("remove_harness", "adaptive_lift")


def remove_training_only_actions(env_cfg) -> list[str]:
    """Remove non-policy action terms that should not become LEAPP outputs."""
    removed: list[str] = []
    actions_cfg = getattr(env_cfg, "actions", None)
    if actions_cfg is None:
        return removed

    for action_name in TRAINING_ONLY_ACTIONS:
        if getattr(actions_cfg, action_name, None) is not None:
            delattr(actions_cfg, action_name)
            removed.append(action_name)

    curriculum_cfg = getattr(env_cfg, "curriculum", None)
    for curriculum_name in TRAINING_ONLY_CURRICULA:
        if curriculum_cfg is not None and getattr(curriculum_cfg, curriculum_name, None) is not None:
            delattr(curriculum_cfg, curriculum_name)

    return removed


def prepare_training_only_actions_for_evaluation(env_cfg) -> tuple[list[str], list[str]]:
    """Remove assistance while retaining deterministic default targets for non-policy joints."""
    removed: list[str] = []
    held_at_default: list[str] = []
    actions_cfg = getattr(env_cfg, "actions", None)
    if actions_cfg is None:
        return removed, held_at_default

    for action_name in ASSISTANCE_ACTIONS:
        if getattr(actions_cfg, action_name, None) is not None:
            delattr(actions_cfg, action_name)
            removed.append(action_name)

    for action_name in DEFAULT_POSITION_ACTIONS:
        action_cfg = getattr(actions_cfg, action_name, None)
        if action_cfg is not None:
            action_cfg.randomize = False
            held_at_default.append(action_name)

    curriculum_cfg = getattr(env_cfg, "curriculum", None)
    for curriculum_name in TRAINING_ONLY_CURRICULA:
        if curriculum_cfg is not None and getattr(curriculum_cfg, curriculum_name, None) is not None:
            delattr(curriculum_cfg, curriculum_name)

    return removed, held_at_default
