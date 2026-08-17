# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patch Isaac Lab LEAPP fallback action export for AGILE action terms.

Isaac Lab's fallback exporter records ``term.processed_actions`` for action terms
whose writes were not intercepted. AGILE has two cases that need stricter handling:

* helper actions with zero action dimension, such as lift assistance, are not
  policy outputs and should not become dynamic LEAPP outputs;
* custom joint-position terms still need semantic joint-target metadata so
  downstream runtimes can map the exported output back to robot joints.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

AGILE_COMMAND_OBSERVATIONS = frozenset({"motion_anchor_pos_b", "motion_anchor_ori_b"})


def build_command_element_names(*, command_term: Any, command_cfg: Any, result: torch.Tensor) -> list[str] | None:
    """Resolve LEAPP element names for exported command tensors."""
    dim = int(result.shape[-1]) if result.ndim > 0 else 0
    configured_names = getattr(command_cfg, "element_names", None)
    if configured_names is not None:
        names = _flat_names(configured_names)
        if len(names) == dim:
            return names

    term_names = getattr(command_term, "command_names", None)
    if term_names is not None:
        names = _flat_names(term_names)
        if len(names) == dim:
            return names

    joint_names = getattr(getattr(getattr(command_term, "robot", None), "data", None), "joint_names", None)
    if getattr(command_term, "motion", None) is not None and joint_names is not None:
        names = _flat_names(joint_names)
        if dim == 2 * len(names):
            return [f"ref_joint_pos/{name}" for name in names] + [f"ref_joint_vel/{name}" for name in names]

    return None


def _flat_names(names: Any) -> list[str]:
    """Return a flat string list from LEAPP-style element names."""
    if not isinstance(names, list | tuple):
        return []
    if not names:
        return []
    if all(isinstance(item, str) for item in names):
        return [str(item) for item in names]
    if len(names) == 1 and isinstance(names[0], list | tuple):
        return [str(item) for item in names[0]]
    return []


def build_command_observation_input(
    *,
    func_name: str | None,
    result: torch.Tensor,
    command_name: str | None,
    build_command_connection: Callable[[str], Any],
) -> dict[str, Any] | None:
    """Return TensorSemantics args for AGILE command-derived observations.

    ``command_name`` is the Isaac Lab command term used to compute the
    observation. The exported LEAPP input keeps the observation function name so
    runtimes can provide the already-computed deployment value directly.
    """
    del command_name
    if func_name not in AGILE_COMMAND_OBSERVATIONS:
        return None
    return {
        "name": func_name,
        "ref": result,
        "kind": None,
        "element_names": None,
        "extra": build_command_connection(func_name),
    }


def build_processed_action_fallback(
    *,
    term_name: str,
    term: Any,
    processed: torch.Tensor | None,
    scene_key: str,
    select_element_names: Callable[[Any, Any], list[str] | None],
    build_write_connection: Callable[[str, str], Any],
) -> dict[str, Any] | None:
    """Return fallback TensorSemantics constructor args for one action term.

    The return value intentionally uses plain Python data so this function can be
    unit-tested without importing Isaac Sim, Isaac Lab's export patcher, or LEAPP.
    """
    if not isinstance(processed, torch.Tensor):
        return None
    if processed.numel() == 0 or (processed.ndim > 0 and processed.shape[-1] == 0):
        return None
    if getattr(term, "action_dim", processed.shape[-1]) == 0:
        return None

    asset = getattr(term, "_asset", None)
    real_asset = getattr(asset, "_real_asset", asset)
    joint_ids = getattr(term, "_joint_ids", None)
    joint_names = getattr(real_asset, "joint_names", None)

    fallback: dict[str, Any] = {
        "name": term_name,
        "ref": processed.clone(),
        "kind": None,
        "element_names": None,
        "extra": None,
    }
    if joint_ids is not None and joint_names is not None:
        fallback["kind"] = "target/joint/position"
        fallback["element_names"] = select_element_names(joint_names, joint_ids)
        fallback["extra"] = build_write_connection(scene_key, "set_joint_position_target_index")
    return fallback


def install_leapp_export_fallback_patch() -> None:
    """Install the AGILE fallback patch onto Isaac Lab's LEAPP ExportPatcher."""
    from leapp import annotate
    from leapp.utils.tensor_description import TensorSemantics

    from isaaclab.utils.leapp import export_annotator
    from isaaclab.utils.leapp.leapp_semantics import select_element_names
    from isaaclab.utils.leapp.utils import build_command_connection, build_write_connection

    export_patcher_cls = export_annotator.ExportPatcher
    if not getattr(export_patcher_cls, "_agile_processed_action_fallback_patch", False):

        def _collect_processed_action_fallbacks(self, action_manager) -> list[TensorSemantics]:
            logger = export_annotator.logging.getLogger(__name__)
            fallback_terms: set[str] = set()
            tensors: list[TensorSemantics] = []
            for term_name, term in action_manager._terms.items():
                if term_name in self._captured_write_term_names:
                    continue
                processed = getattr(term, "processed_actions", None)
                scene_key = self._action_term_scene_keys.get(term_name, "robot")
                fallback = build_processed_action_fallback(
                    term_name=term_name,
                    term=term,
                    processed=processed,
                    scene_key=scene_key,
                    select_element_names=select_element_names,
                    build_write_connection=build_write_connection,
                )
                if fallback is None:
                    continue

                if fallback["kind"] == "target/joint/position":
                    logger.warning(
                        "Action term '%s' did not write to any asset directly. Falling back to processed_actions "
                        "as semantic joint position targets.",
                        term_name,
                    )
                else:
                    logger.warning(
                        "Action term '%s' did not write to any asset directly. Falling back to processed_actions "
                        "as an untyped export output.",
                        term_name,
                    )
                tensors.append(TensorSemantics(**fallback))
                fallback_terms.add(term_name)
            self._fallback_term_names = fallback_terms
            return tensors

        export_patcher_cls._collect_processed_action_fallbacks = _collect_processed_action_fallbacks
        export_patcher_cls._agile_processed_action_fallback_patch = True

    if getattr(export_patcher_cls, "_agile_command_observation_patch", False):
        return

    def _wrap_generated_commands(self, original_func, term_cfg):
        task_name = self.task_name
        command_name_from_cfg = term_cfg.params.get("command_name")

        def wrapped(env, command_name=None, **kwargs):
            result = original_func(env, command_name, **kwargs)
            leapp_input_name = command_name or command_name_from_cfg or "commands"
            command_term = None
            command_cfg = None
            try:
                command_term = env.command_manager.get_term(leapp_input_name)
                command_cfg = command_term.cfg
            except (AttributeError, KeyError):
                pass
            sem = TensorSemantics(
                name=leapp_input_name,
                ref=result,
                kind=getattr(command_cfg, "cmd_kind", None),
                element_names=build_command_element_names(
                    command_term=command_term,
                    command_cfg=command_cfg,
                    result=result,
                ),
                extra=build_command_connection(leapp_input_name),
            )
            return annotate.input_tensors(task_name, sem)

        wrapped.__name__ = original_func.__name__
        return wrapped

    def _wrap_agile_command_observation(self, original_func, term_cfg):
        task_name = self.task_name
        command_name_from_cfg = term_cfg.params.get("command_name")
        func_name = getattr(original_func, "__name__", None)

        def wrapped(env, command_name=None, **kwargs):
            result = original_func(env, command_name, **kwargs)
            command_input = build_command_observation_input(
                func_name=func_name,
                result=result,
                command_name=command_name or command_name_from_cfg,
                build_command_connection=build_command_connection,
            )
            if command_input is None:
                return result
            return annotate.input_tensors(task_name, TensorSemantics(**command_input))

        wrapped.__name__ = getattr(original_func, "__name__", "unknown")
        return wrapped

    original_patch_observation_manager = export_patcher_cls._patch_observation_manager

    def _patch_observation_manager(self, obs_manager, proxy_env):
        for group_name, term_cfgs in obs_manager._group_obs_term_cfgs.items():
            if self.required_obs_groups is not None and group_name not in self.required_obs_groups:
                continue
            for term_cfg in term_cfgs:
                original_func = term_cfg.func
                func_name = getattr(original_func, "__name__", None)
                if func_name not in AGILE_COMMAND_OBSERVATIONS:
                    continue
                term_cfg.func = self._wrap_agile_command_observation(original_func, term_cfg)

        return original_patch_observation_manager(self, obs_manager, proxy_env)

    export_patcher_cls._wrap_generated_commands = _wrap_generated_commands
    export_patcher_cls._wrap_agile_command_observation = _wrap_agile_command_observation
    export_patcher_cls._patch_observation_manager = _patch_observation_manager
    export_patcher_cls._agile_command_observation_patch = True
