# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backport the PhysX articulation COM cache fixes to Isaac Lab 3.0.0-beta2.

This reproduces the cache behavior from upstream PRs
`#6268 <https://github.com/isaac-sim/IsaacLab/pull/6268>`_ and
`#6150 <https://github.com/isaac-sim/IsaacLab/pull/6150>`_. Remove this module
once AGILE uses an official Isaac Lab release containing both fixes.
"""

from __future__ import annotations

from functools import wraps
from importlib.metadata import version
from typing import Any

_TARGET_ISAACLAB_VERSION = "3.0.0b2"
_PATCH_SENTINEL = "_agile_body_com_cache_patch_applied"
_BODY_COM_DEPENDENT_BUFFER_NAMES = (
    "_root_com_pose_w",
    "_root_com_vel_w",
    "_root_link_vel_w",
    "_body_com_pose_w",
    "_body_com_vel_w",
    "_body_link_vel_w",
    "_root_link_lin_vel_b",
    "_root_link_ang_vel_b",
    "_root_com_lin_vel_b",
    "_root_com_ang_vel_b",
    "_root_state_w",
    "_root_link_state_w",
    "_root_com_state_w",
    "_body_state_w",
    "_body_link_state_w",
    "_body_com_state_w",
    "_body_com_jacobian_w",
    "_mass_matrix",
    "_gravity_compensation_forces",
)


def _preflight_body_com_dependents(data: Any) -> tuple[Any, ...]:
    buffers = []
    for name in _BODY_COM_DEPENDENT_BUFFER_NAMES:
        try:
            buffer = getattr(data, name)
        except AttributeError as exc:
            raise RuntimeError(f"Isaac Lab beta2 data is missing expected member {name}") from exc
        if not hasattr(buffer, "timestamp"):
            raise RuntimeError(f"Isaac Lab beta2 data member {name} is missing timestamp")
        buffers.append(buffer)
    return tuple(buffers)


def _invalidate_body_com_dependents(buffers: tuple[Any, ...]) -> None:
    for buffer in buffers:
        buffer.timestamp = -1.0


def _patch_classes(articulation_cls: type, data_cls: type) -> bool:
    if getattr(data_cls, _PATCH_SENTINEL, False):
        return False

    original_property = getattr(data_cls, "body_com_pose_b", None)
    if not isinstance(original_property, property) or original_property.fget is None:
        raise RuntimeError("Isaac Lab beta2 body_com_pose_b is not the expected property")

    method_names = (
        "set_coms_index",
        "write_joint_state_to_sim_index",
        "write_joint_position_to_sim_index",
    )
    original_methods = {}
    for method_name in method_names:
        original_method = getattr(articulation_cls, method_name, None)
        if not callable(original_method):
            raise RuntimeError(f"Isaac Lab beta2 articulation is missing expected member {method_name}")
        original_methods[method_name] = original_method

    @wraps(original_property.fget)
    def body_com_pose_b(self):
        cached_timestamp = self._body_com_pose_b.timestamp
        if cached_timestamp < 0.0:
            return original_property.fget(self)
        self._body_com_pose_b.timestamp = self._sim_timestamp
        try:
            return original_property.fget(self)
        finally:
            self._body_com_pose_b.timestamp = cached_timestamp

    original_set_coms = original_methods["set_coms_index"]

    @wraps(original_set_coms)
    def set_coms_index(self, *args, **kwargs):
        dependent_buffers = _preflight_body_com_dependents(self.data)
        resolved_env_ids = self._resolve_env_ids(kwargs.get("env_ids"))
        resolved_body_ids = self._resolve_body_ids(kwargs.get("body_ids"))
        is_partial = resolved_env_ids.shape[0] != self.num_instances or resolved_body_ids.shape[0] != self.num_bodies
        if self.data._body_com_pose_b.timestamp < 0.0 and is_partial:
            _ = self.data.body_com_pose_b
        result = original_set_coms(self, *args, **kwargs)
        self.data._body_com_pose_b.timestamp = self.data._sim_timestamp
        _invalidate_body_com_dependents(dependent_buffers)
        return result

    wrapped_writers = {}
    for method_name in method_names[1:]:
        original_writer = original_methods[method_name]

        @wraps(original_writer)
        def preserve_com_cache(self, *args, __original=original_writer, **kwargs):
            cached_timestamp = self.data._body_com_pose_b.timestamp
            try:
                return __original(self, *args, **kwargs)
            finally:
                self.data._body_com_pose_b.timestamp = cached_timestamp

        wrapped_writers[method_name] = preserve_com_cache

    data_cls.body_com_pose_b = property(
        body_com_pose_b,
        original_property.fset,
        original_property.fdel,
        original_property.__doc__,
    )
    articulation_cls.set_coms_index = set_coms_index
    for method_name, writer in wrapped_writers.items():
        setattr(articulation_cls, method_name, writer)

    setattr(data_cls, _PATCH_SENTINEL, True)
    return True


def apply_physx_articulation_com_cache_patch() -> bool:
    if version("isaaclab") != _TARGET_ISAACLAB_VERSION:
        return False

    from isaaclab_physx.assets.articulation.articulation import Articulation
    from isaaclab_physx.assets.articulation.articulation_data import ArticulationData

    return _patch_classes(Articulation, ArticulationData)
