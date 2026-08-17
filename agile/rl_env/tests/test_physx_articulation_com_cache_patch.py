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

from __future__ import annotations

import ast
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import pytest

from agile.isaaclab_extras import physx_articulation_com_cache as patch_module
from agile.isaaclab_extras.physx_articulation_com_cache import (
    _BODY_COM_DEPENDENT_BUFFER_NAMES,
    _patch_classes,
)

BODY_COM_DEPENDENTS = (
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


def test_patch_logic_import_does_not_load_physx_runtime():
    code = (
        "import sys; "
        "import agile.isaaclab_extras.physx_articulation_com_cache; "
        "assert 'omni.physics' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_monkey_patch_package_registers_physx_com_cache_patch():
    project_root = Path(__file__).resolve().parents[3]

    package_tree = ast.parse((project_root / "agile/isaaclab_extras/monkey_patches/__init__.py").read_text())
    package_imports_patch = any(
        isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module is None
        and any(alias.name == "physx_articulation_com_cache" for alias in node.names)
        for node in package_tree.body
    )
    assert package_imports_patch, "monkey_patches must register the PhysX COM cache patch"


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _call_name(node.value)
        return f"{owner}.{node.attr}" if owner else node.attr
    return None


@pytest.mark.parametrize(
    ("script_path", "constructor_name"),
    (
        ("scripts/play.py", "ManagerBasedRLEnv"),
        ("scripts/export_policy.py", "gym.make"),
    ),
)
def test_patch_bundle_is_registered_after_app_launcher_startup_and_before_construction(script_path, constructor_name):
    project_root = Path(__file__).resolve().parents[3]
    script_tree = ast.parse((project_root / script_path).read_text())
    simulation_app_assignment = next(
        node
        for node in script_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "simulation_app" for target in node.targets)
    )
    patch_import = next(
        (
            node
            for node in script_tree.body
            if isinstance(node, ast.Import)
            and any(alias.name == "agile.isaaclab_extras.monkey_patches" for alias in node.names)
        ),
        None,
    )
    constructor = next(
        node
        for node in ast.walk(script_tree)
        if isinstance(node, ast.Call) and _call_name(node.func) == constructor_name
    )
    assert patch_import is not None, f"{script_path} must activate the AGILE monkey-patch bundle"
    assert simulation_app_assignment.lineno < patch_import.lineno < constructor.lineno


class _ComBufferData:
    def __init__(self, values):
        self.values = deepcopy(values)

    def assign(self, values):
        self.values[:] = deepcopy(values)


class _Buffer:
    def __init__(self, timestamp: float = -1.0, data=None):
        self.timestamp = timestamp
        self.data = Mock() if data is None else data


class _FakeData:
    def __init__(self):
        self._sim_timestamp = 0.0
        self.vendor_coms = [[f"vendor-{env_id}-{body_id}" for body_id in range(3)] for env_id in range(4)]
        cold_coms = [[f"cold-{env_id}-{body_id}" for body_id in range(3)] for env_id in range(4)]
        self._root_view = Mock()
        self._root_view.get_coms.return_value.view.return_value = self.vendor_coms
        self._body_com_pose_b = _Buffer(data=_ComBufferData(cold_coms))
        self._body_com_pose_b_ta = self._body_com_pose_b.data.values
        for name in BODY_COM_DEPENDENTS:
            setattr(self, name, _Buffer(timestamp=7.0))

    @property
    def body_com_pose_b(self):
        if self._body_com_pose_b.timestamp < self._sim_timestamp:
            self._body_com_pose_b.data.assign(self._root_view.get_coms().view())
            self._body_com_pose_b.timestamp = self._sim_timestamp
        return self._body_com_pose_b_ta


class _FakeArticulation:
    def __init__(self):
        self.data = _FakeData()
        self.num_instances = 4
        self.num_bodies = 3
        self.original_set_coms_calls = 0
        self.original_joint_writer_calls = []

    def _resolve_env_ids(self, env_ids):
        return SimpleNamespace(shape=(self.num_instances if env_ids is None else len(env_ids),))

    def _resolve_body_ids(self, body_ids):
        return SimpleNamespace(shape=(self.num_bodies if body_ids is None else len(body_ids),))

    def set_coms_index(self, *, coms, body_ids=None, env_ids=None, full_data=False):
        self.original_set_coms_calls += 1
        resolved_env_ids = list(range(self.num_instances)) if env_ids is None else env_ids
        resolved_body_ids = list(range(self.num_bodies)) if body_ids is None else body_ids
        target = self.data._body_com_pose_b.data.values
        for env_offset, env_id in enumerate(resolved_env_ids):
            for body_offset, body_id in enumerate(resolved_body_ids):
                source_env_id = env_id if full_data else env_offset
                source_body_id = body_id if full_data else body_offset
                target[env_id][body_id] = coms[source_env_id][source_body_id]

    def write_joint_state_to_sim_index(self, **kwargs):
        self.original_joint_writer_calls.append(("write_joint_state_to_sim_index", kwargs))
        self.data._body_com_pose_b.timestamp = -1.0
        return sentinel.original_joint_writer_result

    def write_joint_position_to_sim_index(self, **kwargs):
        self.original_joint_writer_calls.append(("write_joint_position_to_sim_index", kwargs))
        self.data._body_com_pose_b.timestamp = -1.0
        return sentinel.original_joint_writer_result


def make_fresh_fake_classes():
    class FreshFakeData(_FakeData):
        pass

    class FreshFakeArticulation:
        _resolve_env_ids = _FakeArticulation._resolve_env_ids
        _resolve_body_ids = _FakeArticulation._resolve_body_ids
        set_coms_index = _FakeArticulation.set_coms_index
        write_joint_state_to_sim_index = _FakeArticulation.write_joint_state_to_sim_index
        write_joint_position_to_sim_index = _FakeArticulation.write_joint_position_to_sim_index

        def __init__(self):
            self.data = FreshFakeData()
            self.num_instances = 4
            self.num_bodies = 3
            self.original_set_coms_calls = 0
            self.original_joint_writer_calls = []

    return FreshFakeArticulation, FreshFakeData


def test_body_com_dependent_buffer_names_match_upstream():
    assert _BODY_COM_DEPENDENT_BUFFER_NAMES == BODY_COM_DEPENDENTS


def test_warm_body_com_cache_survives_sim_timestamp_updates():
    articulation_cls, data_cls = make_fresh_fake_classes()
    assert _patch_classes(articulation_cls, data_cls)
    data = data_cls()

    first_value = data.body_com_pose_b
    data._sim_timestamp = 1.0
    second_value = data.body_com_pose_b

    assert first_value is data._body_com_pose_b_ta
    assert second_value is data._body_com_pose_b_ta
    assert data._root_view.get_coms.call_count == 1
    assert data._body_com_pose_b.timestamp == 0.0


@pytest.mark.parametrize(
    "writer_name",
    ("write_joint_state_to_sim_index", "write_joint_position_to_sim_index"),
)
def test_joint_writers_preserve_only_a_warm_com_cache(writer_name):
    articulation_cls, data_cls = make_fresh_fake_classes()
    _patch_classes(articulation_cls, data_cls)
    articulation = articulation_cls()
    articulation.data._sim_timestamp = 5.0
    articulation.data._body_com_pose_b.timestamp = 2.0

    warm_result = getattr(articulation, writer_name)(env_ids=sentinel.warm_env_ids)
    assert articulation.data._body_com_pose_b.timestamp == 2.0
    assert articulation.original_joint_writer_calls == [
        (writer_name, {"env_ids": sentinel.warm_env_ids}),
    ]
    assert warm_result is sentinel.original_joint_writer_result

    articulation.data._body_com_pose_b.timestamp = -1.0
    cold_result = getattr(articulation, writer_name)(env_ids=sentinel.cold_env_ids)
    assert articulation.data._body_com_pose_b.timestamp == -1.0
    assert articulation.original_joint_writer_calls == [
        (writer_name, {"env_ids": sentinel.warm_env_ids}),
        (writer_name, {"env_ids": sentinel.cold_env_ids}),
    ]
    assert cold_result is sentinel.original_joint_writer_result


def test_partial_com_write_primes_cold_cache_once_and_preserves_unselected_values():
    articulation_cls, data_cls = make_fresh_fake_classes()
    _patch_classes(articulation_cls, data_cls)
    articulation = articulation_cls()
    articulation.data._sim_timestamp = 3.0

    articulation.set_coms_index(
        coms=[["updated-1-0", "updated-1-2"], ["updated-3-0", "updated-3-2"]],
        env_ids=[1, 3],
        body_ids=[0, 2],
    )

    assert articulation.original_set_coms_calls == 1
    assert articulation.data._root_view.get_coms.call_count == 1
    assert articulation.data._body_com_pose_b.timestamp == 3.0
    assert articulation.data._body_com_pose_b.data.values == [
        ["vendor-0-0", "vendor-0-1", "vendor-0-2"],
        ["updated-1-0", "vendor-1-1", "updated-1-2"],
        ["vendor-2-0", "vendor-2-1", "vendor-2-2"],
        ["updated-3-0", "vendor-3-1", "updated-3-2"],
    ]
    assert all(getattr(articulation.data, name).timestamp == -1.0 for name in BODY_COM_DEPENDENTS)


def test_partial_com_write_uses_warm_cache_without_refetching_and_preserves_unselected_values():
    articulation_cls, data_cls = make_fresh_fake_classes()
    _patch_classes(articulation_cls, data_cls)
    articulation = articulation_cls()
    _ = articulation.data.body_com_pose_b
    articulation.data._sim_timestamp = 3.0

    articulation.set_coms_index(coms=[["updated-1-2"]], env_ids=[1], body_ids=[2])

    assert articulation.original_set_coms_calls == 1
    assert articulation.data._root_view.get_coms.call_count == 1
    assert articulation.data._body_com_pose_b.timestamp == 3.0
    assert articulation.data._body_com_pose_b.data.values == [
        ["vendor-0-0", "vendor-0-1", "vendor-0-2"],
        ["vendor-1-0", "vendor-1-1", "updated-1-2"],
        ["vendor-2-0", "vendor-2-1", "vendor-2-2"],
        ["vendor-3-0", "vendor-3-1", "vendor-3-2"],
    ]


def test_full_com_write_overwrites_cold_cache_without_priming():
    articulation_cls, data_cls = make_fresh_fake_classes()
    _patch_classes(articulation_cls, data_cls)
    articulation = articulation_cls()
    full_coms = [[f"updated-{env_id}-{body_id}" for body_id in range(3)] for env_id in range(4)]

    articulation.set_coms_index(coms=full_coms, full_data=True)

    assert articulation.original_set_coms_calls == 1
    articulation.data._root_view.get_coms.assert_not_called()
    assert articulation.data._body_com_pose_b.data.values == full_coms


def test_missing_dependent_fails_before_com_setter_or_cache_mutation():
    articulation_cls, data_cls = make_fresh_fake_classes()
    _patch_classes(articulation_cls, data_cls)
    articulation = articulation_cls()
    delattr(articulation.data, "_body_state_w")
    original_coms = deepcopy(articulation.data._body_com_pose_b.data.values)

    with pytest.raises(RuntimeError, match="_body_state_w"):
        articulation.set_coms_index(coms=[["updated-1-2"]], env_ids=[1], body_ids=[2])

    assert articulation.original_set_coms_calls == 0
    articulation.data._root_view.get_coms.assert_not_called()
    assert articulation.data._body_com_pose_b.timestamp == -1.0
    assert articulation.data._body_com_pose_b.data.values == original_coms
    assert all(
        getattr(articulation.data, name).timestamp == 7.0 for name in BODY_COM_DEPENDENTS if name != "_body_state_w"
    )


def test_class_patch_is_idempotent():
    articulation_cls, data_cls = make_fresh_fake_classes()
    assert _patch_classes(articulation_cls, data_cls)
    assert not _patch_classes(articulation_cls, data_cls)


def test_non_beta2_version_returns_without_importing_physx(monkeypatch):
    monkeypatch.setattr(patch_module, "version", lambda _: "7.3.0")
    assert not patch_module.apply_physx_articulation_com_cache_patch()


def test_malformed_beta2_class_fails_before_mutation():
    articulation_cls, data_cls = make_fresh_fake_classes()
    delattr(articulation_cls, "set_coms_index")
    with pytest.raises(RuntimeError, match="set_coms_index"):
        _patch_classes(articulation_cls, data_cls)
    assert not getattr(data_cls, "_agile_body_com_cache_patch_applied", False)
