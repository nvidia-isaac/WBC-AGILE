# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _load_patch_module():
    patch_path = Path(__file__).parents[2] / "isaaclab_extras/monkey_patches/terrain_importer_plane_patch.py"
    spec = importlib.util.spec_from_file_location("terrain_importer_plane_patch_test", patch_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plain_ground_plane_avoids_beta2_legacy_material_paths(monkeypatch):
    patch = _load_patch_module()
    spawned_cfg = Mock()
    ground_plane_cfg = Mock(return_value=spawned_cfg)
    monkeypatch.setattr(patch.sim_utils, "GroundPlaneCfg", ground_plane_cfg)
    importer = SimpleNamespace(
        cfg=SimpleNamespace(prim_path="/World/ground", physics_material=None, visual_material=None),
        terrain_prim_paths=[],
        terrain_names=[],
    )

    patch._import_ground_plane(importer, "terrain", size=(4.0, 6.0))

    ground_plane_cfg.assert_called_once_with(physics_material=None, size=(4.0, 6.0), color=None)
    spawned_cfg.func.assert_called_once_with("/World/ground/terrain", spawned_cfg)
    assert importer.terrain_prim_paths == ["/World/ground/terrain"]


def test_materialized_ground_planes_keep_isaac_lab_behavior(monkeypatch):
    patch = _load_patch_module()
    original_importer = Mock()
    monkeypatch.setattr(patch, "_ORIGINAL_IMPORT_GROUND_PLANE", original_importer)
    importer = SimpleNamespace(
        cfg=SimpleNamespace(physics_material=object(), visual_material=None),
    )

    patch._import_ground_plane(importer, "terrain", size=(4.0, 6.0))

    original_importer.assert_called_once_with(importer, "terrain", (4.0, 6.0))


def test_duplicate_plain_ground_plane_matches_isaac_lab_error():
    patch = _load_patch_module()
    importer = SimpleNamespace(
        cfg=SimpleNamespace(prim_path="/World/ground", physics_material=None, visual_material=None),
        terrain_prim_paths=["/World/ground/terrain"],
        terrain_names=["terrain"],
    )

    with pytest.raises(ValueError, match="already exists"):
        patch._import_ground_plane(importer, "terrain")
