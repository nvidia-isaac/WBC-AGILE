"""Backport the beta2 ground-plane importer compatibility fix.

Isaac Lab 3.0.0b2's default ground-plane USD no longer contains the legacy
``Plane`` collision prim or ``Looks/theGrid`` shader assumed by its terrain
importer. Evaluation uses a plain plane, so spawn it without those obsolete
material overrides. Remove this module once AGILE upgrades past beta2.
"""

from __future__ import annotations

from importlib.metadata import version

from isaaclab import sim as sim_utils
from isaaclab.terrains.terrain_importer import TerrainImporter

_TARGET_ISAACLAB_VERSION = "3.0.0b2"
_PATCH_SENTINEL = "_agile_plain_ground_plane_patch_applied"
_ORIGINAL_IMPORT_GROUND_PLANE = TerrainImporter.import_ground_plane


def _import_ground_plane(self: TerrainImporter, name: str, size: tuple[float, float] = (2.0e6, 2.0e6)):
    """Spawn a plain beta2 ground plane without legacy material overrides."""
    if self.cfg.physics_material is not None or self.cfg.visual_material is not None:
        return _ORIGINAL_IMPORT_GROUND_PLANE(self, name, size)

    prim_path = self.cfg.prim_path + f"/{name}"
    if prim_path in self.terrain_prim_paths:
        raise ValueError(
            f"A terrain with the name '{name}' already exists. Existing terrains: {', '.join(self.terrain_names)}."
        )
    self.terrain_prim_paths.append(prim_path)
    ground_plane_cfg = sim_utils.GroundPlaneCfg(physics_material=None, size=size, color=None)
    ground_plane_cfg.func(prim_path, ground_plane_cfg)


def apply_terrain_importer_plane_patch() -> bool:
    """Patch only the Isaac Lab beta release affected by the legacy USD paths."""
    if version("isaaclab") != _TARGET_ISAACLAB_VERSION or getattr(TerrainImporter, _PATCH_SENTINEL, False):
        return False
    TerrainImporter.import_ground_plane = _import_ground_plane
    setattr(TerrainImporter, _PATCH_SENTINEL, True)
    return True


apply_terrain_importer_plane_patch()
