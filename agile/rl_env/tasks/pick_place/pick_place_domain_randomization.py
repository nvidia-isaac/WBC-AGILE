# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import math
import random

import torch

from isaaclab.assets import AssetBase
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

from agile.rl_env.tasks.pick_place.pick_place_tracking_env_cfg import EventCfg


def randomize_ground_texture(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    textures: list[str],
    texture_rotation: tuple[float, float] = (0.0, 0.0),
):
    """Randomize the visual texture of the ground plane using Replicator API.

    This changes only the visual appearance of the ground, not the physics.
    The ground prim path is assumed to be /World/ground.

    Args:
        env: The environment instance.
        env_ids: Environment indices being reset/randomized.
        textures: List of texture paths for ground randomization.
        texture_rotation: Range for random texture rotation in radians.
    """
    # Check replicate_physics is disabled
    if env.cfg.scene.replicate_physics:
        raise RuntimeError(
            "Unable to randomize ground texture with scene replication enabled."
            " Please set 'replicate_physics' to False in 'InteractiveSceneCfg'."
        )

    # Lazy import - only available inside Isaac Sim runtime
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("omni.replicator.core")
    import omni.replicator.core as rep

    # Convert rotation from radians to degrees
    texture_rotation_deg = tuple(math.degrees(angle) for angle in texture_rotation)

    # Get the ground plane prim - the terrain importer uses /World/ground
    ground_prim_path = "/World/ground"
    prims_group = rep.get.prims(path_pattern=ground_prim_path)

    with prims_group:
        rep.randomizer.texture(
            textures=textures,
            project_uvw=True,
            texture_rotate=rep.distribution.uniform(*texture_rotation_deg),
        )


def randomize_dome_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (3500.0, 5000.0),
    textures: list[str] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    """Randomize dome light intensity, color, and texture.

    Args:
        env: The environment instance.
        env_ids: Environment indices being reset/randomized.
        intensity_range: Range for random intensity sampling.
        textures: List of HDR texture paths for background randomization. If None or empty, no texture is applied.
        asset_cfg: Scene entity config for the light.
    """
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]

    # Sample and set random intensity
    new_intensity = random.uniform(intensity_range[0], intensity_range[1])
    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    intensity_attr.Set(new_intensity)

    # Randomize dome light texture (HDR background)
    if textures:
        new_texture = random.choice(textures)
        texture_file_attr = light_prim.GetAttribute("inputs:texture:file")
        if texture_file_attr:
            texture_file_attr.Set(new_texture)


_DOME_LIGHT_PARAMS = {
    "asset_cfg": SceneEntityCfg("light"),
    "intensity_range": (2500.0, 3500.0),
    "textures": [
        "omniverse://isaac-dev.ov.nvidia.com/NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr",
        "omniverse://isaac-dev.ov.nvidia.com/NVIDIA/Assets/Skies/Indoor/adams_place_bridge_4k.hdr",
        "omniverse://isaac-dev.ov.nvidia.com/NVIDIA/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
    ],
}

_GROUND_TEXTURE_PARAMS = {
    "textures": [
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Stone/Marble/Marble_BaseColor.png",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
    ],
    "texture_rotation": (0.0, 2 * math.pi),
}


@configclass
class RecordRandomizationEventCfg(EventCfg):
    """Event configuration with visual randomization for data recording."""

    # Initialize dome light at startup (runs once when environment is created)
    randomize_light_startup = EventTerm(
        func=randomize_dome_light,
        mode="startup",
        params=_DOME_LIGHT_PARAMS,
    )

    # Randomize dome light at intervals during simulation
    randomize_light = EventTerm(
        func=randomize_dome_light,
        mode="interval",
        interval_range_s=(1.0, 1.5),
        is_global_time=True,
        params=_DOME_LIGHT_PARAMS,
    )

    # Initialize ground texture at startup
    rand_ground_texture_startup = EventTerm(
        func=randomize_ground_texture,
        mode="startup",
        params=_GROUND_TEXTURE_PARAMS,
    )

    # Ground texture randomization at intervals (visual only, physics unchanged)
    rand_ground_texture = EventTerm(
        func=randomize_ground_texture,
        mode="interval",
        interval_range_s=(1.0, 1.5),
        is_global_time=True,
        params=_GROUND_TEXTURE_PARAMS,
    )
