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

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .hf_terrains_cfg import HfRandomUniformTerrainDifficultyCfg

ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=100.0,
    num_rows=12,  # num different difficulties
    num_cols=36,  # num terrains per same difficulty level
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.01,
            grid_width=0.45,
            grid_height_range=(0.01, 0.15),
            platform_width=0.1,
        ),
        "boxes_small": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1,
            grid_width=0.15,
            grid_height_range=(0.01, 0.15),
            platform_width=0.1,
        ),
        "random_rough": HfRandomUniformTerrainDifficultyCfg(
            proportion=0.05, noise_range=(0.01, 0.1), noise_step=0.02, border_width=0.25
        ),
        "random_rough_small": HfRandomUniformTerrainDifficultyCfg(
            proportion=0.1,
            noise_range=(0.01, 0.05),
            noise_step=0.04,
            border_width=0.25,
        ),
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.1,
            rail_thickness_range=(0.01, 0.2),
            rail_height_range=(0.01, 0.2),
            platform_width=2.0,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.01, 0.15),
            step_width=0.5,
            platform_width=1.0,
        ),
        "inverted_pyramid_stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.01, 0.15),
            step_width=0.5,
            platform_width=1.0,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.2, amplitude_range=(0.01, 1.0), num_waves=1),
        "wave_small": terrain_gen.HfWaveTerrainCfg(proportion=0.2, amplitude_range=(0.01, 0.5), num_waves=3),
    },
)


LESS_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=100.0,
    num_rows=20,  # num different difficulties
    num_cols=16,  # num terrains per same difficulty level
    horizontal_scale=0.4,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes_small": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.15,
            grid_height_range=(0.0, 0.02),
            platform_width=0.1,
        ),
        "random_rough_small": HfRandomUniformTerrainDifficultyCfg(
            proportion=1.0,
            noise_range=(0.01, 0.1),
            noise_step=0.1,
            border_width=0.4,
        ),
    },
)

MEDIUM_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=100.0,
    num_rows=20,  # num different difficulties
    num_cols=15,  # num terrains per same difficulty level
    horizontal_scale=0.4,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes_small": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.15,
            grid_height_range=(0.0, 0.04),
            platform_width=0.1,
        ),
        "random_rough_small": HfRandomUniformTerrainDifficultyCfg(
            proportion=0.2,
            noise_range=(0.01, 0.2),
            noise_step=0.1,
            border_width=0.4,
        ),
        "wave_small": terrain_gen.HfWaveTerrainCfg(proportion=0.2, amplitude_range=(0.01, 0.25), num_waves=3),
    },
)


STAND_UP_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=100.0,
    num_rows=20,  # num different difficulties
    num_cols=16,  # num terrains per same difficulty level
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough_small": HfRandomUniformTerrainDifficultyCfg(
            proportion=1.0,
            noise_range=(0.01, 0.05),
            noise_step=0.02,
            border_width=0.4,
        ),
    },
)
