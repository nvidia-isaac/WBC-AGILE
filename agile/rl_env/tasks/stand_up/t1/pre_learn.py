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

"""Pre-learn hook for stand-up task to collect fallen states before training."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agile.rl_env.mdp.events import (
    FallenStateDataset,
    FallenStateDatasetCfg,
    compute_fallen_state_cache_key,
    get_fallen_state_cache_path,
    reset_from_fallen_dataset,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pre_learn(env: ManagerBasedRLEnv, task_name: str, agent_cfg) -> None:
    """Pre-learn hook called before training starts.

    This function sets up the fallen state dataset for efficient resets.
    It pre-collects diverse fallen poses so episodes can reset instantly
    instead of simulating 5 seconds of falling each time.

    Args:
        env: The unwrapped environment instance.
        task_name: Name of the task (e.g., 'StandUp-T1-v0').
        agent_cfg: The agent configuration object.
    """
    # Check if fallen state dataset is configured
    dataset_cfg: FallenStateDatasetCfg | None = getattr(agent_cfg, "fallen_state_dataset_cfg", None)
    if dataset_cfg is None:
        return

    # Get the reset_from_fallen_dataset event term
    reset_event = _get_reset_event(env)

    # Create dataset
    dataset = FallenStateDataset(cfg=dataset_cfg)

    # Try to load from cache
    if dataset_cfg.cache_enabled:
        cache_path = _get_cache_path(env, task_name, dataset_cfg)

        if dataset.load(cache_path):
            logger.info(f"Loaded fallen state dataset from cache: {cache_path}")
            reset_event.set_dataset(dataset)
            return

        logger.info(f"No valid cache found at {cache_path}, will collect new dataset.")

    # Collect new dataset
    logger.info("Starting fallen state collection...")
    dataset.collect(env, verbose=True)

    # Save to cache
    if dataset_cfg.cache_enabled:
        cache_path = _get_cache_path(env, task_name, dataset_cfg)
        dataset.save(cache_path)
        logger.info(f"Saved fallen state dataset to cache: {cache_path}")

    # Inject dataset into reset event
    reset_event.set_dataset(dataset)


def _get_reset_event(env: ManagerBasedRLEnv) -> reset_from_fallen_dataset:
    """Get the reset_from_fallen_dataset event term from the environment.

    Raises:
        AssertionError: If the event term is not found.
    """
    event_manager = env.event_manager

    if "reset" in event_manager.active_terms:
        for term_name in event_manager.active_terms["reset"]:
            term_cfg = event_manager.get_term_cfg(term_name)
            # For class-based terms, term_cfg.func is the instance
            if isinstance(term_cfg.func, reset_from_fallen_dataset):
                return term_cfg.func

    raise AssertionError(
        "No reset_from_fallen_dataset event term found in environment. "
        "Add EventTerm(func=mdp.reset_from_fallen_dataset, ...) to your EventsCfg."
    )


def _get_cache_path(env: ManagerBasedRLEnv, task_name: str, dataset_cfg: FallenStateDatasetCfg) -> str:
    """Compute the cache path for fallen states."""
    terrain_cfg = None
    if hasattr(env.scene, "terrain") and env.scene.terrain is not None:
        terrain_gen = env.scene.terrain.cfg.terrain_generator
        if terrain_gen is not None:
            # Use full terrain config for cache key to ensure invalidation on any change
            # This includes all sub_terrain parameters (roughness, height, etc.)
            terrain_cfg = _serialize_terrain_config(terrain_gen)

    cache_key = compute_fallen_state_cache_key(task_name, terrain_cfg)
    return get_fallen_state_cache_path(dataset_cfg.cache_dir, cache_key)


def _serialize_terrain_config(terrain_gen) -> dict:
    """Serialize terrain generator config for deterministic cache key computation.

    Converts the terrain config to a dictionary, filtering out non-serializable
    fields (like class references) while preserving all parameter values that
    affect terrain generation.
    """
    # Use configclass's to_dict() for deep serialization
    terrain_dict = terrain_gen.to_dict()

    # Filter out non-serializable fields that don't affect terrain geometry
    keys_to_remove = ["class_type", "use_cache", "cache_dir"]
    for key in keys_to_remove:
        terrain_dict.pop(key, None)

    # Also filter class_type from sub_terrains
    if "sub_terrains" in terrain_dict and terrain_dict["sub_terrains"]:
        for sub_terrain_cfg in terrain_dict["sub_terrains"].values():
            if isinstance(sub_terrain_cfg, dict):
                sub_terrain_cfg.pop("class_type", None)
                sub_terrain_cfg.pop("function", None)

    return terrain_dict
