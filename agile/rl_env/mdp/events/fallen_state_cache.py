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

"""Cache utilities for fallen state dataset.

These functions are in a separate file to allow unit testing without Isaac Sim dependencies.
"""

import hashlib
import json
import os

# Cache format version - increment this when the dataset format changes
# to invalidate old caches
_CACHE_VERSION = 3  # v3: Include full terrain config in cache key


def compute_fallen_state_cache_key(task_name: str, terrain_cfg: dict | None) -> str:
    """Compute a cache key for fallen states based on task name and terrain config.

    Args:
        task_name: Name of the task (e.g., 'Template-Isaac-Stand-Up-T1-v0')
        terrain_cfg: Terrain generator configuration dictionary (deeply nested)

    Returns:
        Cache key string suitable for filename
    """
    # Hash terrain config if available
    if terrain_cfg is not None:
        # Use JSON with sort_keys for deterministic serialization of nested dicts
        terrain_str = json.dumps(terrain_cfg, sort_keys=True, default=str)
        terrain_hash = hashlib.md5(terrain_str.encode()).hexdigest()[:8]
    else:
        terrain_hash = "default"

    # Clean task name for filename
    task_clean = task_name.replace("-", "_").replace(" ", "_")

    return f"fallen_states_v{_CACHE_VERSION}_{task_clean}_{terrain_hash}.pt"


def get_fallen_state_cache_path(cache_dir: str, cache_key: str) -> str:
    """Get the full path for a fallen state cache file.

    Args:
        cache_dir: Directory for cache files
        cache_key: Cache key from compute_fallen_state_cache_key

    Returns:
        Full path to cache file
    """
    return os.path.join(cache_dir, cache_key)
