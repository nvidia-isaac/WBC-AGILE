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

"""Unit tests for MDP event functions.

These tests use importlib to load fallen_state_cache directly, avoiding the
package import chain which requires Isaac Sim.
"""

import importlib.util
import os
import unittest

# Load fallen_state_cache module directly to avoid Isaac Sim dependencies
# in the parent package __init__.py files
_module_path = os.path.join(os.path.dirname(__file__), "..", "mdp", "events", "fallen_state_cache.py")
_spec = importlib.util.spec_from_file_location("fallen_state_cache", _module_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

compute_fallen_state_cache_key = _module.compute_fallen_state_cache_key
get_fallen_state_cache_path = _module.get_fallen_state_cache_path


class TestFallenStateDatasetCacheKey(unittest.TestCase):
    """Tests for fallen state dataset cache key computation."""

    def test_same_config_produces_same_key(self):
        """Same terrain config should produce the same cache key."""
        terrain_cfg = {"size": (8.0, 8.0), "seed": 42, "num_rows": 5, "num_cols": 10}

        key1 = compute_fallen_state_cache_key("StandUp-T1-v0", terrain_cfg)
        key2 = compute_fallen_state_cache_key("StandUp-T1-v0", terrain_cfg)

        self.assertEqual(key1, key2)

    def test_different_seed_produces_different_key(self):
        """Different seed should produce different cache key."""
        cfg1 = {"size": (8.0, 8.0), "seed": 42, "num_rows": 5, "num_cols": 10}
        cfg2 = {"size": (8.0, 8.0), "seed": 43, "num_rows": 5, "num_cols": 10}

        key1 = compute_fallen_state_cache_key("StandUp-T1-v0", cfg1)
        key2 = compute_fallen_state_cache_key("StandUp-T1-v0", cfg2)

        self.assertNotEqual(key1, key2)

    def test_different_size_produces_different_key(self):
        """Different terrain size should produce different cache key."""
        cfg1 = {"size": (8.0, 8.0), "seed": 42}
        cfg2 = {"size": (10.0, 10.0), "seed": 42}

        key1 = compute_fallen_state_cache_key("StandUp-T1-v0", cfg1)
        key2 = compute_fallen_state_cache_key("StandUp-T1-v0", cfg2)

        self.assertNotEqual(key1, key2)

    def test_different_task_produces_different_key(self):
        """Different task name should produce different cache key."""
        terrain_cfg = {"size": (8.0, 8.0), "seed": 42}

        key1 = compute_fallen_state_cache_key("StandUp-T1-v0", terrain_cfg)
        key2 = compute_fallen_state_cache_key("StandUp-G1-v0", terrain_cfg)

        self.assertNotEqual(key1, key2)

    def test_none_terrain_config_produces_default_key(self):
        """None terrain config should produce a key with 'default' hash."""
        key = compute_fallen_state_cache_key("StandUp-T1-v0", None)

        self.assertIn("default", key)

    def test_nested_config_changes_produce_different_key(self):
        """Changes in nested config (like sub_terrains) should produce different keys."""
        cfg1 = {
            "size": (8.0, 8.0),
            "seed": 42,
            "sub_terrains": {"flat": {"height": 0.0}, "rough": {"height": 0.1}},
        }
        cfg2 = {
            "size": (8.0, 8.0),
            "seed": 42,
            "sub_terrains": {"flat": {"height": 0.0}, "rough": {"height": 0.2}},  # Different height
        }

        key1 = compute_fallen_state_cache_key("StandUp-T1-v0", cfg1)
        key2 = compute_fallen_state_cache_key("StandUp-T1-v0", cfg2)

        self.assertNotEqual(key1, key2)

    def test_cache_key_contains_version(self):
        """Cache key should contain a version identifier."""
        key = compute_fallen_state_cache_key("StandUp-T1-v0", {"seed": 42})

        # Should contain version prefix (e.g., "fallen_states_v3_")
        self.assertTrue(key.startswith("fallen_states_v"))

    def test_cache_key_ends_with_pt(self):
        """Cache key should end with .pt extension."""
        key = compute_fallen_state_cache_key("StandUp-T1-v0", {"seed": 42})

        self.assertTrue(key.endswith(".pt"))


class TestGetFallenStateCachePath(unittest.TestCase):
    """Tests for cache path generation."""

    def test_cache_path_combines_dir_and_key(self):
        """Cache path should combine directory and key correctly."""
        cache_dir = "/tmp/fallen_states"
        cache_key = "fallen_states_v3_task_abc123.pt"

        path = get_fallen_state_cache_path(cache_dir, cache_key)

        self.assertEqual(path, "/tmp/fallen_states/fallen_states_v3_task_abc123.pt")

    def test_cache_path_handles_trailing_slash(self):
        """Cache path should work with or without trailing slash in dir."""
        cache_key = "test.pt"

        path1 = get_fallen_state_cache_path("/tmp/cache", cache_key)
        path2 = get_fallen_state_cache_path("/tmp/cache/", cache_key)

        # Both should produce valid paths (os.path.join handles this)
        self.assertTrue(path1.endswith("test.pt"))
        self.assertTrue(path2.endswith("test.pt"))


if __name__ == "__main__":
    unittest.main()
