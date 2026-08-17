# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from scripts import sim2mujoco_eval


def test_explicit_duration_overrides_evaluation_scenario() -> None:
    scenario = SimpleNamespace(episode_length_s=100.0)
    assert sim2mujoco_eval._resolve_duration(10.0, scenario) == 10.0


def test_scenario_duration_is_used_when_cli_duration_is_omitted() -> None:
    scenario = SimpleNamespace(episode_length_s=100.0)
    assert sim2mujoco_eval._resolve_duration(None, scenario) == 100.0


def test_default_duration_is_used_without_cli_or_scenario() -> None:
    assert sim2mujoco_eval._resolve_duration(None, None) == 10.0


def test_reference_motion_initial_state_requires_motion_tracker() -> None:
    class ProviderWithoutTracker:
        pass

    try:
        sim2mujoco_eval._resolve_initial_state("reference_motion", ProviderWithoutTracker(), ["j0"])
    except ValueError as exc:
        assert "reference_motion" in str(exc)
    else:
        raise AssertionError("expected reference_motion without a motion tracker to fail")
