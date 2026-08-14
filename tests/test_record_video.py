# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for efficient offscreen video recording."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from agile.isaaclab_extras.record_video import (
    EfficientRecordVideo,
    _offscreen_video_is_only_kit_consumer,
    _update_tracked_asset_recording_camera,
)


class _FakeVideoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    render_mode = "rgb_array"

    def __init__(self, *, fail_step: bool = False):
        settings = {"/isaaclab/video/enabled": True}
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Discrete(1)
        self.render_enabled = True
        self.video_recorder = SimpleNamespace(_backend="kit")
        self.scene = SimpleNamespace(_sensor_renderer_types=lambda: ())
        self.sim = SimpleNamespace(
            has_gui=lambda: False,
            visualizers=(),
            get_setting=settings.__getitem__,
            set_setting=settings.__setitem__,
        )
        self.has_rtx_sensors = True
        self.step_render_values: list[bool] = []
        self.step_video_enabled_values: list[bool] = []
        self.capture_render_values: list[bool] = []
        self.capture_video_enabled_values: list[bool] = []
        self._fail_step = fail_step

    def step(self, action):
        del action
        self.step_render_values.append(self.render_enabled)
        self.step_video_enabled_values.append(self.sim.get_setting("/isaaclab/video/enabled"))
        if self._fail_step:
            raise RuntimeError("step failed")
        return 0, 0.0, False, False, {}

    def render(self):
        self.capture_render_values.append(self.render_enabled)
        self.capture_video_enabled_values.append(self.sim.get_setting("/isaaclab/video/enabled"))
        return np.ones((2, 2, 3), dtype=np.uint8)


def _headless_launcher(**overrides):
    values = {"_headless": True, "_livestream": 0, "_xr": False}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_headless_kit_video_is_eligible_even_with_global_rtx_sensors():
    env = _FakeVideoEnv()

    assert env.has_rtx_sensors is True
    assert _offscreen_video_is_only_kit_consumer(env, _headless_launcher())


def test_non_rgb_render_mode_is_not_eligible():
    env = _FakeVideoEnv()
    env.render_mode = "human"

    assert not _offscreen_video_is_only_kit_consumer(env, _headless_launcher())


def test_non_kit_video_backend_is_not_eligible():
    env = _FakeVideoEnv()
    env.video_recorder._backend = "opencv"

    assert not _offscreen_video_is_only_kit_consumer(env, _headless_launcher())


def test_gui_simulation_is_not_eligible():
    env = _FakeVideoEnv()
    env.sim.has_gui = lambda: True

    assert not _offscreen_video_is_only_kit_consumer(env, _headless_launcher())


def test_non_headless_launch_is_not_eligible():
    env = _FakeVideoEnv()

    assert not _offscreen_video_is_only_kit_consumer(env, _headless_launcher(_headless=False))


def test_livestream_launch_is_not_eligible():
    env = _FakeVideoEnv()

    assert not _offscreen_video_is_only_kit_consumer(env, _headless_launcher(_livestream=1))


def test_xr_launch_is_not_eligible():
    env = _FakeVideoEnv()

    assert not _offscreen_video_is_only_kit_consumer(env, _headless_launcher(_xr=True))


def test_task_renderer_sensors_are_not_eligible():
    env = _FakeVideoEnv()
    env.scene._sensor_renderer_types = lambda: ("camera",)

    assert not _offscreen_video_is_only_kit_consumer(env, _headless_launcher())


def test_unknown_task_renderer_state_is_not_eligible():
    env = _FakeVideoEnv()
    env.scene._sensor_renderer_types = None

    assert not _offscreen_video_is_only_kit_consumer(env, _headless_launcher())


def test_visualizer_that_pumps_kit_is_not_eligible():
    env = _FakeVideoEnv()
    env.sim.visualizers = (SimpleNamespace(pumps_app_update=lambda: True),)

    assert not _offscreen_video_is_only_kit_consumer(env, _headless_launcher())


def test_unknown_visualizer_is_not_eligible():
    env = _FakeVideoEnv()
    env.sim.visualizers = (SimpleNamespace(),)

    assert not _offscreen_video_is_only_kit_consumer(env, _headless_launcher())


def test_standalone_visualizer_is_eligible():
    env = _FakeVideoEnv()
    env.sim.visualizers = (SimpleNamespace(pumps_app_update=lambda: False),)

    assert _offscreen_video_is_only_kit_consumer(env, _headless_launcher())


def test_wrapper_suppresses_only_the_underlying_step_and_restores_state(tmp_path):
    env = _FakeVideoEnv()
    wrapper = EfficientRecordVideo(
        env,
        app_launcher=_headless_launcher(),
        video_folder=str(tmp_path / "videos"),
        step_trigger=lambda step: step == 0,
        video_length=2,
    )

    wrapper.step(0)

    assert env.step_render_values == [False]
    assert env.step_video_enabled_values == [False]
    assert env.capture_render_values == [True]
    assert env.capture_video_enabled_values == [True]
    assert len(wrapper.recorded_frames) == 1
    assert env.render_enabled is True
    assert env.sim.get_setting("/isaaclab/video/enabled") is True
    wrapper.recorded_frames.clear()


def test_wrapper_restores_render_state_when_step_raises(tmp_path):
    env = _FakeVideoEnv(fail_step=True)
    wrapper = EfficientRecordVideo(
        env,
        app_launcher=_headless_launcher(),
        video_folder=str(tmp_path / "videos"),
        step_trigger=lambda step: step == 0,
        video_length=2,
    )

    with pytest.raises(RuntimeError, match="step failed"):
        wrapper.step(0)

    assert env.step_render_values == [False]
    assert env.step_video_enabled_values == [False]
    assert env.render_enabled is True
    assert env.sim.get_setting("/isaaclab/video/enabled") is True


def test_asset_tracking_video_camera_updates_kit_recorder_from_robot_root():
    calls = []
    root_pos = SimpleNamespace(torch=np.array([[10.0, 20.0, 0.5]]))
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            viewer=SimpleNamespace(
                origin_type="asset_root",
                asset_name="robot",
                env_index=0,
                eye=(-2.5, -5.0, 2.0),
                lookat=(0.0, 0.0, 0.75),
                cam_prim_path="/OmniverseKit_Persp",
            )
        ),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_pos_w=root_pos))},
        video_recorder=SimpleNamespace(
            _backend="kit",
            cfg=SimpleNamespace(eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0)),
            _capture=SimpleNamespace(
                cfg=SimpleNamespace(
                    eye=(7.5, 7.5, 7.5),
                    lookat=(0.0, 0.0, 0.0),
                    camera_prim_path="/OmniverseKit_Persp",
                )
            ),
        ),
        sim=SimpleNamespace(set_camera_view=lambda *, eye, target: calls.append(("sim", eye, target))),
    )

    updated = _update_tracked_asset_recording_camera(
        env, set_kit_camera_view=lambda path, *, eye, target: calls.append(("kit", path, eye, target))
    )

    expected_eye = (7.5, 15.0, 2.5)
    expected_target = (10.0, 20.0, 1.25)
    assert updated is True
    assert env.video_recorder.cfg.eye == expected_eye
    assert env.video_recorder.cfg.lookat == expected_target
    assert env.video_recorder._capture.cfg.eye == expected_eye
    assert env.video_recorder._capture.cfg.lookat == expected_target
    assert calls == [
        ("sim", expected_eye, expected_target),
        ("kit", "/OmniverseKit_Persp", expected_eye, expected_target),
    ]
