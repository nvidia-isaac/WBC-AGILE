# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import gymnasium as gym


def _tuple3(values: Any) -> tuple[float, float, float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return (float(values[0]), float(values[1]), float(values[2]))


def _set_kit_camera_view(
    camera_prim_path: str, *, eye: tuple[float, float, float], target: tuple[float, float, float]
) -> None:
    from isaacsim.core.rendering_manager import ViewportManager

    ViewportManager.set_camera_view(camera_prim_path, eye=list(eye), target=list(target))


def _update_tracked_asset_recording_camera(
    env: gym.Env | Any,
    *,
    set_kit_camera_view: Any = _set_kit_camera_view,
) -> bool:
    """Synchronize Isaac Lab's fixed Kit recorder camera with an asset-root viewer.

    Isaac Lab's ``VideoRecorder`` copies only ``viewer.eye`` and ``viewer.lookat`` as world-space
    coordinates when the recorder is constructed.  It does not apply ``viewer.origin_type`` to the
    recorded MP4 camera.  For generated terrains, environment 0 is often far from the world origin,
    so the default recording shows mostly terrain.  This helper computes the world-space camera pose
    from the tracked asset root before each render.
    """
    unwrapped = getattr(env, "unwrapped", env)
    viewer_cfg = getattr(getattr(unwrapped, "cfg", None), "viewer", None)
    recorder = getattr(unwrapped, "video_recorder", None)
    if viewer_cfg is None or recorder is None:
        return False
    if getattr(viewer_cfg, "origin_type", None) != "asset_root":
        return False
    if getattr(recorder, "_backend", None) != "kit":
        return False

    asset_name = getattr(viewer_cfg, "asset_name", None)
    if asset_name is None:
        return False

    scene = getattr(unwrapped, "scene", None)
    try:
        asset = scene[asset_name]
        root_positions = asset.data.root_pos_w.torch
        origin = _tuple3(root_positions[getattr(viewer_cfg, "env_index", 0)])
    except (AttributeError, KeyError, IndexError, TypeError):
        return False

    eye_offset = _tuple3(viewer_cfg.eye)
    lookat_offset = _tuple3(viewer_cfg.lookat)
    eye = tuple(origin[i] + eye_offset[i] for i in range(3))
    target = tuple(origin[i] + lookat_offset[i] for i in range(3))

    cfg = getattr(recorder, "cfg", None)
    if cfg is not None:
        cfg.eye = eye
        cfg.lookat = target
    capture_cfg = getattr(getattr(recorder, "_capture", None), "cfg", None)
    if capture_cfg is not None:
        capture_cfg.eye = eye
        capture_cfg.lookat = target

    sim = getattr(unwrapped, "sim", None)
    if sim is not None and hasattr(sim, "set_camera_view"):
        sim.set_camera_view(eye=eye, target=target)

    camera_prim_path = getattr(capture_cfg, "camera_prim_path", None) or getattr(viewer_cfg, "cam_prim_path", None)
    if camera_prim_path is not None:
        set_kit_camera_view(camera_prim_path, eye=eye, target=target)
    return True


def _offscreen_video_is_only_kit_consumer(env: gym.Env, app_launcher: Any) -> bool:
    unwrapped = env.unwrapped
    recorder = getattr(unwrapped, "video_recorder", None)
    if getattr(unwrapped, "render_mode", None) != "rgb_array":
        return False
    if getattr(recorder, "_backend", None) != "kit":
        return False
    if not getattr(app_launcher, "_headless", False):
        return False
    if getattr(app_launcher, "_livestream", 1) != 0 or getattr(app_launcher, "_xr", True):
        return False

    sim = unwrapped.sim
    has_gui = getattr(sim, "has_gui", False)
    has_gui_enabled = has_gui() if callable(has_gui) else has_gui
    if has_gui_enabled:
        return False

    sensor_renderer_types = getattr(unwrapped.scene, "_sensor_renderer_types", None)
    if sensor_renderer_types is None or sensor_renderer_types():
        return False

    for visualizer in getattr(sim, "visualizers", ()):
        pumps_app_update = getattr(visualizer, "pumps_app_update", None)
        if pumps_app_update is None or pumps_app_update():
            return False
    return True


class _SuppressStepKitPumping(gym.Wrapper):
    """Suppress Kit pumping only while delegating an environment step."""

    def step(self, action: Any):
        unwrapped = self.env.unwrapped
        previous_render_enabled = unwrapped.render_enabled
        video_enabled_setting = "/isaaclab/video/enabled"
        previous_video_enabled = unwrapped.sim.get_setting(video_enabled_setting)
        unwrapped.render_enabled = False
        unwrapped.sim.set_setting(video_enabled_setting, False)
        try:
            return super().step(action)
        finally:
            unwrapped.render_enabled = previous_render_enabled
            unwrapped.sim.set_setting(video_enabled_setting, previous_video_enabled)


class _TrackAssetRootKitRecordingCamera(gym.Wrapper):
    """Keep Isaac Lab's fixed Kit video recorder framed on the configured asset root."""

    def render(self):
        _update_tracked_asset_recording_camera(self.env)
        return super().render()


class EfficientRecordVideo(gym.wrappers.RecordVideo):
    """Record video while avoiding a redundant Kit pump inside eligible environment steps."""

    def __init__(self, env: gym.Env, *args: Any, app_launcher: Any, **kwargs: Any):
        env = _TrackAssetRootKitRecordingCamera(env)
        if _offscreen_video_is_only_kit_consumer(env, app_launcher):
            env = _SuppressStepKitPumping(env)
        super().__init__(env, *args, **kwargs)
