# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless frame capture from MuJocoSimulation."""

import numpy as np
import pytest

pytestmark = pytest.mark.e2e  # mujoco offscreen render needs a GL backend (EGL/osmesa)

_MJCF = """
<mujoco>
  <worldbody>
    <body name="base"><freejoint/><geom type="sphere" size="0.2" mass="1"/></body>
  </worldbody>
</mujoco>
"""


def test_capture_frame_returns_rgb(tmp_path):
    from agile.sim2mujoco.simulation import MuJocoSimulation

    mjcf = tmp_path / "ball.xml"
    mjcf.write_text(_MJCF)
    cfg = {
        "scene": {"physics_dt": 0.005, "decimation": 1},
        "articulations": {
            "robot": {
                "joint_names": [],
                "default_joint_pos": [],
                "default_joint_stiffness": [],
                "default_joint_damping": [],
            }
        },
    }
    sim = MuJocoSimulation(cfg, "cpu", enable_viewer=False, mjcf_path=str(mjcf))
    # Default size (1280x720) is larger than MuJoCo's 640 default offscreen framebuffer, so this
    # exercises the framebuffer enlargement (a smaller size would not).
    frame = sim.capture_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (720, 1280, 3)
    assert frame.dtype == np.uint8
