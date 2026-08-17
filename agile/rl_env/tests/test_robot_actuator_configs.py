# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab.actuators import ImplicitActuatorCfg

from agile.rl_env.assets.robots.booster_t1 import T1_DELAYED_DC_CFG
from agile.rl_env.assets.robots.unitree_g1 import (
    G1_29DOF,
    G1_29DOF_DELAYED_DC_MOTOR,
    G1_29DOF_HEIGHT_TRACKING,
    G1_W_HANDS_AGILE_CFG,
    G1_29DOF_BeyondMimic,
)
from agile.rl_env.mdp.actuators import DelayedDCMotorCfg, DelayedImplicitActuatorCfg

SHIPPED_ROBOT_CONFIGS = {
    "g1_locomotion": G1_29DOF,
    "g1_locomotion_dc": G1_29DOF_DELAYED_DC_MOTOR,
    "g1_height_tracking": G1_29DOF_HEIGHT_TRACKING,
    "g1_pick_place": G1_W_HANDS_AGILE_CFG,
    "t1": T1_DELAYED_DC_CFG,
}


@pytest.mark.parametrize(("config_name", "robot_cfg"), SHIPPED_ROBOT_CONFIGS.items())
def test_shipped_robot_configs_use_delayed_dc_motors(config_name, robot_cfg):
    for actuator_name, actuator_cfg in robot_cfg.actuators.items():
        assert isinstance(actuator_cfg, DelayedDCMotorCfg), f"{config_name}.{actuator_name}"
        assert actuator_cfg.effort_limit_sim is not None
        assert actuator_cfg.velocity_limit_sim is not None
        assert actuator_cfg.saturation_effort is not None


@pytest.mark.parametrize("robot_cfg", [G1_29DOF, G1_W_HANDS_AGILE_CFG])
def test_converted_g1_body_actuators_randomize_delay(robot_cfg):
    for actuator_name, actuator_cfg in robot_cfg.actuators.items():
        if actuator_name == "hands":
            continue
        assert actuator_cfg.min_delay == 0
        assert actuator_cfg.max_delay == 4


def test_dex3_hand_actuators_have_zero_delay():
    hand_cfg = G1_W_HANDS_AGILE_CFG.actuators["hands"]
    assert hand_cfg.min_delay == 0
    assert hand_cfg.max_delay == 0


@pytest.mark.parametrize(
    ("robot_cfg", "expected"),
    [
        (G1_29DOF, {"legs": 180.0, "feet": 80.0, "waist": 120.0, "arms": 40.0}),
        (
            G1_W_HANDS_AGILE_CFG,
            {
                "legs": 180.0,
                "feet": 80.0,
                "waist": 120.0,
                "left_arms": 40.0,
                "right_arms": 40.0,
                "hands": 0.76,
            },
        ),
    ],
)
def test_converted_g1_actuators_use_group_saturation_efforts(robot_cfg, expected):
    assert {name: cfg.saturation_effort for name, cfg in robot_cfg.actuators.items()} == expected


def test_motion_tracking_uses_pre_dc_implicit_actuators():
    for actuator_name in ("legs", "feet", "waist", "waist_yaw"):
        actuator_cfg = G1_29DOF_BeyondMimic.actuators[actuator_name]
        assert isinstance(actuator_cfg, DelayedImplicitActuatorCfg), actuator_name
        assert actuator_cfg.min_delay == 0
        assert actuator_cfg.max_delay == 0

    assert type(G1_29DOF_BeyondMimic.actuators["arms"]) is ImplicitActuatorCfg


def _assert_gains(actuator_cfg, stiffness, damping):
    assert actuator_cfg.stiffness == pytest.approx(stiffness)
    assert actuator_cfg.damping == pytest.approx(damping)


def test_generic_g1_locomotion_uses_reference_dc_gains():
    for actuator_name, actuator_cfg in G1_29DOF.actuators.items():
        reference_cfg = G1_29DOF_DELAYED_DC_MOTOR.actuators[actuator_name]
        _assert_gains(actuator_cfg, reference_cfg.stiffness, reference_cfg.damping)


@pytest.mark.parametrize(
    ("actuator_name", "stiffness", "damping"),
    [
        (
            "legs",
            {
                ".*_hip_pitch_joint": 40.17923847137318,
                ".*_hip_roll_joint": 99.09842777666113,
                ".*_hip_yaw_joint": 40.17923847137318,
                ".*_knee_joint": 99.09842777666113,
            },
            {
                ".*_hip_pitch_joint": 2.5578897650279457,
                ".*_hip_roll_joint": 6.3088018534966395,
                ".*_hip_yaw_joint": 2.5578897650279457,
                ".*_knee_joint": 6.3088018534966395,
            },
        ),
        ("feet", 28.50124619574858, 1.814445686584846),
        ("waist", 28.50124619574858, 1.814445686584846),
        ("waist_yaw", 40.17923847137318, 2.5578897650279457),
        (
            "arms",
            {
                ".*_shoulder_pitch_joint": 14.25062309787429,
                ".*_shoulder_roll_joint": 14.25062309787429,
                ".*_shoulder_yaw_joint": 14.25062309787429,
                ".*_elbow_joint": 14.25062309787429,
                ".*_wrist_roll_joint": 14.25062309787429,
                ".*_wrist_pitch_joint": 16.77832748089279,
                ".*_wrist_yaw_joint": 16.77832748089279,
            },
            {
                ".*_shoulder_pitch_joint": 0.907222843292423,
                ".*_shoulder_roll_joint": 0.907222843292423,
                ".*_shoulder_yaw_joint": 0.907222843292423,
                ".*_elbow_joint": 0.907222843292423,
                ".*_wrist_roll_joint": 0.907222843292423,
                ".*_wrist_pitch_joint": 1.06814150219,
                ".*_wrist_yaw_joint": 1.06814150219,
            },
        ),
    ],
)
def test_beyond_mimic_retains_system_identified_gains(actuator_name, stiffness, damping):
    _assert_gains(G1_29DOF_BeyondMimic.actuators[actuator_name], stiffness, damping)


@pytest.mark.parametrize(
    ("actuator_name", "stiffness", "damping"),
    [
        (
            "waist",
            {"waist_yaw_joint": 300.0, "waist_roll_joint": 300.0, "waist_pitch_joint": 300.0},
            {"waist_yaw_joint": 5.0, "waist_roll_joint": 5.0, "waist_pitch_joint": 5.0},
        ),
        (
            "left_arms",
            dict.fromkeys(G1_W_HANDS_AGILE_CFG.actuators["left_arms"].joint_names_expr, 100.0),
            {
                "left_shoulder_pitch_joint": 2.0,
                "left_shoulder_roll_joint": 1.0,
                "left_shoulder_yaw_joint": 0.4,
                "left_elbow_joint": 1.0,
                "left_wrist_roll_joint": 0.2,
                "left_wrist_pitch_joint": 0.2,
                "left_wrist_yaw_joint": 0.2,
            },
        ),
        (
            "right_arms",
            dict.fromkeys(G1_W_HANDS_AGILE_CFG.actuators["right_arms"].joint_names_expr, 100.0),
            {
                "right_shoulder_pitch_joint": 2.0,
                "right_shoulder_roll_joint": 1.0,
                "right_shoulder_yaw_joint": 0.4,
                "right_elbow_joint": 1.0,
                "right_wrist_roll_joint": 0.2,
                "right_wrist_pitch_joint": 0.2,
                "right_wrist_yaw_joint": 0.2,
            },
        ),
        ("hands", 2.0, 0.2),
    ],
)
def test_pick_place_retains_task_specific_upper_body_gains(actuator_name, stiffness, damping):
    _assert_gains(G1_W_HANDS_AGILE_CFG.actuators[actuator_name], stiffness, damping)


@pytest.mark.parametrize("actuator_name", ["legs", "feet"])
def test_pick_place_retains_system_identified_lower_body_gains(actuator_name):
    expected_cfg = G1_29DOF_BeyondMimic.actuators[actuator_name]
    _assert_gains(
        G1_W_HANDS_AGILE_CFG.actuators[actuator_name],
        expected_cfg.stiffness,
        expected_cfg.damping,
    )
