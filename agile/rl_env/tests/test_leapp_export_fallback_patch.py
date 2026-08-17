# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from agile.isaaclab_extras.leapp_export_fallback_patch import (
    build_command_element_names,
    build_command_observation_input,
    build_processed_action_fallback,
)


def test_processed_action_fallback_skips_zero_dim_actions() -> None:
    fallback = build_processed_action_fallback(
        term_name="lift",
        term=SimpleNamespace(action_dim=0),
        processed=torch.empty(0),
        scene_key="robot",
        select_element_names=lambda names, ids: names,
        build_write_connection=lambda entity, method: {"isaaclab_connection": f"write:{entity}:{method}"},
    )

    assert fallback is None


def test_processed_action_fallback_marks_joint_targets_semantically() -> None:
    term = SimpleNamespace(
        action_dim=2,
        _joint_ids=[1, 0],
        _asset=SimpleNamespace(joint_names=["left_hip", "right_hip"]),
    )

    fallback = build_processed_action_fallback(
        term_name="joint_pos",
        term=term,
        processed=torch.tensor([[0.1, -0.2]]),
        scene_key="robot",
        select_element_names=lambda names, ids: [names[index] for index in ids],
        build_write_connection=lambda entity, method: {"isaaclab_connection": f"write:{entity}:{method}"},
    )

    assert fallback is not None
    assert fallback["name"] == "joint_pos"
    assert torch.equal(fallback["ref"], torch.tensor([[0.1, -0.2]]))
    assert fallback["kind"] == "target/joint/position"
    assert fallback["element_names"] == ["right_hip", "left_hip"]
    assert fallback["extra"] == {"isaaclab_connection": "write:robot:set_joint_position_target_index"}


def test_command_observation_input_marks_motion_anchor_ori_as_command_input() -> None:
    result = build_command_observation_input(
        func_name="motion_anchor_ori_b",
        result=torch.zeros(1, 6),
        command_name="motion",
        build_command_connection=lambda name: {"isaaclab_connection": f"command:{name}"},
    )

    assert result is not None
    assert result["name"] == "motion_anchor_ori_b"
    assert torch.equal(result["ref"], torch.zeros(1, 6))
    assert result["kind"] is None
    assert result["element_names"] is None
    assert result["extra"] == {"isaaclab_connection": "command:motion_anchor_ori_b"}


def test_command_observation_input_ignores_regular_observations() -> None:
    result = build_command_observation_input(
        func_name="base_ang_vel",
        result=torch.zeros(1, 3),
        command_name="motion",
        build_command_connection=lambda name: {"isaaclab_connection": f"command:{name}"},
    )

    assert result is None


def test_command_element_names_prefers_configured_names() -> None:
    names = build_command_element_names(
        command_term=SimpleNamespace(command_names=["ignored"]),
        command_cfg=SimpleNamespace(element_names=["lin_x", "lin_y", "yaw"]),
        result=torch.zeros(1, 3),
    )

    assert names == ["lin_x", "lin_y", "yaw"]


def test_command_element_names_uses_term_command_names() -> None:
    names = build_command_element_names(
        command_term=SimpleNamespace(command_names=["lin_x", "lin_y", "yaw"]),
        command_cfg=SimpleNamespace(element_names=None),
        result=torch.zeros(1, 3),
    )

    assert names == ["lin_x", "lin_y", "yaw"]


def test_command_element_names_derives_motion_reference_names() -> None:
    command_term = SimpleNamespace(
        motion=object(),
        robot=SimpleNamespace(data=SimpleNamespace(joint_names=["left_hip", "right_hip"])),
    )

    names = build_command_element_names(
        command_term=command_term,
        command_cfg=SimpleNamespace(element_names=None),
        result=torch.zeros(1, 4),
    )

    assert names == [
        "ref_joint_pos/left_hip",
        "ref_joint_pos/right_hip",
        "ref_joint_vel/left_hip",
        "ref_joint_vel/right_hip",
    ]
