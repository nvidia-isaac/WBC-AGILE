from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from agile.rl_env.mdp.actions.actions_cfg import RandomActionCfg
from agile.rl_env.mdp.actions.random_actions import RandomPositionAction
from agile.rl_env.mdp.actions.velocity_profiles import EMAVelocityProfileCfg, LinearVelocityProfileCfg


def make_action(velocity_profile_cfg=None):
    num_envs, num_joints = 3, 2
    default = torch.tensor([[0.1, -0.2]]).repeat(num_envs, 1)
    asset = MagicMock()
    asset.device = torch.device("cpu")
    asset.num_joints = num_joints
    asset.joint_names = ["j0", "j1"]
    asset.find_joints.return_value = ([0, 1], ["j0", "j1"])
    asset.data.default_joint_pos = SimpleNamespace(torch=default)
    asset.data.joint_pos_limits = SimpleNamespace(
        torch=torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]).repeat(num_envs, 1, 1)
    )
    asset.data.joint_vel_limits = SimpleNamespace(torch=torch.ones(num_envs, num_joints))
    command = SimpleNamespace(command=torch.zeros(num_envs, 4))
    env = MagicMock()
    env.num_envs = num_envs
    env.device = "cpu"
    env.step_dt = 0.02
    env.scene = {"robot": asset}
    env.command_manager.get_term.return_value = command
    if velocity_profile_cfg is None:
        velocity_profile_cfg = EMAVelocityProfileCfg(ema_coefficient_range=(0.25, 0.25))
    cfg = RandomActionCfg(
        asset_name="robot",
        joint_names=["j0", "j1"],
        sample_range=(0.1, 1.5),
        no_random_when_walking=True,
        command_name="base_velocity",
        velocity_profile_cfg=velocity_profile_cfg,
    )
    return RandomPositionAction(cfg, env), asset, command, env


def test_pre_resample_output_is_default_pose():
    action, _, _, _ = make_action()
    action._time_to_resample_sample.fill_(100.0)
    action.process_actions(torch.empty(3, 0))
    torch.testing.assert_close(action.processed_actions, action._offset)


def test_command_term_is_looked_up_only_during_construction():
    action, _, _, env = make_action()
    action._time_to_resample_sample.fill_(100.0)
    action.process_actions(torch.empty(3, 0))
    action.process_actions(torch.empty(3, 0))
    env.command_manager.get_term.assert_called_once_with("base_velocity")


def test_apply_writes_once_per_process_step():
    action, asset, _, _ = make_action()
    action._time_to_resample_sample.fill_(100.0)
    action.process_actions(torch.empty(3, 0))
    for _ in range(4):
        action.apply_actions()
    assert asset.set_joint_position_target.call_count == 1
    action.process_actions(torch.empty(3, 0))
    for _ in range(4):
        action.apply_actions()
    assert asset.set_joint_position_target.call_count == 2


def test_partial_reset_restores_only_selected_ema_state_and_rearms_walking_transition(mocker):
    action, _, command, _ = make_action()
    profile = action._velocity_profile
    env_ids = torch.tensor([1])

    action._raw_actions = torch.full((3, 1), 9.0)
    action._processed_actions.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    action._target_processed_actions.copy_(action._processed_actions + 10.0)
    action._time_since_last_sample.copy_(torch.tensor([0.4, 0.5, 0.6]))
    action._time_to_resample_sample.copy_(torch.tensor([100.0, 101.0, 102.0]))
    action._previous_is_walking.fill_(True)
    action._target_write_pending = False

    position_state_names = (
        "_current_position",
        "_previous_position",
        "_initial_position",
        "_target_position",
    )
    for index, name in enumerate(position_state_names):
        getattr(profile, name).copy_(torch.arange(6, dtype=torch.float32).reshape(3, 2) + 10.0 * index)
    profile._current_velocity.fill_(7.0)
    profile._velocity_scale.fill_(0.75)
    profile._is_active.fill_(True)

    action_state_before = {
        name: getattr(action, name).clone()
        for name in (
            "_raw_actions",
            "_processed_actions",
            "_target_processed_actions",
            "_time_since_last_sample",
            "_time_to_resample_sample",
        )
    }
    profile_state_before = {
        name: getattr(profile, name).clone()
        for name in (*position_state_names, "_current_velocity", "_velocity_scale", "_is_active")
    }

    with patch("torch.rand", return_value=torch.tensor([0.5])) as rand:
        action.reset(env_ids)

    expected = action_state_before["_raw_actions"].clone()
    expected[env_ids] = 0.0
    torch.testing.assert_close(action._raw_actions, expected)
    for name in ("_processed_actions", "_target_processed_actions"):
        expected = action_state_before[name].clone()
        expected[env_ids] = action._offset[env_ids]
        torch.testing.assert_close(getattr(action, name), expected)
    for name in position_state_names:
        expected = profile_state_before[name].clone()
        expected[env_ids] = action._offset[env_ids]
        torch.testing.assert_close(getattr(profile, name), expected)
    for name in ("_current_velocity", "_velocity_scale"):
        expected = profile_state_before[name].clone()
        expected[env_ids] = 0.0
        torch.testing.assert_close(getattr(profile, name), expected)
    expected_active = profile_state_before["_is_active"].clone()
    expected_active[env_ids] = False
    assert torch.equal(profile._is_active, expected_active)
    assert torch.equal(action._previous_is_walking, torch.tensor([True, False, True]))
    torch.testing.assert_close(action._time_since_last_sample, torch.tensor([0.4, 0.0, 0.6]))
    torch.testing.assert_close(action._time_to_resample_sample, torch.tensor([100.0, 0.8, 102.0]))
    rand.assert_called_once()
    assert action._target_write_pending

    redirect = mocker.spy(profile, "redirect_target")
    command.command[:, 0] = 1.0
    action.process_actions(torch.empty(3, 0))
    assert torch.equal(redirect.call_args.kwargs["env_mask"], torch.tensor([False, True, False]))


def test_non_ema_reset_keeps_existing_randomization_state():
    action, _, _, _ = make_action(LinearVelocityProfileCfg())
    action._raw_actions = torch.full((3, 1), 9.0)
    action._processed_actions.fill_(2.0)
    action._target_processed_actions.fill_(3.0)
    action._time_since_last_sample.fill_(4.0)
    action._time_to_resample_sample.fill_(5.0)
    action._previous_is_walking.fill_(True)
    action._target_write_pending = False
    action_state_before = {
        name: value.clone()
        for name, value in vars(action).items()
        if isinstance(value, torch.Tensor) and name != "_raw_actions"
    }
    profile_state_before = {
        name: value.clone() for name, value in vars(action._velocity_profile).items() if isinstance(value, torch.Tensor)
    }

    action.reset(torch.tensor([1]))

    torch.testing.assert_close(action._raw_actions, torch.tensor([[9.0], [0.0], [9.0]]))
    for name, expected in action_state_before.items():
        torch.testing.assert_close(getattr(action, name), expected)
    for name, expected in profile_state_before.items():
        torch.testing.assert_close(getattr(action._velocity_profile, name), expected)
    assert not action._target_write_pending


def test_standing_to_walking_redirects_only_transitions(mocker):
    action, _, command, _ = make_action()
    action._time_to_resample_sample.fill_(100.0)
    action.process_actions(torch.empty(3, 0))
    redirect = mocker.spy(action._velocity_profile, "redirect_target")
    command.command[[0, 2], 0] = 1.0
    action.process_actions(torch.empty(3, 0))
    mask = redirect.call_args.kwargs["env_mask"]
    assert torch.equal(mask, torch.tensor([True, False, True]))
    torch.testing.assert_close(redirect.call_args.args[0], action._offset)


def test_steady_walking_does_not_redirect_or_draw_random_values(mocker):
    action, _, command, _ = make_action()
    action._time_to_resample_sample.fill_(100.0)
    command.command[:, 0] = 1.0
    action.process_actions(torch.empty(3, 0))
    redirect = mocker.spy(action._velocity_profile, "redirect_target")
    sample_scales = mocker.spy(action._velocity_profile, "_sample_velocity_scales")
    targets_before = action._target_processed_actions.clone()
    with patch("torch.rand", wraps=torch.rand) as rand:
        action.process_actions(torch.empty(3, 0))
    redirect.assert_called_once()
    assert torch.equal(redirect.call_args.kwargs["env_mask"], torch.zeros(3, dtype=torch.bool))
    torch.testing.assert_close(redirect.call_args.args[0], action._offset)
    sample_scales.assert_not_called()
    rand.assert_not_called()
    torch.testing.assert_close(action._target_processed_actions, targets_before)


def test_walking_environments_are_excluded_from_scheduled_resampling(mocker):
    action, _, command, _ = make_action()
    command.command[[0, 2], 0] = 1.0
    action._time_since_last_sample.fill_(2.0)
    action._time_to_resample_sample.fill_(1.0)
    deadlines_before = action._time_to_resample_sample.clone()
    set_target = mocker.spy(action._velocity_profile, "set_target")
    action.process_actions(torch.empty(3, 0))
    assert torch.equal(set_target.call_args.kwargs["env_ids"], torch.tensor([1]))
    torch.testing.assert_close(action._time_since_last_sample[[0, 2]], torch.full((2,), 2.02))
    torch.testing.assert_close(action._time_to_resample_sample[[0, 2]], deadlines_before[[0, 2]])
    assert action._time_since_last_sample[1] == 0.0


def test_return_to_standing_resamples_an_overdue_environment(mocker):
    action, _, command, _ = make_action()
    command.command[0, 0] = 1.0
    action._time_since_last_sample[0] = 2.0
    action._time_to_resample_sample[0] = 1.0
    action.process_actions(torch.empty(3, 0))
    set_target = mocker.spy(action._velocity_profile, "set_target")
    command.command[0, 0] = 0.0
    action.process_actions(torch.empty(3, 0))
    assert 0 in set_target.call_args.kwargs["env_ids"].tolist()
