import torch

from agile.rl_env.mdp.actions.velocity_profiles import EMAVelocityProfile, EMAVelocityProfileCfg


def make_profile(num_envs: int = 3, num_joints: int = 2) -> EMAVelocityProfile:
    cfg = EMAVelocityProfileCfg(
        ema_coefficient_range=(0.25, 0.25),
        position_tolerance=1.0e-4,
        velocity_tolerance=1.0e-4,
    )
    limits = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]).repeat(num_envs, 1, 1)
    return EMAVelocityProfile(
        cfg,
        num_envs=num_envs,
        num_joints=num_joints,
        device=torch.device("cpu"),
        joint_limits=limits,
        velocity_limits=torch.ones(num_envs, num_joints),
    )


def test_initialize_state_uses_default_pose_without_sampling():
    profile = make_profile()
    default = torch.tensor([[0.1, -0.2], [0.2, -0.3], [0.3, -0.4]])
    profile.initialize_state(default)
    torch.testing.assert_close(profile._current_position, default)
    torch.testing.assert_close(profile._previous_position, default)
    torch.testing.assert_close(profile._initial_position, default)
    torch.testing.assert_close(profile._target_position, default)
    assert not profile._is_active.any()
    assert torch.count_nonzero(profile._velocity_scale) == 0


def test_redirect_target_preserves_ema_trajectory_state():
    profile = make_profile()
    profile._current_position.copy_(torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
    profile._previous_position.copy_(profile._current_position - 0.1)
    profile._initial_position.copy_(profile._current_position - 0.2)
    profile._current_velocity.fill_(0.7)
    profile._velocity_scale.copy_(torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]))
    state_before = [
        value.clone()
        for value in (
            profile._current_position,
            profile._previous_position,
            profile._initial_position,
            profile._current_velocity,
            profile._velocity_scale,
        )
    ]
    mask = torch.tensor([True, False, True])
    targets = torch.zeros(3, 2)
    profile.redirect_target(targets, env_mask=mask)
    torch.testing.assert_close(profile._target_position[mask], targets[mask])
    assert torch.equal(profile._is_active, mask)
    for actual, expected in zip(
        (
            profile._current_position,
            profile._previous_position,
            profile._initial_position,
            profile._current_velocity,
            profile._velocity_scale,
        ),
        state_before,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_redirect_target_with_empty_mask_preserves_every_state_tensor(monkeypatch):
    profile = make_profile()
    profile._initial_position.copy_(torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
    profile._current_position.copy_(profile._initial_position + 1.0)
    profile._target_position.copy_(profile._initial_position + 2.0)
    profile._previous_position.copy_(profile._initial_position + 3.0)
    profile._current_velocity.copy_(profile._initial_position + 4.0)
    profile._velocity_scale.copy_(profile._initial_position + 5.0)
    profile._is_active.copy_(torch.tensor([True, False, True]))
    state_names = (
        "_initial_position",
        "_current_position",
        "_target_position",
        "_previous_position",
        "_current_velocity",
        "_velocity_scale",
        "_is_active",
    )
    state_before = {name: getattr(profile, name).clone() for name in state_names}
    monkeypatch.setattr(torch.Tensor, "__bool__", lambda self: (_ for _ in ()).throw(AssertionError("sync")))

    profile.redirect_target(torch.full((3, 2), -10.0), env_mask=torch.zeros(3, dtype=torch.bool))

    for name, expected in state_before.items():
        assert torch.equal(getattr(profile, name), expected)


def test_fixed_coefficient_matches_legacy_recurrence():
    profile = make_profile(num_envs=1)
    current = torch.tensor([[0.2, -0.4]])
    target = torch.tensor([[0.6, 0.0]])
    profile.initialize_state(current)
    profile.set_target(current, target, env_ids=torch.tensor([0]))
    torch.testing.assert_close(profile.compute_next_position(0.02), 0.25 * target + 0.75 * current)


def test_completion_snaps_exactly_to_target():
    profile = make_profile(num_envs=1)
    profile.cfg.position_tolerance = 0.01
    profile.cfg.velocity_tolerance = 1.0
    profile._current_position.fill_(0.999)
    profile._target_position.fill_(1.0)
    profile._velocity_scale.fill_(0.25)
    profile._is_active.fill_(True)
    result = profile.compute_next_position(0.02)
    torch.testing.assert_close(result, profile._target_position, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(profile._current_velocity) == 0
    assert not profile._is_active[0]


def test_inactive_default_pose_outside_limits_is_returned_exactly():
    profile = make_profile(num_envs=1)
    default = torch.tensor([[1.5, -1.5]])
    profile.initialize_state(default)

    result = profile.compute_next_position(0.02)

    torch.testing.assert_close(result, default, rtol=0.0, atol=0.0)


def test_compute_does_not_convert_tensors_to_python_bool(monkeypatch):
    profile = make_profile(num_envs=1)
    profile.initialize_state(torch.zeros(1, 2))
    profile.set_target(torch.zeros(1, 2), torch.ones(1, 2), env_ids=torch.tensor([0]))
    monkeypatch.setattr(torch.Tensor, "__bool__", lambda self: (_ for _ in ()).throw(AssertionError("sync")))
    profile.compute_next_position(0.02)
