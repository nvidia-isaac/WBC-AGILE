import importlib.util
from types import SimpleNamespace

import torch
import yaml

from agile.rl_env.mdp.actions.random_actions import RandomPositionAction


def _load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_remove_training_only_actions_drops_random_pos_for_history_export():
    module = _load_module("export_pruning", "agile/rl_env/rsl_rl/export_pruning.py")

    env_cfg = SimpleNamespace(
        actions=SimpleNamespace(
            joint_pos=object(),
            random_pos=object(),
            random_upper_body_pos=object(),
            harness=object(),
            lift=object(),
        ),
        curriculum=SimpleNamespace(remove_harness=object(), adaptive_lift=object(), terrain_levels=object()),
    )

    removed = module.remove_training_only_actions(env_cfg)

    assert removed == ["harness", "lift", "random_pos", "random_upper_body_pos"]
    assert hasattr(env_cfg.actions, "joint_pos")
    assert not hasattr(env_cfg.actions, "harness")
    assert not hasattr(env_cfg.actions, "lift")
    assert not hasattr(env_cfg.actions, "random_pos")
    assert not hasattr(env_cfg.actions, "random_upper_body_pos")
    assert not hasattr(env_cfg.curriculum, "remove_harness")
    assert not hasattr(env_cfg.curriculum, "adaptive_lift")
    assert hasattr(env_cfg.curriculum, "terrain_levels")


def test_prepare_training_only_actions_for_evaluation_holds_default_joint_positions():
    module = _load_module("eval_action_preparation", "agile/rl_env/rsl_rl/export_pruning.py")
    random_pos = SimpleNamespace(randomize=True)
    random_upper_body_pos = SimpleNamespace(randomize=True)
    env_cfg = SimpleNamespace(
        actions=SimpleNamespace(
            joint_pos=object(),
            random_pos=random_pos,
            random_upper_body_pos=random_upper_body_pos,
            harness=object(),
            lift=object(),
        ),
        curriculum=SimpleNamespace(remove_harness=object(), adaptive_lift=object(), terrain_levels=object()),
    )

    removed, held_at_default = module.prepare_training_only_actions_for_evaluation(env_cfg)

    assert removed == ["harness", "lift"]
    assert held_at_default == ["random_pos", "random_upper_body_pos"]
    assert env_cfg.actions.random_pos is random_pos
    assert env_cfg.actions.random_upper_body_pos is random_upper_body_pos
    assert random_pos.randomize is False
    assert random_upper_body_pos.randomize is False
    assert not hasattr(env_cfg.curriculum, "remove_harness")
    assert not hasattr(env_cfg.curriculum, "adaptive_lift")
    assert hasattr(env_cfg.curriculum, "terrain_levels")


def test_random_position_action_holds_default_positions_when_randomization_is_disabled():
    default_joint_pos = torch.tensor([[1.3, -1.3]])
    term = SimpleNamespace(
        cfg=SimpleNamespace(randomize=False),
        _offset=default_joint_pos,
        _processed_actions=torch.zeros_like(default_joint_pos),
        _target_processed_actions=torch.zeros_like(default_joint_pos),
        _target_write_pending=False,
    )

    RandomPositionAction.process_actions(term, torch.empty(1, 0))

    torch.testing.assert_close(term._processed_actions, default_joint_pos)
    torch.testing.assert_close(term._target_processed_actions, default_joint_pos)
    assert term._target_write_pending is True


def test_update_leapp_export_metadata_preserves_upstream_input_names(tmp_path):
    module = _load_module("leapp_export_metadata", "agile/rl_env/rsl_rl/leapp_export_metadata.py")

    yaml_path = tmp_path / "Velocity-G1-History-v0.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "Velocity-G1-History-v0": {
                        "inputs": [
                            {
                                "name": "base_velocity",
                                "kind": "command/body/velocity",
                                "shape": [1, 3],
                                "element_names": [["lin_vel_x", "lin_vel_y", "ang_vel_z"]],
                            },
                            {
                                "name": "velocity_height",
                                "kind": "command/body/velocity_height",
                                "shape": [1, 4],
                                "element_names": [["lin_vel_x", "lin_vel_y", "ang_vel_z", "height"]],
                            },
                        ]
                    }
                },
                "pipeline": {},
            }
        )
    )

    module.update_leapp_export_metadata(yaml_path, frequency_hz=50.0)

    updated = yaml.safe_load(yaml_path.read_text())
    inputs = updated["models"]["Velocity-G1-History-v0"]["inputs"]
    assert inputs[0]["element_names"] == [["lin_vel_x", "lin_vel_y", "ang_vel_z"]]
    assert inputs[1]["element_names"] == [["lin_vel_x", "lin_vel_y", "ang_vel_z", "height"]]
    assert updated["pipeline"]["configs"]["frequency"] == 50.0


def test_update_leapp_export_metadata_preserves_upstream_frequency(tmp_path):
    module = _load_module("leapp_export_metadata", "agile/rl_env/rsl_rl/leapp_export_metadata.py")

    yaml_path = tmp_path / "Velocity-G1-History-v0.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "models": {},
                "pipeline": {
                    "configs": {
                        "frequency": 60.0,
                    }
                },
            }
        )
    )

    module.update_leapp_export_metadata(yaml_path, frequency_hz=50.0)

    updated = yaml.safe_load(yaml_path.read_text())
    assert updated["pipeline"]["configs"]["frequency"] == 60.0


def test_update_leapp_export_metadata_writes_full_robot_defaults(tmp_path):
    module = _load_module("leapp_export_metadata", "agile/rl_env/rsl_rl/leapp_export_metadata.py")

    yaml_path = tmp_path / "Velocity-G1-History-v0.yaml"
    yaml_path.write_text(yaml.safe_dump({"models": {}, "pipeline": {}}))

    module.update_leapp_export_metadata(
        yaml_path,
        frequency_hz=50.0,
        robot_articulation={
            "joint_names": ["waist_yaw_joint", "left_hip_pitch_joint"],
            "default_joint_pos": [0.0, -0.1],
            "default_joint_stiffness": [200.0, 100.0],
            "default_joint_damping": [20.0, 10.0],
        },
    )

    updated = yaml.safe_load(yaml_path.read_text())
    robot = updated["agile"]["articulations"]["robot"]
    assert robot["joint_names"] == ["waist_yaw_joint", "left_hip_pitch_joint"]
    assert robot["default_joint_pos"] == [0.0, -0.1]
    assert robot["default_joint_stiffness"] == [200.0, 100.0]
    assert robot["default_joint_damping"] == [20.0, 10.0]


def test_leapp_export_uses_the_inference_checkpoint_load_configuration() -> None:
    vendored_export = open("third_party/isaaclab_leapp_export/export.py").read()
    export_wrapper = open("scripts/export_policy_leapp.py").read()

    assert "runner.load(resume_path, load_cfg=checkpoint_load_cfg)" in vendored_export
    assert "checkpoint_load_cfg=make_rsl_rl_inference_load_cfg(agent_cfg)" in export_wrapper
