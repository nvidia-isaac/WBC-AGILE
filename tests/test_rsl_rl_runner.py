from types import SimpleNamespace

from agile.rl_env.rsl_rl.runner import make_rsl_rl_inference_load_cfg


def _agent_cfg(algorithm_class_name: str) -> SimpleNamespace:
    return SimpleNamespace(algorithm=SimpleNamespace(class_name=algorithm_class_name))


def test_distillation_inference_loads_only_the_student_policy() -> None:
    assert make_rsl_rl_inference_load_cfg(_agent_cfg("Distillation")) == {
        "student": True,
        "teacher": False,
        "optimizer": False,
        "iteration": False,
    }


def test_ppo_inference_keeps_default_checkpoint_loading() -> None:
    assert make_rsl_rl_inference_load_cfg(_agent_cfg("PPO")) is None


def test_eval_uses_the_inference_checkpoint_load_configuration() -> None:
    eval_script = open("scripts/eval.py").read()

    assert "make_rsl_rl_inference_load_cfg(agent_cfg)" in eval_script
