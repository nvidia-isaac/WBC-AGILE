import pytest

from agile.evaluation.cli_exit import close_simulation_app, run_main_with_simulation_app


class _FastShutdownApp:
    """Mirror SimulationApp.close terminating the process during fast shutdown."""

    def close(self, *, exit_code: int = 0) -> None:
        raise SystemExit(exit_code)


def test_evaluation_failure_survives_simulation_app_fast_shutdown() -> None:
    def fail_evaluation() -> None:
        raise RuntimeError("rollout had a non-timeout termination")

    with pytest.raises(SystemExit) as raised:
        run_main_with_simulation_app(fail_evaluation, _FastShutdownApp())

    assert raised.value.code == 1


def test_unsuccessful_command_survives_simulation_app_fast_shutdown() -> None:
    with pytest.raises(SystemExit) as raised:
        close_simulation_app(_FastShutdownApp(), exit_code=1)

    assert raised.value.code == 1
