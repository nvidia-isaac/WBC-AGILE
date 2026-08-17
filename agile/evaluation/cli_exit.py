"""Process-boundary handling for Isaac Sim evaluation commands."""

from collections.abc import Callable
from typing import NoReturn, Protocol


class _SimulationApp(Protocol):
    def close(self, *, exit_code: int = 0) -> None: ...


def close_simulation_app(simulation_app: _SimulationApp, *, exit_code: int) -> NoReturn:
    """Close Kit without allowing fast shutdown to replace the command's status."""
    simulation_app.close(exit_code=exit_code)
    raise SystemExit(exit_code)


def run_main_with_simulation_app(main: Callable[[], None], simulation_app: _SimulationApp) -> NoReturn:
    """Run an evaluation and preserve its status through Kit fast shutdown."""
    exit_code = 0
    try:
        main()
    except Exception as error:
        import traceback

        print(f"\n[ERROR] Evaluation failed with exception: {error}", flush=True)
        traceback.print_exc()
        exit_code = 1
    finally:
        # Kit fast shutdown can call os._exit() inside close(). Passing the status
        # here prevents it from replacing an evaluation failure with exit code 0.
        close_simulation_app(simulation_app, exit_code=exit_code)
