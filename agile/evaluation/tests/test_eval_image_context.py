from pathlib import Path


def test_evaluation_runtime_dependencies_are_in_docker_context() -> None:
    ignored = Path(".dockerignore").read_text().splitlines()
    assert "agile/sim2mujoco/" not in ignored


def test_downloaded_external_assets_stay_out_of_docker_context() -> None:
    ignored = Path(".dockerignore").read_text().splitlines()

    assert "external_assets/" in ignored
    assert "agile/rl_env/assets/robot_menagerie/" in ignored


def test_eval_image_does_not_bytecode_compile_isaac_dependencies() -> None:
    dockerfile = Path("workflows/Dockerfile").read_text()

    assert "uv sync --frozen --no-dev --no-install-project --compile-bytecode" not in dockerfile
    assert "uv sync --frozen --no-dev --compile-bytecode" not in dockerfile
