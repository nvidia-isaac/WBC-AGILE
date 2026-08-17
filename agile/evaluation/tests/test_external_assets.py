from __future__ import annotations

from pathlib import Path

from agile.evaluation import external_assets


def test_external_asset_paths_are_public_downloads() -> None:
    assert external_assets.UNITREE_MUJOCO.url == "https://github.com/unitreerobotics/unitree_mujoco.git"
    assert external_assets.UNITREE_MUJOCO.revision == "ae6a8403e272733e9996ef59990880330496177f"
    assert external_assets.G1_MJCF == Path("external_assets/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")

    assert external_assets.BOOSTER_ASSETS.url == "https://github.com/BoosterRobotics/booster_assets.git"
    assert external_assets.BOOSTER_ASSETS.revision == "508cbee6ca9ae6fbc8c0b38dd58785a6f3fc61a2"
    assert external_assets.T1_MJCF == Path("external_assets/booster_assets/robots/T1/T1_23dof.xml")


def test_pyproject_exposes_download_entrypoint() -> None:
    pyproject = Path("pyproject.toml").read_text()

    assert "[project.scripts]" in pyproject
    assert 'agile-download-assets = "agile.evaluation.external_assets:main"' in pyproject


def test_downloader_uses_sparse_checkouts() -> None:
    source = Path("agile/evaluation/external_assets.py").read_text()

    assert "sparse-checkout" in source
    assert "unitree_robots/g1" in source
    assert "robots/T1" in source
