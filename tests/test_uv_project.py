from __future__ import annotations

import os
import pathlib
import tomllib
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_pyproject() -> dict:
    pyproject_path = _REPO_ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as file:
        return tomllib.load(file)


class TestUvProject(unittest.TestCase):
    def test_pyproject_declares_isaac_lab_beta2_x86_dependency_contract(self) -> None:
        pyproject = _load_pyproject()

        dependencies = pyproject["project"]["dependencies"]

        self.assertIn("isaaclab[isaacsim]==3.0.0b2", dependencies)
        self.assertIn("pytest>=9.1.1", dependencies)
        self.assertIn("torch==2.11.0", dependencies)
        self.assertIn("torchaudio==2.11.0", dependencies)
        self.assertIn("torchvision==0.26.0", dependencies)
        self.assertIn("rsl-rl-lib==5.4.1", dependencies)
        self.assertIn("leapp==0.5.2", dependencies)

    def test_pyproject_configures_uv_indexes_for_isaac_sim_resolution(self) -> None:
        pyproject = _load_pyproject()

        uv_config = pyproject["tool"]["uv"]
        indexes = pyproject["tool"]["uv"]["index"]
        sources = pyproject["tool"]["uv"]["sources"]

        self.assertEqual(uv_config["index-strategy"], "unsafe-best-match")
        self.assertEqual(uv_config["prerelease"], "allow")
        self.assertEqual(
            uv_config["environments"],
            ["sys_platform == 'linux' and platform_machine == 'x86_64'"],
        )
        self.assertTrue(any(index["url"] == "https://pypi.nvidia.com" for index in indexes))
        self.assertTrue(any(index.get("name") == "pytorch-cu128" for index in indexes))
        self.assertFalse(any(index.get("name") in {"pytorch-cu126", "pytorch-cu130"} for index in indexes))
        for package in ("torch", "torchaudio", "torchvision"):
            package_sources = sources[package]
            self.assertEqual(
                package_sources,
                [
                    {
                        "index": "pytorch-cu128",
                        "marker": "sys_platform == 'linux' and platform_machine == 'x86_64'",
                    }
                ],
            )

    def test_documented_script_entrypoints_are_executable_scripts(self) -> None:
        for script in (
            "scripts/eval.py",
            "scripts/export_policy.py",
            "scripts/play.py",
            "scripts/sim2mujoco_eval.py",
            "scripts/train.py",
            "scripts/verify_rsl_rl.py",
            "scripts/wandb_sweep/init_sweep.py",
            "scripts/wandb_sweep/run_sweep.py",
            "scripts/wandb_sweep/train_wrapper.py",
        ):
            script_path = _REPO_ROOT / script

            self.assertTrue(script_path.read_text().startswith("#!"))
            self.assertTrue(os.access(script_path, os.X_OK))

    def test_leapp_export_drops_all_zero_dimensional_training_actions(self) -> None:
        eval_script = (_REPO_ROOT / "scripts/eval.py").read_text()
        export_script = (_REPO_ROOT / "scripts/export_policy_leapp.py").read_text()
        export_pruning = (_REPO_ROOT / "agile/rl_env/rsl_rl/export_pruning.py").read_text()

        self.assertIn("remove_training_only_actions(env_cfg)", export_script)
        self.assertIn("prepare_training_only_actions_for_evaluation(env_cfg)", eval_script)
        self.assertIn(
            'TRAINING_ONLY_ACTIONS = ("harness", "lift", "random_pos", "random_upper_body_pos")', export_pruning
        )

    def test_docs_use_direct_uv_script_entrypoints(self) -> None:
        paths_to_check = [
            _REPO_ROOT / "README.md",
            *_REPO_ROOT.glob("docs/source/*.md"),
            *_REPO_ROOT.glob("scripts/wandb_sweep/*.yaml"),
            *_REPO_ROOT.glob("workflows/*.yaml"),
            _REPO_ROOT / ".gitlab-ci.yml",
        ]

        for path in paths_to_check:
            if not path.is_file():
                continue
            text = path.read_text()
            self.assertNotIn("isaaclab.sh -p scripts/", text, msg=str(path.relative_to(_REPO_ROOT)))
            self.assertNotIn("uv run python scripts/", text, msg=str(path.relative_to(_REPO_ROOT)))
            self.assertNotIn("uv run --frozen python scripts/", text, msg=str(path.relative_to(_REPO_ROOT)))
            self.assertNotIn(
                "uv run --frozen --offline --no-sync python scripts/",
                text,
                msg=str(path.relative_to(_REPO_ROOT)),
            )

    def test_pytest_is_the_test_entrypoint(self) -> None:
        pyproject = _load_pyproject()
        pytest_config = pyproject["tool"]["pytest"]["ini_options"]

        self.assertIn("e2e: end-to-end tests", pytest_config["markers"][0])
        self.assertIn("agile/evaluation/tests", pytest_config["testpaths"])
        ci_path = _REPO_ROOT / ".gitlab-ci.yml"
        if ci_path.is_file():
            ci_config = ci_path.read_text()
            self.assertEqual(ci_config.count('uv run --frozen pytest -m "not e2e"'), 2)
            self.assertEqual(ci_config.count("uv run --frozen pytest -m e2e"), 2)
            self.assertNotIn("resource_group: isaac-gpu-tests", ci_config)

            for legacy_entrypoint in (
                "tests/run_unit_tests.sh",
                "tests/test_e2e_ci_locally.sh",
                "tests/test_all_tasks_e2e.py",
                "tests/test_deterministic_eval_e2e.py",
                "tests/test_sim2mujoco_e2e.py",
            ):
                self.assertNotIn(legacy_entrypoint, ci_config)

        conftest = (_REPO_ROOT / "conftest.py").read_text()
        self.assertIn("Unit tests must not launch AppLauncher", conftest)
        self.assertIn('(config.option.markexpr or "").strip() == "e2e"', conftest)
        self.assertIn('not collection_path.name.endswith("_e2e.py")', conftest)
        self.assertIn('collection_path.name.endswith("_e2e.py")', conftest)

    def test_gpu_e2e_jobs_are_serialized(self) -> None:
        ci_path = _REPO_ROOT / ".gitlab-ci.yml"
        if ci_path.is_file():
            self.assertEqual(ci_path.read_text().count("resource_group: agile-gpu-e2e"), 2)

    def test_e2e_test_files_are_marked(self) -> None:
        e2e_test_files = sorted((_REPO_ROOT / "tests").glob("*_e2e.py"))
        self.assertGreater(len(e2e_test_files), 0)

        for path in e2e_test_files:
            text = path.read_text()
            self.assertIn("pytestmark" + " = pytest.mark.e2e", text, msg=str(path.relative_to(_REPO_ROOT)))

    def test_marked_e2e_tests_use_the_e2e_filename_convention(self) -> None:
        e2e_marker = "pytestmark" + " = pytest.mark.e2e"
        pytest_paths = _load_pyproject()["tool"]["pytest"]["ini_options"]["testpaths"]
        for pytest_path in pytest_paths:
            for path in (_REPO_ROOT / pytest_path).rglob("test_*.py"):
                text = path.read_text()
                if e2e_marker in text:
                    self.assertTrue(path.name.endswith("_e2e.py"), msg=str(path.relative_to(_REPO_ROOT)))

    def test_non_e2e_tests_do_not_construct_app_launcher(self) -> None:
        pytest_paths = _load_pyproject()["tool"]["pytest"]["ini_options"]["testpaths"]
        for pytest_path in pytest_paths:
            for path in (_REPO_ROOT / pytest_path).rglob("test_*.py"):
                if path.name.endswith("_e2e.py"):
                    continue
                self.assertNotIn("AppLauncher" + "(", path.read_text(), msg=str(path.relative_to(_REPO_ROOT)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
