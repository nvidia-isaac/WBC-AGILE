from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock


def _load_bootstrap_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "rsl_rl" / "bootstrap.py"
    spec = importlib.util.spec_from_file_location("agile_rsl_rl_bootstrap_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRslRlBootstrap(unittest.TestCase):
    def test_ensure_rsl_rl_patch_skips_when_patch_is_already_applied(self) -> None:
        bootstrap = _load_bootstrap_module()

        def fail_if_called(*args, **kwargs):
            raise AssertionError("patch command should not run when rsl_rl already has AGILE additions")

        with (
            mock.patch.object(bootstrap, "_is_patch_applied", return_value=True),
            mock.patch.object(bootstrap.subprocess, "run", side_effect=fail_if_called),
        ):
            bootstrap.ensure_rsl_rl_patch()

    def test_ensure_rsl_rl_patch_applies_existing_patch_once(self) -> None:
        bootstrap = _load_bootstrap_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            rsl_rl_parent = Path(tmp_dir) / "site-packages"
            rsl_rl_parent.mkdir()
            applied = {"value": False}
            calls = []

            def fake_is_patch_applied():
                return applied["value"]

            def fake_run(command, *, check, cwd):
                calls.append((command, check, cwd))
                applied["value"] = True
                return subprocess.CompletedProcess(command, 0)

            with (
                mock.patch.object(bootstrap, "_is_patch_applied", side_effect=fake_is_patch_applied),
                mock.patch.object(bootstrap, "_locate_rsl_rl_parent", return_value=rsl_rl_parent),
                mock.patch.object(bootstrap.subprocess, "run", side_effect=fake_run),
            ):
                bootstrap.ensure_rsl_rl_patch()

        self.assertEqual(len(calls), 1)
        command, check, cwd = calls[0]
        self.assertTrue(check)
        self.assertEqual(cwd, rsl_rl_parent)
        self.assertEqual(command[:4], ["patch", "--forward", "--batch", "-p1"])
        self.assertEqual(Path(command[-1]).name, "rsl_rl_5_4_1_agile.patch")

    def test_ensure_rsl_rl_patch_removes_stale_bytecode_before_verifying(self) -> None:
        bootstrap = _load_bootstrap_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            rsl_rl_parent = Path(tmp_dir) / "site-packages"
            bytecode_cache = rsl_rl_parent / "rsl_rl" / "models" / "__pycache__"
            bytecode_cache.mkdir(parents=True)
            (bytecode_cache / "__init__.cpython-312.pyc").write_bytes(b"stale")
            patch_ran = {"value": False}

            def fake_is_patch_applied():
                return patch_ran["value"] and not bytecode_cache.exists()

            def fake_run(command, *, check, cwd):
                patch_ran["value"] = True
                return subprocess.CompletedProcess(command, 0)

            with (
                mock.patch.object(bootstrap, "_is_patch_applied", side_effect=fake_is_patch_applied),
                mock.patch.object(bootstrap, "_locate_rsl_rl_parent", return_value=rsl_rl_parent),
                mock.patch.object(bootstrap.subprocess, "run", side_effect=fake_run),
            ):
                bootstrap.ensure_rsl_rl_patch()

            self.assertFalse(bytecode_cache.exists())

    def test_ensure_rsl_rl_patch_raises_if_patch_command_does_not_add_agile_extensions(self) -> None:
        bootstrap = _load_bootstrap_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            rsl_rl_parent = Path(tmp_dir) / "site-packages"
            rsl_rl_parent.mkdir()

            with (
                mock.patch.object(bootstrap, "_is_patch_applied", return_value=False),
                mock.patch.object(bootstrap, "_locate_rsl_rl_parent", return_value=rsl_rl_parent),
                mock.patch.object(
                    bootstrap.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(["patch"], 0),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "AGILE RSL-RL patch did not apply cleanly"):
                    bootstrap.ensure_rsl_rl_patch()


if __name__ == "__main__":
    unittest.main(verbosity=2)
