"""Standard-library mock tests only; no preparation, ML imports or training."""
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("colab_runner", ROOT / "colab_train_submit.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def args(root, *extra):
    return runner.parse_args(["--root", str(root), "--data-dir", str(root / "source"),
                              "--output-dir", str(root / "backup"), *extra])


class PlanTests(unittest.TestCase):
    def test_order_and_no_obsolete_script(self):
        with tempfile.TemporaryDirectory() as temp:
            a = args(Path(temp))
            commands = runner.plan_commands(a)
            paths = [Path(c[1]).name if len(c) > 1 else "" for c in commands]
            expected = ["00_load_and_overview.py", "04_relational.py", "build_base.py",
                        "features_v1.py", "features_v2.py", "features_v4.py"]
            self.assertEqual([name for name in paths if name in expected], expected)
            ratio_index = next(i for i, c in enumerate(commands) if c[-1] == "ratios")
            self.assertLess(ratio_index, paths.index("features_v1.py"))
            self.assertEqual([c[-1] for c in commands if "--stage" in c], ["bootstrap", "validate", "refit"])
            self.assertFalse(any("run_all.py" in token or "features_v3.py" in token for c in commands for token in c))

    def test_pinned_environment_and_explicit_gpu_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            a = args(Path(temp), "--install-deps", "--device", "cuda", "--allow-cpu-fallback")
            commands = runner.plan_commands(a)
            self.assertIn("virtualenv==20.31.2", commands[0])
            pip_checks = [c for c in commands if c[-2:] == ["pip", "check"]]
            self.assertEqual(len(pip_checks), 1)
            self.assertIn(".colab-venv", pip_checks[0][0])
            probe = next(c for c in commands if "--_action" in c and "backend" in c)
            self.assertIn("--allow-cpu-fallback", probe)

    def test_dry_run_has_no_data_or_output_io(self):
        argv = ["--root", "X:/missing-root", "--data-dir", "X:/missing-data",
                "--output-dir", "X:/missing-output", "--install-deps", "--mount-drive", "--dry-run"]
        with patch.object(runner, "_copy_inputs", side_effect=AssertionError("data read")),              patch.object(runner, "run_checked", side_effect=AssertionError("command executed")),              patch.object(Path, "read_text", side_effect=AssertionError("file read")),              patch.object(Path, "mkdir", side_effect=AssertionError("output created")),              contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(runner.main(argv), 0)
        result = json.loads(out.getvalue())
        self.assertEqual(result["branch"], "codex/target-070")

    def test_stage_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            for stage, expected in (("prepare", []), ("bootstrap", ["bootstrap"]),
                                     ("validate", ["bootstrap", "validate"])):
                commands = runner.plan_commands(args(Path(temp), "--stage", stage))
                self.assertEqual([c[-1] for c in commands if "--stage" in c], expected)


class FileTests(unittest.TestCase):
    def test_no_differing_raw_input_is_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, raw = root / "source", root / "data/raw"
            source.mkdir()
            raw.mkdir(parents=True)
            for name in runner.INPUTS:
                (source / name).write_text("synthetic", encoding="utf-8")
            (raw / "train.csv").write_text("different existing data", encoding="utf-8")
            with self.assertRaises(ValueError):
                runner._copy_inputs(args(root))
            self.assertEqual((raw / "train.csv").read_text(), "different existing data")
            self.assertFalse((raw / "test.csv").exists())

    def test_same_input_directory_is_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "data/raw"
            raw.mkdir(parents=True)
            for name in runner.INPUTS:
                (raw / name).write_text("synthetic", encoding="utf-8")
            a = args(root)
            a.data_dir = raw
            self.assertEqual(set(runner._copy_inputs(a)), set(runner.INPUTS))

    def test_resume_hash_and_upstream_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "mock.txt"
            artifact.write_text("first", encoding="utf-8")
            saved = {"dependencies": {"prepare": "previous"},
                     "artifacts": {"mock.txt": runner._hash(artifact)}}
            manifest = {"stages": {"bootstrap": saved}}
            self.assertTrue(runner._resume_stage(root, manifest, "bootstrap", {"prepare": "previous"}))
            with self.assertRaises(ValueError):
                runner._resume_stage(root, manifest, "bootstrap", {"prepare": "changed"})
            artifact.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                runner._resume_stage(root, manifest, "bootstrap", {"prepare": "previous"})

    def test_restore_rejects_corrupt_prepared_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = args(root)
            backup = root / "backup/podium_v6"
            backup.mkdir(parents=True)
            runner._json(backup / "manifest.json", {"stages": {
                "prepare": {"artifacts": {"data/interim/mock.parquet": "expected-hash"}}}})
            cached = backup / "prepared_cache/data/interim/mock.parquet"
            cached.parent.mkdir(parents=True)
            cached.write_text("corrupt synthetic cache")
            with self.assertRaises(ValueError):
                runner._restore(a)

    def test_run_checked_raises_original_subprocess_error(self):
        with tempfile.TemporaryDirectory() as temp:
            command = ["python", "fake.py"]
            error = subprocess.CalledProcessError(7, command)
            with patch.object(runner.subprocess, "run", side_effect=error) as mocked,                  contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError) as caught:
                    runner.run_checked(command, cwd=Path(temp), log_path=Path(temp) / "mock.log")
            self.assertEqual(caught.exception.returncode, 7)
            self.assertTrue(mocked.call_args.kwargs["check"])
            self.assertNotIn("shell", mocked.call_args.kwargs)


class MainTests(unittest.TestCase):
    def test_all_stage_orchestration_with_mock_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._setup(root)
            argv = ["--root", str(root), "--data-dir", str(root / "source"),
                    "--output-dir", str(root / "backup"), "--stage", "all"]
            seen = []
            def mock_run(command, **kwargs):
                run = root / "runs/podium_v6"
                if command[-1] == "environment":
                    runner._json(run / "environment.json", {
                        "python": "3.12.0", "versions": {}, "installed_distributions": {}})
                if "--stage" in command:
                    stage = command[-1]
                    seen.append(stage)
                    if stage == "bootstrap":
                        runner._json(run / "inputs.json", {"dataset_year": 2026})
                if "--_action" in command and command[-1] == "validate":
                    runner._json(run / "validation_report.json", {"feature_schema_hashes": {"ensemble": "schema"}})
                    (run / "RUN_REPORT.md").write_text("mock report, no training")
            with (
                patch.object(runner.sys, "version_info", (3, 12, 0)),
                patch.object(runner, "_copy_inputs", return_value={"train.csv": "hash"}),
                patch.object(runner, "_capture", side_effect=lambda cmd, cwd: "commit" if "rev-parse" in cmd else ""),
                patch.object(runner, "_hash", return_value="hash"),
                patch.object(runner, "_stage_artifacts", return_value={"prepared.mock": "hash"}),
                patch.object(runner, "run_checked", side_effect=mock_run),
                patch.object(runner, "_backup"),
                contextlib.redirect_stdout(io.StringIO())
            ):
                self.assertEqual(runner.main(argv), 0)
            self.assertEqual(seen, ["bootstrap", "validate", "refit"])
            manifest = runner._read_json(root / "runs/podium_v6/manifest.json")
            self.assertEqual(set(manifest["stages"]), {"prepare", "bootstrap", "validate", "refit"})
            self.assertEqual(manifest["dataset_year"], 2026)
            self.assertEqual(manifest["last_completed_stage"], "all")

    def _setup(self, root):
        (root / "configs").mkdir()
        (root / "configs/podium.json").write_text(json.dumps({
            "members": [{"seed": 17}], "splits": {}, "profiles": ["ensemble"]}))
        (root / "prepared.mock").write_text("fixture")

    def test_main_failure_preserves_exit_even_when_backup_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._setup(root)
            argv = ["--root", str(root), "--data-dir", str(root / "source"),
                    "--output-dir", str(root / "backup"), "--stage", "prepare"]
            with patch.object(runner.sys, "version_info", (3, 12, 0)),                  patch.object(runner, "_copy_inputs", return_value={"train.csv": "hash"}),                  patch.object(runner, "_capture", side_effect=["commit", ""]),                  patch.object(runner, "_hash", return_value="hash"),                  patch.object(runner, "run_checked", side_effect=subprocess.CalledProcessError(7, ["mock"])),                  patch.object(runner, "_backup", side_effect=OSError("Drive unavailable")),                  contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(runner.main(argv), 7)
            self.assertIn("Drive unavailable", err.getvalue())
            manifest = json.loads((root / "runs/podium_v6/manifest.json").read_text())
            self.assertEqual(manifest["last_failure"]["exit_code"], 7)

    def test_mock_prepare_and_verified_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._setup(root)
            argv = ["--root", str(root), "--data-dir", str(root / "source"),
                    "--output-dir", str(root / "backup"), "--stage", "prepare"]
            prepare_calls = []
            def mock_run(command, **kwargs):
                if command[-1] == "environment":
                    runner._json(root / "runs/podium_v6/environment.json", {
                        "python": "3.12.0", "versions": {}, "installed_distributions": {}})
                if "00_load_and_overview.py" in " ".join(command):
                    prepare_calls.append(command)
            with patch.object(runner.sys, "version_info", (3, 12, 0)),                  patch.object(runner, "_copy_inputs", return_value={"train.csv": "hash"}),                  patch.object(runner, "_capture", side_effect=lambda cmd, cwd: "commit" if "rev-parse" in cmd else ""),                  patch.object(runner, "_hash", return_value="hash"),                  patch.object(runner, "_stage_artifacts", return_value={"prepared.mock": "hash"}),                  patch.object(runner, "run_checked", side_effect=mock_run),                  patch.object(runner, "_backup"),                  contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(argv), 0)
                self.assertEqual(runner.main(argv + ["--resume"]), 0)
            self.assertEqual(len(prepare_calls), 1)
            manifest = runner._read_json(root / "runs/podium_v6/manifest.json")
            self.assertIn("prepare", manifest["stages"])

    def test_import_does_not_load_ml_dependencies(self):
        code = ("import runpy,sys;runpy.run_path(sys.argv[1],run_name='runner_import_test');"
                "assert not {'numpy','pandas','lightgbm'}.intersection(sys.modules)")
        result = subprocess.run([sys.executable, "-c", code, str(ROOT / "colab_train_submit.py")],
                                check=True, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
