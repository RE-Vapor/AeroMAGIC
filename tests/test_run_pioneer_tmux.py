import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_pioneer_tmux.sh"


class RunPioneerTmuxTests(unittest.TestCase):
    def _run_inner(self, executable_name, debug_profile="quick", legacy=False):
        executable = shutil.which(executable_name)
        self.assertIsNotNone(executable)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        run_dir = root / "run"
        repo_dir = root / "repo"
        run_dir.mkdir()
        repo_dir.mkdir()
        command = [
            "bash",
            str(RUNNER),
            "--inside-tmux",
            "0",
            str(run_dir),
            executable,
            "fake.json",
        ]
        if not legacy:
            command.append(debug_profile)
        command.append(str(repo_dir))
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        status = dict(
            line.split("=", 1)
            for line in (run_dir / "status.txt").read_text(encoding="utf-8").splitlines()
        )
        return completed, status

    def test_success_exit_is_persisted_after_function_locals_expire(self):
        completed, status = self._run_inner("true")
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(status["exit_code"], "0")
        self.assertIn("finished_at_utc", status)

    def test_failure_exit_is_persisted_after_function_locals_expire(self):
        completed, status = self._run_inner("false")
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(status["exit_code"], "1")
        self.assertIn("finished_at_utc", status)

    def test_debug_profile_is_forwarded_to_planner(self):
        completed, status = self._run_inner("echo", "pioneer-50")
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(status["exit_code"], "0")
        self.assertIn(
            "test_magician_planning.py -c fake.json --debug-profile pioneer-50",
            completed.stdout,
        )

    def test_legacy_inside_signature_defaults_to_quick(self):
        completed, status = self._run_inner("echo", legacy=True)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(status["exit_code"], "0")
        self.assertIn("--debug-profile quick", completed.stdout)

    def test_outer_runner_derives_profile_manifest_and_rejects_mismatch(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        repo_dir = root / "repo"
        runner = repo_dir / "scripts" / RUNNER.name
        test_configs = repo_dir / "configs" / "test"
        debug_profiles = repo_dir / "configs" / "debug"
        fake_bin = root / "bin"
        for directory in (runner.parent, test_configs, debug_profiles, fake_bin):
            directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RUNNER, runner)

        profile = {
            "name": "pioneer-50",
            "description": "test fixture",
            "debug_only": True,
            "coverage_comparable": False,
            "output_suffix": "debug_pioneer50",
            "overrides": {
                "validation_n_proxy_points": 100000,
                "beam_width": 3,
                "beam_steps": 3,
                "validation_n_interpolation_steps": 1,
                "validation_n_poses_in_trajectory": 49,
                "experiment_budget_observations": 50,
                "validation_max_start_positions": 1,
                "random_seed": 8,
                "torch_seed": 9,
            },
        }
        (debug_profiles / "pioneer-50.json").write_text(
            json.dumps(profile), encoding="utf-8"
        )
        for name, run_dir in (
            ("derived.json", "../artifacts/attempt-001"),
            ("mismatch.json", "../artifacts/mismatch"),
        ):
            (test_configs / name).write_text(
                json.dumps(
                    {
                        "test_scenes": ["HKUST"],
                        "debug_profile": "pioneer-50",
                        "experiment_run_dir": run_dir,
                        "pioneer_planner_state_mode": "position_only",
                        "pioneer_filter_occupied_position_candidates": True,
                        "validation_require_complete_occupied_pose": True,
                        "pioneer_cubemap_rig_frame": "world",
                        "pioneer_cubemap_extrinsics_version": (
                            "pytorch3d-world-axes-v1"
                        ),
                        "pioneer_canonical_orientation_indices": [2, 0],
                    }
                ),
                encoding="utf-8",
            )

        tmux_log = root / "tmux.log"
        fake_tmux = fake_bin / "tmux"
        fake_tmux.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$*\" >> \"$FAKE_TMUX_LOG\"\n"
            "if [ \"$1\" = has-session ]; then exit 1; fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_tmux.chmod(0o755)
        fake_sha256sum = fake_bin / "sha256sum"
        fake_sha256sum.write_text(
            "#!/bin/sh\n"
            "\"$PIONEER_PYTHON\" -c 'import hashlib, sys; "
            "p = sys.argv[1]; print(hashlib.sha256(open(p, \"rb\").read()).hexdigest(), p)' "
            "\"$1\"\n",
            encoding="utf-8",
        )
        fake_sha256sum.chmod(0o755)

        subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_dir,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True
        )
        subprocess.run(
            [
                "git",
                "add",
                "--",
                "scripts/run_pioneer_tmux.sh",
                "configs/debug/pioneer-50.json",
                "configs/test/derived.json",
                "configs/test/mismatch.json",
            ],
            cwd=repo_dir,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "commit.gpgSign=false",
                "-c",
                "core.hooksPath=/dev/null",
                "commit",
                "-qm",
                "test fixture",
            ],
            cwd=repo_dir,
            check=True,
        )

        environment = os.environ.copy()
        environment.update(
            {
                "FAKE_TMUX_LOG": str(tmux_log),
                "PATH": str(fake_bin) + os.pathsep + environment["PATH"],
                "PIONEER_PYTHON": sys.executable,
            }
        )
        derived_run_dir = root / "artifacts" / "attempt-001"
        completed = subprocess.run(
            [
                "bash",
                str(runner),
                "pioneer-outer-derived",
                "0",
                str(derived_run_dir),
                "derived.json",
            ],
            cwd=repo_dir,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        manifest = dict(
            line.split("=", 1)
            for line in (derived_run_dir / "manifest.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        self.assertEqual(manifest["debug_profile"], "pioneer-50")
        self.assertEqual(manifest["scene"], "HKUST")
        self.assertEqual(manifest["expected_observations"], "50")
        self.assertEqual(manifest["expected_real_face_renders"], "300")
        self.assertEqual(manifest["planner_state_mode"], "position_only")
        self.assertEqual(
            manifest["filter_occupied_position_candidates"], "true"
        )
        self.assertEqual(manifest["require_complete_occupied_pose"], "true")
        self.assertEqual(manifest["planner_state_dimension"], "3")
        self.assertEqual(manifest["cubemap_rig_frame"], "world")
        self.assertEqual(
            manifest["cubemap_extrinsics_version"],
            "pytorch3d-world-axes-v1",
        )
        self.assertEqual(manifest["canonical_orientation_indices"], "[2,0]")
        self.assertEqual(manifest["raw_action_branches_per_parent"], "6")
        self.assertEqual(manifest["orientation_action_branches_per_parent"], "0")
        self.assertIn("new-session", tmux_log.read_text(encoding="utf-8"))

        tmux_log.write_text("", encoding="utf-8")
        mismatch_run_dir = root / "artifacts" / "mismatch"
        mismatch = subprocess.run(
            [
                "bash",
                str(runner),
                "pioneer-outer-mismatch",
                "0",
                str(mismatch_run_dir),
                "mismatch.json",
                "quick",
            ],
            cwd=repo_dir,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(mismatch.returncode, 2)
        self.assertIn("does not match config debug_profile pioneer-50", mismatch.stderr)
        self.assertFalse(mismatch_run_dir.exists())
        self.assertNotIn("new-session", tmux_log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
