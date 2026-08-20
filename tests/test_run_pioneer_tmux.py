from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_pioneer_tmux.sh"


class RunPioneerTmuxTests(unittest.TestCase):
    def _run_inner(self, executable_name):
        executable = shutil.which(executable_name)
        self.assertIsNotNone(executable)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        run_dir = root / "run"
        repo_dir = root / "repo"
        run_dir.mkdir()
        repo_dir.mkdir()
        completed = subprocess.run(
            [
                "bash",
                str(RUNNER),
                "--inside-tmux",
                "0",
                str(run_dir),
                executable,
                "fake.json",
                str(repo_dir),
            ],
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


if __name__ == "__main__":
    unittest.main()
