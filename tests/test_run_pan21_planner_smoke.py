from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_pan21_planner_smoke.sh"


class Pan21PlannerRunnerTests(unittest.TestCase):
    def test_runner_pins_two_observation_profile_and_tmux(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("PROFILE=pan21-two-observation", text)
        self.assertIn("expected_observations=2", text)
        self.assertIn("validation_start_position_override=5,3,1,2,0", text)
        self.assertIn("tmux new-session", text)
        self.assertIn("refusing to run from a dirty worktree", text)
        self.assertIn('config["dataset_path"] = str(run_dir / "data" / "Macarons++")', text)
        self.assertIn("isolated_dataset_overlay", text)
        self.assertIn("isolated_results_root", text)
        self.assertIn("spatial policy hash does not match debug profile", text)
        self.assertIn("validation_position_index_bounds", text)
        self.assertNotIn("PAN-24", text)


if __name__ == "__main__":
    unittest.main()
