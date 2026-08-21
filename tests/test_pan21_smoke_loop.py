import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from macarons.utility.ue5_observation_contract import (
    make_synthetic_plane_bundle,
    write_bundle,
)
from scripts.validate_pan21_smoke_loop import run_smoke


class Pan21SmokeLoopTests(unittest.TestCase):
    def test_shell_runner_uses_repo_module_and_creates_only_parent(self):
        runner = Path(__file__).resolve().parents[1] / "scripts" / "run_pan21_smoke_loop.sh"
        text = runner.read_text(encoding="utf-8")
        self.assertIn("mkdir -p -- \"$(dirname \"$output_dir\")\"", text)
        self.assertIn("-m scripts.validate_pan21_smoke_loop", text)
        self.assertIn('cd "$repo_root"', text)

    def test_two_bundle_loop_and_atomic_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p0 = make_synthetic_plane_bundle(
                image_size=7,
                request_id="p0-request",
                frame_id="p0-frame",
                capture_timestamp_ns=1_725_000_000_000_000_001,
            )
            front_mask = p0.faces[0].valid_mask.copy()
            front_depth = p0.faces[0].depth_range_m.copy()
            front_mask[3, 3] = False
            front_depth[3, 3] = np.nan
            p0 = dataclasses.replace(
                p0,
                faces=(
                    dataclasses.replace(
                        p0.faces[0],
                        valid_mask=front_mask,
                        depth_range_m=front_depth,
                    ),
                    *p0.faces[1:],
                ),
            )
            # A zero-length synthetic move is enough for the unit fixture; use
            # adjacent Planner state indices while keeping the same world pose.
            p1 = dataclasses.replace(
                p0,
                request_id="p1-request",
                frame_id="p1-frame",
                capture_timestamp_ns=1_725_000_000_000_000_002,
                faces=tuple(
                    dataclasses.replace(
                        face,
                        request_id="p1-request",
                        frame_id="p1-frame",
                        capture_timestamp_ns=1_725_000_000_000_000_002,
                    )
                    for face in p0.faces
                ),
            )
            p0_manifest = write_bundle(p0, root / "p0")
            p1_manifest = write_bundle(p1, root / "p1")
            position = [0.2, 0.6, -0.4]
            metrics = root / "metrics.json"
            metrics.write_text(
                json.dumps(
                    {
                        "planner": "pioneer",
                        "run": {"validation_initial_ue_manifest": str(p0_manifest)},
                        "trajectory": {
                            "observation_count": 2,
                            "planner_state_indices": [[5, 3, 1], [5, 3, 2]],
                            "positions": [position, position],
                        },
                        "planner_search": {"totals": {"generated_candidate_count": 1}},
                        "pioneer_observation": {"bundles": [{"depth_source": "UE5"}]},
                    }
                ),
                encoding="utf-8",
            )
            report = run_smoke(
                p0_manifest=p0_manifest,
                p1_manifest=p1_manifest,
                planner_metrics=metrics,
                output_dir=root / "output",
                device_name="cpu",
                proxy_count=128,
            )
            self.assertEqual(report["result"], "PASS")
            self.assertEqual(report["observation_count"], 2)
            self.assertTrue(report["negative_test"]["zero_update"])
            self.assertEqual(report["final_state"]["surface_scene"]["fill_count"], 2)
            self.assertEqual(report["final_state"]["proxy_scene"]["occupancy_update_count"], 2)


if __name__ == "__main__":
    unittest.main()
