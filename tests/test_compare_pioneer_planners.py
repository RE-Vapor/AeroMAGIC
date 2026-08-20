import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_pioneer_planners import (  # noqa: E402
    ComparisonError,
    build_comparison,
    main,
    render_markdown,
)


LEGACY_CONFIG = (
    ROOT
    / "configs"
    / "test"
    / "test_pioneer_eiffel_pan11_legacy_pose5d_quick_config.json"
)
POSITION_CONFIG = (
    ROOT
    / "configs"
    / "test"
    / "test_pioneer_eiffel_pan11_position_only_quick_config.json"
)
PROFILES_DIR = ROOT / "configs" / "debug"


class ComparePioneerPlannersTests(unittest.TestCase):
    def _metrics(self, mode, *, trajectory_seconds=None, orientation_proposals=None):
        legacy = mode == "legacy_pose5d"
        state_dimension = 5 if legacy else 3
        rig_frame = "body" if legacy else "world"
        extrinsics_version = (
            "pytorch3d-body-aligned-v1"
            if legacy
            else "pytorch3d-world-axes-v1"
        )
        raw = 60 if legacy else 36
        translation = 36
        orientation = 24 if legacy else 0
        if orientation_proposals is not None:
            orientation = orientation_proposals
        generated = 60 if legacy else 36
        rendered = 40 if legacy else 26
        valid = rendered
        retained = 18
        history_bundles = 9
        base_run_id = (
            "PAN-11-pioneer-eiffel-legacy-pose5d"
            if legacy
            else "PAN-11-pioneer-eiffel-position-only"
        )
        return {
            "schema_version": 1,
            "planner": "pioneer",
            "scene": "eiffel",
            "start_index": 0,
            "run": {
                "run_id": f"{base_run_id}_debug_quick",
                "seed": 8,
                "torch_seed": 9,
                "budget_observations": 3,
                "debug_profile": "quick",
                "planning_observation_mode": "cubemap6",
                "pioneer_planner_state_mode": mode,
                "planner_state_dimension": state_dimension,
                "pioneer_cubemap_rig_frame": rig_frame,
                "pioneer_cubemap_extrinsics_version": extrinsics_version,
                "pioneer_canonical_orientation_indices": None if legacy else [2, 0],
            },
            "trajectory": {"observation_count": 3},
            "planner_search": {
                "schema_version": 1,
                "state_mode": mode,
                "state_dimension": state_dimension,
                "cubemap_rig_frame": rig_frame,
                "cubemap_extrinsics_version": extrinsics_version,
                "totals": {
                    "parent_beam_count": 6,
                    "raw_action_proposal_count": raw,
                    "translation_action_proposal_count": translation,
                    "orientation_action_proposal_count": orientation,
                    "generated_candidate_count": generated,
                    "valid_state_candidate_count": valid,
                    "observed_rejected_candidate_count": 4,
                    "collision_rejected_candidate_count": 0,
                    "rendered_candidate_count": rendered,
                    "retained_beam_count": retained,
                    "search_seconds": 5.0 if legacy else 3.5,
                },
            },
            "pioneer_observation": {
                "bundle_count": 3,
                "real_face_render_count": 18,
                "imagined_bundle_render_count": history_bundles + rendered,
                "imagined_face_render_count": (history_bundles + rendered) * 6,
                "imagined_history_bundle_render_count": history_bundles,
                "imagined_history_face_render_count": history_bundles * 6,
                "imagined_candidate_bundle_render_count": rendered,
                "imagined_candidate_face_render_count": rendered * 6,
            },
            "latency": {
                "trajectory_seconds": (
                    trajectory_seconds
                    if trajectory_seconds is not None
                    else (12.0 if legacy else 10.0)
                )
            },
        }

    def _write_json(self, root, name, value):
        path = Path(root) / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def _write_status(self, root, name, wall_seconds):
        path = Path(root) / name
        path.write_text(
            "started_at_utc=2026-08-20T00:00:00Z\n"
            f"finished_at_utc=2026-08-20T00:00:{wall_seconds:02d}Z\n"
            "exit_code=0\n",
            encoding="utf-8",
        )
        return path

    def _write_manifest(self, run_dir, config_path, *, legacy):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        state_mode = "legacy_pose5d" if legacy else "position_only"
        dimension = 5 if legacy else 3
        rig = "body" if legacy else "world"
        version = (
            "pytorch3d-body-aligned-v1"
            if legacy
            else "pytorch3d-world-axes-v1"
        )
        config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
        values = {
            "schema_version": "1",
            "planner": "pioneer",
            "observation_mode": "cubemap6",
            "scene": "eiffel",
            "git_commit": "a" * 40,
            "config": config_path.name,
            "config_sha256": config_sha,
            "debug_profile": "quick",
            "expected_observations": "3",
            "expected_real_face_renders": "18",
            "experiment_run_dir": str(run_dir.resolve()),
            "planner_state_mode": state_mode,
            "planner_state_dimension": str(dimension),
            "cubemap_rig_frame": rig,
            "cubemap_extrinsics_version": version,
        }
        path = run_dir / "manifest.txt"
        path.write_text(
            "".join(f"{key}={value}\n" for key, value in values.items()),
            encoding="utf-8",
        )
        return path

    def _build(
        self,
        temporary,
        *,
        legacy_metrics=None,
        position_metrics=None,
        legacy_wall_seconds=20,
        position_wall_seconds=16,
    ):
        legacy_run = Path(temporary) / "legacy-run"
        position_run = Path(temporary) / "position-run"
        legacy_metrics_dir = legacy_run / "metrics_debug_quick"
        position_metrics_dir = position_run / "metrics_debug_quick"
        legacy_metrics_dir.mkdir(parents=True)
        position_metrics_dir.mkdir(parents=True)
        legacy_path = self._write_json(
            legacy_metrics_dir,
            "legacy_metrics.json",
            legacy_metrics or self._metrics("legacy_pose5d"),
        )
        position_path = self._write_json(
            position_metrics_dir,
            "position_metrics.json",
            position_metrics or self._metrics("position_only"),
        )
        legacy_status = self._write_status(
            legacy_run, "status.txt", legacy_wall_seconds
        )
        position_status = self._write_status(
            position_run, "status.txt", position_wall_seconds
        )
        legacy_manifest = self._write_manifest(
            legacy_run, LEGACY_CONFIG, legacy=True
        )
        position_manifest = self._write_manifest(
            position_run, POSITION_CONFIG, legacy=False
        )
        return build_comparison(
            legacy_config_path=LEGACY_CONFIG,
            position_config_path=POSITION_CONFIG,
            legacy_metrics_path=legacy_path,
            position_metrics_path=position_path,
            legacy_status_path=legacy_status,
            position_status_path=position_status,
            legacy_manifest_path=legacy_manifest,
            position_manifest_path=position_manifest,
            profiles_dir=PROFILES_DIR,
        )

    def test_builds_strict_controlled_pair_and_reports_measured_direction(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self._build(temporary)

        self.assertTrue(report["fairness"]["validated"])
        self.assertEqual(report["fairness"]["budget_observations"], 3)
        self.assertEqual(report["runs"]["legacy_pose5d"]["state_dimension"], 5)
        self.assertEqual(report["runs"]["position_only"]["state_dimension"], 3)
        self.assertEqual(
            report["position_only_minus_legacy_pose5d"]["planner_search"][
                "orientation_action_proposal_count"
            ],
            -24.0,
        )
        self.assertEqual(
            report["wall_time_interpretation"]["direction"],
            "position_only_shorter",
        )
        markdown = render_markdown(report)
        self.assertIn("Legal candidates (derived)", markdown)
        self.assertIn("2.000000 s shorter", markdown)
        self.assertIn("4.000000 s shorter", markdown)

    def test_rejects_controlled_pair_seed_mismatch_before_reporting(self):
        with tempfile.TemporaryDirectory() as temporary:
            position_config = json.loads(POSITION_CONFIG.read_text(encoding="utf-8"))
            position_config["random_seed"] = 99
            position_config_path = self._write_json(
                temporary, "position_config.json", position_config
            )
            legacy_metrics = self._write_json(
                temporary, "legacy_metrics.json", self._metrics("legacy_pose5d")
            )
            position_metrics = self._write_json(
                temporary, "position_metrics.json", self._metrics("position_only")
            )
            legacy_status = self._write_status(
                temporary, "legacy_status.txt", 20
            )
            position_status = self._write_status(
                temporary, "position_status.txt", 16
            )
            with self.assertRaisesRegex(ComparisonError, "random_seed"):
                build_comparison(
                    legacy_config_path=LEGACY_CONFIG,
                    position_config_path=position_config_path,
                    legacy_metrics_path=legacy_metrics,
                    position_metrics_path=position_metrics,
                    legacy_status_path=legacy_status,
                    position_status_path=position_status,
                    legacy_manifest_path=Path(temporary) / "unused-legacy-manifest",
                    position_manifest_path=Path(temporary) / "unused-position-manifest",
                    profiles_dir=PROFILES_DIR,
                )

    def test_rejects_position_only_orientation_proposals(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ComparisonError, "zero orientation"):
                self._build(
                    temporary,
                    position_metrics=self._metrics(
                        "position_only", orientation_proposals=1
                    ),
                )

    def test_longer_position_run_is_not_presented_as_a_speedup(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self._build(
                temporary,
                position_metrics=self._metrics(
                    "position_only", trajectory_seconds=15.0
                ),
                position_wall_seconds=25,
            )
        interpretation = report["wall_time_interpretation"]
        self.assertEqual(interpretation["direction"], "position_only_longer")
        self.assertIn("5.000000 s longer", interpretation["statement"])
        self.assertIn(
            "3.000000 s longer",
            report["trajectory_time_interpretation"]["statement"],
        )

    def test_cli_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            legacy_run = Path(temporary) / "legacy-run"
            position_run = Path(temporary) / "position-run"
            (legacy_run / "metrics_debug_quick").mkdir(parents=True)
            (position_run / "metrics_debug_quick").mkdir(parents=True)
            legacy_metrics = self._write_json(
                legacy_run / "metrics_debug_quick",
                "legacy_metrics.json",
                self._metrics("legacy_pose5d"),
            )
            position_metrics = self._write_json(
                position_run / "metrics_debug_quick",
                "position_metrics.json",
                self._metrics("position_only"),
            )
            legacy_status = self._write_status(
                legacy_run, "status.txt", 20
            )
            position_status = self._write_status(
                position_run, "status.txt", 16
            )
            legacy_manifest = self._write_manifest(
                legacy_run, LEGACY_CONFIG, legacy=True
            )
            position_manifest = self._write_manifest(
                position_run, POSITION_CONFIG, legacy=False
            )
            output_json = Path(temporary) / "comparison.json"
            output_markdown = Path(temporary) / "comparison.md"
            return_code = main(
                [
                    "--legacy-config",
                    str(LEGACY_CONFIG),
                    "--position-config",
                    str(POSITION_CONFIG),
                    "--legacy-metrics",
                    str(legacy_metrics),
                    "--position-metrics",
                    str(position_metrics),
                    "--legacy-status",
                    str(legacy_status),
                    "--position-status",
                    str(position_status),
                    "--legacy-manifest",
                    str(legacy_manifest),
                    "--position-manifest",
                    str(position_manifest),
                    "--output-json",
                    str(output_json),
                    "--output-markdown",
                    str(output_markdown),
                ]
            )
            self.assertEqual(return_code, 0)
            self.assertTrue(output_json.is_file())
            self.assertTrue(output_markdown.is_file())
            self.assertEqual(
                json.loads(output_json.read_text(encoding="utf-8"))["schema_version"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
