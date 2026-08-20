import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from register_pioneer_experiment import _parser, register_experiment  # noqa: E402


BASE_COMMIT = "1" * 40
FINAL_COMMIT = "2" * 40


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _base_registry():
    return {
        "schema_version": "1.0",
        "generated_at": "2026-08-19T00:00:00+00:00",
        "record_count": 1,
        "category_counts": {"real_mesh_smoke": 1},
        "status_counts": {"PASS": 1},
        "experiments": [
            {
                "experiment_id": "OLD-EIFFEL",
                "category": "real_mesh_smoke",
                "planner": "MAGICIAN",
                "status": "PASS",
                "provenance": {"provenance_grade": "A"},
            }
        ],
        "artifact_sets": {},
        "provenance_grade_counts": {"A": 1},
    }


def _base_markdown():
    return """# Registry

- Experiment records：**1**
- Provenance grades：**A=1 / B=0 / C=0**

| Category | Count |
|---|---:|
| real_mesh_smoke | 1 |
"""


class RegisterPioneerExperimentTests(unittest.TestCase):
    def _fixture(self, temporary):
        root = Path(temporary)
        registry_json = root / "registry.json"
        registry_md = root / "registry.md"
        run_dir = root / "run"
        run_dir.mkdir()
        _write_json(registry_json, _base_registry())
        registry_md.write_text(_base_markdown(), encoding="utf-8")
        (run_dir / "manifest.txt").write_text(
            "planner=pioneer\nobservation_mode=cubemap6\nscene=eiffel\nconfig=config.json\n",
            encoding="utf-8",
        )
        (run_dir / "status.txt").write_text(
            "started_at_utc=2026-08-20T00:00:00Z\nexit_code=0\n",
            encoding="utf-8",
        )
        (run_dir / "run.log").write_text("verified run bytes\n", encoding="utf-8")
        _write_json(
            run_dir / "config.json",
            {
                "planning_observation_mode": "cubemap6",
                "pioneer_face_size": 128,
                "pioneer_face_fov_degrees": 90,
                "random_seed": 8,
                "torch_seed": 9,
                "experiment_budget_observations": 3,
                "beam_width": 3,
                "beam_steps": 3,
                "validation_n_proxy_points": 100000,
                "kind_depth_map": "GT",
                "debug_only": True,
                "coverage_comparable": False,
            },
        )
        online = run_dir / "metrics" / "pioneer_eiffel_0.online.json"
        capture_dir = root / "capture" / "training" / "0" / "frames"
        capture_dir.mkdir(parents=True)
        (capture_dir / "bundle-000-front.pt").write_bytes(b"six-face-capture")
        _write_json(
            online,
            {
                "planner": "pioneer",
                "scene": "eiffel",
                "start_index": 0,
                "capture_dir": str(capture_dir),
                "run": {
                    "seed": 8,
                    "torch_seed": 9,
                    "budget_observations": 3,
                    "planning_observation_mode": "cubemap6",
                    "pioneer_face_count": 6,
                    "pioneer_face_size": 128,
                    "pioneer_face_fov_degrees": 90,
                    "coverage_comparable": False,
                },
                "coverage": [{"raw": 12.0, "normalized": 0.25}],
                "trajectory": {
                    "observation_count": 3,
                    "path_length_scene_units": 4.5,
                    "final_point_count": 1200,
                },
                "latency": {
                    "trajectory_seconds": 12.5,
                    "provider_seconds": 1.5,
                    "geometry_seconds": 2.5,
                },
                "cuda": {"peak_allocated_mib": 1000.0, "peak_reserved_mib": 1100.0},
                "pioneer_observation": {
                    "bundle_count": 3,
                    "real_face_render_count": 18,
                    "imagined_bundle_render_count": 10,
                    "imagined_face_render_count": 60,
                    "imagined_history_bundle_render_count": 1,
                    "imagined_history_face_render_count": 6,
                    "imagined_candidate_bundle_render_count": 9,
                    "imagined_candidate_face_render_count": 54,
                    "bundles": [
                        {
                            "bundle_id": bundle_id,
                            "face_count": 6,
                            "face_names": [
                                "front",
                                "back",
                                "left",
                                "right",
                                "up",
                                "down",
                            ],
                        }
                        for bundle_id in range(3)
                    ],
                },
            },
        )
        return registry_json, registry_md, run_dir, online

    def test_append_hash_and_same_id_update_are_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry_json, registry_md, run_dir, online = self._fixture(temporary)
            lmdb_dir = Path(temporary) / "lmdb"
            lmdb_dir.mkdir()
            (lmdb_dir / "data.mdb").write_bytes(b"lmdb-bytes")
            kwargs = {
                "registry_json": registry_json,
                "registry_md": registry_md,
                "run_dir": run_dir,
                "online_metrics": online,
                "experiment_id": "PAN-10-PIONEER-EIFFEL-QUICK",
                "scientific_run_commit": BASE_COMMIT,
                "final_branch_commit": FINAL_COMMIT,
                "status": "PASS",
                "artifact_roots": [lmdb_dir],
            }
            first = register_experiment(**kwargs)
            self.assertEqual(first["status"], "PASS")
            self.assertEqual(first["multica_issue"], "PAN-10")
            self.assertEqual(first["notes"], "Debug-only coverage is not comparable.")
            self.assertTrue(first["pioneer"]["cubemap6"])
            self.assertEqual(first["proxy_points"], 100000)
            self.assertEqual(first["pioneer"]["render_counts"]["imagined_face_render_count"], 60)
            self.assertEqual(first["pioneer"]["vram"]["peak_reserved_mib"], 1100.0)
            self.assertFalse(first["pioneer"]["coverage"]["comparable"])
            self.assertTrue(first["pioneer"]["cubemap6_metrics_verified"])

            registry = json.loads(registry_json.read_text(encoding="utf-8"))
            self.assertEqual(registry["record_count"], 2)
            self.assertEqual(registry["category_counts"]["real_mesh_smoke"], 2)
            self.assertEqual(registry["status_counts"]["PASS"], 2)
            artifact_set = registry["artifact_sets"][first["artifacts"]]
            self.assertEqual(artifact_set["issue"], "PAN-10")
            self.assertNotIn("planner_search", first["pioneer"])
            self.assertNotIn(
                "pioneer_planner_state_mode",
                first["provenance"]["normalized_config"],
            )
            log_artifact = artifact_set["artifacts"]["run.log"]
            expected_hash = hashlib.sha256((run_dir / "run.log").read_bytes()).hexdigest()
            self.assertEqual(log_artifact["sha256"], expected_hash)
            capture_dir = Path(
                json.loads(online.read_text(encoding="utf-8"))["capture_dir"]
            )
            capture_artifacts = [
                artifact
                for artifact in artifact_set["artifacts"].values()
                if artifact["path"]
                == str((capture_dir / "bundle-000-front.pt").resolve())
            ]
            self.assertEqual(len(capture_artifacts), 1)
            self.assertEqual(
                capture_artifacts[0]["sha256"],
                hashlib.sha256(b"six-face-capture").hexdigest(),
            )
            lmdb_artifacts = [
                artifact
                for artifact in artifact_set["artifacts"].values()
                if artifact["path"] == str((lmdb_dir / "data.mdb").resolve())
            ]
            self.assertEqual(len(lmdb_artifacts), 1)
            self.assertEqual(
                lmdb_artifacts[0]["sha256"],
                hashlib.sha256(b"lmdb-bytes").hexdigest(),
            )
            normalized = first["provenance"]["normalized_config"]
            expected_semantic_hash = hashlib.sha256(
                json.dumps(
                    normalized,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(
                first["provenance"]["normalized_config_sha256"],
                expected_semantic_hash,
            )

            metrics = json.loads(online.read_text(encoding="utf-8"))
            metrics["coverage"][-1]["normalized"] = 0.5
            _write_json(online, metrics)
            second = register_experiment(**kwargs)
            registry = json.loads(registry_json.read_text(encoding="utf-8"))
            self.assertEqual(second["final_normalized_coverage"], 0.5)
            self.assertEqual(
                sum(
                    record["experiment_id"] == "PAN-10-PIONEER-EIFFEL-QUICK"
                    for record in registry["experiments"]
                ),
                1,
            )
            markdown = registry_md.read_text(encoding="utf-8")
            self.assertEqual(markdown.count("PIONEER_LIVE_SECTION_START"), 1)
            self.assertEqual(markdown.count("| PAN-10-PIONEER-EIFFEL-QUICK |"), 1)
            self.assertIn("| real_mesh_smoke | 2 |", markdown)
            self.assertIn("- Provenance grades：**A=2 / B=0 / C=0**", markdown)
            self.assertIn(
                "- Experiment records：**2**（1 historical v1.0 IDs + 1 live PIONEER IDs）",
                markdown,
            )

    def test_pan11_retains_issue_and_position_only_search_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry_json, registry_md, run_dir, online = self._fixture(temporary)
            metrics = json.loads(online.read_text(encoding="utf-8"))
            metrics["run"].update(
                {
                    "pioneer_planner_state_mode": "position_only",
                    "planner_state_dimension": 3,
                    "pioneer_cubemap_rig_frame": "world",
                    "pioneer_cubemap_extrinsics_version": (
                        "pytorch3d-world-axes-v1"
                    ),
                    "pioneer_canonical_orientation_indices": [2, 0],
                }
            )
            expected_totals = {
                "parent_beam_count": 4,
                "raw_action_proposal_count": 24,
                "translation_action_proposal_count": 24,
                "orientation_action_proposal_count": 0,
                "generated_candidate_count": 20,
                "valid_state_candidate_count": 18,
                "observed_rejected_candidate_count": 2,
                "collision_rejected_candidate_count": 3,
                "rendered_candidate_count": 9,
                "retained_beam_count": 6,
                "search_seconds": 0.75,
            }
            metrics["planner_search"] = {
                "schema_version": 1,
                "state_mode": "position_only",
                "state_dimension": 3,
                "cubemap_rig_frame": "world",
                "cubemap_extrinsics_version": "pytorch3d-world-axes-v1",
                "totals": expected_totals,
            }
            _write_json(online, metrics)

            record = register_experiment(
                registry_json=registry_json,
                registry_md=registry_md,
                run_dir=run_dir,
                online_metrics=online,
                experiment_id="PAN-11-PIONEER-EIFFEL-POSITION-ONLY",
                scientific_run_commit=BASE_COMMIT,
                final_branch_commit=FINAL_COMMIT,
                status="PASS",
                issue="PAN-11",
            )

            self.assertEqual(record["status"], "PASS")
            self.assertEqual(record["multica_issue"], "PAN-11")
            self.assertEqual(
                record["notes"],
                "Issue provenance: PAN-11. Debug-only coverage is not comparable.",
            )
            self.assertEqual(record["provenance"]["issue"], "PAN-11")
            self.assertEqual(
                record["pioneer"]["pioneer_planner_state_mode"], "position_only"
            )
            self.assertEqual(record["pioneer"]["planner_state_dimension"], 3)
            self.assertEqual(
                record["pioneer"]["pioneer_cubemap_rig_frame"], "world"
            )
            self.assertEqual(
                record["pioneer"]["pioneer_cubemap_extrinsics_version"],
                "pytorch3d-world-axes-v1",
            )
            self.assertEqual(
                record["pioneer"]["pioneer_canonical_orientation_indices"],
                [2, 0],
            )
            self.assertTrue(record["pioneer"]["pan11_contract_verified"])
            self.assertEqual(
                record["pioneer"]["planner_search"]["totals"], expected_totals
            )
            self.assertEqual(
                record["pioneer"]["planner_search"]["state_mode"],
                "position_only",
            )
            self.assertEqual(
                record["provenance"]["normalized_config"][
                    "pioneer_planner_state_mode"
                ],
                "position_only",
            )
            registry = json.loads(registry_json.read_text(encoding="utf-8"))
            artifact_set = registry["artifact_sets"][record["artifacts"]]
            self.assertEqual(artifact_set["issue"], "PAN-11")
            self.assertIn(
                "| PAN-11-PIONEER-EIFFEL-POSITION-ONLY |",
                registry_md.read_text(encoding="utf-8"),
            )

            metrics["run"]["pioneer_cubemap_extrinsics_version"] = "arbitrary-v1"
            metrics["planner_search"]["cubemap_extrinsics_version"] = "arbitrary-v1"
            _write_json(online, metrics)
            invalid_version = register_experiment(
                registry_json=registry_json,
                registry_md=registry_md,
                run_dir=run_dir,
                online_metrics=online,
                experiment_id="PAN-11-PIONEER-EIFFEL-INVALID-VERSION",
                scientific_run_commit=BASE_COMMIT,
                final_branch_commit=FINAL_COMMIT,
                status="PASS",
                issue="PAN-11",
            )
            self.assertEqual(invalid_version["status"], "UNKNOWN")
            self.assertFalse(
                invalid_version["pioneer"]["pan11_contract_verified"]
            )

            metrics["run"]["pioneer_cubemap_extrinsics_version"] = (
                "pytorch3d-world-axes-v1"
            )
            metrics["planner_search"]["cubemap_extrinsics_version"] = (
                "pytorch3d-world-axes-v1"
            )
            del metrics["planner_search"]["totals"]["rendered_candidate_count"]
            _write_json(online, metrics)
            missing_counter = register_experiment(
                registry_json=registry_json,
                registry_md=registry_md,
                run_dir=run_dir,
                online_metrics=online,
                experiment_id="PAN-11-PIONEER-EIFFEL-MISSING-COUNTER",
                scientific_run_commit=BASE_COMMIT,
                final_branch_commit=FINAL_COMMIT,
                status="PASS",
                issue="PAN-11",
            )
            self.assertEqual(missing_counter["status"], "UNKNOWN")

    def test_cli_issue_defaults_to_pan10_and_accepts_pan11(self):
        required = [
            "--registry-json",
            "registry.json",
            "--registry-md",
            "registry.md",
            "--run-dir",
            "run",
            "--online-metrics",
            "metrics.json",
            "--experiment-id",
            "experiment",
            "--scientific-run-commit",
            BASE_COMMIT,
            "--final-branch-commit",
            FINAL_COMMIT,
            "--status",
            "PASS",
        ]
        self.assertEqual(_parser().parse_args(required).issue, "PAN-10")
        self.assertEqual(
            _parser().parse_args([*required, "--issue", "PAN-11"]).issue,
            "PAN-11",
        )

    def test_requested_pass_is_withheld_when_evidence_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_json = root / "registry.json"
            registry_md = root / "registry.md"
            run_dir = root / "run"
            run_dir.mkdir()
            _write_json(registry_json, _base_registry())
            registry_md.write_text(_base_markdown(), encoding="utf-8")
            (run_dir / "manifest.txt").write_text(
                "planner=pioneer\nobservation_mode=cubemap6\nscene=eiffel\n",
                encoding="utf-8",
            )

            record = register_experiment(
                registry_json=registry_json,
                registry_md=registry_md,
                run_dir=run_dir,
                online_metrics=root / "missing.online.json",
                experiment_id="PAN-10-MISSING-EVIDENCE",
                scientific_run_commit="unknown",
                final_branch_commit="unknown",
                status="PASS",
            )
            self.assertEqual(record["status"], "UNKNOWN")
            self.assertIsNone(record["final_normalized_coverage"])
            self.assertIsNone(record["pioneer"]["timing"]["trajectory_seconds"])
            self.assertFalse(
                record["provenance"]["provenance_checks"]["completion_status_evidence"]
            )
            self.assertIn("No PASS claim", record["conclusion"])
            registry = json.loads(registry_json.read_text(encoding="utf-8"))
            self.assertEqual(registry["status_counts"]["UNKNOWN"], 1)

    def test_requested_pass_is_withheld_without_measured_bundle_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry_json, registry_md, run_dir, online = self._fixture(temporary)
            metrics = json.loads(online.read_text(encoding="utf-8"))
            del metrics["pioneer_observation"]["bundles"]
            _write_json(online, metrics)

            record = register_experiment(
                registry_json=registry_json,
                registry_md=registry_md,
                run_dir=run_dir,
                online_metrics=online,
                experiment_id="PAN-10-INCOMPLETE-BUNDLE-EVIDENCE",
                scientific_run_commit=BASE_COMMIT,
                final_branch_commit=FINAL_COMMIT,
                status="PASS",
            )
            self.assertEqual(record["status"], "UNKNOWN")
            self.assertFalse(record["pioneer"]["cubemap6_metrics_verified"])

    def test_pass_conclusion_is_neutral_across_observation_budgets(self):
        for budget, debug_profile in ((3, "quick"), (50, "pioneer-50")):
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as temporary:
                registry_json, registry_md, run_dir, online = self._fixture(temporary)
                config_path = run_dir / "config.json"
                config = json.loads(config_path.read_text(encoding="utf-8"))
                config["experiment_budget_observations"] = budget
                config["debug_profile"] = debug_profile
                _write_json(config_path, config)

                metrics = json.loads(online.read_text(encoding="utf-8"))
                metrics["run"]["budget_observations"] = budget
                metrics["run"]["debug_profile"] = debug_profile
                metrics["trajectory"]["observation_count"] = budget
                metrics["pioneer_observation"]["bundle_count"] = budget
                metrics["pioneer_observation"]["real_face_render_count"] = 6 * budget
                metrics["pioneer_observation"]["bundles"] = [
                    {
                        "bundle_id": bundle_id,
                        "face_count": 6,
                        "face_names": [
                            "front",
                            "back",
                            "left",
                            "right",
                            "up",
                            "down",
                        ],
                    }
                    for bundle_id in range(budget)
                ]
                _write_json(online, metrics)

                record = register_experiment(
                    registry_json=registry_json,
                    registry_md=registry_md,
                    run_dir=run_dir,
                    online_metrics=online,
                    experiment_id=f"PAN-10-PIONEER-EIFFEL-{budget}OBS",
                    scientific_run_commit=BASE_COMMIT,
                    final_branch_commit=FINAL_COMMIT,
                    status="PASS",
                )

                self.assertEqual(record["status"], "PASS")
                self.assertEqual(
                    record["conclusion"],
                    "Verified PIONEER six-face cubemap run completed with exit code 0.",
                )
                self.assertNotIn("quick", record["conclusion"].lower())


if __name__ == "__main__":
    unittest.main()
