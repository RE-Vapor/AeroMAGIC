import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from register_pioneer_experiment import register_experiment  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
