import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


import scripts.register_pan29_derived_artifact as registry_module
from scripts.register_pan29_derived_artifact import register_derived_artifact


IDS = (0, 5, 10, 14, 19)
FACES = ("front", "back", "left", "right", "up", "down")
SOURCE_ID = "PAN-11-PIONEER-HKUST-POSITION-ONLY-20OBS-20260821"
DERIVED_ID = "PAN-29-HKUST-GT20OBS-UE5-REPLAY-PREVIEW-20260822"
SOURCE_COMMIT = "d" * 40
PAN29_COMMIT = "e" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _file_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _tree_hash(replays: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in replays:
        digest.update(f"{row['observation_id']:06d}".encode("ascii"))
        digest.update(b"\0")
        digest.update(row["erp"]["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_tree_hash(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_sha256sums(root: Path) -> Path:
    output = root / "SHA256SUMS"
    rows = []
    for path in sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate != output
    ):
        rows.append(f"{_sha256(path)}  ./{path.relative_to(root).as_posix()}")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output


def _make_registry(root: Path, source_artifacts: list[dict]) -> tuple[Path, Path]:
    source_artifacts = {
        f"source-{index}": dict(value)
        for index, value in enumerate(source_artifacts)
    }
    registry = {
        "schema_version": "1.0",
        "generated_at": "2026-08-21T00:00:00+00:00",
        "record_count": 2,
        "category_counts": {"real_mesh_smoke": 2},
        "status_counts": {"PASS": 2},
        "experiments": [
            {
                "experiment_id": "OLD-EIFFEL",
                "category": "real_mesh_smoke",
                "status": "PASS",
                "provenance": {"provenance_grade": "A"},
            },
            {
                "experiment_id": SOURCE_ID,
                "category": "real_mesh_smoke",
                "status": "PASS",
                "artifacts": "ART-SOURCE-PAN11",
                "provenance": {
                    "provenance_grade": "A",
                    "scientific_run_commit": SOURCE_COMMIT,
                    "scientific_run_commit_full": SOURCE_COMMIT,
                },
            },
        ],
        "artifact_sets": {
            "ART-SOURCE-PAN11": {
                "experiment_id": SOURCE_ID,
                "issue": "PAN-11",
                "hash_semantics": "SHA256 of the live artifact file bytes",
                "artifacts": source_artifacts,
            }
        },
        "provenance_grade_counts": {"A": 2},
        "repair_traceability_counts": {"complete": 2},
    }
    registry_json = root / "registry.json"
    registry_md = root / "registry.md"
    _write_json(registry_json, registry)
    registry_md.write_text(
        "# Pioneer Registry\n\n"
        "- Experiment records: **2**\n\n"
        "## Scientific experiments\n\n"
        "Existing scientific content.\n",
        encoding="utf-8",
    )
    return registry_json, registry_md


def _make_pan29_artifacts(root: Path) -> dict:
    source_root = root / "source"
    source_root.mkdir(parents=True)
    source_run_manifest = source_root / "manifest.txt"
    source_config = source_root / "config.json"
    source_metrics = source_root / "metrics.json"
    source_preview = source_root / "preview.png"
    source_preview_sidecar = source_root / "preview.json"
    source_lmdb = source_root / "lmdb"
    source_lmdb.mkdir()
    (source_lmdb / "data.mdb").write_bytes(b"pan11-lmdb")
    source_config.write_bytes(b"pan11-config")
    source_metrics.write_bytes(b"pan11-metrics")
    source_preview.write_bytes(b"pan11-preview")
    source_run_manifest.write_text(
        f"git_commit={SOURCE_COMMIT}\n"
        f"config_sha256={_sha256(source_config)}\n"
        "debug_profile=pioneer-20\n"
        "expected_observations=20\n"
        "expected_real_face_renders=120\n",
        encoding="utf-8",
    )

    capture_root = source_root / "capture"
    all_frame_paths = []
    all_image_paths = []
    records_by_id = {}
    observation_rows = []
    for observation_id in range(20):
        frame_records = []
        image_records = []
        frame_dir = capture_root / "frames" / f"{observation_id:06d}"
        image_dir = capture_root / "imgs" / f"{observation_id:06d}"
        frame_dir.mkdir(parents=True)
        image_dir.mkdir(parents=True)
        for name in ("bundle", *FACES):
            path = frame_dir / f"{name}.pt"
            path.write_bytes(f"frame-{observation_id}-{name}".encode())
            all_frame_paths.append(path)
            record = _file_record(path)
            record["path"] = path.relative_to(capture_root).as_posix()
            record["bytes"] = record.pop("size_bytes")
            frame_records.append(record)
        for face in FACES:
            path = image_dir / f"{face}.png"
            path.write_bytes(f"image-{observation_id}-{face}".encode())
            all_image_paths.append(path)
            record = _file_record(path)
            record["path"] = path.relative_to(capture_root).as_posix()
            record["bytes"] = record.pop("size_bytes")
            image_records.append(record)
        records_by_id[observation_id] = (frame_records, image_records)
    _write_json(
        source_preview_sidecar,
        {
            "schema_version": 1,
            "sources": {
                "capture_image_count": 120,
                "capture_images_root": str(capture_root / "imgs"),
                "capture_images_tree_sha256": _file_tree_hash(
                    all_image_paths, capture_root / "imgs"
                ),
                "metrics": _file_record(source_metrics),
                "lmdb": {
                    "path": str(source_lmdb),
                    "data_sha256": _sha256(source_lmdb / "data.mdb"),
                },
            },
        },
    )
    for observation_id in IDS:
        frame_records, image_records = records_by_id[observation_id]
        observation_rows.append(
            {
                "observation_id": observation_id,
                "source_capture_timestamp_ns": 1000 + observation_id,
                "source_capture_timestamp_utc": f"time-{observation_id}",
                "planner_position_scene_units": [observation_id, 2, 3],
                "ue_position_cm": [500 * observation_id, 1500, 1000],
                "within_pan13_conservative_fly_volume": observation_id == 0,
                "source_artifacts": {
                    "frames": frame_records,
                    "images": image_records,
                    "transaction_marker": None,
                },
            }
        )

    plan = {
        "schema_version": "pan29.ue5-postrun-replay.v1",
        "task": "PAN-29",
        "artifact_role": "post_run_visualization_only",
        "planner_input_unchanged": True,
        "ue5_render_is_postrun_visualization_only": True,
        "selected_bundle_ids": list(IDS),
        "source": {
            "registry_experiment_id": SOURCE_ID,
            "scientific_run_commit": SOURCE_COMMIT,
            "run_dir": str(source_root),
            "run_manifest": _file_record(source_run_manifest),
            "config": _file_record(source_config),
            "metrics": _file_record(source_metrics),
            "capture_root": str(capture_root),
            "lmdb": {
                "path": str(source_lmdb),
                "data_sha256": _sha256(source_lmdb / "data.mdb"),
                "key": "HKUST/0",
            },
            "original_preview": {
                "path": str(source_preview),
                "sha256": _sha256(source_preview),
                "sidecar_path": str(source_preview_sidecar),
                "sidecar_sha256": _sha256(source_preview_sidecar),
            },
            "depth_source": "GT",
            "scene": "HKUST",
            "planner_state_mode": "position_only",
        },
        "observations": observation_rows,
    }
    plan_path = root / "replay_plan.json"
    _write_json(plan_path, plan)

    run_manifest = root / "run_manifest.txt"
    run_manifest.write_text(
        f"git_commit={PAN29_COMMIT}\n"
        f"ue_project_sha256={'1' * 64}\n"
        f"ue_default_engine_sha256={'2' * 64}\n"
        f"ue_level_sha256={'3' * 64}\n"
        f"capture_config_sha256={'4' * 64}\n",
        encoding="utf-8",
    )
    replays = []
    for observation_id in IDS:
        processed = root / "processed" / f"{observation_id:06d}"
        processed.mkdir(parents=True)
        raw = processed / "raw.json"
        canonical = processed / "canonical.json"
        erp = processed / "erp.png"
        raw.write_bytes(f"raw-{observation_id}".encode())
        canonical.write_bytes(f"canonical-{observation_id}".encode())
        erp.write_bytes(f"erp-{observation_id}".encode())
        receipt = {
            "schema_version": "pan29.ue5-replay-receipt.v1",
            "result": "PASS",
            "observation_id": observation_id,
            "source_capture_timestamp_ns": 1000 + observation_id,
            "ue5_actual_capture_timestamp_ns": 2000 + observation_id,
            "planner_position_scene_units": [observation_id, 2, 3],
            "requested_position_ue_cm": [500 * observation_id, 1500, 1000],
            "position_max_abs_error_scene_units": 0.0,
            "face_basis_max_abs_error": 0.0,
            "face_names": list(FACES),
            "within_pan13_conservative_fly_volume": observation_id == 0,
            "planner_input_unchanged": True,
            "ue5_role": "post_run_visualization_only",
            "sources": {
                "plan": _file_record(plan_path),
                "raw_manifest": _file_record(raw),
                "canonical_manifest": _file_record(canonical),
            },
            "erp": {
                "path": "erp.png",
                "sha256": _sha256(erp),
                "width": 1024,
                "height": 512,
                "byte_deterministic_reprojection": True,
            },
        }
        receipt_path = processed / "receipt.json"
        _write_json(receipt_path, receipt)
        replays.append(
            {
                "observation_id": observation_id,
                "receipt": _file_record(receipt_path),
                "erp": {
                    **receipt["erp"],
                    "path": str(erp.resolve()),
                },
                "source_capture_timestamp_ns": 1000 + observation_id,
                "ue5_actual_capture_timestamp_ns": 2000 + observation_id,
                "within_pan13_conservative_fly_volume": observation_id == 0,
            }
        )
    result = {
        "schema_version": "pan29.ue5-postrun-replay-result.v1",
        "task": "PAN-29",
        "result": "PASS",
        "artifact_role": "post_run_visualization_only",
        "planner_input_unchanged": True,
        "observation_ids": list(IDS),
        "replay_count": 5,
        "out_of_pan13_fly_policy_ids": [5, 10, 14, 19],
        "source_scientific_run_commit": SOURCE_COMMIT,
        "implementation_commit": PAN29_COMMIT,
        "run_manifest": {
            **_file_record(run_manifest),
            "ue_project_sha256": "1" * 64,
            "ue_default_engine_sha256": "2" * 64,
            "ue_level_sha256": "3" * 64,
            "capture_config_sha256": "4" * 64,
        },
        "plan": _file_record(plan_path),
        "erp_tree_sha256": _tree_hash(replays),
        "replays": replays,
        "claims": {
            "same_observation_id_pose_source_timestamp": True,
            "same_wall_clock_capture_time": False,
            "rgb_pixel_parity": False,
            "depth_parity": False,
            "coverage_parity": False,
        },
    }
    result_path = root / "replay_result.json"
    _write_json(result_path, result)

    preview = root / "preview" / "pan29.png"
    preview.parent.mkdir()
    preview.write_bytes(b"pan29-preview")
    preview_sidecar = {
        "schema_version": "pan29.preview.v1",
        "task": "PAN-29",
        "preview": {
            **_file_record(preview),
            "width": 2240,
            "height": 1948,
            "mode": "RGB",
        },
        "sources": {
            "metrics": _file_record(source_metrics),
            "replay_plan": _file_record(plan_path),
            "replay_result": _file_record(result_path),
            "original_preview": _file_record(source_preview),
            "all_source_face_count": 120,
            "all_source_face_tree_sha256": "5" * 64,
            "displayed_source_face_count": 30,
            "displayed_source_face_tree_sha256": "6" * 64,
            "ue5_erp_count": 5,
            "ue5_erp_tree_sha256": "7" * 64,
        },
        "summary": {
            "scene": "HKUST",
            "source_depth": "GT",
            "source_observation_count": 20,
            "displayed_observation_ids": list(IDS),
            "planner_face_image_count": 30,
            "ue5_erp_count": 5,
            "planner_input_unchanged": True,
            "ue5_role": "post_run_visualization_only",
            "out_of_pan13_fly_policy_ids": [5, 10, 14, 19],
        },
    }
    sidecar_path = preview.with_suffix(".json")
    _write_json(sidecar_path, preview_sidecar)
    preview_commit = {
        "schema_version": "pan29.preview-commit.v1",
        "preview_path": str(preview.resolve()),
        "preview_sha256": _sha256(preview),
        "sidecar_path": str(sidecar_path.resolve()),
        "sidecar_sha256": _sha256(sidecar_path),
    }
    commit_path = preview.with_suffix(".commit.json")
    _write_json(commit_path, preview_commit)
    run_log = root / "run.log"
    preview_log = root / "preview.log"
    status = root / "status.txt"
    listed_extra = root / "captures" / "000000" / "ue_capture.log"
    listed_extra.parent.mkdir(parents=True)
    run_log.write_text("PAN-29 replay completed\n", encoding="utf-8")
    preview_log.write_text("PAN-29 preview completed\n", encoding="utf-8")
    listed_extra.write_text("UE capture evidence\n", encoding="utf-8")
    status.write_text(
        "started_at_utc=2026-08-22T00:00:00Z\n"
        "snapshot_integrity_preflight=PASS\n"
        "snapshot_integrity_postflight=PASS\n"
        "finished_at_utc=2026-08-22T00:01:00Z\n"
        "exit_code=0\n",
        encoding="utf-8",
    )
    sha256sums = _write_sha256sums(root)
    source_registry_artifacts = [
        _file_record(source_run_manifest),
        _file_record(source_config),
        _file_record(source_metrics),
        _file_record(source_lmdb / "data.mdb"),
        _file_record(source_preview),
        _file_record(source_preview_sidecar),
        *(_file_record(path) for path in all_frame_paths),
    ]
    return {
        "result": result_path,
        "sidecar": sidecar_path,
        "commit": commit_path,
        "source_registry_artifacts": source_registry_artifacts,
        "erp": root / "processed" / "000010" / "erp.png",
        "status": status,
        "sha256sums": sha256sums,
        "run_log": run_log,
        "preview_log": preview_log,
        "listed_extra": listed_extra,
        "unselected_source_image": capture_root / "imgs" / "000007" / "down.png",
    }


class RegisterPAN29DerivedArtifactTests(unittest.TestCase):
    def test_registers_derived_receipt_without_mutating_scientific_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _make_pan29_artifacts(root / "attempt-001")
            registry_json, registry_md = _make_registry(
                root, artifacts["source_registry_artifacts"]
            )
            before = json.loads(registry_json.read_text(encoding="utf-8"))
            source_paths = [
                row["path"]
                for row in before["artifact_sets"]["ART-SOURCE-PAN11"][
                    "artifacts"
                ].values()
            ]
            self.assertEqual(sum("/frames/" in path for path in source_paths), 140)
            self.assertEqual(sum("/imgs/" in path for path in source_paths), 0)
            protected = {
                key: copy.deepcopy(before[key])
                for key in (
                    "record_count",
                    "category_counts",
                    "status_counts",
                    "experiments",
                    "artifact_sets",
                    "provenance_grade_counts",
                    "repair_traceability_counts",
                )
            }

            record = register_derived_artifact(
                registry_json=registry_json,
                registry_md=registry_md,
                replay_result_path=artifacts["result"],
                preview_sidecar_path=artifacts["sidecar"],
                preview_commit_path=artifacts["commit"],
                derived_artifact_id=DERIVED_ID,
            )

            registry = json.loads(registry_json.read_text(encoding="utf-8"))
            self.assertEqual(
                {key: registry[key] for key in protected}, protected
            )
            self.assertEqual(len(registry["derived_artifacts"]), 1)
            self.assertEqual(record, registry["derived_artifacts"][0])
            self.assertEqual(record["derived_artifact_id"], DERIVED_ID)
            self.assertEqual(record["source_experiment_id"], SOURCE_ID)
            self.assertEqual(record["artifact_role"], "post_run_visualization_only")
            self.assertTrue(record["planner_input_unchanged"])
            self.assertEqual(record["observation_ids"], list(IDS))
            self.assertFalse(record["claims"]["rgb_pixel_parity"])
            self.assertFalse(record["claims"]["depth_parity"])
            self.assertFalse(record["claims"]["coverage_parity"])
            self.assertEqual(
                record["source_scientific_run"]["commit"], SOURCE_COMMIT
            )
            self.assertEqual(
                record["source_scientific_run"]["frame_file_count"], 140
            )
            self.assertEqual(
                record["source_scientific_run"]["image_file_count"], 120
            )
            self.assertEqual(
                len(record["source_scientific_run"]["frames_tree_sha256"]), 64
            )
            self.assertEqual(
                len(record["source_scientific_run"]["images_tree_sha256"]), 64
            )
            self.assertEqual(record["pan29"]["implementation_commit"], PAN29_COMMIT)
            self.assertEqual(len(record["receipts"]), 5)
            artifact_set = registry["derived_artifact_sets"][
                record["derived_artifact_set_id"]
            ]
            self.assertEqual(artifact_set["derived_artifact_id"], DERIVED_ID)
            self.assertEqual(artifact_set["source_experiment_id"], SOURCE_ID)
            self.assertEqual(len(artifact_set["artifacts"]), 30)
            for name in ("status", "run_log", "preview_log", "sha256sums"):
                self.assertIn(name, artifact_set["artifacts"])
            self.assertEqual(record["pan29"]["run_status"]["exit_code"], 0)
            self.assertEqual(
                record["pan29"]["run_status"]["snapshot_integrity_preflight"],
                "PASS",
            )

            markdown = registry_md.read_text(encoding="utf-8")
            self.assertEqual(
                markdown.count("<!-- PIONEER_DERIVED_ARTIFACTS_START -->"), 1
            )
            self.assertEqual(markdown.count(f"| `{DERIVED_ID}` |"), 1)
            self.assertIn("not scientific experiment records", markdown)

            first_json_bytes = registry_json.read_bytes()
            first_md_bytes = registry_md.read_bytes()
            second = register_derived_artifact(
                registry_json=registry_json,
                registry_md=registry_md,
                replay_result_path=artifacts["result"],
                preview_sidecar_path=artifacts["sidecar"],
                preview_commit_path=artifacts["commit"],
                derived_artifact_id=DERIVED_ID,
            )
            self.assertEqual(second, record)
            self.assertEqual(registry_json.read_bytes(), first_json_bytes)
            self.assertEqual(registry_md.read_bytes(), first_md_bytes)

    def test_tampered_artifact_fails_before_any_registry_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _make_pan29_artifacts(root / "attempt-001")
            registry_json, registry_md = _make_registry(
                root, artifacts["source_registry_artifacts"]
            )
            before_json = registry_json.read_bytes()
            before_md = registry_md.read_bytes()
            artifacts["erp"].write_bytes(b"tampered")

            with self.assertRaisesRegex(ValueError, "ERP.*SHA256"):
                register_derived_artifact(
                    registry_json=registry_json,
                    registry_md=registry_md,
                    replay_result_path=artifacts["result"],
                    preview_sidecar_path=artifacts["sidecar"],
                    preview_commit_path=artifacts["commit"],
                    derived_artifact_id=DERIVED_ID,
                )
            self.assertEqual(registry_json.read_bytes(), before_json)
            self.assertEqual(registry_md.read_bytes(), before_md)

    def test_same_id_with_different_valid_content_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _make_pan29_artifacts(root / "attempt-001")
            second = _make_pan29_artifacts(root / "attempt-002")
            registry_json, registry_md = _make_registry(
                root,
                [
                    *first["source_registry_artifacts"],
                    *second["source_registry_artifacts"],
                ],
            )
            register_derived_artifact(
                registry_json=registry_json,
                registry_md=registry_md,
                replay_result_path=first["result"],
                preview_sidecar_path=first["sidecar"],
                preview_commit_path=first["commit"],
                derived_artifact_id=DERIVED_ID,
            )
            before_json = registry_json.read_bytes()
            before_md = registry_md.read_bytes()

            with self.assertRaisesRegex(ValueError, "different canonical content"):
                register_derived_artifact(
                    registry_json=registry_json,
                    registry_md=registry_md,
                    replay_result_path=second["result"],
                    preview_sidecar_path=second["sidecar"],
                    preview_commit_path=second["commit"],
                    derived_artifact_id=DERIVED_ID,
                )
            self.assertEqual(registry_json.read_bytes(), before_json)
            self.assertEqual(registry_md.read_bytes(), before_md)

    def test_failed_status_or_missing_checksums_fail_before_registry_write(self):
        for case in ("status-fail", "duplicate-status-key", "missing-checksums"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                artifacts = _make_pan29_artifacts(root / "attempt-001")
                registry_json, registry_md = _make_registry(
                    root, artifacts["source_registry_artifacts"]
                )
                if case == "status-fail":
                    artifacts["status"].write_text(
                        "started_at_utc=2026-08-22T00:00:00Z\n"
                        "snapshot_integrity_preflight=FAIL\n"
                        "snapshot_integrity_postflight=PASS\n"
                        "finished_at_utc=2026-08-22T00:01:00Z\n"
                        "exit_code=0\n",
                        encoding="utf-8",
                    )
                    _write_sha256sums(artifacts["result"].parent)
                    expected = "status is not a successful finalized run"
                elif case == "duplicate-status-key":
                    artifacts["status"].write_text(
                        "started_at_utc=2026-08-22T00:00:00Z\n"
                        "snapshot_integrity_preflight=PASS\n"
                        "snapshot_integrity_preflight=PASS\n"
                        "snapshot_integrity_postflight=PASS\n"
                        "finished_at_utc=2026-08-22T00:01:00Z\n"
                        "exit_code=0\n",
                        encoding="utf-8",
                    )
                    _write_sha256sums(artifacts["result"].parent)
                    expected = "invalid or duplicate PAN-29 status"
                else:
                    artifacts["sha256sums"].unlink()
                    expected = "SHA256SUMS"
                before_json = registry_json.read_bytes()
                before_md = registry_md.read_bytes()
                with self.assertRaisesRegex((ValueError, FileNotFoundError), expected):
                    register_derived_artifact(
                        registry_json=registry_json,
                        registry_md=registry_md,
                        replay_result_path=artifacts["result"],
                        preview_sidecar_path=artifacts["sidecar"],
                        preview_commit_path=artifacts["commit"],
                        derived_artifact_id=DERIVED_ID,
                    )
                self.assertEqual(registry_json.read_bytes(), before_json)
                self.assertEqual(registry_md.read_bytes(), before_md)

    def test_tampered_sha256sums_listed_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _make_pan29_artifacts(root / "attempt-001")
            registry_json, registry_md = _make_registry(
                root, artifacts["source_registry_artifacts"]
            )
            before_json = registry_json.read_bytes()
            before_md = registry_md.read_bytes()
            artifacts["listed_extra"].write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256SUMS.*ue_capture.log"):
                register_derived_artifact(
                    registry_json=registry_json,
                    registry_md=registry_md,
                    replay_result_path=artifacts["result"],
                    preview_sidecar_path=artifacts["sidecar"],
                    preview_commit_path=artifacts["commit"],
                    derived_artifact_id=DERIVED_ID,
                )
            self.assertEqual(registry_json.read_bytes(), before_json)
            self.assertEqual(registry_md.read_bytes(), before_md)

    def test_unregistered_png_tree_is_bound_by_registered_preview_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _make_pan29_artifacts(root / "attempt-001")
            registry_json, registry_md = _make_registry(
                root, artifacts["source_registry_artifacts"]
            )
            before_json = registry_json.read_bytes()
            before_md = registry_md.read_bytes()
            artifacts["unselected_source_image"].write_bytes(b"tampered-png")
            with self.assertRaisesRegex(
                ValueError, "original preview sidecar differs from the source trees"
            ):
                register_derived_artifact(
                    registry_json=registry_json,
                    registry_md=registry_md,
                    replay_result_path=artifacts["result"],
                    preview_sidecar_path=artifacts["sidecar"],
                    preview_commit_path=artifacts["commit"],
                    derived_artifact_id=DERIVED_ID,
                )
            self.assertEqual(registry_json.read_bytes(), before_json)
            self.assertEqual(registry_md.read_bytes(), before_md)

    def test_second_replace_failure_rolls_back_both_registry_files_byte_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _make_pan29_artifacts(root / "attempt-001")
            registry_json, registry_md = _make_registry(
                root, artifacts["source_registry_artifacts"]
            )
            before_json = registry_json.read_bytes()
            before_md = registry_md.read_bytes()
            calls = 0

            def fail_second_replace(source: Path, target: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected second replace failure")
                source.replace(target)

            with mock.patch.object(
                registry_module, "_replace_file", side_effect=fail_second_replace
            ), self.assertRaisesRegex(OSError, "injected second replace failure"):
                register_derived_artifact(
                    registry_json=registry_json,
                    registry_md=registry_md,
                    replay_result_path=artifacts["result"],
                    preview_sidecar_path=artifacts["sidecar"],
                    preview_commit_path=artifacts["commit"],
                    derived_artifact_id=DERIVED_ID,
                )
            self.assertGreaterEqual(calls, 3)
            self.assertEqual(registry_json.read_bytes(), before_json)
            self.assertEqual(registry_md.read_bytes(), before_md)


if __name__ == "__main__":
    unittest.main()
