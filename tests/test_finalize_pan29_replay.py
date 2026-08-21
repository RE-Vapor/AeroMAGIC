import hashlib
import json
from pathlib import Path
import tempfile
import unittest


from scripts.finalize_pan29_replay import finalize_replay


IDS = (0, 5, 10, 14, 19)
FACES = ("front", "back", "left", "right", "up", "down")
PIONEER_FACE_ROTATIONS = {
    "front": ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "back": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
    "left": ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    "right": ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
    "up": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "down": ((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
}
CONTRACT_TO_PIONEER = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def transpose(matrix):
    return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))


def matmul(left, right):
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def transform(rotation, translation):
    return [
        [*rotation[0], translation[0]],
        [*rotation[1], translation[1]],
        [*rotation[2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def source_semantic(observation_id, timestamp_ns, timestamp_utc, position):
    faces = [
        {
            "face_name": face_name,
            "T_world_from_cam_rotation": [
                list(row) for row in PIONEER_FACE_ROTATIONS[face_name]
            ],
            "camera_center_scene_units": list(position),
            "camera_center_max_abs_error_scene_units": 0.0,
            "basis_max_abs_error": 0.0,
        }
        for face_name in FACES
    ]
    basis_payload = [
        {
            "face_name": face["face_name"],
            "T_world_from_cam_rotation": face["T_world_from_cam_rotation"],
        }
        for face in faces
    ]
    basis_sha = hashlib.sha256(
        json.dumps(basis_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": "pan29.source-bundle-semantic-validation.v1",
        "result": "PASS",
        "bundle_id": observation_id,
        "source_capture_timestamp_ns": timestamp_ns,
        "source_capture_timestamp_utc": timestamp_utc,
        "camera_center_scene_units": list(position),
        "camera_center_max_abs_error_scene_units": 0.0,
        "face_basis_max_abs_error": 0.0,
        "faces": faces,
    }, basis_sha


class FinalizePAN29ReplayTests(unittest.TestCase):
    def _fixture(self, root: Path):
        source = root / "source"
        source.mkdir()
        config = source / "config.json"
        metrics = source / "metrics.json"
        lmdb = source / "data.mdb"
        preview = source / "preview.png"
        for path in (config, metrics, lmdb, preview):
            path.write_bytes(path.name.encode())
        source_capture = source / "capture"
        source_capture.mkdir()
        rows = []
        for observation_id in IDS:
            source_artifact = source_capture / f"{observation_id:06d}.bin"
            source_artifact.write_bytes(str(observation_id).encode())
            position = [observation_id, 2, 3]
            timestamp_ns = 1000 + observation_id
            timestamp_utc = f"t-{observation_id}"
            semantic, _ = source_semantic(
                observation_id, timestamp_ns, timestamp_utc, position
            )
            rows.append(
                {
                    "observation_id": observation_id,
                    "replay_request_id": f"pan29-test-{observation_id:06d}",
                    "replay_frame_id": f"observation-{observation_id:06d}",
                    "source_capture_timestamp_ns": timestamp_ns,
                    "source_capture_timestamp_utc": timestamp_utc,
                    "planner_position_scene_units": position,
                    "ue_position_cm": [500 * observation_id, 1500, 1000],
                    "within_pan13_conservative_fly_volume": observation_id == 0,
                    "source_semantic_validation": semantic,
                    "source_artifacts": {
                        "frames": [artifact(source_artifact, source_capture)],
                        "images": [],
                        "transaction_marker": None,
                    },
                }
            )
        plan = {
            "schema_version": "pan29.ue5-postrun-replay.v1",
            "selected_bundle_ids": list(IDS),
            "planner_input_unchanged": True,
            "source": {
                "scientific_run_commit": "d" * 40,
                "scene_units_per_meter": 0.2,
                "capture_root": str(source_capture),
                "config": {"path": str(config), "sha256": sha256(config)},
                "metrics": {"path": str(metrics), "sha256": sha256(metrics)},
                "lmdb": {"path": str(source), "data_sha256": sha256(lmdb)},
                "original_preview": {"path": str(preview), "sha256": sha256(preview)},
            },
            "observations": rows,
        }
        plan_path = root / "replay_plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")

        processed = root / "processed"
        captures = root / "captures"
        for row in rows:
            observation_id = row["observation_id"]
            formatted = f"{observation_id:06d}"
            request_id = row["replay_request_id"]
            frame_id = row["replay_frame_id"]
            actual_timestamp = 2000 + observation_id
            capture = captures / formatted
            raw_root = capture / "raw_bundle"
            canonical_root = capture / "canonical_bundle"
            raw_root.mkdir(parents=True)
            canonical_root.mkdir(parents=True)
            replay_source = {
                "schema_version": "pan29.replay-source.v1",
                "observation_id": observation_id,
                "source_bundle_id": observation_id,
                "source_capture_timestamp_ns": row["source_capture_timestamp_ns"],
                "source_capture_timestamp_utc": row["source_capture_timestamp_utc"],
                "planner_position_scene_units": row["planner_position_scene_units"],
                "requested_position_ue_cm": row["ue_position_cm"],
                "source_metrics_sha256": sha256(metrics),
                "within_pan13_conservative_fly_volume": row[
                    "within_pan13_conservative_fly_volume"
                ],
                "planner_input_unchanged": True,
                "ue5_role": "post_run_visualization_only",
            }
            request = {
                "schema_version": "pan15.capture-request.v1",
                "scenario": "hkust",
                "level_path": "/Game/PAN13_Derived/HKUST_ZUp_QA",
                "position_actor_label": None,
                "position_ue_cm": row["ue_position_cm"],
                "request_id": request_id,
                "frame_id": frame_id,
                "replay_source": replay_source,
            }
            request_path = capture / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")

            raw_faces = []
            canonical_faces = []
            canonical_rgb_hashes = {}
            contract_position = (
                row["ue_position_cm"][0] / 100.0,
                -row["ue_position_cm"][1] / 100.0,
                row["ue_position_cm"][2] / 100.0,
            )
            contract_from_pioneer = transpose(CONTRACT_TO_PIONEER)
            for face_name in FACES:
                raw_face_root = raw_root / "faces" / face_name
                raw_face_root.mkdir(parents=True)
                records = {}
                for name in (
                    "rgb_uint8.bin",
                    "rgb.png",
                    "scene_depth.exr",
                    "scene_readback.bin",
                    "device_depth.exr",
                    "device_readback.bin",
                    "mask.bin",
                ):
                    path = raw_face_root / name
                    path.write_bytes(f"{observation_id}:{face_name}:{name}".encode())
                    records[name] = artifact(path, raw_root)
                pioneer_rotation = PIONEER_FACE_ROTATIONS[face_name]
                raw_faces.append(
                    {
                        "face_name": face_name,
                        "request_id": request_id,
                        "frame_id": frame_id,
                        "capture_timestamp_ns": actual_timestamp,
                        "T_pioneer_world_from_cam": transform(
                            pioneer_rotation, row["planner_position_scene_units"]
                        ),
                        "rgb_uint8": records["rgb_uint8.bin"],
                        "rgb_png": records["rgb.png"],
                        "scene_depth_r": {
                            "raw_exr": records["scene_depth.exr"],
                            "python_readback": records["scene_readback.bin"],
                        },
                        "device_depth_r": {
                            "raw_exr": records["device_depth.exr"],
                            "python_readback": records["device_readback.bin"],
                        },
                        "python_readback_candidate_mask": records["mask.bin"],
                    }
                )

                canonical_face_root = canonical_root / "faces" / face_name
                canonical_face_root.mkdir(parents=True)
                canonical_assets = {}
                for field in ("rgb_uint8", "depth_range_m", "valid_mask"):
                    path = canonical_face_root / f"{field}.npy"
                    path.write_bytes(f"{observation_id}:{face_name}:{field}".encode())
                    canonical_assets[field] = artifact(path, canonical_root)
                canonical_rgb_hashes[face_name] = canonical_assets["rgb_uint8"][
                    "sha256"
                ]
                contract_rotation = matmul(contract_from_pioneer, pioneer_rotation)
                canonical_faces.append(
                    {
                        "face_name": face_name,
                        "request_id": request_id,
                        "frame_id": frame_id,
                        "capture_timestamp_ns": actual_timestamp,
                        "T_world_from_cam": transform(
                            contract_rotation, contract_position
                        ),
                        "assets": canonical_assets,
                    }
                )

            raw_manifest = {
                "schema_version": "pan15.ue5-raw-rgbd.v1",
                "request_id": request_id,
                "frame_id": frame_id,
                "capture_timestamp_ns": actual_timestamp,
                "requested_position_ue_cm": row["ue_position_cm"],
                "shared_optical_center_ue_cm": row["ue_position_cm"],
                "replay_source": replay_source,
                "faces": raw_faces,
            }
            raw_path = raw_root / "manifest.json"
            raw_path.write_text(json.dumps(raw_manifest), encoding="utf-8")
            canonical_manifest = {
                "schema_version": "pioneer.ue5-observation.v1",
                "request_id": request_id,
                "frame_id": frame_id,
                "capture_timestamp_ns": actual_timestamp,
                "position_world_m": list(contract_position),
                "face_names": list(FACES),
                "faces": canonical_faces,
                "provenance": {"replay_source": replay_source},
            }
            canonical_path = canonical_root / "manifest.json"
            canonical_path.write_text(json.dumps(canonical_manifest), encoding="utf-8")

            directory = processed / formatted
            directory.mkdir(parents=True)
            erp = directory / "erp.png"
            erp.write_bytes(f"erp-{observation_id}".encode())
            receipt = {
                "schema_version": "pan29.ue5-replay-receipt.v1",
                "result": "PASS",
                "observation_id": observation_id,
                "request_id": request_id,
                "frame_id": frame_id,
                "source_capture_timestamp_ns": row["source_capture_timestamp_ns"],
                "source_capture_timestamp_utc": row["source_capture_timestamp_utc"],
                "ue5_actual_capture_timestamp_ns": actual_timestamp,
                "planner_position_scene_units": row["planner_position_scene_units"],
                "requested_position_ue_cm": row["ue_position_cm"],
                "measured_position_ue_cm": row["ue_position_cm"],
                "measured_position_scene_units": row["planner_position_scene_units"],
                "ue_position_max_abs_error_cm": 0.0,
                "position_max_abs_error_scene_units": 0.0,
                "face_basis_max_abs_error": 0.0,
                "face_names": list(FACES),
                "within_pan13_conservative_fly_volume": row[
                    "within_pan13_conservative_fly_volume"
                ],
                "planner_input_unchanged": True,
                "ue5_role": "post_run_visualization_only",
                "source_semantic_validation": {
                    "schema_version": "pan29.source-bundle-semantic-validation.v1",
                    "result": "PASS",
                    "basis_sha256": source_semantic(
                        observation_id,
                        row["source_capture_timestamp_ns"],
                        row["source_capture_timestamp_utc"],
                        row["planner_position_scene_units"],
                    )[1],
                    "camera_center_max_abs_error_scene_units": 0.0,
                    "face_basis_max_abs_error": 0.0,
                },
                "sources": {
                    "plan": {"path": str(plan_path), "sha256": sha256(plan_path)},
                    "request": {
                        "path": str(request_path),
                        "sha256": sha256(request_path),
                    },
                    "raw_manifest": {"path": str(raw_path), "sha256": sha256(raw_path)},
                    "canonical_manifest": {
                        "path": str(canonical_path),
                        "sha256": sha256(canonical_path),
                    },
                    "canonical_rgb_array_sha256": canonical_rgb_hashes,
                },
                "erp": {
                    "path": "erp.png",
                    "sha256": sha256(erp),
                    "width": 1024,
                    "height": 512,
                    "mode": "RGB",
                    "byte_deterministic_reprojection": True,
                },
            }
            (directory / "receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
        run_manifest = root / "run-manifest.txt"
        run_manifest.write_text(
            "git_commit=" + "e" * 40 + "\n"
            "replay_plan_sha256=" + sha256(plan_path) + "\n"
            "ue_project_sha256=" + "1" * 64 + "\n"
            "ue_default_engine_sha256=" + "2" * 64 + "\n"
            "ue_level_sha256=" + "3" * 64 + "\n"
            "capture_config_sha256=" + "4" * 64 + "\n",
            encoding="utf-8",
        )
        return plan_path, processed, run_manifest

    def test_finalizes_exact_five_and_rehashes_source_and_replay_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, processed, run_manifest = self._fixture(root)
            result = finalize_replay(
                plan_path=plan,
                processed_root=processed,
                run_manifest_path=run_manifest,
            )
            self.assertEqual(result["result"], "PASS")
            self.assertEqual(result["replay_count"], 5)
            self.assertEqual(result["observation_ids"], list(IDS))
            self.assertEqual(result["out_of_pan13_fly_policy_ids"], [5, 10, 14, 19])
            self.assertTrue(result["planner_input_unchanged"])
            self.assertEqual(len(result["replays"]), 5)
            self.assertEqual(len(result["erp_tree_sha256"]), 64)

    def test_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, processed, run_manifest = self._fixture(root)
            payload = json.loads(plan.read_text())
            Path(payload["source"]["metrics"]["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source metrics"):
                finalize_replay(
                    plan_path=plan,
                    processed_root=processed,
                    run_manifest_path=run_manifest,
                )

    def test_rejects_run_manifest_bound_to_another_replay_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, processed, run_manifest = self._fixture(root)
            payload = run_manifest.read_text().replace(
                f"replay_plan_sha256={sha256(plan)}",
                f"replay_plan_sha256={'f' * 64}",
            )
            run_manifest.write_text(payload)
            with self.assertRaisesRegex(ValueError, "replay_plan_sha256 differs"):
                finalize_replay(
                    plan_path=plan,
                    processed_root=processed,
                    run_manifest_path=run_manifest,
                )

    def test_rejects_capture_against_tampered_plan_pinned_source_basis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, processed, run_manifest = self._fixture(root)
            payload = json.loads(plan.read_text())
            payload["observations"][1]["source_semantic_validation"]["faces"][4][
                "T_world_from_cam_rotation"
            ] = [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
            plan.write_text(json.dumps(payload))
            new_plan_sha = sha256(plan)
            run_manifest.write_text(
                run_manifest.read_text().replace(
                    next(
                        line
                        for line in run_manifest.read_text().splitlines()
                        if line.startswith("replay_plan_sha256=")
                    ),
                    "replay_plan_sha256=" + new_plan_sha,
                )
            )
            tampered_faces = payload["observations"][1]["source_semantic_validation"][
                "faces"
            ]
            tampered_basis_sha = hashlib.sha256(
                json.dumps(
                    [
                        {
                            "face_name": face["face_name"],
                            "T_world_from_cam_rotation": face[
                                "T_world_from_cam_rotation"
                            ],
                        }
                        for face in tampered_faces
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            for observation_id in IDS:
                receipt_path = processed / f"{observation_id:06d}" / "receipt.json"
                receipt = json.loads(receipt_path.read_text())
                receipt["sources"]["plan"]["sha256"] = new_plan_sha
                if observation_id == 5:
                    receipt["source_semantic_validation"]["basis_sha256"] = (
                        tampered_basis_sha
                    )
                receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "source basis"):
                finalize_replay(
                    plan_path=plan,
                    processed_root=processed,
                    run_manifest_path=run_manifest,
                )

    def test_rejects_tampered_request_and_recursive_assets(self):
        mutators = (
            lambda root: (root / "captures/000005/request.json").write_text("{}"),
            lambda root: (
                root / "captures/000005/raw_bundle/faces/front/rgb.png"
            ).write_bytes(b"tampered-raw"),
            lambda root: (
                root / "captures/000005/canonical_bundle/faces/front/rgb_uint8.npy"
            ).write_bytes(b"tampered-canonical"),
        )
        expected = ("request", "raw capture asset", "canonical asset")
        for mutate, message in zip(mutators, expected):
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                plan, processed, run_manifest = self._fixture(root)
                mutate(root)
                with self.assertRaisesRegex(ValueError, message):
                    finalize_replay(
                        plan_path=plan,
                        processed_root=processed,
                        run_manifest_path=run_manifest,
                    )

    def test_rejects_wrong_capture_path_erp_path_and_tampered_identity(self):
        def use_external_raw(root: Path):
            receipt_path = root / "processed/000005/receipt.json"
            receipt = json.loads(receipt_path.read_text())
            original = root / "captures/000005/raw_bundle/manifest.json"
            external = root / "external-raw.json"
            external.write_bytes(original.read_bytes())
            receipt["sources"]["raw_manifest"] = {
                "path": str(external),
                "sha256": sha256(external),
            }
            receipt_path.write_text(json.dumps(receipt))

        def use_external_erp(root: Path):
            receipt_path = root / "processed/000005/receipt.json"
            receipt = json.loads(receipt_path.read_text())
            external = root / "external-erp.png"
            external.write_bytes((root / "processed/000005/erp.png").read_bytes())
            receipt["erp"]["path"] = str(external)
            receipt_path.write_text(json.dumps(receipt))

        def change_receipt_identity(root: Path):
            receipt_path = root / "processed/000005/receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["request_id"] = "another-request"
            receipt_path.write_text(json.dumps(receipt))

        for mutate, message in (
            (use_external_raw, "expected capture path"),
            (use_external_erp, "ERP path"),
            (change_receipt_identity, "request_id differs from plan"),
        ):
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                plan, processed, run_manifest = self._fixture(root)
                mutate(root)
                with self.assertRaisesRegex(ValueError, message):
                    finalize_replay(
                        plan_path=plan,
                        processed_root=processed,
                        run_manifest_path=run_manifest,
                    )

    def test_recomputes_pose_and_basis_errors_from_manifests(self):
        mutators = (
            (0, 3, 1.0, "raw face front source position"),
            (0, 0, 0.25, "raw face front basis"),
        )
        for row_index, column_index, delta, message in mutators:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                plan, processed, run_manifest = self._fixture(root)
                raw_path = root / "captures/000005/raw_bundle/manifest.json"
                raw = json.loads(raw_path.read_text())
                raw["faces"][0]["T_pioneer_world_from_cam"][row_index][
                    column_index
                ] += delta
                raw_path.write_text(json.dumps(raw))
                receipt_path = root / "processed/000005/receipt.json"
                receipt = json.loads(receipt_path.read_text())
                receipt["sources"]["raw_manifest"]["sha256"] = sha256(raw_path)
                receipt_path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, message):
                    finalize_replay(
                        plan_path=plan,
                        processed_root=processed,
                        run_manifest_path=run_manifest,
                    )

        canonical_mutators = (
            (0, 3, 1.0, "canonical source position"),
            (0, 0, 0.25, "canonical face front basis"),
        )
        for row_index, column_index, delta, message in canonical_mutators:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                plan, processed, run_manifest = self._fixture(root)
                canonical_path = root / "captures/000005/canonical_bundle/manifest.json"
                canonical = json.loads(canonical_path.read_text())
                canonical["faces"][0]["T_world_from_cam"][row_index][column_index] += (
                    delta
                )
                if column_index == 3:
                    canonical["position_world_m"][row_index] += delta
                canonical_path.write_text(json.dumps(canonical))
                receipt_path = root / "processed/000005/receipt.json"
                receipt = json.loads(receipt_path.read_text())
                receipt["sources"]["canonical_manifest"]["sha256"] = sha256(
                    canonical_path
                )
                receipt_path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, message):
                    finalize_replay(
                        plan_path=plan,
                        processed_root=processed,
                        run_manifest_path=run_manifest,
                    )


if __name__ == "__main__":
    unittest.main()
