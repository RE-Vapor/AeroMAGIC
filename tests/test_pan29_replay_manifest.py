import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch


from scripts.make_pan29_replay_manifest import (
    BUNDLE_IDS,
    build_replay_manifest,
)


FACE_NAMES = ("front", "back", "left", "right", "up", "down")
EXPECTED_WORLD_FROM_CAMERA_ROTATIONS = {
    "front": ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "back": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
    "left": ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    "right": ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
    "up": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "down": ((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
}


class PAN29ReplayManifestTests(unittest.TestCase):
    def _fixture(self, root: Path, observation_count: int = 20):
        capture = root / "memory" / "training" / "0"
        frames = capture / "frames"
        images = capture / "imgs"
        bundles = []
        positions = []
        for bundle_id in range(observation_count):
            frame_dir = frames / f"{bundle_id:06d}"
            image_dir = images / f"{bundle_id:06d}"
            frame_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)
            position = [float(bundle_id), 45.0 + bundle_id, -2.0 * bundle_id]
            center = torch.tensor([position], dtype=torch.float32)
            timestamp_ns = 1_780_000_000_000_000_000 + bundle_id
            timestamp_utc = f"2026-08-20T00:00:{bundle_id:02d}Z"
            torch.save(
                {
                    "bundle_id": bundle_id,
                    "face_names": FACE_NAMES,
                    "camera_center": center,
                    "capture_timestamp_unix_ns": timestamp_ns,
                    "capture_timestamp_utc": timestamp_utc,
                    "rig_frame": "world",
                    "extrinsics_version": "pytorch3d-world-axes-v1",
                },
                frame_dir / "bundle.pt",
            )
            for face in FACE_NAMES:
                world_from_camera = torch.tensor(
                    EXPECTED_WORLD_FROM_CAMERA_ROTATIONS[face], dtype=torch.float32
                )
                world_to_camera = world_from_camera.transpose(0, 1)
                translation = -(world_to_camera @ center.reshape(3, 1))
                torch.save(
                    {
                        "bundle_id": bundle_id,
                        "face_name": face,
                        "camera_center": center,
                        "capture_timestamp_unix_ns": timestamp_ns,
                        "capture_timestamp_utc": timestamp_utc,
                        "rig_frame": "world",
                        "extrinsics_version": "pytorch3d-world-axes-v1",
                        "R_opencv_world_to_camera": world_to_camera,
                        "T_opencv_world_to_camera": translation,
                    },
                    frame_dir / f"{face}.pt",
                )
                (image_dir / f"{face}.png").write_bytes(
                    f"png:{bundle_id}:{face}".encode()
                )
            positions.append(position)
            bundles.append(
                {
                    "bundle_id": bundle_id,
                    "capture_timestamp_unix_ns": timestamp_ns,
                    "capture_timestamp_utc": timestamp_utc,
                    "face_count": 6,
                    "face_names": list(FACE_NAMES),
                    "rig_frame": "world",
                    "extrinsics_version": "pytorch3d-world-axes-v1",
                }
            )
        metrics = {
            "planner": "pioneer",
            "scene": "HKUST",
            "start_index": 0,
            "capture_dir": str(frames),
            "scene_units_per_meter": 0.2,
            "run": {
                "debug_profile": (
                    "pioneer-20" if observation_count == 20 else "pioneer-low-altitude-50"
                ),
                "planning_observation_mode": "cubemap6",
                "depth_source": "GT",
                "pioneer_planner_state_mode": "position_only",
                "planning_state_dimension": 3,
                "pioneer_cubemap_rig_frame": "world",
                "pioneer_cubemap_extrinsics_version": "pytorch3d-world-axes-v1",
            },
            "trajectory": {
                "observation_count": observation_count,
                "positions": positions,
            },
            "pioneer_observation": {
                "bundle_count": observation_count,
                "real_face_render_count": 6 * observation_count,
                "bundles": bundles,
            },
        }
        metrics_path = root / "metrics.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "test_scenes": ["HKUST"],
                    "use_perfect_depth_map": True,
                    "kind_depth_map": "GT",
                    "pioneer_planner_state_mode": "position_only",
                    "pioneer_cubemap_rig_frame": "world",
                    "pioneer_cubemap_extrinsics_version": "pytorch3d-world-axes-v1",
                }
            ),
            encoding="utf-8",
        )
        config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
        run_manifest = root / "manifest.txt"
        run_manifest.write_text(
            "\n".join(
                (
                    "schema_version=1",
                    "planner=pioneer",
                    "scene=HKUST",
                    "git_commit=d015da4d6dc2dce78da75f9627a01be6dca71663",
                    f"config_sha256={config_sha}",
                    "debug_profile="
                    + (
                        "pioneer-20"
                        if observation_count == 20
                        else "pioneer-low-altitude-50"
                    ),
                    f"expected_observations={observation_count}",
                    f"expected_real_face_renders={6 * observation_count}",
                    "planner_state_mode=position_only",
                    "planner_state_dimension=3",
                    "cubemap_rig_frame=world",
                    "cubemap_extrinsics_version=pytorch3d-world-axes-v1",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        spatial_policy = root / "spatial_policy.json"
        spatial_policy.write_text(
            json.dumps(
                {
                    "coordinate_frame": "PAN13 localized Z-up UE centimetres",
                    "fly_volume_aabb_ue_cm": {
                        "min": [-10000.0, -10000.0, 20000.0],
                        "max": [10000.0, 10000.0, 26000.0],
                    },
                }
            ),
            encoding="utf-8",
        )
        lmdb = root / "trajectory.lmdb"
        lmdb.mkdir()
        (lmdb / "data.mdb").write_bytes(b"lmdb-data")
        preview = root / "original-preview.png"
        preview.write_bytes(b"original-preview")
        preview.with_suffix(".json").write_text("{}\n", encoding="utf-8")
        return (
            metrics_path,
            config_path,
            run_manifest,
            capture,
            spatial_policy,
            lmdb,
            preview,
        )

    def test_builds_five_hash_bound_postrun_replay_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics, config, run_manifest, capture, policy, lmdb, preview = self._fixture(root)
            payload = build_replay_manifest(
                metrics_path=metrics,
                source_config_path=config,
                source_run_manifest_path=run_manifest,
                capture_root=capture,
                spatial_policy_path=policy,
                source_lmdb_path=lmdb,
                original_preview_path=preview,
                bundle_ids=BUNDLE_IDS,
                source_registry_id=(
                    "PAN-11-PIONEER-HKUST-POSITION-ONLY-20OBS-20260821"
                ),
            )

            self.assertEqual(payload["schema_version"], "pan29.ue5-postrun-replay.v1")
            self.assertEqual(payload["selected_bundle_ids"], [0, 5, 10, 14, 19])
            self.assertTrue(payload["planner_input_unchanged"])
            self.assertTrue(payload["ue5_render_is_postrun_visualization_only"])
            self.assertEqual(payload["source"]["depth_source"], "GT")
            self.assertEqual(
                payload["source"]["scientific_run_commit"],
                "d015da4d6dc2dce78da75f9627a01be6dca71663",
            )
            self.assertEqual(
                payload["source"]["lmdb"]["data_sha256"],
                hashlib.sha256(b"lmdb-data").hexdigest(),
            )
            self.assertEqual(
                payload["source"]["original_preview"]["sha256"],
                hashlib.sha256(b"original-preview").hexdigest(),
            )
            self.assertEqual(
                payload["coordinate_transform"]["version"],
                "pan21-planner-to-ue-cm-v1",
            )

            rows = payload["observations"]
            self.assertEqual([row["observation_id"] for row in rows], list(BUNDLE_IDS))
            self.assertEqual(rows[1]["display_observation_number"], 6)
            self.assertEqual(rows[1]["planner_position_scene_units"], [5.0, 50.0, -10.0])
            self.assertEqual(rows[1]["ue_position_cm"], [2500.0, -5000.0, 25000.0])
            self.assertEqual(
                rows[1]["source_capture_timestamp_ns"],
                1_780_000_000_000_000_005,
            )
            self.assertEqual(rows[1]["replay_frame_id"], "observation-000005")
            self.assertFalse(rows[1]["source_bundle_transaction_marker_present"])
            self.assertTrue(rows[0]["within_pan13_conservative_fly_volume"])
            self.assertFalse(rows[2]["within_pan13_conservative_fly_volume"])
            self.assertEqual(len(rows[0]["source_artifacts"]["frames"]), 7)
            self.assertEqual(len(rows[0]["source_artifacts"]["images"]), 6)
            semantic = rows[1]["source_semantic_validation"]
            self.assertEqual(
                semantic["schema_version"],
                "pan29.source-bundle-semantic-validation.v1",
            )
            self.assertEqual(semantic["result"], "PASS")
            self.assertEqual(semantic["bundle_id"], 5)
            self.assertEqual(
                semantic["camera_center_scene_units"], [5.0, 50.0, -10.0]
            )
            self.assertLessEqual(
                semantic["camera_center_max_abs_error_scene_units"], 1e-6
            )
            self.assertLessEqual(semantic["face_basis_max_abs_error"], 1e-6)
            self.assertEqual(
                [face["face_name"] for face in semantic["faces"]],
                list(FACE_NAMES),
            )
            self.assertEqual(
                semantic["faces"][0]["T_world_from_cam_rotation"],
                [list(row) for row in EXPECTED_WORLD_FROM_CAMERA_ROTATIONS["front"]],
            )
            for group in ("frames", "images"):
                for artifact in rows[0]["source_artifacts"][group]:
                    self.assertEqual(len(artifact["sha256"]), 64)

    def test_builds_pan30_selection_from_fifty_observation_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics, config, run_manifest, capture, policy, lmdb, preview = (
                self._fixture(root, observation_count=50)
            )
            selected = (0, 12, 24, 36, 49)
            payload = build_replay_manifest(
                metrics_path=metrics,
                source_config_path=config,
                source_run_manifest_path=run_manifest,
                capture_root=capture,
                spatial_policy_path=policy,
                source_lmdb_path=lmdb,
                original_preview_path=preview,
                bundle_ids=selected,
                source_registry_id="PAN-30-PIONEER-HKUST-LOWALT-50OBS",
                task="PAN-30",
                replay_request_prefix="pan30-hkust-lowalt50",
            )

            self.assertEqual(payload["task"], "PAN-30")
            self.assertEqual(payload["selected_bundle_ids"], list(selected))
            self.assertEqual(payload["source"]["observation_count"], 50)
            self.assertEqual(
                payload["observations"][-1]["replay_request_id"],
                "pan30-hkust-lowalt50-000049",
            )

    def test_rejects_noncanonical_selection_and_manifest_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics, config, run_manifest, capture, policy, lmdb, preview = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "five unique"):
                build_replay_manifest(
                    metrics_path=metrics,
                    source_config_path=config,
                    source_run_manifest_path=run_manifest,
                    capture_root=capture,
                    spatial_policy_path=policy,
                    source_lmdb_path=lmdb,
                    original_preview_path=preview,
                    bundle_ids=(0, 5, 10, 19),
                    source_registry_id="record",
                )

            config.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config SHA256"):
                build_replay_manifest(
                    metrics_path=metrics,
                    source_config_path=config,
                    source_run_manifest_path=run_manifest,
                    capture_root=capture,
                    spatial_policy_path=policy,
                    source_lmdb_path=lmdb,
                    original_preview_path=preview,
                    bundle_ids=BUNDLE_IDS,
                    source_registry_id="record",
                )

    def test_rejects_source_tensor_identity_timestamp_center_name_or_basis_mismatch(self):
        cases = (
            ("bundle ID", "bundle.pt", "bundle_id", "payload has another bundle ID"),
            (
                "face bundle ID",
                "front.pt",
                "face_bundle_id",
                "face front has another bundle ID",
            ),
            ("timestamp", "front.pt", "timestamp", "timestamp differs"),
            ("camera center", "left.pt", "center", "camera center differs"),
            ("face name", "right.pt", "face_name", "another face name"),
            ("basis", "up.pt", "basis", "basis differs"),
        )
        for label, filename, mutation, expected_error in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (
                    metrics,
                    config,
                    run_manifest,
                    capture,
                    policy,
                    lmdb,
                    preview,
                ) = self._fixture(root)
                path = capture / "frames" / "000005" / filename
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if mutation == "bundle_id":
                    payload["bundle_id"] = 6
                elif mutation == "face_bundle_id":
                    payload["bundle_id"] = 6
                elif mutation == "timestamp":
                    payload["capture_timestamp_unix_ns"] += 1
                elif mutation == "center":
                    payload["camera_center"] = payload["camera_center"] + torch.tensor(
                        [[0.01, 0.0, 0.0]], dtype=torch.float32
                    )
                elif mutation == "face_name":
                    payload["face_name"] = "left"
                elif mutation == "basis":
                    center = payload["camera_center"].reshape(3, 1)
                    world_from_camera = torch.tensor(
                        EXPECTED_WORLD_FROM_CAMERA_ROTATIONS["down"],
                        dtype=torch.float32,
                    )
                    world_to_camera = world_from_camera.transpose(0, 1)
                    payload["R_opencv_world_to_camera"] = world_to_camera
                    payload["T_opencv_world_to_camera"] = -(
                        world_to_camera @ center
                    )
                else:  # pragma: no cover - the tuple above is closed.
                    self.fail(f"unknown mutation {mutation}")
                torch.save(payload, path)

                with self.assertRaisesRegex(ValueError, expected_error):
                    build_replay_manifest(
                        metrics_path=metrics,
                        source_config_path=config,
                        source_run_manifest_path=run_manifest,
                        capture_root=capture,
                        spatial_policy_path=policy,
                        source_lmdb_path=lmdb,
                        original_preview_path=preview,
                        bundle_ids=BUNDLE_IDS,
                        source_registry_id="record",
                    )


if __name__ == "__main__":
    unittest.main()
