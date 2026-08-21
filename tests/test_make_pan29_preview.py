import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.make_pan29_preview import generate_pan29_preview


IDS = (0, 5, 10, 14, 19)
FACES = ("front", "back", "left", "right", "up", "down")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(paths, root):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


class MakePAN29PreviewTests(unittest.TestCase):
    def _fixture(self, root: Path):
        capture = root / "capture"
        images_root = capture / "imgs"
        bundles = []
        all_images = []
        for bundle_id in range(20):
            directory = images_root / f"{bundle_id:06d}"
            directory.mkdir(parents=True)
            for face_index, face in enumerate(FACES):
                rgb = np.zeros((32, 32, 3), dtype=np.uint8)
                rgb[..., 0] = bundle_id * 8
                rgb[..., 1] = face_index * 35
                rgb[..., 2] = 100
                path = directory / f"{face}.png"
                Image.fromarray(rgb).save(path)
                all_images.append(path)
            bundles.append(
                {
                    "bundle_id": bundle_id,
                    "face_count": 6,
                    "face_names": list(FACES),
                    "capture_timestamp_unix_ns": 1000 + bundle_id,
                    "capture_timestamp_utc": f"2026-08-20T00:00:{bundle_id:02d}Z",
                }
            )
        metrics = {
            "planner": "pioneer",
            "scene": "HKUST",
            "capture_dir": str(capture / "frames"),
            "coverage": [
                {"normalized": 0.1 + 0.02 * bundle_id} for bundle_id in range(20)
            ],
            "trajectory": {
                "observation_count": 20,
                "positions": [[bundle_id, 50, -bundle_id] for bundle_id in range(20)],
            },
            "pioneer_observation": {
                "bundle_count": 20,
                "real_face_render_count": 120,
                "bundles": bundles,
            },
            "run": {"planning_observation_mode": "cubemap6"},
        }
        metrics_path = root / "metrics.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        original_preview = root / "original.png"
        Image.new("RGB", (8, 8)).save(original_preview)
        original_sidecar = {
            "sources": {
                "capture_images_tree_sha256": tree_hash(all_images, images_root),
                "capture_image_count": 120,
            }
        }
        original_preview.with_suffix(".json").write_text(
            json.dumps(original_sidecar), encoding="utf-8"
        )
        plan = {
            "schema_version": "pan29.ue5-postrun-replay.v1",
            "selected_bundle_ids": list(IDS),
            "source": {
                "depth_source": "GT",
                "metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)},
                "capture_root": str(capture),
                "original_preview": {
                    "path": str(original_preview),
                    "sha256": sha256(original_preview),
                    "sidecar_path": str(original_preview.with_suffix(".json")),
                    "sidecar_sha256": sha256(original_preview.with_suffix(".json")),
                },
            },
            "observations": [
                {
                    "observation_id": bundle_id,
                    "source_capture_timestamp_ns": 1000 + bundle_id,
                    "source_capture_timestamp_utc": f"2026-08-20T00:00:{bundle_id:02d}Z",
                    "planner_position_scene_units": [bundle_id, 50, -bundle_id],
                    "within_pan13_conservative_fly_volume": bundle_id == 0,
                }
                for bundle_id in IDS
            ],
        }
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        erp_root = root / "erps"
        replays = []
        for bundle_id in IDS:
            directory = erp_root / f"{bundle_id:06d}"
            directory.mkdir(parents=True)
            rgb = np.zeros((64, 128, 3), dtype=np.uint8)
            rgb[..., 0] = 50 + bundle_id
            rgb[..., 1] = np.linspace(0, 255, 128, dtype=np.uint8)
            erp = directory / "erp.png"
            Image.fromarray(rgb).save(erp)
            replays.append(
                {
                    "observation_id": bundle_id,
                    "erp": {
                        "path": str(erp),
                        "sha256": sha256(erp),
                        "width": 128,
                        "height": 64,
                    },
                    "within_pan13_conservative_fly_volume": bundle_id == 0,
                }
            )
        run_manifest = root / "run_manifest.txt"
        run_manifest.write_text("git_commit=" + "a" * 40 + "\n", encoding="utf-8")
        (root / "status.txt").write_text(
            "started_at_utc=2026-08-22T00:00:00Z\n"
            "snapshot_integrity_preflight=PASS\n"
            "snapshot_integrity_postflight=PASS\n",
            encoding="utf-8",
        )
        result = {
            "schema_version": "pan29.ue5-postrun-replay-result.v1",
            "result": "PASS",
            "planner_input_unchanged": True,
            "observation_ids": list(IDS),
            "replay_count": 5,
            "out_of_pan13_fly_policy_ids": [5, 10, 14, 19],
            "run_manifest": {
                "path": str(run_manifest),
                "sha256": sha256(run_manifest),
            },
            "plan": {"path": str(plan_path), "sha256": sha256(plan_path)},
            "replays": replays,
        }
        result_path = root / "replay_result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return metrics_path, plan_path, result_path, images_root

    def test_generates_five_row_planner_plus_ue5_preview_and_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics, plan, result, _ = self._fixture(root)
            output = root / "preview" / "pan29.png"
            sidecar = generate_pan29_preview(
                metrics_path=metrics,
                replay_result_path=result,
                output_path=output,
            )
            self.assertTrue(output.is_file())
            with Image.open(output) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertGreater(image.width, 1500)
                self.assertGreater(image.height, 1200)
            self.assertEqual(sidecar["summary"]["displayed_observation_ids"], list(IDS))
            self.assertEqual(sidecar["summary"]["planner_face_image_count"], 30)
            self.assertEqual(sidecar["summary"]["ue5_erp_count"], 5)
            self.assertTrue(sidecar["summary"]["planner_input_unchanged"])
            self.assertEqual(
                sidecar["summary"]["ue5_role"], "post_run_visualization_only"
            )
            self.assertEqual(sidecar["sources"]["all_source_face_count"], 120)
            self.assertEqual(
                sidecar["summary"]["out_of_pan13_fly_policy_ids"],
                [5, 10, 14, 19],
            )
            self.assertEqual(sidecar["sources"]["replay_result"]["path"], str(result.resolve()))
            persisted = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(persisted, sidecar)
            commit = json.loads(output.with_suffix(".commit.json").read_text())
            self.assertEqual(commit["preview_sha256"], sha256(output))
            self.assertEqual(commit["sidecar_sha256"], sha256(output.with_suffix(".json")))

    def test_rejects_tampered_unselected_source_face(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics, _, result, images = self._fixture(root)
            rgb = np.full((32, 32, 3), 255, dtype=np.uint8)
            Image.fromarray(rgb).save(images / "000007" / "down.png")
            with self.assertRaisesRegex(ValueError, "source face tree"):
                generate_pan29_preview(
                    metrics_path=metrics,
                    replay_result_path=result,
                    output_path=root / "invalid.png",
                )

    def test_rejects_unpublished_or_failed_postflight_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics, _, result, _ = self._fixture(root)
            unpublished = root / "replay_result.pending.json"
            result.replace(unpublished)
            with self.assertRaisesRegex(ValueError, "published replay_result"):
                generate_pan29_preview(
                    metrics_path=metrics,
                    replay_result_path=unpublished,
                    output_path=root / "unpublished.png",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics, _, result, _ = self._fixture(root)
            (root / "status.txt").write_text(
                "snapshot_integrity_preflight=PASS\n"
                "snapshot_integrity_postflight=FAIL\n"
                "exit_code=2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "postflight"):
                generate_pan29_preview(
                    metrics_path=metrics,
                    replay_result_path=result,
                    output_path=root / "failed.png",
                )


if __name__ == "__main__":
    unittest.main()
