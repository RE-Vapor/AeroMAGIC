import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import lmdb
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_pioneer_preview import FACE_NAMES, generate_preview  # noqa: E402


class MakePioneerPreviewTests(unittest.TestCase):
    def _fixture(self, temporary, bundle_count=3):
        root = Path(temporary)
        frames = root / "memory" / "training" / "0" / "frames"
        images = frames.parent / "imgs"
        frames.mkdir(parents=True)
        bundles = []
        for bundle_id in range(bundle_count):
            image_dir = images / f"{bundle_id:06d}"
            image_dir.mkdir(parents=True)
            for face_index, face in enumerate(FACE_NAMES):
                rgb = np.zeros((16, 16, 3), dtype=np.uint8)
                rgb[:, :, 0] = (30 * bundle_id) % 256
                rgb[:, :, 1] = 20 * face_index
                rgb[:, :, 2] = 100
                Image.fromarray(rgb).save(image_dir / f"{face}.png")
            bundles.append(
                {
                    "bundle_id": bundle_id,
                    "face_count": 6,
                    "face_names": list(FACE_NAMES),
                    "face_point_counts": [bundle_id + face for face in range(6)],
                }
            )

        metrics_path = root / "pioneer.online.json"
        metrics = {
            "planner": "pioneer",
            "scene": "eiffel",
            "start_index": 0,
            "capture_dir": str(frames),
            "run": {
                "run_id": "test_pioneer_preview",
                "planning_observation_mode": "cubemap6",
                "pioneer_face_size": 16,
            },
            "coverage": [
                {"normalized": 0.1 + 0.2 * bundle_id / max(1, bundle_count - 1)}
                for bundle_id in range(bundle_count)
            ],
            "trajectory": {
                "positions": [
                    [float(bundle_id), 1.0, float(bundle_id % 5)]
                    for bundle_id in range(bundle_count)
                ],
                "path_length_scene_units": 2.0,
            },
            "latency": {"trajectory_seconds": 12.5},
            "cuda": {"peak_reserved_mib": 1000.0},
            "pioneer_observation": {
                "bundle_count": bundle_count,
                "real_face_render_count": 6 * bundle_count,
                "imagined_candidate_face_render_count": 54,
                "bundles": bundles,
            },
        }
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

        lmdb_path = root / "trajectory.lmdb"
        environment = lmdb.open(str(lmdb_path), map_size=2 * 1024 * 1024)
        payload = {
            "points": np.asarray(
                [[0, 0, 0], [1, 0.5, 1], [2, 1, 0], [1, 1.5, 2]],
                dtype=np.float32,
            ),
            "points_color": np.asarray(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
                dtype=np.float32,
            ),
        }
        with environment.begin(write=True) as transaction:
            transaction.put(b"eiffel/0", pickle.dumps(payload))
        environment.close()
        return metrics_path, lmdb_path, images

    def test_generates_comprehensive_preview_and_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            metrics_path, lmdb_path, images = self._fixture(temporary)
            output = Path(temporary) / "preview" / "pioneer.png"
            source_hash = hashlib.sha256(
                (images / "000000" / "front.png").read_bytes()
            ).hexdigest()

            sidecar = generate_preview(
                metrics_path=metrics_path,
                lmdb_path=lmdb_path,
                output_path=output,
                max_points=100,
                dpi=40,
            )

            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".json").is_file())
            persisted_sidecar = json.loads(
                output.with_suffix(".json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted_sidecar, sidecar)
            with Image.open(output) as image:
                self.assertEqual(image.mode, "RGBA")
                self.assertGreater(image.width, 500)
                self.assertGreater(image.height, 300)
            self.assertEqual(sidecar["summary"]["bundle_count"], 3)
            self.assertEqual(sidecar["summary"]["displayed_bundle_ids"], [0, 1, 2])
            self.assertEqual(sidecar["summary"]["face_count"], 18)
            self.assertEqual(sidecar["summary"]["reconstructed_point_count"], 4)
            self.assertEqual(sidecar["sources"]["capture_image_count"], 18)
            self.assertEqual(
                sidecar["sources"]["lmdb"]["data_sha256"],
                hashlib.sha256((lmdb_path / "data.mdb").read_bytes()).hexdigest(),
            )
            self.assertEqual(len(sidecar["sources"]["capture_images_tree_sha256"]), 64)
            self.assertEqual(
                sidecar["preview"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest()
            )
            self.assertEqual(
                source_hash,
                hashlib.sha256((images / "000000" / "front.png").read_bytes()).hexdigest(),
            )

    def test_long_preview_samples_rows_but_validates_every_face(self):
        with tempfile.TemporaryDirectory() as temporary:
            metrics_path, lmdb_path, images = self._fixture(temporary, bundle_count=50)
            output = Path(temporary) / "pioneer-50.png"
            sidecar = generate_preview(
                metrics_path=metrics_path,
                lmdb_path=lmdb_path,
                output_path=output,
                max_points=100,
                dpi=20,
            )
            self.assertEqual(sidecar["summary"]["bundle_count"], 50)
            self.assertEqual(sidecar["summary"]["face_count"], 300)
            self.assertEqual(sidecar["sources"]["capture_image_count"], 300)
            self.assertEqual(sidecar["summary"]["displayed_bundle_count"], 6)
            self.assertEqual(
                sidecar["summary"]["displayed_bundle_ids"],
                [0, 9, 19, 29, 39, 49],
            )
            with Image.open(output) as image:
                self.assertLess(image.height, 5000)

            (images / "000007" / "down.png").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "missing PIONEER face images"):
                generate_preview(
                    metrics_path=metrics_path,
                    lmdb_path=lmdb_path,
                    output_path=Path(temporary) / "invalid.png",
                    max_points=100,
                    dpi=20,
                )


if __name__ == "__main__":
    unittest.main()
