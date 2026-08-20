import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).parents[1] / "tools" / "convert_huge_scene_to_magician.py"
SPEC = importlib.util.spec_from_file_location("huge_converter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


class HugeConverterTest(unittest.TestCase):
    def test_coordinate_transform_is_invertible_and_right_handed(self):
        points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
        transformed = CONVERTER.transform_points(points)
        np.testing.assert_allclose(transformed, [[1.0, 3.0, -2.0], [-4.0, -6.0, -5.0]])
        self.assertAlmostEqual(np.linalg.det(CONVERTER.T_HUGE_TO_MAGIC[:3, :3]), 1.0)
        homogeneous = np.column_stack((transformed, np.ones(len(transformed))))
        recovered = (homogeneous @ np.linalg.inv(CONVERTER.T_HUGE_TO_MAGIC).T)[:, :3]
        np.testing.assert_allclose(recovered, points)

    def test_obj_audit_handles_negative_indices_and_polygons(self):
        with tempfile.TemporaryDirectory() as temp_text:
            path = Path(temp_text) / "mesh.obj"
            path.write_text(
                "v 0 0 0\n"
                "v 1 0 0\n"
                "v 1 1 0\n"
                "v 0 1 0\n"
                "vt 0 0\n"
                "o quad\n"
                "f -4/1 -3/1 -2/1 -1/1\n",
                encoding="utf-8",
            )
            audit = CONVERTER.parse_obj(path)
            self.assertEqual(audit["counts"]["faces"], 1)
            self.assertEqual(audit["counts"]["non_triangular_faces"], 1)
            self.assertEqual(audit["counts"]["negative_index_faces"], 1)
            self.assertEqual(audit["counts"]["triangles_after_fan"], 2)
            self.assertEqual(audit["counts"]["degenerate_triangles"], 0)

    def test_end_to_end_geometry_only_conversion_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as temp_text:
            root = Path(temp_text)
            env = root / "huge" / "1_office"
            mesh_dir = env / "terra_ply"
            ply_dir = env / "3dgs_ply"
            landmarks_dir = env / "location_gen"
            mesh_dir.mkdir(parents=True)
            ply_dir.mkdir(parents=True)
            landmarks_dir.mkdir(parents=True)
            (mesh_dir / "metadata.xml").write_text(
                "<ModelMetadata><SRS>EPSG:32650</SRS>"
                "<SRSOrigin>100,200,102.332</SRSOrigin></ModelMetadata>\n",
                encoding="utf-8",
            )
            (ply_dir / "metadata.xml").write_text(
                "<ModelMetadata><SRS>ENU:30.2,119.9</SRS>"
                "<SRSOrigin>0,0,102.332</SRSOrigin></ModelMetadata>\n",
                encoding="utf-8",
            )
            (mesh_dir / "simplified_mesh.obj").write_text(
                "v -10 -10 -85\n"
                "v 10 -10 -85\n"
                "v 10 10 -85\n"
                "v -10 10 -85\n"
                "v -10 -10 -75\n"
                "v 10 -10 -75\n"
                "v 10 10 -75\n"
                "v -10 10 -75\n"
                "f 1 3 2\nf 1 4 3\n"
                "f 5 6 7\nf 5 7 8\n"
                "f 1 2 6\nf 1 6 5\n"
                "f 2 3 7\nf 2 7 6\n"
                "f 3 4 8\nf 3 8 7\n"
                "f 4 1 5\nf 4 5 8\n",
                encoding="utf-8",
            )
            (ply_dir / "point_cloud_utm50.ply").write_text(
                "ply\nformat ascii 1.0\nelement vertex 4\n"
                "property float x\nproperty float y\nproperty float z\nend_header\n"
                "-10 -10 -85\n10 -10 -85\n10 10 -75\n-10 10 -75\n",
                encoding="ascii",
            )
            (landmarks_dir / "landmark_merged_s.txt").write_text(
                "#x\ty\tz\tlabel\n0\t0\t-80\ttest building\n", encoding="utf-8"
            )
            output_root = root / "magician_data"
            common_args = [
                "--huge-data-root",
                str(root / "huge"),
                "--magician-data-root",
                str(output_root),
                "--no-preview",
            ]
            self.assertEqual(CONVERTER.main(common_args), 0)
            output = output_root / "huge_1_office"
            manifest = json.loads((output / "conversion_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["materials"]["rgb_mesh_compatible"])
            self.assertEqual(manifest["coordinate_system"]["scale"], 1.0)
            occupancy = torch.load(output / "occupied_pose.pt", map_location="cpu")
            self.assertEqual(occupancy["X_idx"].shape[1], 3)
            self.assertEqual(occupancy["occupied"].dtype, torch.bool)
            settings = json.loads((output / "settings.json").read_text(encoding="utf-8"))
            self.assertEqual(len(settings["camera"]["start_positions"]), 5)
            deterministic_files = [
                output / "huge_1_office.obj",
                output / "occupied_pose.pt",
                output / "settings.json",
            ]
            before = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in deterministic_files
            }
            with self.assertRaises(FileExistsError):
                CONVERTER.main(common_args)
            manual = output / "manual.txt"
            manual.write_text("must survive\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                CONVERTER.main([*common_args, "--overwrite"])
            self.assertEqual(manual.read_text(encoding="utf-8"), "must survive\n")
            manual.unlink()
            self.assertEqual(CONVERTER.main([*common_args, "--overwrite"]), 0)
            after = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in deterministic_files
            }
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
