import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_ntile_workflow import prepare_workflow


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        path.write_bytes(value)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrepareNTileWorkflowTests(unittest.TestCase):
    def _fixture(self, root):
        scene = "joined"
        source = root / "source" / scene
        output = root / "output"
        settings = {
            "camera": {
                "x_min": [0, 0, 0],
                "x_max": [30, 10, 10],
                "pose_l": 3,
                "pose_w": 1,
                "pose_h": 1,
                "pose_n_theta": 2,
                "pose_n_azim": 4,
                "start_positions": [[1, 0, 0, 0, 0]],
            }
        }
        assembly = {
            "scene_name": scene,
            "members": [{"name": "west"}, {"name": "center"}, {"name": "east"}],
            "output": {
                "bounds": {"minimum": [0, 0, 0], "maximum": [30, 10, 10]}
            },
            "shared_transform": {"scale": 1.0},
        }
        _write(source / f"{scene}.obj", b"obj")
        _write(source / f"{scene}.mtl", b"mtl")
        _write(source / "occupied_pose.pt", b"occupied")
        _write(source / "settings.json", settings)
        _write(
            source / "occupied_pose.json",
            {"X_idx": [[0, 0, 0]], "occupied": [False]},
        )
        _write(source / "assembly-manifest.json", assembly)
        baseline = root / "baseline.json"
        params = root / "params.json"
        calibration = root / "calibration.json"
        _write(
            baseline,
            {
                "numGPU": 9,
                "dataset_path": "baseline",
                "experiment_metrics_enabled": False,
                "experiment_param_overrides": {"unrelated": 7},
            },
        )
        _write(
            params,
            {
                "_data": {"scene_scale_factor": 1.0},
                "_depth_module": {"zfar": 100.0},
            },
        )
        _write(
            calibration,
            {
                "scene": scene,
                "unit_contract": {"threshold_unit": "runtime_scene_unit"},
                "calibration": {
                    "recommended_sensor_range_scene_units": 40.0,
                    "hard_cap_scene_units": 80.0,
                },
            },
        )
        asset_hashes = {
            name: _sha(source / name)
            for name in (
                f"{scene}.obj",
                f"{scene}.mtl",
                "settings.json",
                "occupied_pose.pt",
                "assembly-manifest.json",
            )
        }
        manifest = root / "workflow.json"
        spec = {
            "schema_version": 1,
            "issue": "MYL-test",
            "scene": scene,
            "source_scene_dir": str(source),
            "assembly_manifest": str(source / "assembly-manifest.json"),
            "baseline_config": str(baseline),
            "params_config": str(params),
            "calibration_report": str(calibration),
            "output_root": str(output),
            "python": "/env/bin/python",
            "gpu": 2,
            "run_id": "three_tiles",
            "budget_observations": 11,
            "start_grid_index": [0, 0, 0, 0, 0],
            "tile_partition": {
                "axis": 0,
                "boundaries": [10.0, 20.0],
                "tile_ids": ["west", "center", "east"],
            },
            "asset_sha256": asset_hashes,
        }
        _write(manifest, spec)
        return manifest, spec

    def test_prepares_isolated_three_tile_config_and_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, spec = self._fixture(Path(temporary))
            prepared_path = prepare_workflow(manifest)
            prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
            config = json.loads(Path(prepared["config"]).read_text(encoding="utf-8"))
            self.assertEqual(config["validation_n_poses_in_trajectory"], 10)
            self.assertEqual(config["experiment_param_overrides"]["sensor_range"], 40.0)
            self.assertTrue(config["experiment_tile_metrics_enabled"])
            self.assertEqual(
                config["experiment_tile_partition"]["tile_ids"],
                spec["tile_partition"]["tile_ids"],
            )
            self.assertIn(
                "interval_semantics", config["experiment_tile_partition"]
            )
            self.assertEqual(prepared["assembly_members"], ["west", "center", "east"])
            self.assertEqual(prepared["gate_accept_command"][-3], "gate")
            self.assertTrue(
                (Path(prepared["dataset_view"]) / "joined.obj").is_symlink()
            )

    def test_rejects_member_count_and_asset_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, spec = self._fixture(Path(temporary))
            spec["tile_partition"]["tile_ids"] = ["west", "east"]
            spec["tile_partition"]["boundaries"] = [10.0]
            _write(manifest, spec)
            with self.assertRaisesRegex(ValueError, "same length"):
                prepare_workflow(manifest)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, spec = self._fixture(Path(temporary))
            spec["asset_sha256"]["joined.obj"] = "0" * 64
            _write(manifest, spec)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                prepare_workflow(manifest)


if __name__ == "__main__":
    unittest.main()
