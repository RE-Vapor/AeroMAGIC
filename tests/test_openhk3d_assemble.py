import tempfile
import unittest
from pathlib import Path

from tools.openhk3d_assemble import (
    AssemblyError,
    GridCoord,
    SharedTransform,
    _load_raw_tile,
    adjacent_tiles,
    generate_settings,
    merge_materials,
    merge_objects,
    parse_grid_from_obj_name,
)


OBJ_TEMPLATE = """mtllib {mtl}\n\
v {x0} 0 0\n\
v {x1} 0 0\n\
v {x0} 1 0\n\
vt 0 0\n\
vt 1 0\n\
vt 0 1\n\
vn 0 0 1\n\
usemtl wall\n\
f 1/1/1 2/2/1 3/3/1\n"""


def write_tile(root: Path, name: str, x: int, y: int, x0: float, x1: float, texture: bytes) -> Path:
    tile = root / name
    tile.mkdir()
    obj_name = "Tile_{:+d}_{:+d}.obj".format(x, y)
    mtl_name = "Tile_{:+d}_{:+d}.mtl".format(x, y)
    (tile / obj_name).write_text(
        OBJ_TEMPLATE.format(mtl=mtl_name, x0=x0, x1=x1), encoding="utf-8"
    )
    (tile / mtl_name).write_text("newmtl wall\nmap_Kd shared.jpg\n", encoding="utf-8")
    (tile / "shared.jpg").write_bytes(texture)
    return tile


class OpenHK3DAssemblyTests(unittest.TestCase):
    def test_grid_parser_and_adjacency(self):
        self.assertEqual(parse_grid_from_obj_name("Tile_+301_+146.obj"), GridCoord(301, 146))
        with self.assertRaises(AssemblyError):
            parse_grid_from_obj_name("scene.obj")

    def test_shared_transform_matches_the_existing_tile_axis_contract(self):
        transform = SharedTransform(0.1, (45225.0, 21975.0, 96.0))
        self.assertEqual(transform.point((45235.0, 21955.0, 101.0)), (1.0, 0.5, 2.0))
        self.assertEqual(transform.normal((1.0, 2.0, 3.0)), (1.0, 3.0, -2.0))

    def test_discovery_reports_only_edge_neighbors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_tile(root, "tile-7", 301, 146, 0, 1, b"a")
            write_tile(root, "tile-8", 302, 146, 1, 2, b"b")
            write_tile(root, "diagonal", 302, 147, 1, 2, b"c")
            result = adjacent_tiles(root, "tile-7")
            self.assertEqual([item["name"] for item in result["neighbors"]], ["tile-8"])

    def test_merge_offsets_indices_and_namespaces_textures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tile_a = write_tile(root, "tile-7", 301, 146, 0, 1, b"first")
            tile_b = write_tile(root, "tile-8", 302, 146, 1, 2, b"second")
            assets = [_load_raw_tile(tile_a), _load_raw_tile(tile_b)]
            output = root / "output"
            output.mkdir()
            maps = merge_materials(assets, output, output / "scene.mtl")
            stats = merge_objects(
                assets,
                SharedTransform(0.1, (0.0, 0.0, 0.0)),
                maps,
                output / "scene.obj",
                "scene.mtl",
            )
            self.assertEqual(stats.vertices, 6)
            self.assertEqual(stats.faces, 2)
            obj = (output / "scene.obj").read_text(encoding="utf-8")
            self.assertIn("f 1/1/1 2/2/1 3/3/1", obj)
            self.assertIn("f 4/4/2 5/5/2 6/6/2", obj)
            self.assertIn("usemtl tile-7__wall", obj)
            self.assertIn("usemtl tile-8__wall", obj)
            self.assertEqual((output / "textures/tile-7/shared.jpg").read_bytes(), b"first")
            self.assertEqual((output / "textures/tile-8/shared.jpg").read_bytes(), b"second")

    def test_settings_expansion_preserves_camera_world_start(self):
        template = {
            "scene": {
                "grid_l": 2,
                "grid_w": 2,
                "grid_h": 2,
                "cell_capacity": 1000,
                "cell_resolution": 0.05,
                "x_min": [-1, -1, -1],
                "x_max": [1, 1, 1],
                "visibility_ratio": 0.99,
            },
            "camera": {
                "pose_l": 2,
                "pose_w": 2,
                "pose_h": 2,
                "pose_n_theta": 5,
                "pose_n_azim": 10,
                "x_min": [-1, -1, -1],
                "x_max": [1, 1, 1],
                "start_positions": [[0, 0, 0, 1, 2]],
                "contrast_factor": 1.2,
            },
        }
        from tools.openhk3d_assemble import Bounds

        result = generate_settings(
            template,
            Bounds((-1, -1, -1), (1, 1, 1)),
            Bounds((-3, -1, -1), (1, 1, 1)),
            0,
        )
        self.assertEqual(result["camera"]["pose_l"], 4)
        self.assertEqual(result["camera"]["start_positions"][0][:3], [2, 0, 0])
        self.assertEqual(template["camera"]["start_positions"][0][:3], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
