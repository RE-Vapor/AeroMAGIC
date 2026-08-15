# Incremental OpenHK3D scene assembly

`tools/openhk3d_assemble.py` converts one raw OpenHK3D tile plus one adjacent
tile into a single MAGICIAN scene. The generated directory can be supplied as
`--base` in the next run, giving the reusable operation:

```text
MAGICIAN_scene(N tiles) + adjacent raw tile -> MAGICIAN_scene(N+1 tiles)
```

The tool does not infer a geographic CRS. It uses the integer coordinates in
`Tile_<x>_<y>.obj` only for adjacency and preserves relative placement from the
source OBJ coordinates. For a first assembly, the base tile's source bounding
box center is the default origin. Every later assembly inherits the transform
recorded in `assembly-manifest.json`, so the coordinate frame never recenters.

## Requirements

- The MAGICIAN Python environment, including PyTorch.
- Blender `>=4.2,<4.3`, callable through an executable path.
- Each raw tile directory must contain exactly one `Tile_<x>_<y>.obj`, its MTL
  libraries, and every referenced texture.
- The base tile or assembled scene must contain a working `settings.json`.

The Blender version is intentionally narrow because the Wavefront importer API
is part of the validation contract.

## Discover an adjacent tile

```bash
python tools/openhk3d_assemble.py discover \
  --dataset-root /path/to/12-NW-6C \
  --base 12-NW-6C-7
```

Only shared-edge neighbors are returned. Diagonal tiles are excluded.

## Assemble tile 7 and tile 8

```bash
python tools/openhk3d_assemble.py assemble \
  --dataset-root /path/to/12-NW-6C \
  --base 12-NW-6C-7 \
  --add 12-NW-6C-8 \
  --output /path/to/Macarons++/12-NW-6C-7_8 \
  --blender /path/to/blender
```

The output directory must not exist. This prevents an incomplete run from
overwriting an earlier scene. Work is staged next to the destination and moved
into place only after all enabled gates pass.

To append another tile, point `--base` to the preceding output directory and
keep `--dataset-root` pointed at the raw tile collection:

```bash
python tools/openhk3d_assemble.py assemble \
  --dataset-root /path/to/12-NW-6C \
  --base /path/to/Macarons++/12-NW-6C-7_8 \
  --add 12-NW-6C-9 \
  --output /path/to/Macarons++/12-NW-6C-7_8_9 \
  --blender /path/to/blender
```

## What is generated

- `<scene>.obj` and `<scene>.mtl`: one deterministic mesh package. Positive
  OBJ indices are offset without changing topology; normals and vertices use
  the same shared axis transform. No ICP, Boolean, remesh, welding, or vertex
  deduplication is performed.
- `textures/<asset>/...`: per-input namespaces prevent silent filename
  overwrites, including when two tiles contain different bytes under the same
  basename.
- `settings.json`: extends the base settings while retaining approximate scene
  and camera grid cell sizes. Existing start poses are remapped by world
  position when a lower bound expands.
- `occupied_pose.json` and `occupied_pose.pt`: all camera-grid positions,
  classified by Blender BVH proximity plus majority ray parity.
- `validation_report.json`: Blender import, texture, occupancy, and start-pose
  checks. `magician_loader_report.json` records a CPU smoke test through the
  repository's real `SceneDataset` and `load_scene` path.
- `assembly-manifest.json`: member grid coordinates, frozen shared transform,
  hashes, counts, bounds, and validation claims needed by the next run.

The seam report is diagnostic. It records bounding-box gap/overlap and sampled
nearest boundary-vertex distances, but it never declares a geometry-continuity
PASS without an approved tolerance. The loader smoke test proves that the
scene can be parsed; it does not prove that a planning trajectory succeeds.

## Diagnostic-only intermediate

`--skip-blender` can create an intermediate package for inspection on a host
without Blender. It intentionally omits `occupied_pose.pt` and records the
Blender gate as skipped. Such an output is not a completed MAGICIAN scene.
`--skip-magician-loader` retains the Blender and occupancy gates but omits the
repository loader smoke test; it is likewise not sufficient for a compatibility
claim.
