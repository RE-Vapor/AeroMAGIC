# HUGE-Bench `1_office` conversion

This is a single-scene feasibility path for converting the official HUGE-Bench
`1_office` release into MAGICIAN's `data/Macarons++` layout. It keeps the HUGE
mesh in metres and does not modify MAGICIAN's planning logic.

## Revisions and inputs

- MAGICIAN: `1aad178728910d0a514a7c69b09f339fa558223f`
- RaDe-GS: `5f5cd3b0ecb329645bde45216fa826e9e85a7e1c`
- HUGE-Bench: `d000b99794e9da85dff117db61ef874659b91ae8`
- Hugging Face dataset revision: `f9bed5c1da172aecd5e3942848ee9174599ec59a`
- Archive: `archives/3DGS_Mesh_Envs_1_office.tar`
  (`7,397,201,920` bytes,
  SHA-256 `35a2583455438eb029ed3ee954f0d0afe29c9159e1cc4924ebb16b533af3142e`)

The archive contains only:

- `data_3d/1_office/3dgs_ply/point_cloud_utm50.ply`
- `data_3d/1_office/terra_ply/simplified_mesh.obj`

Do not download the complete environment or trajectory bundles for this check.
Use the exact dataset revision and the archive hash above.

## MAGICIAN scene contract

`SceneDataset` selects the first root-level `.obj`, then reads `settings.json`
and `occupied_pose.pt`. The 14 default scenes additionally all contain one MTL
and one diffuse texture because `capture_image` uses `SoftPhongShader`.

| Consumer | Converted source and treatment |
| --- | --- |
| root OBJ | HUGE `terra_ply/simplified_mesh.obj`; vertices and normals receive the proper frame rotation |
| MTL, UV, textures | copied and reference-checked if present; never invented |
| `settings.json` scene AABB | exact transformed mesh AABB |
| `settings.json` camera envelope | published HUGE task-0 horizontal landmarks plus 50 m margin and the full `z=-20..40 m` activity band |
| five-dimensional poses | `[l,w,h,theta,azim]`, using MAGICIAN's own grid and angle formula |
| five start poses | deterministic free grid positions facing the landmark median |
| `occupied_pose.pt` | regenerated for every camera-grid position from mesh proximity; `X_idx:int64`, `occupied:bool` |
| RGB/depth/mask, K, R/T | produced by MAGICIAN at runtime; not copied from future HUGE observations |

## Coordinate conversion

HUGE metadata identifies a right-handed, metre-scale local ENU/UTM-relative
frame with `+Z` up. PyTorch3D/MAGICIAN uses `+Y` up. The converter applies the
proper rotation `(E,N,U) -> (X,Y,Z) = (E,U,-N)`:

```text
T_huge_to_magic = [[1,  0, 0, 0],
                   [0,  0, 1, 0],
                   [0, -1, 0, 0],
                   [0,  0, 0, 1]]
```

Its determinant is `+1`, its scale is `1.0`, and unchanged face winding retains
handedness. The conversion manifest records mesh, PLY and landmark AABBs plus
three explicit landmark correspondences.

## Convert and validate

Run from the MAGICIAN repository with the environment that provides PyTorch,
PyTorch3D, trimesh, rtree, plyfile and matplotlib:

```bash
python tools/convert_huge_scene_to_magician.py \
  --huge-data-root /path/to/HUGE-Bench/trajectory_generation/scene_annotations/data_3d \
  --env-id 1_office \
  --magician-data-root ./data/Macarons++ \
  --output-scene-name huge_1_office \
  --random-seed 670 \
  --huge-commit d000b99794e9da85dff117db61ef874659b91ae8 \
  --magician-commit 1aad178728910d0a514a7c69b09f339fa558223f \
  --archive-path /path/to/archives/3DGS_Mesh_Envs_1_office.tar
```

Use `--dry-run` for a read-only full asset audit. An existing output is refused
unless `--overwrite` is explicit; overwrite is allowed only when the existing
manifest proves that the converter owns every file.

The GPU loader check is:

```bash
python tools/validate_huge_magician_scene.py \
  --scene-dir ./data/Macarons++/huge_1_office \
  --device cuda:0 --image-height 128 --image-width 128 --zfar 1600 \
  --report-path ./validation_report.json
```

`configs/test/test_huge_1_office_smoke_config.json` fixes the random seeds,
selects this scene, sets `beam_width=2` and `beam_steps=2`; its paired model
configuration preserves metre scale, raises the far plane for this kilometre
scale scene, and limits the run to one planning action.

## Published-asset limitation

The released `simplified_mesh.obj` is geometry-only: it has no `mtllib`, UVs or
texture. It can pass MAGICIAN's mesh loader and depth/mask rasterizer, but
MAGICIAN's `SoftPhongShader` raises `ValueError: Meshes does not have textures`.
This is a real RGB harness incompatibility. Do not add a dummy texture and do
not call the geometry-only result a full planning smoke-test pass. A faithful
next step is a renderer adapter that renders the official 3DGS PLY in the same
camera frame, with an explicit observation/planner/evaluator permission boundary.
