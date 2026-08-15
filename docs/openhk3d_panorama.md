# OpenHK3D GT panorama pilot

`tools/openhk3d_panorama.py` renders a validated assembled OpenHK3D scene as
paired RGB and ground-truth range-depth panoramas. It is the first data-stage
pilot for panoramic Active Mapping; it does not change MAGICIAN's perspective
camera or planning logic.

The output camera uses a 2:1 equirectangular projection with a 360-degree
horizontal field of view and a 180-degree vertical field of view. RGB is stored
as RGBA PNG, where alpha zero marks unobserved background. Depth is stored as a
32-bit OpenEXR Z pass and represents radial distance from the camera center in
MAGICIAN world units.

By default the tool renders every validated `start_positions` entry from the
scene's `settings.json`. It reproduces MAGICIAN's position-grid, elevation, and
azimuth formulas and refuses occupied start positions. The panoramic frame is
kept level with world up at `+Y`; the start pose azimuth sets longitude zero,
while the perspective elevation is recorded as provenance but does not tilt the
equirectangular horizon.

```bash
python tools/openhk3d_panorama.py \
  --scene-dir /path/to/Macarons++/12-NW-6C-7_8 \
  --output /path/to/panoramas/12-NW-6C-7_8-equirect-gt \
  --blender /path/to/blender \
  --compute-device CUDA
```

The output directory must not exist. Rendering occurs in a sibling staging
directory and is moved into place only after all configured poses have RGB and
non-empty depth outputs.

Each `poses/<number>/` directory contains `rgb.png`, `depth.exr`, and
`pose.json`. `panorama-manifest.json` records the source hashes, exact Blender
runtime and compute device, projection convention, pose extrinsics, file hashes,
depth statistics, quality gates, and a canonical scope fingerprint.

The panorama output does not establish that MAGICIAN can consume a panoramic
observation. That requires a later adapter for spherical visibility, depth
unprojection, view-state updates, and panoramic model inputs. The assembled
tile seam also remains diagnostic until a continuity tolerance is approved.
