## MYL-18 Stage-4 protocol

Both runs use official HUGE `point_cloud_utm50.ply` RGB observations through
the 3DGS adapter, the five configured starts, beam width/steps `10 x 10`, 101
observations per start, collision checking, fixed seeds 670/670, and the legacy
MAGICIAN point gathering radius `2 x gathering_factor`.

The planner contract remains 256x456. The official 102.6M-Gaussian PLY is
rasterized on the verified 128x128 PyTorch3D square-NDC grid and its RGB is
bilinearly resized to the renderer's 256x456 pixel grid: native rectangular
rasterization exceeded 24 GB on an RTX 3090. DA3 depth and perfect mesh z-buffer
depth both remain native 256x456; no depth is resized from the lower-resolution
3DGS render.

The depth intervention is the only intended observation-source difference:

- DA3 uses `Depth-Anything-3@3d835ec1` and the explicit calibration
  `huge_1_office = 1.0 scene unit/m`.
- perfect depth uses the benchmark mesh z-buffer with the HUGE camera's
  1-1600 m range. Its RGB still comes from the official 3DGS, not from the
  untextured mesh.

The calibration is not inferred from MAGICIAN's historical
`scene_scale_factor=10`. The source Terra metadata declares EPSG:32650, whose
Cartesian axes use metres; the Gaussian metadata is local ENU at the matching
origin height. `conversion_manifest.json` records a rigid determinant-1
`(E,N,U)->(X,Y,Z)=(E,U,-N)` transform with scale `1.0`, and the fixed-view probe
validated 3DGS/mesh depth without scale fitting (median relative errors 0.461%
expected and 0.220% median). The HUGE-specific model config therefore keeps
`scene_scale_factor=1.0` and the DA3 calibration is explicitly `1.0`.
