# PAN-23 UE 5.4 analytic project

`unreal/PAN23/PioneerPAN23.uproject` is the paper-stage content-only UE project.
It uses the installed UE 5.4.4 build and does not copy PAN-13 source assets.

Static validation, level construction, and one offscreen smoke capture:

```bash
python3 -m unittest tests.test_pan23_ue_project -v
bash unreal/PAN23/Scripts/run_build_analytic_level.sh /tmp/pan23-build
PAN23_GPU_INDEX=1 bash unreal/PAN23/Scripts/run_offscreen_smoke.sh /tmp/pan23-smoke
```

The level contains a known 5 m front plane, asymmetric axis markers, a box
crossing the front/right boundary, and open sky.  The saved rig has one
`PAN23_SixFaceRigRoot` and six attached `SceneCapture2D` actors.  The capture
report numerically records the shared optical centre, each fixed rotation and
forward axis, the common 90 degree FOV, resolution, WorldToMeters, RHI, engine
version, per-face RGB checksum, and the fixed lighting preset.

The preset uses SkyAtmosphere, DirectionalLight, SkyLight, and cast shadows.
Auto exposure, motion blur, temporal upsampling, and Lumen are disabled.  A
VolumetricCloud actor is intentionally absent because clouds are non-blocking
for paper stage A.

PAN-23 produces RGB only.  It deliberately does not decide UE depth encoding,
export canonical depth, implement RPC/shared memory, run trajectories, or add
dataset/weather/ERP machinery; those are outside this task.
