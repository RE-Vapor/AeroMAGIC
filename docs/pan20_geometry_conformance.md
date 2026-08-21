# PAN-20 minimal UE5 geometry conformance

PAN-20 consumes the immutable PAN-15 analytic and HKUST raw manifests. It
does not recapture or alter either source fixture.

```bash
OPENCV_IO_ENABLE_OPENEXR=1 \
  /home/ubuntu/anaconda3/envs/magician_mve/bin/python \
  scripts/validate_pan20_geometry.py \
  --analytic-raw-manifest <PAN15 analytic>/raw_bundle/manifest.json \
  --hkust-raw-manifest <PAN15 HKUST>/raw_bundle/manifest.json \
  --output <new PAN20 artifact directory>
```

The analytic evidence establishes that UE `SCS_SCENE_DEPTH` in the exported
RGBA16F OpenEXR R channel is optical-axis camera-z in UE world units. With
`WorldToMeters=100`, the only canonical conversion is:

```text
z_m = EXR_R / WorldToMeters
range_m = z_m * sqrt(((u-cx)/fx)^2 + ((v-cy)/fy)^2 + 1)
```

For UE raster pixel centres, `fx=fy=N/(2*tan(FOV/2))` and
`cx=cy=(N-1)/2`. The provisional PAN-15 value `(N-1)/2` for focal length is
preserved in the raw manifest but corrected by the validated adapter.

The check covers the known plane, all six direction markers, signed-cardinal
pose structure, sky/no-hit mapping, PAN-10 direction-Gram agreement, HKUST
world point-cloud bounds, and all twelve cubemap seams. The output is atomic
and contains validated bundles, a machine-readable report, a visualization,
and per-file SHA-256 checksums.

The one-pose seam gate is deliberately robust to distant vegetation and
occlusion boundaries: all 12 adjacencies must be present, validity agreement
must be at least 99%, maximum per-seam median point distance at most 2 m, and
maximum relative p95 at most 5%. Absolute p95/max values remain in the report
and are not used to hide boundary outliers.

This remains a one-pose, one-resolution sanity check. RGBA16F carries binary16
quantization and cannot distinguish a valid value at the 65504-world-unit
ceiling from overflow/no-hit; such samples are invalidated. No PAN-10 fusion,
visibility union, gain accounting, or Planner code is changed.
