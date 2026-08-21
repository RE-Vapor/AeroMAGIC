# PAN-15 fixed-pose UE RGB-D capture

The paper-stage exchange is intentionally file based:

```text
request.json -> UE raw six-face capture -> checksums -> candidate Contract v1 bundle
```

Run the analytic fixture and the read-only PAN-13 HKUST scene from a clean shell:

```bash
bash scripts/run_pan15_capture.sh analytic /tmp/pan15-analytic
bash scripts/run_pan15_capture.sh hkust /tmp/pan15-hkust
```

Each published artifact contains the immutable request, UE log, raw bundle,
candidate canonical bundle, adapter log, exit codes, and provenance.  The
wrapper writes into a sibling temporary directory and renames it only after UE
capture, all per-file checksums, Contract validation, and adapter serialization
succeed.  Failed attempts are retained below an `attempts/` directory.

For every face the raw bundle preserves RGB bytes and PNG, UE
`SCS_SCENE_DEPTH` and `SCS_DEVICE_DEPTH` in RGBA16F OpenEXR targets (the R
channel is authoritative), diagnostic Python readback copies, pixel intrinsics,
a right-handed `T_world_from_cam`, fixed capture metadata, and detailed
capture/readback/export timings. Failed R32F and RGBA32F attempts are retained
as evidence: UE 5.4/Vulkan returned no Python samples for those paths. The
RGBA16F Python readback clips or normalizes scene depth, so it is explicitly
non-authoritative. Binary16 source precision and its 65504-world-unit limit are
known limitations pending PAN-20 validation.

The candidate adapter verifies and reads the OpenEXR R channel, provisionally
treats `SCS_SCENE_DEPTH` as ray distance in UE world units, recomputes the
valid/no-hit mask, and converts valid samples to metres. This hypothesis is marked
`candidate_only_pending_PAN-20`; PAN-15 does not claim final depth or coordinate
correctness.  Raw values remain available so PAN-20 can test and, if needed,
replace the conversion without recapturing.

No online RPC, shared memory, trajectory loop, Planner modification, weather
matrix, ERP export, or dataset release machinery is included.
