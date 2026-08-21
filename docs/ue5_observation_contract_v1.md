# UE5 observation Contract v1 (PAN-19)

This is the minimal file/manifest boundary for the paper-stage UE5 smoke test.
One request is accepted only as a complete, checksum-verified six-face bundle in
the order `front/back/left/right/up/down`.

The canonical planner-side depth is Euclidean ray range in metres.  Camera-z is
converted only in the adapter with:

```text
range = z * sqrt(((u-cx)/fx)^2 + ((v-cy)/fy)^2 + 1)
```

Sky, no-hit, and out-of-range pixels use `valid_mask=false` and `NaN` depth.
Each face carries a pixel-space `K_pixel` and the only persisted pose is the
right-handed 4x4 `T_world_from_cam`.  All six faces must share request, frame,
timestamp, optical centre, image size, and 90 degree FOV.

The module validates before any mapping mutation and adapts the existing PAN-10
observation object without modifying PAN-10 fusion, visibility union, gain
deduplication, or planner logic.

CPU-only verification:

```bash
python3 -m unittest tests.test_ue5_observation_contract -v
python3 scripts/make_pan19_contract_fixture.py --output /tmp/pan19-contract-v1
```

`manifest.json` stores per-array paths, byte counts, and SHA-256 digests.  The
writer refuses to overwrite an existing bundle and publishes its temporary
directory atomically only after every face has been serialized.  The fixture
generator also saves `negative_missing_down_manifest.json` and proves that the
validator rejects it.
