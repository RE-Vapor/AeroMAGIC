# PAN-21 UE5 provider smoke

PAN-21 keeps the original MAGICIAN provider as the default for paper experiments.
The UE5 provider is a debug-only file/manifest exchange used to prove that the
existing Planner can consume a realistic full-sphere observation frontend.

## Provider switch

Original provider (paper experiments): run the existing MAGICIAN configuration
without `validation_initial_ue_manifest`.  No PAN-21 setting changes the default.

UE5 p0 provider (two-observation debug profile):

```bash
PIONEER_INITIAL_UE_MANIFEST=/absolute/p0/canonical_bundle/manifest.json \
  ./scripts/run_pan21_planner_smoke.sh SESSION GPU /absolute/planner-run
```

The current Planner consumes p0 through PAN-19 validation and PAN-10 fusion,
then writes its selected p1 to the online metrics trajectory.  Capture p1 with
the PAN-15 file provider and close the two-observation loop with:

```bash
CUDA_VISIBLE_DEVICES=1 ./scripts/run_pan21_smoke_loop.sh \
  /absolute/p0/canonical_bundle/manifest.json \
  /absolute/p1/canonical_bundle/manifest.json \
  /absolute/planner-run/metrics_debug_pan21_two_observation/pioneer_HKUST_0.online.json \
  /absolute/pan21-acceptance cuda:0
```

The acceptance command refuses dirty worktrees, missing inputs, or an existing
output path.  It checksum-validates both real UE5 bundles, runs one incomplete
bundle negative test before any state mutation, and consumes p0/p1 with the
existing `process_cubemap_observation` and `update_proxy_state` paths.

## Scope and limitations

- This is exactly two observations and one Planner move.  It is not a long
  trajectory, benchmark, main-table result, ablation, or Dataset v1 pipeline.
- Coverage is `debug_only` and non-comparable.
- The exchange is offline files, not RPC/shared memory, and has no throughput
  claim or broad fault-injection matrix.
- PAN-10 remains authoritative for six-face fusion, visibility union, gain
  deduplication, and the main Planner algorithm.
- After this smoke and PAN-14 final review pass, paper-stage UE5 development stops.
