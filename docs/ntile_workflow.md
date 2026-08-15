# Manifest-driven N-tile planning workflow

The N-tile workflow keeps geometry assembly, runtime units, planning range,
experiment isolation, and coverage accounting in one fail-closed contract. It
does not infer seams from a scene name. A manifest supplies an ordered partition
on runtime world axis 0, 1, or 2; `N - 1` strictly increasing boundaries define
exactly `N` tile IDs using intervals `(-inf,b0]`, `(b0,b1]`, ..., `(bN,+inf)`.

## Prepare, gate, run, and accept

Copy `configs/test/ntile_12-nw-6c-7_8.example.json`, replace the machine-local
paths and GPU, then run:

```bash
python scripts/prepare_ntile_workflow.py --manifest workflow.json
```

The preparer refuses a nonempty output directory and validates all of the
following before it creates the isolated dataset view and configs:

- assembly member count equals the declared tile count;
- every boundary lies strictly inside the assembled runtime bounds;
- OBJ, MTL, settings, occupied-pose, and assembly-manifest SHA-256 values match;
- the selected start index is in the camera lattice and explicitly unoccupied;
- the calibration report belongs to the scene and its range is within both its
  geometry hard cap and renderer `zfar`;
- only the documented isolation, budget, gate, metrics, GPU, and explicit
  `sensor_range` keys differ from the baseline configuration.

The generated `manifest.json` contains argv arrays named `gate_command`,
`gate_accept_command`, `command`, and `accept_command`. Run them in that order.
The two-observation gate must succeed before the full trajectory is started.
Both acceptance steps require a live range-gate pass and exact recombination of
per-tile covered/reference counts into global coverage. Full acceptance also
requires reference and covered points in every declared tile.

Online metrics are opt-in through `experiment_tile_metrics_enabled=true` and
`experiment_tile_partition`. They are emitted under `tile_metrics`; the older
two-tile `cross_tile` output remains backward compatible and is not required by
the generic workflow.

## Add one more tile

To extend an accepted N-tile scene to N+1 tiles:

1. Assemble the adjacent geometry and regenerate `settings.json`,
   `occupied_pose.pt`, and `occupied_pose.json`; keep the assembly member order
   consistent with the chosen partition axis.
2. Add the member to `assembly-manifest.json`, insert its separating runtime
   boundary in sorted order, and add its stable tile ID at the corresponding
   position. There is no code change for a third or later tile.
3. Recompute and replace every asset hash. Re-run sensor-range calibration for
   the expanded renderer/camera bounds and preregistered start-view evidence.
4. Choose a fresh output root and run ID, prepare the workflow, pass the short
   gate, then run and accept the full budget.

Do not reuse a prior calibration report, occupied-pose evidence, output root, or
asset hashes after geometry changes. A rejected preflight or gate is evidence
to repair the assembly/calibration contract, not a reason to relax it.
