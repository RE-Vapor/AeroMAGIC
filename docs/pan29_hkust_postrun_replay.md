# PAN-29: HKUST post-run UE5 observation replay

PAN-29 is a derived visualization workflow for the accepted PAN-11 HKUST GT
20-observation run. It does not rerun the Planner and it does not feed UE5
images back into mapping.

The source experiment is fixed by
`configs/test/test_pioneer_hkust_pan11_position_only_20obs_a1_config.json`.
The workflow replays observation IDs `0, 5, 10, 14, 19` in UE5, preserving
each source observation ID, Planner position, and source capture timestamp.
The real UE5 capture time is recorded separately.

## Output contract

Each replay produces:

- one atomic PAN-15 raw RGB-D six-face bundle;
- one validated canonical observation bundle;
- one deterministic 1024 x 512 equirectangular RGB panorama;
- one receipt binding the source identity, requested and measured pose, full
  six-face basis, source and UE5 timestamps, and all hashes.

The final preview contains five rows. Each row shows the six RGB faces that the
PAN-11 Planner actually used and the UE5 panorama rendered later at the same
logical observation pose. It explicitly labels UE5 as
`post_run_visualization_only`.

PAN-11 and UE5 use different visual assets, geometry detail, materials,
lighting, and rendering paths. PAN-29 therefore makes no RGB-pixel, depth, or
coverage-parity claim.

## Coordinate and policy boundary

Planner positions are transformed to UE centimetres as:

```text
[ue_x, ue_y, ue_z] = [500 * planner_x, 500 * planner_z, 500 * planner_y]
```

The PAN-29 rig includes the polar-face roll required to reproduce the complete
`pytorch3d-world-axes-v1` basis, not merely the face forward vectors.

All five poses are permitted for offline rendering. Only observation `0` lies
inside the conservative PAN-13 fly-policy AABB; observations `5, 10, 14, 19`
must remain labelled as post-hoc visualization outside that policy.

## Execution

Run from a clean PAN-29 worktree. The runner refuses an existing output path,
an existing tmux session, a dirty checkout, or a GPU already using 2 GiB or
more:

```bash
scripts/run_pan29_ue5_replay.sh \
  pan29-hkust-gt20obs-ue5-replay-a1 \
  1 \
  /home/ubuntu/Projects/Pioneer/experiments/runs/integration-all-experiments/PAN-29-hkust-gt20obs-ue5-replay/attempt-001
```

The run publishes `replay_result.json` and the preview only after the source,
checkout, UE project, level, configuration, and script postflight checks pass.
Failed capture, adaptation, processing, or preview material is retained under
the attempt's `attempts/` directory and is never promoted to a PASS result.

## Evidence publication

PAN-29 is recorded as a derived visualization linked to the existing PAN-11
scientific experiment. It must not increment or alter the scientific
experiment registry counts, and it must not replace the original PAN-11
preview or any source artifact.

Registration is intentionally a separate post-visual-QA gate, rather than an
automatic runner side effect. First run the command against copies of the live
JSON and Markdown files, inspect the derived record, and back up the live files.
Then use the same command on the live pair:

```bash
scripts/register_pan29_derived_artifact.py \
  --registry-json /home/ubuntu/Projects/Pioneer/experiments/Pioneer_experiment_registry_v1.0.json \
  --registry-md /home/ubuntu/Projects/Pioneer/experiments/Pioneer_experiment_registry_v1.0.md \
  --replay-result "$RUN/replay_result.json" \
  --preview-sidecar "$RUN/preview/hkust_gt20obs_pan29_ue5_replay_preview.json" \
  --preview-commit "$RUN/preview/hkust_gt20obs_pan29_ue5_replay_preview.commit.json" \
  --derived-artifact-id PAN-29-HKUST-GT20OBS-UE5-REPLAY-PREVIEW-20260822
```

An identical second invocation is a byte-level no-op. Reusing the ID for
different content is rejected.
