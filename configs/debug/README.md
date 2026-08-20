# Planning debug profiles

These profiles are compute-only overlays for an existing test config. They do
not replace the scene, model, depth source, calibration, collision, or renderer
settings in that base config.

| Profile | Proxy points | Beam | Interpolation steps | Captured observations | Starts | Purpose |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `quick` | 100,000 | 3 x 3 | 1 | 3 | 1 | Fast logic and wiring check |
| `magician` | 200,000 | 5 x 5 | 1 | 11 | Moderate MAGICIAN-style debug run |
| `large-scene` | 200,000 | 3 x 3 | 1 | 21 | Longer traversal with bounded branching |

All profiles fix `random_seed=8` and `torch_seed=9`. They also set
`debug_only=true` and `coverage_comparable=false`. Coverage from a debug profile
must not be compared with a paper or formal-experiment config.

Select a profile on top of any existing test config:

```bash
python test_magician_planning.py \
  -c test_in_default_scenes_config.json \
  --debug-profile quick
```

The SCONE entry point accepts the same option:

```bash
python test_scenes.py \
  -c test_in_default_scenes_config.json \
  --debug-profile large-scene
```

For a config-pinned run, add a top-level selector instead:

```json
{
  "debug_profile": "magician"
}
```

Do not set a different profile in the CLI and the config. Profile values take
precedence over the matching compute fields in the base config. Debug LMDB,
memory, metrics, and result paths receive a profile-specific suffix so they do
not share output locations with the base run.

To return to formal parameters, omit `--debug-profile` and remove the optional
`debug_profile` key. The base config itself is never rewritten.
