# Description of config files

Below is a detailed description of all the hyperparameters involved in evaluating our models.<br>

## 1. Large-scale 3D scenes exploration and reconstruction with an RGB camera

| Parameter | Type | Description |
| :----- | :-----: | :----- |
| `numGPU` | int  | GPU device to use for evaluation. |
| `dataset_path` | str | Path to the data directory. |
| `test_scenes` | List of str | List of scenes to explore during evaluation. Strings should be equal to the scene directory names in the dataset folder. |
| `params_name` | str | Name of the config file corresponding to the model to be evaluated. |
| `model_name` | str | Name of the weights file corresponding to the model to be evaluated. |
| `results_json_name` | str | Name of the json file in which results will be saved. |
| `test_resolution` | str | Distance threshold used to compute surface coverage during evaluation. |
| `use_perfect_depth_map` | bool | Selects renderer ground-truth depth when `true`. This flag has priority over `kind_depth_map`; the default remains `true` for backward-compatible evaluation. |
| `kind_depth_map` | str | Case-insensitive registered depth backend selected only when `use_perfect_depth_map` is `false`. Whitespace is ignored. Missing, empty, `GT`, `NONE`, or unregistered values fail during startup; there is no GT fallback. `DA3` selects the pinned Depth Anything 3 adapter. |
| `da3_model_id` / `da3_model_revision` | str | Hugging Face model and immutable revision. The default nested model supplies metric depth. |
| `da3_window_size` | int | Maximum number of recent RGB/pose frames used for pose-conditioned inference. |
| `da3_process_res` / `da3_process_res_method` | int / str | DA3 preprocessing resolution and resize method. |
| `da3_output_height` / `da3_output_width` | int | Planning output size; defaults to `256 x 456`. |
| `da3_confidence_percentile` | float / null | Optional calibrated per-frame confidence percentile. The default is `null`, which records confidence but does not discard finite valid depth using an uncalibrated threshold. |
| `da3_cache_enabled` | bool | Enables persistent DA3 result caching. `false` performs inference without reading or writing cache files; outputs must remain numerically equivalent. |
| `da3_scene_units_per_meter` | object | Required in DA3 mode: explicit positive `scene_units_per_meter` calibration keyed by every requested scene name. Missing scenes fail before model/dataset setup. `scene_scale_factor` is never treated as physical-unit evidence. |
| `validation_n_poses_in_trajectory` | int | Optional short-run override. `2` exercises three captured views because the planners include pose zero. Omit it for the production trajectory length. |
| `validation_max_start_positions` | int | Optional number of configured start poses to exercise. Omit it to run every start pose. |
| `validation_memory_dir_name` | str | Optional plain directory name that isolates validation captures from production/test memories. Paths and traversal components are rejected. |
| `validation_n_gt_surface_points` / `validation_n_proxy_points` | int | Optional short-run capacity overrides. Omit them to retain the training configuration. |
| `scene_mesh_transforms` | object | Optional per-scene exporter/world-coordinate adaptation. Each entry may specify an axis permutation, axis signs, translation, and positive preprocessing scale. Missing scenes use the identity transform. |
| `scene_texture_atlas_size` | int | Optional positive per-face texture atlas resolution used by both planning entry points and the real-scene CUDA gate. Defaults to the legacy value `32`; large textured meshes can lower it to bound loader memory without changing geometry or planning settings. |
| `validation_use_occupied_pose` | bool | Defaults to `true`. Set `false` only for a validated custom scene without `occupied_pose.pt`; mesh collision checks remain independent. |

Depth-source selection is intentionally strict. With `use_perfect_depth_map=true`,
the GT provider is constructed and `kind_depth_map` is ignored, even if it is
missing or names another backend. With `use_perfect_depth_map=false`, the named
non-GT backend must already be registered. DA3 imports and model construction
remain lazy, so the GT path does not load DA3. The DA3 cache is stored below the
captured-frame directory and is isolated by RGB content, camera parameters,
model/revision, preprocessing, explicit per-scene metric scale, and adapter
version. Pose translations are converted from calibrated scene units to meters
before pose-conditioned inference, then metric depth is converted back to scene
units. Training uses
the separate `use_perfect_depth` option and is unchanged.

`scene_metric_calibrations.json` records the evidence and arithmetic behind
accepted metric scales. `test_da3_eiffel_real_mesh_config.json` is the bounded
three-view configuration used to validate both planning entry points against a
real Macarons++ mesh; its Eiffel value is copied from that evidence manifest.
`test_da3_12-nw-6c-5_real_mesh_config.json` applies the scene's explicit
metric-tile recentering/z-up-to-y-up transform and does not reuse Eiffel's
calibration.
`test_da3_12-nw-6c-7_real_mesh_config.json` records an explicit identity
transform because its uploaded OBJ already materializes recentering, y-up
rotation, and the 0.1 preprocessing scale. Its `.obj.bak` source and independent
`settings.json` envelope establish the separate `1.0 scene unit/m` calibration;
reapplying the 12-NW-6C-5 transform would move the mesh outside all planning
bounds.

When DA3 is provided as an isolated dependency overlay rather than installed
in the active environment, use `scripts/run_with_da3_overlay.py` and set
`MAGICIAN_DA3_APPEND_PATHS` to the source/dependency directories. The paths are
appended after normal site-packages so an overlay cannot shadow the validated
CUDA Torch/PyTorch3D build.

## 1.1 Eiffel fair experiment and diagnostics

`scripts/run_eiffel_fair_experiments.py` generates reproducible Eiffel-only
GT/DA3 fair pairs plus confidence, scale, window, process-resolution, and
mapping ablations.  `--generate-only --suite all` materializes every config and
records unselected runs without using a GPU.  Executed runs persist online-only
metrics (depth/confidence summaries, geometry counts, coverage, trajectory,
latency, and CUDA memory) in both LMDB and JSON.

`scripts/analyze_planning_diagnostics.py` is intentionally a separate post-run
process.  It is the only experiment component that reads captured renderer
`zbuf/mask`, and writes `diagnostic_only=true` / `feedback_to_online_planner=false`.
Test-fit scale/shift is therefore a diagnostic, never an online DA3 correction.
See `LabLog/阶段 4：Eiffel 公平实验、诊断与调参记录.md` for exact commands,
runtime provenance, measured results, failed-attempt disclosure, and the
single-scene conclusion boundary.

## 2. 3D object reconstruction with a depth sensor

| Parameter | Type | Description |
| :----- | :-----: | :----- |
| `numGPU` | int  | GPU device to use for evaluation. |
| `data_path` | str | Path to the data directory.  |
| `params_name` | str | Name of the config file corresponding to the surface coverage gain module to be evaluated.  |
| `scone_occ_model_name` | str | Name of the weights file corresponding to the occupancy probability module to be evaluated. |
| `scone_vis_model_name` | str | Name of the weights file corresponding to the surface coverage gain module to be evaluated. |
| `pc_size` | int | Number of points in the sequence processed by the self-attention unit of the surface coverage gain. |
| `n_view_max` | int |  Maximum number of views for reconstruction. Starting from one random initial view, the model iteratively predicts NBVs and captures up to `n_view_max` depth maps. |
| `test_novel` | bool | If True, starts a test on categories of objects never seen during training. |
| `results_json_name` | str | Name of the json file in which results will be saved. |
