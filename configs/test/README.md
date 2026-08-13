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
