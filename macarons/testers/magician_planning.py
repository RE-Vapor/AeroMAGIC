import os
import sys
import gc
import shutil
import time
from ..utility.macarons_utils import *
from ..utility.utils import count_parameters
from ..utility.gaussian_utils import CamerasWrapper, convert_camera_from_pytorch3d_to_gs
from ..utility.magician_utils import *
from ..utility.planning_depth import (
    apply_planning_validation_limits,
    compute_planning_coverage,
    create_scene_depth_providers,
    path_is_blocked,
    process_planning_depth_frame,
    scene_texture_atlas_size,
    set_planning_seeds,
    update_proxy_state,
    validation_uses_occupied_pose,
)
from ..utility.scene_transform import (
    resolve_scene_mesh_transform,
    transform_scene_vertices,
)
from ..utility.experiment_metrics import (
    create_trajectory_metrics_recorder,
    write_online_metrics,
)
from ..utility.cross_tile_diagnostics import audit_neighbor_generation
from ..utility.huge_3dgs_adapter import (
    capture_planning_observation,
    create_scene_rgb_providers,
)
from ..utility.planning_observations import (
    CUBEMAP_BODY_EXTRINSICS_VERSION,
    CUBEMAP_RIG_FRAME_BODY,
    CUBEMAP_RIG_FRAME_WORLD,
    CUBEMAP_WORLD_EXTRINSICS_VERSION,
    build_cubemap_cameras,
    capture_cubemap_observation,
    process_cubemap_observation,
    visible_union_from_depth_maps,
)
from ..utility.position_only_planning import (
    PositionOnlyPlannerState,
    PositionOnlySpec,
    pose_index_to_xyz,
    position_neighbors,
    position_only_structure_audit,
    xyz_to_canonical_pose_index,
)
import trimesh
import lmdb

# ==================== RaDe-GS Integration ====================
RADE_GS_PATH = os.path.join(os.path.dirname(__file__), "../../RaDe-GS")
if RADE_GS_PATH not in sys.path:
    sys.path.insert(0, RADE_GS_PATH)


class SimpleGaussianModel:
    def __init__(self, means, opacities, scales, rotations, colors, device):
        """
        Args:
            means: (N, 3) locations
            opacities: (N, 1) opacity[0, 1]
            scales: (N, 3) 
            rotations: (N, 4) 
            colors: (N, 3) 
        """
        self.device = device
        self._xyz = means.to(device)
        self._opacity = self.inverse_sigmoid(opacities.to(device))  # logit
        self._scaling = torch.log(scales.to(device))  # log
        self._rotation = rotations.to(device)
        self._colors_precomp = colors.to(device)  

        self.active_sh_degree = 0
        self.max_sh_degree = 0
        self.max_radii2D = torch.zeros(means.shape[0], device=device)

    @staticmethod
    def inverse_sigmoid(x, eps=1e-6):
        """ logit: logit(x) = log(x / (1-x))"""
        x = torch.clamp(x, eps, 1 - eps)
        return torch.log(x / (1 - x))

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        # none
        return torch.zeros(self._xyz.shape[0], 1, 3, device=self.device)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def get_opacity_with_3D_filter(self):
        return self.get_opacity

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_scaling_n_opacity_with_3D_filter(self):
        return self.get_scaling, self.get_opacity

    @property
    def get_colors_precomp(self):
        return self._colors_precomp


def render_gaussian_depth(gaussian_means, gaussian_opacities, gaussian_scales,
                          gaussian_rotations, gaussian_colors, gs_camera, device,
                          bg_color=None, kernel_size=0.1):
    """

    Args:
        gaussian_means: (N, 3) 
        gaussian_opacities: (N, 1)
        gaussian_scales: (N, 3) 
        gaussian_rotations: (N, 4) 
        gaussian_colors: (N, 3) 
        gs_camera: GSCamera 
        device: torch device
        bg_color: 
        kernel_size: Mip-Splatting kernel size

    Returns:
        rendered_depth: (1, H, W) median depth
        rendered_image: (3, H, W) RGB image
    """
    if bg_color is None:
        bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)

    gaussians = SimpleGaussianModel(
        means=gaussian_means,
        opacities=gaussian_opacities,
        scales=gaussian_scales,
        rotations=gaussian_rotations,
        colors=gaussian_colors,
        device=device
    )
    import math
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    tanfovx = math.tan(gs_camera.FoVx * 0.5)
    tanfovy = math.tan(gs_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(gs_camera.image_height),
        image_width=int(gs_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=gs_camera.world_view_transform,
        projmatrix=gs_camera.full_proj_transform,
        sh_degree=0,
        campos=gs_camera.camera_center,
        prefiltered=False,
        require_coord=False,
        require_depth=True,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussians.get_xyz
    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=False, device=device)
    scales, opacity = gaussians.get_scaling_n_opacity_with_3D_filter
    rotations = gaussians.get_rotation
    colors_precomp = gaussians.get_colors_precomp

    with torch.no_grad():
        rendered_image, radii, _, _, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )

    return rendered_median_depth, rendered_image


def _uses_pioneer_observation(params):
    mode = getattr(params, "planning_observation_mode", "single")
    if mode not in {"single", "cubemap6"}:
        raise ValueError(
            "planning_observation_mode must be either 'single' or 'cubemap6'."
        )
    return mode == "cubemap6"


def _pioneer_planner_state_mode(params):
    mode = getattr(params, "pioneer_planner_state_mode", "legacy_pose5d")
    if mode not in {"legacy_pose5d", "position_only"}:
        raise ValueError(
            "pioneer_planner_state_mode must be 'legacy_pose5d' or "
            "'position_only'."
        )
    if mode == "position_only" and not _uses_pioneer_observation(params):
        raise ValueError(
            "position_only planner state requires planning_observation_mode="
            "cubemap6."
        )
    return mode


def _pioneer_rig_frame(params):
    default = (
        CUBEMAP_RIG_FRAME_WORLD
        if _pioneer_planner_state_mode(params) == "position_only"
        else CUBEMAP_RIG_FRAME_BODY
    )
    rig_frame = getattr(params, "pioneer_cubemap_rig_frame", None) or default
    if rig_frame not in {CUBEMAP_RIG_FRAME_BODY, CUBEMAP_RIG_FRAME_WORLD}:
        raise ValueError("pioneer_cubemap_rig_frame must be 'body' or 'world'.")
    if (
        _pioneer_planner_state_mode(params) == "position_only"
        and rig_frame != CUBEMAP_RIG_FRAME_WORLD
    ):
        raise ValueError("position_only planner state requires a world cubemap rig.")
    return rig_frame


def _pioneer_extrinsics_version(params):
    rig_frame = _pioneer_rig_frame(params)
    expected = (
        CUBEMAP_WORLD_EXTRINSICS_VERSION
        if rig_frame == CUBEMAP_RIG_FRAME_WORLD
        else CUBEMAP_BODY_EXTRINSICS_VERSION
    )
    configured = (
        getattr(params, "pioneer_cubemap_extrinsics_version", None) or expected
    )
    if configured != expected:
        raise ValueError(
            "pioneer_cubemap_extrinsics_version does not match the selected "
            f"rig frame; expected {expected!r}."
        )
    return expected


def _position_only_spec(camera, params):
    canonical = getattr(params, "pioneer_canonical_orientation_indices", None)
    if canonical is None:
        raise ValueError(
            "position_only requires pioneer_canonical_orientation_indices."
        )
    if not isinstance(canonical, (list, tuple)) or len(canonical) != 2:
        raise ValueError(
            "pioneer_canonical_orientation_indices must contain two integers."
        )
    return PositionOnlySpec(
        position_shape=(camera.pose_l, camera.pose_w, camera.pose_h),
        orientation_shape=(camera.pose_n_elev, camera.pose_n_azim),
        canonical_orientation_index=tuple(canonical),
    )


def _capture_planner_observation(camera, mesh, rgb_provider, params):
    if not _uses_pioneer_observation(params):
        return capture_planning_observation(camera, mesh, rgb_provider)
    if rgb_provider is not None:
        raise ValueError(
            "PIONEER cubemap6 currently supports the Python mesh/GT RGB provider only."
        )
    bundle = capture_cubemap_observation(
        camera=camera,
        mesh=mesh,
        face_size=int(getattr(params, "pioneer_face_size", 256)),
        ambient_light_intensity=float(params.ambient_light_intensity),
        rgb_provider=rgb_provider,
        rig_frame=_pioneer_rig_frame(params),
    )
    camera.last_observation_bundle = bundle
    return bundle


def _render_pioneer_gaussian_visibility(
    *,
    points,
    gaussian_means,
    gaussian_opacities,
    gaussian_scales,
    gaussian_rotations,
    gaussian_colors,
    reference_camera,
    camera,
    params,
    device,
    render_kind,
    metrics_recorder=None,
):
    """Render a full-sphere imagined observation and OR visibility once."""

    face_size = int(getattr(params, "pioneer_face_size", 256))
    znear_value = getattr(reference_camera, "znear", None)
    if torch.is_tensor(znear_value):
        znear_value = float(znear_value.reshape(-1)[0].detach().cpu().item())
    elif znear_value is None:
        znear_value = 1.0
    else:
        znear_value = float(znear_value)
    face_cameras = build_cubemap_cameras(
        center=reference_camera.get_camera_center(),
        znear=znear_value,
        zfar=float(camera.zfar),
        device=device,
        reference_camera=reference_camera,
        rig_frame=_pioneer_rig_frame(params),
    )
    depth_maps = {}
    for face_name, face_camera in face_cameras.items():
        face_camera.K = (
            face_camera.get_projection_transform().get_matrix().transpose(-1, -2)
        )
        gs_camera = convert_camera_from_pytorch3d_to_gs(
            face_camera,
            height=face_size,
            width=face_size,
            device=device,
        )[0]
        rendered_depth, _ = render_gaussian_depth(
            gaussian_means=gaussian_means,
            gaussian_opacities=gaussian_opacities,
            gaussian_scales=gaussian_scales,
            gaussian_rotations=gaussian_rotations,
            gaussian_colors=gaussian_colors,
            gs_camera=gs_camera,
            device=device,
            bg_color=torch.tensor([1.0, 1.0, 1.0], device=device),
            kernel_size=0.01,
        )
        depth_maps[face_name] = rendered_depth[0]
    if metrics_recorder is not None:
        metrics_recorder.record_imagined_bundle_render(
            face_renders=len(depth_maps), kind=render_kind
        )
    return visible_union_from_depth_maps(
        points=points,
        face_cameras=face_cameras,
        depth_maps=depth_maps,
        depth_tolerance=1.0,
    )


def update_gaussian_colors_from_novelty(novelty_values):
    """
    use novelty_values update Gaussian colors

    Args:
        novelty_values: 
            0 = unknown → white [1,1,1]
            1 = known → black [0,0,0]
    """
    inverted_values = 1.0 - novelty_values
    colors = inverted_values.unsqueeze(1).repeat(1, 3)
    return colors

# ==================== End RaDe-GS Integration ====================

dir_path = os.path.abspath(os.path.dirname(__file__))
# data_path = os.path.join(dir_path, "../../../../../../datasets/rgb")
data_path = os.path.join(dir_path, "../../data/scenes")
results_dir = os.path.join(dir_path, "../../results/scene_exploration")
weights_dir = os.path.join(dir_path, "../../weights/macarons")
configs_dir = os.path.join(dir_path, "../../configs/macarons")

def setup_test(params, model_path, device, verbose=True, use_occupied_pose=True):
    # Create dataloader
    _, _, test_dataloader = get_dataloader(train_scenes=params.train_scenes,
                                           val_scenes=params.val_scenes,
                                           test_scenes=params.test_scenes,
                                           batch_size=1,
                                           ddp=False, jz=False,
                                           world_size=None, ddp_rank=None,
                                           data_path=params.data_path,
                                           use_occupied_pose=use_occupied_pose)
    print("\nThe following scenes will be used to test the model:")
    for batch, elem in enumerate(test_dataloader):
        print(elem['scene_name'][0])

    # Create model
    macarons = load_pretrained_macarons(pretrained_model_path=params.pretrained_model_path,
                                        device=device, learn_pose=params.learn_pose)


    trained_weights = torch.load(model_path, map_location=device, weights_only=False)
    macarons.load_state_dict(trained_weights["model_state_dict"], ddp=True)  # todo: replace by params.ddp
    depth_losses = np.array(trained_weights["depth_losses"])
    depth_losses_per_epoch = (depth_losses[::2] + depth_losses[1::2]) / 2
    # depth_losses_per_epoch = depth_losses
    print("\nModel name:", model_path)
    print("\nThe model has", (count_parameters(macarons.depth) + count_parameters(macarons.scone)) / 1e6,
          "trainable parameters.")
    print("It has been trained for", trained_weights["epoch"], "epochs.")
    print("The loss was:", depth_losses_per_epoch[-1], depth_losses_per_epoch[-1] * 3 / 4)
    print(params.n_alpha, "additional frames are used for depth prediction.")

    # Creating memory
    print("\nUsing memory folders", params.memory_dir_name)
    scene_memory_paths = []
    for scene_name in params.test_scenes:
        scene_path = os.path.join(test_dataloader.dataset.data_path, scene_name)
        scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
        scene_memory_paths.append(scene_memory_path)
    memory = Memory(scene_memory_paths=scene_memory_paths, n_trajectories=params.n_memory_trajectories,
                    current_epoch=0, verbose=verbose)

    return test_dataloader, macarons, memory


def setup_test_scene(params,
                     mesh,
                     settings,
                     mirrored_scene,
                     device,
                     mirrored_axis=None,
                     surface_scene_feature_dim=1,
                     test_resolution=0.05,
                     covered_scene_feature_dim=1):
    """
    Setup the different scene objects used for prediction and performance evaluation.

    :param params:
    :param mesh:
    :param settings:
    :param device:
    :param is_master:
    :return:
    """

    # Initialize gt_scene: we use this scene to store gt surface points to evaluate the performance of the model.
    # This scene is not used for supervision during training, since the model is self-supervised from RGB data
    # captured in real-time.
    gt_scene = Scene(x_min=settings.scene.x_min,
                     x_max=settings.scene.x_max,
                     grid_l=settings.scene.grid_l,
                     grid_w=settings.scene.grid_w,
                     grid_h=settings.scene.grid_h,
                     cell_capacity=params.surface_cell_capacity,
                     cell_resolution=test_resolution * params.scene_scale_factor,
                     n_proxy_points=params.n_proxy_points,
                     device=device,
                     view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                     feature_dim=3,
                     mirrored_scene=mirrored_scene,
                     mirrored_axis=mirrored_axis)  # We use colors as features

    covered_scene = Scene(x_min=settings.scene.x_min,
                          x_max=settings.scene.x_max,
                          grid_l=settings.scene.grid_l,
                          grid_w=settings.scene.grid_w,
                          grid_h=settings.scene.grid_h,
                          cell_capacity=params.surface_cell_capacity,
                          cell_resolution=test_resolution * params.scene_scale_factor,
                          n_proxy_points=params.n_proxy_points,
                          device=device,
                          view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                          feature_dim=covered_scene_feature_dim,
                          mirrored_scene=mirrored_scene,
                          mirrored_axis=mirrored_axis)  # We use colors as features

    # We fill gt_scene with points sampled on the surface of the ground truth mesh
    gt_surface, gt_normals, gt_surface_colors = get_scene_gt_surface(gt_scene=gt_scene,
                                                         verts=mesh.verts_list()[0],
                                                         faces=mesh.faces_list()[0],
                                                         n_surface_points=params.n_gt_surface_points,
                                                         return_colors=True,
                                                         mesh=mesh)
    gt_scene.fill_cells(gt_surface, features=gt_surface_colors)

    # Initialize surface_scene: we store in this scene the surface points computed by the depth model from RGB images
    surface_scene = Scene(x_min=settings.scene.x_min,
                          x_max=settings.scene.x_max,
                          grid_l=settings.scene.grid_l,
                          grid_w=settings.scene.grid_w,
                          grid_h=settings.scene.grid_h,
                          cell_capacity=params.surface_cell_capacity,
                          cell_resolution=None,
                          n_proxy_points=params.n_proxy_points,
                          device=device,
                          view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                          feature_dim=surface_scene_feature_dim,  # We use visibility history as features
                          mirrored_scene=mirrored_scene,
                          mirrored_axis=mirrored_axis)

    # Initialize proxy_scene: we store in this scene the proxy points
    proxy_scene = Scene(x_min=settings.scene.x_min,
                        x_max=settings.scene.x_max,
                        grid_l=settings.scene.grid_l,
                        grid_w=settings.scene.grid_w,
                        grid_h=settings.scene.grid_h,
                        cell_capacity=params.proxy_cell_capacity,
                        cell_resolution=params.proxy_cell_resolution,
                        n_proxy_points=params.n_proxy_points,
                        device=device,
                        view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                        feature_dim=1,  # We use proxy points indices as features
                        mirrored_scene=mirrored_scene,
                        score_threshold=params.score_threshold,
                        mirrored_axis=mirrored_axis)
    proxy_scene.initialize_proxy_points()

    return gt_scene, covered_scene, surface_scene, proxy_scene


def setup_test_camera(params,
                      mesh, intersector, start_cam_idx,
                      settings,
                      occupied_pose_data,
                      device,
                      training_frames_path,
                      mirrored_scene=False,
                      mirrored_axis=None,
                      rgb_provider=None):
    """
    Setup the camera used for prediction.

    :param params:
    :param mesh:
    :param start_cam_idx:
    :param settings:
    :param occupied_pose_data:
    :param device:
    :param training_frames_path:
    :return:
    """
    # Default camera to initialize the renderer
    n_camera = 1
    camera_dist = [10 * params.scene_scale_factor] * n_camera  # 10
    camera_elev = [30] * n_camera
    camera_azim = [260] * n_camera  # 160
    R, T = look_at_view_transform(camera_dist, camera_elev, camera_azim)
    zfar = params.zfar
    fov_camera = FoVPerspectiveCameras(R=R, T=T, zfar=zfar, device=device)

    renderer = get_rgb_renderer(image_height=params.image_height,
                                image_width=params.image_width,
                                ambient_light_intensity=params.ambient_light_intensity,
                                cameras=fov_camera,
                                device=device,
                                max_faces_per_bin=200000
                                )

    # Initialize camera
    camera = Camera(x_min=settings.camera.x_min, x_max=settings.camera.x_max,
                    pose_l=settings.camera.pose_l, pose_w=settings.camera.pose_w, pose_h=settings.camera.pose_h,
                    pose_n_elev=settings.camera.pose_n_elev, pose_n_azim=settings.camera.pose_n_azim,
                    n_interpolation_steps=params.n_interpolation_steps, zfar=params.zfar,
                    renderer=renderer,
                    device=device,
                    contrast_factor=settings.camera.contrast_factor,
                    gathering_factor=params.gathering_factor,
                    occupied_pose_data=occupied_pose_data,
                    save_dir_path=training_frames_path,
                    mirrored_scene=mirrored_scene,
                    mirrored_axis=mirrored_axis)  # Change or remove this path during inference or test


    state_mode = _pioneer_planner_state_mode(params)
    pioneer_observation = _uses_pioneer_observation(params)
    if state_mode == "position_only":
        spec = _position_only_spec(camera, params)
        planner_state = PositionOnlyPlannerState(spec)
        start_pose_idx = torch.as_tensor(
            start_cam_idx, dtype=torch.long, device=camera.device
        )
        start_state_idx = pose_index_to_xyz(start_pose_idx)
        canonical_start_idx = xyz_to_canonical_pose_index(start_state_idx, spec)
        camera.pioneer_position_spec = spec
        camera.pioneer_planner_state = planner_state
        camera.pioneer_state_index_history = torch.zeros(
            0, 3, dtype=torch.long, device=camera.device
        )
        camera.initialize_camera(start_cam_idx=canonical_start_idx)
        planner_state.capture_and_commit(
            start_state_idx,
            _capture_planner_observation,
            camera,
            mesh,
            rgb_provider,
            params,
        )
        camera.pioneer_state_index_history = torch.vstack(
            (camera.pioneer_state_index_history, start_state_idx.reshape(1, 3))
        )
        print(
            "PIONEER planner contract: state=position_only dim=3 "
            f"canonical_orientation={list(spec.canonical_orientation_index)} "
            f"rig={_pioneer_rig_frame(params)} "
            f"extrinsics={_pioneer_extrinsics_version(params)}"
        )
        print("PIONEER action branches: legacy_raw=10 pan10_effective=6 position_only=6 orientation=0")
        print("PIONEER structure audit:", dict(position_only_structure_audit(spec)))
    elif pioneer_observation:
        # Legacy Camera owns a five-dimensional pose graph and records visits
        # as soon as it moves.  This path is retained only for controlled A/B
        # comparison and backwards compatibility.
        camera.initialize_camera(start_cam_idx=start_cam_idx)
        _capture_planner_observation(camera, mesh, rgb_provider, params)
        print(
            "PIONEER planner contract: state=legacy_pose5d dim=5 "
            f"rig={_pioneer_rig_frame(params)} "
            f"extrinsics={_pioneer_extrinsics_version(params)}"
        )
    else:
        camera.initialize_camera(start_cam_idx=start_cam_idx)
        _capture_planner_observation(camera, mesh, rgb_provider, params)

    return camera


def compute_magician_trajectory(params, macarons, camera, gt_scene, surface_scene,
                           proxy_scene, covered_scene, mesh, intersector, device, settings,
                           depth_provider, test_resolution=0.05,
                           compute_collision=False, metrics_recorder=None,
                           rgb_provider=None):

    macarons.eval()

    # compute scene_scales
    scene_bbox_x = settings.scene.x_max[0] - settings.scene.x_min[0]
    scene_bbox_y = settings.scene.x_max[1] - settings.scene.x_min[1]
    scene_bbox_z = settings.scene.x_max[2] - settings.scene.x_min[2]
    scene_scale = (scene_bbox_x + scene_bbox_y + scene_bbox_z) / 3.0
    print(f"Scene scale computed: {scene_scale:.2f} (bbox: x={scene_bbox_x:.2f}, y={scene_bbox_y:.2f}, z={scene_bbox_z:.2f})")

    full_pc = torch.zeros(0, 3, device=device)
    full_pc_colors = torch.zeros(0, 3, device=device)
    full_pc_idx = torch.zeros(0, 1, device=device)
    coverage_evolution = []
    pose_i = 0
    pioneer_observation = _uses_pioneer_observation(params)
    pioneer_state_mode = _pioneer_planner_state_mode(params)
    position_only = pioneer_state_mode == "position_only"
    position_spec = getattr(camera, "pioneer_position_spec", None)
    position_state = getattr(camera, "pioneer_planner_state", None)
    if position_only and (
        not isinstance(position_spec, PositionOnlySpec)
        or not isinstance(position_state, PositionOnlyPlannerState)
    ):
        raise RuntimeError(
            "position_only camera is missing its planner spec/observation state."
        )

    def camera_pose_index(planner_state_index):
        if position_only:
            return xyz_to_canonical_pose_index(planner_state_index, position_spec)
        return planner_state_index

    def camera_pose_from_state(planner_state_index):
        return camera.get_pose_from_idx(camera_pose_index(planner_state_index))

    K_matrix = None
    
    def process_current_frame():
        if pioneer_observation:
            bundle = camera.last_observation_bundle
            frame_data = process_cubemap_observation(
                bundle=bundle,
                proxy_points=proxy_scene.proxy_points,
                gathering_factor=(
                    params.gathering_factor
                    * getattr(params, "planning_gathering_factor_multiplier", 2.0)
                ),
                sensor_range=params.sensor_range,
                voxel_size=float(
                    getattr(params, "pioneer_voxel_size", test_resolution)
                ),
                device=device,
            )
            if pose_i == 0 and len(frame_data["part_pc"]) == 0:
                raise RuntimeError(
                    "PIONEER first observation contains no points within sensor_range."
                )
            if metrics_recorder is not None:
                metrics_recorder.record_observation_bundle(
                    frame_data["bundle_stats"],
                    provider_seconds=float(getattr(bundle, "capture_seconds", 0.0)),
                    geometry_seconds=float(frame_data.get("geometry_seconds", 0.0)),
                )
            return frame_data
        return process_planning_depth_frame(
            camera=camera,
            depth_provider=depth_provider,
            proxy_scene=proxy_scene,
            device=device,
            gathering_factor=(
                params.gathering_factor
                * getattr(params, "planning_gathering_factor_multiplier", 2.0)
            ),
            sensor_range=params.sensor_range,
            metrics_recorder=metrics_recorder,
            enforce_sensor_range_gate=(
                pose_i == 0 and getattr(params, "planning_range_gate_enabled", False)
            ),
            sensor_range_gate_quantile=getattr(
                params, "planning_range_gate_quantile", 0.9
            ),
            sensor_range_gate_min_points=getattr(
                params, "planning_range_gate_min_points", 1
            ),
        )
            

    while pose_i <= params.n_poses_in_trajectory:
        if pose_i % 10 == 0:
            print("Processing pose", str(pose_i) + "...")
        
        camera.fov_camera_0 = camera.fov_camera

        if pose_i > 0 and pose_i % params.recompute_surface_every_n_loop == 0:
            print("Recomputing surface...")
            fill_surface_scene(surface_scene, full_pc,
                               random_sampling_max_size=params.n_gt_surface_points,
                               min_n_points_per_cell_fill=3,
                               progressive_fill=params.progressive_fill,
                               max_n_points_per_fill=params.max_points_per_progressive_fill)

        frame_data = process_current_frame()

        # Unpdate scene
        part_pc_features = torch.zeros(len(frame_data['part_pc']), 1, device=device)
        covered_scene.fill_cells(frame_data['part_pc'], features=part_pc_features)
        surface_scene.fill_cells(frame_data['part_pc'], features=part_pc_features)
        full_pc = torch.vstack((full_pc, frame_data['part_pc']))
        full_pc_colors = torch.vstack((full_pc_colors, frame_data['part_pc_features']))
        part_pc_idx = torch.full((frame_data['part_pc'].shape[0], 1), pose_i, device=device)
        full_pc_idx = torch.vstack((full_pc_idx, part_pc_idx))

        update_proxy_state(
            camera=camera,
            proxy_scene=proxy_scene,
            frame_data=frame_data,
            carving_tolerance=params.carving_tolerance,
        )

        surface_scene.set_all_features_to_value(value=1.)

        # Compute coverage gain for evaulation
        current_coverage, current_cov = compute_planning_coverage(
            gt_scene=gt_scene,
            covered_scene=covered_scene,
            surface_epsilon=2 * test_resolution * params.scene_scale_factor,
            normalization=settings.scene.visibility_ratio,
        )
        if pose_i % 5 == 0:
            print("==========current coverage:", current_coverage)
        coverage_evolution.append(current_cov)
        if metrics_recorder is not None:
            metrics_recorder.record_coverage(current_coverage, current_cov)
            metrics_recorder.record_cross_tile_coverage(
                gt_scene=gt_scene,
                covered_scene=covered_scene,
                reconstruction_points=full_pc,
                surface_epsilon=2 * test_resolution * params.scene_scale_factor,
                normalization=settings.scene.visibility_ratio,
                global_raw=current_coverage,
                global_normalized=current_cov,
            )

        if pose_i >= params.n_poses_in_trajectory:
            break

        # Occupancy field prediction
        with torch.no_grad():
            X_world, view_harmonics, occ_probs = compute_scene_occupancy_probability_field(
                params, macarons.scone, camera, surface_scene, proxy_scene, device
            )
        # We only keep the points with occupancy value larger than 0.5
        filtered_X_world = X_world[occ_probs.squeeze() > 0.5]
        n_points = filtered_X_world.shape[0]
        gaussian_means = filtered_X_world  # (N, 3) 
        occ_values = occ_probs[occ_probs.squeeze() > 0.5] 

        # Convert occupancy field to Imagined Gaussians
        gaussian_opacities = occ_values    # (N, 1) 
        gaussian_scales = torch.ones(n_points, 3, device=device) * (0.7154/2)  
        gaussian_rotations = torch.tensor([[1, 0, 0, 0]], device=device, dtype=torch.float32).repeat(n_points, 1) 
        novelty_values = torch.zeros(n_points, device=device)  # (N,)
        gaussian_colors = update_gaussian_colors_from_novelty(novelty_values)  # (N, 3)

        if pose_i == 0 and not pioneer_observation:
            sample_X_cam = camera.X_cam_history[0].view(1, 3)
            sample_V_cam = camera.V_cam_history[0].view(1, 2)
            R_sample, T_sample = get_camera_RT(sample_X_cam, sample_V_cam)
            sample_camera = FoVPerspectiveCameras(R=R_sample, T=T_sample, zfar=camera.zfar, device=device)
            K_matrix = sample_camera.get_projection_transform().get_matrix().transpose(-1, -2)

        # 1. initialize all novelty_values to 0
        novelty_values = torch.zeros(n_points, device=device)

        # 2. revisit all previous cameras
        history_length = len(camera.X_cam_history)

        for cam_idx in range(history_length):
            current_X_cam = camera.X_cam_history[cam_idx]
            current_V_cam = camera.V_cam_history[cam_idx]
            X_cam = current_X_cam.view(1, 3)
            V_cam = current_V_cam.view(1, 2)
            R_cam, T_cam = get_camera_RT(X_cam, V_cam)
            current_fov_camera = FoVPerspectiveCameras(R=R_cam, T=T_cam, zfar=camera.zfar, device=device)
            with torch.no_grad():
                if pioneer_observation:
                    current_visible_mask = _render_pioneer_gaussian_visibility(
                        points=filtered_X_world,
                        gaussian_means=gaussian_means,
                        gaussian_opacities=gaussian_opacities,
                        gaussian_scales=gaussian_scales,
                        gaussian_rotations=gaussian_rotations,
                        gaussian_colors=gaussian_colors,
                        reference_camera=current_fov_camera,
                        camera=camera,
                        params=params,
                        device=device,
                        render_kind="history",
                        metrics_recorder=metrics_recorder,
                    )
                else:
                    current_fov_camera.K = K_matrix
                    gs_camera = convert_camera_from_pytorch3d_to_gs(
                        current_fov_camera,
                        height=camera.image_height,
                        width=camera.image_width,
                        device=device,
                    )[0]
                    rendered_depth, _ = render_gaussian_depth(
                        gaussian_means=gaussian_means,
                        gaussian_opacities=gaussian_opacities,
                        gaussian_scales=gaussian_scales,
                        gaussian_rotations=gaussian_rotations,
                        gaussian_colors=gaussian_colors,
                        gs_camera=gs_camera,
                        device=device,
                        bg_color=torch.tensor([1.0, 1.0, 1.0], device=device),
                        kernel_size=0.01,
                    )
                    current_visible_mask = camera.check_point_visibility_from_depth(
                        filtered_X_world,
                        current_fov_camera,
                        rendered_depth[0],
                        depth_tolerance=1.0,
                    )
                # update the novelty along the visited cameras
                novelty_values[current_visible_mask] = 1.0

        print(f"historical: {novelty_values.sum().item()}/{n_points}")

        cross_tile_enabled = bool(
            metrics_recorder is not None and metrics_recorder.cross_tile_enabled
        )
        planning_diagnostic = None
        if cross_tile_enabled:
            seam_x = metrics_recorder.cross_tile_seam_x
            tile_2_min_x_index = metrics_recorder.cross_tile_min_x_index
            imagined_tile_2 = filtered_X_world[:, 0] > seam_x
            novel_mask = novelty_values <= 0
            current_pose_index = [int(value) for value in camera.cam_idx.cpu().tolist()]
            seam_from_index = torch.tensor(
                metrics_recorder.cross_tile_gate_from_index,
                device=camera.device,
                dtype=torch.long,
            )
            seam_to_index = torch.tensor(
                metrics_recorder.cross_tile_gate_to_index,
                device=camera.device,
                dtype=torch.long,
            )
            seam_from_pose, _ = camera.get_pose_from_idx(seam_from_index)
            seam_to_pose, _ = camera.get_pose_from_idx(seam_to_index)
            seam_from_xyz = seam_from_pose[:3]
            seam_to_xyz = seam_to_pose[:3]
            planning_diagnostic = {
                'frame_id': pose_i,
                'current_pose_index': current_pose_index,
                'current_xyz': [float(value) for value in camera.X_cam[0].cpu().tolist()],
                'imagined_gaussians': {
                    'tile_1': int((~imagined_tile_2).sum().item()),
                    'tile_2': int(imagined_tile_2.sum().item()),
                },
                'novel_imagined_gaussians': {
                    'tile_1': int((novel_mask & ~imagined_tile_2).sum().item()),
                    'tile_2': int((novel_mask & imagined_tile_2).sum().item()),
                },
                'beam_steps': [],
                'direct_crossing_candidate_generated': False,
                'direct_crossing_candidate_legal': False,
                'seam_segment_collision': {
                    'from_pose_index': metrics_recorder.cross_tile_gate_from_index,
                    'to_pose_index': metrics_recorder.cross_tile_gate_to_index,
                    'from_xyz': [
                        float(value) for value in seam_from_xyz.cpu().tolist()
                    ],
                    'to_xyz': [float(value) for value in seam_to_xyz.cpu().tolist()],
                    'gt_mesh': bool(
                        line_segment_mesh_intersection(
                            seam_from_xyz.cpu().numpy(),
                            seam_to_xyz.cpu().numpy(),
                            intersector,
                        )
                    ),
                    'predicted_point_cloud': bool(
                        line_segment_intersects_point_cloud_region(
                            filtered_X_world, seam_from_xyz, seam_to_xyz
                        )
                    ),
                },
            }

        # 3. Beam Search 
        remaining_steps = params.n_poses_in_trajectory + 1 - history_length
        print(f"Beam search remain: {remaining_steps} steps")

        # initialize beam search
        initial_pose_idx = (
            pose_index_to_xyz(camera.cam_idx) if position_only else camera.cam_idx
        )
        beams = [{
            'trajectory': [],
            'novelty_values': novelty_values.clone(),
            'score': novelty_values.sum().item(),
            'total_coverage_gain': 0.0,  
            'current_pose_idx': initial_pose_idx
        }]

        # settings for beam search
        beam_width = params.beam_width
        for bs_i in range(params.beam_steps):
            print(f"Beam search step {bs_i + 1}/{params.beam_steps}")

            all_candidates = []
            search_step_started = time.perf_counter()
            search_step_metrics = {
                "planning_iteration": pose_i,
                "beam_step": bs_i,
                "parent_beam_count": 0,
                "raw_action_proposal_count": 0,
                "translation_action_proposal_count": 0,
                "orientation_action_proposal_count": 0,
                "generated_candidate_count": 0,
                "valid_state_candidate_count": 0,
                "observed_rejected_candidate_count": 0,
                "collision_rejected_candidate_count": 0,
                "rendered_candidate_count": 0,
                "retained_beam_count": 0,
            }
            step_diagnostic = None
            if cross_tile_enabled:
                step_diagnostic = {
                    'beam_step': bs_i,
                    'attempted_candidate_count': 0,
                    'generated_candidate_count': 0,
                    'legal_candidate_count': 0,
                    'generated_tile_2_candidate_count': 0,
                    'legal_tile_2_candidate_count': 0,
                    'rejections': [],
                    'candidates': [],
                }

            # extend to every beams
            for beam_index, beam in enumerate(beams):
                search_step_metrics["parent_beam_count"] += 1
                if position_only:
                    search_step_metrics["raw_action_proposal_count"] += 6
                    search_step_metrics["translation_action_proposal_count"] += 6
                    neighbor_indices = position_neighbors(
                        beam["current_pose_idx"], position_spec
                    )
                    valid_neighbors = position_state.valid_neighbors(
                        beam["current_pose_idx"]
                    )
                    observed_rejected = (
                        len(neighbor_indices) - len(valid_neighbors)
                        if any(
                            not position_state.is_observed(row)
                            for row in neighbor_indices
                        )
                        else 0
                    )
                else:
                    search_step_metrics["raw_action_proposal_count"] += 10
                    search_step_metrics["translation_action_proposal_count"] += 6
                    search_step_metrics["orientation_action_proposal_count"] += 4
                    neighbor_indices = camera.get_neighboring_poses(
                        pose_idx=beam['current_pose_idx']
                    )
                    if pioneer_observation and getattr(
                        params, "pioneer_remove_rotation_only_candidates", True
                    ):
                        translated = torch.any(
                            neighbor_indices[:, :3]
                            != beam["current_pose_idx"][:3].view(1, 3),
                            dim=1,
                        )
                        neighbor_indices = neighbor_indices[translated]
                    visited_flags = [
                        camera.get_pose_from_idx(row)[1] for row in neighbor_indices
                    ]
                    valid_neighbors = camera.get_valid_neighbors(
                        neighbor_indices=neighbor_indices, mesh=mesh
                    )
                    observed_rejected = (
                        sum(bool(value) for value in visited_flags)
                        if any(not bool(value) for value in visited_flags)
                        else 0
                    )
                search_step_metrics["generated_candidate_count"] += len(
                    neighbor_indices
                )
                search_step_metrics["valid_state_candidate_count"] += len(
                    valid_neighbors
                )
                search_step_metrics["observed_rejected_candidate_count"] += int(
                    observed_rejected
                )

                if cross_tile_enabled:
                    parent_pose_index = [
                        int(value) for value in beam['current_pose_idx'].cpu().tolist()
                    ]
                    if position_only:
                        position_bounds = (
                            camera.pose_l,
                            camera.pose_w,
                            camera.pose_h,
                        )
                        position_actions = (
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        )
                        attempts = []
                        for action in position_actions:
                            candidate = [
                                parent_pose_index[i] + action[i] for i in range(3)
                            ]
                            boundary = any(
                                value < 0 or value >= position_bounds[i]
                                for i, value in enumerate(candidate)
                            )
                            attempts.append(
                                {
                                    'pose_index': candidate,
                                    'rejection_reason': 'boundary' if boundary else None,
                                }
                            )
                        generation = {
                            'attempted_count': len(position_actions),
                            'attempts': attempts,
                        }
                    else:
                        generation = audit_neighbor_generation(
                            parent_pose_index,
                            (
                                camera.pose_l,
                                camera.pose_w,
                                camera.pose_h,
                                camera.pose_n_elev,
                                camera.pose_n_azim,
                            ),
                        )
                    step_diagnostic['attempted_candidate_count'] += generation[
                        'attempted_count'
                    ]
                    for attempt in generation['attempts']:
                        if attempt['rejection_reason'] == 'boundary':
                            step_diagnostic['rejections'].append({
                                'beam_parent': beam_index,
                                **attempt,
                            })
                    generated_indices = [
                        [int(value) for value in row.cpu().tolist()]
                        for row in neighbor_indices
                    ]
                    step_diagnostic['generated_candidate_count'] += len(
                        generated_indices
                    )
                    step_diagnostic['generated_tile_2_candidate_count'] += sum(
                        row[0] >= tile_2_min_x_index for row in generated_indices
                    )
                    if (
                        bs_i == 0
                        and parent_pose_index[0] == tile_2_min_x_index - 1
                        and any(
                            row[0] == tile_2_min_x_index
                            for row in generated_indices
                        )
                    ):
                        planning_diagnostic[
                            'direct_crossing_candidate_generated'
                        ] = True
                    has_unvisited = any(
                        (
                            not position_state.is_observed(row)
                            if position_only
                            else not camera.get_pose_from_idx(row)[1]
                        )
                        for row in neighbor_indices
                    )
                    if has_unvisited:
                        for row in neighbor_indices:
                            row_visited = (
                                position_state.is_observed(row)
                                if position_only
                                else camera.get_pose_from_idx(row)[1]
                            )
                            if row_visited:
                                step_diagnostic['rejections'].append({
                                    'beam_parent': beam_index,
                                    'pose_index': [
                                        int(value) for value in row.cpu().tolist()
                                    ],
                                    'rejection_reason': 'visited',
                                })

                rendering_candidate = []
                idx_candidate = []

                current_pose, _ = camera_pose_from_state(beam['current_pose_idx'])
                X_current, _, _ = camera.get_camera_parameters_from_pose(current_pose)
                current_loc = X_current[0].cpu().numpy()

                for row in valid_neighbors:
                    neighbor_pose, _ = camera_pose_from_state(row)
                    X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)
                    target_loc = X_neighbor[0].cpu().numpy()

                    collision_reason = None

                    if bs_i == 0 and getattr(params, "planning_shared_collision_gate", False):
                        if path_is_blocked(
                            current_loc,
                            target_loc,
                            intersector,
                            compute_collision=compute_collision,
                            intersection_fn=line_segment_mesh_intersection,
                        ):
                            collision_reason = 'gt_mesh_collision'
                    elif bs_i == 0:
                        if line_segment_mesh_intersection(current_loc, target_loc, intersector):
                            collision_reason = 'gt_mesh_collision'
                    elif getattr(params, "planning_shared_collision_gate", False):
                        # we use occupancy points to check for future collisions.
                        if compute_collision and line_segment_intersects_point_cloud_region(
                            filtered_X_world, X_current[0], X_neighbor[0]
                        ):
                            collision_reason = 'predicted_point_cloud_collision'
                    elif line_segment_intersects_point_cloud_region(
                        filtered_X_world, X_current[0], X_neighbor[0]
                    ):
                        collision_reason = 'predicted_point_cloud_collision'

                    if collision_reason is not None:
                        search_step_metrics[
                            "collision_rejected_candidate_count"
                        ] += 1
                        if cross_tile_enabled:
                            step_diagnostic['rejections'].append({
                                'beam_parent': beam_index,
                                'pose_index': [int(value) for value in row.cpu().tolist()],
                                'xyz': [float(value) for value in X_neighbor[0].cpu().tolist()],
                                'rejection_reason': collision_reason,
                            })
                        continue

                    rendering_candidate.append(fov_neighbor)
                    idx_candidate.append(row)
                    if cross_tile_enabled:
                        row_list = [int(value) for value in row.cpu().tolist()]
                        step_diagnostic['legal_candidate_count'] += 1
                        step_diagnostic['legal_tile_2_candidate_count'] += int(
                            row_list[0] >= tile_2_min_x_index
                        )
                        if (
                            bs_i == 0
                            and parent_pose_index[0] == tile_2_min_x_index - 1
                            and row_list[0] == tile_2_min_x_index
                        ):
                            planning_diagnostic['direct_crossing_candidate_legal'] = True

                if len(rendering_candidate) == 0:
                    continue

                # rendering for every pose
                for j, pose_idx in enumerate(idx_candidate):
                    fov_camera = rendering_candidate[j]

                    # update colors
                    current_novelty= beam['novelty_values']
                    gaussian_colors = update_gaussian_colors_from_novelty(current_novelty)

                    with torch.no_grad():
                        if pioneer_observation:
                            visible_mask = _render_pioneer_gaussian_visibility(
                                points=filtered_X_world,
                                gaussian_means=gaussian_means,
                                gaussian_opacities=gaussian_opacities,
                                gaussian_scales=gaussian_scales,
                                gaussian_rotations=gaussian_rotations,
                                gaussian_colors=gaussian_colors,
                                reference_camera=fov_camera,
                                camera=camera,
                                params=params,
                                device=device,
                                render_kind="candidate",
                                metrics_recorder=metrics_recorder,
                            )
                            coverage_gain = (
                                visible_mask & (current_novelty <= 0)
                            ).sum().item()
                        else:
                            fov_camera.K = K_matrix
                            gs_camera = convert_camera_from_pytorch3d_to_gs(
                                fov_camera,
                                height=camera.image_height,
                                width=camera.image_width,
                                device=device,
                            )[0]
                            rendered_depth, rendered_image = render_gaussian_depth(
                                gaussian_means=gaussian_means,
                                gaussian_opacities=gaussian_opacities,
                                gaussian_scales=gaussian_scales,
                                gaussian_rotations=gaussian_rotations,
                                gaussian_colors=gaussian_colors,
                                gs_camera=gs_camera,
                                device=device,
                                bg_color=torch.tensor([1.0, 1.0, 1.0], device=device),
                                kernel_size=0.01,
                            )
                            depth_map = rendered_depth[0]
                            visible_mask = camera.check_point_visibility_from_depth(
                                filtered_X_world,
                                fov_camera,
                                depth_map,
                                depth_tolerance=1.0,
                            )
                            valid_depth_mask = depth_map > 0
                            grayscale = rendered_image.mean(dim=0)
                            depth_threshold = scene_scale / 2.0
                            if valid_depth_mask.any():
                                depth_weight = (
                                    (depth_map / depth_threshold) ** 2
                                ).clamp_max(1.0)
                                coverage_gain = (
                                    grayscale
                                    * depth_weight
                                    * valid_depth_mask.float()
                                ).sum().item()
                            else:
                                coverage_gain = 0.0

                        new_novelty = current_novelty.clone()
                        new_novelty[visible_mask] = 1.0

                        new_total_coverage_gain = beam['total_coverage_gain'] + coverage_gain

                        all_candidates.append({
                            'trajectory': beam['trajectory'] + [pose_idx],
                            'novelty_values': new_novelty,
                            'coverage_gain': coverage_gain,  # single step
                            'total_coverage_gain': new_total_coverage_gain, 
                            'current_pose_idx': pose_idx,
                            '_diagnostic': ({
                                'beam_parent': beam_index,
                                'pose_index': [int(value) for value in pose_idx.cpu().tolist()],
                                'xyz': [float(value) for value in fov_camera.get_camera_center()[0].cpu().tolist()],
                                'tile': (
                                    'tile_2'
                                    if int(pose_idx[0].item()) >= tile_2_min_x_index
                                    else 'tile_1'
                                ),
                                'coverage_gain': float(coverage_gain),
                                'total_coverage_gain': float(new_total_coverage_gain),
                                'trajectory': [
                                    [int(value) for value in item.cpu().tolist()]
                                    for item in beam['trajectory'] + [pose_idx]
                                ],
                            } if cross_tile_enabled else None),
                        })
                        search_step_metrics["rendered_candidate_count"] += 1

            if len(all_candidates) == 0:
                print("No valid candidates found!")
                if cross_tile_enabled:
                    planning_diagnostic['beam_steps'].append(step_diagnostic)
                if not any(beam["trajectory"] for beam in beams):
                    beams = []
                search_step_metrics["search_seconds"] = (
                    time.perf_counter() - search_step_started
                )
                if metrics_recorder is not None:
                    metrics_recorder.record_planner_search_step(
                        search_step_metrics
                    )
                print("Planner search step summary:", search_step_metrics)
                break

            # coverage gains based on rgb imgs
            all_candidates.sort(key=lambda x: x['total_coverage_gain'], reverse=True)
            if cross_tile_enabled:
                for rank, candidate in enumerate(all_candidates, start=1):
                    candidate['_diagnostic']['rank'] = rank
                    candidate['_diagnostic']['top_10'] = rank <= beam_width
                    step_diagnostic['candidates'].append(candidate['_diagnostic'])
                tile_1_candidates = [
                    item['_diagnostic'] for item in all_candidates
                    if item['_diagnostic']['tile'] == 'tile_1'
                ]
                tile_2_candidates = [
                    item['_diagnostic'] for item in all_candidates
                    if item['_diagnostic']['tile'] == 'tile_2'
                ]
                best_tile_1 = tile_1_candidates[0] if tile_1_candidates else None
                best_tile_2 = tile_2_candidates[0] if tile_2_candidates else None
                step_diagnostic['best_tile_1'] = best_tile_1
                step_diagnostic['best_tile_2'] = best_tile_2
                step_diagnostic['best_tile_2_minus_tile_1_coverage_gain'] = (
                    best_tile_2['coverage_gain'] - best_tile_1['coverage_gain']
                    if best_tile_1 is not None and best_tile_2 is not None
                    else None
                )
                step_diagnostic['best_tile_2_minus_tile_1_total_coverage_gain'] = (
                    best_tile_2['total_coverage_gain']
                    - best_tile_1['total_coverage_gain']
                    if best_tile_1 is not None and best_tile_2 is not None
                    else None
                )
                step_diagnostic['top_10_trajectories_with_tile_2'] = sum(
                    any(
                        pose[0] >= tile_2_min_x_index
                        for pose in candidate['_diagnostic']['trajectory']
                    )
                    for candidate in all_candidates[:beam_width]
                )
                planning_diagnostic['beam_steps'].append(step_diagnostic)
            beams = all_candidates[:beam_width]
            search_step_metrics["retained_beam_count"] = len(beams)
            search_step_metrics["search_seconds"] = (
                time.perf_counter() - search_step_started
            )
            if metrics_recorder is not None:
                metrics_recorder.record_planner_search_step(search_step_metrics)
            print("Planner search step summary:", search_step_metrics)
            # print(f"Step {bs_i + 1}: Best total_coverage_gain = {beams[0]['total_coverage_gain']:.2f}, Score = {beams[0]['score']}/{n_points}, Current step gain = {beams[0].get('coverage_gain', 0):.2f}")
            print(f"Top {min(beam_width, len(all_candidates))} beams selected from {len(all_candidates)} candidates")

        if len(beams) > 0:
            best_beam = beams[0]
            best_trajectory = best_beam['trajectory']
        else:
            print("No valid trajectory found!")
            if cross_tile_enabled:
                planning_diagnostic['selected'] = None
                metrics_recorder.record_planning_diagnostic(planning_diagnostic)
            break

        # move one step
        next_idx = best_trajectory[0]
        print(f"move one step: pose_idx = {next_idx}")
        if cross_tile_enabled:
            selected_pose = [int(value) for value in next_idx.cpu().tolist()]
            first_step = planning_diagnostic['beam_steps'][0]
            rejection_reason_counts = {}
            for rejection in first_step['rejections']:
                reason = rejection['rejection_reason']
                rejection_reason_counts[reason] = (
                    rejection_reason_counts.get(reason, 0) + 1
                )
            planning_diagnostic['candidate_summary'] = {
                'attempted_count': first_step['attempted_candidate_count'],
                'generated_count': first_step['generated_candidate_count'],
                'legal_count': first_step['legal_candidate_count'],
                'generated_tile_2_count': first_step[
                    'generated_tile_2_candidate_count'
                ],
                'legal_tile_2_count': first_step[
                    'legal_tile_2_candidate_count'
                ],
                'rejection_reason_counts': rejection_reason_counts,
            }
            planning_diagnostic['selected'] = {
                'pose_index': selected_pose,
                'tile': (
                    'tile_2'
                    if selected_pose[0] >= tile_2_min_x_index
                    else 'tile_1'
                ),
                'reason': 'highest_total_coverage_gain',
                'winning_trajectory': [
                    [int(value) for value in item.cpu().tolist()]
                    for item in best_trajectory
                ],
                'coverage_gain': float(best_beam['coverage_gain']),
                'total_coverage_gain': float(best_beam['total_coverage_gain']),
            }
            metrics_recorder.record_planning_diagnostic(planning_diagnostic)

        interpolation_step = 1
        for i in range(camera.n_interpolation_steps):
            next_camera_idx = camera_pose_index(next_idx)
            camera.update_camera(
                next_camera_idx, interpolation_step=interpolation_step
            )
            if position_only:
                position_state.capture_and_commit(
                    next_idx,
                    _capture_planner_observation,
                    camera,
                    mesh,
                    rgb_provider,
                    params,
                )
                camera.pioneer_state_index_history = torch.vstack(
                    (
                        camera.pioneer_state_index_history,
                        next_idx.reshape(1, 3),
                    )
                )
            else:
                _capture_planner_observation(camera, mesh, rgb_provider, params)
            interpolation_step += 1

        pose_i += 1

    print("Coverage Evolution:", coverage_evolution)
    
    return coverage_evolution, camera.X_cam_history, camera.V_cam_history, full_pc, full_pc_colors, full_pc_idx
        
def run_magician_test(params_name,
             model_name,
             results_json_name,
             numGPU,
             test_scenes,
             test_resolution=0.05,
             use_perfect_depth_map=False,
             compute_collision=False,
             load_json=False,
             dataset_path=None,
             test_params=None):

    params_path = os.path.join(configs_dir, params_name)
    weights_path = os.path.join(weights_dir, model_name)
    results_json_path = os.path.join(results_dir, results_json_name)

    params = load_params(params_path)
    params.test_scenes = test_scenes
    params.jitter_probability = 0.
    params.symmetry_probability = 0.
    params.anomaly_detection = False
    params.memory_dir_name = "test_memory_" + str(numGPU)

    params.jz = False
    params.numGPU = numGPU
    params.WORLD_SIZE = 1
    params.batch_size = 1
    params.total_batch_size = 1

    max_start_positions = apply_planning_validation_limits(params, test_params)
    set_planning_seeds(test_params or {})

    for name, default in (
        ("planning_observation_mode", "single"),
        ("pioneer_face_size", 256),
        ("pioneer_face_fov_degrees", 90.0),
        ("pioneer_voxel_size", test_resolution),
        ("pioneer_remove_rotation_only_candidates", True),
        ("pioneer_planner_state_mode", "legacy_pose5d"),
        ("pioneer_cubemap_rig_frame", None),
        ("pioneer_cubemap_extrinsics_version", None),
        ("pioneer_canonical_orientation_indices", None),
    ):
        setattr(params, name, getattr(test_params, name, default))
    if _uses_pioneer_observation(params):
        state_mode = _pioneer_planner_state_mode(params)
        if params.pioneer_cubemap_rig_frame is None:
            params.pioneer_cubemap_rig_frame = (
                CUBEMAP_RIG_FRAME_WORLD
                if state_mode == "position_only"
                else CUBEMAP_RIG_FRAME_BODY
            )
        expected_extrinsics = (
            CUBEMAP_WORLD_EXTRINSICS_VERSION
            if params.pioneer_cubemap_rig_frame == CUBEMAP_RIG_FRAME_WORLD
            else CUBEMAP_BODY_EXTRINSICS_VERSION
        )
        if params.pioneer_cubemap_extrinsics_version is None:
            params.pioneer_cubemap_extrinsics_version = expected_extrinsics
        _pioneer_extrinsics_version(params)
        if state_mode == "position_only":
            canonical = params.pioneer_canonical_orientation_indices
            if (
                not isinstance(canonical, (list, tuple))
                or len(canonical) != 2
                or any(type(value) is not int for value in canonical)
            ):
                raise ValueError(
                    "position_only requires two integer "
                    "pioneer_canonical_orientation_indices."
                )
        if float(params.pioneer_face_fov_degrees) != 90.0:
            raise ValueError("PIONEER cubemap faces require exactly 90 degree FOV.")
        if (
            isinstance(params.pioneer_face_size, bool)
            or not isinstance(params.pioneer_face_size, int)
            or params.pioneer_face_size < 16
        ):
            raise ValueError("pioneer_face_size must be an integer >= 16.")
        if not use_perfect_depth_map:
            raise ValueError(
                "PIONEER cubemap6 pilot currently requires use_perfect_depth_map=true."
            )
        if params.n_interpolation_steps != 1:
            raise ValueError(
                "PIONEER cubemap6 currently requires n_interpolation_steps=1 "
                "so every captured bundle is processed exactly once."
            )

    if dataset_path is None:
        params.data_path = data_path
    else:
        params.data_path = dataset_path

    # Setup device
    device = setup_device(params, None)

    depth_providers = create_scene_depth_providers(
        test_params,
        scene_names=test_scenes,
        use_perfect_depth_map=use_perfect_depth_map,
        kind_depth_map=getattr(test_params, 'kind_depth_map', None),
        scene_scale_factor=params.scene_scale_factor,
        znear=params.znear,
        zfar=params.zfar,
        device=device,
    )
    rgb_providers = create_scene_rgb_providers(
        test_params, scene_names=test_scenes, device=device
    )

    # Setup model and dataloader
    dataloader, macarons, memory = setup_test(
        params,
        weights_path,
        device,
        use_occupied_pose=validation_uses_occupied_pose(test_params),
    )

    params.beam_width = test_params.beam_width
    params.beam_steps = test_params.beam_steps

    lmdb_dir = os.path.join(results_dir, test_params.lmdb_dir_name)
    os.makedirs(lmdb_dir, exist_ok=True)
    print(f"\nLMDB database directory: {lmdb_dir}")

    for i in range(len(dataloader.dataset)):
        scene_dict = dataloader.dataset[i]

        scene_names = [scene_dict['scene_name']]
        obj_names = [scene_dict['obj_name']]
        all_settings = [scene_dict['settings']]
        occupied_pose_datas = [scene_dict.get('occupied_pose')]

        batch_size = len(scene_names)

        for i_scene in range(batch_size):
            mesh = None
            torch.cuda.empty_cache()

            scene_name = scene_names[i_scene]
            depth_provider = depth_providers[scene_name]
            rgb_provider = rgb_providers[scene_name]
            obj_name = obj_names[i_scene]
            settings = all_settings[i_scene]
            settings = Settings(settings, device, params.scene_scale_factor)
            occupied_pose_data = occupied_pose_datas[i_scene]
            print("\nScene name:", scene_name)
            print("-------------------------------------")

            scene_path = os.path.join(dataloader.dataset.data_path, scene_name)
            mesh_path = os.path.join(scene_path, obj_name)
            # segmented_mesh_path = os.path.join(scene_path, 'segmented.obj')

            mirrored_scene = False
            mirrored_axis = None
            mesh_transform = resolve_scene_mesh_transform(test_params, scene_name)

            # Load mesh
            mesh = load_scene(mesh_path, params.scene_scale_factor, device,
                              mirror=mirrored_scene, mirrored_axis=mirrored_axis,
                              texture_atlas_size=scene_texture_atlas_size(test_params),
                              mesh_transform=mesh_transform)
           
            mesh_for_check = trimesh.load(mesh_path)

            if isinstance(mesh_for_check, trimesh.Scene):
                mesh_for_check = mesh_for_check.dump(concatenate=True)
            mesh_for_check.vertices = transform_scene_vertices(
                mesh_for_check.vertices,
                mesh_transform,
                scene_scale_factor=params.scene_scale_factor,
            )

            intersector = mesh_for_check.ray

            print("Mesh Vertices shape:", mesh.verts_list()[0].shape)
            print("Min Vert:", torch.min(mesh.verts_list()[0], dim=0)[0],
                  "\nMax Vert:", torch.max(mesh.verts_list()[0], dim=0)[0])

            # Use memory info to set frames and poses path
            scene_memory_path = os.path.join(scene_path, params.memory_dir_name)

            torch.cuda.empty_cache()

            start_position_count = len(settings.camera.start_positions)
            if max_start_positions is not None:
                start_position_count = min(start_position_count, max_start_positions)
            for start_cam_idx_i in range(start_position_count):
                start_cam_idx = settings.camera.start_positions[start_cam_idx_i]
                print("\n" + "="*60)
                print(f"Start cam index {start_cam_idx_i} for {scene_name}: {start_cam_idx}")
                print("="*60)

                # Each start_cam_idx_i gets its own trajectory number
                trajectory_nb = start_cam_idx_i
                training_frames_path = memory.get_trajectory_frames_path(scene_memory_path, trajectory_nb)
                print(f"Using trajectory folder: {training_frames_path}")

                # Setup the Scene and Camera objects
                gt_scene, covered_scene, surface_scene, proxy_scene = None, None, None, None
                gc.collect()
                torch.cuda.empty_cache()
                gt_scene, covered_scene, surface_scene, proxy_scene = setup_test_scene(params,
                                                                                       mesh,
                                                                                       settings,
                                                                                       mirrored_scene,
                                                                                       device,
                                                                                       mirrored_axis=mirrored_axis,
                                                                                       test_resolution=test_resolution)

                # Start telemetry before the initial real observation so its
                # six face renders and CUDA peak are included.
                metrics_recorder = create_trajectory_metrics_recorder(
                    test_params,
                    planner=(
                        "pioneer" if _uses_pioneer_observation(params) else "magician"
                    ),
                    scene=scene_name,
                    start_index=start_cam_idx_i,
                    capture_dir=training_frames_path,
                    device=device,
                )

                # clear_folder(training_frames_path)
                camera = setup_test_camera(params, mesh, intersector, start_cam_idx, settings, occupied_pose_data,
                                           device, training_frames_path,
                                           mirrored_scene=mirrored_scene, mirrored_axis=mirrored_axis,
                                           rgb_provider=rgb_provider)
                if metrics_recorder is not None and _uses_pioneer_observation(params):
                    audit_spec = getattr(camera, "pioneer_position_spec", None)
                    if audit_spec is None:
                        current_orientation = tuple(
                            int(value)
                            for value in camera.cam_idx[3:].detach().cpu().tolist()
                        )
                        audit_spec = PositionOnlySpec(
                            position_shape=(
                                camera.pose_l,
                                camera.pose_w,
                                camera.pose_h,
                            ),
                            orientation_shape=(
                                camera.pose_n_elev,
                                camera.pose_n_azim,
                            ),
                            canonical_orientation_index=current_orientation,
                        )
                    metrics_recorder.record_planner_structure(
                        position_only_structure_audit(audit_spec)
                    )
                print(camera.X_cam_history[0], camera.V_cam_history[0])

                coverage_evolution, X_cam_history, V_cam_history, full_pc, full_pc_colors, full_pc_idx = compute_magician_trajectory(params, macarons,
                                                                                      camera,
                                                                                      gt_scene, surface_scene,
                                                                                      proxy_scene, covered_scene,
                                                                                      mesh,
                                                                                      intersector,
                                                                                      device,
                                                                                      settings,
                                                                                      depth_provider=depth_provider,
                                                                                      test_resolution=test_resolution,
                                                                                      compute_collision=compute_collision,
                                                                                      metrics_recorder=metrics_recorder,
                                                                                      rgb_provider=rgb_provider)

                experiment_metrics = None
                experiment_metrics_path = None
                planner_state_index_history = getattr(
                    camera, "pioneer_state_index_history", camera.cam_idx_history
                )
                if metrics_recorder is not None:
                    experiment_metrics = metrics_recorder.finalize(
                        X_cam_history=X_cam_history,
                        V_cam_history=V_cam_history,
                        final_point_count=len(full_pc),
                        pose_index_history=camera.cam_idx_history,
                        planner_state_index_history=planner_state_index_history,
                    )
                    experiment_metrics_path = write_online_metrics(
                        test_params, experiment_metrics
                    )
                

                # Open LMDB, save data, then close
                print(f"\n=== Saving trajectory data to LMDB ===")
                lmdb_env = lmdb.open(lmdb_dir, map_size=30 * 1024 * 1024 * 1024)

                # Save trajectory data to LMDB
                lmdb_key = f"{scene_name}/{start_cam_idx_i}"
                trajectory_data = {
                    'coverage': coverage_evolution,
                    'X_cam_history': X_cam_history.cpu().numpy(),
                    'V_cam_history': V_cam_history.cpu().numpy(),
                    'planner_state_mode': _pioneer_planner_state_mode(params),
                    'planner_state_index_history': (
                        planner_state_index_history.cpu().numpy()
                    ),
                    'points': full_pc.cpu().numpy(),
                    'points_color': full_pc_colors.cpu().numpy(),
                    'experiment_metrics': experiment_metrics,
                    'experiment_metrics_path': experiment_metrics_path,
                }
                save_to_lmdb(lmdb_env, lmdb_key, trajectory_data)

                # Close LMDB
                lmdb_env.close()
                print(f"Closed LMDB database for {scene_name}/{start_cam_idx_i}\n")

                # Cleanup: Keep only imgs folder, delete frames/depths/occupancy folders
                # cleanup_trajectory_folders(training_frames_path, keep_folders=['imgs'])
                # print(f"Finished processing trajectory {start_cam_idx_i}\n")

    print("All trajectories computed.")
