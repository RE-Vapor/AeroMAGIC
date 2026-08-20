import argparse
from macarons.testers.magician_planning import *
from macarons.utility.debug_profiles import DEBUG_PROFILE_NAMES, apply_debug_profile

dir_path = os.path.abspath(os.path.dirname(__file__))
test_configs_dir = os.path.join(dir_path, "./configs/test/")
debug_profiles_dir = os.path.join(dir_path, "./configs/debug/")


if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='Script to test a full macarons model in large 3D scenes.')
    parser.add_argument('-c', '--config', type=str, help='name of the config file. '
                                                         'Default is "test_in_default_scenes_config.json".')
    parser.add_argument('--debug-profile', choices=DEBUG_PROFILE_NAMES,
                        help='optional compute-only debug overlay for the selected config')
    parser.add_argument(
        '--debug-profiles-dir',
        default=debug_profiles_dir,
        help='directory containing the selected debug profile JSON',
    )
    parser.add_argument(
        '--macarons-params-path',
        help='optional immutable snapshot of the base Macarons parameter JSON',
    )

    args = parser.parse_args()

    if args.config:
        params_name = args.config
    else:
        params_name = "test_in_default_scenes_config.json"

    params_name = os.path.join(test_configs_dir, params_name)
    test_params = load_params(params_name)
    debug_profile = apply_debug_profile(
        test_params,
        cli_profile_name=args.debug_profile,
        profiles_dir=args.debug_profiles_dir,
    )
    if debug_profile is not None:
        print(
            f"Debug profile: {debug_profile['name']} -- "
            "coverage_comparable=false; do not compare this run with formal experiments."
        )


    with torch.no_grad():
        run_magician_test(params_name=test_params.params_name,
                 model_name=test_params.model_name,
                 results_json_name=test_params.results_json_name,
                 numGPU=test_params.numGPU,
                 test_scenes=test_params.test_scenes,
                 test_resolution=test_params.test_resolution,
                 use_perfect_depth_map=test_params.use_perfect_depth_map,
                 compute_collision=test_params.compute_collision,
                 load_json=test_params.load_json,
                 dataset_path=test_params.dataset_path,
                 test_params=test_params,
                 params_path_override=args.macarons_params_path)
