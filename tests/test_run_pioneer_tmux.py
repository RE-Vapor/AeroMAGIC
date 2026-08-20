import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_pioneer_tmux.sh"
DA3_BOOTSTRAP = ROOT / "scripts" / "run_with_da3_overlay.py"


class RunPioneerTmuxTests(unittest.TestCase):
    def _run_inner(self, executable_name, debug_profile="quick", legacy=False):
        executable = shutil.which(executable_name)
        self.assertIsNotNone(executable)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        run_dir = root / "run"
        repo_dir = root / "repo"
        run_dir.mkdir()
        repo_dir.mkdir()
        command = [
            "bash",
            str(RUNNER),
            "--inside-tmux",
            "0",
            str(run_dir),
            executable,
            "fake.json",
        ]
        if not legacy:
            command.append(debug_profile)
        command.append(str(repo_dir))
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        status = dict(
            line.split("=", 1)
            for line in (run_dir / "status.txt").read_text(encoding="utf-8").splitlines()
        )
        return completed, status

    def test_success_exit_is_persisted_after_function_locals_expire(self):
        completed, status = self._run_inner("true")
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(status["exit_code"], "0")
        self.assertIn("finished_at_utc", status)

    def test_failure_exit_is_persisted_after_function_locals_expire(self):
        completed, status = self._run_inner("false")
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(status["exit_code"], "1")
        self.assertIn("finished_at_utc", status)

    def test_debug_profile_is_forwarded_to_planner(self):
        completed, status = self._run_inner("echo", "pioneer-50")
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(status["exit_code"], "0")
        self.assertIn(
            "test_magician_planning.py -c fake.json --debug-profile pioneer-50",
            completed.stdout,
        )

    def test_legacy_inside_signature_defaults_to_quick(self):
        completed, status = self._run_inner("echo", legacy=True)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(status["exit_code"], "0")
        self.assertIn("--debug-profile quick", completed.stdout)

    def test_enforced_inner_run_rejects_checkout_changed_after_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            run_dir = root / "run"
            repo.mkdir()
            run_dir.mkdir()
            marker = repo / "tracked.txt"
            marker.write_text("clean\n", encoding="utf-8")
            entrypoint = repo / "test_magician_planning.py"
            entrypoint.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('changed during run\\n')\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "add", "--", "tracked.txt", "test_magician_planning.py"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-c",
                    "commit.gpgSign=false",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                cwd=repo,
                check=True,
            )
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            config = run_dir / "config.json"
            profile = run_dir / "quick.json"
            macarons_params = run_dir / "macarons_params.json"
            config.write_text("{}\n", encoding="utf-8")
            profile.write_text("{}\n", encoding="utf-8")
            macarons_params.write_text("{}\n", encoding="utf-8")
            (run_dir / "manifest.txt").write_text(
                "\n".join(
                    (
                        f"git_commit={commit}",
                        "config_snapshot=config.json",
                        f"config_sha256={hashlib.sha256(config.read_bytes()).hexdigest()}",
                        "debug_profile_snapshot=quick.json",
                        f"debug_profile_sha256={hashlib.sha256(profile.read_bytes()).hexdigest()}",
                        "macarons_params_snapshot=macarons_params.json",
                        f"macarons_params_sha256={hashlib.sha256(macarons_params.read_bytes()).hexdigest()}",
                        "depth_source=GT",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PIONEER_ENFORCE_RUN_SNAPSHOT"] = "1"
            completed = subprocess.run(
                [
                    "bash",
                    str(RUNNER),
                    "--inside-tmux",
                    "0",
                    str(run_dir),
                    sys.executable,
                    "ignored.json",
                    "quick",
                    str(repo),
                ],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("became dirty", completed.stderr)
            status = dict(
                line.split("=", 1)
                for line in (run_dir / "status.txt").read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(status["snapshot_integrity_preflight"], "PASS")
            self.assertEqual(status["snapshot_integrity_postflight"], "FAIL")
            self.assertEqual(status["exit_code"], "2")

    def test_outer_runner_derives_profile_manifest_and_rejects_mismatch(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        repo_dir = root / "repo"
        runner = repo_dir / "scripts" / RUNNER.name
        da3_bootstrap = repo_dir / "scripts" / DA3_BOOTSTRAP.name
        test_configs = repo_dir / "configs" / "test"
        debug_profiles = repo_dir / "configs" / "debug"
        macarons_configs = repo_dir / "configs" / "macarons"
        fake_bin = root / "bin"
        for directory in (
            runner.parent, test_configs, debug_profiles, macarons_configs, fake_bin
        ):
            directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RUNNER, runner)
        shutil.copy2(DA3_BOOTSTRAP, da3_bootstrap)
        fake_entrypoint = repo_dir / "test_magician_planning.py"
        fake_entrypoint.write_text(
            "import os\n"
            "from pathlib import Path\n"
            "target = os.environ.get('PIONEER_TEST_TAMPER_ASSET')\n"
            "if target:\n"
            "    Path(target).write_bytes(b'tampered-during-run')\n",
            encoding="utf-8",
        )

        profile = {
            "name": "pioneer-50",
            "description": "test fixture",
            "debug_only": True,
            "coverage_comparable": False,
            "output_suffix": "debug_pioneer50",
            "overrides": {
                "validation_n_proxy_points": 100000,
                "beam_width": 3,
                "beam_steps": 3,
                "validation_n_interpolation_steps": 1,
                "validation_n_poses_in_trajectory": 49,
                "experiment_budget_observations": 50,
                "validation_max_start_positions": 1,
                "random_seed": 8,
                "torch_seed": 9,
            },
        }
        (debug_profiles / "pioneer-50.json").write_text(
            json.dumps(profile), encoding="utf-8"
        )
        scene_root = repo_dir / "data" / "Macarons++" / "HKUST"
        weight_path = repo_dir / "weights" / "macarons" / "trained_macarons.pth"
        scene_root.mkdir(parents=True)
        weight_path.parent.mkdir(parents=True)
        scene_files = {
            "adaptation_manifest.json": b"adaptation",
            "12-NW-6C.obj": b"mtllib 12-NW-6C.mtl\n",
            "12-NW-6C.mtl": b"newmtl fixture\nmap_Kd texture.jpg\n",
            "texture.jpg": b"texture",
            "settings.json": b"settings",
            "occupied_pose.pt": b"occupied",
        }
        for name, payload in scene_files.items():
            (scene_root / name).write_bytes(payload)
        fixture_texture_tree = hashlib.sha256()
        for name in ("12-NW-6C.mtl", "texture.jpg"):
            fixture_texture_tree.update(name.encode("utf-8"))
            fixture_texture_tree.update(b"\0")
            fixture_texture_tree.update(
                hashlib.sha256(scene_files[name]).hexdigest().encode("ascii")
            )
            fixture_texture_tree.update(b"\n")
        weight_path.write_bytes(b"weight")
        macarons_params = macarons_configs / "macarons_default_training_config.json"
        macarons_params.write_text('{"znear":0.5,"zfar":750.0}\n', encoding="utf-8")
        calibration_path = test_configs / "scene_metric_calibrations.json"
        calibration_path.write_text(
            json.dumps(
                {
                    "calibrations": {
                        "HKUST": {
                            key: {
                                "path": f"data/Macarons++/HKUST/{name}",
                                "sha256": hashlib.sha256(scene_files[name]).hexdigest(),
                            }
                            for key, name in {
                                "adaptation_manifest": "adaptation_manifest.json",
                                "mesh": "12-NW-6C.obj",
                                "settings": "settings.json",
                            }.items()
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        source_files = {
            "__init__.py": b"",
            "api.py": b"PROVENANCE_FIXTURE = True\n",
        }
        source_tree_digest = hashlib.sha256()
        for relative_path, payload in sorted(source_files.items()):
            source_tree_digest.update(relative_path.encode("utf-8"))
            source_tree_digest.update(b"\0")
            source_tree_digest.update(hashlib.sha256(payload).hexdigest().encode("ascii"))
            source_tree_digest.update(b"\n")
        source_tree_sha256 = source_tree_digest.hexdigest()
        for name, run_dir in (
            ("derived.json", "../artifacts/attempt-001"),
            ("mismatch.json", "../artifacts/mismatch"),
        ):
            (test_configs / name).write_text(
                json.dumps(
                    {
                        "test_scenes": ["HKUST"],
                        "params_name": "macarons_default_training_config.json",
                        "macarons_params_sha256": hashlib.sha256(
                            macarons_params.read_bytes()
                        ).hexdigest(),
                        "debug_profile": "pioneer-50",
                        "experiment_run_dir": run_dir,
                        "pioneer_planner_state_mode": "position_only",
                        "pioneer_filter_occupied_position_candidates": True,
                        "validation_require_complete_occupied_pose": True,
                        "pioneer_cubemap_rig_frame": "world",
                        "pioneer_cubemap_extrinsics_version": (
                            "pytorch3d-world-axes-v1"
                        ),
                        "pioneer_canonical_orientation_indices": [2, 0],
                        "scene_texture_tree_sha256": fixture_texture_tree.hexdigest(),
                        "use_perfect_depth_map": False,
                        "kind_depth_map": "DA3",
                        "da3_model_id": "depth-anything/DA3NESTED-GIANT-LARGE",
                        "da3_model_revision": "model-revision",
                        "da3_model_config_sha256": hashlib.sha256(
                            b"model-config"
                        ).hexdigest(),
                        "da3_model_weights_sha256": hashlib.sha256(
                            b"model-weights"
                        ).hexdigest(),
                        "da3_source_revision": "source-revision",
                        "da3_source_tree_sha256": source_tree_sha256,
                        "da3_window_size": 3,
                        "da3_process_res": 504,
                        "da3_process_res_method": "upper_bound_resize",
                        "da3_output_height": 256,
                        "da3_output_width": 256,
                        "da3_confidence_percentile": None,
                        "da3_cache_enabled": True,
                        "da3_cache_dir": "../artifacts/attempt-001/da3-cache",
                        "da3_scene_units_per_meter": {"HKUST": 0.2},
                    }
                ),
                encoding="utf-8",
            )

        tmux_log = root / "tmux.log"
        fake_tmux = fake_bin / "tmux"
        fake_tmux.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$*\" >> \"$FAKE_TMUX_LOG\"\n"
            "if [ \"$1\" = has-session ]; then exit 1; fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_tmux.chmod(0o755)
        fake_sha256sum = fake_bin / "sha256sum"
        fake_sha256sum.write_text(
            "#!/bin/sh\n"
            "\"$PIONEER_PYTHON\" -c 'import hashlib, sys; "
            "p = sys.argv[1]; print(hashlib.sha256(open(p, \"rb\").read()).hexdigest(), p)' "
            "\"$1\"\n",
            encoding="utf-8",
        )
        fake_sha256sum.chmod(0o755)

        subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_dir,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True
        )
        subprocess.run(
            [
                "git",
                "add",
                "--",
                "scripts/run_pioneer_tmux.sh",
                "scripts/run_with_da3_overlay.py",
                "test_magician_planning.py",
                "configs/debug/pioneer-50.json",
                "configs/test/derived.json",
                "configs/test/mismatch.json",
                "configs/test/scene_metric_calibrations.json",
                "configs/macarons/macarons_default_training_config.json",
                "data/Macarons++/HKUST/adaptation_manifest.json",
                "data/Macarons++/HKUST/12-NW-6C.obj",
                "data/Macarons++/HKUST/12-NW-6C.mtl",
                "data/Macarons++/HKUST/texture.jpg",
                "data/Macarons++/HKUST/settings.json",
                "data/Macarons++/HKUST/occupied_pose.pt",
                "weights/macarons/trained_macarons.pth",
            ],
            cwd=repo_dir,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "commit.gpgSign=false",
                "-c",
                "core.hooksPath=/dev/null",
                "commit",
                "-qm",
                "test fixture",
            ],
            cwd=repo_dir,
            check=True,
        )

        environment = os.environ.copy()
        hf_home = root / "hf-home"
        overlay_deps = root / "overlay-deps"
        overlay_source = root / "overlay-source-source-revision"
        for directory in (hf_home, overlay_deps, overlay_source):
            directory.mkdir()
        model_snapshot = (
            hf_home
            / "hub"
            / "models--depth-anything--DA3NESTED-GIANT-LARGE"
            / "snapshots"
            / "model-revision"
        )
        model_snapshot.mkdir(parents=True)
        (model_snapshot / "config.json").write_bytes(b"model-config")
        model_weights = model_snapshot / "model.safetensors"
        model_weights.write_bytes(b"model-weights")
        da3_package = overlay_source / "depth_anything_3"
        da3_package.mkdir()
        for relative_path, payload in source_files.items():
            (da3_package / relative_path).write_bytes(payload)
        environment.update(
            {
                "FAKE_TMUX_LOG": str(tmux_log),
                "PATH": str(fake_bin) + os.pathsep + environment["PATH"],
                "PIONEER_PYTHON": sys.executable,
                "PIONEER_DA3_HF_HOME": str(hf_home),
                "PIONEER_DA3_APPEND_PATHS": os.pathsep.join(
                    (str(overlay_source), str(overlay_deps))
                ),
            }
        )
        derived_run_dir = root / "artifacts" / "attempt-001"
        completed = subprocess.run(
            [
                "bash",
                str(runner),
                "pioneer-outer-derived",
                "0",
                str(derived_run_dir),
                "derived.json",
            ],
            cwd=repo_dir,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        manifest = dict(
            line.split("=", 1)
            for line in (derived_run_dir / "manifest.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        self.assertEqual(manifest["debug_profile"], "pioneer-50")
        self.assertEqual(
            manifest["config_sha256"],
            hashlib.sha256((derived_run_dir / "config.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            manifest["debug_profile_sha256"],
            hashlib.sha256(
                (derived_run_dir / "pioneer-50.json").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(manifest["scene"], "HKUST")
        self.assertEqual(manifest["macarons_params_name"], macarons_params.name)
        self.assertEqual(
            manifest["macarons_params_sha256"],
            hashlib.sha256(macarons_params.read_bytes()).hexdigest(),
        )
        self.assertEqual(manifest["expected_observations"], "50")
        self.assertEqual(manifest["expected_real_face_renders"], "300")
        self.assertEqual(manifest["planner_state_mode"], "position_only")
        self.assertEqual(
            manifest["filter_occupied_position_candidates"], "true"
        )
        self.assertEqual(manifest["require_complete_occupied_pose"], "true")
        self.assertEqual(manifest["planner_state_dimension"], "3")
        self.assertEqual(manifest["cubemap_rig_frame"], "world")
        self.assertEqual(
            manifest["cubemap_extrinsics_version"],
            "pytorch3d-world-axes-v1",
        )
        self.assertEqual(manifest["canonical_orientation_indices"], "[2,0]")
        self.assertEqual(manifest["raw_action_branches_per_parent"], "6")
        self.assertEqual(manifest["orientation_action_branches_per_parent"], "0")
        self.assertEqual(manifest["depth_source"], "DA3")
        self.assertEqual(manifest["use_perfect_depth_map"], "false")
        self.assertEqual(manifest["kind_depth_map"], "DA3")
        self.assertEqual(manifest["da3_window_size"], "3")
        self.assertEqual(
            manifest["da3_model_config_sha256"], hashlib.sha256(b"model-config").hexdigest()
        )
        self.assertEqual(
            manifest["da3_model_weights_sha256"], hashlib.sha256(b"model-weights").hexdigest()
        )
        self.assertEqual(manifest["da3_source_tree_sha256"], source_tree_sha256)
        self.assertEqual(
            manifest["da3_import_package_tree_sha256"], source_tree_sha256
        )
        self.assertEqual(manifest["da3_process_res"], "504")
        self.assertEqual(manifest["da3_process_res_method"], "upper_bound_resize")
        self.assertEqual(manifest["da3_output_size"], "256x256")
        self.assertEqual(manifest["da3_confidence_percentile"], "null")
        self.assertEqual(manifest["da3_cache_enabled"], "true")
        self.assertEqual(
            manifest["da3_cache_dir"],
            str((repo_dir / "../artifacts/attempt-001/da3-cache").resolve()),
        )
        self.assertEqual(manifest["da3_scene_units_per_meter"], "0.2")
        self.assertEqual(
            manifest["da3_import_origin"],
            str((overlay_source / "depth_anything_3" / "api.py").resolve()),
        )
        self.assertEqual(manifest["scene_asset_hashes_verified"], "1")
        self.assertRegex(manifest["scene_texture_tree_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(manifest["hf_hub_offline"], "1")
        self.assertEqual(manifest["hf_home"], str(hf_home))
        tmux_text = tmux_log.read_text(encoding="utf-8")
        self.assertIn("new-session", tmux_text)
        self.assertIn("PIONEER_USE_DA3=1", tmux_text)
        self.assertIn("PIONEER_ENFORCE_RUN_SNAPSHOT=1", tmux_text)
        self.assertIn("HF_HUB_OFFLINE=1", tmux_text)
        self.assertIn(str(hf_home), tmux_text)
        self.assertIn("run_with_da3_overlay.py", manifest["command"])
        self.assertIn(str(derived_run_dir / "config.json"), manifest["command"])
        self.assertIn("--debug-profiles-dir", manifest["command"])
        self.assertIn("--macarons-params-path", manifest["command"])

        inner_environment = environment.copy()
        inner_environment.update(
            {
                "PIONEER_ENFORCE_RUN_SNAPSHOT": "1",
                "PIONEER_USE_DA3": "1",
                "HF_HOME": str(hf_home),
                "HF_HUB_OFFLINE": "1",
                "MAGICIAN_DA3_APPEND_PATHS": environment[
                    "PIONEER_DA3_APPEND_PATHS"
                ],
                "PIONEER_TEST_TAMPER_ASSET": str(model_weights),
            }
        )
        tampered = subprocess.run(
            [
                "bash",
                str(runner),
                "--inside-tmux",
                "0",
                str(derived_run_dir),
                sys.executable,
                "derived.json",
                "pioneer-50",
                str(repo_dir),
            ],
            cwd=repo_dir,
            env=inner_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(tampered.returncode, 2, tampered.stdout + tampered.stderr)
        status = dict(
            line.split("=", 1)
            for line in (derived_run_dir / "status.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        self.assertEqual(status["snapshot_integrity_preflight"], "PASS")
        self.assertEqual(status["snapshot_integrity_postflight"], "FAIL")
        self.assertEqual(status["exit_code"], "2")
        self.assertIn(
            "runtime asset changed after manifest creation: da3_model_weights",
            tampered.stdout + tampered.stderr,
        )
        model_weights.write_bytes(b"model-weights")

        tmux_log.write_text("", encoding="utf-8")
        mismatch_run_dir = root / "artifacts" / "mismatch"
        mismatch = subprocess.run(
            [
                "bash",
                str(runner),
                "pioneer-outer-mismatch",
                "0",
                str(mismatch_run_dir),
                "mismatch.json",
                "quick",
            ],
            cwd=repo_dir,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(mismatch.returncode, 2)
        self.assertIn("does not match config debug_profile pioneer-50", mismatch.stderr)
        self.assertFalse(mismatch_run_dir.exists())
        self.assertNotIn("new-session", tmux_log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
