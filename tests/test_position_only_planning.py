from types import SimpleNamespace
import unittest

import torch

from macarons.utility.position_only_planning import (
    FULL_SPHERE_FACE_NAMES,
    POSITION_ONLY_STATE_VERSION,
    PositionOnlyPlannerState,
    PositionOnlySpec,
    pose_index_to_xyz,
    position_neighbors,
    position_only_structure_audit,
    xyz_to_canonical_pose_index,
)


class _Face:
    def __init__(self, name):
        self.name = name


def _bundle(*, names=FULL_SPHERE_FACE_NAMES, mode="cubemap6", render_count=None):
    names = tuple(names)
    return SimpleNamespace(
        faces=tuple(_Face(name) for name in names),
        render_count=len(names) if render_count is None else render_count,
        metadata={"observation_mode": mode},
    )


class PositionOnlyPlanningTests(unittest.TestCase):
    def setUp(self):
        self.spec = PositionOnlySpec(
            position_shape=(6, 12, 6),
            orientation_shape=(5, 10),
            canonical_orientation_index=(2, 0),
        )

    def test_eiffel_5d_pose_projects_to_xyz_and_canonical_camera_adapter(self):
        pose = torch.tensor([2, 9, 3, 4, 5], dtype=torch.long)
        xyz = pose_index_to_xyz(pose)
        canonical_pose = xyz_to_canonical_pose_index(xyz, self.spec)

        self.assertEqual(xyz.tolist(), [2, 9, 3])
        self.assertEqual(tuple(xyz.shape), (3,))
        self.assertEqual(canonical_pose.tolist(), [2, 9, 3, 2, 0])
        self.assertEqual(pose.tolist(), [2, 9, 3, 4, 5])

        batched = torch.tensor([[2, 9, 3, 4, 5], [1, 2, 3, 0, 9]])
        canonical_batch = xyz_to_canonical_pose_index(
            pose_index_to_xyz(batched), self.spec
        )
        self.assertEqual(
            canonical_batch.tolist(), [[2, 9, 3, 2, 0], [1, 2, 3, 2, 0]]
        )

    def test_neighbors_are_translation_only_without_clamped_self_or_duplicates(self):
        center = torch.tensor([2, 5, 2], dtype=torch.long)
        neighbors = position_neighbors(center, self.spec)

        self.assertEqual(tuple(neighbors.shape), (6, 3))
        self.assertEqual(len(torch.unique(neighbors, dim=0)), 6)
        self.assertTrue(bool((torch.abs(neighbors - center).sum(dim=1) == 1).all()))
        self.assertFalse(bool((neighbors == center).all(dim=1).any()))

        corner = torch.tensor([0, 0, 0], dtype=torch.long)
        corner_neighbors = position_neighbors(corner, self.spec)
        self.assertEqual(tuple(corner_neighbors.shape), (3, 3))
        self.assertEqual(
            {tuple(row) for row in corner_neighbors.tolist()},
            {(1, 0, 0), (0, 1, 0), (0, 0, 1)},
        )

    def test_seen_state_commits_only_after_complete_full_sphere_capture(self):
        state = PositionOnlyPlannerState(self.spec)
        xyz = torch.tensor([2, 9, 3], dtype=torch.long)

        # Converting or traversing to a camera pose is not an observation.
        self.assertEqual(
            xyz_to_canonical_pose_index(xyz, self.spec).tolist(), [2, 9, 3, 2, 0]
        )
        self.assertFalse(state.is_observed(xyz))

        def failed_capture():
            raise RuntimeError("sixth face failed")

        with self.assertRaisesRegex(RuntimeError, "sixth face failed"):
            state.capture_and_commit(xyz, failed_capture)
        self.assertFalse(state.is_observed(xyz))
        self.assertEqual(state.observed_positions, frozenset())

        with self.assertRaisesRegex(ValueError, "canonical six faces"):
            state.capture_and_commit(
                xyz, lambda: _bundle(names=FULL_SPHERE_FACE_NAMES[:-1])
            )
        self.assertFalse(state.is_observed(xyz))

        with self.assertRaisesRegex(ValueError, "observation_mode"):
            state.capture_and_commit(xyz, lambda: _bundle(mode="single"))
        self.assertFalse(state.is_observed(xyz))

        completed = _bundle()
        self.assertIs(state.capture_and_commit(xyz, lambda: completed), completed)
        self.assertTrue(state.is_observed(xyz))
        self.assertEqual(state.observed_positions, frozenset({(2, 9, 3)}))

        # A legacy pose with a different body yaw maps to the same seen XYZ.
        different_yaw = torch.tensor([2, 9, 3, 2, 9], dtype=torch.long)
        self.assertTrue(state.is_observed(pose_index_to_xyz(different_yaw)))

    def test_unseen_neighbors_use_committed_xyz_not_legacy_orientation(self):
        state = PositionOnlyPlannerState(self.spec)
        center = torch.tensor([2, 5, 2], dtype=torch.long)
        target = torch.tensor([3, 5, 2], dtype=torch.long)

        self.assertIn(target.tolist(), state.unseen_neighbors(center).tolist())
        state.capture_and_commit(target, lambda: _bundle())
        self.assertNotIn(target.tolist(), state.unseen_neighbors(center).tolist())

        # Preserve the legacy planner's dead-end behavior: prefer unseen
        # positions, but allow backtracking once every adjacent XYZ has really
        # been captured.
        for neighbor in position_neighbors(center, self.spec):
            if not state.is_observed(neighbor):
                state.capture_and_commit(neighbor, lambda: _bundle())
        self.assertEqual(len(state.unseen_neighbors(center)), 0)
        self.assertEqual(
            {tuple(row) for row in state.valid_neighbors(center).tolist()},
            {tuple(row) for row in position_neighbors(center, self.spec).tolist()},
        )

    def test_structure_audit_distinguishes_raw_legacy_and_pan10_effective_actions(self):
        audit = position_only_structure_audit(self.spec)

        self.assertEqual(audit["schema_version"], 1)
        self.assertEqual(audit["state_version"], POSITION_ONLY_STATE_VERSION)
        self.assertEqual(audit["legacy_pose_state_dimension"], 5)
        self.assertEqual(audit["position_only_state_dimension"], 3)
        self.assertEqual(audit["legacy_pose_state_count"], 21600)
        self.assertEqual(audit["position_only_state_count"], 432)
        self.assertEqual(audit["orientation_state_multiplier"], 50)
        self.assertEqual(audit["legacy_raw_action_branch_count"], 10)
        self.assertEqual(audit["pan10_effective_action_branch_count"], 6)
        self.assertEqual(audit["position_only_action_branch_count"], 6)
        self.assertEqual(audit["removed_orientation_action_branch_count"], 4)
        self.assertEqual(audit["canonical_orientation_index"], [2, 0])

    def test_invalid_shapes_indices_and_partial_bundle_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "position_shape"):
            PositionOnlySpec((6, 12), (5, 10))
        with self.assertRaisesRegex(ValueError, "canonical_orientation_index"):
            PositionOnlySpec((6, 12, 6), (5, 10), (5, 0))
        with self.assertRaisesRegex(ValueError, "final dimension 5"):
            pose_index_to_xyz([1, 2, 3])
        with self.assertRaisesRegex(TypeError, "integer indices"):
            pose_index_to_xyz(torch.tensor([1.0, 2.0, 3.0, 2.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "outside position_shape"):
            xyz_to_canonical_pose_index([6, 0, 0], self.spec)

        state = PositionOnlyPlannerState(self.spec)
        with self.assertRaisesRegex(ValueError, "render_count=6"):
            state.capture_and_commit(
                [0, 0, 0], lambda: _bundle(render_count=5)
            )
        self.assertEqual(state.observed_positions, frozenset())


if __name__ == "__main__":
    unittest.main()
