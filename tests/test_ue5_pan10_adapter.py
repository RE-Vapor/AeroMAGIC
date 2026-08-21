import unittest

import numpy as np
import torch

from macarons.utility.planning_observations import unproject_depth_to_world
from macarons.utility.ue5_geometry_conformance import backproject_face
from macarons.utility.ue5_observation_contract import make_synthetic_plane_bundle
from macarons.utility.ue5_pan10_adapter import (
    UE_CONTRACT_TO_MAGICIAN,
    canonical_bundle_to_pan10,
)


class UE5PAN10AdapterTests(unittest.TestCase):
    def test_pixel_rays_and_world_axes_match_canonical_geometry(self):
        canonical = make_synthetic_plane_bundle(
            image_size=11,
        )
        adapted = canonical_bundle_to_pan10(
            canonical,
            device="cpu",
            scene_units_per_meter=0.2,
            bundle_id=0,
        )
        self.assertEqual(adapted.render_count, 6)
        self.assertEqual(adapted.metadata["depth_source"], "UE5")
        for source, face in zip(canonical.faces, adapted.faces):
            expected, _ = backproject_face(source)
            expected = (
                expected @ UE_CONTRACT_TO_MAGICIAN.T * 0.2
            ).astype(np.float32)
            actual = unproject_depth_to_world(face.depth_z, face.camera)
            mask = face.valid_mask[0, ..., 0]
            np.testing.assert_allclose(
                actual[mask].detach().cpu().numpy(),
                expected,
                rtol=2e-5,
                atol=2e-5,
            )

    def test_rejects_invalid_scale_before_pan10(self):
        canonical = make_synthetic_plane_bundle(image_size=7)
        with self.assertRaisesRegex(ValueError, "scene_units_per_meter"):
            canonical_bundle_to_pan10(
                canonical,
                device=torch.device("cpu"),
                scene_units_per_meter=0.0,
                bundle_id=0,
            )


if __name__ == "__main__":
    unittest.main()
