#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull" / "experiments" / "photo_mask_guard_20260917"))

from photo_carve_warp import mask_samples

from run_fgpatch import default_out_root, write_viz_instances

class MaskSamplesTest(unittest.TestCase):
    def test_rejects_background_and_out_of_bounds_samples(self):
        mask = np.array([[True, False], [True, True]])
        actual = mask_samples(
            mask,
            np.array([0.1, 1.0, -0.1, 2.1]),
            np.array([0.1, 0.1, 1.0, 1.0]),
        )
        np.testing.assert_array_equal(actual, [True, False, False, False])

    def test_output_root_tracks_input_hull_and_reduction(self):
        self.assertEqual(
            default_out_root("srp_hull_mv2_v12_am1_fp", "second"),
            "srp_hull_mv2_v12_am1_fp_photo_fgpatch_second",
        )
    def test_writes_legacy_grayfill_instances(self):
        occupancy = np.array([[[True, False]]])
        grid_min = np.array([0.1, 0.2, 0.3])
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            self.assertTrue(write_viz_instances(
                out, occupancy, grid_min, 0.005, "srp_hull_mv2_v12_am1"
            ))
            saved = np.load(out / "instances.npz")
            np.testing.assert_array_equal(saved["labels"], np.zeros_like(occupancy))
            np.testing.assert_array_equal(saved["occupancy"], occupancy)
            np.testing.assert_array_equal(saved["grid_min"], grid_min)
            self.assertEqual(float(saved["voxel_size"]), 0.005)


if __name__ == "__main__":
    unittest.main()
