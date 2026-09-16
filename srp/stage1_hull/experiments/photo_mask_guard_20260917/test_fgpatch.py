#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))

from photo_carve_warp import mask_samples


class MaskSamplesTest(unittest.TestCase):
    def test_rejects_background_and_out_of_bounds_samples(self):
        mask = np.array([[True, False], [True, True]])
        actual = mask_samples(
            mask,
            np.array([0.1, 1.0, -0.1, 2.1]),
            np.array([0.1, 0.1, 1.0, 1.0]),
        )
        np.testing.assert_array_equal(actual, [True, False, False, False])


if __name__ == "__main__":
    unittest.main()
