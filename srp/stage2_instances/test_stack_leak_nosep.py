#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Synthetic contract tests for the clean-separation metrics."""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stack_leak_nosep import label_stats  # noqa: E402


class CleanSeparationTests(unittest.TestCase):
    def stats(self, prediction, gt=None):
        gt = gt if gt is not None else [1, 1, 1, 1, 2, 2, 2, 2]
        return label_stats(np.array(gt), np.array(prediction), {1: "lower", 2: "upper"})

    def test_clean_separation_has_only_correct_voxels(self):
        stat = self.stats([10, 10, 10, 10, 20, 20, 20, 20])
        for obj in (1, 2):
            self.assertEqual(stat["per"][obj]["correct_frac"], 1.0)
            self.assertEqual(stat["per"][obj]["leak_frac"], 0.0)
            self.assertEqual(stat["per"][obj]["unassigned_frac"], 0.0)
            self.assertEqual(stat["per"][obj]["fragment_frac"], 0.0)

    def test_full_merge_is_directional_leak_and_low_purity(self):
        stat = self.stats([10, 10, 10, 10, 10, 10, 10, 10])
        self.assertEqual(stat["per"][1]["correct_frac"], 1.0)
        self.assertEqual(stat["per"][2]["leak_frac"], 1.0)
        self.assertEqual(stat["per"][2]["leak_by_obj_frac"], {1: 1.0})
        self.assertEqual(stat["purity"](10), 0.5)

    def test_stack_top_contamination_is_detected_despite_different_main_instances(self):
        stat = self.stats([10, 10, 10, 10, 10, 10, 10, 20, 20, 20], [1] * 5 + [2] * 5)
        lower, upper = stat["per"][1], stat["per"][2]
        self.assertNotEqual(lower["main_i"], upper["main_i"])
        self.assertEqual(upper["leak_by_obj_frac"], {1: 0.4})
        self.assertEqual(upper["correct_frac"], 0.6)
        self.assertEqual(upper["unassigned_frac"], 0.0)

    def test_unassigned_voxels_are_not_mistaken_for_clean_separation(self):
        stat = self.stats([10, 10, 10, 10, 0, 0, 0, 0])
        upper = stat["per"][2]
        self.assertEqual(upper["leak_frac"], 0.0)
        self.assertEqual(upper["unassigned_frac"], 1.0)
        self.assertEqual(upper["correct_frac"], 0.0)

    def test_same_object_split_is_reported_as_fragmentation(self):
        stat = self.stats([10, 10, 11, 11, 20, 20, 20, 20])
        lower = stat["per"][1]
        self.assertEqual(lower["correct_frac"], 1.0)
        self.assertEqual(lower["correct_instance_count"], 2)
        self.assertEqual(lower["largest_correct_frac"], 0.5)
        self.assertEqual(lower["fragment_frac"], 0.5)


if __name__ == "__main__":
    unittest.main()
