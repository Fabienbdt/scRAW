import math
import unittest

import numpy as np

from scraw.metrics import compute_metrics


class BalancedRareAccuracyTests(unittest.TestCase):
    def test_averages_rare_class_recalls(self) -> None:
        labels_true = np.array(
            ["major"] * 90
            + ["medium"] * 6
            + ["rare_a"] * 3
            + ["rare_b"],
            dtype=object,
        )
        labels_pred = labels_true.copy()
        labels_pred[-1] = "major"

        metrics = compute_metrics(labels_true, labels_pred)

        self.assertEqual(metrics["RareACC"], 0.75)
        self.assertEqual(metrics["BalancedRareACC"], 0.5)

    def test_uses_strict_five_percent_threshold(self) -> None:
        labels_true = np.array(
            ["major"] * 92 + ["boundary"] * 5 + ["rare"] * 3,
            dtype=object,
        )
        labels_pred = labels_true.copy()
        labels_pred[labels_true == "boundary"] = "major"

        metrics = compute_metrics(labels_true, labels_pred)

        self.assertEqual(metrics["BalancedRareACC"], 1.0)

    def test_is_nan_without_rare_classes(self) -> None:
        labels_true = np.array(["a"] * 50 + ["b"] * 50, dtype=object)

        metrics = compute_metrics(labels_true, labels_true.copy())

        self.assertTrue(math.isnan(metrics["BalancedRareACC"]))


if __name__ == "__main__":
    unittest.main()
