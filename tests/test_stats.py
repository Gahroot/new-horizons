import math
import random
import unittest

from horizons import stats


class StatsTests(unittest.TestCase):
    def test_exact_permutation_minimum(self):
        # 3 vs 3 perfectly separated: exact one-sided p = 1 / C(6, 3) = 0.05
        self.assertAlmostEqual(stats.permutation_test([4, 5, 6], [1, 2, 3], alternative="greater"), 0.05)
        self.assertAlmostEqual(stats.permutation_test([1, 2, 3], [4, 5, 6], alternative="less"), 0.05)
        self.assertEqual(stats.permutation_test([1, 2, 3], [4, 5, 6], alternative="greater"), 1.0)

    def test_identical_groups_not_significant(self):
        self.assertEqual(stats.permutation_test([1, 1, 1], [1, 1, 1], alternative="two-sided"), 1.0)

    def test_monte_carlo_detects_real_effect_and_not_noise(self):
        r = random.Random(7)
        a = [r.gauss(1.0, 1) for _ in range(40)]
        b = [r.gauss(0.0, 1) for _ in range(40)]
        self.assertLess(stats.permutation_test(a, b), 0.01)
        c = [r.gauss(0.0, 1) for _ in range(40)]
        self.assertGreater(stats.permutation_test(c, b, alternative="two-sided"), 0.05)

    def test_paired_sign_flip_exact(self):
        # 3 positive differences: only the all-plus pattern is as extreme -> p = 1/8
        self.assertAlmostEqual(stats.paired_permutation_test([0.2, 0.1, 0.3], alternative="greater"), 0.125)
        self.assertAlmostEqual(stats.paired_permutation_test([-0.2, -0.1, -0.3], alternative="less"), 0.125)
        self.assertEqual(stats.paired_permutation_test([-0.2, -0.1, -0.3], alternative="greater"), 1.0)
        self.assertEqual(stats.paired_permutation_test([]), 1.0)

    def test_paired_sign_flip_monte_carlo(self):
        r = random.Random(3)
        shift = [0.5 + r.gauss(0, 0.2) for _ in range(30)]
        self.assertLess(stats.paired_permutation_test(shift), 0.01)
        noise = [r.gauss(0, 1) for _ in range(30)]
        self.assertGreater(stats.paired_permutation_test(noise, alternative="two-sided"), 0.05)

    def test_bootstrap_ci_contains_true_difference(self):
        lo, hi = stats.bootstrap_ci([5, 6, 7, 6, 5], [1, 2, 3, 2, 1])
        self.assertLess(lo, 4)
        self.assertGreater(hi, 3.5)
        self.assertTrue(all(math.isnan(x) for x in stats.bootstrap_ci([], [1])))

    def test_holm(self):
        self.assertEqual(stats.holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])
        self.assertEqual(stats.holm([0.5]), [0.5])
        self.assertEqual(stats.holm([0.9, 0.9]), [1.0, 1.0])  # capped at 1


if __name__ == "__main__":
    unittest.main()
