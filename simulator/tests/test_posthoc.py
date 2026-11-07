import unittest

import numpy as np

from simulator.posthoc_metrics import mcnemar_exact, task_progress, tip_over, permutation_paired


class PosthocMetricTests(unittest.TestCase):
    def test_grasping_progress_reaches_one_at_lift_threshold(self):
        z = [0.025, 0.075, 0.125, 0.10]
        prog = task_progress('grasping', [[0.4, 0.0, h] for h in z], None, 0.10, 0.03)
        np.testing.assert_allclose(prog, [0.0, 0.5, 1.0, 0.75])

    def test_pushing_progress_reaches_one_at_target_tolerance(self):
        target = [0.0, 0.20, 0.0258]
        pos = [[0.0, 0.0, 0.025], [0.0, 0.085, 0.025], [0.0, 0.17, 0.025], [0.0, 0.25, 0.025]]
        prog = task_progress('pushing', pos, target, 0.10, 0.03)
        np.testing.assert_allclose(prog, [0.0, 0.5, 1.0, 0.8823529], rtol=1e-6)

    def test_tip_over_threshold_matches_cube_on_edge(self):
        on_edge = 0.025 * (2 ** 0.5 - 1)  # 1.04 cm rise for a 5 cm cube
        self.assertTrue(tip_over([[0, 0, 0.025], [0, 0, 0.025 + on_edge]], 0.010))
        self.assertFalse(tip_over([[0, 0, 0.025], [0, 0, 0.030]], 0.010))

    def test_mcnemar_exact(self):
        self.assertEqual(mcnemar_exact([1, 0, 1], [1, 0, 1]), 1.0)
        # 10 discordant pairs all in one direction: p = 2 / 2**10.
        a, b = [1] * 10 + [0] * 5, [0] * 10 + [0] * 5
        self.assertAlmostEqual(mcnemar_exact(a, b), 2 / 2 ** 10)

    def test_permutation_detects_consistent_shift(self):
        rng = np.random.default_rng(0)
        self.assertLess(permutation_paired(np.full(30, 1.0), 2000, rng), 0.01)
        self.assertEqual(permutation_paired(np.zeros(30), 2000, rng), 1.0)


if __name__ == '__main__':
    unittest.main()
