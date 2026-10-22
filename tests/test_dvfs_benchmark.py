import unittest

import numpy as np

from jetson_orin_nx.benchmark.benchmark_dvfs import integrate_power_samples


class DVFSBenchmarkTests(unittest.TestCase):
    def test_integrate_power_samples_uses_full_window(self):
        samples = [
            {"timestamp_monotonic_s": 0.0, "power_w": 5.0},
            {"timestamp_monotonic_s": 0.5, "power_w": 10.0},
            {"timestamp_monotonic_s": 1.0, "power_w": 15.0},
        ]
        energy = integrate_power_samples(samples)
        self.assertTrue(np.isfinite(energy))
        self.assertGreater(energy, 0.0)

    def test_integrate_power_samples_rejects_missing_data(self):
        self.assertIsNone(integrate_power_samples([]))
        self.assertIsNone(integrate_power_samples([{"timestamp_monotonic_s": 0.0, "power_w": None}]))


if __name__ == "__main__":
    unittest.main()
