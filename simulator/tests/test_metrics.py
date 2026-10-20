import unittest
from simulator.evaluation.metrics import TaskConfig, EpisodeMetrics, summarize
from simulator.evaluation.contacts import unintended


class MetricsTests(unittest.TestCase):
    def make(self, task='grasping', **kwargs):
        return EpisodeMetrics(task, [.45, -.2, .025], [.45, 0, .025], TaskConfig(**kwargs))

    def test_grasp_requires_height_bilateral_contacts_and_hold(self):
        m = self.make()
        m.update(.1, [.45, -.2, .2], [0]*3, False)
        m.update(.2, [.45, -.2, .2], [0]*3, True)
        m.update(1.1, [.45, -.2, .2], [0]*3, True)
        self.assertFalse(m.success)
        m.update(1.2, [.45, -.2, .2], [0]*3, True)
        self.assertTrue(m.success)
        self.assertAlmostEqual(m.result()['completion_time_s'], 1.2)

    def test_hold_resets_on_drop_or_contact_loss(self):
        m = self.make()
        m.update(.1, [.45, -.2, .2], [0]*3, True)
        m.update(.9, [.45, -.2, .2], [0]*3, False)
        m.update(1.1, [.45, -.2, .2], [0]*3, True)
        m.update(1.5, [.45, -.2, .2], [0]*3, True)
        self.assertFalse(m.success)
        m.update(2.1, [.45, -.2, .2], [0]*3, True)
        self.assertTrue(m.success)

    def test_push_requires_target_height_and_low_speed(self):
        m = self.make('pushing')
        m.update(.1, [.45, 0, .025], [.1, 0, 0])
        m.update(.2, [.45, 0, .025], [0]*3)
        m.update(1.2, [.45, 0, .025], [0]*3)
        self.assertTrue(m.success)

    def test_push_lift_then_place_never_passes(self):
        m = self.make('pushing', timeout_s=2)
        m.update(.1, [.45, -.1, .1], [0]*3)
        m.update(.2, [.45, 0, .025], [0]*3)
        m.update(2, [.45, 0, .025], [0]*3)
        self.assertEqual(m.result()['reason'], 'timeout')

    def test_failure_collision_and_summary_denominators(self):
        good = self.make()
        good.update(.1, [.45, -.2, .2], [0]*3, True, True)
        good.update(1.1, [.45, -.2, .2], [0]*3, True)
        bad = self.make(timeout_s=2)
        bad.update(2, [.45, -.2, .025], [0]*3)
        s = summarize([good.result(), bad.result()])
        self.assertEqual(s['task_success_rate_pct'], 50)
        self.assertEqual(s['collision_rate_pct'], 50)
        self.assertAlmostEqual(s['task_completion_time_s'], 1.1)
        self.assertIsNone(summarize([bad.result()])['task_completion_time_s'])

    def test_fall_and_bad_samples(self):
        m = self.make()
        with self.assertRaises(ValueError): m.update(0, [0]*3, [0]*3)
        with self.assertRaises(ValueError): m.update(.1, [float('nan')]*3, [0]*3)
        m.update(.1, [.45, -.2, -.03], [0]*3)
        self.assertEqual(m.result()['reason'], 'object_fell')
        with self.assertRaises(RuntimeError): m.update(.2, [0]*3, [0]*3)

    def test_completion_after_deadline_is_failure(self):
        m = self.make(timeout_s=1)
        m.update(.1, [.45, -.2, .2], [0]*3, True)
        m.update(1.2, [.45, -.2, .2], [0]*3, True)
        self.assertFalse(m.result()['success'])

    def test_collision_exclusions_and_argument_order(self):
        cases = [('/World/robot/panda_leftfinger', '/World/Cube', False),
                 ('/World/robot/panda_hand', '/World/Cube', True),
                 ('/World/Cube', '/World/Table', False),
                 ('/World/Cube', '/World/ground_plane', True),
                 ('/World/robot/panda_link0', '/World/Table', False),
                 ('/World/robot/panda_link2', '/World/Table', True)]
        for a, b, expected in cases:
            self.assertEqual(unintended(a, b), expected)
            self.assertEqual(unintended(b, a), expected)

    def test_invalid_config(self):
        for kwargs in ({'timeout_s': 0}, {'hold_s': 31}, {'lift_m': float('nan')}):
            with self.assertRaises(ValueError): TaskConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
